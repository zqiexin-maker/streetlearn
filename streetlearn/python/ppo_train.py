# Copyright 2026 The StreetLearn Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language permissions and
# limitations under the License.

"""StreetLearn 的 PyTorch PPO 训练脚本（替换 IMPALA 版 experiment.py）。

============================================================================
PPO 训练总览（本文件是整条训练闭环的"总指挥"）
============================================================================
训练使用 on-policy PPO（GAE + clipped surrogate）；课程阶段由独立评估决定。

一次完整迭代（train 里 while 循环的一个 update）流程：

  1) Rollout 收集  (4 env × 128 步)：
     每步用当前策略 agent 前向得到 (logits, value, hp, cp, hg, cg, aux)，采样动作，
     与环境交互，把 obs/action/logp/value/reward/done 以及本步用的
     双 LSTM 记忆 (hp,cp,hg,cg) 全部存进 rollout buffer（recurrent-PPO 的关键：存隐状态）。

  2) Bootstrap 估值：用收集结束后的最后一帧再前向一次，得到 last_values，
     作为 GAE 递推的 "下一步价值"（截断项）。

  3) 计算 GAE：把 reward 序列转成 advantage / return（见 compute_gae）。

  4) PPO 更新 (ppo_epochs 轮，每轮打乱成 minibatch)：
     对每个样本用保存的 (hp, cp, hg, cg) 做前向，算 clipped 策略损失 + 价值损失
     + 熵正则（可选 + 辅助 heading 损失），反向传播更新参数。

  5) 日志 / 存档。

关键思想（on-policy）：每次更新用的样本都是"当前策略刚采的"，所以每个
update 都要重新 rollout；样本不能被无限复用（PPO 用 clip 把策略限制在离旧
策略不远的地方，允许多复用几轮）。
============================================================================
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import csv
import json
import math
import re
import os
import shutil
import tempfile
import time
import multiprocessing as mp
from collections import deque
from datetime import timedelta

try:
  import fcntl
except ImportError:  # Windows / 非 POSIX：没有进程级 flock，退化为无互斥。
  fcntl = None

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributions import Categorical
from torch.distributed.elastic.multiprocessing.errors import record

from streetlearn.python.environment import default_config
from streetlearn.python.environment import streetlearn as streetlearn_env
from streetlearn.python.environment.streetlearn import get_action_set
from streetlearn.python.agents.ppo_agent import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_META_DEFAULTS,
    PPOAgent,
    SUPPORTED_TOPOLOGIES,
    agent_kwargs_from_checkpoint,
    count_parameters,
    describe_agent,
    model_meta,
)
from streetlearn.python.checkpoint_utils import write_latest_pointer


# ImageNet 归一化常数（ResNet18 视觉编码器需要；与 PPOAgent 的方向栏遮蔽共用一份）。
# 注意：这些值在 obs_to_batch 里会被转成与 view_t 同设备的张量，避免
# "cuda 张量无法转 numpy" 的问题（早期 bug 之一）。
_IMAGENET_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.array(IMAGENET_STD, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 向量化环境（每个 OS 进程持有一个 StreetLearn 实例）
# --------------------------------------------------------------------------- #
# LevelDB 的数据表文件后缀：原版 leveldb 用 .ldb，部分构建/移植版本用 .sst。
# 两种都是不可变的 sstable，可安全硬链接共享。
_TABLE_FILE_EXTS = (".sst", ".ldb")

# 副本完整性标记的内容：改动它即可让所有旧缓存一次性失效。
_CACHE_MARKER = "streetlearn-cache-v1\n"
# 首选副本被别的进程占用时，最多再试几个备选目录（_a1 / _a2 / ... / _aN）。
_CACHE_MAX_ALT = 8
# 已持有的副本锁 fd。必须在 worker 进程整个生命周期内保持打开：fd 一关（含被
# GC 回收）内核会立刻释放 flock，互斥就失效了。
_CACHE_LOCK_FDS = []


def _link_or_copy_file(src, dst):
  """优先硬链接（零额外磁盘、秒级），失败（跨分区 EXDEV 等）退回复制。"""
  try:
    os.link(src, dst)
    return "link"
  except OSError:
    shutil.copy2(src, dst)
    return "copy"


def _try_lock_copy(dst):
  """尝试独占 dst 这份副本，成功返回 fd-like 真值，已被占用返回 None。

  为什么需要进程级互斥：副本目录名只由 (源路径 hash, worker id) 决定，而
  worker id = rank * num_envs + i。**两个并发的 run**（或上一次崩掉但子进程
  还活着的 run）会算出同一个目录名，两个进程同时打开同一份 LevelDB → 抢同一
  个 LOCK 文件 → "IO error: lock .../LOCK: Resource temporarily unavailable"。
  flock 是内核态锁，进程退出（含 SIGKILL）时自动释放，不会留下死锁。
  """
  if fcntl is None:
    return 1  # 非 POSIX：无法互斥，直接放行（本地单 run 不会发生争用）
  try:
    fd = os.open(dst + ".slock", os.O_CREAT | os.O_RDWR, 0o644)
  except OSError:
    return None
  try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except OSError:
    os.close(fd)
    return None
  _CACHE_LOCK_FDS.append(fd)
  return fd


def _leveldb_lock_busy(db_dir):
  """LevelDB 自己的 LOCK 正被别的进程持有时返回 True。

  .slock 只拦得住带这套机制的进程；旧版本代码的训练、别的工具打开的副本只
  持有 LevelDB 的 LOCK（fcntl 记录锁），所以这里按 LevelDB 的加锁方式试探
  一次：拿得到就立刻释放，说明没人在用。
  """
  if fcntl is None:
    return False
  lock_path = os.path.join(db_dir, "LOCK")
  if not os.path.isfile(lock_path):
    return False
  try:
    fd = os.open(lock_path, os.O_RDWR)
  except OSError:
    return False
  try:
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except OSError:
    return True
  else:
    fcntl.lockf(fd, fcntl.LOCK_UN)
    return False
  finally:
    os.close(fd)


def _lock_holders(path):
  """从 /proc/locks 找出对 path 持有记录锁的进程，返回 'PID(命令行)' 列表。"""
  try:
    inode = os.stat(path).st_ino
    with open("/proc/locks") as f:
      lines = f.read().splitlines()
  except OSError:
    return []
  holders = []
  for line in lines:
    parts = line.split()
    if "->" in parts:  # 排队等锁的，不是持有者
      continue
    for i, token in enumerate(parts):
      if token.count(":") == 2:  # MAJ:MIN:INODE
        if (token.rsplit(":", 1)[1] == str(inode) and i > 0
                and parts[i - 1].isdigit()):
          pid = parts[i - 1]
          try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
              cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
          except OSError:
            cmd = "?"
          holders.append("%s(%s)" % (pid, cmd[:90]))
        break
  return holders


def _copy_is_complete(dst):
  try:
    with open(os.path.join(dst, ".streetlearn_cache_complete"), "r",
              encoding="ascii") as marker_file:
      return marker_file.read() == _CACHE_MARKER
  except OSError:
    return False


def _build_cache_copy(src, dst):
  """构造一份副本到 dst：先写 dst.tmp.<pid> 再原子 rename。

  用临时目录过渡是为了避免并发构造的中间状态被别人当成完整副本读取。
  返回 (硬链接文件数, 真正复制的文件数)。
  """
  tmp = "%s.tmp.%d" % (dst, os.getpid())
  if os.path.isdir(tmp):
    shutil.rmtree(tmp)
  if os.path.isdir(dst):  # 上一次构造到一半留下的残骸 / 旧 bundle
    shutil.rmtree(dst)
  n_link = n_copy = 0
  for root, _dirs, files in os.walk(src):
    rel = os.path.relpath(root, src)
    dst_root = tmp if rel == "." else os.path.join(tmp, rel)
    os.makedirs(dst_root, exist_ok=True)
    for f in files:
      s = os.path.join(root, f)
      d = os.path.join(dst_root, f)
      if f.endswith(_TABLE_FILE_EXTS):
        if _link_or_copy_file(s, d) == "link":
          n_link += 1
        else:
          n_copy += 1
      else:
        shutil.copy2(s, d)
        n_copy += 1
  # 兜底：源目录可能本就没有 LOCK（从未被打开过），补一个空的独立 LOCK。
  lock = os.path.join(tmp, "LOCK")
  if not os.path.isfile(lock):
    open(lock, "wb").close()
  with open(os.path.join(tmp, ".streetlearn_cache_complete"), "w",
            encoding="ascii") as marker_file:
    marker_file.write(_CACHE_MARKER)
  try:
    os.rename(tmp, dst)
  except OSError:
    # 并发情况下别人可能已经 rename 成功（dst 非空）。只要它完整就直接用。
    if os.path.isdir(tmp):
      shutil.rmtree(tmp, ignore_errors=True)
    if not (_copy_is_complete(dst) and os.path.isfile(os.path.join(dst, "LOCK"))):
      raise
  return n_link, n_copy


def _copy_dataset_for_worker(dataset_path, worker_id, cache_dir=None):
  """给每个 worker 一份独立的（只读）LevelDB 数据集"副本"。

  背景 / 早期 bug：LevelDB 同一时刻只允许一个进程持有写锁。安装的 plyvel
  版本不支持 read_only=True，于是 N 个并行 worker 同时打开同一份数据集目录
  会报 "LOCK: Resource temporarily unavailable"。解决办法：每个 worker 把
  数据集复制一份到临时目录（每份副本有自己的 LOCK）再打开。

  磁盘优化（2026-08-30）：数据集的体积几乎全在数据表文件（*.sst / *.ldb，
  依 leveldb 版本而定）上，而它们是
  LevelDB 的**不可变**数据文件（写入后永远不会被原地改写，leveldb 只会新增
  文件或 unlink 旧文件）。因此数据表文件用**硬链接**共享 inode——N 个 worker
  共用同一份物理数据，磁盘开销约等于 0，且复制是秒级（不用真拷几个 GB）。
  只有 LOCK / CURRENT / MANIFEST / LOG / *.log 这些小文件需要真正复制：
    - LOCK 必须是独立 inode：硬链接等于共享同一把锁，等于没复制；
    - CURRENT/MANIFEST/*.log/LOG 在 leveldb 打开或恢复时可能被改写，
      硬链接原地改写会污染源数据集，必须独立文件。
  跨设备（cache_dir 与数据集不在同一分区，os.link 报 EXDEV）时自动退回
  全量复制，行为与旧版一致。

  副本按"源路径 hash + worker id"稳定命名并缓存，后续运行直接复用，避免
  重复拷贝。若源数据集有变动，需手动删除缓存目录下 streetlearn_ds_*。

  并发安全（2026-10-04）：同名副本用 flock 做**进程级独占**；首选目录被别的
  run（或上一次崩溃残留的 worker）占用时，自动顺延到 _a1 / _a2 ... 备选目录
  并重新构造一份（构造仍走硬链接，代价很低）。所有候选都被占用才报错。
  另外直接试探 LevelDB 自己的 LOCK：旧版本代码留下的进程不持有 .slock，
  只看 .slock 会把正被它们打开的副本当成空闲。
  """
  src = os.path.abspath(dataset_path)
  h = hashlib.md5(src.encode("utf-8")).hexdigest()[:10]
  base = cache_dir if cache_dir else tempfile.gettempdir()
  os.makedirs(base, exist_ok=True)
  base_name = "streetlearn_ds_%s_w%d" % (h, worker_id)
  reason = "no candidate tried"
  for alt in range(_CACHE_MAX_ALT + 1):
    name = base_name if alt == 0 else "%s_a%d" % (base_name, alt)
    dst = os.path.join(base, name)
    if _try_lock_copy(dst) is None:
      reason = "copy in use by another process: %s" % dst
      continue
    if _leveldb_lock_busy(dst):
      holders = _lock_holders(os.path.join(dst, "LOCK"))
      reason = "LevelDB LOCK of %s held by %s" % (dst, ", ".join(holders) or "another process")
      print("[worker %d] WARN: %s；改用备选副本。持有者多半是之前没退出的训练进程，"
            "建议 kill 掉（它们还占着内存）。" % (worker_id, reason), flush=True)
      continue
    if _copy_is_complete(dst) and os.path.isfile(os.path.join(dst, "LOCK")):
      print("[worker %d] reusing cached dataset copy: %s" % (worker_id, dst),
            flush=True)
      return dst
    print("[worker %d] preparing dataset copy %s -> %s (hardlink for table "
          "files)" % (worker_id, src, dst), flush=True)
    try:
      n_link, n_copy = _build_cache_copy(src, dst)
    except OSError as exc:
      reason = "failed to build %s: %s" % (dst, exc)
      continue
    print("[worker %d] dataset ready at %s (%d hardlinked, %d copied files)"
          % (worker_id, dst, n_link, n_copy), flush=True)
    return dst
  raise RuntimeError(
      "[worker %d] 无法取得可用的数据集副本：%d 个候选目录全部不可用（最后一个"
      "原因：%s）。常见原因是之前的训练还有进程活着，清理："
      "ps -eo pid,etime,args | grep -E 'ppo_train|spawn_main' 后 kill 掉残留进程，"
      "或给本次 run 单独指定 --ds_cache_dir。" % (worker_id, _CACHE_MAX_ALT + 1, reason))


def _worker(remote, parent_remote, env_args, worker_id, num_envs):
    """Worker 进程：拥有一个独立的 StreetLearn 环境，通过 Pipe 与主进程通信。

    通信协议（由 SubprocVecEnv 发送命令）：
      - ("reset", None)        -> 返回 env.observation()
      - ("step", action_idx)   -> 返回 (obs, reward, done, info)
      - ("close", None)        -> 关闭管道并退出
    多进程让 4 个环境真正并行（跨 GIL），提升采样吞吐。
    """
    parent_remote.close()
    # 每个 worker 一次只投影一张小图；OpenCV 默认按核数开线程，6 卡 × 32 环境
    # = 192 个 worker 同时开线程只会互相抢 CPU。
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    dataset_path, config, cache_dir, need_copy = env_args

    def worker_seed(seed):
        # 每个 worker 叠加不同偏移：各环境的起点/目标序列不同，但跨 run 可复现。
        return (int(seed) + worker_id * 100003) % (2 ** 32)
    # 并行 worker 需要独立数据集副本以避免 LevelDB 锁冲突。
    # 注意：多 rank（world_size>1）时即便每 rank 只有 1 个环境，各 rank 也会去
    # 打开同一份原始数据集 → 同样撞锁，所以这里由调用方显式传入是否需要副本。
    if need_copy:
        dataset_path = _copy_dataset_for_worker(dataset_path, worker_id,
                                                cache_dir)
    cfg = copy.deepcopy(config)
    cfg["seed"] = config["seed"] + worker_id * 100003  # 用不同种子偏移让各环境 episode 去相关
    game = default_config.CreateGame(cfg["game_name"], cfg)
    # 数据集打不开时 StreetLearnEngine.Create 返回 None → AssertionError（LevelDB
    # 的具体原因由 dataset.py 用 logging.error 打在上面）。偶发的瞬时占用（例如
    # 上一个持有者正在退出）重试几次；重试耗尽后给出带持有进程号的明确报错。
    env = None
    last_error = None
    for attempt in range(3):
        try:
            env = streetlearn_env.StreetLearn(dataset_path, cfg, game)
            break
        except AssertionError as exc:
            last_error = exc
            print("[worker %d] opening dataset %s failed (attempt %d/3)"
                  % (worker_id, dataset_path, attempt + 1), flush=True)
            time.sleep(2.0 * (attempt + 1))
    if env is None:
        holders = _lock_holders(os.path.join(dataset_path, "LOCK"))
        raise RuntimeError(
            "[worker %d] 数据集打开失败（已重试 3 次）：%s。数据集路径=%s。%s"
            % (worker_id, last_error, dataset_path,
               ("LevelDB 的 LOCK 被进程 %s 持有，多半是之前没退出的训练进程，"
                "kill 掉再启动。" % ", ".join(holders)) if holders else
               "具体原因见上面 'ERROR:root:Unable to open dataset' 那一行。"))

    def reset_with_band_retry():
        for _ in range(100):
            try:
                env.reset()
                return env.observation()
            except ValueError as exc:
                if not str(exc).startswith("No goal in requested distance band") and \
                        not str(exc).startswith("No eligible goal"):
                    raise
        return {"__reset_error__": "No eligible start/goal pair after 100 resets"}

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                remote.send(reset_with_band_retry())
            elif cmd == "step":
                action_idx = int(data)
                action_vec = env.action_set()[action_idx]
                obs, reward, done, info = env.step(action_vec)
                remote.send((obs, float(reward), bool(done), info))
            elif cmd == "set_max_target_distance":
                # P1：课程学习——更新该 worker 游戏的目标距离上限。
                meters, strict = data
                env.game.set_max_target_distance(meters, strict)
                remote.send("ok")
            elif cmd == "set_target_distance_band":
                env.game.set_target_distance_band(*data)
                remote.send("ok")
            elif cmd == "set_end_on_success":
                env.game.set_end_on_success(data)
                remote.send("ok")
            elif cmd == "set_eval_mode":
                # 课程测试：固定起点/目标 RNG、每 episode 单目标、严格距离段、
                # 到达即结束。测试直接复用训练环境，不再另起进程。
                seed, min_meters, max_meters = data
                np.random.seed(worker_seed(seed))
                env.game.set_goal_seed(worker_seed(seed))
                env.game.set_single_goal_mode(True)
                env.game.set_target_distance_band(min_meters, max_meters)
                env.game.set_end_on_success(True)
                remote.send("ok")
            elif cmd == "set_train_mode":
                # 测试结束后恢复训练采样：多目标、全局 RNG、训练距离段。
                min_meters, max_meters, strict, reseed = data
                env.game.clear_goal_seed()
                env.game.set_single_goal_mode(False)
                env.game.set_end_on_success(False)
                if min_meters is None:
                    env.game.set_max_target_distance(max_meters, strict)
                else:
                    env.game.set_target_distance_band(min_meters, max_meters)
                if reseed is not None:
                    np.random.seed(worker_seed(reseed))
                remote.send("ok")
            elif cmd == "setup_held_out":
                # P2b：held-out 泛化——配置目标 in/out 拆分。
                fraction, seed, use_held_out = data
                env.game.setup_held_out(fraction, seed, use_held_out)
                remote.send("ok")
            elif cmd == "set_goal_seed":
                # 评估 head-to-head：固定目标采样 RNG（独立于全局 np.random），
                # 同时把全局 np.random 也 seed 住（控制 reset 时的起点重生成），
                # 使 Agent/Oracle 两遍面对完全相同的 (起点,目标)。仅在评估调用
                # 时触发，训练不会调用，故不影响训练的起点多样性。
                np.random.seed(worker_seed(data))
                env.game.set_goal_seed(worker_seed(data))
                remote.send("ok")
            elif cmd == "set_single_goal_mode":
                # 评估 head-to-head：每 episode 单目标，保证 Agent/Oracle 同步。
                env.game.set_single_goal_mode(data)
                remote.send("ok")
            elif cmd == "get_map":
                # 轨迹可视化：导出整张图所有 pano 的经纬度，用来画全图散点底图。
                # env.graph 在 reset 时建立，须在 reset 之后调用。
                panos = [env.engine.GetMetadata(pid).pano for pid in env.graph]
                remote.send(np.array([(p.coords.lat, p.coords.lng)
                                      for p in panos], dtype=np.float64))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise ValueError("Unknown command %r" % cmd)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        remote.close()


class SubprocVecEnv:
    """极简的多进程向量化环境（spawn 方式，CPU 引擎）。

    作用：把 N 个 StreetLearn 环境打包成一个"向量环境"，对外接口和一个环境
    一样——一次 step 返回 N 个环境的下一组观测。这是 PPO 获取并行 on-policy
    样本的手段（同步、锁步的多进程 actor，契合 PPO 的同策略本性；而非 IMPALA
    的异步 actor-learner 架构）。

    某个 worker 进程退出后（启动时打不开数据集、被 OOM killer 杀掉……），管道
    读写会抛 BrokenPipeError / EOFError；这里统一换成说明哪些 worker 退出了的
    RuntimeError，具体原因看日志里对应 worker 的 Traceback。
    """

    _PIPE_ERRORS = (BrokenPipeError, EOFError, ConnectionResetError)

    def __init__(self, dataset_path, config, num_envs, cache_dir=None,
                 worker_id_offset=0, force_copy=False):
        self.n_envs = num_envs
        self.worker_ids = [i + worker_id_offset for i in range(num_envs)]
        ctx = mp.get_context("spawn")
        # 每对进程用一对 Pipe 通信（主进程持 remote，子进程持 work_remote）。
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(num_envs)])
        self.ps = []
        # 只要存在"别的进程也可能打开同一份源数据集"的情况就必须各用一份副本：
        # 多环境并行、带了 worker 偏移（评测/多 rank），或调用方显式要求。
        need_copy = num_envs > 1 or worker_id_offset > 0 or force_copy
        for i, (remote, work_remote) in enumerate(
                zip(self.remotes, self.work_remotes)):
            # daemon：主进程因异常退出时（续训检查失败、OOM、Ctrl-C……）解释器会
            # 先终止 daemon 子进程。非 daemon 的话退出时要 join 它们，而它们正
            # 阻塞在 recv() 等命令，主进程就挂住不退，worker 也一直占着副本锁和内存。
            # worker 自己不再创建子进程，满足 daemon 的限制。
            p = ctx.Process(
                target=_worker,
                args=(work_remote, remote,
                      (dataset_path, config, cache_dir, need_copy),
                      self.worker_ids[i],
                      num_envs),
                daemon=True)
            p.start()
            work_remote.close()  # 主进程关闭自己这端的 work_remote
            self.ps.append(p)

    def _workers_died(self, exc):
        time.sleep(1.0)  # 让刚退出的进程把 exitcode 填好
        dead = ["%d(exit %s)" % (self.worker_ids[i], p.exitcode)
                for i, p in enumerate(self.ps) if not p.is_alive()]
        return RuntimeError(
            "%d/%d 个环境 worker 进程已退出：%s%s。原因见日志里这些 worker 的 "
            "Traceback；常见原因：数据集副本被之前没退出的训练进程锁住、数据集"
            "路径不对、内存不足被 OOM killer 杀掉（exit -9）。（%s）"
            % (len(dead), self.n_envs, ", ".join(dead[:12]),
               " ..." if len(dead) > 12 else "", type(exc).__name__))

    def _send_all(self, indices, messages):
        try:
            for i, msg in zip(indices, messages):
                self.remotes[i].send(msg)
            return [self.remotes[i].recv() for i in indices]
        except self._PIPE_ERRORS as exc:
            raise self._workers_died(exc) from exc

    def _broadcast(self, cmd, data):
        indices = range(self.n_envs)
        return self._send_all(indices, [(cmd, data)] * self.n_envs)

    def reset(self, index=None):
        if index is None:
            return self.reset_indices(range(self.n_envs))
        return self.reset_indices([index])[0]

    def reset_indices(self, indices):
        """并行重置指定的几个环境，返回与 indices 对齐的观测列表。"""
        indices = list(indices)
        results = self._send_all(indices, [("reset", None)] * len(indices))
        return [self._check_reset(result) for result in results]

    @staticmethod
    def _check_reset(obs):
        if isinstance(obs, dict) and "__reset_error__" in obs:
            raise RuntimeError(obs["__reset_error__"])
        return obs

    def step(self, actions, indices=None):
        """锁步：所有（或 indices 指定的）环境同时走一步，再统一回收结果。

        indices 给出时 actions 与之一一对应，其余环境原地不动（评估时已跑完
        配额的环境不再推进）。返回的四个列表都与 indices 对齐。
        """
        indices = list(range(self.n_envs) if indices is None else indices)
        if len(actions) != len(indices):
            raise ValueError("step() got %d actions for %d envs"
                             % (len(actions), len(indices)))
        results = self._send_all(indices, [("step", a) for a in actions])
        obs = [r[0] for r in results]
        rews = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return obs, rews, dones, infos

    def close(self):
        # 某个 worker 已经崩溃时管道是断的：不能让清理本身再抛异常盖住原始错误。
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for p in self.ps:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

    def set_eval_mode(self, seed, min_meters, max_meters):
        """课程测试：单目标、目标严格落在 (min, max] 米、到达即结束、固定种子。"""
        self._broadcast("set_eval_mode", (seed, min_meters, max_meters))

    def set_train_mode(self, min_meters, max_meters, strict=True, reseed=None):
        """恢复训练采样；reseed 给出时重新播种 worker 的全局 RNG。"""
        self._broadcast("set_train_mode", (min_meters, max_meters, strict, reseed))

    # ---- P1/P2 控制接口：把指令广播到所有 worker 的 CourierGame ----
    def set_max_target_distance(self, meters, strict=False):
        """课程学习：限制目标选取半径（米）；严格模式不放宽距离。"""
        self._broadcast("set_max_target_distance", (meters, strict))

    def set_target_distance_band(self, min_meters, max_meters):
        self._broadcast("set_target_distance_band", (min_meters, max_meters))

    def set_end_on_success(self, flag):
        self._broadcast("set_end_on_success", flag)

    def setup_held_out(self, fraction, seed=0, use_held_out=True):
        """held-out 泛化：配置目标 in/out 拆分（训练用 in，评估用 out）。"""
        self._broadcast("setup_held_out", (fraction, seed, use_held_out))

    # ---- 评估 head-to-head 控制接口 ----
    def set_goal_seed(self, seed):
        """评估 head-to-head：用独立 RNG 固定目标采样（需 Agent/Oracle 两遍
        用相同 --goal_seed 与 --seed，且 --num_envs 一致）。worker 内部会再叠加
        worker_id 偏移，使各环境目标不同但跨 run 可复现。"""
        self._broadcast("set_goal_seed", seed)

    def set_single_goal_mode(self, flag):
        """评估 head-to-head：每 episode 仅 1 个目标（到达后不重抽）。"""
        self._broadcast("set_single_goal_mode", flag)

    # ---- 轨迹可视化接口 ----
    def get_map(self, index=0):
        """返回整张图所有 pano 的坐标 [N,2]，每行 (lat, lng)。

        各 worker 的图相同，只向一个 worker 请求；须在 reset 之后调用。"""
        return self._send_all([index], [("get_map", None)])[0]


# --------------------------------------------------------------------------- #
# 观测预处理（把环境返回的 obs 字典批次化成网络输入张量）
# --------------------------------------------------------------------------- #
def obs_to_batch(obs_list, device, view_shape, prev_reward=None, prev_action=None):
    """把一组环境观测（obs 字典列表）转换成批次张量。

    P0：agent 不看底层图，只消费 view_image（RGB）+ 6 维标量。
    差异 3（P1）：额外接收上一步奖励/动作 r_{t-1}/a_{t-1}，并行送两路 LSTM。
    返回 (view, scalar, prev_reward_t, prev_action_t)：
      view:            [N, 3, H, W] 已做 ImageNet 归一化的 float
      scalar:          [N, 6] float
      prev_reward_t:   [N, 1] 上一步奖励（原值，agent 内部 clamp 到 [-1,1]）
      prev_action_t:   [N]   long，上一步动作的索引（用于 one-hot）

    scalar 的拼接（对应论文里的低维导航/目标信号，且不含 neighbors）：
      [0]   = sin(yaw)            # 朝向角转 2D 单位向量，避免 359°/1° 跳变
      [1]   = cos(yaw)
      [2:4] = latlng              # 当前经纬度（已缩放到 [0,1]）
      [4:6] = target_latlng       # 目标经纬度（论文的连续 (lat,long) 目标描述符）

    prev_reward / prev_action 为 None 时（如 minibatch 未提供）回退为全零，
    即等价于 episode 起始（r_0=a_0=0）。长度需等于 N（obs 数量）。

    输入遮蔽（--mask_status_bar / --mask_position / --mask_yaw）不在这里做，
    统一在 PPOAgent.mask_inputs 里：采样、序列更新、bootstrap、评估和轨迹
    可视化都经过网络前向，设置又随 checkpoint 保存，不会出现某一路漏掉遮蔽。
    """
    n = len(obs_list)
    views = (np.stack([np.asarray(o["view_image"]).reshape(view_shape)
                       for o in obs_list]).astype(np.float32)
             / 255.0)
    view_t = torch.from_numpy(views).to(device)
    # 归一化常数必须和 view_t 同设备；用 numpy 数组会触发 cuda->host 转换并报错。
    mean = torch.as_tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    view_t = (view_t - mean) / std

    scalars = np.zeros((n, 6), dtype=np.float32)
    for i, o in enumerate(obs_list):
        yaw_deg = float(np.asarray(o["yaw"]).reshape(-1)[0])
        yaw = np.deg2rad(yaw_deg)
        scalars[i, 0] = math.sin(yaw)
        scalars[i, 1] = math.cos(yaw)
        scalars[i, 2:4] = np.asarray(o["latlng"], dtype=np.float32).reshape(2)
        scalars[i, 4:6] = np.asarray(
            o["target_latlng"], dtype=np.float32).reshape(2)
    scalar_t = torch.from_numpy(scalars).to(device)

    if prev_reward is None:
        pr_t = torch.zeros(n, 1, dtype=torch.float32, device=device)
    else:
        pr_t = torch.as_tensor(
            np.asarray(prev_reward, dtype=np.float32).reshape(n, 1),
            dtype=torch.float32, device=device)
    if prev_action is None:
        pa_t = torch.zeros(n, dtype=torch.long, device=device)
    else:
        pa_t = torch.as_tensor(
            np.asarray(prev_action, dtype=np.int64).reshape(n),
            dtype=torch.long, device=device)
    return view_t, scalar_t, pr_t, pa_t


def obs_to_seq(obs_by_env, device, view_shape, prev_r_by_env=None,
               prev_a_by_env=None):
    """[P0-1] 把一条 rollout 的观测打包成 time-major 序列张量。

    与 obs_to_batch 的区别：输入是 T 个长度 B 的 obs 列表（或 [B][T] 嵌套），
    输出 [T, B, ...]——供 PPOAgent.forward_seq 整段重展开（BPTT）。

    Args:
      obs_by_env:    [B][T] 嵌套（env 外层、时间内层）的 obs dict 列表。
      prev_r_by_env: [B][T] float，每步"当时喂给网络"的上一步奖励（原值）。
      prev_a_by_env: [B][T] int。
    Returns:
      view [T,B,3,H,W]、scalar [T,B,6]、prev_r [T,B]、prev_a [T,B]。
    """
    b = len(obs_by_env)
    t = len(obs_by_env[0])
    # 转置成 [T][B] 再复用 obs_to_batch 逐时间步批次化，保证与采集路径
    # 的预处理（/255 + ImageNet 归一化 + scalar 布局）逐比特一致。
    views, scalars = [], []
    prs = (np.zeros((t, b), dtype=np.float32) if prev_r_by_env is None
           else np.array(prev_r_by_env, dtype=np.float32).T)  # [B][T]->[T,B]
    pas = (np.zeros((t, b), dtype=np.int64) if prev_a_by_env is None
           else np.array(prev_a_by_env, dtype=np.int64).T)
    for tt in range(t):
        obs_t = [obs_by_env[e][tt] for e in range(b)]
        view_t, scalar_t, _, _ = obs_to_batch(
            obs_t, device, view_shape,
            prev_reward=prs[tt], prev_action=pas[tt])
        views.append(view_t)
        scalars.append(scalar_t)
    view = torch.stack(views, dim=0)      # [T, B, 3, H, W]
    scalar = torch.stack(scalars, dim=0)  # [T, B, 6]
    prev_r = torch.as_tensor(prs, dtype=torch.float32, device=device)  # [T,B]
    prev_a = torch.as_tensor(pas, dtype=torch.long, device=device)     # [T,B]
    return view, scalar, prev_r, prev_a


# --------------------------------------------------------------------------- #
# PPO 核心
# --------------------------------------------------------------------------- #
# GAE 实现版本，随 checkpoint 保存。1 = 旧版（非末步读 dones[:, t+1]，done 错后
# 一位）；2 = 修正后（统一读本步 dones[:, t]，默认）。续训版本不同的 checkpoint
# 会告警。--gae_version 1 只为"修复前后"对照实验保留，见 compute_gae_v1。
GAE_VERSION = 2


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    """广义优势估计（Generalized Advantage Estimation）。

    这是 PPO / A2C 系列的核心：用价值函数把"多步累计奖励"折合成"单步优势"，
    优势 A(s,a)=Q(s,a)-V(s)，表示"这一步比平均水平好多少"。

    输入：
      rewards/values/dones: [N, T]  numpy 数组（N 环境，T 步）
      last_values:         [N]       rollout 结束后下一帧的 bootstrap 价值
    输出：
      advantages [N, T], returns [N, T]（return = advantage + value，即 GAE 版的
      目标价值，用于训练 critic）

    dones[:, t] 是执行第 t 个动作**之后** episode 是否结束（env.step 返回的
    done，即 done_store[e][t]）。结束后的下一状态属于新 episode，它的价值和
    后续优势都不能传回第 t 步。公式（从后往前递推）：
      mask_t   = 1 - dones[:, t]
      delta_t  = r_t + gamma * V(s_{t+1}) * mask_t - V(s_t)
      A_t      = delta_t + gamma * lam * mask_t * A_{t+1}
      return_t = A_t + V(s_t)
    最后一步的 V(s_{t+1}) 用 last_values。lam 控制偏差-方差权衡：lam=1 偏向
    高方差低偏差（MC），lam=0 退化为单步 TD。

    修正记录（GAE_VERSION=2）：旧版非末步读 dones[:, t+1]，把 done 往后错了
    一位——终止步会接上新 episode 首帧的价值，终止前一步反而被提前截断。
    例：rewards=[0,1,0] values=[0,0,100] dones=[F,T,F] last=100 gamma=0.99
    lam=0.95，旧版优势 [0, 99.06, -1]，正确为 [0.9405, 1, -1]。
    PPOAgent.forward_seq 里清空记忆用 dones[t-1] 是对的（处理第 t 步观测前看
    上一动作是否结束 episode），与这里的时间位置不同，不要一起改。

    episode 到 frame_cap 截断也按终止处理（不 bootstrap），沿用现有定义；要区分
    截断与真正终止，需另存截断前的最终观测来估值，不能用重置后的观测。
    """
    n, t = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(n, dtype=np.float32)
    for step in reversed(range(t)):
        mask = 1.0 - dones[:, step].astype(np.float32)
        if step == t - 1:
            next_values = last_values
        else:
            next_values = values[:, step + 1]
        delta = (rewards[:, step]
                 + gamma * next_values * mask
                 - values[:, step])
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[:, step] = last_gae
    returns = advantages + values
    return advantages, returns


def compute_gae_v1(rewards, values, dones, last_values, gamma, lam):
    """修复前的 GAE（GAE_VERSION=1），**有错**，只为"修复前后"对照实验保留。

    非末步读 dones[:, t+1]：终止步接上新 episode 首帧的价值，终止前一步被提前
    截断（见 compute_gae 的修正记录）。用 --gae_version 1 显式选择。
    """
    n, t = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(n, dtype=np.float32)
    for step in reversed(range(t)):
        if step == t - 1:
            nextnonterminal = 1.0 - dones[:, step].astype(np.float32)
            next_values = last_values
        else:
            nextnonterminal = 1.0 - dones[:, step + 1].astype(np.float32)
            next_values = values[:, step + 1]
        delta = (rewards[:, step]
                 + gamma * next_values * nextnonterminal
                 - values[:, step])
        last_gae = delta + gamma * lam * nextnonterminal * last_gae
        advantages[:, step] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(agent, optimizer, rollout, view_shape, device, args):
    """[P0-1/P0-4 重写] 序列重展开式 PPO 更新（recurrent-PPO 正确形态）。

    与旧版的三处本质区别（对应调研的 R3/R5 根因）：
      1) minibatch 按 **env** 切分、整条轨迹为单位保持时序——不再展平打乱
         (env, step)。只有这样才能从 rollout 起始隐状态整段重展开。
      2) 每个 minibatch 用 agent.forward_seq 对 [T, B, ...] 整段前向，
         梯度贯穿时间（BPTT 恢复）；done 处状态清零在 forward_seq 内部完成。
      3) 优势归一化用**整批 [N,T]** 统计（DD-PPO 指出 per-minibatch 归一化
         在稀疏奖励导航任务上放大 TD 噪声），可用 --norm_adv 关闭。

    约束：num_minibatches <= num_envs（每份 minibatch 至少 1 条完整轨迹）。
    每个 minibatch 计算 clipped surrogate + value MSE + 熵 + 可选辅助损失，
    记录 approx_kl / clip_fraction / explained_variance 供健康监控。
    """
    n, t = rollout["rewards"].shape
    num_mb = max(1, min(args.num_minibatches, n))

    # ---- 整批优势归一化（在 minibatch 划分之前，用全部 N*T 个样本统计）----
    advantages = rollout["advantages"].reshape(-1).copy()
    returns = rollout["returns"].reshape(-1).copy()
    if getattr(args, "norm_adv", True):
        if dist.is_available() and dist.is_initialized():
            adv64 = advantages.astype(np.float64)
            moments = torch.tensor(
                [adv64.sum(), np.dot(adv64, adv64), adv64.size],
                dtype=torch.float64, device=device)
            dist.all_reduce(moments, op=dist.ReduceOp.SUM)
            adv_mean = (moments[0] / moments[2]).item()
            adv_std = math.sqrt(max(0.0, (moments[1] / moments[2]).item()
                                    - adv_mean ** 2)) + 1e-8
        else:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
    adv_np = advantages.reshape(n, t)
    ret_np = returns.reshape(n, t)

    # ---- 按 env 划分 minibatch（整条轨迹为单位；每个 epoch 重新打乱分组）----
    env_ids = np.arange(n)
    np.random.shuffle(env_ids)
    groups = np.array_split(env_ids, num_mb)

    hp0, cp0, hg0, cg0 = rollout["init_states"]   # 各 [1, N, hidden]（CPU）

    total = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0,
             "aux": 0.0, "kl": 0.0, "clipfrac": 0.0, "explained_var": 0.0}
    for g in groups:
        if len(g) == 0:
            continue
        obs_g = [rollout["obs"][e] for e in g]
        pr_g = [rollout["prev_r"][e] for e in g]
        pa_g = [rollout["prev_a"][e] for e in g]
        view, scalar, prev_r, prev_a = obs_to_seq(
            obs_g, device, view_shape, pr_g, pa_g)          # [T, B, ...]
        dones = torch.as_tensor(
            np.array([rollout["dones"][e] for e in g], dtype=np.float32).T,
            dtype=torch.float32, device=device)              # [T, B]
        act = torch.as_tensor(
            np.array([rollout["actions"][e] for e in g], dtype=np.int64).T,
            dtype=torch.long, device=device)                 # [T, B]
        logp_old = torch.as_tensor(
            np.array([rollout["logp"][e] for e in g], dtype=np.float32).T,
            dtype=torch.float32, device=device)              # [T, B]
        adv = torch.as_tensor(
            np.array([adv_np[e] for e in g]).T,
            dtype=torch.float32, device=device)              # [T, B]
        ret = torch.as_tensor(
            np.array([ret_np[e] for e in g]).T,
            dtype=torch.float32, device=device)              # [T, B]

        # 起始状态切片 [1, B, hidden]（无 grad——行为策略产生）。
        hs = (hp0[:, g].to(device), cp0[:, g].to(device),
              hg0[:, g].to(device), cg0[:, g].to(device))

        # ---- 序列重展开（BPTT）：整段前向，梯度跨时间步 ----
        logits, value, aux = agent(view, scalar, prev_r, prev_a,
                                   *hs, dones=dones)
        action_dist = Categorical(logits=logits)
        logp = action_dist.log_prob(act)                      # [T, B]
        entropy = action_dist.entropy().mean()

        # PPO clipped surrogate（flatten 到 [T*B] 统一归约）。
        ratio = torch.exp(logp - logp_old)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(value.reshape(-1), ret.reshape(-1))
        loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

        # 辅助 heading 任务（--aux_heading 开启时）：goal LSTM 输出预测
        # 当前朝向 (sinθ, cosθ)，与真实 yaw 对齐（P1 会改为 16 桶分类）。
        if aux is not None:
            aux_target = scalar[..., 0:2]     # [T, B, 2] (sin(yaw), cos(yaw))
            aux_loss = F.mse_loss(aux, aux_target)
            loss = loss + args.aux_coef * aux_loss
            total["aux"] += aux_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # ---- 健康监控指标（flatten 后统计）----
        with torch.no_grad():
            logratio = logp - logp_old
            kl = ((ratio - 1.0) - logratio).mean().item()      # k3 估计
            clipfrac = ((ratio - 1.0).abs() > args.clip).float().mean().item()
            var_ret = ret.reshape(-1).var().item()
            ev = (1.0 - (ret.reshape(-1) - value.reshape(-1)).var().item()
                  / (var_ret + 1e-8))
        total["loss"] += loss.item()
        total["policy"] += policy_loss.item()
        total["value"] += value_loss.item()
        total["entropy"] += entropy.item()
        total["kl"] += kl
        total["clipfrac"] += clipfrac
        total["explained_var"] += ev

    count = max(1, len(groups))
    return {k: v / count for k, v in total.items()}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _bbox_from_graph(graph):
    """StreetLearnGraph proto -> latlng 归一化范围 {"bbox_lat_min": ..., ...}。

    优先用 proto 自带的 min_coords / max_coords；缺失或退化（max <= min）时
    按全部 pano 的坐标重新计算。
    """
    lo, hi = graph.min_coords, graph.max_coords
    if (graph.HasField("min_coords") and graph.HasField("max_coords")
            and hi.lat > lo.lat and hi.lng > lo.lng):
        lat_min, lat_max, lng_min, lng_max = lo.lat, hi.lat, lo.lng, hi.lng
    else:
        lats = [p.coords.lat for p in graph.pano]
        lngs = [p.coords.lng for p in graph.pano]
        if not lats or max(lats) <= min(lats) or max(lngs) <= min(lngs):
            raise RuntimeError("数据集图里没有可用的 pano 坐标，无法计算经纬度范围")
        lat_min, lat_max = min(lats), max(lats)
        lng_min, lng_max = min(lngs), max(lngs)
    return {"bbox_lat_min": float(lat_min), "bbox_lat_max": float(lat_max),
            "bbox_lng_min": float(lng_min), "bbox_lng_max": float(lng_max)}


def dataset_latlng_bbox(dataset_path):
    """读取数据集的经纬度范围，作为 latlng / target_latlng 观测的归一化 bbox。

    背景：DEFAULT_CONFIG 的 bbox 是全球范围（-90..90 / -180..180），一座城市
    缩放后只变化约 1e-4 量级，agent 的 GoalEmbed / StateEmbed 几乎看到常数。
    按数据集自身范围缩放后，城市内坐标铺满 [0,1]。
    """
    from streetlearn.engine.python.dataset import Dataset
    from streetlearn.proto import streetlearn_pb2

    dataset = Dataset.create(dataset_path)
    if dataset is None:
        raise RuntimeError("无法打开数据集 %s" % dataset_path)
    try:
        graph = streetlearn_pb2.StreetLearnGraph()
        if not dataset.get_graph(graph):
            raise RuntimeError("数据集 %s 里读不到图（panos_connectivity）"
                               % dataset_path)
    finally:
        # 释放 LevelDB 目录锁：单 worker 时 worker 会直接打开同一个目录。
        dataset.close()
    return _bbox_from_graph(graph)


def parse_latlng_bbox(spec, dataset_path=None):
    """把 --latlng_bbox 的取值解析成 latlng 归一化范围 dict。

    三种写法：
      "dataset"  -> 读数据集图的 min/max 坐标（默认；换数据集自动跟着变）。
      "world"    -> -90..90 / -180..180（旧行为，城市内输入几乎不变，仅复现
                    旧 checkpoint），返回 None 表示沿用 DEFAULT_CONFIG。
      "lat_min,lat_max,lng_min,lng_max"
                -> 显式写死，例如曼哈顿 "40.695,40.788,-74.028,-73.940"。
                   好处是跨 run 恒定、不依赖数据集 proto，且能自己收紧范围
                   换更高归一化精度（bbox 越小，单位 [0,1] 代表的米数越少）。
    """
    spec = (spec or "dataset").strip()
    key = spec.lower()
    if key == "dataset":
        return dataset_latlng_bbox(dataset_path)
    if key == "world":
        return None
    try:
        parts = [float(x) for x in spec.split(",")]
    except ValueError:
        raise ValueError(
            "--latlng_bbox 无法解析: %r（应为 dataset / world / "
            "'lat_min,lat_max,lng_min,lng_max'）" % spec)
    if len(parts) != 4:
        raise ValueError(
            "--latlng_bbox 显式写法需要 4 个数（lat_min,lat_max,lng_min,"
            "lng_max），当前给了 %d 个: %r" % (len(parts), spec))
    lat_min, lat_max, lng_min, lng_max = parts
    if lat_max <= lat_min or lng_max <= lng_min:
        raise ValueError(
            "--latlng_bbox 的 max 必须大于 min: %r（注意顺序是 "
            "lat_min,lat_max,lng_min,lng_max）" % spec)
    return {"bbox_lat_min": lat_min, "bbox_lat_max": lat_max,
            "bbox_lng_min": lng_min, "bbox_lng_max": lng_max}


def latlng_bbox_note(config):
    """一行说明 config 使用的 latlng 归一化范围（训练 / 评估 / 可视化日志用）。"""
    lat_min = config.get("bbox_lat_min", -90.0)
    lat_max = config.get("bbox_lat_max", 90.0)
    lng_min = config.get("bbox_lng_min", -180.0)
    lng_max = config.get("bbox_lng_max", 180.0)
    note = "lat [%.5f, %.5f]  lng [%.5f, %.5f]" % (lat_min, lat_max,
                                                   lng_min, lng_max)
    if (lat_min, lat_max, lng_min, lng_max) == (-90.0, 90.0, -180.0, 180.0):
        note += ("  <- WORLD bbox (old normalization: city coordinates barely "
                 "vary; a model trained this way cannot see where the goal is)")
    return note


def reward_config_note(config):
    """一行说明 config 里实际生效的奖励设置（硬币 / 塑形窗口 / 到达阈值）。"""
    ratio = float(config.get("shaping_radius_ratio", 0.0) or 0.0)
    window = ("adaptive: clamp(%g x initial distance, %g, %g) m"
              % (ratio, 1.05 * config["min_radius_meters"],
                 config["max_radius_meters"]) if ratio > 0
              else "fixed %g m" % config["max_radius_meters"])
    return ("proportion_of_panos_with_coins=%g reward_per_coin=%g "
            "min_radius_meters=%g max_radius_meters=%g shaping_radius_ratio=%g "
            "shaping_scale=%g (shaping window %s)"
            % (config["proportion_of_panos_with_coins"], config["reward_per_coin"],
               config["min_radius_meters"], config["max_radius_meters"], ratio,
               float(config.get("shaping_scale", 1.0)), window))


def build_config(args, latlng_bbox=None):
    """构造训练用的 config 字典（分辨率、观测集合、游戏、动作集等）。

    latlng_bbox：latlng / target_latlng 观测的归一化范围（dataset_latlng_bbox
    的返回值）。None 时沿用 DEFAULT_CONFIG 的全球范围，仅用于复现旧实验。
    bbox 随 config 存进 checkpoint，评估 / 轨迹可视化加载后自动沿用同一口径。

    这里集中定义了训练超参的默认值，其中：
      - width/height=160/120、graph_*=160/120：刻意用小分辨率换吞吐，显存够可调大。
      - observations：环境要产出的观测字段，必须与 obs_to_batch / PPOAgent 对齐
        （少了某个字段，obs_to_batch 取 o["..."] 会 KeyError）。
      - game_name="courier_game"：用快递投送任务（见 courier_game.py 的奖励）。
      - auto_reset=False：episode 边界由 PPO 自己管理（done 时手动 reset，并清 LSTM 记忆）。
    最后用 default_config.ApplyDefaults 合入引擎层的其余默认值。
    """
    base = {
        # 分辨率（差异 5 预备）：默认 160×120；论文对齐可用 --width/--height 84。
        "width": args.width,
        "height": args.height,
        "graph_width": args.width,
        "graph_height": args.height,
        "status_height": 10,
        "field_of_view": 60,
        "min_graph_depth": 200,
        "max_graph_depth": 200,
        "max_cache_size": 1000,
        "full_graph": True,
        "sample_graph_depth": False,
        "frame_cap": args.frame_cap,      # 单 episode 最大步数（远目标需要更大）
        "goal_timeout": args.goal_timeout,  # 单个目标超时步数（与 frame_cap 同步更合理）
        # 到达阈值(米) / 塑形窗口外边界上限(米)；必须 0 < min < max。
        # 实际塑形半径 = clamp(shaping_radius_ratio × 目标初始距离,
        #                    min_radius×1.05, max_radius_meters)
        # ratio<=0 时退化为原版的固定窗口（= max_radius_meters）。
        "min_radius_meters": args.min_radius_meters,
        "max_radius_meters": args.max_radius_meters,
        "shaping_radius_ratio": float(args.shaping_radius_ratio),
        "shaping_scale": float(args.shaping_scale),
        # P0：agent 不看底层图（与论文一致，只用第一人称 RGB + yaw/latlng/target）。
        "observations": ["view_image", "yaw", "latlng", "target_latlng"],
        # 差异 9（P1）：目标表示。默认 target_latlng（论文最终选择）；
        # target_landmarks 已留接口但默认不启用。
        "goal_type": args.goal_type,
        # 差异 5（P2）：视觉骨干。resnet18（默认，有意增强）/ small_cnn（论文小卷积）。
        "vision_backbone": args.vision_backbone,
        "game_name": "courier_game",
        "action_spec": "streetlearn_fast_rotate",
        "rotation_speed": 22.5,
        "auto_reset": False,  # PPO manages episode boundaries itself
        "seed": args.seed,
    }
    # 关闭硬币奖励
    if getattr(args, "no_coins", False):
        base["proportion_of_panos_with_coins"] = 0.0
        base["reward_per_coin"] = 0.0
    if latlng_bbox:
        base.update(latlng_bbox)
    return default_config.ApplyDefaults(base)


def safe_torch_save(path, state, min_free_gb=1.0):
    """先写临时文件再替换，保存失败时保留上一个有效 checkpoint。

    返回 True 表示保存成功，False 表示被跳过/失败。
    - 先检查目标目录剩余空间：低于 min_free_gb 直接跳过（避免写到一半 ENOSPC 把
      整个训练进程拖崩，现象就是用户遇到的
      `PytorchStreamWriter failed writing file ... file write failed`）。
    - torch.save 包在 try/except 里，任何异常（磁盘满/权限/损坏）都只告警。
    """
    d = os.path.dirname(path) or "."
    try:
        free = shutil.disk_usage(d).free
    except OSError:
        free = None
    if free is not None and free < min_free_gb * (1 << 30):
        free_gb = free / (1 << 30)
        print("[PPO] WARN: disk free %.2f GB < %.2f GB, skip save %s"
              % (free_gb, min_free_gb, path))
        return False
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=".ppo_", suffix=".pt", dir=d)
        os.close(fd)
        torch.save(state, tmp)
        os.replace(tmp, path)
        return True
    except Exception as e:
        print("[PPO] WARN: torch.save failed, skip save %s: %s" % (path, e))
        return False
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _rotate_checkpoints(output_dir, keep_last):
    """只保留最近 keep_last 个周期性检查点，删除更旧的，防止 checkpoint 堆积把盘吃满。"""
    if keep_last <= 0:
        return
    files = glob.glob(os.path.join(output_dir, "ppo_iter_*.pt"))

    def _upd(fn):
        m = re.search(r"ppo_iter_(\d+)\.pt$", fn)
        return int(m.group(1)) if m else -1

    files.sort(key=_upd)
    for old in files[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass


def _parse_curriculum_frames(spec):
    """解析 --curriculum_frames：'帧:半径,帧:半径,...' -> [(frames, radius)]。

    帧数支持科学计数（'5e7'）；半径 <=0 表示全图（FULL）。
    """
    anchors = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        f, r = part.split(":")
        anchors.append((float(f), float(r)))
    anchors.sort(key=lambda x: x[0])
    return anchors


def _parse_curriculum_stages(spec, arrival_radius):
    stages = [float(item.strip()) for item in spec.split(",") if item.strip()]
    if not stages or stages[0] <= arrival_radius or any(
            not math.isfinite(radius) or radius <= 0 for radius in stages):
        raise ValueError("curriculum_stages must start beyond min_radius_meters")
    if any(left >= right for left, right in zip(stages, stages[1:])):
        raise ValueError("curriculum_stages must be strictly increasing")
    return stages


def _curriculum_radius(global_step, args):
    """[P0-3] 按全局帧数（global_step）计算课程半径（米）。

    旧版按 update 计数分段：默认 1200 updates × 512 帧 ≈ 61 万帧就把半径
    从 500m 拉到全图，课程只占论文 1B 预算的 0.06%（论文校准值：24h 课程
    ≈ 5.18 亿帧，约占 52%）。改按帧锚定后与吞吐解耦——多开环境或断点续训
    都不会扭曲课程-帧对应关系（原版 wall-clock 课程在续训时会重置回 500m，
    按帧锚定天然免疫）。

    --curriculum_frames='5e7:500,2e8:1000,5e8:2000,1e9:0' 的含义：
    5 千万帧前保持 500m；5e7→2e8 帧线性过渡到 1000m；……；1e9 帧起全图。
    返回值 0.0 表示"全图"（调用处会转成 None）。
    """
    anchors = _parse_curriculum_frames(args.curriculum_frames)
    if not anchors:
        return 0.0
    if global_step <= anchors[0][0]:
        return anchors[0][1]
    if global_step >= anchors[-1][0]:
        return anchors[-1][1]
    for (f0, r0), (f1, r1) in zip(anchors, anchors[1:]):
        if f0 <= global_step < f1:
            frac = (global_step - f0) / (f1 - f0)
            return r0 + frac * (r1 - r0)
    return anchors[-1][1]


# curriculum_eval.csv 的列：每行 = 一次阶段测试（全部 rank 汇总后）。
CURRICULUM_EVAL_FIELDS = [
    "stage", "lower_m", "upper_m", "update", "global_step", "stage_frames",
    "seed", "episodes", "successes", "success_rate", "all_success",
    "mean_spl", "max_goal_distance", "max_success_distance",
    "mean_success_steps", "max_success_steps", "pass_streak", "promoted",
    "eval_seconds"]

# train_metrics.csv 的列（与 plot_ppo_log.py 的读取一致）。
# 奖励按来源分开记（最近 100 个 episode 的均值）：arrival / shaping_reward /
# coin_reward 是环境里各分支实际发出的原始奖励；clipped_ep_reward 是进 GAE 的
# 奖励之和（reward_clip=abs_one 时是"每步总奖励先截断再累加"）。截断作用在每步
# 的总奖励上，所以它不等于各分量分别截断后相加，也不能反推分量。
# window_episodes：这些均值背后的 episode 数（各卡合计）；为 0 时均值列留空。
METRICS_FIELDS = ["update", "global_step", "fps", "loss", "policy", "value",
                  "entropy", "aux", "kl", "clipfrac", "explained_var",
                  "mean_ep_reward", "mean_spl",
                  "arrival", "oracle", "coverage", "cur_radius",
                  "shaping_reward", "coin_reward", "clipped_ep_reward",
                  "window_episodes"]


def _aggregate_curriculum_eval(local, distributed, device):
    """把各 rank 的测试结果合成一次完整测试（计数求和，距离/步数/耗时取最大）。"""
    sums = torch.tensor(
        [local["episodes"], local["successes"], local["spl_sum"],
         local["success_steps_sum"]], dtype=torch.float64, device=device)
    maxes = torch.tensor(
        [local["max_goal_distance"], local["max_success_distance"],
         local["max_success_steps"], local["seconds"]],
        dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxes, op=dist.ReduceOp.MAX)

    episodes = int(round(sums[0].item()))
    successes = int(round(sums[1].item()))
    if episodes < 1:
        raise RuntimeError("Curriculum evaluation produced no episodes")
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "all_success": int(successes == episodes),
        "mean_spl": sums[2].item() / episodes,
        "max_goal_distance": maxes[0].item(),
        "max_success_distance": maxes[1].item(),
        "mean_success_steps": sums[3].item() / successes if successes else 0.0,
        "max_success_steps": int(round(maxes[2].item())),
        "eval_seconds": round(maxes[3].item(), 1),
    }


def _open_csv_log(path, fields, resumed_update):
    """打开 CSV 日志（rank 0）。返回 (file, DictWriter)。

    续训时先丢掉 checkpoint 之后（update > resumed_update）写入的行：这些行
    来自崩溃前、未存进 checkpoint 的那段训练，保留会让曲线出现重复/回跳。
    旧表头缺的新列留空，统一改写成当前表头后再追加。
    """
    kept = None
    if resumed_update is not None and os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        kept = []
        for row in rows:
            try:
                if int(float(row.get("update") or 0)) <= resumed_update:
                    kept.append(row)
            except ValueError:
                continue
        if len(kept) != len(rows):
            print("[PPO] resume: dropped %d row(s) after update %d from %s"
                  % (len(rows) - len(kept), resumed_update, path))
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in kept or []:
        writer.writerow(row)
    f.flush()
    return f, writer


def train(args):
    """PPO 训练主循环。"""
    # ppo_eval 反过来依赖本模块，放到函数里导入避免循环导入。
    from streetlearn.python.ppo_eval import evaluate_curriculum

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))
    distributed = world_size > 1
    is_main = rank == 0
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but initialization failed on rank %d "
                           "(LOCAL_RANK=%d). Check CUDA_VISIBLE_DEVICES and "
                           "verify cuda:0 in a standalone Python process." %
                           (rank, local_rank))
    if distributed:
        if args.cuda:
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError("LOCAL_RANK exceeds visible GPU count")
            torch.cuda.set_device(local_rank)
        # NCCL 集合通信默认 10 分钟超时，超时后 watchdog 会直接杀掉整个任务。
        # 首次构建数据集副本、阶段测试时各 rank 进度可能相差数分钟，故显式放宽。
        # 不用 CUDA 时走 gloo（仅供 CPU 上调试多进程逻辑）。
        dist.init_process_group(
            backend="nccl" if args.cuda else "gloo", init_method="env://",
            timeout=timedelta(minutes=args.ddp_timeout_minutes))
    args.world_size = world_size
    staged = args.curriculum and args.curriculum_mode == "performance"
    stages = (_parse_curriculum_stages(args.curriculum_stages,
                                       args.min_radius_meters) if staged else [])
    if staged and (args.curriculum_eval_episodes < 1 or
                   args.curriculum_eval_envs < 0 or
                   args.curriculum_eval_interval < 1 or
                   args.curriculum_min_stage_frames < 0 or
                   args.curriculum_max_stage_frames < args.curriculum_min_stage_frames or
                   not 0 < args.curriculum_pass_rate <= 1 or
                   args.curriculum_confirmations < 1):
        raise ValueError("Invalid performance curriculum settings")
    device = (torch.device("cuda", local_rank) if args.cuda and distributed else
              torch.device("cuda" if args.cuda else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # dataset bbox 会打开 LevelDB；只让 rank 0 读取，再广播给其余 rank。
    latlng_bbox = (parse_latlng_bbox(args.latlng_bbox, args.dataset_path)
                   if is_main else None)
    if distributed:
        bbox_msg = [latlng_bbox]
        dist.broadcast_object_list(bbox_msg, src=0)
        latlng_bbox = bbox_msg[0]
    config = build_config(args, latlng_bbox)
    if is_main:
        print("[PPO] latlng bbox: %s" % latlng_bbox_note(config))
        # 实际生效的奖励配置（合并环境默认值之后）：确认硬币 / 塑形开关真的生效，
        # 而不是只改了 run.py 里的文字。同一份 config 会存进 checkpoint。
        print("[PPO] reward config: %s" % reward_config_note(config))
    mask_status_rows = int(config["status_height"]) if args.mask_status_bar else 0
    if args.mask_status_bar and mask_status_rows <= 0:
        raise ValueError("--mask_status_bar 需要 status_height > 0（当前 %r，图像里"
                         "没有方向栏）" % config["status_height"])
    # 离散动作数：从动作集规格推算（streetlearn_fast_rotate -> 5 个动作）。
    num_actions = get_action_set(
        config["action_spec"], config["rotation_speed"]).shape[0]
    # P0：标量观测固定为 6 维（yaw sin/cos + latlng(2) + target_latlng(2)），
    # 不再包含 neighbors（agent 不看底层图）。
    scalar_dim = 6

    if is_main:
        print("[PPO] world_size=%d envs_per_gpu=%d total_envs=%d "
              "device=%s num_actions=%d scalar_dim=%d aux_heading=%s"
              % (world_size, args.num_envs, world_size * args.num_envs,
                 device, num_actions, scalar_dim, args.aux_heading))

    # 实例化策略网络与优化器（环境进程在续训检查通过之后才创建，见下）。
    agent = PPOAgent(
        num_actions=num_actions,
        hidden_size=args.hidden_size,
        aux_heading=args.aux_heading,
        pretrained_vision=args.pretrained_vision,
        vision_backbone=args.vision_backbone,
        goal_type=args.goal_type,
        model_topology=args.model_topology,
        mask_position=args.mask_position,
        mask_yaw=args.mask_yaw,
        mask_status_rows=mask_status_rows,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    num_parameters = count_parameters(agent)
    if is_main:
        print("[PPO] model: %s" % describe_agent(agent))
    gae_fn = compute_gae if args.gae_version == GAE_VERSION else compute_gae_v1
    if is_main and args.gae_version != GAE_VERSION:
        print("[PPO] WARN: --gae_version %d：使用修复前（done 索引错后一位）的 GAE，"
              "只用于修复前后的对照实验。" % args.gae_version)

    stage_state = {
        "stages": stages, "index": 0, "stage_start_step": 0,
        "last_eval_step": 0, "eval_count": 0, "pass_streak": 0,
        "best_rate": -1.0, "best_spl": -1.0, "best_step": None,
        "first_threshold_step": None, "first_100_step": None, "history": [],
        "max_success_distance": 0.0, "complete": False,
    } if staged else None
    global_step = 0
    update = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        # 兼容旧 checkpoint：先按当前默认值补齐它缺的键，再比对。
        # 新增环境配置（如 shaping_radius_ratio）后，旧 ckpt 的 config 里没有这些
        # 键，严格 != 比较会直接拒绝续训。补齐后只对"真正取值不同"的项报错。
        saved_config = dict(saved.get("config") or {})
        legacy_keys = [key for key in config if key not in saved_config]
        for key in legacy_keys:
            saved_config[key] = config[key]
        diff_keys = sorted(key for key in config
                           if saved_config.get(key) != config[key])
        if diff_keys:
            raise ValueError(
                "Resume checkpoint %s: environment config differs from this run "
                "in %d key(s): %s. Resume with the original settings, or start "
                "a new run (new --output_dir, no --resume)."
                % (args.resume, len(diff_keys),
                   ", ".join("%s=%r (now %r)" % (k, saved_config.get(k), config[k])
                             for k in diff_keys)))
        if legacy_keys and is_main:
            print("[PPO] WARN: 旧 checkpoint 缺少 %d 个配置项，已按本次运行值沿用: %s"
                  % (len(legacy_keys), legacy_keys))
        # 网络结构 / 输入遮蔽必须一致：结构不同参数形状就对不上；遮蔽不同则
        # 等于中途换了输入信息，训练历史就不再是同一个实验。
        saved_model = agent_kwargs_from_checkpoint(saved)
        model_diff = [key for key in MODEL_META_DEFAULTS
                      if saved_model[key] != getattr(agent, key)]
        if model_diff:
            raise ValueError(
                "Resume checkpoint %s: network differs from this run in %s. "
                "Resume with the original --model_topology / --mask_* / "
                "--hidden_size settings, or start a new run (new --output_dir, "
                "no --resume)."
                % (args.resume, ", ".join("%s=%r (now %r)" % (
                    key, saved_model[key], getattr(agent, key)) for key in model_diff)))
        saved_gae = int(saved.get("gae_version", 1))
        if saved_gae != args.gae_version and is_main:
            print("[PPO] WARN: %s 是用 GAE v%d%s 训练的，本次续训用 v%d。两种 GAE 的"
                  "训练历史混在一起了；要比较修复前后的效果，请各自从头训练。"
                  % (args.resume, saved_gae,
                     "（修复前，done 索引错后一位）" if saved_gae == 1 else "",
                     args.gae_version))
        saved_state = saved.get("curriculum_state")
        if staged and (not saved_state or saved_state.get("stages") != stages):
            raise ValueError(
                "Resume checkpoint %s has %s, but this run uses curriculum_stages=%s. "
                "Start a new run (new --output_dir, no --resume) instead."
                % (args.resume,
                   ("curriculum_stages=%s" % saved_state.get("stages"))
                   if saved_state else "no performance-curriculum state",
                   stages))
        agent.load_state_dict(saved["agent"])
        optimizer.load_state_dict(saved["optimizer"])
        global_step = int(saved["global_step"])
        update = int(saved["update"])
        if staged:
            stage_state = saved_state
            stage_state.setdefault("best_step", None)
        if is_main:
            print("[PPO] resumed update=%d steps=%d from %s" %
                  (update, global_step, args.resume))
    resumed_update = update if args.resume else None

    # 创建向量化环境（多进程并行采样）。放在续训检查之后：checkpoint 不匹配时
    # 直接报错退出，不必先拉起一批环境进程、复制数据集副本。
    env = SubprocVecEnv(args.dataset_path, config, args.num_envs,
                        cache_dir=args.ds_cache_dir,
                        worker_id_offset=rank * args.num_envs,
                        force_copy=world_size > 1)

    # P2b：held-out 泛化——训练时把目标限制在 in 集合（留出 fraction 比例作为
    # 未见过的目标，供后续 --held_out 评估测试泛化）。0 表示不限制（默认）。
    if getattr(args, "held_out_fraction", 0.0) > 0.0:
        env.setup_held_out(args.held_out_fraction, seed=args.seed,
                           use_held_out=False)
        if is_main:
            print("[PPO] held-out masking ON: training targets restricted to in-set "
                  "(%.0f%% held out)" % (100.0 * args.held_out_fraction))

    if distributed:
        train_model = (DDP(agent, device_ids=[local_rank], output_device=local_rank)
                       if args.cuda else DDP(agent))
    else:
        train_model = agent
    torch.manual_seed(args.seed + rank * 1_000_003)
    np.random.seed(args.seed + rank * 1_000_003)

    # 训练目标距离段：性能课程下为 (到达阈值, 当前档上限]——把一开始就落在
    # 到达半径内（首步即"到达"、白拿全奖）的目标排除掉，与阶段测试同口径。
    if staged:
        env.set_train_mode(args.min_radius_meters, stages[stage_state["index"]],
                           strict=True)
    elif args.curriculum and args.curriculum_mode == "frames":
        radius = _curriculum_radius(global_step, args)
        env.set_max_target_distance(None if radius <= 0 else radius)
    obs_list = env.reset()

    # 双 LSTM 初始记忆（全零）：hp/cp 为 policy_lstm，hg/cg 为 goal_lstm，
    # 形状均为 [1, num_envs, hidden]。
    hp, cp, hg, cg = agent.initial_state(args.num_envs, device)

    # 差异 3：追踪每个环境"上一步"的奖励与动作 r_{t-1}/a_{t-1}，
    # 作为 LSTM 的显式历史输入（episode 起始时均为 0）。
    prev_rewards = np.zeros(args.num_envs, dtype=np.float32)
    prev_actions = np.zeros(args.num_envs, dtype=np.int64)

    # 统计量。
    ep_rewards = np.zeros(args.num_envs, dtype=np.float32)
    ep_successes = deque(maxlen=100)  # 最近 100 个 episode 的累计奖励（成功代理指标）
    ep_spl = deque(maxlen=100)        # 最近 100 个 episode 的 SPL（路径效率）
    # P2a：与论文 Table 2 对齐的口径——只统计"到达全奖"，排除塑形/硬币。
    ep_arrival = deque(maxlen=100)    # 最近 100 个 episode 的到达全奖累计
    ep_oracle = deque(maxlen=100)     # 最近 100 个 episode 的 BFS Oracle 上界
    ep_coverage = deque(maxlen=100)   # 到达 / Oracle（覆盖率上限）
    # 奖励按来源分开记：塑形 / 硬币是环境对应分支实际发出的原始奖励（到达见
    # ep_arrival），另记进 GAE 的累计奖励——不从总奖励事后反推分量。
    ep_shaping = deque(maxlen=100)
    ep_coin = deque(maxlen=100)
    ep_clipped = deque(maxlen=100)
    ep_clipped_sum = np.zeros(args.num_envs, dtype=np.float32)
    clip_rewards = getattr(args, "reward_clip", "abs_one") == "abs_one"
    cur_radius = None                 # 当前课程半径（用于日志）
    start_step = global_step
    t0 = time.time()

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.output_dir, "run_config_history.jsonl"), "a") as f:
            f.write(json.dumps({"started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "parameters": vars(args)}, ensure_ascii=False) + "\n")

    def checkpoint_state():
        # 网络结构 / 遮蔽设置（model_meta）随权重一起存：评估和轨迹可视化按它
        # 重建同一个网络（agent_kwargs_from_checkpoint）。
        return {
            "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
            "update": update, "global_step": global_step, "config": config,
            "num_actions": num_actions, "scalar_dim": scalar_dim,
            **model_meta(agent),
            "pretrained_vision": args.pretrained_vision,
            "num_parameters": num_parameters,
            "gae_version": args.gae_version,
            "curriculum_state": copy.deepcopy(stage_state),
        }

    def save_latest(path):
        """保存"最新训练状态"并更新 checkpoint_latest.json（断点恢复读它）。"""
        if not safe_torch_save(path, checkpoint_state(), args.min_free_gb):
            return False
        write_latest_pointer(args.output_dir, path, update, global_step)
        return True

    eval_f = eval_w = None
    if staged and is_main:
        eval_f, eval_w = _open_csv_log(
            os.path.join(args.output_dir, "curriculum_eval.csv"),
            CURRICULUM_EVAL_FIELDS, resumed_update)

    def write_curriculum_summary(status):
        if not staged or not is_main:
            return
        passed = stage_state["history"]
        distances = [item["max_success_distance"] for item in passed]
        distances.append(stage_state["max_success_distance"])
        summary = {
            "status": status, "global_step": global_step, "update": update,
            "stages": stages,
            "stages_passed": len(passed),
            "highest_passed_m": passed[-1]["upper_m"] if passed else 0,
            "max_observed_success_m": max(distances),
            "current_stage": stage_state["index"] + 1,
            "current_stage_m": stages[stage_state["index"]],
            "current_stage_frames": global_step - stage_state["stage_start_step"],
            "current_stage_evals": stage_state["eval_count"],
            "current_stage_first_100_step": stage_state["first_100_step"],
            "current_stage_best_rate": stage_state["best_rate"],
            "current_stage_best_step": stage_state.get("best_step"),
            "passed_stages": passed,
        }
        with open(os.path.join(args.output_dir, "curriculum_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # ---- 结构化日志：把指标落盘成 CSV（比解析 stdout 更稳，便于画图/归档）----
    # 同时保留控制台 print 方便实时看。CSV 列与 plot_ppo_log.py 的读取顺序一致。
    metrics_f = metrics_w = None
    if is_main:
        metrics_f, metrics_w = _open_csv_log(
            os.path.join(args.output_dir, "train_metrics.csv"),
            METRICS_FIELDS, resumed_update)

    def close_training():
        for log_file in (metrics_f, eval_f):
            if log_file and not log_file.closed:
                log_file.close()
        env.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()

    # ===== 主循环：每个 update 收集一批 rollout 并训练 =====
    stop_reason = ("all_stages_passed" if staged and stage_state["complete"]
                   else "total_timesteps")
    while global_step < args.total_timesteps and stop_reason == "total_timesteps":
        update += 1
        if staged:
            cur_radius = stages[stage_state["index"]]
        elif args.curriculum and args.curriculum_mode == "frames":
            radius = _curriculum_radius(global_step, args)
            # 0 / 负半径视为"全图"（None = 不限制）。
            cur_radius = None if radius <= 0 else radius
            env.set_max_target_distance(cur_radius)
        else:
            cur_radius = None
        n, t = args.num_envs, args.steps
        # ---- [P0-1] 只存 rollout 起始隐状态（4×[1,N,hidden]，CPU）----
        # 旧版逐步存输出状态有两重错误：(a) 存的是消费完 obs[tt] 之后的状态，
        # 回放再前向一次 = LSTM 每步推进两帧；(b) 更新时单步喂入使 BPTT=1。
        # 现在只存起始状态，更新时经 forward_seq 整段重展开。
        init_states = (hp.detach().cpu().clone(), cp.detach().cpu().clone(),
                       hg.detach().cpu().clone(), cg.detach().cpu().clone())
        # ---- 存储区：为 N 个环境、T 步预分配（用 list of lists 存 Python 对象）----
        roll_obs = [[None] * t for _ in range(n)]
        act_store = [[0] * t for _ in range(n)]
        logp_store = [[0.0] * t for _ in range(n)]
        val_store = [[0.0] * t for _ in range(n)]
        rew_store = [[0.0] * t for _ in range(n)]
        done_store = [[False] * t for _ in range(n)]
        # 差异 3：每步"当时喂给网络"的上一步 reward/action（更新时按序列回放）。
        prev_r_store = [[0.0] * t for _ in range(n)]
        prev_a_store = [[0] * t for _ in range(n)]

        # ---- 1) Rollout 收集 ----
        for tt in range(t):
            # 把当前 obs 批次化成网络输入（P0：view + 6 维标量，无图；
            # 差异 3：带上一步 reward/action r_{t-1}/a_{t-1}）。
            view_t, scalar_t, pr_t, pa_t = obs_to_batch(
                obs_list, device,
                view_shape=(3, config["height"], config["width"]),
                prev_reward=prev_rewards, prev_action=prev_actions)
            with torch.no_grad():
                # 用上一帧的双 LSTM 记忆做单步前向，得到本帧的动作分布与价值。
                logits, value, hp, cp, hg, cg, _ = train_model(
                    view_t, scalar_t, pr_t, pa_t, hp, cp, hg, cg)
            action_dist = Categorical(logits=logits)
            act = action_dist.sample()          # [N] 按当前策略采样动作
            logp = action_dist.log_prob(act)    # [N] 采样动作的对数概率
            value = value.squeeze(-1)           # [N] 价值 [B,1]->[B]

            for e in range(n):
                roll_obs[e][tt] = obs_list[e]
                act_store[e][tt] = int(act[e])
                logp_store[e][tt] = float(logp[e])
                val_store[e][tt] = float(value[e])
                # [P0-1] 不再逐步存 LSTM 状态——只存本 update 的起始状态
                # （见循环前的 init_states），更新时 forward_seq 整段重展开。
                # 差异 3：存下本步当时用的"上一步" reward/action（更新时回放）。
                prev_r_store[e][tt] = float(prev_rewards[e])
                prev_a_store[e][tt] = int(prev_actions[e])

            # 与环境交互（动作是 Python 列表，发给各子进程）。
            next_obs, rewards, dones, infos = env.step(act.cpu().tolist())
            # 训练 episode 都跑满 frame_cap，各环境往往在同一步结束：并行重置，
            # 不要一个个排队（每次重置都要在整张图上选目标 + BFS 最短路）。
            done_envs = [e for e in range(n) if dones[e]]
            if done_envs:
                for e, fresh_obs in zip(done_envs, env.reset_indices(done_envs)):
                    next_obs[e] = fresh_obs
            for e in range(n):
                rew_store[e][tt] = rewards[e]
                done_store[e][tt] = dones[e]
                ep_rewards[e] += rewards[e]
                # 与进 GAE 的奖励同口径：每步的总奖励先截断再累加。
                ep_clipped_sum[e] += (min(1.0, max(-1.0, rewards[e]))
                                      if clip_rewards else rewards[e])
                if dones[e]:
                    # episode 结束：记录成功指标，重置累计奖励。
                    ep_successes.append(ep_rewards[e])
                    spl = infos[e].get("spl", 0.0)
                    ep_spl.append(spl if spl else 0.0)
                    # P2a：论文口径——只统计到达全奖（排除塑形/硬币），并记录
                    # BFS Oracle 上界与覆盖率，用于与论文 Table 2 对齐。
                    ep_arrival.append(infos[e].get("arrival_reward", 0.0))
                    ep_oracle.append(infos[e].get("oracle_reward", 0.0))
                    ep_coverage.append(infos[e].get("coverage", 0.0))
                    ep_shaping.append(infos[e].get("shaping_reward", 0.0))
                    ep_coin.append(infos[e].get("coin_reward", 0.0))
                    ep_clipped.append(float(ep_clipped_sum[e]))
                    ep_rewards[e] = 0.0
                    ep_clipped_sum[e] = 0.0
                    # 新 episode：清零该环境的双 LSTM 记忆，避免上一个 episode 的记忆泄漏。
                    hp[:, e:e + 1, :] = 0.0
                    cp[:, e:e + 1, :] = 0.0
                    hg[:, e:e + 1, :] = 0.0
                    cg[:, e:e + 1, :] = 0.0
                    # 新 episode：上一步 reward/action 清零（r_0=a_0=0）。
                    prev_rewards[e] = 0.0
                    prev_actions[e] = 0
                else:
                    # 非结束：本步的 reward/action 成为下一步的"上一步"输入。
                    prev_rewards[e] = float(rewards[e])
                    prev_actions[e] = int(act[e])
            obs_list = next_obs
            global_step += n * world_size

        # ---- 2) Bootstrap 估值：用 rollout 结束后的最后一帧前向得 last_values ----
        with torch.no_grad():
            view_t, scalar_t, pr_t, pa_t = obs_to_batch(
                obs_list, device,
                view_shape=(3, config["height"], config["width"]),
                prev_reward=prev_rewards, prev_action=prev_actions)
            # 注意解包顺序：agent.forward 返回 (logits, value, hp, cp, hg, cg, aux)，
            # value 是第二个。早期 bug：把 logits（[B,5]）误当价值会导致 compute_gae
            # 广播失败 (4,5)(4,)。
            _, last_values, _, _, _, _, _ = train_model(
                view_t, scalar_t, pr_t, pa_t, hp, cp, hg, cg)
        last_values = last_values.squeeze(-1).cpu().numpy().astype(np.float32)

        # ---- 3) 计算 GAE：把 reward 序列转成 advantage / return ----
        # [P0-2] 论文语义："clipped at 1 for the purpose of gradient
        # calculations"——奖励裁剪后进 GAE/价值目标（原值量级 O(100-800)，
        # 会把价值 MSE 撑到压过策略项）。日志与 episode 统计仍用原值。
        # 等价于原版 IMPALA 的 reward_clipping='abs_one'
        # （experiment.py:470-471，tf.clip_by_value(rewards, -1, 1)）。
        rewards_np = np.array(rew_store, dtype=np.float32)
        if getattr(args, "reward_clip", "abs_one") == "abs_one":
            rewards_for_gae = np.clip(rewards_np, -1.0, 1.0)
        else:
            rewards_for_gae = rewards_np
        values_np = np.array(val_store, dtype=np.float32)
        dones_np = np.array(done_store, dtype=bool)
        advantages, returns = gae_fn(
            rewards_for_gae, values_np, dones_np, last_values,
            args.gamma, args.lam)

        # 组装成 rollout 字典，交给 ppo_update。
        rollout = {
            "obs": roll_obs,
            "actions": act_store,
            "logp": logp_store,
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards_np,
            "dones": done_store,
            "init_states": init_states,  # [P0-1] 4×[1,N,hidden] rollout 起始状态
            "prev_r": prev_r_store,  # 差异 3：更新时回放的上一步 reward
            "prev_a": prev_a_store,  # 差异 3：更新时回放的上一步 action
        }

        # ---- 4) PPO 多 epoch 更新 ----
        stats = {"loss": 0, "policy": 0, "value": 0, "entropy": 0, "aux": 0,
                 "kl": 0, "clipfrac": 0, "explained_var": 0}
        for _ in range(args.ppo_epochs):
            s = ppo_update(
                train_model, optimizer, rollout,
                (3, config["height"], config["width"]),
                device, args)
            for k in stats:
                stats[k] += s[k]
        for k in stats:
            stats[k] /= args.ppo_epochs  # 多轮取平均

        # ---- 5) 日志 ----
        if update % args.log_interval == 0:
            fps = (global_step - start_step) / max(1e-6, time.time() - t0)
            series = (ep_successes, ep_spl, ep_arrival, ep_oracle, ep_coverage,
                      ep_shaping, ep_coin, ep_clipped)
            if distributed:
                keys = tuple(stats)
                values = [stats[key] for key in keys]
                values += [item for samples in series
                           for item in (sum(samples), len(samples))]
                totals = torch.tensor(values, dtype=torch.float64, device=device)
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                stats = {key: (totals[i] / world_size).item()
                         for i, key in enumerate(keys)}
                means = [(totals[len(keys) + 2 * i] /
                          max(1, totals[len(keys) + 2 * i + 1].item())).item()
                         for i in range(len(series))]
                window_episodes = int(totals[len(keys) + 1].item())
            else:
                means = [float(np.mean(samples)) if samples else 0.0
                         for samples in series]
                window_episodes = len(ep_successes)
            (mean_ep, mean_spl, mean_arrival, mean_oracle, mean_cov,
             mean_shaping, mean_coin, mean_clipped) = means

            def episode_metric(value):
                # 统计窗口里还没有结束的 episode（刚开训 / 晋级后清空了窗口）时写空，
                # 曲线上是断开而不是掉到 0。
                return round(value, 3) if window_episodes else ""
            aux_str = " aux=%.3f" % stats["aux"] if args.aux_heading else ""
            cur_rad_str = ("%.0fm" % cur_radius) if cur_radius else "FULL"
            # [P1-5] 健康监控：approx_kl<0.02 正常（>0.05 报警）；clip_fraction
            # 健康 0.05-0.2；explained_var 应随训练转正。
            if is_main:
                print("[PPO] update=%d steps=%d fps=%.0f loss=%.3f "
                  "pol=%.3f val=%.3f ent=%.3f kl=%.4f clipf=%.3f ev=%.3f "
                  "mean_ep_reward=%.2f "
                  "mean_spl=%.3f arrival=%.2f oracle=%.2f cov=%.3f cur=%s%s "
                  "shaping=%.2f coin=%.2f clipped=%.2f eps=%d"
                  % (update, global_step, fps, stats["loss"], stats["policy"],
                     stats["value"], stats["entropy"], stats["kl"],
                     stats["clipfrac"], stats["explained_var"], mean_ep, mean_spl,
                     mean_arrival, mean_oracle, mean_cov, cur_rad_str, aux_str,
                     mean_shaping, mean_coin, mean_clipped, window_episodes))

                # 落盘结构化指标（便于离线画图，不依赖 stdout 解析）。
                metrics_w.writerow({
                "update": update,
                "global_step": global_step,
                "fps": round(fps, 1),
                "loss": round(stats["loss"], 4),
                "policy": round(stats["policy"], 4),
                "value": round(stats["value"], 4),
                "entropy": round(stats["entropy"], 4),
                "aux": round(stats["aux"], 4) if args.aux_heading else 0.0,
                "kl": round(stats["kl"], 6),
                "clipfrac": round(stats["clipfrac"], 4),
                "explained_var": round(stats["explained_var"], 4),
                "mean_ep_reward": episode_metric(mean_ep),
                "mean_spl": episode_metric(mean_spl),
                "arrival": episode_metric(mean_arrival),
                "oracle": episode_metric(mean_oracle),
                "coverage": episode_metric(mean_cov),
                "cur_radius": (-1 if cur_radius is None else round(cur_radius, 1)),
                "shaping_reward": episode_metric(mean_shaping),
                "coin_reward": episode_metric(mean_coin),
                "clipped_ep_reward": episode_metric(mean_clipped),
                "window_episodes": window_episodes,
                })
                metrics_f.flush()  # 训练中断时保留已有指标

        evaluated = False
        if staged and \
                global_step - stage_state["stage_start_step"] >= args.curriculum_min_stage_frames and \
                global_step - stage_state["last_eval_step"] >= args.curriculum_eval_interval:
            evaluated = True
            stage_index = stage_state["index"]
            lower = (args.min_radius_meters if stage_index == 0
                     else stages[stage_index - 1])
            upper = stages[stage_index]
            seed = args.seed + 1_000_000 + stage_index * 100_000 + \
                stage_state["eval_count"] * 997
            # 100 条测试路线按 rank 均分（余数给前几个 rank），每个 rank 再按
            # 固定配额分给自己的训练环境，各 rank 结果最后求和。
            eval_base, eval_remainder = divmod(args.curriculum_eval_episodes,
                                                world_size)
            local_eval_episodes = eval_base + int(rank < eval_remainder)
            local_result = None
            eval_error = None
            try:
                if distributed:
                    # 测试用 eval 模式（BatchNorm 用滑动统计量）：先与 rank 0 对齐，
                    # 保证各卡测的是同一个策略。
                    for buffer in agent.buffers():
                        dist.broadcast(buffer, src=0)
                local_result = evaluate_curriculum(
                    agent, env, config, lower, upper, seed, local_eval_episodes,
                    max_envs=args.curriculum_eval_envs, rank=rank,
                    world_size=world_size)
            except Exception as exc:  # 交给下面统一处理，避免其他 rank 卡在集合通信
                eval_error = exc

            if distributed:
                eval_ok = torch.tensor([int(eval_error is None)],
                                       dtype=torch.int32, device=device)
                dist.all_reduce(eval_ok, op=dist.ReduceOp.MIN)
                eval_failed = not eval_ok.item()
            else:
                eval_failed = eval_error is not None
            if eval_failed:
                close_training()
                if eval_error is not None:
                    raise RuntimeError(
                        "Curriculum evaluation failed on rank %d: %s"
                        % (rank, eval_error)) from eval_error
                raise RuntimeError(
                    "Curriculum evaluation failed on another rank; "
                    "see that rank's log under torchrun_logs/")

            result = _aggregate_curriculum_eval(local_result, distributed, device)
            if is_main:
                stage_state["eval_count"] += 1
                stage_state["last_eval_step"] = global_step
                stage_state["max_success_distance"] = max(
                    stage_state["max_success_distance"], result["max_success_distance"])
                if result["all_success"] and stage_state["first_100_step"] is None:
                    stage_state["first_100_step"] = global_step
                passed = result["success_rate"] >= args.curriculum_pass_rate
                if passed and stage_state["first_threshold_step"] is None:
                    stage_state["first_threshold_step"] = global_step
                stage_state["pass_streak"] = (stage_state["pass_streak"] + 1
                                               if passed else 0)
                promoted = stage_state["pass_streak"] >= args.curriculum_confirmations
                stage_frames = global_step - stage_state["stage_start_step"]
                print("[PPO] stage=%d band=(%.0f,%.0f]m steps=%d success=%d/%d "
                      "(%.1f%%)%s streak=%d/%d max_success=%.1fm "
                      "mean_success_steps=%.1f eval=%.0fs"
                      % (stage_index + 1, lower, upper, global_step,
                         result["successes"], result["episodes"],
                         100 * result["success_rate"],
                         " ALL" if result["all_success"] else "",
                         stage_state["pass_streak"], args.curriculum_confirmations,
                         result["max_success_distance"],
                         result["mean_success_steps"], result["eval_seconds"]),
                      flush=True)
                eval_w.writerow({
                    "stage": stage_index + 1, "lower_m": lower, "upper_m": upper,
                    "update": update, "global_step": global_step,
                    "stage_frames": stage_frames, "seed": seed,
                    "pass_streak": stage_state["pass_streak"],
                    "promoted": int(promoted), **result,
                })
                eval_f.flush()
                # 本档最优权重：成功率优先，SPL 次之。
                score = (result["success_rate"], result["mean_spl"])
                if score > (stage_state["best_rate"], stage_state["best_spl"]):
                    previous_best = (stage_state["best_rate"], stage_state["best_spl"],
                                     stage_state.get("best_step"))
                    stage_state["best_rate"], stage_state["best_spl"] = score
                    stage_state["best_step"] = global_step
                    best_path = os.path.join(args.output_dir,
                                             "ppo_best_stage_%02d_%gm.pt" %
                                             (stage_index + 1, upper))
                    if not safe_torch_save(best_path, checkpoint_state(), args.min_free_gb):
                        (stage_state["best_rate"], stage_state["best_spl"],
                         stage_state["best_step"]) = previous_best
                        close_training()
                        raise RuntimeError("Cannot save best checkpoint %s (disk full?)"
                                           % best_path)
                    print("[PPO] best of stage %d -> %s" % (stage_index + 1, best_path))
                if promoted:
                    stage_state["history"].append({
                        "stage": stage_index + 1, "upper_m": upper,
                        "first_pass_step": stage_state["first_threshold_step"],
                        "promoted_step": global_step,
                        "stage_frames": stage_frames,
                        "evals": stage_state["eval_count"],
                        "first_100_step": stage_state["first_100_step"],
                        "best_rate": stage_state["best_rate"],
                        "max_success_distance": stage_state["max_success_distance"],
                    })
                    if stage_index + 1 == len(stages):
                        stage_state["complete"] = True
                        stop_reason = "all_stages_passed"
                    else:
                        stage_state.update({
                            "index": stage_index + 1, "stage_start_step": global_step,
                            "last_eval_step": global_step, "eval_count": 0,
                            "pass_streak": 0, "best_rate": -1.0,
                            "best_spl": -1.0, "best_step": None,
                            "first_threshold_step": None,
                            "first_100_step": None, "max_success_distance": 0.0,
                        })
                        # 切换前权重：刚通过第 stage_index+1 档、尚未练下一档。
                        before_path = os.path.join(args.output_dir,
                                                   "ppo_before_stage_%02d_%gm.pt" %
                                                   (stage_index + 2, stages[stage_index + 1]))
                        if not safe_torch_save(before_path, checkpoint_state(), args.min_free_gb):
                            close_training()
                            raise RuntimeError("Cannot save checkpoint %s (disk full?)"
                                               % before_path)
                        print("[PPO] promoted to stage %d (%gm) -> %s"
                              % (stage_index + 2, stages[stage_index + 1], before_path))
                elif stage_frames >= args.curriculum_max_stage_frames:
                    stop_reason = "stage_budget_exhausted"
                    print("[PPO] stage %d not passed within %d frames; stopping."
                          % (stage_index + 1, args.curriculum_max_stage_frames))
                write_curriculum_summary(
                    "training" if stop_reason == "total_timesteps" else stop_reason)

            if distributed:
                state_msg = [(stage_state, stop_reason) if is_main else None]
                dist.broadcast_object_list(state_msg, src=0)
                stage_state, stop_reason = state_msg[0]
            if stop_reason == "total_timesteps":
                # 测试借用了训练环境：恢复训练采样（可能已晋级到下一档），
                # 换一组随机种子后全部重开 episode，LSTM 记忆一并清零。
                cur_radius = stages[stage_state["index"]]
                env.set_train_mode(args.min_radius_meters, cur_radius,
                                   strict=True, reseed=seed + 7919)
                obs_list = env.reset()
                hp, cp, hg, cg = agent.initial_state(args.num_envs, device)
                prev_rewards.fill(0)
                prev_actions.fill(0)
                ep_rewards.fill(0)
                ep_clipped_sum.fill(0)
                if stage_state["index"] != stage_index:
                    for series in (ep_successes, ep_spl, ep_arrival,
                                   ep_oracle, ep_coverage, ep_shaping,
                                   ep_coin, ep_clipped):
                        series.clear()

        # ---- 存档（容错：磁盘满也只告警、不崩训练）----
        # 每次阶段测试后也存一次，保证最新的 ppo_iter 总包含最新的课程进度。
        if is_main and (update % args.save_interval == 0 or evaluated):
            ckpt = os.path.join(args.output_dir, "ppo_iter_%d.pt" % update)
            if save_latest(ckpt):
                print("[PPO] saved checkpoint -> %s" % ckpt)
                _rotate_checkpoints(args.output_dir, args.keep_last)

    # 训练结束存最终检查点（容错）。
    if is_main:
        ckpt = os.path.join(args.output_dir, "ppo_final.pt")
        if save_latest(ckpt):
            print("[PPO] training finished (%s) -> %s" % (stop_reason, ckpt))
        else:
            print("[PPO] WARN: final checkpoint save failed (disk full?); "
                  "train_metrics.csv is preserved.")
        write_curriculum_summary(stop_reason)
    if distributed:
        if args.cuda:
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()
    close_training()


def parse_args(argv=None):
    """解析命令行超参（PPO 的标准超参集合）。"""
    p = argparse.ArgumentParser(description="StreetLearn PPO training (PyTorch)")
    p.add_argument("--local-rank", "--local_rank", type=int, default=0,
                   help=argparse.SUPPRESS)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=4,
                   help="每个 GPU rank 的并行环境数（总数再乘 WORLD_SIZE）。")
    p.add_argument("--steps", type=int, default=128,
                   help="rollout horizon T per env")          # 每个环境的收集步数
    p.add_argument("--total_timesteps", type=int, default=2_000_000)  # 总样本数上限
    # [P0-4] epochs 4->2、minibatches 4->1：512 帧上 16 个优化步/更新是
    # 复用过度主因；且 P0-1 后 minibatch 按 env 切，num_minibatches<=num_envs。
    p.add_argument("--ppo_epochs", type=int, default=2)      # 每批数据复用更新的轮数
    p.add_argument("--num_minibatches", type=int, default=1) # 每轮按 env 切几组（<=num_envs）
    p.add_argument("--gamma", type=float, default=0.99)     # 折扣因子
    p.add_argument("--lam", type=float, default=0.95)       # GAE lambda（偏差-方差权衡）
    p.add_argument("--gae_version", type=int, choices=[1, GAE_VERSION],
                   default=GAE_VERSION,
                   help="GAE 实现。2=修正后（默认）；1=修复前（done 索引错后一位，"
                        "有错），只用于在同一份代码里做修复前后的对照实验。")
    # [P0-4] clip 0.2->0.1：PPO 表 5 / Ye et al. 2020 导航任务常用 0.1。
    p.add_argument("--clip", type=float, default=0.1)       # PPO 裁剪范围（1±clip）
    # [P1-4] ent 0.01->0.004：对齐论文附录 C.2（repo 的 0.00025 是继承自
    # scalable_agent DMLab 的漏改值）。
    p.add_argument("--ent_coef", type=float, default=0.004)  # 熵正则系数（鼓励探索）
    # [P1-4] vf 0.5->0.25：对齐原版有效值（flag 0.5 × compute_baseline_loss
    # 内部硬编码 0.5）。
    p.add_argument("--vf_coef", type=float, default=0.25)   # 价值损失系数
    # [P0-4] 优势归一化开关：整批统计（非 per-minibatch）；DD-PPO 在导航任务
    # 上关闭它才稳定，做对照实验用。
    p.add_argument("--norm_adv", action="store_true", default=True,
                   help="整批优势归一化（默认开）。用 --no_norm_adv 关闭"
                        "做对照（DD-PPO 在导航任务上报告关闭更稳）。")
    p.add_argument("--no_norm_adv", dest="norm_adv", action="store_false")
    p.add_argument("--lr", type=float, default=2.5e-4)      # 学习率
    p.add_argument("--max_grad_norm", type=float, default=0.5)  # 梯度裁剪上限
    p.add_argument("--hidden_size", type=int, default=512)  # LSTM 隐状态维
    # P0：辅助 heading(θ) 任务开关（默认关）。开启后 agent 用 goal LSTM 输出预测
    # 当前朝向 (sinθ, cosθ)，作为自监督辅助损失，帮助学出空间朝向表征。
    p.add_argument("--aux_heading", action="store_true", default=False)
    p.add_argument("--aux_coef", type=float, default=0.5)  # 辅助 heading 损失权重
    p.add_argument("--pretrained_vision", action="store_true", default=False)  # 视觉预训练
    # 结构 / 输入对照实验（见 ppo_agent.py 模块说明）。设置随 checkpoint 保存，
    # 评估与轨迹可视化按存档重建；续训时必须与 checkpoint 一致。
    p.add_argument("--model_topology", choices=list(SUPPORTED_TOPOLOGIES),
                   default="parallel",
                   help="双 LSTM 连接方式。parallel=当前并联实验结构（两路输出拼接后"
                        "接双头，旧 checkpoint 都是它）；serial=CityNav 式串联（goal "
                        "LSTM 输出经 64 维 tanh 瓶颈送进 policy LSTM，双头只读 policy "
                        "LSTM），容量和输入不变。")
    p.add_argument("--mask_position", action="store_true", default=False,
                   help="输入遮蔽实验：自身经纬度 scalar[2:4] 固定为 0（目标经纬度保留）。")
    p.add_argument("--mask_yaw", action="store_true", default=False,
                   help="输入遮蔽实验：自身朝向 sin/cos(yaw) scalar[0:2] 固定为 0。")
    p.add_argument("--mask_status_bar", action="store_true", default=False,
                   help="输入遮蔽实验：图像底部的邻居方向栏（status_height 行）固定为"
                        "黑色；图像尺寸和上方投影区不变（不要用 status_height=0 代替，"
                        "那会改变投影画面）。")
    # 差异 9（P1）：目标表示。默认 target_latlng（论文最终选择）；
    # target_landmarks 仅占位、默认不启用（启用即 NotImplementedError）。
    p.add_argument("--latlng_bbox", type=str, default="dataset",
                   help="latlng / target_latlng 观测的归一化范围。dataset=按数据集"
                        "经纬度范围缩放到 [0,1]（默认）；world=旧行为（-90..90 / "
                        "-180..180，城市内输入几乎不变），只用于复现旧实验；也可显式"
                        "写死四元组 'lat_min,lat_max,lng_min,lng_max'（例：曼哈顿 "
                        "'40.695,40.788,-74.028,-73.940'），跨 run 恒定且可自己收紧"
                        "范围换精度。范围随 config 存进 checkpoint，评估 / 轨迹可视"
                        "化自动沿用。")
    p.add_argument("--goal_type", type=str, default="target_latlng",
                   choices=["target_latlng", "target_landmarks"],
                   help="目标描述符类型。target_latlng=绝对经纬度（论文默认，启用）；"
                        "target_landmarks=地标表示（可选但未接入，默认不启用）。")
    # 差异 5（P2）：视觉编码器与分辨率。默认 resnet18 @ 160×120（有意增强）；
    # 论文对齐可用 small_cnn @ 84×84。
    p.add_argument("--vision_backbone", type=str, default="resnet18",
                   choices=["resnet18", "small_cnn"],
                   help="视觉编码器骨干。resnet18=更强特征（默认）；"
                        "small_cnn=论文风格小卷积（配合 --width/--height 84 对齐论文）。")
    p.add_argument("--width", type=int, default=160,
                   help="观测宽度（像素）。论文用 84。")
    p.add_argument("--height", type=int, default=120,
                   help="观测高度（像素）。论文用 84（正方形）。")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output_dir", type=str, default="ppo_ckpts")
    p.add_argument("--resume", type=str, default=None,
                   help="从已有 .pt 恢复网络、优化器、训练帧数和课程阶段。")
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--frame_cap", type=int, default=1000,
                   help="单 episode 最大步数（frame_cap）。远目标需要更大，"
                        "否则 agent 还没走到目标就超时。")
    p.add_argument("--goal_timeout", type=int, default=1000,
                   help="单个 courier 目标的超时步数。建议与 frame_cap 保持一致，"
                        "让 agent 有完整一回合去追一个远目标。")
    p.add_argument("--min_radius_meters", type=float, default=100.0,
                   help="到达目标的距离阈值（米），小于此值视为到达并发放全奖。")
    p.add_argument("--max_radius_meters", type=float, default=200.0,
                   help="塑形窗口外边界的**上限**（米）。自适应模式下实际半径 = "
                        "clamp(shaping_radius_ratio × 目标初始距离, "
                        "min_radius×1.05, 本值)；固定模式（ratio<=0）下即为固定窗口。")
    p.add_argument("--shaping_radius_ratio", type=float, default=0.0,
                   help="塑形窗口自适应比例。>0 时按每个目标的初始直线距离缩放："
                        "0.4 复刻原版几何（200m 窗口 / 500m 课程起点），越大越早"
                        "进入塑形（越像稠密进度条）。<=0 = 固定窗口（原版行为）。")
    p.add_argument("--shaping_scale", type=float, default=1.0,
                   help="塑形奖励相对到达全奖的折扣（0,1]。奖励进 GAE 前会被裁到 "
                        "±1，给塑形打折才能在裁剪后依然保留「到达 >> 靠近」的序关系。")
    p.add_argument("--reward_clip", choices=["abs_one", "none"], default="abs_one",
                   help="奖励裁剪（等价于原版 IMPALA 的 reward_clipping=abs_one）："
                        "abs_one=把每步奖励裁到 [-1,1] 后进 GAE/价值目标（默认，与"
                        "论文'clipped at 1 for the purpose of gradient calculations'"
                        "一致）；none=用原始奖励量级（可达数百）。")
    p.add_argument("--keep_last", type=int, default=3,
                   help="周期性检查点最多保留最近几个（轮转删除更旧的），"
                        "防止 checkpoint 堆积把磁盘写满。0=全部保留。")
    p.add_argument("--ds_cache_dir", type=str, default=None,
                   help="多 worker 数据集副本（.sst 硬链接）的存放目录。默认用"
                        "系统临时目录（Linux 上通常是 /tmp）；若 /tmp 所在分区"
                        "空间不足，可指到大分区，如 --ds_cache_dir /home/wcl/tmp。"
                        "注意：该目录与数据集需在同一分区才能命中硬链接优化，"
                        "跨分区会自动退回全量复制。")
    p.add_argument("--min_free_gb", type=float, default=1.0,
                   help="存档前若目标盘剩余空间低于该值(GB)则跳过保存，"
                        "避免写到一半磁盘满导致训练进程崩溃。")
    p.add_argument("--cuda", action="store_true", default=False)
    p.add_argument("--ddp_timeout_minutes", type=float, default=120.0,
                   help="多卡训练集合通信的超时（分钟）。NCCL 默认只有 10 分钟，"
                        "阶段测试或首次构建数据集副本时各卡进度会相差数分钟。")
    p.add_argument("--no_coins", action="store_true", default=False)

    # ---- 课程学习：按评估达标晋级；frames 保留旧实验口径 ----
    p.add_argument("--curriculum", action="store_true", default=False,
                   help="开启课程学习。")
    p.add_argument("--curriculum_mode", choices=["performance", "frames"],
                   default="performance")
    p.add_argument("--curriculum_stages", type=str,
                   default="200,300,500,800,1200,2000,4000,6000",
                   help="逐档目标距离上限（米），严格递增。")
    p.add_argument("--curriculum_eval_interval", type=int, default=500_000,
                   help="两次阶段测试之间至少训练的环境帧数。")
    p.add_argument("--curriculum_min_stage_frames", type=int, default=1_000_000,
                   help="每档首次测试前至少训练的环境帧数。")
    p.add_argument("--curriculum_max_stage_frames", type=int, default=10_000_000,
                   help="每档训练帧数上限；未达标时停止并报告。")
    p.add_argument("--curriculum_eval_episodes", type=int, default=100,
                   help="每次阶段测试的单目标任务数。")
    p.add_argument("--curriculum_eval_envs", type=int, default=0,
                   help="阶段测试直接借用训练环境：每个 rank 最多用几个训练环境"
                        "并行跑测试路线，0 = 全部。每个环境的路线数是事先固定的"
                        "配额，不会因为短路线先跑完而偏向成功的路线。")
    p.add_argument("--curriculum_pass_rate", type=float, default=0.95,
                   help="阶段晋级最低成功率；1.0 表示 100%%。")
    p.add_argument("--curriculum_confirmations", type=int, default=2,
                   help="连续达标多少次才晋级。")
    p.add_argument("--curriculum_frames", type=str,
                   default="5e7:500,2e8:1000,5e8:2000,1e9:0",
                   help="[P0-3] 课程锚点，格式 '帧数:半径(米),...'（半径 0=全图）。"
                        "默认按论文校准：24h 课程 ≈ 5.18e8 帧（6000 步/s），"
                        "即 1B 预算的约 52% 处完成课程。帧数支持科学计数。")
    # 旧参数保留解析但已不参与调度（deprecated，见 _curriculum_radius）。
    p.add_argument("--curriculum_start_radius", type=float, default=500.0,
                   help="(deprecated) 由 --curriculum_frames 取代。")
    p.add_argument("--curriculum_end_radius", type=float, default=0.0,
                   help="(deprecated) 由 --curriculum_frames 取代。")
    p.add_argument("--curriculum_updates", type=int, default=50,
                   help="(deprecated) 由 --curriculum_frames 取代。")

    # ---- P2b held-out 泛化 ----
    p.add_argument("--held_out_fraction", type=float, default=0.0,
                   help="留出比例（论文 0.25）：训练目标限制在 in 集合，留出部分"
                        "供 --held_out 评估测试泛化。0=不限制。")
    return p.parse_args(argv)


@record
def main():
    # 强制用 spawn 启动多进程（干净的解释器，避免 fork 死锁/线程问题）。
    mp.set_start_method("spawn", force=True)
    train(parse_args())


if __name__ == "__main__":
    main()
