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
设计原则：引擎层与环境层（StreetLearn / CourierGame）完全不动，只把"学习器"
从 IMPALA（V-trace + off-policy 校正）换成 on-policy PPO（GAE + clipped surrogate）。

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
import math
import re
import os
import shutil
import tempfile
import time
import multiprocessing as mp
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from streetlearn.python.environment import default_config
from streetlearn.python.environment import streetlearn as streetlearn_env
from streetlearn.python.environment.streetlearn import get_action_set
from streetlearn.python.agents.ppo_agent import PPOAgent


# ImageNet 归一化常数（ResNet18 视觉编码器需要）。
# 注意：这些值在 obs_to_batch 里会被转成与 view_t 同设备的张量，避免
# "cuda 张量无法转 numpy" 的问题（早期 bug 之一）。
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 向量化环境（每个 OS 进程持有一个 StreetLearn 实例）
# --------------------------------------------------------------------------- #
def _link_or_copy_file(src, dst):
  """优先硬链接（零额外磁盘、秒级），失败（跨分区 EXDEV 等）退回复制。"""
  try:
    os.link(src, dst)
    return "link"
  except OSError:
    shutil.copy2(src, dst)
    return "copy"


def _copy_dataset_for_worker(dataset_path, worker_id, cache_dir=None):
  """给每个 worker 一份独立的（只读）LevelDB 数据集"副本"。

  背景 / 早期 bug：LevelDB 同一时刻只允许一个进程持有写锁。安装的 plyvel
  版本不支持 read_only=True，于是 N 个并行 worker 同时打开同一份数据集目录
  会报 "LOCK: Resource temporarily unavailable"。解决办法：每个 worker 把
  数据集复制一份到临时目录（每份副本有自己的 LOCK）再打开。

  磁盘优化（2026-08-30）：数据集的体积几乎全在 *.sst 文件上，而 .sst 是
  LevelDB 的**不可变**数据文件（写入后永远不会被原地改写，leveldb 只会新增
  文件或 unlink 旧文件）。因此 .sst 用**硬链接**共享 inode——N 个 worker
  共用同一份物理数据，磁盘开销约等于 0，且复制是秒级（不用真拷几个 GB）。
  只有 LOCK / CURRENT / MANIFEST / LOG / *.log 这些小文件需要真正复制：
    - LOCK 必须是独立 inode：硬链接等于共享同一把锁，等于没复制；
    - CURRENT/MANIFEST/*.log/LOG 在 leveldb 打开或恢复时可能被改写，
      硬链接原地改写会污染源数据集，必须独立文件。
  跨设备（cache_dir 与数据集不在同一分区，os.link 报 EXDEV）时自动退回
  全量复制，行为与旧版一致。

  副本按"源路径 hash + worker id"稳定命名并缓存，后续运行直接复用，避免
  重复拷贝。若源数据集有变动，需手动删除缓存目录下 streetlearn_ds_*。
  """
  src = os.path.abspath(dataset_path)
  h = hashlib.md5(src.encode("utf-8")).hexdigest()[:10]
  base = cache_dir if cache_dir else tempfile.gettempdir()
  os.makedirs(base, exist_ok=True)
  dst = os.path.join(base, "streetlearn_ds_%s_w%d" % (h, worker_id))
  if os.path.isdir(dst) and os.path.isfile(os.path.join(dst, "LOCK")):
    print("[worker %d] reusing cached dataset copy: %s" % (worker_id, dst))
    return dst
  print("[worker %d] preparing dataset copy %s -> %s (hardlink for .sst)"
        % (worker_id, src, dst))
  if os.path.isdir(dst):
    shutil.rmtree(dst)
  n_link = n_copy = 0
  for root, _dirs, files in os.walk(src):
    rel = os.path.relpath(root, src)
    dst_root = dst if rel == "." else os.path.join(dst, rel)
    os.makedirs(dst_root, exist_ok=True)
    for f in files:
      s = os.path.join(root, f)
      d = os.path.join(dst_root, f)
      if f.endswith(".sst"):
        if _link_or_copy_file(s, d) == "link":
          n_link += 1
        else:
          n_copy += 1
      else:
        shutil.copy2(s, d)
        n_copy += 1
  # 兜底：源目录可能本就没有 LOCK（从未被打开过），补一个空的独立 LOCK。
  lock = os.path.join(dst, "LOCK")
  if not os.path.isfile(lock):
    open(lock, "wb").close()
  print("[worker %d] dataset ready at %s (%d hardlinked, %d copied files)"
        % (worker_id, dst, n_link, n_copy))
  return dst


def _worker(remote, parent_remote, env_args, worker_id, num_envs):
    """Worker 进程：拥有一个独立的 StreetLearn 环境，通过 Pipe 与主进程通信。

    通信协议（由 SubprocVecEnv 发送命令）：
      - ("reset", None)        -> 返回 env.observation()
      - ("step", action_idx)   -> 返回 (obs, reward, done, info)
      - ("close", None)        -> 关闭管道并退出
    多进程让 4 个环境真正并行（跨 GIL），提升采样吞吐。
    """
    parent_remote.close()
    dataset_path, config, cache_dir = env_args
    # 并行 worker 需要独立数据集副本以避免 LevelDB 锁冲突；单 worker 可直接打开原数据。
    if num_envs > 1:
        dataset_path = _copy_dataset_for_worker(dataset_path, worker_id,
                                                cache_dir)
    cfg = copy.deepcopy(config)
    cfg["seed"] = config["seed"] + worker_id * 100003  # 用不同种子偏移让各环境 episode 去相关
    game = default_config.CreateGame(cfg["game_name"], cfg)
    env = streetlearn_env.StreetLearn(dataset_path, cfg, game)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                env.reset()
                remote.send(env.observation())
            elif cmd == "step":
                action_idx = int(data)
                action_vec = env.action_set()[action_idx]
                obs, reward, done, info = env.step(action_vec)
                remote.send((obs, float(reward), bool(done), info))
            elif cmd == "set_max_target_distance":
                # P1：课程学习——更新该 worker 游戏的目标距离上限。
                env.game.set_max_target_distance(data)
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
                np.random.seed(data + worker_id * 100003)
                env.game.set_goal_seed(data + worker_id * 100003)
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
    """

    def __init__(self, dataset_path, config, num_envs, cache_dir=None):
        self.n_envs = num_envs
        ctx = mp.get_context("spawn")
        # 每对进程用一对 Pipe 通信（主进程持 remote，子进程持 work_remote）。
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(num_envs)])
        self.ps = []
        for i, (remote, work_remote) in enumerate(
                zip(self.remotes, self.work_remotes)):
            p = ctx.Process(
                target=_worker,
                args=(work_remote, remote, (dataset_path, config, cache_dir), i,
                      num_envs))
            p.start()
            work_remote.close()  # 主进程关闭自己这端的 work_remote
            self.ps.append(p)

    def reset(self, index=None):
        if index is None:
            for remote in self.remotes:
                remote.send(("reset", None))
            return [remote.recv() for remote in self.remotes]
        self.remotes[index].send(("reset", None))
        return self.remotes[index].recv()

    def step(self, actions):
        # 锁步：所有环境同时走一步，再统一回收结果。
        for remote, a in zip(self.remotes, actions):
            remote.send(("step", a))
        results = [remote.recv() for remote in self.remotes]
        obs = [r[0] for r in results]
        rews = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return obs, rews, dones, infos

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()

    # ---- P1/P2 控制接口：把指令广播到所有 worker 的 CourierGame ----
    def set_max_target_distance(self, meters):
        """课程学习：限制目标选取半径（米）；meters=None 表示全图。"""
        for remote in self.remotes:
            remote.send(("set_max_target_distance", meters))
        for remote in self.remotes:
            remote.recv()

    def setup_held_out(self, fraction, seed=0, use_held_out=True):
        """held-out 泛化：配置目标 in/out 拆分（训练用 in，评估用 out）。"""
        for remote in self.remotes:
            remote.send(("setup_held_out", (fraction, seed, use_held_out)))
        for remote in self.remotes:
            remote.recv()

    # ---- 评估 head-to-head 控制接口 ----
    def set_goal_seed(self, seed):
        """评估 head-to-head：用独立 RNG 固定目标采样（需 Agent/Oracle 两遍
        用相同 --goal_seed 与 --seed，且 --num_envs 一致）。worker 内部会再叠加
        worker_id 偏移，使各环境目标不同但跨 run 可复现。"""
        for remote in self.remotes:
            remote.send(("set_goal_seed", seed))
        for remote in self.remotes:
            remote.recv()

    def set_single_goal_mode(self, flag):
        """评估 head-to-head：每 episode 仅 1 个目标（到达后不重抽）。"""
        for remote in self.remotes:
            remote.send(("set_single_goal_mode", flag))
        for remote in self.remotes:
            remote.recv()

    # ---- 轨迹可视化接口 ----
    def get_map(self, index=0):
        """返回整张图所有 pano 的坐标 [N,2]，每行 (lat, lng)。

        各 worker 的图相同，只向一个 worker 请求；须在 reset 之后调用。"""
        self.remotes[index].send(("get_map", None))
        return self.remotes[index].recv()


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

    公式（从后往前递推）：
      delta_t  = r_t + gamma * V(s_{t+1}) * (1-done) - V(s_t)
      A_t      = delta_t + gamma * lam * (1-done) * A_{t+1}
      return_t = A_t + V(s_t)
    lam 控制偏差-方差权衡：lam=1 偏向高方差低偏差（MC），lam=0 退化为单步 TD。
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
        last_gae = (delta
                    + gamma * lam * nextnonterminal * last_gae)
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
        logits, value, aux = agent.forward_seq(
            view, scalar, prev_r, prev_a, dones, *hs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act)                             # [T, B]
        entropy = dist.entropy().mean()

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
        # 奖励塑形范围：默认 min=100/max=200；远距离目标建议把 max 调大，
        # 让 agent 在接近目标途中就能获得非零奖励信号。
        "min_radius_meters": args.min_radius_meters,
        "max_radius_meters": args.max_radius_meters,
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
    """容错存档：磁盘不足或写入失败时不崩溃训练，仅告警并跳过。

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
    try:
        torch.save(state, path)
        return True
    except Exception as e:
        print("[PPO] WARN: torch.save failed, skip save %s: %s" % (path, e))
        # 清理可能写坏的半截文件，避免下次加载时解压报错
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return False


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


def train(args):
    """PPO 训练主循环。"""
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available())
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    latlng_bbox = parse_latlng_bbox(args.latlng_bbox, args.dataset_path)
    config = build_config(args, latlng_bbox)
    print("[PPO] latlng bbox: %s" % latlng_bbox_note(config))
    # 离散动作数：从动作集规格推算（streetlearn_fast_rotate -> 5 个动作）。
    num_actions = get_action_set(
        config["action_spec"], config["rotation_speed"]).shape[0]
    # P0：标量观测固定为 6 维（yaw sin/cos + latlng(2) + target_latlng(2)），
    # 不再包含 neighbors（agent 不看底层图）。
    scalar_dim = 6

    print("[PPO] device=%s  num_actions=%d  scalar_dim=%d  aux_heading=%s"
          % (device, num_actions, scalar_dim, args.aux_heading))

    # 创建向量化环境（多进程并行采样）。
    env = SubprocVecEnv(args.dataset_path, config, args.num_envs,
                        cache_dir=args.ds_cache_dir)

    # P2b：held-out 泛化——训练时把目标限制在 in 集合（留出 fraction 比例作为
    # 未见过的目标，供后续 --held_out 评估测试泛化）。0 表示不限制（默认）。
    if getattr(args, "held_out_fraction", 0.0) > 0.0:
        env.setup_held_out(args.held_out_fraction, seed=args.seed,
                           use_held_out=False)
        print("[PPO] held-out masking ON: training targets restricted to in-set "
              "(%.0f%% held out)" % (100.0 * args.held_out_fraction))

    obs_list = env.reset()

    # 实例化策略网络与优化器。
    agent = PPOAgent(
        num_actions=num_actions,
        hidden_size=args.hidden_size,
        aux_heading=args.aux_heading,
        pretrained_vision=args.pretrained_vision,
        vision_backbone=args.vision_backbone,
        goal_type=args.goal_type,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

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
    cur_radius = None                 # 当前课程半径（用于日志）
    global_step = 0
    update = 0
    t0 = time.time()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 结构化日志：把指标落盘成 CSV（比解析 stdout 更稳，便于画图/归档）----
    # 同时保留控制台 print 方便实时看。CSV 列与 plot_ppo_log.py 的读取顺序一致。
    metrics_csv_path = os.path.join(args.output_dir, "train_metrics.csv")
    metrics_fields = ["update", "global_step", "fps", "loss", "policy", "value",
                      "entropy", "aux", "kl", "clipfrac", "explained_var",
                      "mean_ep_reward", "mean_spl",
                      "arrival", "oracle", "coverage", "cur_radius"]
    metrics_f = open(metrics_csv_path, "w", newline="")
    metrics_w = csv.DictWriter(metrics_f, fieldnames=metrics_fields)
    metrics_w.writeheader()

    # ===== 主循环：每个 update 收集一批 rollout 并训练 =====
    while global_step < args.total_timesteps:
        update += 1
        # ---- P0-3 课程学习：按全局帧数锚定（与吞吐解耦，断点续训免疫）----
        if getattr(args, "curriculum", False):
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
                logits, value, hp, cp, hg, cg, _ = agent(
                    view_t, scalar_t, pr_t, pa_t, hp, cp, hg, cg)
            dist = Categorical(logits=logits)
            act = dist.sample()                 # [N] 按当前策略采样动作
            logp = dist.log_prob(act)           # [N] 采样动作的对数概率
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
            for e in range(n):
                rew_store[e][tt] = rewards[e]
                done_store[e][tt] = dones[e]
                ep_rewards[e] += rewards[e]
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
                    ep_rewards[e] = 0.0
                    next_obs[e] = env.reset(e)
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
            global_step += n

        # ---- 2) Bootstrap 估值：用 rollout 结束后的最后一帧前向得 last_values ----
        with torch.no_grad():
            view_t, scalar_t, pr_t, pa_t = obs_to_batch(
                obs_list, device,
                view_shape=(3, config["height"], config["width"]),
                prev_reward=prev_rewards, prev_action=prev_actions)
            # 注意解包顺序：agent.forward 返回 (logits, value, hp, cp, hg, cg, aux)，
            # value 是第二个。早期 bug：把 logits（[B,5]）误当价值会导致 compute_gae
            # 广播失败 (4,5)(4,)。
            _, last_values, _, _, _, _, _ = agent(
                view_t, scalar_t, pr_t, pa_t, hp, cp, hg, cg)
        last_values = last_values.squeeze(-1).cpu().numpy().astype(np.float32)

        # ---- 3) 计算 GAE：把 reward 序列转成 advantage / return ----
        # [P0-2] 论文语义："clipped at 1 for the purpose of gradient
        # calculations"——奖励裁剪后进 GAE/价值目标（原值量级 O(100-800)，
        # 会把价值 MSE 撑到压过策略项）。日志与 episode 统计仍用原值。
        rewards_np = np.array(rew_store, dtype=np.float32)
        values_np = np.array(val_store, dtype=np.float32)
        dones_np = np.array(done_store, dtype=bool)
        advantages, returns = compute_gae(
            np.clip(rewards_np, -1.0, 1.0), values_np, dones_np, last_values,
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
                agent, optimizer, rollout,
                (3, config["height"], config["width"]),
                device, args)
            for k in stats:
                stats[k] += s[k]
        for k in stats:
            stats[k] /= args.ppo_epochs  # 多轮取平均

        # ---- 5) 日志 ----
        if update % args.log_interval == 0:
            fps = global_step / max(1e-6, time.time() - t0)
            mean_ep = (np.mean(ep_successes).item()
                       if ep_successes else 0.0)
            mean_spl = (np.mean(ep_spl).item() if ep_spl else 0.0)
            # P2a：论文口径指标（到达全奖 / Oracle 上界 / 覆盖率）。
            mean_arrival = (np.mean(ep_arrival).item() if ep_arrival else 0.0)
            mean_oracle = (np.mean(ep_oracle).item() if ep_oracle else 0.0)
            mean_cov = (np.mean(ep_coverage).item() if ep_coverage else 0.0)
            aux_str = " aux=%.3f" % stats["aux"] if args.aux_heading else ""
            cur_rad_str = ("%.0fm" % cur_radius) if cur_radius else "FULL"
            # [P1-5] 健康监控：approx_kl<0.02 正常（>0.05 报警）；clip_fraction
            # 健康 0.05-0.2；explained_var 应随训练转正。
            print("[PPO] update=%d steps=%d fps=%.0f loss=%.3f "
                  "pol=%.3f val=%.3f ent=%.3f kl=%.4f clipf=%.3f ev=%.3f "
                  "mean_ep_reward=%.2f "
                  "mean_spl=%.3f arrival=%.2f oracle=%.2f cov=%.3f cur=%s%s"
                  % (update, global_step, fps, stats["loss"], stats["policy"],
                     stats["value"], stats["entropy"], stats["kl"],
                     stats["clipfrac"], stats["explained_var"], mean_ep, mean_spl,
                     mean_arrival, mean_oracle, mean_cov, cur_rad_str, aux_str))

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
                "mean_ep_reward": round(mean_ep, 3),
                "mean_spl": round(mean_spl, 3),
                "arrival": round(mean_arrival, 3),
                "oracle": round(mean_oracle, 3),
                "coverage": round(mean_cov, 3),
                "cur_radius": (-1 if cur_radius is None else round(cur_radius, 1)),
            })
            metrics_f.flush()  # 及时刷盘，训练中途崩溃也能保留已有指标

        # ---- 存档（容错：磁盘满也只告警、不崩训练）----
        if update % args.save_interval == 0:
            ckpt = os.path.join(args.output_dir, "ppo_iter_%d.pt" % update)
            if safe_torch_save(ckpt, {
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "global_step": global_step,
                "config": config,
                "num_actions": num_actions,
                "scalar_dim": scalar_dim,
                "hidden_size": args.hidden_size,
                "aux_heading": args.aux_heading,
                "goal_type": args.goal_type,
                "vision_backbone": args.vision_backbone,
            }, min_free_gb=args.min_free_gb):
                print("[PPO] saved checkpoint -> %s" % ckpt)
                _rotate_checkpoints(args.output_dir, args.keep_last)

    # 训练结束存最终检查点（容错）。
    ckpt = os.path.join(args.output_dir, "ppo_final.pt")
    if safe_torch_save(ckpt, {
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "global_step": global_step,
        "config": config,
        "num_actions": num_actions,
        "scalar_dim": scalar_dim,
        "hidden_size": args.hidden_size,
        "aux_heading": args.aux_heading,
        "goal_type": args.goal_type,
        "vision_backbone": args.vision_backbone,
    }, min_free_gb=args.min_free_gb):
        print("[PPO] training finished -> %s" % ckpt)
    else:
        print("[PPO] WARN: final checkpoint save failed (disk full?); "
              "train_metrics.csv is preserved.")
    metrics_f.close()  # 关闭结构化日志
    env.close()


def parse_args():
    """解析命令行超参（PPO 的标准超参集合）。"""
    p = argparse.ArgumentParser(description="StreetLearn PPO training (PyTorch)")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=4)        # 并行环境数
    p.add_argument("--steps", type=int, default=128,
                   help="rollout horizon T per env")          # 每个环境的收集步数
    p.add_argument("--total_timesteps", type=int, default=2_000_000)  # 总样本数上限
    # [P0-4] epochs 4->2、minibatches 4->1：512 帧上 16 个优化步/更新是
    # 复用过度主因；且 P0-1 后 minibatch 按 env 切，num_minibatches<=num_envs。
    p.add_argument("--ppo_epochs", type=int, default=2)      # 每批数据复用更新的轮数
    p.add_argument("--num_minibatches", type=int, default=1) # 每轮按 env 切几组（<=num_envs）
    p.add_argument("--gamma", type=float, default=0.99)     # 折扣因子
    p.add_argument("--lam", type=float, default=0.95)       # GAE lambda（偏差-方差权衡）
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
                   help="塑形奖励外边界（米）。在 min~max 之间按距离线性给奖励；"
                        "大于此值奖励为 0。对于远距离目标建议调大（如 1000~2000），"
                        "否则 agent 在远距离完全没有奖励信号。")
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
    p.add_argument("--no_coins", action="store_true", default=False)

    # ---- P0-3 课程学习（按帧锚定；旧的按 update 参数已废弃）----
    p.add_argument("--curriculum", action="store_true", default=False,
                   help="开启课程学习：目标半径按帧数锚点渐进放宽到全图。")
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
    p.add_argument("--curriculum_stages", type=str, default=None,
                   help="(deprecated) 由 --curriculum_frames 取代。")

    # ---- P2b held-out 泛化 ----
    p.add_argument("--held_out_fraction", type=float, default=0.0,
                   help="留出比例（论文 0.25）：训练目标限制在 in 集合，留出部分"
                        "供 --held_out 评估测试泛化。0=不限制。")
    return p.parse_args()


if __name__ == "__main__":
    # 强制用 spawn 启动多进程（干净的解释器，避免 fork 死锁/线程问题）。
    mp.set_start_method("spawn", force=True)
    train(parse_args())
