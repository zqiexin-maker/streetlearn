#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StreetLearn 一键启动器：训练 / 评估 / 画图，配置有默认值且支持命令行覆盖。

用法（无需记命令行参数）：
    python run.py train          # PPO 训练（读取下面 TRAIN 配置区）
    python run.py train --resume auto                # 断点恢复：接着最近一次没跑完的训练
    python run.py train --resume runs/xxx/ppo_iter_120.pt   # 从指定 checkpoint 续训
    python run.py train --nproc_per_node 6 --num_envs 32    # 6 GPU 同步训练（每卡 32 环境）
    python run.py train --curriculum_pass_rate 1.0          # 每次测试都要 100/100 才算达标
    python run.py train --exp B --seed 2                    # 对照实验预设（见 EXPERIMENTS）
    python run.py selftest       # 训练前自检：GAE 数值、串/并联网络、输入遮蔽、存档往返
    python run.py eval           # Agent 贪心评估（读取 EVAL 配置区）
    python run.py eval_oracle    # Oracle 实跑评估（同一配置区，自动加 --oracle）
    python run.py eval_all       # 先 Agent 后 Oracle，两遍同种子 head-to-head
    python run.py traj           # 轨迹可视化：跑 N 条导航，叠画在同一张全图地图上
    python run.py traj --checkpoints a.pt,b.pt        # 多 checkpoint 逐个出轨迹图
    python run.py traj --ckpt_dir runs/xxx            # 目录通配全部 .pt 逐个出图
    python run.py plot_log       # 画训练曲线（TRAIN.output_dir 下找 train_metrics.csv）
    python run.py plot_log --ckpt_dir runs/xxx        # 曲线上标注各 checkpoint 位置
    python run.py plot_curriculum                     # 画课程阶段测试图（curriculum_eval.csv）
    python run.py plot_eval      # 画评估对比图（读 EVAL.eval_csv）
    python run.py report         # 训练后一键出全部图：曲线 + 课程测试 + 评估 + 各档轨迹
    python run.py demo           # 引擎+游戏快速自检（不需要 checkpoint）

配置约定：
  - 所有配置项与底层脚本的命令行参数一一对应；设为 None 且无联动源的项不传
    （走底层默认值）。任何配置项都可以用 --键名 值 在命令行临时覆盖。
  - 【输出目录】TRAIN.output_dir 为 None 时：train 新建
    runs/ppo_<GPU数>gpu_<环境数>env_<时间戳>；续训默认写回 checkpoint 所在目录；
    eval / traj / plot_* / report 自动取 runs/ 下最近一次训练。
    这些命令的产物（CSV / PNG / npy）统一写到 TRAIN.output_dir，与 checkpoint、
    train_metrics.csv 同一文件夹；配置区里的 eval_csv / out_png / *_png 只写文件名。
  - 【自动联动】必须等于其他参数的项，配置区一律写 None，运行时自动从源参数
    复制并打印出来（详见 _resolve），改一处即全局生效：
        TRAIN.goal_timeout      = TRAIN.frame_cap        （底层建议保持一致）
        EVAL.max_steps          = TRAIN.frame_cap        （评估口径要求相等）
        EVAL.checkpoint         = TRAIN.output_dir/ppo_final.pt
        EVAL.eval_csv           = TRAIN.output_dir/<eval_csv 文件名>
        EVAL.goal_seed          = COMMON.seed            （head-to-head 对比）
        EVAL.held_out_fraction  = TRAIN.held_out_fraction（out 集合切分一致）
        TRAJ.out_dir            = TRAIN.output_dir
  - 每条命令运行前会做配置校验（_validate_*），联动或取值错误会直接报错退出，
    不会带着错误参数进入训练/评估。
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import subprocess
import sys
import time

from streetlearn.python.checkpoint_utils import find_latest_checkpoint

# ============================================================================
# 通用配置
# ============================================================================
COMMON = {
    # 数据集路径（LevelDB 目录，如 manhattan_2021_lowres）——必填
    "dataset_path": r"/home/gong/qw/manhattan_2021_lowres",
    # 并行环境数（多卡训练时是每张卡的环境数）
    "num_envs": 32,
    # 随机种子（评估时 Agent/Oracle 保持一致才能严格对比）
    "seed": 1234,
    # 是否用 GPU
    "cuda": True,
}

# ============================================================================
# 训练配置（python -m streetlearn.python.ppo_train）
# ============================================================================
TRAIN = {
    # 本次默认实验：200→6000m；每档至少 100 万帧，每 50 万帧测 100 条路；
    # 连续两次 >=95% 才晋级。命令行参数可覆盖这里的任意值，实际值另存 run_config.json。
    # 输出目录：checkpoint / train_metrics.csv / curriculum_eval.csv 都写到这里。
    # None = 训练时新建 runs/ppo_<GPU数>gpu_<环境数>env_<时间戳>（续训时 = checkpoint
    # 所在目录）；评估 / 画图时取 runs/ 下最近一次训练。
    "output_dir": None,
    # 单机同步训练使用的 GPU 数；1=普通单卡，6=由 torchrun 启动 6 个 rank。
    # COMMON.num_envs 是每张卡的环境数，实际总环境数为二者的乘积。
    "nproc_per_node": 1,
    # 断点恢复：None = 从头训；"auto" = output_dir（未指定时为 runs/ 下最近一次
    # 同配置、未跑完的训练）里最新的 checkpoint；也可直接给 .pt 路径。
    # 续训恢复网络、优化器、累计帧数和课程进度；episode 从头开始。
    "resume": None,
    # 总训练帧数上限（所有 GPU 合计）
    "total_timesteps": 100000000,
    # rollout horizon（每个环境每次收集的步数）
    "steps": 256,
    # PPO 超参（一般不用动）
    "ppo_epochs": 2,
    "num_minibatches": 4,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "lam": 0.95,
    # GAE 实现：2 = 修正后（done 统一读本步）；1 = 修复前（done 错后一位，有错），
    # 只用于修复前后的对照实验（--exp gae_v1）
    "gae_version": 2,
    "clip": 0.1,
    "ent_coef": 0.004,
    "vf_coef": 0.25,
    "max_grad_norm": 0.5,
    "hidden_size": 512,
    # 观测分辨率 / 视觉骨干（论文对齐：small_cnn + 84x84）
    "vision_backbone": "resnet18",
    "width": 160,
    "height": 120,
    "latlng_bbox": "40.695,40.788,-74.028,-73.940",
    # 单 episode 最大步数
    "frame_cap": 2000,
    # courier 目标超时步数；None = 自动取 frame_cap
    "goal_timeout": None,
    # 到达阈值(米) / 塑形窗口外边界上限(米)；必须 0 < min < cap
    "min_radius_meters": 100.0,
    "max_radius_meters": 800.0,
    # 塑形窗口自适应比例（>0 生效）：R_out = clamp(ratio × 目标初始距离,
    # min_radius×1.05, max_radius_meters)。0.4 = 复刻原版几何（200m 窗口 vs
    # 500m 课程起点）——保证"进入塑形区间那一刻奖励≈0"，否则 200m 目标配 800m
    # 窗口会让整条轨迹泡在塑形带里，塑形吞掉 90%+ 回报（实测 arrival 59 vs
    # mean_ep_reward 916）。<=0 = 固定窗口（原版行为）。
    "shaping_radius_ratio": 0.4,
    # 塑形奖励相对到达全奖的折扣（0,1]；1.0 = 不打折）
    "shaping_scale": 1.0,
    # 奖励裁剪：abs_one=每步奖励裁到 [-1,1] 后进 GAE（原版 IMPALA 口径，默认）；
    # none=用原始量级
    "reward_clip": "abs_one",
    # 关闭金币奖励（环境默认 50% 的 pano 放 +1 金币，裁剪后和到达目标一样都是
    # +1，会把策略引向"捡金币"而不是"去目标"）。--no_coins false 可重新打开。
    "no_coins": True,
    # 双 LSTM 连接方式：parallel = 当前并联结构（两路输出拼接后接双头）；
    # serial = goal LSTM 输出经 64 维 tanh 瓶颈送进 policy LSTM、双头只读 policy
    # LSTM（官方 CityNav 的连接方式，容量与输入不变）。随 checkpoint 保存。
    "model_topology": "parallel",
    # 输入遮蔽实验（目标经纬度始终保留，张量形状不变）：自身经纬度 / 自身朝向 /
    # 图像底部 10 行的邻居方向栏。随 checkpoint 保存，评估 / 轨迹自动沿用。
    "mask_position": False,
    "mask_yaw": False,
    "mask_status_bar": False,
    # 课程学习：performance=按测试成功率晋级；frames=旧版按帧自动扩距
    # performance 模式下训练目标取 (min_radius_meters, 当前档上限]，测试路线取
    # (上一档上限, 当前档上限]。
    "curriculum": True,
    "curriculum_mode": "performance",
    "curriculum_stages": "200,300,500,800,1200,2000,4000,6000",
    "curriculum_eval_interval": 500000,
    "curriculum_min_stage_frames": 1000000,
    "curriculum_max_stage_frames": 10000000,
    "curriculum_eval_episodes": 100,
    # 测试借用训练环境：每张卡最多用几个训练环境并行跑测试路线；0 = 全部
    "curriculum_eval_envs": 0,
    "curriculum_pass_rate": 0.95,
    "curriculum_confirmations": 2,
    # 仅 curriculum_mode=frames 时使用
    "curriculum_frames": "5e6:500,1.5e7:1000,3e7:2000,4.2e7:4000,4.8e7:6000",
    # held-out 留出比例（0=不分；论文 0.25）；评估的 out 集合自动与此一致
    "held_out_fraction": 0.0,
    # 每多少个 update 存一次 ppo_iter_*.pt（每次阶段测试后也会存一次）；
    # 断点恢复最多丢这么多个 update 的进度。
    "save_interval": 10,
    # 检查点轮转保留数 & 磁盘保护
    "keep_last": 3,
    "min_free_gb": 1.0,
    # 多卡集合通信超时（分钟）；NCCL 默认 10 分钟不够阶段测试/首次建副本用
    "ddp_timeout_minutes": 120,
    # 多 worker 数据集副本目录；None = 项目目录下的 tmp_ds（eval / traj 共用）。
    # 最好与数据集位于同一文件系统，便于用硬链接节省空间。
    "ds_cache_dir": None,
    # 辅助 heading 自监督任务（默认关）
    "aux_heading": None,         # 例如 True 开启
    # 视觉预训练
    "pretrained_vision": None,   # 例如 True 开启
}

# ============================================================================
# 对照实验预设：python run.py train --exp B --seed 2（bash：EXP=B SEED=2 bash run_6gpu.sh）
# 预设只改列出的几项，其余沿用 TRAIN；命令行参数优先级更高；多个预设用逗号
# 叠加（--exp B,serial）。输出目录自动带上预设名和种子。
# 每组都从头训练（不要拿一组的权重续训另一组），同样的训练帧数，至少 3 个种子。
# ============================================================================
EXPERIMENTS = {
    # 基线：TRAIN 原样（GAE 已修复、关硬币、按目标距离自适应的塑形窗口、并联、不遮蔽）。
    # 结构 / 输入实验都和它比。
    "base": {},
    # 修复前的 GAE（done 错后一位），其余同 base：和 base 比，看修复本身的效果。
    "gae_v1": {"gae_version": 1},
    # 奖励对照：固定塑形窗口（原版做法）。注意 TRAIN 默认是自适应窗口，只改
    # max_radius_meters 并不等于固定窗口，所以这里同时把 shaping_radius_ratio 置 0。
    # A：保留硬币 + 800m（共同参照）；B 对 A 看硬币的影响；C 对 B 看塑形半径的影响。
    # 若 200m 明显拖慢早期成功率，可追加 --exp C --max_radius_meters 400（单独标记）。
    "A": {"no_coins": False, "shaping_radius_ratio": 0.0, "max_radius_meters": 800.0},
    "B": {"no_coins": True, "shaping_radius_ratio": 0.0, "max_radius_meters": 800.0},
    "C": {"no_coins": True, "shaping_radius_ratio": 0.0, "max_radius_meters": 200.0},
    # 网络连接：只改连接方式，容量 / 输入不变。
    "serial": {"model_topology": "serial"},
    # 输入遮蔽：分别去掉自身位置、朝向、方向栏，以及三者全去（目标始终保留）。
    "mask_position": {"mask_position": True},
    "mask_yaw": {"mask_yaw": True},
    "mask_status_bar": {"mask_status_bar": True},
    "mask_all": {"mask_position": True, "mask_yaw": True, "mask_status_bar": True},
}

# ============================================================================
# 评估配置（python -m streetlearn.python.ppo_eval）
# ============================================================================
EVAL = {
    # checkpoint 路径；None = 自动取 TRAIN.output_dir/ppo_final.pt
    "checkpoint": None,
    # 评估 episode 数（按固定配额分给各环境）
    "num_episodes": 40,
    # 单 episode 最大步数；None = 自动取 TRAIN.frame_cap（评估口径必须相等）
    "max_steps": None,
    # 固定目标采样种子；None = 自动取 COMMON.seed（Agent/Oracle 一致即 head-to-head）
    "goal_seed": None,
    # 目标距离段：stage = checkpoint 所在课程档（如 800m 档测 (500,800] 米）；
    # full = 全图不限距离；或 "LOW,HIGH"（米）
    "goal_band": "stage",
    # held-out 泛化测试（True=目标只从留出集合抽）
    "held_out": None,            # 例如 True 开启
    # 留出比例；None = 自动取 TRAIN.held_out_fraction（保证 out 集合与训练一致）
    "held_out_fraction": None,
    # 结果 CSV 文件名（写到 TRAIN.output_dir 下；plot_eval 读这个文件）
    "eval_csv": "eval_results.csv",
}

# ============================================================================
# 轨迹可视化配置（python -m streetlearn.python.ppo_traj_viz）
# ============================================================================
TRAJ = {
    # checkpoint 路径；None = 自动取 TRAIN.output_dir/ppo_final.pt
    # （配置了 checkpoints / ckpt_dir 时此项被忽略）
    "checkpoint": None,
    # 多 checkpoint 批量出图：逗号分隔路径或列表，每个各出一张轨迹图 + 对比 CSV
    "checkpoints": None,         # 例如 "ppo_iter_180.pt,ppo_final.pt"
    # 或直接给目录 + 通配符，取该目录下全部匹配的 .pt（按轮次排序）
    "ckpt_dir": None,
    "ckpt_pattern": "*.pt",
    # 画几条轨迹（全部叠画在同一张全图地图上，按编号区分）
    "num_trajectories": 9,
    # 单条轨迹最大步数；None = 自动取 TRAIN.frame_cap（与评估口径一致）
    "max_steps": None,
    # 并行环境数（可视化只记录 env 0 的轨迹，1 即够）
    "num_envs": 1,
    # 固定目标采样种子；None = 自动取 COMMON.seed（可复现同一批任务）
    "goal_seed": None,
    # 目标距离段：stage = 每个 checkpoint 所在课程档；full = 全图；或 "LOW,HIGH"
    "goal_band": "stage",
    # 输出目录；None = 自动取 TRAIN.output_dir（与训练输出同一文件夹）
    "out_dir": None,
    # 图文件名（写到 out_dir 下，轨迹数据 trajectories.npy 也在 out_dir）
    "out_png": "traj_map.png",
}

# ============================================================================
# 画图配置
# ============================================================================
PLOT = {
    # plot_log：训练指标来源（None → 自动找 TRAIN.output_dir/train_metrics.csv）
    "log_src": None,
    # plot_log：在曲线上标注 checkpoint 位置（逗号分隔路径 / 目录+通配符）
    "checkpoints": None,         # 例如 "ppo_iter_180.pt,ppo_final.pt"
    "ckpt_dir": None,
    "ckpt_pattern": "*.pt",
    # 下面几张图的文件名都写到 TRAIN.output_dir 下
    "log_png": "ppo_curves_1.png",
    "smooth_win": 20,            # 平滑窗口
    # plot_curriculum：课程阶段测试总览图
    "curriculum_png": "curriculum_eval.png",
    # plot_eval：评估对比图
    "eval_png": "ppo_eval_triple_1.png",
}

# ============================================================================
# 以下为解析 / 校验 / 调度逻辑，一般不用改
# ============================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = "runs"

# 判断"能否接着这个 run 续训"时比较的训练参数（都会影响环境语义、网络结构或
# 课程进度）。GAE 修复之前的 run 的 run_config.json 里没有 model_topology /
# mask_* 这几项，因此不会被 --resume auto 接上（新旧 GAE 的训练历史不能混）。
_RESUME_KEYS = ("curriculum", "curriculum_mode", "curriculum_stages", "width",
                "height", "frame_cap", "goal_timeout", "min_radius_meters",
                "max_radius_meters", "shaping_radius_ratio", "shaping_scale",
                "reward_clip", "no_coins", "vision_backbone", "hidden_size",
                "aux_heading", "model_topology", "mask_position", "mask_yaw",
                "mask_status_bar", "gae_version", "latlng_bbox", "seed")
# 配置区写 None 表示"不传、走底层默认 False"的开关，比较时按 False 处理。
_FLAG_DEFAULT_FALSE = ("aux_heading",)


def _split_paths(value):
    """'a.pt,b.pt' / ['a.pt', 'b.pt'] / None -> 路径列表（去空白、去重）。"""
    if not value:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = []
    for it in items:
        it = str(it).strip()
        if it and it not in out:
            out.append(it)
    return out


class ConfigError(Exception):
    """配置错误：报错并退出，不进入训练/评估。"""


def _run_mtime(run_dir):
    """一个 run 最近一次被训练写入的时间。"""
    times = [os.path.getmtime(p) for p in
             (os.path.join(run_dir, name) for name in
              ("train_metrics.csv", "run_config.json", "checkpoint_latest.json"))
             if os.path.exists(p)]
    return max(times) if times else 0.0


def _latest_run_dir():
    """runs/ 下最近一次训练的目录（有 train_metrics.csv 的）。"""
    runs = [d for d in glob.glob(os.path.join(RUNS_DIR, "*"))
            if os.path.isfile(os.path.join(d, "train_metrics.csv"))]
    return max(runs, key=_run_mtime) if runs else None


def _has_run_artifacts(run_dir):
    return bool(run_dir) and (
        os.path.isfile(os.path.join(run_dir, "train_metrics.csv")) or
        bool(glob.glob(os.path.join(run_dir, "ppo_*.pt"))))


def _same_value(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return str(a).replace(" ", "") == str(b).replace(" ", "")


def _run_compatible(run_dir, train):
    """该 run 的 run_config.json 与本次训练参数在关键项上是否一致。"""
    try:
        with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return False
    merged = {**COMMON, **train}
    for key in _RESUME_KEYS:
        value = merged.get(key)
        if value is None and key in _FLAG_DEFAULT_FALSE:
            value = False
        if value is None:
            continue
        if key not in saved or not _same_value(value, saved[key]):
            return False
    return True


def _auto_resume_checkpoint(train):
    """runs/ 下最近一次"同配置、没跑完"的训练里最新的 checkpoint。

    已经写出 ppo_final.pt 的 run 视为跑完了（全部通过 / 某档预算用尽 /
    帧数到上限），不会被自动接上；要接着跑请显式 --resume <路径>。
    """
    runs = [os.path.dirname(p) for p in
            glob.glob(os.path.join(RUNS_DIR, "*", "run_config.json"))]
    for run_dir in sorted(runs, key=_run_mtime, reverse=True):
        if os.path.isfile(os.path.join(run_dir, "ppo_final.pt")):
            continue
        if not _run_compatible(run_dir, train):
            continue
        ckpt = find_latest_checkpoint(run_dir)
        if ckpt:
            return ckpt
    return None


def _resolve(command, overrides):
    """把必须联动的参数自动复制（None → 从源参数取值）。

    返回 (train, ev, tr) 三个解析后的副本；被自动填充的项会打印出来。
    """
    train = dict(TRAIN)
    ev = dict(EVAL)
    tr = dict(TRAJ)
    auto = []

    # 输出目录容错：TRAIN.output_dir 常写成实验名（如 ppo_6gpu_32env_xxx），
    # 但实际产物归档在 runs/<实验名> 下。这里自动补上 runs/ 前缀，
    # 省去每次改配置；远端若直接写全路径则不受影响。
    od = train.get("output_dir")
    if od and not os.path.isdir(od):
        alt = os.path.join(RUNS_DIR, os.path.basename(str(od).rstrip("/\\")))
        if os.path.isdir(alt):
            train["output_dir"] = alt
            auto.append("TRAIN.output_dir -> %s (runs/ 归档目录)" % alt)

    if command in ("train", "latest_ckpt"):
        resume = train.get("resume")
        if resume is not None and str(resume).strip().lower() in (
                "", "none", "no", "false", "0"):
            resume = None
        if resume is not None and str(resume).strip().lower() in ("auto", "latest"):
            train["_resume_auto"] = True
            if train.get("output_dir"):
                resume = find_latest_checkpoint(train["output_dir"])
                where = train["output_dir"]
            else:
                resume = _auto_resume_checkpoint(train)
                where = "runs/ 下同配置、未跑完的训练"
            auto.append("TRAIN.resume = %s" % (
                resume if resume else "无（%s 里没有可续训的 checkpoint，从头训练）"
                % where))
        train["resume"] = resume
        if resume and not train.get("output_dir"):
            # 续训默认写回原目录：CSV 接着追加，checkpoint 轮转也在同一处。
            train["output_dir"] = os.path.dirname(os.path.abspath(resume))
            auto.append("TRAIN.output_dir = checkpoint 所在目录 %s"
                        % train["output_dir"])
        if not train.get("output_dir"):
            # 对照实验的目录名带上预设名和种子，方便按组汇总多种子结果。
            tag = ("_%s_seed%d" % (train["_exp"], COMMON["seed"])
                   if train.get("_exp") else "")
            train["output_dir"] = os.path.join(
                RUNS_DIR, "ppo_%dgpu_%denv%s_%s" % (
                    train["nproc_per_node"], COMMON["num_envs"], tag,
                    time.strftime("%Y%m%d_%H%M%S")))
            auto.append("TRAIN.output_dir = %s（新建）" % train["output_dir"])
    elif command not in ("demo", "selftest") and not train.get("output_dir"):
        # 显式给了 checkpoint 时，产物写到 checkpoint 所在目录；否则取最近一次训练。
        hint = None
        if command.startswith("eval"):
            hint = ev.get("checkpoint")
        elif command == "traj":
            hint = (tr.get("checkpoint") or next(iter(_split_paths(tr.get("checkpoints"))), None)
                    or (tr.get("ckpt_dir") and os.path.join(tr["ckpt_dir"], "x.pt")))
        if hint:
            train["output_dir"] = os.path.dirname(os.path.abspath(hint))
            auto.append("TRAIN.output_dir = %s（checkpoint 所在目录）" % train["output_dir"])
        else:
            latest = _latest_run_dir()
            if latest:
                train["output_dir"] = latest
                auto.append("TRAIN.output_dir = %s（runs/ 下最近一次训练）" % latest)

    if train.get("goal_timeout") is None:
        train["goal_timeout"] = train["frame_cap"]
        auto.append("TRAIN.goal_timeout = TRAIN.frame_cap = %d" % train["frame_cap"])

    if train.get("ds_cache_dir") is None:
        train["ds_cache_dir"] = os.path.join(ROOT, "tmp_ds")
        auto.append("TRAIN.ds_cache_dir = %s" % train["ds_cache_dir"])

    if command in ("train", "latest_ckpt", "demo", "selftest") or \
            not train.get("output_dir"):
        # 下面这些只对评估 / 画图有意义（且都依赖输出目录）。
        _print_auto(auto)
        return train, ev, tr

    if ev.get("max_steps") is None:
        ev["max_steps"] = train["frame_cap"]
        auto.append("EVAL.max_steps = TRAIN.frame_cap = %d" % train["frame_cap"])

    if ev.get("checkpoint") is None:
        ev["checkpoint"] = os.path.join(train["output_dir"], "ppo_final.pt")
        auto.append("EVAL.checkpoint = %s" % ev["checkpoint"])

    # 输出目录联动：评估结果 CSV 与训练产物放在同一文件夹。
    ev["eval_csv"] = os.path.join(train["output_dir"], ev["eval_csv"])
    auto.append("EVAL.eval_csv = %s" % ev["eval_csv"])

    if ev.get("goal_seed") is None:
        ev["goal_seed"] = COMMON["seed"]
        auto.append("EVAL.goal_seed = COMMON.seed = %d" % COMMON["seed"])

    if ev.get("held_out_fraction") is None:
        ev["held_out_fraction"] = train.get("held_out_fraction", 0.0)
        auto.append("EVAL.held_out_fraction = TRAIN.held_out_fraction = %g"
                    % ev["held_out_fraction"])

    if tr.get("checkpoint") is None:
        tr["checkpoint"] = os.path.join(train["output_dir"], "ppo_final.pt")
        auto.append("TRAJ.checkpoint = %s" % tr["checkpoint"])

    if tr.get("max_steps") is None:
        tr["max_steps"] = train["frame_cap"]
        auto.append("TRAJ.max_steps = TRAIN.frame_cap = %d" % train["frame_cap"])

    if tr.get("goal_seed") is None:
        tr["goal_seed"] = COMMON["seed"]
        auto.append("TRAJ.goal_seed = COMMON.seed = %d" % COMMON["seed"])

    if tr.get("out_dir") is None:
        tr["out_dir"] = train["output_dir"]
        auto.append("TRAJ.out_dir = TRAIN.output_dir = %s" % tr["out_dir"])

    _print_auto(auto)
    return train, ev, tr


def _print_auto(auto):
    if auto:
        print("[run] 自动联动参数：")
        for a in auto:
            print("    %s" % a)


def _fail(msg):
    raise ConfigError(msg)


def _warn(msg):
    print("[run] 警告: %s" % msg)


def _check_positive(cfg, key, can_be_zero=False):
    v = cfg.get(key)
    if v is None:
        return
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        _fail("%s 必须是数字，当前是 %r" % (key, v))
    if v < 0 or (v == 0 and not can_be_zero):
        _fail("%s 必须%s 0，当前是 %r" % (key, ">= " if can_be_zero else "> ", v))


def _require_output_dir(t):
    if not t.get("output_dir"):
        _fail("没有指定 --output_dir，runs/ 下也找不到训练结果（train_metrics.csv）。"
              "请用 --output_dir 指向训练输出目录。")


def _validate_common(check_cuda=True):
    dp = COMMON.get("dataset_path")
    if not dp or not isinstance(dp, str):
        _fail("COMMON.dataset_path 未配置（数据集 LevelDB 目录，必填）")
    if not os.path.isdir(dp):
        _fail("COMMON.dataset_path 不存在: %r\n"
              "    请确认数据集已下载并解压，路径写正确。" % dp)
    if not isinstance(COMMON.get("num_envs"), int) or COMMON["num_envs"] < 1:
        _fail("COMMON.num_envs 必须是 >= 1 的整数，当前是 %r" % COMMON.get("num_envs"))
    if check_cuda and COMMON.get("cuda"):
        try:
            import torch  # noqa: F401
            if not torch.cuda.is_available():
                _warn("cuda=True 但 torch.cuda.is_available()=False，"
                      "底层脚本会自动回退 CPU（训练会很慢）。")
        except ImportError:
            pass  # 本机没装 torch（可能在别的机器上跑），跳过检查


def _validate_latlng_bbox(v):
    """校验 TRAIN.latlng_bbox：None / "dataset" / "world" / 显式四元组。

    显式写法 "lat_min,lat_max,lng_min,lng_max"（如曼哈顿
    40.695,40.788,-74.028,-73.940），要求 4 个数、顺序正确、max > min，
    且纬度落在 [-90,90]、经度落在 [-180,180]。
    """
    if v is None or v in ("dataset", "world"):
        return
    try:
        parts = [float(x) for x in str(v).split(",")]
    except ValueError:
        _fail('TRAIN.latlng_bbox 必须是 None / "dataset" / "world" / '
              "'lat_min,lat_max,lng_min,lng_max'，当前是 %r" % v)
    if len(parts) != 4:
        _fail("TRAIN.latlng_bbox 显式写法需要 4 个数"
              "（lat_min,lat_max,lng_min,lng_max），当前 %d 个: %r"
              % (len(parts), v))
    lat_min, lat_max, lng_min, lng_max = parts
    if lat_max <= lat_min or lng_max <= lng_min:
        _fail("TRAIN.latlng_bbox 的 max 必须大于 min: %r"
              "（顺序是 lat_min,lat_max,lng_min,lng_max）" % v)
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        _fail("TRAIN.latlng_bbox 纬度越界或顺序错误: %r（应在 -90..90 内）" % v)
    if not (-180.0 <= lng_min < lng_max <= 180.0):
        _fail("TRAIN.latlng_bbox 经度越界或顺序错误: %r（应在 -180..180 内）" % v)


def _validate_goal_band(name, v):
    s = str(v or "full").strip().lower()
    if s in ("full", "none", "stage"):
        return
    try:
        lower, upper = (float(x) for x in s.split(","))
    except ValueError:
        _fail("%s.goal_band 必须是 stage / full / 'LOW,HIGH'（米），当前是 %r" % (name, v))
    if not 0 <= lower < upper:
        _fail("%s.goal_band 需要 0 <= LOW < HIGH，当前是 %r" % (name, v))


def _validate_train(t, check_cuda=True):
    nproc = t.get("nproc_per_node")
    if not isinstance(nproc, int) or nproc < 1:
        _fail("TRAIN.nproc_per_node 必须是 >= 1 的整数")
    if check_cuda and nproc > 1 and not COMMON.get("cuda"):
        _warn("cuda=False 的多进程训练走 gloo/CPU，只适合调试流程。")
    if check_cuda and COMMON.get("cuda"):
        try:
            import torch
        except ImportError:
            if nproc > 1:
                _fail("多卡训练需要先安装 PyTorch CUDA 版本")
        else:
            cuda_available = torch.cuda.is_available()
            if nproc > 1:
                gpu_count = torch.cuda.device_count()
                if not cuda_available:
                    _fail("CUDA 初始化失败：PyTorch 枚举到 %d 张 GPU，但无法使用 CUDA"
                          "（torch.cuda.is_available()=False）。请检查"
                          " CUDA_VISIBLE_DEVICES、驱动和容器的 GPU 映射；"
                          "先在独立 Python 进程中测试 cuda:0。" % gpu_count)
                if gpu_count < nproc:
                    _fail("可用 GPU 数量不足：需要 %d 张，当前 %d 张" %
                          (nproc, gpu_count))
            elif not cuda_available:
                _fail("CUDA 初始化失败：torch.cuda.is_available()=False。"
                      "指定了 --cuda true，训练不会自动改用 CPU；"
                      "请先用独立 Python 进程测试 cuda:0。")
    if check_cuda and t.get("ds_cache_dir"):
        try:
            os.makedirs(t["ds_cache_dir"], exist_ok=True)
        except OSError as exc:
            _fail("无法创建数据集缓存目录 %r：%s。请用 --ds_cache_dir 指定"
                  "当前用户可写的目录。" % (t["ds_cache_dir"], exc))
    if t.get("resume") and not os.path.isfile(t["resume"]):
        _fail("TRAIN.resume 文件不存在: %r" % t["resume"])
    if (not t.get("resume") and not t.get("_resume_auto")
            and _has_run_artifacts(t.get("output_dir"))):
        _fail("输出目录 %r 里已有训练产物（train_metrics.csv / ppo_*.pt）。\n"
              "    接着这个 run 训练：加 --resume auto；\n"
              "    重新训练：换一个新的 --output_dir（或不指定，自动新建），"
              "否则旧的曲线/CSV 会被覆盖、checkpoint 会被轮转删除。"
              % t["output_dir"])
    _check_positive(t, "total_timesteps")
    _check_positive(t, "steps")
    _check_positive(t, "lr")
    _check_positive(t, "frame_cap")
    _check_positive(t, "goal_timeout")
    _check_positive(t, "hidden_size")
    _check_positive(t, "width")
    _check_positive(t, "height")
    _check_positive(t, "keep_last", can_be_zero=True)
    _check_positive(t, "min_free_gb", can_be_zero=True)
    _check_positive(t, "ddp_timeout_minutes")

    if t.get("steps") and t.get("total_timesteps") and \
            t["steps"] > t["total_timesteps"]:
        _fail("TRAIN.steps(%d) > TRAIN.total_timesteps(%d)，一轮 rollout 都凑不满"
              % (t["steps"], t["total_timesteps"]))

    nm = t.get("num_minibatches")
    if not isinstance(nm, int) or nm < 1:
        _fail("TRAIN.num_minibatches 必须是 >= 1 的整数，当前是 %r" % nm)
    if nm > COMMON["num_envs"]:
        _fail("TRAIN.num_minibatches(%d) > COMMON.num_envs(%d)，"
              "底层要求 num_minibatches <= num_envs" % (nm, COMMON["num_envs"]))

    _validate_latlng_bbox(t.get("latlng_bbox"))

    if t.get("vision_backbone") not in ("resnet18", "small_cnn"):
        _fail("TRAIN.vision_backbone 必须是 resnet18 或 small_cnn，当前是 %r"
              % t.get("vision_backbone"))
    if t.get("model_topology") not in ("parallel", "serial"):
        _fail("TRAIN.model_topology 必须是 parallel 或 serial，当前是 %r"
              % t.get("model_topology"))
    for key in ("mask_position", "mask_yaw", "mask_status_bar"):
        if not isinstance(t.get(key), bool):
            _fail("TRAIN.%s 必须是 True / False，当前是 %r" % (key, t.get(key)))
    if t.get("gae_version") not in (1, 2):
        _fail("TRAIN.gae_version 必须是 2（修正后）或 1（修复前，仅对照实验），"
              "当前是 %r" % t.get("gae_version"))
    if t.get("gae_version") == 1:
        _warn("gae_version=1：使用修复前（done 索引错后一位）的 GAE，只用于对照实验。")

    rmin, rmax = t.get("min_radius_meters"), t.get("max_radius_meters")
    if not (isinstance(rmin, (int, float)) and rmin > 0):
        _fail("TRAIN.min_radius_meters 必须 > 0，当前是 %r" % rmin)
    if not (isinstance(rmax, (int, float)) and rmax > 0):
        _fail("TRAIN.max_radius_meters 必须 > 0，当前是 %r" % rmax)
    if rmin >= rmax:
        _fail("TRAIN.min_radius_meters(%g) 必须 < TRAIN.max_radius_meters(%g)，"
              "否则塑形奖励区间为空" % (rmin, rmax))

    ratio = t.get("shaping_radius_ratio")
    if not isinstance(ratio, (int, float)) or math.isnan(ratio):
        _fail("TRAIN.shaping_radius_ratio 必须是数字（<=0 表示固定窗口），当前是 %r"
              % ratio)
    scale = t.get("shaping_scale")
    if not isinstance(scale, (int, float)) or not (0 < scale <= 1):
        _fail("TRAIN.shaping_scale 必须在 (0,1] 内，当前是 %r" % scale)
    if t.get("reward_clip") not in ("abs_one", "none"):
        _fail("TRAIN.reward_clip 必须是 'abs_one' 或 'none'，当前是 %r"
              % t.get("reward_clip"))

    hf = t.get("held_out_fraction")
    if not isinstance(hf, (int, float)) or not (0 <= hf < 1):
        _fail("TRAIN.held_out_fraction 必须在 [0,1) 内，当前是 %r" % hf)

    if t.get("curriculum") and t.get("curriculum_mode") == "performance":
        try:
            stages = [float(x.strip()) for x in t["curriculum_stages"].split(",")]
        except (ValueError, AttributeError):
            _fail("TRAIN.curriculum_stages 格式应为 '200,300,500,...'")
        if (not stages or any(not math.isfinite(x) for x in stages) or
                stages[0] <= rmin or
                any(not (a < b) for a, b in zip(stages, stages[1:]))):
            _fail("TRAIN.curriculum_stages 必须从到达阈值之外开始且严格递增")
        for key in ("curriculum_eval_interval", "curriculum_eval_episodes",
                    "curriculum_confirmations"):
            _check_positive(t, key)
        _check_positive(t, "curriculum_eval_envs", can_be_zero=True)
        _check_positive(t, "curriculum_min_stage_frames", can_be_zero=True)
        _check_positive(t, "curriculum_max_stage_frames")
        if t["curriculum_max_stage_frames"] < t["curriculum_min_stage_frames"]:
            _fail("每档最大训练帧数不能小于首次测试前的帧数")
        if not 0 < t["curriculum_pass_rate"] <= 1:
            _fail("TRAIN.curriculum_pass_rate 必须在 (0,1] 内")
    elif t.get("curriculum") and t.get("curriculum_mode") == "frames":
        spec = str(t.get("curriculum_frames") or "")
        points = []
        try:
            for item in spec.split(","):
                f, r = item.strip().split(":")
                points.append((float(f), float(r)))
        except ValueError:
            _fail('TRAIN.curriculum_frames 格式错误: %r\n'
                  "    正确格式: '帧数:半径(米),...'（帧数支持科学计数，半径 0=全图），"
                  "如 '5e7:500,2e8:1000,1e9:0'" % spec)
        if not points:
            _fail("TRAIN.curriculum_frames 为空：开启 curriculum 时至少要一个锚点")
        if any(r < 0 for _, r in points):
            _fail("TRAIN.curriculum_frames 中的半径不能为负: %r" % spec)
        frames = [f for f, _ in points]
        if frames != sorted(frames):
            _fail("TRAIN.curriculum_frames 的锚点帧数必须严格递增: %r" % spec)
    elif t.get("curriculum"):
        _fail("TRAIN.curriculum_mode 必须是 performance 或 frames")


def _validate_eval(t, ev):
    _require_output_dir(t)
    ckpt = ev.get("checkpoint")
    if not ckpt or not isinstance(ckpt, str):
        _fail("EVAL.checkpoint 未解析出路径（配 None 会自动取 "
              "TRAIN.output_dir/ppo_final.pt）")
    if not os.path.isfile(ckpt):
        _fail("EVAL.checkpoint 文件不存在: %r\n"
              "    先跑 python run.py train，或把 checkpoint 指到已有 .pt 文件。" % ckpt)
    _check_positive(ev, "num_episodes")
    _check_positive(ev, "max_steps")
    gs = ev.get("goal_seed")
    if gs is not None and (not isinstance(gs, int) or gs < 0):
        _fail("EVAL.goal_seed 必须是非负整数或 None，当前是 %r" % gs)
    _validate_goal_band("EVAL", ev.get("goal_band"))
    hf = ev.get("held_out_fraction")
    if not isinstance(hf, (int, float)) or not (0 <= hf < 1):
        _fail("EVAL.held_out_fraction 必须在 [0,1) 内，当前是 %r" % hf)
    if ev.get("held_out") and not t.get("held_out_fraction"):
        _warn("EVAL.held_out=True 但 TRAIN.held_out_fraction=0：训练没留 out 集合，"
              "评估的'held-out'目标其实训练时见过，不是严格的泛化测试。")
    if ev.get("max_steps") != t.get("frame_cap"):
        _fail("EVAL.max_steps(%r) 必须等于 TRAIN.frame_cap(%r)（论文口径要求；"
              "把 EVAL.max_steps 设为 None 可自动联动）"
              % (ev.get("max_steps"), t.get("frame_cap")))


def _validate_traj(t, tr):
    _require_output_dir(t)
    _validate_goal_band("TRAJ", tr.get("goal_band"))
    multi = _split_paths(tr.get("checkpoints"))
    if tr.get("ckpt_dir"):
        if not os.path.isdir(tr["ckpt_dir"]):
            _fail("TRAJ.ckpt_dir 目录不存在: %r" % tr["ckpt_dir"])
        multi += sorted(glob.glob(os.path.join(tr["ckpt_dir"],
                                               tr.get("ckpt_pattern") or "*.pt")))
    if multi:
        # 多 checkpoint 批量模式：逐个校验存在性
        for p in multi:
            if not os.path.isfile(p):
                _fail("TRAJ 批量模式下的 checkpoint 文件不存在: %r" % p)
        return
    ckpt = tr.get("checkpoint")
    if not ckpt or not isinstance(ckpt, str):
        _fail("TRAJ.checkpoint 未解析出路径（配 None 会自动取 "
              "TRAIN.output_dir/ppo_final.pt）")
    if not os.path.isfile(ckpt):
        _fail("TRAJ.checkpoint 文件不存在: %r\n"
              "    先跑 python run.py train，或把 checkpoint 指到已有 .pt 文件。" % ckpt)
    _check_positive(tr, "num_trajectories")
    _check_positive(tr, "max_steps")
    gs = tr.get("goal_seed")
    if gs is not None and (not isinstance(gs, int) or gs < 0):
        _fail("TRAJ.goal_seed 必须是非负整数或 None，当前是 %r" % gs)
    if not isinstance(tr.get("num_envs"), int) or tr["num_envs"] < 1:
        _fail("TRAJ.num_envs 必须是 >= 1 的整数，当前是 %r" % tr.get("num_envs"))
    if not tr.get("out_dir") or not isinstance(tr.get("out_dir"), str):
        _fail("TRAJ.out_dir 必须是非空字符串，当前是 %r" % tr.get("out_dir"))
    if not tr.get("out_png") or not isinstance(tr.get("out_png"), str):
        _fail("TRAJ.out_png 必须是非空字符串，当前是 %r" % tr.get("out_png"))
    if tr.get("max_steps") != t.get("frame_cap"):
        _fail("TRAJ.max_steps(%r) 必须等于 TRAIN.frame_cap(%r)（与评估口径一致；"
              "把 TRAJ.max_steps 设为 None 可自动联动）"
              % (tr.get("max_steps"), t.get("frame_cap")))


def _validate_plot_log(t):
    _require_output_dir(t)
    src = PLOT.get("log_src") or os.path.join(t["output_dir"], "train_metrics.csv")
    if os.path.isfile(src):
        return src
    alt = os.path.join(t["output_dir"], "train.log")
    if os.path.isfile(alt):
        return alt
    _fail("在 TRAIN.output_dir(%r) 下找不到 train_metrics.csv 或 train.log，"
          "无法画训练曲线。先跑 python run.py train，或用 PLOT.log_src 指定来源。"
          % t["output_dir"])


def _validate_plot_curriculum(t):
    _require_output_dir(t)
    src = os.path.join(t["output_dir"], "curriculum_eval.csv")
    if not os.path.isfile(src):
        _fail("%r 不存在：只有 curriculum_mode=performance 的训练、且已经做过"
              "至少一次阶段测试才会有它。" % src)
    return src


def _validate_plot_eval(ev):
    src = ev.get("eval_csv")
    if not src or not os.path.isfile(src):
        _fail("EVAL.eval_csv(%r) 不存在，无法画评估对比图。"
              "先跑 python run.py eval_all 生成结果。" % src)
    return src


def _flag(key, val):
    """None/False → 不传；True → 只传开关；其他 → --key value。"""
    if val is None or val is False:
        return []
    if val is True:
        return ["--%s" % key]
    return ["--%s" % key, str(val)]


def _run(module, extra_args, nproc_per_node=1, log_dir=None):
    if nproc_per_node > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node", str(nproc_per_node), "--tee", "3"]
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            cmd += ["--log-dir", log_dir]
        cmd += ["-m", module] + extra_args
    else:
        cmd = [sys.executable, "-m", module] + extra_args
    print("[run] %s\n" % " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def cmd_train(train, ev, tr=None):
    args = []
    for k, v in {**COMMON, **train}.items():
        if k == "nproc_per_node" or k.startswith("_"):
            continue
        # 子开关未开时，不传其配套参数（如 --curriculum 关时不传 curriculum_frames）
        if k == "curriculum_frames" and (
                not train.get("curriculum") or train.get("curriculum_mode") != "frames"):
            continue
        if k == "held_out_fraction" and not train.get("held_out_fraction"):
            continue
        args += _flag(k, v)
    if train["resume"]:
        print("[run] 续训：%s -> 输出目录 %s" % (train["resume"], train["output_dir"]))
    return _run("streetlearn.python.ppo_train", args,
                nproc_per_node=train["nproc_per_node"],
                log_dir=os.path.join(ROOT, train["output_dir"],
                                     "torchrun_logs"))


def _eval_args(ev, train, oracle=False):
    args = []
    cfg = {**COMMON, **ev}
    cfg["ds_cache_dir"] = train["ds_cache_dir"]
    if oracle:
        cfg["oracle"] = True
    for k, v in cfg.items():
        # held-out 未开启时不传 held_out_fraction，避免误改采样行为
        if k == "held_out_fraction" and not cfg.get("held_out"):
            continue
        args += _flag(k, v)
    return args


def cmd_eval(train, ev, tr=None, oracle=False):
    return _run("streetlearn.python.ppo_eval", _eval_args(ev, train, oracle))


def cmd_eval_all(train, ev, tr=None):
    rc1 = cmd_eval(train, ev, oracle=False)
    rc2 = cmd_eval(train, ev, oracle=True)
    print("\n[run] Agent 与 Oracle 两遍使用相同 seed/goal_seed/num_envs/num_episodes，"
          "即严格 head-to-head 对比。")
    return rc1 or rc2


def cmd_traj(train, ev, tr):
    """轨迹可视化：跑 TRAJ.num_trajectories 条导航，叠画在同一张全图地图上。

    支持多 checkpoint 批量：TRAJ.checkpoints（逗号分隔）或 TRAJ.ckpt_dir
    （目录通配），每个 checkpoint 各出一张轨迹图 + traj_summary.csv 对比表。

    参数取 {COMMON, TRAJ} 的并集（checkpoint/max_steps/goal_seed 已联动），
    与 ppo_traj_viz.py 的命令行参数一一对应。
    """
    args = []
    cfg = {**COMMON, **tr}
    # num_envs 联动取 TRAJ 自己的（默认 1，可视化够用且最省内存）。
    cfg["num_envs"] = tr["num_envs"]
    cfg["ds_cache_dir"] = train["ds_cache_dir"]
    # checkpoints 列表 -> 逗号分隔字符串传给底层脚本
    if cfg.get("checkpoints") is not None:
        cfg["checkpoints"] = ",".join(_split_paths(cfg["checkpoints"]))
    for k, v in cfg.items():
        args += _flag(k, v)
    return _run("streetlearn.python.ppo_traj_viz", args)


def cmd_plot_log(train, ev, tr=None):
    src = _validate_plot_log(train)
    png = os.path.join(train["output_dir"], PLOT["log_png"])
    args = [src, png, "--smooth", str(PLOT["smooth_win"])]
    # 可选：在曲线上标注多个 checkpoint 的位置（竖虚线 + 标签）
    if PLOT.get("checkpoints"):
        args += ["--checkpoints", ",".join(_split_paths(PLOT["checkpoints"]))]
    if PLOT.get("ckpt_dir"):
        args += ["--ckpt-dir", str(PLOT["ckpt_dir"])]
        if PLOT.get("ckpt_pattern"):
            args += ["--ckpt-pattern", str(PLOT["ckpt_pattern"])]
    # plot_ppo_log 的平滑参数名是 --smooth（传 --smooth_win 会被当成输出文件名）。
    return _run("streetlearn.tools.plot_ppo_log", args)


def cmd_plot_curriculum(train, ev=None, tr=None):
    src = _validate_plot_curriculum(train)
    png = os.path.join(train["output_dir"], PLOT["curriculum_png"])
    return _run("streetlearn.tools.plot_curriculum", [src, png])


def cmd_plot_eval(train, ev, tr=None):
    src = _validate_plot_eval(ev)
    png = os.path.join(train["output_dir"], PLOT["eval_png"])
    return _run("streetlearn.tools.plot_ppo_eval", [src, png])


def cmd_report(train, ev, tr):
    """训练结束后一键出图（缺输入的步骤跳过，其余照常）：

      1) plot_log         训练曲线
      2) plot_curriculum  课程阶段测试总览
      3) eval_all + plot_eval   EVAL.checkpoint（默认 ppo_final.pt）在
                                EVAL.goal_band（默认所在档位）上的 Agent / Oracle 对比
      4) traj             各档最优权重 ppo_best_stage_*.pt 各出一张轨迹图
                          （TRAJ.goal_band 默认 stage：每张图用该档自己的距离段）
    """
    od = train["output_dir"]
    steps = []

    def step(name, fn):
        print("\n[run] ===== report: %s =====" % name, flush=True)
        try:
            rc = fn()
        except ConfigError as exc:
            print("[run] 跳过 %s：%s" % (name, exc))
            steps.append((name, "skipped"))
            return
        steps.append((name, "ok" if rc == 0 else "failed (rc=%s)" % rc))

    step("plot_log", lambda: cmd_plot_log(train, ev))
    step("plot_curriculum", lambda: cmd_plot_curriculum(train))

    def run_eval_all():
        _validate_common(check_cuda=True)
        _validate_eval(train, ev)
        return cmd_eval_all(train, ev) or cmd_plot_eval(train, ev)
    step("eval_all + plot_eval", run_eval_all)

    def run_traj():
        best = sorted(glob.glob(os.path.join(od, "ppo_best_stage_*.pt")))
        tr2 = dict(tr)
        if best and not (tr.get("checkpoints") or tr.get("ckpt_dir")):
            tr2["checkpoints"] = ",".join(best)
        _validate_common(check_cuda=True)
        _validate_traj(train, tr2)
        return cmd_traj(train, ev, tr2)
    step("traj", run_traj)

    print("\n[run] report 结果（输出目录 %s）：" % od)
    for name, status in steps:
        print("    %-22s %s" % (name, status))
    return 0 if all(s in ("ok", "skipped") for _, s in steps) else 1


def cmd_latest_ckpt(train, ev=None, tr=None):
    """打印 --resume auto 会选中的 checkpoint（没有则输出空行），供 shell 脚本用。"""
    print(train.get("resume") or "", file=sys.__stdout__)
    return 0


def cmd_demo(train, ev, tr=None):
    return _run("streetlearn.tools.run_nav",
                ["--dataset_path", COMMON["dataset_path"], "--mode", "game"])


def cmd_selftest(train=None, ev=None, tr=None):
    """训练前自检（只用 CPU，十几秒）：GAE 数值与回合边界、并联 / 串联网络的
    形状与梯度、段内 done 清空记忆、输入遮蔽、checkpoint 保存再加载。"""
    if not os.path.isdir(os.path.join(ROOT, "tests")):
        _fail("找不到 tests/ 目录（上传代码时漏了？）")
    return _run("unittest", ["discover", "-s", "tests", "-t", ".", "-v"])


# 每条命令: (执行函数, 需要的校验组, 可在命令行覆盖的配置组)
COMMANDS = {
    "train":       (cmd_train,     ("common", "train"), ("common", "train")),
    "eval":        (cmd_eval,      ("common", "eval"), ("common", "train", "eval")),
    "eval_oracle": (lambda t, e, tr: cmd_eval(t, e, oracle=True),
                    ("common", "eval"), ("common", "train", "eval")),
    "eval_all":    (cmd_eval_all,  ("common", "eval"), ("common", "train", "eval")),
    "traj":        (cmd_traj,      ("common", "traj"), ("common", "train", "traj")),
    "plot_log":    (cmd_plot_log,  (), ("train", "plot")),
    "plot_curriculum": (cmd_plot_curriculum, (), ("train", "plot")),
    "plot_eval":   (cmd_plot_eval, (), ("train", "eval", "plot")),
    "report":      (cmd_report,    (), ("common", "train", "eval", "traj", "plot")),
    "latest_ckpt": (cmd_latest_ckpt, (), ("common", "train")),
    "demo":        (cmd_demo,      ("common",), ("common",)),
    "selftest":    (cmd_selftest,  (), ()),
}


def _pop_exp(argv):
    """从命令行取出 --exp NAME[,NAME...]（或 --exp=...），返回 (预设名列表, 其余参数)。"""
    names, rest = [], []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--exp" or arg.startswith("--exp="):
            if arg == "--exp":
                if i + 1 >= len(argv):
                    _fail("--exp 后面要跟预设名：%s" % " / ".join(EXPERIMENTS))
                value, i = argv[i + 1], i + 2
            else:
                value, i = arg.split("=", 1)[1], i + 1
            names += [n.strip() for n in value.split(",") if n.strip()]
            continue
        rest.append(arg)
        i += 1
    return names, rest


def _apply_experiment(command, names):
    """把 --exp 预设合进 TRAIN（在命令行逐项覆盖之前，命令行优先）。"""
    if not names:
        return
    if command not in ("train", "latest_ckpt"):
        _fail("--exp 只用于 train / latest_ckpt；评估和画图按 checkpoint 里保存的"
              "设置运行，不需要预设。")
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        _fail("未知的实验预设 %s，可选：%s" % (", ".join(unknown), " / ".join(EXPERIMENTS)))
    for name in names:
        TRAIN.update(EXPERIMENTS[name])
        print("[run] 实验预设 %s: %s" % (name, EXPERIMENTS[name] or "（TRAIN 原样）"))
    TRAIN["_exp"] = "-".join(names)


def _cli_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("1", "true", "yes", "on"):
        return True
    if value.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("布尔值应为 true 或 false")


def _cli_int(value):
    """整数参数也接受科学计数（帧数常写成 1e8 / 5e5）。"""
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("需要整数，收到 %r" % value)
    if not math.isfinite(number) or number != int(number):
        raise argparse.ArgumentTypeError("需要整数，收到 %r" % value)
    return int(number)


def _apply_cli_overrides(command, argv):
    """--键名 值 覆盖配置区；同名键在多个配置组里时一起覆盖（如 goal_band）。"""
    groups = {"common": COMMON, "train": TRAIN, "eval": EVAL,
              "traj": TRAJ, "plot": PLOT}
    owners = {}
    for group_name in COMMANDS[command][2]:
        for key in groups[group_name]:
            owners.setdefault(key, []).append(group_name)
    type_hints = {"goal_timeout": _cli_int, "checkpoint": str, "max_steps": _cli_int,
                  "goal_seed": _cli_int, "held_out_fraction": float,
                  "held_out": _cli_bool, "aux_heading": _cli_bool,
                  "pretrained_vision": _cli_bool, "resume": str,
                  "out_dir": str, "log_src": str, "output_dir": str,
                  "checkpoints": str, "ckpt_dir": str, "ckpt_pattern": str}
    parser = argparse.ArgumentParser(prog="python run.py " + command,
                                     description="命令行参数覆盖 run.py 中的默认配置")
    for key, group_names in owners.items():
        values = [groups[g][key] for g in group_names]
        value = next((v for v in values if v is not None), None)
        value_type = (type_hints.get(key, str) if value is None else
                      _cli_bool if isinstance(value, bool) else
                      _cli_int if isinstance(value, int) else type(value))
        options = {"default": argparse.SUPPRESS, "type": value_type}
        if value_type is _cli_bool:
            options.update(nargs="?", const=True)
        parser.add_argument("--" + key, **options)
    overrides = vars(parser.parse_args(argv))
    for key, value in overrides.items():
        for group_name in owners[key]:
            groups[group_name][key] = value
    if overrides:
        print("[run] 命令行覆盖: %s" % overrides)
    return overrides


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("可用命令: %s" % " / ".join(COMMANDS))
        sys.exit(1)

    # 相对路径（output_dir / checkpoint 等）统一以项目根目录为基准：校验与
    # 子进程（cwd=ROOT）看到的是同一个文件，从任意目录启动都一致。
    os.chdir(ROOT)
    command = sys.argv[1]
    fn, checks, _ = COMMANDS[command]
    try:
        # latest_ckpt 的标准输出只留 checkpoint 路径，其余提示都转到 stderr。
        quiet = contextlib.redirect_stdout(sys.stderr) if command == "latest_ckpt" \
            else contextlib.nullcontext()
        with quiet:
            exp_names, argv = _pop_exp(sys.argv[2:])
            _apply_experiment(command, exp_names)
            overrides = _apply_cli_overrides(command, argv)
            if command == "latest_ckpt":
                TRAIN["resume"] = "auto"
            train, ev, tr = _resolve(command, overrides)
            if "common" in checks:
                _validate_common(check_cuda=command != "train")
            if command == "train":
                _validate_train(train, check_cuda=True)
            if "eval" in checks:
                _validate_eval(train, ev)
            if "traj" in checks:
                _validate_traj(train, tr)
            # plot_* 的校验在各自 cmd 里做（返回解析出的 src）
        sys.exit(fn(train, ev, tr))
    except ConfigError as e:
        print("\n[run] 配置错误: %s\n" % e, file=sys.stderr if command == "latest_ckpt"
              else sys.stdout)
        sys.exit(2)


if __name__ == "__main__":
    main()
