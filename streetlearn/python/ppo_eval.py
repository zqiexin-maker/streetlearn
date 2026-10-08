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

"""StreetLearn 评估脚本（论文口径：Agent 贪心 + Oracle 实跑 + T_1/2 + 失败率）。

与训练循环解耦，独立运行。覆盖论文 Table 2 / 4 的核心评估口径：
  - Agent 贪心（argmax）评估：到达全奖、BFS Oracle 上界、SPL、覆盖率、
    T_1/2（覆盖一半目标距离所需步数）、任务失败率。
  - Oracle 实跑（差异 6）：用环境 info 里的 bearing_to_next_pano 驱动离散动作，
    真正把 Oracle 导航跑一遍，给出 Oracle 的"实际"到达/SPL/T_1/2/失败率，
    而不只是 BFS 节点数上界。
  - held-out 泛化：目标只从未训练见过的 out 集合抽（--held_out）。
  - 目标距离段（--goal_band）：full = 全图（默认）；stage = checkpoint 记录的
    课程档位，如 300m 档测 (200, 300] 米；也可写 "LOW,HIGH"（米）。

每个环境事先分到固定数量的 episode（配额），各自只跑自己的那几条：被统计的
就是"每个环境的前 k 条路线"。旧做法是"谁先跑完谁领下一条、凑够 N 条就停"，
短（多为成功）的路线先跑完，长（多为失败）的路线被截掉，成功率会系统性偏高，
Agent / Oracle 两遍也对不上同一批路线。

用法示例：
  # Agent 贪心评估（与训练 frame_cap 一致的 --max_steps）
  python -m streetlearn.python.ppo_eval \\
      --checkpoint ppo_ckpts/ppo_final.pt \\
      --dataset_path .../manhattan_2021_lowres \\
      --cuda --num_envs 4 --num_episodes 40 --max_steps 5000

  # Oracle 实跑评估（论文上界，真实导航）
  python -m streetlearn.python.ppo_eval --checkpoint ppo_ckpts/ppo_final.pt \\
      --dataset_path .../manhattan_2021_lowres --oracle --num_envs 4 \\
      --num_episodes 40 --max_steps 5000
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

from streetlearn.python.agents.ppo_agent import (
    agent_kwargs_from_checkpoint,
    describe_agent,
)
from streetlearn.python.ppo_train import (
    SubprocVecEnv,
    latlng_bbox_note,
    obs_to_batch,
    PPOAgent,
    reward_config_note,
)

# Oracle 朝向容差（论文用 30°；与 oracle_agent.py 一致）。
TOL_BEARING = 30.0

# success_steps：成功 episode 到达目标所用的动作数（均值；没有成功时为 0）。
# 只看它不够——只会完成简单任务也能得到很短的平均步数，要和 fail_rate 一起看。
EVAL_CSV_FIELDS = ["mode", "tag", "episodes", "arrival", "spl", "oracle",
                   "coverage", "t_half", "fail_rate", "success_steps", "goal_band",
                   "checkpoint"]


# --------------------------------------------------------------------------- #
# 固定配额的 episode 调度（课程测试与独立评估共用）
# --------------------------------------------------------------------------- #
def split_quota(total, n_envs):
    """把 total 条路线尽量均分给 n_envs 个环境（余数给前几个）。"""
    base, extra = divmod(int(total), int(n_envs))
    return [base + int(i < extra) for i in range(int(n_envs))]


class AgentPolicy(object):
    """贪心（argmax）Agent：按环境维护双 LSTM 记忆与上一步 reward/action。"""

    def __init__(self, agent, n_envs, device, view_shape):
        self.agent = agent
        self.device = device
        self.view_shape = view_shape
        self.state = list(agent.initial_state(n_envs, device))  # hp, cp, hg, cg
        self.prev_rewards = np.zeros(n_envs, dtype=np.float32)
        self.prev_actions = np.zeros(n_envs, dtype=np.int64)

    def reset(self, e):
        for s in self.state:
            s[:, e:e + 1, :] = 0.0
        self.prev_rewards[e] = 0.0
        self.prev_actions[e] = 0

    def act(self, indices, obs):
        """只对 indices 这几个环境前向（已跑完配额的环境不参与）。"""
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        view, scalar, pr, pa = obs_to_batch(
            obs, self.device, self.view_shape,
            self.prev_rewards[indices], self.prev_actions[indices])
        with torch.no_grad():
            logits, _, hp, cp, hg, cg, _ = self.agent(
                view, scalar, pr, pa, *(s.index_select(1, idx) for s in self.state))
        for s, new in zip(self.state, (hp, cp, hg, cg)):
            s.index_copy_(1, idx, new)
        return torch.argmax(logits, dim=1).cpu().tolist()

    def observe(self, e, action, reward, done, info):
        if done:
            self.reset(e)
        else:
            self.prev_rewards[e] = float(reward)
            self.prev_actions[e] = int(action)


class OraclePolicy(object):
    """Oracle：沿 BFS 最短路的下一个 pano 方位角转向 / 前进。"""

    def __init__(self, n_envs, rot_speed):
        self.rot_speed = rot_speed
        self.last_info = [None] * n_envs

    def reset(self, e):
        self.last_info[e] = None

    def act(self, indices, obs):
        # 首步先前进，拿到首个 info（含 bearing_to_next_pano）。
        return [0 if self.last_info[e] is None else
                _bearing_to_action(self.last_info[e].get("bearing_to_next_pano", 0.0),
                                   self.rot_speed)
                for e in indices]

    def observe(self, e, action, reward, done, info):
        self.last_info[e] = None if done else info


def run_quota_episodes(env, policy, quotas, frame_cap, on_step=None,
                       on_episode_end=None, on_reset=None):
    """环境 e 恰好连续跑 quotas[e] 个 episode；配额为 0 的环境不参与。

    每个 episode 至多 frame_cap + 1 步，因此总迭代数有上界；超过上界说明
    episode 没能结束（环境异常），直接报错而不是无限循环。
    """
    remaining = list(quotas)
    active = [e for e in range(len(quotas)) if remaining[e] > 0]
    if not active:
        return
    obs = [None] * len(quotas)

    def start(indices):
        for e, o in zip(indices, env.reset_indices(indices)):
            obs[e] = o
            policy.reset(e)
            if on_reset:
                on_reset(e)

    start(active)
    max_iters = max(quotas) * (frame_cap + 1) + 16
    for _ in range(max_iters):
        if not active:
            return
        actions = policy.act(active, [obs[e] for e in active])
        next_obs, rewards, dones, infos = env.step(actions, indices=active)
        restart = []
        for k, e in enumerate(active):
            obs[e] = next_obs[k]
            if on_step:
                on_step(e, infos[k])
            policy.observe(e, actions[k], rewards[k], dones[k], infos[k])
            if dones[k]:
                if on_episode_end:
                    on_episode_end(e, infos[k])
                remaining[e] -= 1
                if remaining[e] > 0:
                    restart.append(e)
        if restart:
            start(restart)
        active = [e for e in active if remaining[e] > 0]
    if active:
        raise RuntimeError("%d env(s) did not finish their episodes within %d steps"
                           % (len(active), max_iters))


# --------------------------------------------------------------------------- #
# 课程阶段测试（训练中调用，借用训练环境）
# --------------------------------------------------------------------------- #
def evaluate_curriculum(agent, env, config, lower_m, upper_m, seed,
                        episode_count, max_envs=0, rank=0, world_size=1):
    """在 (lower_m, upper_m] 米距离段内跑本 rank 分到的单目标测试路线。

    直接借用训练用的 SubprocVecEnv（测试期间训练环境本来就闲着），省去每次
    测试重新起进程、建图、算扩展图的几分钟。调用方负责在测试后用
    set_train_mode 恢复训练采样并重置全部环境。

    返回本 rank 的计数/求和量，由训练脚本跨 rank 汇总。
    """
    started = time.monotonic()
    empty = {"episodes": 0, "successes": 0, "spl_sum": 0.0,
             "success_steps_sum": 0, "max_goal_distance": 0.0,
             "max_success_distance": 0.0, "max_success_steps": 0,
             "seconds": 0.0}
    if episode_count <= 0:
        return empty
    n_used = env.n_envs if max_envs <= 0 else min(max_envs, env.n_envs)
    n_used = min(n_used, episode_count)
    quotas = split_quota(episode_count, n_used) + [0] * (env.n_envs - n_used)
    print("[PPO] curriculum eval rank=%d/%d: band=(%.0f,%.0f]m episodes=%d "
          "envs=%d" % (rank, world_size, lower_m, upper_m, episode_count, n_used),
          flush=True)

    records = []  # (success, goal_distance, spl, steps)

    def on_episode_end(e, info):
        distance = float(info["initial_distance_to_goal"])
        if not lower_m < distance <= upper_m:
            raise RuntimeError("Evaluation goal outside distance band (%.0f, %.0f]: "
                               "%.2f" % (lower_m, upper_m, distance))
        success = int(info.get("num_successes", 0)) > 0
        records.append((success, distance,
                        float(info.get("spl_without_last_goal", 0.0) or 0.0),
                        int(info["num_steps_this_goal"]) if success else 0))

    env.set_eval_mode(seed, lower_m, upper_m)
    device = next(agent.parameters()).device
    policy = AgentPolicy(agent, env.n_envs, device,
                         (3, config["height"], config["width"]))
    was_training = agent.training
    agent.eval()
    try:
        run_quota_episodes(env, policy, quotas, config["frame_cap"],
                           on_episode_end=on_episode_end)
    finally:
        agent.train(was_training)
    if len(records) != episode_count:
        raise RuntimeError("Evaluation finished %d/%d episodes"
                           % (len(records), episode_count))
    seconds = time.monotonic() - started
    successes = [r for r in records if r[0]]
    print("[PPO] curriculum eval rank=%d/%d finished: %d/%d reached in %.1fs"
          % (rank, world_size, len(successes), len(records), seconds), flush=True)
    return {
        "episodes": len(records),
        "successes": len(successes),
        "spl_sum": float(sum(r[2] for r in records)),
        "success_steps_sum": sum(r[3] for r in successes),
        "max_goal_distance": max(r[1] for r in records),
        "max_success_distance": max((r[1] for r in successes), default=0.0),
        "max_success_steps": max((r[3] for r in successes), default=0),
        "seconds": seconds,
    }


# --------------------------------------------------------------------------- #
# 独立评估（python run.py eval / eval_oracle / eval_all）
# --------------------------------------------------------------------------- #
def resolve_goal_band(spec, ckpt):
    """--goal_band -> (lower_m, upper_m) 或 None（全图）。

    full：全图（不限距离，原行为）；stage：checkpoint 记录的课程档位
    （best / final 是它所在的那一档，before_stage_K 是即将开始的第 K 档）；
    "LOW,HIGH"：显式距离段（米），目标严格落在 (LOW, HIGH] 内。
    """
    spec = str(spec or "full").strip().lower()
    if spec in ("full", "none", ""):
        return None
    if spec == "stage":
        state = ckpt.get("curriculum_state") or {}
        stages = state.get("stages") or []
        if not stages:
            print("[EVAL] checkpoint has no curriculum stage; --goal_band stage "
                  "falls back to the full map")
            return None
        idx = min(int(state.get("index", 0)), len(stages) - 1)
        lower = (stages[idx - 1] if idx > 0
                 else ckpt["config"].get("min_radius_meters", 100.0))
        return float(lower), float(stages[idx])
    try:
        lower, upper = (float(x) for x in spec.split(","))
    except ValueError:
        raise ValueError("--goal_band must be full / stage / 'LOW,HIGH' (meters), "
                         "got %r" % spec)
    if not 0 <= lower < upper:
        raise ValueError("--goal_band needs 0 <= LOW < HIGH, got %r" % spec)
    return lower, upper


def band_label(band):
    return "full" if band is None else "%g-%gm" % band


def _bearing_to_action(bearing, rot_speed):
    """把相对方位角（yaw-relative，[-180,180]）映射成离散动作索引。

    动作集顺序（streetlearn_fast_rotate, rot_speed=22.5）：
      0 = 前进; 1 = 慢左(-22.5); 2 = 快左(-67.5);
      3 = 慢右(+22.5); 4 = 快右(+67.5)。
    """
    if abs(bearing) <= TOL_BEARING:
        return 0  # 朝向容差内 → 前进
    fast_thr = TOL_BEARING + 2.0 * rot_speed
    if bearing > 0:  # 需右转
        return 4 if bearing > fast_thr else 3
    else:  # 需左转
        return 2 if bearing < -fast_thr else 1


class _EvalTracker:
    """跨 episode 评估指标聚合器（Agent 与 Oracle 共用）。

    T_1/2 = 第一次走到"离目标不超过初始直线距离一半"时已用的步数，
    用 info 里的实时距离 distance_to_goal 判断（min_distance_reached 只在
    塑形窗口内更新，窗口小于半程时会漏判）。
    """

    def __init__(self, n_envs):
        self.half_recorded = [False] * n_envs
        self.arrival_recorded = [False] * n_envs
        # 累计指标。
        self.arrivals, self.oracles, self.spls, self.coverages = [], [], [], []
        self.t_half_list = []          # 每个目标的 T_1/2（步）
        self.success_steps = []        # 成功 episode 到达目标所用的步数
        self.ep_success, self.ep_total = 0, 0

    def on_reset(self, e):
        self.half_recorded[e] = False
        self.arrival_recorded[e] = False

    def on_step(self, e, infos_e):
        if (not self.arrival_recorded[e]
                and int(infos_e.get("num_successes", 0) or 0) > 0):
            self.success_steps.append(int(infos_e.get("num_steps_this_goal", 0)))
            self.arrival_recorded[e] = True
        if self.half_recorded[e]:
            return
        init = float(infos_e.get("initial_distance_to_goal", 0.0) or 0.0)
        now = infos_e.get("distance_to_goal")
        if now is None:  # 旧环境没有实时距离时退回 min_distance_reached
            now = infos_e.get("min_distance_reached", 1e9)
        if init > 1.0 and float(now) <= 0.5 * init:
            self.t_half_list.append(int(infos_e.get("num_steps_this_goal", 0)))
            self.half_recorded[e] = True

    def on_episode_end(self, e, infos_e):
        """episode 结束（done）时调用，汇总该 episode。"""
        self.ep_total += 1
        arr = infos_e.get("arrival_reward", 0.0)
        self.arrivals.append(arr)
        self.oracles.append(infos_e.get("oracle_reward", 0.0))
        # 评估固定为单目标模式：info["spl"] = Σspl/(num_successes+1) 是多目标口径
        # （把"未完成的最后一个目标"记 0），单目标到达时会把 SPL 砍半。单目标下
        # 每 episode 至多 1 个目标，应取 spl_without_last_goal（未到达时为 0）。
        s = infos_e.get("spl_without_last_goal", 0.0)
        self.spls.append(s if s else 0.0)
        self.coverages.append(infos_e.get("coverage", 0.0))
        if arr and arr > 0:
            self.ep_success += 1


def _print_and_save(mode, tag, args, arr, orc, spl, cov, t_half, fail, band,
                    success_steps=()):
    n = max(1, len(arr))
    mean_arr = float(np.mean(arr)) if arr else 0.0
    mean_orc = float(np.mean(orc)) if orc else 0.0
    mean_spl = float(np.mean(spl)) if spl else 0.0
    mean_cov = float(np.mean(cov)) if cov else 0.0
    mean_t12 = float(np.mean(t_half)) if t_half else 0.0
    mean_steps = float(np.mean(success_steps)) if success_steps else 0.0
    fail_rate = float(fail)
    print("=" * 68)
    print("[EVAL:%s] mode=%s  episodes=%d  goal_band=%s" % (tag, mode, n, band_label(band)))
    print("  arrival_reward = %.2f" % mean_arr)
    print("  SPL            = %.3f" % mean_spl)
    print("  Oracle reward  = %.2f   (ceiling)" % mean_orc)
    print("  coverage       = %.3f" % mean_cov)
    print("  T_1/2 (steps)  = %.1f   (半程步数)" % mean_t12)
    print("  failure_rate   = %.3f" % fail_rate)
    print("  success_steps  = %s   (成功 episode 的到达步数，%d 条)"
          % ("%.1f" % mean_steps if success_steps else "n/a", len(success_steps)))
    print("=" * 68)
    print("EVAL_CSV mode=%s tag=%s episodes=%d arrival=%.4f spl=%.4f "
          "oracle=%.4f coverage=%.4f t_half=%.4f fail_rate=%.4f success_steps=%.1f"
          % (mode, tag, n, mean_arr, mean_spl, mean_orc, mean_cov,
             mean_t12, fail_rate, mean_steps))
    # 落盘（按 mode+tag 覆盖写，便于多次评估对比）。
    row = {"mode": mode, "tag": tag, "episodes": n,
           "arrival": round(mean_arr, 4), "spl": round(mean_spl, 4),
           "oracle": round(mean_orc, 4), "coverage": round(mean_cov, 4),
           "t_half": round(mean_t12, 4), "fail_rate": round(fail_rate, 4),
           "success_steps": round(mean_steps, 1),
           "goal_band": band_label(band),
           "checkpoint": os.path.basename(args.checkpoint)}
    try:
        os.makedirs(os.path.dirname(args.eval_csv) or ".", exist_ok=True)
        existing = []
        if os.path.exists(args.eval_csv):
            with open(args.eval_csv, "r", newline="") as f:
                existing = [r for r in csv.DictReader(f)
                            if not (r.get("mode") == mode and r.get("tag") == tag)]
        with open(args.eval_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EVAL_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in existing:
                w.writerow(r)
            w.writerow(row)
        print("[EVAL] wrote result -> %s (mode=%s tag=%s)" % (
            args.eval_csv, mode, tag))
    except Exception as e:
        print("[EVAL] warn: failed to write %s: %s" % (args.eval_csv, e))


def run_eval(args):
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available())
                          else "cpu")

    # 本项目自己存的 checkpoint（含 config 字典），PyTorch 2.6+ 默认的
    # weights_only=True 会拒绝加载。
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    rot_speed = float(config.get("rotation_speed", 22.5))
    band = resolve_goal_band(args.goal_band, ckpt)
    # latlng 观测的归一化范围必须与训练一致：环境直接用 checkpoint 保存的 config，
    # 旧 checkpoint（全球范围）与新 checkpoint（数据集范围）各按自己的口径评估。
    print("[EVAL] latlng bbox (from checkpoint): %s" % latlng_bbox_note(config))
    print("[EVAL] reward config (from checkpoint): %s" % reward_config_note(config))
    print("[EVAL] goal band: %s" % band_label(band))

    n_envs = max(1, min(args.num_envs, args.num_episodes))
    quotas = split_quota(args.num_episodes, n_envs)
    goal_seed = args.goal_seed if args.goal_seed is not None else args.seed
    env = SubprocVecEnv(args.dataset_path, config, n_envs,
                        cache_dir=args.ds_cache_dir)
    try:
        if args.held_out:
            env.setup_held_out(args.held_out_fraction, seed=args.seed,
                               use_held_out=True)
            print("[EVAL] held-out ON: targets drawn from OUT set "
                  "(%.0f%% held out)" % (100.0 * args.held_out_fraction))

        # ---- 评估 head-to-head：固定目标采样 ----
        # Agent/Oracle 两遍评估用相同 --goal_seed（默认回退到 --seed）、--seed、
        # --num_envs、--num_episodes：每个环境的 (起点, 目标) 序列由种子决定，
        # 再配合固定配额，两遍跑的是同一批路线。
        env.set_goal_seed(goal_seed)
        env.set_single_goal_mode(True)
        # 到达即结束 episode（与课程阶段测试一致）：单目标下到达后不会再有奖励，
        # 各项指标不变，只是不用空跑到 frame_cap。
        env.set_end_on_success(True)
        if band is not None:
            env.set_target_distance_band(*band)
        print("[EVAL] goal_seed=%d single-goal mode, %d episodes on %d envs "
              "(fixed per-env quota)." % (goal_seed, args.num_episodes, n_envs))

        if args.oracle:
            policy = OraclePolicy(n_envs, rot_speed)
        else:
            # 网络结构 / 输入遮蔽按 checkpoint 元数据重建（旧 checkpoint = 并联、不遮蔽）。
            agent = PPOAgent(**agent_kwargs_from_checkpoint(ckpt)).to(device)
            agent.load_state_dict(ckpt["agent"])
            agent.eval()
            print("[EVAL] model (from checkpoint): %s" % describe_agent(agent))
            policy = AgentPolicy(agent, n_envs, device,
                                 (3, config["height"], config["width"]))

        tracker = _EvalTracker(n_envs)
        t0 = time.time()
        run_quota_episodes(env, policy, quotas, config["frame_cap"],
                           on_step=tracker.on_step,
                           on_episode_end=tracker.on_episode_end,
                           on_reset=tracker.on_reset)
        print("[EVAL] %d episodes in %.1fs" % (tracker.ep_total, time.time() - t0))
    finally:
        env.close()

    mode, tag = ("oracle", "actual") if args.oracle else ("normal", "agent")
    _print_and_save(mode, tag, args, tracker.arrivals,
                    tracker.oracles, tracker.spls, tracker.coverages,
                    tracker.t_half_list,
                    (1.0 - tracker.ep_success / max(1, tracker.ep_total)), band,
                    tracker.success_steps)


def parse_args():
    p = argparse.ArgumentParser(description="StreetLearn PPO evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--num_episodes", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=1000,
                   help="单 episode 最大步数（必须=训练时的 frame_cap）。")
    p.add_argument("--cuda", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--goal_seed", type=int, default=None,
                   help="固定目标采样 RNG 种子；Agent/Oracle 两遍用相同值即可"
                        "抽到完全相同的目标序列，实现严格 head-to-head。默认=None"
                        "时使用 --seed。需配合相同的 --seed 与 --num_envs。")
    p.add_argument("--goal_band", type=str, default="full",
                   help="目标距离段：full=全图（默认）；stage=checkpoint 所在课程档"
                        "（如 300m 档测 (200,300] 米）；或 'LOW,HIGH'（米）。")
    p.add_argument("--held_out", action="store_true", default=False,
                   help="泛化测试：目标只从留出(out)集合抽。")
    p.add_argument("--held_out_fraction", type=float, default=0.25)
    p.add_argument("--oracle", action="store_true", default=False,
                   help="Oracle 实跑评估（差异 6）：用 bearing 真正导航，"
                        "给出论文口径的 Oracle 实际上界与 T_1/2/失败率。")
    p.add_argument("--ds_cache_dir", type=str, default=None,
                   help="多环境数据集副本目录（同训练的 --ds_cache_dir）。默认系统"
                        "临时目录；与数据集不在同一分区时会整份复制，可能写满 /tmp。")
    p.add_argument("--eval_csv", type=str, default="eval_results.csv",
                   help="把评估结果追加写入该 CSV（含 mode/tag 列），"
                        "便于多次评估后画对比图。")
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
