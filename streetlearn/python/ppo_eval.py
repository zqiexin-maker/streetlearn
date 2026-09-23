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

from streetlearn.python.ppo_train import (
    SubprocVecEnv,
    latlng_bbox_note,
    obs_to_batch,
    PPOAgent,
)

# Oracle 朝向容差（论文用 30°；与 oracle_agent.py 一致）。
TOL_BEARING = 30.0


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

    维护每个环境的"当前目标"进度，用于统计 T_1/2（覆盖一半目标距离所需步数）
    与每 episode 的到达/失败情况。
    """

    def __init__(self, n_envs):
        self.n = n_envs
        self.last_goal_id = [None] * n_envs
        self.steps_since_goal = [0] * n_envs
        self.init_dist = [0.0] * n_envs
        self.recorded_half = [False] * n_envs
        # 累计指标。
        self.arrivals, self.oracles, self.spls, self.coverages = [], [], [], []
        self.t_half_list = []          # 每个目标的 T_1/2（步）
        self.ep_success, self.ep_total = 0, 0

    def on_step(self, e, infos_e):
        """每个环境每步调用，更新 T_1/2 追踪。"""
        gid = infos_e.get("current_goal_id", None)
        if gid != self.last_goal_id[e]:
            # 新目标：重置进度。
            self.last_goal_id[e] = gid
            self.steps_since_goal[e] = 0
            self.init_dist[e] = float(infos_e.get("initial_distance_to_goal", 0.0))
            self.recorded_half[e] = False
        else:
            self.steps_since_goal[e] += 1
        if (not self.recorded_half[e]) and self.init_dist[e] > 1.0:
            min_d = float(infos_e.get("min_distance_reached", 1e9))
            if min_d <= 0.5 * self.init_dist[e]:
                self.t_half_list.append(self.steps_since_goal[e])
                self.recorded_half[e] = True

    def on_episode_end(self, infos_e):
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


def _print_and_save(mode, tag, args, arr, orc, spl, cov, t_half, fail):
    n = max(1, len(arr))
    mean_arr = float(np.mean(arr)) if arr else 0.0
    mean_orc = float(np.mean(orc)) if orc else 0.0
    mean_spl = float(np.mean(spl)) if spl else 0.0
    mean_cov = float(np.mean(cov)) if cov else 0.0
    mean_t12 = float(np.mean(t_half)) if t_half else 0.0
    fail_rate = float(fail) if isinstance(fail, float) else (
        (1.0 - float(np.mean(fail))) if fail else 0.0)
    print("=" * 68)
    print("[EVAL:%s] mode=%s  episodes=%d" % (tag, mode, n))
    print("  arrival_reward = %.2f" % mean_arr)
    print("  SPL            = %.3f" % mean_spl)
    print("  Oracle reward  = %.2f   (ceiling)" % mean_orc)
    print("  coverage       = %.3f" % mean_cov)
    print("  T_1/2 (steps)  = %.1f   (半程步数)" % mean_t12)
    print("  failure_rate   = %.3f" % fail_rate)
    print("=" * 68)
    print("EVAL_CSV mode=%s tag=%s episodes=%d arrival=%.4f spl=%.4f "
          "oracle=%.4f coverage=%.4f t_half=%.4f fail_rate=%.4f"
          % (mode, tag, n, mean_arr, mean_spl, mean_orc, mean_cov,
             mean_t12, fail_rate))
    # 落盘（按 mode+tag 覆盖写，便于多次评估对比）。
    row = [mode, tag, n, round(mean_arr, 4), round(mean_spl, 4),
           round(mean_orc, 4), round(mean_cov, 4),
           round(mean_t12, 4), round(fail_rate, 4)]
    try:
        os.makedirs(os.path.dirname(args.eval_csv) or ".", exist_ok=True)
        existing = []
        if os.path.exists(args.eval_csv):
            with open(args.eval_csv, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for r in reader:
                    if len(r) >= 2 and not (r[0] == mode and r[1] == tag):
                        existing.append(r)
        with open(args.eval_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mode", "tag", "episodes", "arrival", "spl",
                        "oracle", "coverage", "t_half", "fail_rate"])
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    num_actions = ckpt["num_actions"]
    scalar_dim = ckpt["scalar_dim"]
    hidden_size = ckpt.get("hidden_size", 512)
    aux_heading = ckpt.get("aux_heading", False)
    goal_type = ckpt.get("goal_type", "target_latlng")
    vision_backbone = ckpt.get("vision_backbone", "resnet18")
    rot_speed = float(config.get("rotation_speed", 22.5))
    # latlng 观测的归一化范围必须与训练一致：环境直接用 checkpoint 保存的 config，
    # 旧 checkpoint（全球范围）与新 checkpoint（数据集范围）各按自己的口径评估。
    print("[EVAL] latlng bbox (from checkpoint): %s" % latlng_bbox_note(config))

    env = SubprocVecEnv(args.dataset_path, config, args.num_envs)
    if args.held_out:
        env.setup_held_out(args.held_out_fraction, seed=args.seed,
                           use_held_out=True)
        print("[EVAL] held-out ON: targets drawn from OUT set "
              "(%.0f%% held out)" % (100.0 * args.held_out_fraction))

    # ---- 评估 head-to-head：固定目标采样 ----
    # Agent/Oracle 两遍评估用相同 --goal_seed（默认回退到 --seed）与 --seed，
    # 且 --num_envs 一致，即可抽到完全相同的 (起点,目标) 序列，得到严格的
    # paper-metric head-to-head 对比（BFS 上界也会一致）。
    goal_seed = args.goal_seed if args.goal_seed is not None else args.seed
    env.set_goal_seed(goal_seed)
    env.set_single_goal_mode(True)
    print("[EVAL] goal_seed=%d -> single-goal head-to-head mode ON. "
          "For a fair Agent-vs-Oracle compare, run BOTH with the SAME "
          "--seed and --goal_seed (and --num_envs)." % goal_seed)

    view_shape = (3, config["height"], config["width"])
    tracker = _EvalTracker(args.num_envs)
    t0 = time.time()

    if args.oracle:
        # ---- Oracle 实跑（差异 6）：用 bearing 真正导航 ----
        last_info = [None] * args.num_envs
        obs_list = env.reset()
        max_iters = int(args.num_episodes / args.num_envs * args.max_steps * 2) + 16
        with torch.no_grad():
            for _ in range(max_iters):
                if tracker.ep_total >= args.num_episodes:
                    break
                actions = []
                for e in range(args.num_envs):
                    if last_info[e] is None:
                        actions.append(0)  # 首步先前进，拿到首个 info(bearing)
                    else:
                        b = last_info[e].get("bearing_to_next_pano", 0.0)
                        actions.append(_bearing_to_action(b, rot_speed))
                next_obs, rewards, dones, infos = env.step(actions)
                for e in range(args.num_envs):
                    tracker.on_step(e, infos[e])
                    last_info[e] = infos[e]
                    if dones[e]:
                        tracker.on_episode_end(infos[e])
                        next_obs[e] = env.reset(e)
                        last_info[e] = None
                obs_list = next_obs
        env.close()
        _print_and_save("oracle", "actual", args, tracker.arrivals,
                        tracker.oracles, tracker.spls, tracker.coverages,
                        tracker.t_half_list,
                        (1.0 - tracker.ep_success / max(1, tracker.ep_total)))
        return

    # ---- Agent 贪心评估（argmax）----
    agent = PPOAgent(
        num_actions=num_actions,
        hidden_size=hidden_size,
        aux_heading=aux_heading,
        pretrained_vision=ckpt.get("pretrained_vision", False),
        vision_backbone=vision_backbone,
        goal_type=goal_type,
    ).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    obs_list = env.reset()
    hp, cp, hg, cg = agent.initial_state(args.num_envs, device)
    prev_rewards = np.zeros(args.num_envs, dtype=np.float32)
    prev_actions = np.zeros(args.num_envs, dtype=np.int64)

    max_iters = int(args.num_episodes / args.num_envs * args.max_steps * 2) + 16
    with torch.no_grad():
        for _ in range(max_iters):
            if tracker.ep_total >= args.num_episodes:
                break
            view_t, scalar_t, pr_t, pa_t = obs_to_batch(
                obs_list, device, view_shape, prev_rewards, prev_actions)
            logits, value, hp, cp, hg, cg, _ = agent(
                view_t, scalar_t, pr_t, pa_t, hp, cp, hg, cg)
            act = torch.argmax(logits, dim=1).cpu().tolist()
            next_obs, rewards, dones, infos = env.step(act)
            for e in range(args.num_envs):
                tracker.on_step(e, infos[e])
                if dones[e]:
                    tracker.on_episode_end(infos[e])
                    next_obs[e] = env.reset(e)
                    hp[:, e:e + 1, :] = 0.0
                    cp[:, e:e + 1, :] = 0.0
                    hg[:, e:e + 1, :] = 0.0
                    cg[:, e:e + 1, :] = 0.0
                    prev_rewards[e] = 0.0
                    prev_actions[e] = 0
                else:
                    prev_rewards[e] = float(rewards[e])
                    prev_actions[e] = int(act[e])
            obs_list = next_obs
    env.close()

    _print_and_save("normal", "agent", args, tracker.arrivals,
                    tracker.oracles, tracker.spls, tracker.coverages,
                    tracker.t_half_list,
                    (1.0 - tracker.ep_success / max(1, tracker.ep_total)))


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
    p.add_argument("--held_out", action="store_true", default=False,
                   help="泛化测试：目标只从留出(out)集合抽。")
    p.add_argument("--held_out_fraction", type=float, default=0.25)
    p.add_argument("--oracle", action="store_true", default=False,
                   help="Oracle 实跑评估（差异 6）：用 bearing 真正导航，"
                        "给出论文口径的 Oracle 实际上界与 T_1/2/失败率。")
    p.add_argument("--eval_csv", type=str, default="eval_results.csv",
                   help="把评估结果追加写入该 CSV（含 mode/tag 列），"
                        "便于多次评估后画对比图。")
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
