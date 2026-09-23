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
# See the License for the specific language governing permissions and
# limitations under the License.

"""轨迹可视化评估：加载 checkpoint，跑 N 条完整导航，把全部轨迹叠画在整张街道图上。

对标 RL_Mapless 的 test() 可视化（Draw_result.draw_trajs）：把 agent 的实际
行走轨迹叠在街景节点图上，直观展示导航策略的行为质量。

与 ppo_eval.py 的区别：
  - ppo_eval：40 集 x 数值指标（arrival/SPL/T_1/2/fail_rate），看数字。
  - 本脚本 ：少量 episode x 全图轨迹图 + 每条轨迹的成功/步数摘要，看行为。

实现要点：
  - 复用 ppo_train 的 SubprocVecEnv / obs_to_batch / PPOAgent，
    与训练/评估完全同构，不重复造轮子。
  - 底图：参照 RL_Mapless 的街景绘图（draw_trajs_street_multi），通过 worker 的
    get_map 接口取整张图所有 pano 的 (lat, lng)，画成灰色散点背景（经度方向乘
    cos(lat)，地图比例不变形）；每个目标外画一圈到达半径（min_radius_meters）。
  - 轨迹记录：info 里没有经纬度，但 obs 的 "latlng"/"target_latlng" 每步都有
    （按 config 的 bbox 缩放到 [0,1]），反缩放回真实经纬度即可，不用改引擎层。
  - 单目标模式（set_single_goal_mode）：每 episode 只追 1 个目标，到达即
    成功，轨迹语义与 RL_Mapless 的 demo 一致。
  - 动作取 argmax（贪心），与 ppo_eval 的 Agent 口径一致；非采样。
  - 画法：轨迹颜色表示成败（绿色实线=到达，红色虚线=未到达），编号标在起点
    与目标旁；右侧列出每条轨迹的步数 / SPL / 起点到目标的直线距离。

用法示例：
  python -m streetlearn.python.ppo_traj_viz \
      --checkpoint ppo_ckpts/ppo_final.pt \
      --dataset_path .../manhattan_2021_lowres \
      --num_envs 1 --num_trajectories 9 --max_steps 1000 \
      --out_dir ppo_ckpts --out_png traj_map.png
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from streetlearn.python.ppo_train import (
    SubprocVecEnv,
    latlng_bbox_note,
    obs_to_batch,
    PPOAgent,
)

# 避免负号渲染成方块；字体顺序与 plot_ppo_eval.py 一致（英文标签用 DejaVu Sans，
# CJK 字体留作中文字形回退）。
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "SimHei", "Microsoft YaHei", "Arial Unicode MS",
]

# 配色（浅色底）：pano 节点灰色散点是低调底图；轨迹颜色只表示成败（状态色），
# 并配合线型（实线/虚线）与文字，不靠颜色也能分辨。
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
FRAME = "#e1e0d9"
RULE = "#c3c2b7"
NODE = "#898781"
REACHED = "#0ca30c"
FAILED = "#d03b3b"
FAILED_DASH = (0, (4, 2))
EARTH_RADIUS_M = 6371000.0


# ---------------------------------------------------------------------------
# worker 侧辅助：从 info 拿不到经纬度，这里把 lat/lng 直接塞进 info。
# StreetLearn.step() 里 info 由 game.get_info() 生成，不含坐标；但 obs 字典
# 的 "latlng"/"target_latlng" 是 [0,1] 缩放值（bbox 归一化），可反缩放回真实
# 经纬度（config 里有 bbox_*_min/max）。本脚本按此反算，避免改引擎层。
# ---------------------------------------------------------------------------
def _rescale_latlng(scaled, env_config):
    """[0,1] 缩放经纬度 -> 真实 (lat, lng)。env_config 为该环境的 config dict。"""
    lat = scaled[0] * (env_config["bbox_lat_max"] - env_config["bbox_lat_min"]) \
        + env_config["bbox_lat_min"]
    lng = scaled[1] * (env_config["bbox_lng_max"] - env_config["bbox_lng_min"]) \
        + env_config["bbox_lng_min"]
    return float(lat), float(lng)


def _haversine_m(a, b):
    """两点 (lat, lng) 之间的大圆距离（米）。"""
    lat1, lng1, lat2, lng2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def _halo(width):
    """背景色描边：轨迹/编号压在街道或其他轨迹上时依然清楚。"""
    return [pe.withStroke(linewidth=width, foreground=SURFACE)]


def _circle(lat, lng, radius_m, n=64):
    """以 (lat, lng) 为圆心、radius_m 米为半径的圆，返回 (lats, lngs)。"""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    dlat = radius_m / (math.pi / 180 * EARTH_RADIUS_M)
    dlng = dlat / math.cos(math.radians(lat))
    return lat + dlat * np.sin(t), lng + dlng * np.cos(t)


def _scale_bar(ax, lat0):
    """左下角比例尺：长度取不超过地图宽度 1/5 的整数公里数。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    km_per_deg_lng = (math.pi / 180 * EARTH_RADIUS_M / 1000
                      * math.cos(math.radians(lat0)))
    target_km = (x1 - x0) * km_per_deg_lng / 5
    nice = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50)
    km = max([k for k in nice if k <= target_km] or [nice[0]])
    bx, by = x0 + 0.05 * (x1 - x0), y0 + 0.03 * (y1 - y0)
    ax.plot([bx, bx + km / km_per_deg_lng], [by, by], color=INK, lw=2.2,
            solid_capstyle="butt", zorder=8, path_effects=_halo(4.5))
    ax.annotate(("%g km" % km), (bx, by), xytext=(0, 5),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=9, color=INK, zorder=8, path_effects=_halo(3))


def _side_panel(ax, trajectories, n_panos, height_in, goal_radius_m):
    """右侧：总成功率、图例、每条轨迹一行的明细表。"""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def y_at(inches):  # 距面板顶部 inches 英寸处的 y 坐标
        return 1.0 - inches / height_in

    n = len(trajectories)
    n_ok = sum(bool(tr["success"]) for tr in trajectories)
    ax.text(0, y_at(0.05), "Reached goal: %d / %d (%.0f%%)"
            % (n_ok, n, 100.0 * n_ok / max(1, n)),
            ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    if n:
        ax.text(0, y_at(0.4), "Mean steps %.0f  ·  mean SPL %.2f"
                % (np.mean([tr["steps"] for tr in trajectories]),
                   np.mean([tr["spl"] for tr in trajectories])),
                ha="left", va="top", fontsize=10.5, color=INK_2)

    keys = [
        (dict(color=REACHED, lw=1.8), "Reached goal"),
        (dict(color=FAILED, lw=1.8, ls=FAILED_DASH), "Did not reach goal"),
        (dict(ls="none", marker="o", ms=7, mfc=SURFACE, mec=INK, mew=1.5),
         "Start"),
        (dict(ls="none", marker="*", ms=14, mfc=INK_2, mec=SURFACE, mew=1.0),
         "Goal (filled with the outcome color)"),
        (dict(ls="none", marker="o", ms=12, mfc="none", mec=INK_2, mew=1.0),
         "Arrival radius around goal (%g m)" % goal_radius_m),
        (dict(ls="none", marker="o", ms=3, mfc=NODE, mec="none", alpha=0.6),
         "Panorama nodes (%s)" % format(n_panos, ",")),
    ]
    inches = 1.0
    for style, text in keys:
        y = y_at(inches)
        if "marker" in style:
            ax.plot([0.04], [y], **style)
        else:
            ax.plot([0.0, 0.08], [y, y], **style)
        ax.text(0.11, y, text, ha="left", va="center", fontsize=10, color=INK)
        inches += 0.28

    inches += 0.25
    columns = [(0.0, "left", "#"), (0.09, "left", "Result"),
               (0.60, "right", "Steps"), (0.77, "right", "SPL"),
               (1.0, "right", "Start-goal")]
    for x, ha, name in columns:
        ax.text(x, y_at(inches), name, ha=ha, va="center", fontsize=9.5,
                color=INK_2)
    ax.plot([0, 1], [y_at(inches + 0.17)] * 2, color=RULE, lw=0.8)
    inches += 0.4

    available = height_in - inches
    row_h = min(0.28, available / max(n, 1))
    shown = n if row_h >= 0.18 else max(0, int(available / 0.18) - 1)
    row_h = max(row_h, 0.18)
    for i, tr in enumerate(trajectories[:shown], start=1):
        y = y_at(inches)
        ok = bool(tr["success"])
        dist_km = _haversine_m(tr["latlng"][0], tr["goal"]) / 1000.0
        ax.text(0.0, y, str(i), ha="left", va="center", fontsize=10,
                fontweight="bold", color=INK)
        ax.plot([0.09, 0.16], [y, y], color=REACHED if ok else FAILED,
                lw=1.8, ls="-" if ok else FAILED_DASH)
        cells = [(0.19, "left", "reached" if ok else "failed"),
                 (0.60, "right", "%d" % tr["steps"]),
                 (0.77, "right", ("%.2f" % tr["spl"]) if ok else "-"),
                 (1.0, "right", "%.2f km" % dist_km)]
        for x, ha, text in cells:
            ax.text(x, y, text, ha=ha, va="center", fontsize=10, color=INK)
        inches += row_h
    if shown < n:
        ax.text(0.0, y_at(inches), "... %d more in trajectories.npy"
                % (n - shown), ha="left", va="center", fontsize=9.5,
                color=INK_2)


def draw_traj_map(coords, trajectories, png, goal_radius_m=100.0, subtitle=""):
    """把全部轨迹叠画在整张 pano 节点散点图上并保存 PNG。

    Args:
      coords: [N, 2] 所有 pano 的 (lat, lng)。
      trajectories: run_traj_viz 记录的轨迹列表
        （latlng / goal / steps / success / spl）。
      png: 输出路径。
      goal_radius_m: 到达半径（米），画在每个目标外；<=0 不画。
      subtitle: 标题下方的补充说明（checkpoint、步数上限等）。
    """
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    (lat_min, lng_min), (lat_max, lng_max) = coords.min(0), coords.max(0)
    pad_lat = 0.02 * max(lat_max - lat_min, 1e-4)
    pad_lng = 0.02 * max(lng_max - lng_min, 1e-4)
    lat_min, lat_max = lat_min - pad_lat, lat_max + pad_lat
    lng_min, lng_max = lng_min - pad_lng, lng_max + pad_lng
    lat0 = 0.5 * (lat_min + lat_max)

    # 等距圆柱投影：纬度 1° 的实地长度是经度 1° 的 1/cos(lat0) 倍。
    aspect = 1.0 / math.cos(math.radians(lat0))
    data_w, data_h = (lng_max - lng_min) / aspect, lat_max - lat_min
    scale = min(10.0 / data_w, 12.0 / data_h)  # 地图区域最大 10 x 12 英寸
    map_w, map_h = data_w * scale, data_h * scale
    left, right, top, bottom, gap, side_w = 0.85, 0.3, 1.2, 0.6, 0.55, 4.4
    fig_w = left + map_w + gap + side_w + right
    fig_h = max(top + map_h + bottom, 7.5)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)

    fig.text(left / fig_w, 1 - 0.3 / fig_h,
             "Agent trajectories on the full street graph",
             ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(left / fig_w, 1 - 0.72 / fig_h,
             "%d trajectories · greedy policy · one goal per episode%s"
             % (len(trajectories), ("  ·  " + subtitle) if subtitle else ""),
             ha="left", va="top", fontsize=10.5, color=INK_2)

    ax = fig.add_axes([left / fig_w, (fig_h - top - map_h) / fig_h,
                       map_w / fig_w, map_h / fig_h])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(lng_min, lng_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(aspect, adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_color(FRAME)
    ax.tick_params(colors=INK_2, labelsize=8, length=2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("longitude", fontsize=9, color=INK_2)
    ax.set_ylabel("latitude", fontsize=9, color=INK_2)

    # 底图：参照 RL_Mapless 的街景绘图，全部 pano 节点画成灰色小散点
    # （栅格化，几万个点也不会让 PNG 渲染变慢）。
    ax.scatter(coords[:, 1], coords[:, 0], s=1.2, c=NODE, alpha=0.6,
               linewidths=0, zorder=1, rasterized=True)

    for i, tr in enumerate(trajectories, start=1):
        path = np.asarray(tr["latlng"], dtype=np.float64).reshape(-1, 2)
        ok = bool(tr["success"])
        color = REACHED if ok else FAILED
        ax.plot(path[:, 1], path[:, 0], color=color, lw=1.8,
                ls="-" if ok else FAILED_DASH, solid_joinstyle="round",
                solid_capstyle="round", zorder=3, path_effects=_halo(3.8))
        ax.plot(path[0, 1], path[0, 0], ls="none", marker="o", ms=7,
                mfc=SURFACE, mec=INK, mew=1.5, zorder=5)
        ax.plot(tr["goal"][1], tr["goal"][0], ls="none", marker="*", ms=15,
                mfc=color, mec=SURFACE, mew=1.2, zorder=6)
        if goal_radius_m > 0:
            ring_lat, ring_lng = _circle(tr["goal"][0], tr["goal"][1],
                                         goal_radius_m)
            ax.plot(ring_lng, ring_lat, color=color, lw=1.0, zorder=4)
        # 编号同时标在起点和目标旁：一眼对上"哪个起点去哪个目标"。
        for (lat, lng), dy in ((path[0], 7), (tr["goal"], -8)):
            ax.annotate(str(i), (lng, lat), xytext=(7, dy),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=9.5, fontweight="bold", color=INK, zorder=7,
                        path_effects=_halo(2.8))
    _scale_bar(ax, lat0)

    side = fig.add_axes([(left + map_w + gap) / fig_w, bottom / fig_h,
                         side_w / fig_w, (fig_h - top - bottom) / fig_h])
    _side_panel(side, trajectories, len(coords), fig_h - top - bottom,
                goal_radius_m)

    os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
    fig.savefig(png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def run_traj_viz(args):
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available())
                          else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    num_actions = ckpt["num_actions"]
    hidden_size = ckpt.get("hidden_size", 512)
    aux_heading = ckpt.get("aux_heading", False)
    goal_type = ckpt.get("goal_type", "target_latlng")
    vision_backbone = ckpt.get("vision_backbone", "resnet18")
    # 环境与 _rescale_latlng 都用 checkpoint 自带 config 的 bbox，新旧口径各自正确。
    print("[TRAJ] latlng bbox (from checkpoint): %s" % latlng_bbox_note(config))

    # ---- 环境：单环境即可（可视化不需要吞吐）；复用评估同款控制接口 ----
    env = SubprocVecEnv(args.dataset_path, config, args.num_envs)
    goal_seed = args.goal_seed if args.goal_seed is not None else args.seed
    env.set_goal_seed(goal_seed)
    env.set_single_goal_mode(True)
    print("[TRAJ] goal_seed=%d single-goal mode ON (到达即成功，episode 结束由"
          " frame_cap 控制)" % goal_seed)

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

    view_shape = (3, config["height"], config["width"])
    n_traj = args.num_trajectories
    trajectories = []       # 每条: {"latlng": [(lat,lng), ...], "goal": (lat,lng),
                            #        "steps": int, "success": bool, "spl": float}
    t0 = time.time()
    ep = 0
    # 起点也算轨迹的一部分（reset 后、step 前记录一次）。
    obs_list = env.reset()
    # 整张图所有 pano 的坐标（画全图散点底图用；图在 reset 时建立，之后不变）。
    coords = env.get_map()
    print("[TRAJ] street graph: %d panos" % len(coords))
    while ep < n_traj:
        hp, cp, hg, cg = agent.initial_state(args.num_envs, device)
        prev_rewards = np.zeros(args.num_envs, dtype=np.float32)
        prev_actions = np.zeros(args.num_envs, dtype=np.int64)

        # episode 起始：记录起点坐标与目标坐标。
        o0 = obs_list[0]
        latlngs = [_rescale_latlng(np.asarray(o0["latlng"]).reshape(-1),
                                   config)]
        goal_ll = _rescale_latlng(
            np.asarray(o0["target_latlng"]).reshape(-1), config)

        success = False
        spl = 0.0
        steps_used = 0
        for t in range(args.max_steps):
            view_t, scalar_t, pr_t, pa_t = obs_to_batch(
                obs_list, device, view_shape, prev_rewards, prev_actions)
            with torch.no_grad():
                logits, _, hp, cp, hg, cg, _ = agent(
                    view_t, scalar_t, pr_t, pa_t, hp, cp, hg, cg)
            act = torch.argmax(logits, dim=1).cpu().tolist()  # 贪心，与 eval 一致
            next_obs, rewards, dones, infos = env.step(act)
            o = next_obs[0]
            latlngs.append(_rescale_latlng(
                np.asarray(o["latlng"]).reshape(-1), config))
            steps_used = t + 1
            prev_rewards[0] = float(rewards[0])
            prev_actions[0] = int(act[0])
            # 到达即停：CourierGame.done() 只在超时/帧耗尽时为 True，到达目标
            # 不会结束 episode，这里用 num_successes 自行检测（单目标模式下
            # 到达后 _reached_this_episode 置位，num_successes 恰为 0→1）。
            if int(infos[0].get("num_successes", 0)) > 0:
                success = True
                spl = float(infos[0].get("spl_without_last_goal", 0.0)
                            or infos[0].get("spl", 0.0))
                break
            if dones[0]:
                break
            obs_list = next_obs

        trajectories.append({
            "latlng": latlngs,
            "goal": goal_ll,
            "steps": steps_used,
            "success": bool(success),
            "spl": spl,
        })
        ep += 1
        print("[TRAJ] episode %d/%d: steps=%d success=%s spl=%.3f (%.1fs)"
              % (ep, n_traj, steps_used, success, spl, time.time() - t0))
        obs_list = env.reset()  # 下一条轨迹重新开局

    env.close()

    # ---- 汇总打印 ----
    n_succ = sum(tr["success"] for tr in trajectories)
    print("=" * 68)
    print("[TRAJ] trajectories=%d  success=%d/%d (%.0f%%)  "
          "ave_steps=%.1f  mean_spl=%.3f"
          % (n_traj, n_succ, n_traj, 100.0 * n_succ / max(1, n_traj),
             float(np.mean([tr["steps"] for tr in trajectories])),
             float(np.mean([tr["spl"] for tr in trajectories]))))
    print("=" * 68)

    # ---- 画全图轨迹 ----
    os.makedirs(args.out_dir, exist_ok=True)
    png = os.path.join(args.out_dir, args.out_png)
    draw_traj_map(coords, trajectories, png,
                  goal_radius_m=float(config.get("min_radius_meters", 100.0)),
                  subtitle="%s · max %d steps · goal_seed %d"
                  % (os.path.basename(args.checkpoint), args.max_steps,
                     goal_seed))
    print("[TRAJ] wrote trajectories map -> %s" % png)

    # ---- 轨迹数据落盘（便于事后重画/分析） ----
    np.save(os.path.join(args.out_dir, "trajectories.npy"),
            np.array(trajectories, dtype=object), allow_pickle=True)
    print("[TRAJ] wrote trajectories data -> %s/trajectories.npy" % args.out_dir)


def parse_args():
    p = argparse.ArgumentParser(description="StreetLearn PPO 轨迹可视化")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=1,
                   help="可视化用 1 个环境即可；>1 时仅记录 env 0 的轨迹。")
    p.add_argument("--num_trajectories", type=int, default=9,
                   help="跑几条轨迹（全部叠画在同一张全图地图上）。")
    p.add_argument("--max_steps", type=int, default=1000,
                   help="单条轨迹最大步数（=训练 frame_cap 口径）。")
    p.add_argument("--cuda", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--goal_seed", type=int, default=None,
                   help="固定目标采样种子（默认取 --seed）。")
    p.add_argument("--out_dir", type=str, default="traj_out")
    p.add_argument("--out_png", type=str, default="traj_map.png")
    return p.parse_args()


if __name__ == "__main__":
    run_traj_viz(parse_args())
