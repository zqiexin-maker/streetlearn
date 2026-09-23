#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把评估结果画成「Agent 贪心 vs Oracle 实跑」对比图（论文口径，BFS 上界作参考线）。

数据源：ppo_eval.py 每次评估把一行结果写入 eval_results.csv，列：
    mode,tag,episodes,arrival,spl,oracle,coverage,t_half,fail_rate
其中：
  - mode=normal, tag=agent    → Agent 贪心评估（argmax）
  - mode=oracle, tag=actual   → Oracle 实跑（用 bearing 真正导航，差异 6）
  - oracle 列 = BFS 上界：每 episode 目标的最短路径节点数，即到达奖励的理论满分

图的内容（上排三个面板，下方数值表）：
    1) 到达奖励 arrival（panos/episode，越高越好）：虚线 = BFS 上界，
       柱顶小字 = 占上界的百分比
    2) 成功率 / SPL（0-1，越高越好）：虚线 = 满分 1.0。ppo_eval 固定单目标
       模式，每 episode 的 coverage 只能是 0 或 1，均值恒等于成功率（=1-失败率），
       两者合并成一组柱，不重复画
    3) T_1/2（走完一半距离所需步数，越低越好）：没有任何目标走到半程时标 n/a
    数值表列出 CSV 的全部指标（含失败率 / coverage）。
各指标量纲不同，每个面板各用自己的 y 轴；Agent 用强调蓝，Oracle 作参照用灰。

用法：
    python run.py eval_all        # Agent + Oracle 两遍评估，结果写进同一个 CSV
    python run.py plot_eval       # 图输出到 TRAIN.output_dir
    # 或直接调用：
    python -m streetlearn.tools.plot_ppo_eval <eval_results.csv> [out.png]

看什么：
    - 蓝柱越接近虚线（BFS 上界 / 满分 1.0），说明学得越接近理论上限；
    - Oracle 灰柱应紧贴虚线（Oracle 是最优导航员），是 Agent 的实际参照；
    - T_1/2 越低越好，灰柱是最优导航员走到半程的步数。
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.ticker import MaxNLocator

# 避免负号渲染成方块。标签全是英文，DejaVu Sans 放最前（CJK 字体没有粗体、
# 拉丁字母也偏宽）；其后的 CJK 字体留作中文字形回退（matplotlib>=3.6 逐字形回退）。
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "SimHei", "Microsoft YaHei", "Arial Unicode MS",
]

# 配色（浅色底）：数据只用两种填充色，文字一律用墨色，网格/基线保持低调。
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
AGENT_COLOR = "#2a78d6"
ORACLE_COLOR = "#898781"
CEILING_STYLE = dict(color=INK_2, lw=1.2, ls=(0, (4, 3)))

# (CSV 行的键, 图例名, 填充色)；顺序固定，Agent 永远在前。
SERIES = (("agent", "Agent (greedy)", AGENT_COLOR),
          ("oracle", "Oracle (actual)", ORACLE_COLOR))
BAR_IN = 0.42   # 柱宽（英寸）
GAP_IN = 0.06   # 同组相邻两柱的间隙（英寸）


def load(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV %s 为空，请先跑 ppo_eval.py 生成数据。" % path)
    return rows


def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _metrics(row):
    """CSV 行 -> 画图用的指标字典；row 为 None（该评估没跑）时返回 None。"""
    if row is None:
        return None
    t_half = _f(row, "t_half")
    return {
        "episodes": int(_f(row, "episodes")),
        "arrival": _f(row, "arrival"),
        "bfs": _f(row, "oracle"),
        "success": 1.0 - _f(row, "fail_rate"),
        "fail": _f(row, "fail_rate"),
        "coverage": _f(row, "coverage"),
        "spl": _f(row, "spl"),
        # t_half=0 是 ppo_eval 在"没有任何目标走到半程"时写的占位值，不是 0 步。
        "t_half": t_half if t_half > 0 else float("nan"),
    }


def _present(m):
    """已跑过的评估：[(图例名, 填充色, 指标), ...]。"""
    return [(label, color, m[key]) for key, label, color in SERIES
            if m[key] is not None]


def _to_data_width(ax, inches):
    """英寸 -> x 数据单位（依赖固定布局，须在 set_xlim 之后调用）。"""
    x0, x1 = ax.get_xlim()
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    return inches * (x1 - x0) / width_in


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", colors=INK_2, labelsize=10, length=0)
    ax.tick_params(axis="x", pad=7, labelcolor=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _panel_title(ax, title, subtitle):
    ax.text(0, 1.13, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0, 1.04, subtitle, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, color=INK_2)


def _bar(ax, x, h, width, color, radius_px=4.0):
    """画一根柱：顶端圆角、基线处直角。须在 xlim/ylim 定好之后调用。"""
    if not np.isfinite(h) or h <= 0:
        return
    x0 = x - width / 2
    to_px = ax.transData.transform
    rx = radius_px / (to_px((1, 0))[0] - to_px((0, 0))[0])
    ry = radius_px / (to_px((0, 1))[1] - to_px((0, 0))[1])
    body = dict(facecolor=color, linewidth=0, zorder=2)
    if h <= 2 * ry:
        ax.add_patch(Rectangle((x0, 0), width, h, **body))
        return
    ax.add_patch(FancyBboxPatch(
        (x0, 0), width, h, boxstyle="round,pad=0,rounding_size=%g" % rx,
        mutation_aspect=ry / rx, **body))
    # 直角矩形盖住下半截的圆角，让柱子方正地立在基线上。
    ax.add_patch(Rectangle((x0, 0), width, h - ry, **body))


def _label(ax, x, h, text, sub=None):
    """柱顶标注：主数值 + 可选的第二行小字；底衬背景色，压在参考虚线之上。"""
    h = h if np.isfinite(h) else 0.0
    box = dict(boxstyle="square,pad=0.12", facecolor=SURFACE, edgecolor="none")
    offset = 3
    if sub:
        ax.annotate(sub, (x, h), xytext=(0, offset), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK_2,
                    bbox=box, zorder=5)
        offset += 15
    ax.annotate(text, (x, h), xytext=(0, offset), textcoords="offset points",
                ha="center", va="bottom", fontsize=10.5, color=INK,
                bbox=box, zorder=5)


def _same_goals(runs):
    """两次评估的 BFS 上界一致（差 <=1%）即视为面对同一批目标。"""
    bfs = [s["bfs"] for _, _, s in runs]
    return max(bfs) - min(bfs) <= 0.01 * max(max(bfs), 1e-9)


def _panel_arrival(ax, runs):
    _style_axes(ax)
    _panel_title(ax, "Arrival reward",
                 "panos per episode · higher is better")
    top = max([s["arrival"] for _, _, s in runs]
              + [s["bfs"] for _, _, s in runs] + [1.0])
    # 右侧多留空白，给 BFS 上界的直接标注腾位置。
    ax.set_xlim(-0.6, 2.1)
    ax.set_ylim(0, top * 1.28)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    width = _to_data_width(ax, BAR_IN)
    # 两次评估目标一致时画共用的上界虚线；不一致时没有共用上界，只看柱顶的
    # "占自己那次上界的百分比"（标题下方另有提示）。
    if _same_goals(runs):
        ceiling = runs[0][2]["bfs"]
        ax.axhline(ceiling, zorder=3, **CEILING_STYLE)
        ax.annotate("BFS ceiling\n%.1f" % ceiling, xy=(1, ceiling),
                    xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=9.5, color=INK_2, linespacing=1.3)
    for i, (_, color, s) in enumerate(runs):
        _bar(ax, i, s["arrival"], width, color)
        sub = ("%.0f%% of BFS" % (100.0 * s["arrival"] / s["bfs"])
               if s["bfs"] > 0 else None)
        _label(ax, i, s["arrival"], "%.1f" % s["arrival"], sub)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([label for label, _, _ in runs])


def _panel_rates(ax, runs):
    _style_axes(ax)
    groups = [("success", "Success rate")]
    if all(abs(s["coverage"] - s["success"]) <= 0.005 for _, _, s in runs):
        # 单目标评估里 coverage 恒等于成功率，合并成一组，免得画两组一样的柱。
        groups[0] = ("success", "Success rate (= coverage)")
    else:
        groups.append(("coverage", "Coverage"))
    groups.append(("spl", "SPL"))
    _panel_title(ax, "Success rate & SPL", "0-1 · higher is better")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(1.0, zorder=3, **CEILING_STYLE)
    width = _to_data_width(ax, BAR_IN)
    step = width + _to_data_width(ax, GAP_IN)
    for gi, (key, _) in enumerate(groups):
        for si, (_, color, s) in enumerate(runs):
            x = gi + (si - (len(runs) - 1) / 2.0) * step
            _bar(ax, x, s[key], width, color)
            _label(ax, x, s[key], "%.2f" % s[key])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([name for _, name in groups])


def _panel_t_half(ax, runs):
    _style_axes(ax)
    finite = [s["t_half"] for _, _, s in runs if np.isfinite(s["t_half"])]
    _panel_title(ax, r"$T_{1/2}$: steps to cover half the distance",
                 "steps · lower is better"
                 + (" · n/a = never got halfway" if len(finite) < len(runs)
                    else ""))
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, max(finite + [1.0]) * 1.2)
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    width = _to_data_width(ax, BAR_IN)
    for i, (_, color, s) in enumerate(runs):
        if np.isfinite(s["t_half"]):
            _bar(ax, i, s["t_half"], width, color)
            _label(ax, i, s["t_half"], "%.0f" % s["t_half"])
        else:
            _label(ax, i, 0.0, "n/a")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([label for label, _, _ in runs])


def _table(ax, m):
    """数值表：CSV 里的全部指标（含失败率 / coverage），供精确读数。"""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    header = ["Episodes", "Arrival", "BFS ceiling", "Arrival / BFS",
              "Success rate", "Failure rate", "Coverage", "SPL",
              r"$T_{1/2}$ (steps)"]
    right_edges = np.linspace(0.30, 1.0, len(header))
    y_header, y_rows = 0.84, (0.50, 0.17)
    for x, name in zip(right_edges, header):
        ax.text(x, y_header, name, ha="right", va="center", fontsize=10,
                color=INK_2)
    ax.plot([0, 1], [0.69, 0.69], color=BASELINE, lw=0.9)
    ax.plot([0, 1], [0.335, 0.335], color=GRID, lw=0.8)
    for y, (key, label, color) in zip(y_rows, SERIES):
        ax.add_patch(Rectangle((0.0, y - 0.07), 0.009, 0.14, facecolor=color,
                               linewidth=0))
        ax.text(0.016, y, label, ha="left", va="center", fontsize=10.5,
                color=INK)
        s = m[key]
        if s is None:
            ax.text(right_edges[0], y, "not run", ha="right", va="center",
                    fontsize=10.5, color=INK_2)
            continue
        pct = (100.0 * s["arrival"] / s["bfs"]) if s["bfs"] > 0 else None
        cells = ["%d" % s["episodes"], "%.1f" % s["arrival"],
                 "%.1f" % s["bfs"], ("%.0f%%" % pct) if pct is not None else "-",
                 "%.2f" % s["success"], "%.2f" % s["fail"],
                 "%.2f" % s["coverage"], "%.2f" % s["spl"],
                 ("%.0f" % s["t_half"]) if np.isfinite(s["t_half"]) else "n/a"]
        for x, cell in zip(right_edges, cells):
            ax.text(x, y, cell, ha="right", va="center", fontsize=10.5,
                    color=INK)


def plot(m):
    runs = _present(m)
    fig = plt.figure(figsize=(15, 8.6), facecolor=SURFACE)
    fig.text(0.05, 0.955, "Evaluation: Agent (greedy) vs Oracle (actual)",
             ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    subtitle = "Episodes: " + ", ".join(
        "%s %d" % (label, s["episodes"]) for label, _, s in runs)
    subtitle += ("  ·  BFS ceiling = shortest-path length (panos) to each "
                 "goal, the maximum arrival reward")
    fig.text(0.05, 0.912, subtitle, ha="left", va="top", fontsize=10.5,
             color=INK_2)
    if len(runs) == 2 and not _same_goals(runs):
        fig.text(0.05, 0.884,
                 "Note: BFS ceilings differ (%.1f vs %.1f), so the two runs "
                 "did not face the same goals. Rerun both with the same "
                 "seed / goal_seed / num_envs for a head-to-head comparison."
                 % (runs[0][2]["bfs"], runs[1][2]["bfs"]),
                 ha="left", va="top", fontsize=10.5, color=INK)

    handles = [Patch(facecolor=color, label=label) for label, color, _ in runs]
    handles.append(Line2D([0], [0], label="Ceiling (BFS / max 1.0)",
                          **CEILING_STYLE))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.043, 0.87),
               ncol=len(handles), frameon=False, fontsize=10.5,
               handlelength=1.8, columnspacing=2.2, labelcolor=INK)

    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0], left=0.05,
                          right=0.985, top=0.73, bottom=0.33, wspace=0.2)
    _panel_arrival(fig.add_subplot(gs[0, 0]), runs)
    _panel_rates(fig.add_subplot(gs[0, 1]), runs)
    _panel_t_half(fig.add_subplot(gs[0, 2]), runs)
    _table(fig.add_axes([0.05, 0.04, 0.935, 0.17]), m)
    return fig


def main():
    if len(sys.argv) < 2:
        print("usage: python plot_ppo_eval.py <eval_results.csv> [out.png]")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "ppo_eval_triple.png"

    rows = load(src)
    # 选两行：Agent 贪心（tag=agent）与 Oracle 实跑（mode=oracle）。
    agent_row = next((r for r in rows if r.get("tag") == "agent"), None)
    oracle_row = next((r for r in rows if r.get("mode") == "oracle"), None)
    if agent_row is None and oracle_row is None:
        raise ValueError(
            "CSV 里找不到 tag=agent 或 mode=oracle 的行，请先跑 ppo_eval.py "
            "(至少跑一次 Agent 评估 + 一次 --oracle 评估)。")

    fig = plot({"agent": _metrics(agent_row), "oracle": _metrics(oracle_row)})
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print("已保存评估对比图 -> %s" % out)


if __name__ == "__main__":
    main()
