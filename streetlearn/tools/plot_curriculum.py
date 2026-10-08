#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把课程阶段测试（curriculum_eval.csv）画成一张总览图。

数据源：ppo_train.py 在性能课程（--curriculum_mode performance）下每次阶段
测试写一行 curriculum_eval.csv：档位、距离段、累计训练帧、成功数、最远成功
距离、到达步数、是否晋级……同目录的 run_config.json / curriculum_summary.json
用来取晋级阈值与训练状态（缺失时用默认值）。

四个面板（前三个横轴是累计训练帧，所有 GPU 合计）：
  1) 测试成功率：每个点是一次 100 条路线的测试；虚线 = 晋级阈值；
     灰底色块区分档位，竖线标出晋级时刻，100/100 的测试外面套一个圈。
  2) 最远成功距离：每次测试里成功到达的最远路线（米），灰带是该档测试距离段；
     一条都没到达的测试留空。
  3) 到达步数：成功路线的平均步数（蓝）与最慢一条（灰）。
  4) 每档训练帧数：已通过的档为蓝色，当前 / 未通过的档为灰色。

用法：
    python run.py plot_curriculum                     # 读 TRAIN.output_dir
    python -m streetlearn.tools.plot_curriculum runs/xxx/curriculum_eval.csv out.png
    python -m streetlearn.tools.plot_curriculum runs/xxx            # 传目录也行
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

# 字体与配色与 plot_ppo_log.py / plot_ppo_eval.py 一致：数据只用强调蓝，
# 上下文（距离段、最慢路线、未通过的档）用灰，文字一律墨色。
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "SimHei", "Microsoft YaHei", "Arial Unicode MS",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"
CONTEXT = "#898781"
STAGE_WASH = "#f0efec"
BAND_WASH = "#e1e0d9"
THRESHOLD_STYLE = dict(color=INK_2, lw=1.2, ls=(0, (4, 3)))
DPI = 130


def _f(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_rows(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        try:
            r = {
                "stage": int(float(row["stage"])),
                "lower": _f(row, "lower_m"),
                "upper": _f(row, "upper_m"),
                "step": _f(row, "global_step"),
                "stage_frames": _f(row, "stage_frames", 0.0),
                "episodes": int(float(row["episodes"])),
                "successes": int(float(row["successes"])),
                "rate": _f(row, "success_rate"),
                "max_success_m": _f(row, "max_success_distance", 0.0),
                "mean_steps": _f(row, "mean_success_steps", 0.0),
                "max_steps": _f(row, "max_success_steps", 0.0),
                "promoted": int(float(row.get("promoted") or 0)),
            }
        except (KeyError, TypeError, ValueError):
            continue
        r["perfect"] = r["episodes"] > 0 and r["successes"] == r["episodes"]
        out.append(r)
    out.sort(key=lambda r: r["step"])
    return out


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def stage_spans(rows):
    """每档：[start, end] 帧、距离段、测试次数、是否已晋级。"""
    spans = {}
    for r in rows:
        s = spans.setdefault(r["stage"], {
            "stage": r["stage"], "lower": r["lower"], "upper": r["upper"],
            "start": r["step"] - r["stage_frames"], "end": r["step"],
            "frames": r["stage_frames"], "tests": 0, "passed": False,
            "first_perfect": None})
        s["start"] = min(s["start"], r["step"] - r["stage_frames"])
        s["end"] = max(s["end"], r["step"])
        s["frames"] = max(s["frames"], r["stage_frames"])
        s["tests"] += 1
        s["passed"] = s["passed"] or bool(r["promoted"])
        if r["perfect"] and s["first_perfect"] is None:
            s["first_perfect"] = r["step"]
    return [spans[k] for k in sorted(spans)]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", colors=INK_2, labelsize=9.5, length=0)
    ax.tick_params(axis="x", pad=6)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _panel_title(ax, title, subtitle):
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 24),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=12.5, fontweight="bold", color=INK)
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=10, color=INK_2)


def _fmt_m(v):
    return "%g km" % (v / 1000.0) if v >= 1000 else "%g m" % v


def _fmt_frames(v, _pos=None):
    """帧数刻度：1.5M / 250K / 64。"""
    a = abs(v)
    if a >= 1e6:
        return "%gM" % round(v / 1e6, 2)
    if a >= 1e3:
        return "%gK" % round(v / 1e3, 1)
    return "%d" % v


def _frames_axis(ax, xmax):
    ax.set_xlim(0, xmax * 1.02)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_frames))
    ax.set_xlabel("training frames (all GPUs)", fontsize=9.5, color=INK_2)


def _stage_backdrop(ax, spans, label_y=None):
    """交替灰底区分档位，晋级处画竖线；label_y（数据坐标）给出时在该高度写档位。"""
    for i, s in enumerate(spans):
        x0, x1 = s["start"], s["end"]
        if i % 2 == 0:
            ax.axvspan(x0, x1, color=STAGE_WASH, lw=0, zorder=0)
        if s["passed"]:
            ax.axvline(x1, color=BASELINE, lw=1.0, zorder=1)
        if label_y is not None:
            ax.text((x0 + x1) / 2, label_y, _fmt_m(s["upper"]), ha="center",
                    va="center", fontsize=8.5, color=INK_2, zorder=5)


def _series(ax, rows, key, color, label, marker_size=6, lw=1.8, zorder=3):
    """按档位断开画线（跨档不连线），每次测试一个点（带背景色描边）。"""
    for stage in sorted({r["stage"] for r in rows}):
        pts = [r for r in rows if r["stage"] == stage]
        x = [r["step"] for r in pts]
        y = [r[key] for r in pts]
        ax.plot(x, y, color=color, lw=lw, solid_capstyle="round",
                solid_joinstyle="round", zorder=zorder)
        ax.plot(x, y, ls="none", marker="o", ms=marker_size, mfc=color,
                mec=SURFACE, mew=1.5, zorder=zorder + 1)
    return Line2D([0], [0], color=color, lw=lw, marker="o", ms=marker_size,
                  mfc=color, mec=SURFACE, label=label)


def panel_success(ax, rows, spans, pass_rate, episodes):
    _style_axes(ax)
    _panel_title(ax, "Test success rate",
                 "%d routes per test in the stage's distance band · dashed = pass "
                 "threshold · ring = all routes reached" % episodes)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _stage_backdrop(ax, spans, label_y=1.08)
    ax.axhline(pass_rate, zorder=2, **THRESHOLD_STYLE)
    ax.annotate("pass %.0f%%" % (100 * pass_rate), xy=(0, pass_rate),
                xycoords=("axes fraction", "data"), xytext=(2, -3),
                textcoords="offset points", ha="left", va="top",
                fontsize=9, color=INK_2)
    _series(ax, rows, "rate", ACCENT, "success rate")
    perfect = [r for r in rows if r["perfect"]]
    if perfect:
        # 100/100 的测试额外套一个圈（形状编码，不靠颜色）。
        ax.plot([r["step"] for r in perfect], [1.0] * len(perfect), ls="none",
                marker="o", ms=12, mfc="none", mec=INK, mew=1.2, zorder=5)


def panel_distance(ax, rows, spans):
    _style_axes(ax)
    _panel_title(ax, "Farthest successful route per test",
                 "meters · gray band = the stage's test distance band · "
                 "gap = no route reached")
    for s in spans:
        x0, x1 = s["start"], max(s["end"], s["start"] + 1)
        ax.fill_between([x0, x1], [s["lower"]] * 2, [s["upper"]] * 2,
                        color=BAND_WASH, lw=0, zorder=1)
    # 没有任何成功的测试不画成 0 米，而是留空。
    rows = [dict(r, max_success_m=(r["max_success_m"] if r["successes"] > 0
                                   else float("nan"))) for r in rows]
    handle = _series(ax, rows, "max_success_m", ACCENT, "farthest success")
    top = max([s["upper"] for s in spans] + [1.0])
    ax.set_ylim(0, top * 1.08)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:,.0f}".format(v)))
    ax.legend(handles=[handle, Patch(facecolor=BAND_WASH, label="test band")],
              loc="upper left", frameon=False, fontsize=9, labelcolor=INK)


def panel_steps(ax, rows):
    _style_axes(ax)
    _panel_title(ax, "Steps to reach the goal",
                 "successful routes only · lower is better")
    hit = [r for r in rows if r["successes"] > 0]
    handles = []
    if hit:
        handles.append(_series(ax, hit, "max_steps", CONTEXT, "slowest success",
                               marker_size=5, lw=1.2, zorder=2))
        handles.append(_series(ax, hit, "mean_steps", ACCENT, "mean"))
        ax.set_ylim(0, max(r["max_steps"] for r in hit) * 1.12 + 1)
        ax.legend(handles=handles[::-1], loc="upper left", frameon=False,
                  fontsize=9, labelcolor=INK)
    else:
        ax.text(0.5, 0.5, "no successful route yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color=INK_2)
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))


def panel_stage_frames(ax, spans, budget):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", colors=INK_2, labelsize=9.5, length=0)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _panel_title(ax, "Training frames per stage",
                 "blue = passed · gray = current / not passed")
    y = np.arange(len(spans))
    frames = [s["frames"] for s in spans]
    # 条宽封顶约 22px（不撑满整格），按保存时的 dpi 换算成数据单位。
    slot_px = (ax.get_position().height * ax.figure.get_figheight() * DPI
               / max(len(spans), 1))
    height = min(0.6, 22.0 / slot_px)
    for yi, s, fr in zip(y, spans, frames):
        ax.barh(yi, fr, height=height, color=ACCENT if s["passed"] else CONTEXT,
                lw=0, zorder=2)
        state = "passed" if s["passed"] else "current"
        if not s["passed"] and budget and s["frames"] >= budget:
            state = "not passed (budget)"
        ax.annotate("%s · %d test%s · %s" % (_fmt_frames(fr), s["tests"],
                                             "" if s["tests"] == 1 else "s", state),
                    (fr, yi), xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(["stage %d · %s" % (s["stage"], _fmt_m(s["upper"]))
                        for s in spans], color=INK)
    ax.set_ylim(len(spans) - 0.5, -0.5)
    xmax = max(frames + [budget or 0.0, 1.0])
    ax.set_xlim(0, xmax * 1.45)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_frames))
    ax.set_xlabel("frames in stage", fontsize=9.5, color=INK_2)
    if budget:
        ax.axvline(budget, zorder=1, **THRESHOLD_STYLE)
        ax.annotate("stage budget", xy=(budget, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -2), textcoords="offset points", ha="left", va="top",
                    fontsize=9, color=INK_2)


def plot(rows, cfg, summary, source):
    spans = stage_spans(rows)
    pass_rate = float(cfg.get("curriculum_pass_rate", 0.95))
    confirmations = int(cfg.get("curriculum_confirmations", 2))
    budget = float(cfg.get("curriculum_max_stage_frames", 0) or 0)
    stages = summary.get("stages") or cfg.get("curriculum_stages") or []
    if isinstance(stages, str):
        stages = [float(x) for x in stages.split(",") if x.strip()]
    episodes = int(round(np.median([r["episodes"] for r in rows])))
    passed = [s for s in spans if s["passed"]]
    status = summary.get("status") or "unknown"

    fig = plt.figure(figsize=(15, 10.8), facecolor=SURFACE)
    fig.text(0.05, 1 - 0.3 / 10.8, "Curriculum tests", ha="left", va="top",
             fontsize=16, fontweight="bold", color=INK)
    subtitle = ("Highest passed: %s (%d/%d stages) · %s frames · %d tests · "
                "status: %s · promote after %d consecutive tests >= %.0f%%"
                % (_fmt_m(passed[-1]["upper"]) if passed else "none",
                   len(passed), max(len(stages), len(spans)),
                   _fmt_frames(rows[-1]["step"]), len(rows),
                   status.replace("_", " "), confirmations, 100 * pass_rate))
    fig.text(0.05, 1 - 0.72 / 10.8, subtitle, ha="left", va="top",
             fontsize=10.5, color=INK_2)
    fig.text(0.05, 1 - 1.02 / 10.8,
             os.path.basename(os.path.dirname(os.path.abspath(source))),
             ha="left", va="top", fontsize=8.5, color=INK_2)

    gs = fig.add_gridspec(2, 2, left=0.05, right=0.975, top=1 - 1.85 / 10.8,
                          bottom=0.6 / 10.8, hspace=0.55, wspace=0.18)
    xmax = rows[-1]["step"]
    ax = fig.add_subplot(gs[0, 0])
    panel_success(ax, rows, spans, pass_rate, episodes)
    _frames_axis(ax, xmax)
    ax = fig.add_subplot(gs[0, 1])
    panel_distance(ax, rows, spans)
    _frames_axis(ax, xmax)
    ax = fig.add_subplot(gs[1, 0])
    _stage_backdrop(ax, spans)
    panel_steps(ax, rows)
    _frames_axis(ax, xmax)
    panel_stage_frames(fig.add_subplot(gs[1, 1]), spans, budget)
    return fig


def main():
    p = argparse.ArgumentParser(description="画课程阶段测试总览图")
    p.add_argument("src", help="curriculum_eval.csv 或其所在目录")
    p.add_argument("out", nargs="?", default=None,
                   help="输出 PNG（默认与 CSV 同目录的 curriculum_eval.png）")
    args = p.parse_args()
    src = args.src
    if os.path.isdir(src):
        src = os.path.join(src, "curriculum_eval.csv")
    if not os.path.isfile(src):
        print("找不到 %s（只有 --curriculum_mode performance 的训练才会写它）" % src)
        sys.exit(1)
    rows = load_rows(src)
    if not rows:
        print("%s 里还没有测试记录（每档至少训练 curriculum_min_stage_frames 帧后"
              "才开始测试）。" % src)
        sys.exit(1)
    run_dir = os.path.dirname(os.path.abspath(src))
    cfg = _read_json(os.path.join(run_dir, "run_config.json"))
    summary = _read_json(os.path.join(run_dir, "curriculum_summary.json"))
    out = args.out or os.path.join(run_dir, "curriculum_eval.png")
    fig = plot(rows, cfg, summary, src)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print("已保存课程测试图 -> %s  (%d 次测试, %d 档)"
          % (out, len(rows), len({r["stage"] for r in rows})))


if __name__ == "__main__":
    main()
