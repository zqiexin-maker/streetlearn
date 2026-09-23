#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 PPO 训练指标画成曲线 PNG（绘图风格与 plot_ppo_eval.py 一致）。

数据源（两种都支持）：
  1) 结构化 CSV（首选）：ppo_train.py 现在每次 log 都会把指标追加写到
     <output_dir>/train_metrics.csv。列包含论文口径的 P2a 指标
     (arrival / oracle / coverage) 与课程半径 (cur_radius)。直接喂这个文件最稳。
  2) 控制台日志：若只有 `2>&1 | tee train.log` 落盘的文本日志，本脚本用
     容错正则提取 [PPO] 行（兼容新旧两种打印格式）。

用法（在训练所在的机器上，例如 3090 服务器）：
    # 推荐：直接吃 CSV
    python plot_ppo_log.py ppo_ckpts/train_metrics.csv ppo_curves.png

    # 兼容：吃控制台日志
    python -m streetlearn.python.ppo_train ... 2>&1 | tee train.log
    python plot_ppo_log.py train.log ppo_curves.png

    # 也可以直接传目录（会自动找 train_metrics.csv / train.log）
    python plot_ppo_log.py ppo_ckpts ppo_curves.png

指标怎么看（与论文 Table 2 对齐）：
    mean_ep_reward / mean_spl / arrival  应随训练【上升】（学到送快递）
    coverage = arrival / oracle          应【上升】（逼近 Oracle 上界）
    oracle                                是 BFS Oracle 理论满分，基本恒定
    entropy                              应【下降】（策略从随机变确定）
    value_loss / policy_loss / loss       应【下降】并趋稳（critic/actor 收敛）
    cur_radius                            课程学习进度（500m → FULL）

图的样式：每个面板一条平滑后的曲线（同一种强调蓝），末端标出最新值；
标题下一行小字说明期望趋势；网格/坐标轴保持低调。
"""
import sys
import os
import re
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 字体顺序与 plot_ppo_eval.py 一致（英文标签用 DejaVu Sans，CJK 字体留作中文字形回退）。
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans CJK JP",
    "SimHei", "Microsoft YaHei", "Arial Unicode MS",
]

# 配色（与 plot_ppo_eval.py 相同）：数据只用强调蓝，文字一律墨色，网格/基线低调。
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
LINE_COLOR = "#2a78d6"


# 容错正则：兼容旧格式（只有 mean_ep_reward/mean_spl）与新格式
# （额外有 arrival / oracle / cov / cur ，以及可选的 aux）。
LINE_RE = re.compile(
    r"\[PPO\] update=(?P<update>\d+) steps=(?P<steps>\d+) fps=(?P<fps>[\d.]+) "
    r"loss=(?P<loss>[-\d.]+) pol=(?P<pol>[-\d.]+) val=(?P<val>[-\d.]+) "
    r"ent=(?P<ent>[-\d.]+) mean_ep_reward=(?P<rew>[-\d.]+) "
    r"mean_spl=(?P<spl>[-\d.]+)"
    r"(?: arrival=(?P<arrival>[-\d.]+))?"
    r"(?: oracle=(?P<oracle>[-\d.]+))?"
    r"(?: cov=(?P<cov>[-\d.]+))?"
    r"(?: cur=(?P<cur>\S+))?"
    r"(?: aux=(?P<aux>[-\d.]+))?"
)


def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _cur_to_float(x):
    """把 cur 字段（'500m' / 'FULL' / -1 / None）转成数值，FULL/None 返 None。"""
    if x is None:
        return None
    s = str(x).strip()
    if s in ("FULL", "", "None", "-1"):
        return None
    m = re.match(r"([\d.]+)", s)
    return float(m.group(1)) if m else None


def parse_log(log_path):
    """从控制台日志文本解析（容错）。返回 (updates, dict_of_lists)。"""
    d = {k: [] for k in
         ("loss", "pol", "val", "ent", "rew", "spl",
          "arrival", "oracle", "cov", "cur", "aux")}
    updates = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            g = m.groupdict()
            updates.append(int(g["update"]))
            d["loss"].append(_to_float(g["loss"]))
            d["pol"].append(_to_float(g["pol"]))
            d["val"].append(_to_float(g["val"]))
            d["ent"].append(_to_float(g["ent"]))
            d["rew"].append(_to_float(g["rew"]))
            d["spl"].append(_to_float(g["spl"]))
            d["arrival"].append(_to_float(g.get("arrival")))
            d["oracle"].append(_to_float(g.get("oracle")))
            d["cov"].append(_to_float(g.get("cov")))
            d["aux"].append(_to_float(g.get("aux")))
            d["cur"].append(_cur_to_float(g.get("cur")))
    return updates, d


def parse_csv(csv_path):
    """从 train_metrics.csv 解析（首选，字段最全）。返回 (updates, dict_of_lists)。"""
    d = {k: [] for k in
         ("loss", "pol", "val", "ent", "rew", "spl",
          "arrival", "oracle", "cov", "cur", "aux")}
    updates = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            updates.append(int(float(row["update"])))
            d["loss"].append(_to_float(row.get("loss")))
            d["pol"].append(_to_float(row.get("policy")))
            d["val"].append(_to_float(row.get("value")))
            d["ent"].append(_to_float(row.get("entropy")))
            d["rew"].append(_to_float(row.get("mean_ep_reward")))
            d["spl"].append(_to_float(row.get("mean_spl")))
            d["arrival"].append(_to_float(row.get("arrival")))
            d["oracle"].append(_to_float(row.get("oracle")))
            d["cov"].append(_to_float(row.get("coverage")))
            d["aux"].append(_to_float(row.get("aux")))
            d["cur"].append(_cur_to_float(row.get("cur_radius")))
    return updates, d


def load_metrics(arg):
    """arg 可以是 .csv / .log / 目录；自动选择数据源。"""
    if os.path.isdir(arg):
        for cand in ("train_metrics.csv", "train.log", "train.log.txt"):
            p = os.path.join(arg, cand)
            if os.path.exists(p):
                return load_metrics(p)
        raise FileNotFoundError("目录 %s 下没找到 train_metrics.csv / train.log" % arg)
    if arg.lower().endswith(".csv"):
        return parse_csv(arg)
    return parse_log(arg)


def smooth(values, window):
    """滑窗均值。对 None / NaN 容错：窗口内缺失值跳过；全缺失则记 None（画图断开）。"""
    if window <= 1 or not values:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        seg = values[lo:i + 1]
        nums = [x for x in seg
                if x is not None
                and not (isinstance(x, float) and math.isnan(x))]
        out.append(sum(nums) / len(nums) if nums else None)
    return out


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", colors=INK_2, labelsize=9.5, length=0)
    ax.tick_params(axis="x", pad=6)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def _panel_title(ax, title, subtitle):
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 24),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=12.5, fontweight="bold", color=INK)
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=10, color=INK_2)


def _fmt(v):
    """末端数值标签：大数取整，小数保留有效位。"""
    a = abs(v)
    if a >= 100:
        return "%.0f" % v
    if a >= 1:
        return "%.2f" % v
    return "%.3f" % v


def plot(updates, series, specs, subtitle):
    """series: {key: 平滑后的值列表}；specs: [(标题, 小字说明, key), ...]。"""
    ncols = 3
    nrows = math.ceil(len(specs) / ncols)
    head, foot, row_h, row_gap = 1.85, 0.6, 2.4, 1.25   # 英寸
    fig_h = head + nrows * row_h + (nrows - 1) * row_gap + foot
    fig = plt.figure(figsize=(15, fig_h), facecolor=SURFACE)
    fig.text(0.05, 1 - 0.3 / fig_h, "PPO training curves", ha="left",
             va="top", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.05, 1 - 0.72 / fig_h, subtitle, ha="left", va="top",
             fontsize=10.5, color=INK_2)
    gs = fig.add_gridspec(nrows, ncols, left=0.05, right=0.975,
                          top=1 - head / fig_h, bottom=foot / fig_h,
                          hspace=row_gap / row_h, wspace=0.22)

    x = np.asarray(updates, dtype=float)
    box = dict(boxstyle="square,pad=0.12", facecolor=SURFACE, edgecolor="none")
    for idx, (title, note, key) in enumerate(specs):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        _style_axes(ax)
        _panel_title(ax, title, note)
        y = np.array([np.nan if v is None else v for v in series[key]],
                     dtype=float)
        ax.plot(x, y, color=LINE_COLOR, lw=1.6, solid_capstyle="round",
                solid_joinstyle="round", zorder=3)
        finite = np.flatnonzero(np.isfinite(y))
        if finite.size:
            # 末端圆点 + 最新值，读数不用对 y 轴。
            i = finite[-1]
            ax.plot(x[i], y[i], ls="none", marker="o", ms=6, mfc=LINE_COLOR,
                    mec=SURFACE, mew=1.5, zorder=4, clip_on=False)
            ax.annotate(_fmt(y[i]), (x[i], y[i]), xytext=(-4, 7),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=10, color=INK, bbox=box, zorder=5)
        if len(x) > 1:
            ax.set_xlim(x[0], x[-1])
        if key in ("cov", "oracle"):
            ax.set_ylim(bottom=0)
        ax.set_xlabel("update", fontsize=9.5, color=INK_2)
    return fig


def main():
    if len(sys.argv) < 2:
        print("usage: python plot_ppo_log.py <train_metrics.csv|train.log|dir> "
              "[out.png] [--smooth N]")
        sys.exit(1)
    src = sys.argv[1]
    out_path = "ppo_curves.png"
    smooth_win = 5
    i = 2
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--smooth":
            smooth_win = int(sys.argv[i + 1])
            i += 2
        else:
            out_path = a
            i += 1

    updates, d = load_metrics(src)
    if not updates:
        print("未在 %s 中找到任何可解析的训练指标，请确认数据源。" % src)
        sys.exit(1)

    # 平滑（不影响原始 CSV，仅画图更可读）
    sd = {k: smooth(v, smooth_win) for k, v in d.items()}

    # 面板规格：(标题, 期望趋势说明, 数据列)。标题统一用英文/ASCII，
    # 避免服务器无 CJK 字体时显示成方框。
    specs = [
        ("Mean episode reward", "higher is better", "rew"),
        ("Mean SPL", "higher is better", "spl"),
        ("Arrival reward", "paper metric · higher is better", "arrival"),
        ("Coverage", "arrival / oracle · higher is better", "cov"),
        ("Oracle reward", "BFS ceiling · roughly constant", "oracle"),
        ("Value loss", "lower is better", "val"),
        ("Policy loss", "clipped surrogate objective", "pol"),
        ("Entropy", "falls as the policy becomes certain", "ent"),
        ("Total loss", "lower is better", "loss"),
    ]
    if any(d["aux"]):
        specs.append(("Aux heading loss", "lower is better", "aux"))
    if any(x is not None for x in d["cur"]):
        specs.append(("Curriculum radius", "meters · gap = full map", "cur"))

    fig = plot(updates, sd, specs,
               "%d updates · smoothing window %d · %s"
               % (len(updates), smooth_win, os.path.basename(src)))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("已保存曲线图 -> %s  (%d 个 update, %d 面板)"
          % (out_path, len(updates), len(specs)))


if __name__ == "__main__":
    main()
