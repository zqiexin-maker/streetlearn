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

    # 在曲线上标出多个 checkpoint 的位置（竖虚线 + 标签）
    python plot_ppo_log.py runs/xxx/train_metrics.csv curves.png \
        --checkpoints runs/xxx/ppo_iter_180.pt,runs/xxx/ppo_final.pt
    python plot_ppo_log.py runs/xxx curves.png --ckpt-dir runs/xxx --ckpt-pattern "ppo_*.pt"

checkpoint 竖线的 update 位置按以下顺序确定（都取不到就跳过并告警）：
    1) 文件名里的轮次（ppo_iter_180.pt -> update 180）
    2) 直接读 .pt 里的 update 字段（需要 torch；ppo_final.pt 走这条）
    3) ppo_best_stage_XX_*.pt -> curriculum_eval.csv 里该 stage 成功率最高的那次 update
    4) ppo_final.pt -> train_metrics.csv 的最后一个 update（近似）


指标怎么看（与论文 Table 2 对齐）：
    mean_ep_reward / mean_spl / arrival  应随训练【上升】（学到送快递）
    shaping / coin                        塑形 / 硬币分支实际发出的原始奖励；总奖励
                                          上升而 arrival 不升，说明涨的是辅助奖励
    clipped                               每步总奖励截断后的累计值（PPO 实际优化的量）
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
import glob
import math
import argparse

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
CKPT_MARKER = "#8a8981"


# 控制台日志 "[PPO] update=..." 行里的 key=value 字段 -> 数据列。按字段名取值，
# 新旧格式（有没有 kl/clipf/ev、arrival/oracle/cov/cur、aux、奖励分量）都能解析。
LOG_FIELDS = {"loss": "loss", "pol": "pol", "val": "val", "ent": "ent",
              "mean_ep_reward": "rew", "mean_spl": "spl", "arrival": "arrival",
              "oracle": "oracle", "cov": "cov", "aux": "aux", "shaping": "shaping",
              "coin": "coin", "clipped": "clipped"}
FIELD_RE = re.compile(r"(\w+)=(\S+)")
SERIES_KEYS = ("loss", "pol", "val", "ent", "rew", "spl", "arrival", "oracle",
               "cov", "cur", "aux", "shaping", "coin", "clipped")
# 旧日志 / 旧 CSV 里没有这几列：记 None，对应面板不画（其余列缺失时按 0 处理）。
OPTIONAL_KEYS = ("shaping", "coin", "clipped")


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


# --------------------------------------------------------------------------- #
# 多 checkpoint：把若干个 .pt 对应的 update 标到曲线上
# --------------------------------------------------------------------------- #
def _split_list(value):
    """'a.pt,b.pt' / ['a.pt', 'b.pt'] / None -> 去重后的路径列表。"""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = []
    for it in items:
        it = str(it).strip()
        if it and it not in out:
            out.append(it)
    return out


def resolve_ckpt_paths(specs, ckpt_dir=None, pattern="*.pt"):
    """--checkpoints 列表 + --ckpt-dir 通配，合成待标注的 checkpoint 路径。"""
    paths = _split_list(specs)
    if ckpt_dir:
        paths += sorted(glob.glob(os.path.join(ckpt_dir, pattern or "*.pt")))
    seen, out = set(), []
    for p in paths:
        key = os.path.abspath(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _ckpt_update_from_file(path):
    """直接读 .pt 里的 update 字段；没有 torch 或读失败返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    for kwargs in ({"map_location": "cpu", "weights_only": True},
                   {"map_location": "cpu"}):
        try:
            state = torch.load(path, **kwargs)
        except TypeError:      # 老版本 torch 不认识 weights_only
            continue
        except Exception:
            return None
        if isinstance(state, dict) and state.get("update") is not None:
            try:
                return int(state["update"])
            except (TypeError, ValueError):
                return None
        return None
    return None


def load_stage_best_updates(src):
    """curriculum_eval.csv -> {stage: 该 stage 成功率最高那次的 update}。

    src 可以是 csv 路径或其所在目录；用于给 ppo_best_stage_XX_*.pt 定位。
    """
    path = src if str(src).lower().endswith(".csv") else \
        os.path.join(src, "curriculum_eval.csv")
    if not os.path.isfile(path):
        return {}
    best = {}
    try:
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    stage = int(float(row["stage"]))
                    rate = float(row["success_rate"])
                    spl = float(row["mean_spl"])
                    upd = int(float(row["update"]))
                except (TypeError, ValueError, KeyError):
                    continue
                cur = best.get(stage)
                if cur is None or (rate, spl) > (cur[0], cur[1]):
                    best[stage] = (rate, spl, upd)
    except OSError:
        return {}
    return {k: v[2] for k, v in best.items()}


def ckpt_update(path, last_update=None, stage_best=None):
    """确定某个 checkpoint 对应的 update。返回 (update 或 None, 来源说明)。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"ppo_iter_(\d+)", stem)
    if m:
        return int(m.group(1)), "filename"
    upd = _ckpt_update_from_file(path)
    if upd is not None:
        return upd, "checkpoint"
    m = re.search(r"ppo_best_stage_(\d+)", stem)
    if m and stage_best:
        stage = int(m.group(1))
        if stage in stage_best:
            return stage_best[stage], "curriculum_eval.csv"
    if "final" in stem and last_update:
        return last_update, "train_metrics.csv (approx)"
    return None, None


def ckpt_label(path):
    """竖线标签：iter180 / best s1 / final / 裁剪后的文件名。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"ppo_iter_(\d+)$", stem)
    if m:
        return "iter%s" % m.group(1)
    m = re.match(r"ppo_best_stage_(\d+)_", stem)
    if m:
        return "best s%s" % m.group(1)
    m = re.match(r"ppo_before_stage_(\d+)_", stem)
    if m:
        return "before s%s" % m.group(1)
    if stem == "ppo_final":
        return "final"
    return re.sub(r"^ppo_", "", stem)[:14]


def parse_log(log_path):
    """从控制台日志文本解析（容错）。返回 (updates, dict_of_lists)。"""
    d = {k: [] for k in SERIES_KEYS}
    updates = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            pos = line.find("[PPO] update=")   # torchrun --tee 会加 [default0]: 前缀
            if pos < 0:
                continue
            fields = dict(FIELD_RE.findall(line[pos:]))
            if "loss" not in fields:
                continue
            try:
                updates.append(int(fields["update"]))
            except (KeyError, ValueError):
                continue
            for name, key in LOG_FIELDS.items():
                default = None if key in OPTIONAL_KEYS else 0.0
                d[key].append(_to_float(fields.get(name), default))
            d["cur"].append(_cur_to_float(fields.get("cur")))
    return updates, d


def parse_csv(csv_path):
    """从 train_metrics.csv 解析（首选，字段最全）。返回 (updates, dict_of_lists)。"""
    d = {k: [] for k in SERIES_KEYS}
    updates = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            updates.append(int(float(row["update"])))
            d["loss"].append(_to_float(row.get("loss")))
            d["pol"].append(_to_float(row.get("policy")))
            d["val"].append(_to_float(row.get("value")))
            d["ent"].append(_to_float(row.get("entropy")))
            # episode 均值在"窗口里还没有结束的 episode"时是空值：画成断开，不当 0。
            d["rew"].append(_to_float(row.get("mean_ep_reward"), None))
            d["spl"].append(_to_float(row.get("mean_spl"), None))
            d["arrival"].append(_to_float(row.get("arrival"), None))
            d["oracle"].append(_to_float(row.get("oracle"), None))
            d["cov"].append(_to_float(row.get("coverage"), None))
            d["aux"].append(_to_float(row.get("aux")))
            d["cur"].append(_cur_to_float(row.get("cur_radius")))
            d["shaping"].append(_to_float(row.get("shaping_reward"), None))
            d["coin"].append(_to_float(row.get("coin_reward"), None))
            d["clipped"].append(_to_float(row.get("clipped_ep_reward"), None))
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


def plot(updates, series, specs, subtitle, markers=()):
    """series: {key: 平滑后的值列表}；specs: [(标题, 小字说明, key), ...]。

    markers: 若干 checkpoint 竖线 [{"update": int, "label": str}, ...]，
    落在 x 区间外的会被跳过。
    """
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
        # checkpoint 竖线：每个面板都画，标签只在第一个面板标（避免满图文字）。
        for k, mk in enumerate(markers):
            u = mk["update"]
            if not (x[0] - 0.5 <= u <= x[-1] + 0.5):
                continue
            ax.axvline(u, color=CKPT_MARKER, lw=1.0, ls=(0, (3, 3)), zorder=2)
            if idx == 0:
                ax.annotate(mk["label"], (u, 0.93 - 0.12 * (k % 4)),
                            xycoords=("data", "axes fraction"),
                            xytext=(3, -2), textcoords="offset points",
                            ha="left", va="top", fontsize=8, color=INK_2)
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
    p = argparse.ArgumentParser(
        description="画 PPO 训练曲线（可选：在曲线上标出多个 checkpoint 的位置）")
    p.add_argument("src", help="train_metrics.csv / train.log / 所在目录")
    p.add_argument("out", nargs="?", default="ppo_curves.png", help="输出 PNG")
    p.add_argument("--smooth", type=int, default=5, help="滑窗大小（默认 5）")
    p.add_argument("--checkpoints", default=None,
                   help="逗号分隔的 .pt 列表，在曲线上画竖线标注")
    p.add_argument("--ckpt-dir", "--ckpt_dir", dest="ckpt_dir", default=None,
                   help="目录下按通配符取全部 .pt 作为标注")
    p.add_argument("--ckpt-pattern", "--ckpt_pattern", dest="ckpt_pattern",
                   default="*.pt", help="配合 --ckpt-dir 的通配符（默认 *.pt）")
    args = p.parse_args()
    src, out_path, smooth_win = args.src, args.out, args.smooth

    updates, d = load_metrics(src)
    if not updates:
        print("未在 %s 中找到任何可解析的训练指标，请确认数据源。" % src)
        sys.exit(1)

    # 平滑（不影响原始 CSV，仅画图更可读）
    sd = {k: smooth(v, smooth_win) for k, v in d.items()}

    # 面板规格：(标题, 期望趋势说明, 数据列)。标题统一用英文/ASCII，
    # 避免服务器无 CJK 字体时显示成方框。
    def has(key):
        return any(v is not None for v in d[key])

    # 奖励按来源分面板：总奖励上升而 arrival 不升，说明涨的是辅助奖励。
    specs = [("Mean episode reward", "raw · arrival + shaping + coins", "rew")]
    if has("clipped"):
        specs.append(("Clipped episode reward",
                      "per-step reward clipped, then summed · what PPO optimizes",
                      "clipped"))
    specs += [
        ("Mean SPL", "higher is better", "spl"),
        ("Arrival reward", "paper metric · higher is better", "arrival"),
    ]
    if has("shaping"):
        specs.append(("Shaping reward", "raw, per episode · auxiliary signal",
                      "shaping"))
    if any(d["coin"]):   # 关硬币时全是 0，不占面板
        specs.append(("Coin reward", "raw, per episode · auxiliary signal", "coin"))
    specs += [
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

    # ---- checkpoint 竖线标注（可选）----
    ckpt_paths = resolve_ckpt_paths(args.checkpoints, args.ckpt_dir,
                                    args.ckpt_pattern)
    csv_dir = src if os.path.isdir(src) else os.path.dirname(os.path.abspath(src))
    stage_best = load_stage_best_updates(csv_dir) if ckpt_paths else {}
    markers = []
    for path in ckpt_paths:
        if not os.path.isfile(path):
            print("警告: checkpoint 不存在，跳过标注: %s" % path)
            continue
        u, source = ckpt_update(path, updates[-1] if updates else None,
                                stage_best)
        if u is None:
            print("警告: 无法确定 %s 对应的 update（文件名无轮次、且 torch 不可用"
                  "或文件里没有 update 字段），跳过标注。" % path)
            continue
        markers.append({"update": u, "label": ckpt_label(path)})
        print("    标注 %-28s update=%-6d (%s)" % (os.path.basename(path), u,
                                                   source))
    # 按 update 排序，竖线与标签顺序一致
    markers.sort(key=lambda m: m["update"])

    subtitle = "%d updates · smoothing window %d · %s" % (
        len(updates), smooth_win, os.path.basename(src.rstrip(os.sep)))
    if markers:
        subtitle += " · %d checkpoints marked" % len(markers)

    fig = plot(updates, sd, specs, subtitle, markers=markers)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("已保存曲线图 -> %s  (%d 个 update, %d 面板%s)"
          % (out_path, len(updates), len(specs),
             (", %d 条 checkpoint 竖线" % len(markers)) if markers else ""))


if __name__ == "__main__":
    main()
