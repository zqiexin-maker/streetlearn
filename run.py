#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StreetLearn 一键启动器：训练 / 评估 / 画图，参数全部集中在这里改。

用法（无需记命令行参数）：
    python run.py train          # PPO 训练（读取下面 TRAIN 配置区）
    python run.py eval           # Agent 贪心评估（读取 EVAL 配置区）
    python run.py eval_oracle    # Oracle 实跑评估（同一配置区，自动加 --oracle）
    python run.py eval_all       # 先 Agent 后 Oracle，两遍同种子 head-to-head
    python run.py traj           # 轨迹可视化：跑 N 条导航，叠画在同一张全图地图上
    python run.py plot_log       # 画训练曲线（TRAIN.output_dir 下找 train_metrics.csv）
    python run.py plot_eval      # 画评估对比图（读 EVAL.eval_csv）
    python run.py demo           # 引擎+游戏快速自检（不需要 checkpoint）

配置约定：
  - 所有配置项与底层脚本的命令行参数一一对应；设为 None 且无联动源的项不传
    （走底层默认值）。
  - 【输出目录】eval / traj / plot_* 的产物（CSV / PNG / npy）统一写到
    TRAIN.output_dir，与 train 的 checkpoint、train_metrics.csv 同一文件夹；
    配置区里的 eval_csv / out_png / log_png / eval_png 只写文件名。
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

import os
import subprocess
import sys

# ============================================================================
# 通用配置
# ============================================================================
COMMON = {
    # 数据集路径（LevelDB 目录，如 manhattan_2021_lowres）——必填
    "dataset_path": r"/home/aiseon/zq/streetlearn-master/manhattan_2021_highres",
    # 并行环境数
    "num_envs": 16,
    # 随机种子（评估时 Agent/Oracle 保持一致才能严格对比）
    "seed": 1234,
    # 是否用 GPU
    "cuda": True,
}

# ============================================================================
# 训练配置（python -m streetlearn.python.ppo_train）
# ============================================================================
TRAIN = {
    # 输出目录：checkpoint / train_metrics.csv 都写到这里
    "output_dir": "ppo_ckpts_1",
    # 总训练帧数上限
    "total_timesteps": 100000000,
    # rollout horizon（每个环境每次收集的步数）
    "steps": 256,
    # PPO 超参（一般不用动）
    "ppo_epochs": 2,
    "num_minibatches": 4,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "lam": 0.95,
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
    # 到达阈值(米) / 塑形奖励外边界(米)；必须 0 < min < max
    "min_radius_meters": 100.0,
    "max_radius_meters": 800.0,
    # 课程学习：None/False=关；True=开（用下面的锚点）
    "curriculum": True,          # 例如 True 开启
    "curriculum_frames": "5e6:500,1.5e7:1000,3e7:2000,4.2e7:4000,4.8e7:6000",
    # held-out 留出比例（0=不分；论文 0.25）；评估的 out 集合自动与此一致
    "held_out_fraction": 0.0,
    # 检查点轮转保留数 & 磁盘保护
    "keep_last": 3,
    "min_free_gb": 1.0,
    # 多 worker 数据集副本目录（默认系统临时目录；空间不足时指到大分区）
    "ds_cache_dir": "/home/aiseon/zq/streetlearn-master/tmp_ds",
    # 辅助 heading 自监督任务（默认关）
    "aux_heading": None,         # 例如 True 开启
    # 视觉预训练
    "pretrained_vision": None,   # 例如 True 开启
}

# ============================================================================
# 评估配置（python -m streetlearn.python.ppo_eval）
# ============================================================================
EVAL = {
    # checkpoint 路径；None = 自动取 TRAIN.output_dir/ppo_final.pt
    "checkpoint": None,
    # 评估 episode 数
    "num_episodes": 40,
    # 单 episode 最大步数；None = 自动取 TRAIN.frame_cap（评估口径必须相等）
    "max_steps": None,
    # 固定目标采样种子；None = 自动取 COMMON.seed（Agent/Oracle 一致即 head-to-head）
    "goal_seed": None,
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
    "checkpoint": None,
    # 画几条轨迹（全部叠画在同一张全图地图上，按编号区分）
    "num_trajectories": 9,
    # 单条轨迹最大步数；None = 自动取 TRAIN.frame_cap（与评估口径一致）
    "max_steps": None,
    # 并行环境数（可视化只记录 env 0 的轨迹，1 即够）
    "num_envs": 1,
    # 固定目标采样种子；None = 自动取 COMMON.seed（可复现同一批任务）
    "goal_seed": None,
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
    # 下面两张图的文件名都写到 TRAIN.output_dir 下
    "log_png": "ppo_curves_1.png",
    "smooth_win": 20,            # 平滑窗口
    # plot_eval：评估对比图
    "eval_png": "ppo_eval_triple_1.png",
}

# ============================================================================
# 以下为解析 / 校验 / 调度逻辑，一般不用改
# ============================================================================
ROOT = os.path.dirname(os.path.abspath(__file__))


class ConfigError(Exception):
    """配置错误：报错并退出，不进入训练/评估。"""


def _resolve():
    """把必须联动的参数自动复制（None → 从源参数取值）。

    返回 (train, ev, tr) 三个解析后的副本；被自动填充的项会打印出来。
    """
    train = dict(TRAIN)
    ev = dict(EVAL)
    tr = dict(TRAJ)
    auto = []

    if train.get("goal_timeout") is None:
        train["goal_timeout"] = train["frame_cap"]
        auto.append("TRAIN.goal_timeout = TRAIN.frame_cap = %d" % train["frame_cap"])

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

    if auto:
        print("[run] 自动联动参数：")
        for a in auto:
            print("    %s" % a)
    return train, ev, tr


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


def _validate_common():
    dp = COMMON.get("dataset_path")
    if not dp or not isinstance(dp, str):
        _fail("COMMON.dataset_path 未配置（数据集 LevelDB 目录，必填）")
    if not os.path.isdir(dp):
        _fail("COMMON.dataset_path 不存在: %r\n"
              "    请确认数据集已下载并解压，路径写正确。" % dp)
    if not isinstance(COMMON.get("num_envs"), int) or COMMON["num_envs"] < 1:
        _fail("COMMON.num_envs 必须是 >= 1 的整数，当前是 %r" % COMMON.get("num_envs"))
    if COMMON.get("cuda"):
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


def _validate_train(t):
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

    rmin, rmax = t.get("min_radius_meters"), t.get("max_radius_meters")
    if not (isinstance(rmin, (int, float)) and rmin > 0):
        _fail("TRAIN.min_radius_meters 必须 > 0，当前是 %r" % rmin)
    if not (isinstance(rmax, (int, float)) and rmax > 0):
        _fail("TRAIN.max_radius_meters 必须 > 0，当前是 %r" % rmax)
    if rmin >= rmax:
        _fail("TRAIN.min_radius_meters(%g) 必须 < TRAIN.max_radius_meters(%g)，"
              "否则塑形奖励区间为空" % (rmin, rmax))

    hf = t.get("held_out_fraction")
    if not isinstance(hf, (int, float)) or not (0 <= hf < 1):
        _fail("TRAIN.held_out_fraction 必须在 [0,1) 内，当前是 %r" % hf)

    if t.get("curriculum"):
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


def _validate_eval(t, ev):
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
    src = PLOT.get("log_src") or os.path.join(t["output_dir"], "train_metrics.csv")
    if os.path.isfile(src):
        return src
    alt = os.path.join(t["output_dir"], "train.log")
    if os.path.isfile(alt):
        return alt
    _fail("在 TRAIN.output_dir(%r) 下找不到 train_metrics.csv 或 train.log，"
          "无法画训练曲线。先跑 python run.py train，或用 PLOT.log_src 指定来源。"
          % t["output_dir"])


def _validate_plot_eval(ev):
    src = ev.get("eval_csv")
    if not os.path.isfile(src):
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


def _run(module, extra_args):
    cmd = [sys.executable, "-m", module] + extra_args
    print("[run] %s\n" % " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def cmd_train(train, ev, tr=None):
    args = []
    for k, v in {**COMMON, **train}.items():
        # 子开关未开时，不传其配套参数（如 --curriculum 关时不传 curriculum_frames）
        if k == "curriculum_frames" and not train.get("curriculum"):
            continue
        if k == "held_out_fraction" and not train.get("held_out_fraction"):
            continue
        args += _flag(k, v)
    return _run("streetlearn.python.ppo_train", args)


def _eval_args(ev, oracle=False):
    args = []
    cfg = {**COMMON, **ev}
    if oracle:
        cfg["oracle"] = True
    for k, v in cfg.items():
        # held-out 未开启时不传 held_out_fraction，避免误改采样行为
        if k == "held_out_fraction" and not cfg.get("held_out"):
            continue
        args += _flag(k, v)
    return args


def cmd_eval(train, ev, tr=None, oracle=False):
    return _run("streetlearn.python.ppo_eval", _eval_args(ev, oracle))


def cmd_eval_all(train, ev, tr=None):
    rc1 = cmd_eval(train, ev, oracle=False)
    rc2 = cmd_eval(train, ev, oracle=True)
    print("\n[run] Agent 与 Oracle 两遍使用相同 seed/goal_seed/num_envs，"
          "即严格 head-to-head 对比。")
    return rc1 or rc2


def cmd_traj(train, ev, tr):
    """轨迹可视化：跑 TRAJ.num_trajectories 条导航，叠画在同一张全图地图上。

    参数取 {COMMON, TRAJ} 的并集（checkpoint/max_steps/goal_seed 已联动），
    与 ppo_traj_viz.py 的命令行参数一一对应。
    """
    args = []
    cfg = {**COMMON, **tr}
    # num_envs 联动取 TRAJ 自己的（默认 1，可视化够用且最省内存）。
    cfg["num_envs"] = tr["num_envs"]
    for k, v in cfg.items():
        args += _flag(k, v)
    return _run("streetlearn.python.ppo_traj_viz", args)


def cmd_plot_log(train, ev, tr=None):
    src = _validate_plot_log(train)
    png = os.path.join(train["output_dir"], PLOT["log_png"])
    # plot_ppo_log 的平滑参数名是 --smooth（传 --smooth_win 会被当成输出文件名）。
    return _run("streetlearn.tools.plot_ppo_log",
                [src, png, "--smooth", str(PLOT["smooth_win"])])


def cmd_plot_eval(train, ev, tr=None):
    src = _validate_plot_eval(ev)
    png = os.path.join(train["output_dir"], PLOT["eval_png"])
    return _run("streetlearn.tools.plot_ppo_eval", [src, png])


def cmd_demo(train, ev, tr=None):
    return _run("streetlearn.tools.run_nav",
                ["--dataset_path", COMMON["dataset_path"], "--mode", "game"])


# 每条命令: (执行函数, 需要的校验组)
COMMANDS = {
    "train":       (cmd_train,     ("common", "train")),
    "eval":        (cmd_eval,      ("common", "train", "eval")),
    "eval_oracle": (lambda t, e, tr: cmd_eval(t, e, oracle=True),
                                    ("common", "train", "eval")),
    "eval_all":    (cmd_eval_all,  ("common", "train", "eval")),
    "traj":        (cmd_traj,      ("common", "train", "traj")),
    "plot_log":    (cmd_plot_log,  ("plot_log",)),
    "plot_eval":   (cmd_plot_eval, ("plot_eval",)),
    "demo":        (cmd_demo,      ("common",)),
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("可用命令: %s" % " / ".join(COMMANDS))
        sys.exit(1)

    # 相对路径（output_dir / checkpoint 等）统一以项目根目录为基准：校验与
    # 子进程（cwd=ROOT）看到的是同一个文件，从任意目录启动都一致。
    os.chdir(ROOT)
    train, ev, tr = _resolve()
    fn, checks = COMMANDS[sys.argv[1]]
    try:
        if "common" in checks:
            _validate_common()
        if "train" in checks:
            _validate_train(train)
        if "eval" in checks:
            _validate_eval(train, ev)
        if "traj" in checks:
            _validate_traj(train, tr)
        # plot_log / plot_eval 的校验在各自 cmd 里做（返回解析出的 src）
        sys.exit(fn(train, ev, tr))
    except ConfigError as e:
        print("\n[run] 配置错误: %s\n" % e)
        sys.exit(2)


if __name__ == "__main__":
    main()
