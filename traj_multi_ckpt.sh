#!/usr/bin/env bash
# ============================================================================ #
# 多 checkpoint 批量轨迹可视化（在装有数据集的 Linux 服务器上运行）
#
# 用法（在项目根目录下）：
#   bash traj_multi_ckpt.sh                                  # runs/ 下最近一次训练的全部 .pt
#   bash traj_multi_ckpt.sh runs/ppo_6gpu_32env_20261007_120000
#   CKPT_PATTERN="ppo_best_stage_*.pt" bash traj_multi_ckpt.sh   # 只看各档最优权重
#   GOAL_BAND=full bash traj_multi_ckpt.sh                   # 目标不限距离（全图）
#   GOAL_BAND=500,800 bash traj_multi_ckpt.sh                # 所有 ckpt 统一测 (500,800] 米
#   DATASET=/home/gong/qw/manhattan_2021_lowres N_TRAJ=12 MAX_STEPS=2000 bash traj_multi_ckpt.sh
#
# GOAL_BAND 默认 stage：每个 checkpoint 用它自己所在课程档的距离段（best_stage_K /
# final 是所在档，before_stage_K 是即将开始的第 K 档）。目标距离段相同的 checkpoint
# 在相同 GOAL_SEED 下面对完全一致的 (起点, 目标) 序列，可直接对比。
#
# 产物（写在 ckpt 目录下）：
#   <ckpt名>_traj_map.png      每个 ckpt 一张全图轨迹
#   trajectories_<ckpt名>.npy  轨迹原始数据
#   traj_summary.csv           横向对比表
# ============================================================================ #
set -euo pipefail

# ---- 可用环境变量覆盖的默认值 ----
DATASET="${DATASET:-/home/gong/qw/manhattan_2021_lowres}"  # StreetLearn LevelDB 数据集
CKPT_DIR="${1:-}"                                          # checkpoint 目录（第 1 参数）
if [[ -z "$CKPT_DIR" ]]; then
    CKPT_DIR=$(ls -td runs/*/ 2>/dev/null | head -n 1 || true)
    CKPT_DIR="${CKPT_DIR%/}"
fi
CKPT_PATTERN="${CKPT_PATTERN:-ppo_*.pt}"                   # 通配符
N_TRAJ="${N_TRAJ:-9}"                                      # 每个 ckpt 跑几条轨迹
MAX_STEPS="${MAX_STEPS:-2000}"                             # 单条最大步数（=frame_cap）
GOAL_SEED="${GOAL_SEED:-1234}"                             # 固定目标采样种子
GOAL_BAND="${GOAL_BAND:-stage}"                            # stage / full / LOW,HIGH
NUM_ENVS="${NUM_ENVS:-1}"                                  # 可视化 1 个环境即可
DS_CACHE_DIR="${DS_CACHE_DIR:-$PWD/tmp_ds}"                # NUM_ENVS>1 时的数据集副本目录
CUDA_FLAG="${CUDA_FLAG:---cuda}"                           # 无 GPU 时置空: CUDA_FLAG= bash ...

if [ ! -d "$DATASET" ]; then
    echo "[ERR] 数据集目录不存在: $DATASET （用 DATASET=路径 覆盖）" >&2
    exit 2
fi
if [ -z "$CKPT_DIR" ] || [ ! -d "$CKPT_DIR" ]; then
    echo "[ERR] checkpoint 目录不存在: '$CKPT_DIR'（作为第 1 个参数传入）" >&2
    exit 2
fi

PYTHON="${PYTHON:-python}"
echo "[TRAJ] dataset = $DATASET"
echo "[TRAJ] ckpt_dir = $CKPT_DIR  pattern = $CKPT_PATTERN  goal_band = $GOAL_BAND"
echo "[TRAJ] n_traj=$N_TRAJ max_steps=$MAX_STEPS goal_seed=$GOAL_SEED"

PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH} $PYTHON -m streetlearn.python.ppo_traj_viz \
    --ckpt_dir "$CKPT_DIR" \
    --ckpt_pattern "$CKPT_PATTERN" \
    --dataset_path "$DATASET" \
    --num_envs "$NUM_ENVS" \
    --num_trajectories "$N_TRAJ" \
    --max_steps "$MAX_STEPS" \
    --goal_seed "$GOAL_SEED" \
    --goal_band "$GOAL_BAND" \
    --ds_cache_dir "$DS_CACHE_DIR" \
    --out_dir "$CKPT_DIR" \
    --out_png "traj_map.png" \
    $CUDA_FLAG

echo "[TRAJ] 完成。轨迹图与 traj_summary.csv 见: $CKPT_DIR"
