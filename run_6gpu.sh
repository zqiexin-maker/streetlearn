#!/usr/bin/env bash
# ============================================================================ #
# 单机 6×RTX 4090 PPO 训练（后台运行；训练结束后自动跑 run.py report 出全部图）
#
# 用法（在项目根目录）：
#   bash run_6gpu.sh                      # 默认：接着 runs/ 下最近一次"同配置、没跑完"
#                                         #       的训练续训；没有就新开一个 run
#   RESUME=none bash run_6gpu.sh          # 强制新开
#   RESUME=runs/xxx/ppo_iter_120.pt bash run_6gpu.sh   # 从指定 checkpoint 续训（写回该目录）
#   OUTPUT=runs/xxx bash run_6gpu.sh      # 指定输出目录；里面有 checkpoint 就从最新的续训
#   POST_REPORT=0 bash run_6gpu.sh        # 训练结束后不自动出图
#   NUM_ENVS=24 bash run_6gpu.sh          # 每卡环境数（总环境数 = 6 × NUM_ENVS）
#   SHAPING_RATIO=0.6 / SHAPING_SCALE=0.5 / REWARD_CLIP=none / CURRICULUM_STAGES=500,800,1200
#   CURRICULUM_EVAL_ENVS=16               # 每卡最多用几个训练环境跑阶段测试（默认全部）
#   EXTRA_ARGS="--curriculum_pass_rate 1.0 --total_timesteps 200000000"   # 其他 run.py train 参数
#   REPORT_EXTRA="--num_episodes 100"     # 追加给训练后 report 的参数
#   EXP=B SEED=2 bash run_6gpu.sh         # 对照实验预设（run.py 的 EXPERIMENTS：base / gae_v1 / A / B /
#                                         # C / serial / mask_position / mask_yaw / mask_status_bar /
#                                         # mask_all，可用逗号叠加），输出目录带预设名和种子
#   SELFTEST=0 bash run_6gpu.sh           # 跳过启动前的 CPU 自检（run.py selftest）
#   KILL_STALE=1 bash run_6gpu.sh         # 先结束之前残留的训练进程（启动前会列出），再启动
#
# 终端里中文显示成乱码（如"纭鍙互缁撴潫"）时，把 SSH 客户端的字符编码改成 UTF-8。
#
# 续训只会自动接上 run_config.json 里关键参数（课程档位、分辨率、塑形、frame_cap 等）
# 与本次一致、且还没写出 ppo_final.pt 的 run；改了这些参数会自动新开，不会再把
# 旧实验的 checkpoint 喂给新配置。
# ============================================================================ #
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/gong/qw/streetlearn}"
WRAPPER="${WRAPPER-/home/gong/qw/with_healthy_gpus.sh}"   # 置空（WRAPPER=）则不用包装脚本
PY="${PY:-/home/gong/qw/.venv/bin/python}"
NPROC="${NPROC:-6}"
NUM_ENVS="${NUM_ENVS:-32}"
RUNS_DIR="$PROJECT_DIR/runs"
DS_CACHE_DIR="${DS_CACHE_DIR:-$PROJECT_DIR/tmp_ds}"
OUTPUT="${OUTPUT:-}"
RESUME="${RESUME:-auto}"
# 兼容旧变量：RESUME_CKPT=<pt> 指定 checkpoint；RESUME_LATEST=0 强制新开。
if [[ -n "${RESUME_CKPT:-}" ]]; then
  RESUME="$RESUME_CKPT"
elif [[ "${RESUME_LATEST:-1}" == "0" ]]; then
  RESUME=none
fi

cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${GPUS:-GPU-c2e2e3aa-922d-b71a-e33e-7ada03e2899b,GPU-ed6b37dd-cfd5-4bf1-08f9-eec91188a0f2,GPU-4b3885ac-a3c0-2d89-7a02-1e0b8c073064,GPU-40eb3b01-f7aa-b531-53f2-9118f175d28d,GPU-6bfe1857-1944-743a-bc9c-05f689b55034,GPU-3b553fab-c11d-578b-752f-1aa5c564aa3d}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

run_py() {
  if [[ -n "$WRAPPER" ]]; then
    bash "$WRAPPER" "$PY" "$@"
  else
    "$PY" "$@"
  fi
}

if [[ ! -x "$PY" ]]; then
  echo "Python 不存在或不可执行：$PY" >&2
  exit 1
fi
if [[ -n "$WRAPPER" && ! -f "$WRAPPER" ]]; then
  echo "GPU 包装脚本不存在：$WRAPPER（不需要的话用 WRAPPER= 关掉）" >&2
  exit 1
fi

# ---- 预检：残留的训练进程（最先做：有残留时什么都不启动，也不建输出目录）----
# 之前的 run 崩掉或卡住后，它的进程（尤其是环境 worker）可能一直活着：锁着
# $DS_CACHE_DIR 里的数据集副本（新 run 会报 "IO error: lock .../LOCK: Resource
# temporarily unavailable"），还占着内存和显存。工具会列出它们属于哪个输出目录、
# 那个目录最后一次写入是什么时候，用来判断那次训练是还在跑还是早就卡住了。
mkdir -p "$RUNS_DIR"
STALE_FILE="$RUNS_DIR/stale_pids.txt"
stale_rc=0
if [[ "${KILL_STALE:-0}" == "1" ]]; then
  # 列出并结束（先 TERM，20 秒后还在的 KILL）；结束不掉就不启动。
  "$PY" -m streetlearn.tools.stale_procs --cache_dir "$DS_CACHE_DIR" --kill || exit 1
else
  "$PY" -m streetlearn.tools.stale_procs --cache_dir "$DS_CACHE_DIR" \
    --pid_file "$STALE_FILE" || stale_rc=$?
fi
if [[ $stale_rc -ne 0 ]]; then
  if [[ "${ALLOW_CONCURRENT:-0}" == "1" ]]; then
    echo "ALLOW_CONCURRENT=1：不结束它们，继续启动（它们占的内存和显存不会释放）。" >&2
  else
    echo "[run_6gpu] NOT STARTED: nothing was launched and no run directory was created." >&2
    echo "[run_6gpu] Stop the processes above and start: KILL_STALE=1 bash run_6gpu.sh" >&2
    echo "训练没有启动（也没有建输出目录）。确认上面这些进程可以结束后：" >&2
    echo "  KILL_STALE=1 bash run_6gpu.sh        # 先结束它们（TERM，20 秒后 KILL），再启动" >&2
    echo "  手动结束：kill \$(cat $STALE_FILE)，不退出再 kill -9 \$(cat $STALE_FILE)" >&2
    echo "  不结束、并存启动（不推荐）：ALLOW_CONCURRENT=1 bash run_6gpu.sh" >&2
    exit 1
  fi
fi

DATASET="${DATASET:-$("$PY" -c 'import run; print(run.COMMON["dataset_path"])')}"
if [[ ! -d "$DATASET" ]]; then
  echo "数据集目录不存在：$DATASET" >&2
  echo "可用 DATASET=/真实/数据集路径 bash run_6gpu.sh 覆盖。" >&2
  exit 1
fi

if [[ -n "$WRAPPER" ]]; then
  bash -n "$WRAPPER"
fi
if [[ "${GPU_CHECK:-1}" == "1" ]]; then
  echo "检查 $NPROC 张 GPU 和 CUDA 张量分配..."
  run_py -c "import torch; n=$NPROC; assert torch.cuda.device_count() == n, torch.cuda.device_count(); [torch.zeros(1, device='cuda:%d' % i) for i in range(n)]; [torch.cuda.synchronize(i) for i in range(n)]; print('%d GPUs OK' % n)"
fi

EXP="${EXP:-}"
if [[ -n "$EXP" ]]; then
  "$PY" - "$EXP" <<'PYEOF' || exit 1
import sys
import run
bad = [n for n in (x.strip() for x in sys.argv[1].split(",")) if n and n not in run.EXPERIMENTS]
if bad:
    sys.exit("未知的实验预设 %s，可选：%s" % (", ".join(bad), " / ".join(run.EXPERIMENTS)))
PYEOF
fi
SEED_EFF="${SEED:-$("$PY" -c 'import run; print(run.COMMON["seed"])')}"

# 启动前自检（CPU，十几秒）：GAE 数值、串 / 并联网络、输入遮蔽、存档往返。
if [[ "${SELFTEST:-1}" == "1" ]]; then
  mkdir -p "$RUNS_DIR"
  echo "启动前自检：run.py selftest ..."
  if ! "$PY" run.py selftest > "$RUNS_DIR/selftest_last.log" 2>&1; then
    tail -n 40 "$RUNS_DIR/selftest_last.log" >&2
    echo "自检失败，完整输出见 $RUNS_DIR/selftest_last.log（确认无碍可用 SELFTEST=0 跳过）" >&2
    exit 1
  fi
  grep -E "^(Ran |OK)" "$RUNS_DIR/selftest_last.log" | tr '\n' ' ' || true
  echo
fi

TRAIN_ARGS=(
  --dataset_path "$DATASET"
  --nproc_per_node "$NPROC"
  --num_envs "$NUM_ENVS"
  --ds_cache_dir "$DS_CACHE_DIR"
)
# 奖励塑形 / 课程的临时覆盖（留空则沿用 run.py 的 TRAIN 配置区）：
#   SHAPING_RATIO=0.6 bash run_6gpu.sh      塑形窗口 = 0.6 × 目标初始距离
#   REWARD_CLIP=none  bash run_6gpu.sh      不裁剪奖励（原始量级进 GAE）
#   CURRICULUM_STAGES=500,800,1200 bash run_6gpu.sh   改课程档位
if [[ -n "${SHAPING_RATIO:-}" ]]; then
  TRAIN_ARGS+=(--shaping_radius_ratio "$SHAPING_RATIO")
fi
if [[ -n "${SHAPING_SCALE:-}" ]]; then
  TRAIN_ARGS+=(--shaping_scale "$SHAPING_SCALE")
fi
if [[ -n "${REWARD_CLIP:-}" ]]; then
  TRAIN_ARGS+=(--reward_clip "$REWARD_CLIP")
fi
if [[ -n "${CURRICULUM_STAGES:-}" ]]; then
  TRAIN_ARGS+=(--curriculum_stages "$CURRICULUM_STAGES")
fi
if [[ -n "${CURRICULUM_EVAL_ENVS:-}" ]]; then
  TRAIN_ARGS+=(--curriculum_eval_envs "$CURRICULUM_EVAL_ENVS")
fi
if [[ -n "$EXP" ]]; then
  TRAIN_ARGS+=(--exp "$EXP")
fi
if [[ -n "${SEED:-}" ]]; then
  TRAIN_ARGS+=(--seed "$SEED")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra <<< "$EXTRA_ARGS"
  TRAIN_ARGS+=("${extra[@]}")
fi
REPORT_ARGS=(--dataset_path "$DATASET" --ds_cache_dir "$DS_CACHE_DIR")
if [[ -n "${REPORT_EXTRA:-}" ]]; then
  read -r -a extra <<< "$REPORT_EXTRA"
  REPORT_ARGS+=("${extra[@]}")
fi

# ---- 决定续训还是新开 ----
OUT_ARGS=()
if [[ -n "$OUTPUT" ]]; then
  OUT_ARGS=(--output_dir "$OUTPUT")
fi
RESUME_CKPT=""
case "$RESUME" in
  none|no|0|"")
    ;;
  auto|latest)
    # run.py latest_ckpt 只在 stdout 打印 checkpoint 路径（没有则为空）。
    RESUME_CKPT=$("$PY" run.py latest_ckpt "${TRAIN_ARGS[@]}" ${OUT_ARGS[@]+"${OUT_ARGS[@]}"} 2>/dev/null | tail -n 1 || true)
    ;;
  *)
    RESUME_CKPT="$RESUME"
    if [[ ! -f "$RESUME_CKPT" ]]; then
      echo "续训检查点不存在：$RESUME_CKPT" >&2
      exit 1
    fi
    ;;
esac

if [[ -n "$RESUME_CKPT" ]]; then
  if [[ -z "$OUTPUT" ]]; then
    OUTPUT=$(dirname "$RESUME_CKPT")
  fi
  TRAIN_ARGS+=(--resume "$RESUME_CKPT")
  LOG="$OUTPUT/train_resume_$(date +%Y%m%d_%H%M%S).log"
  echo "从检查点续训：$RESUME_CKPT"
else
  if [[ -z "$OUTPUT" ]]; then
    # 与 run.py 的命名一致：对照实验带上预设名和种子。
    EXP_TAG=""
    if [[ -n "$EXP" ]]; then
      EXP_TAG="_${EXP//,/-}_seed${SEED_EFF}"
    fi
    OUTPUT="$RUNS_DIR/ppo_${NPROC}gpu_${NUM_ENVS}env${EXP_TAG}_$(date +%Y%m%d_%H%M%S)"
  elif [[ "$RESUME" == "auto" || "$RESUME" == "latest" ]]; then
    # 指定的目录里还没有 checkpoint（例如第一次保存前就崩了）：允许在里面重新开始。
    TRAIN_ARGS+=(--resume auto)
  fi
  LOG="$OUTPUT/train.log"
  echo "没有可续训的 checkpoint，新开训练。"
fi
TRAIN_ARGS+=(--output_dir "$OUTPUT")

echo "输出目录: $OUTPUT"
echo "共享数据缓存: $DS_CACHE_DIR"
mkdir -p "$OUTPUT" "$DS_CACHE_DIR"

# 打印本次的奖励 / 网络 / 课程配置（run.py TRAIN + EXP 预设），便于事后回溯实验。
# 训练日志开头还会打印合并环境默认值之后实际生效的奖励配置和网络结构。
TRAIN_CFG_LINE=$("$PY" - "$EXP" <<'PYEOF'
import sys
import run
t = dict(run.TRAIN)
for name in (x.strip() for x in sys.argv[1].split(",")):
    if name:
        t.update(run.EXPERIMENTS[name])
keys = ("gae_version", "min_radius_meters", "max_radius_meters",
        "shaping_radius_ratio", "shaping_scale", "reward_clip", "no_coins",
        "model_topology", "mask_position", "mask_yaw", "mask_status_bar",
        "curriculum_stages", "curriculum_pass_rate", "curriculum_eval_episodes")
print(" ".join("%s=%s" % (k, t.get(k, "-")) for k in keys))
PYEOF
)
echo "奖励/网络/课程配置（含 EXP=${EXP:-无}，未含其他环境变量覆盖）: $TRAIN_CFG_LINE"

# 后台：训练 →（成功结束后）report：训练曲线 + 课程测试图 + 最终权重 Agent/Oracle
# 评估 + 各档最优权重轨迹图，全部写进 $OUTPUT。忽略 SIGHUP，关掉终端不影响。
(
  trap '' HUP
  rc=0
  run_py "$PROJECT_DIR/run.py" train "${TRAIN_ARGS[@]}" || rc=$?
  if [[ $rc -eq 0 && "${POST_REPORT:-1}" == "1" ]]; then
    echo "[run_6gpu] 训练结束，开始出图：run.py report --output_dir $OUTPUT"
    run_py "$PROJECT_DIR/run.py" report --output_dir "$OUTPUT" "${REPORT_ARGS[@]}" || rc=$?
  fi
  echo "[run_6gpu] 结束，exit=$rc"
  exit $rc
) > "$LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true

echo "后台 PID: $PID"
echo "日志文件: $LOG"
echo "查看日志: tail -f '$LOG'"
echo "各 rank 日志: $OUTPUT/torchrun_logs/"
echo "停止训练: pkill -f 'run.py train' ; pkill -f streetlearn.python.ppo_train"
echo "续训（崩溃/停止后）: 直接再跑 bash run_6gpu.sh，或 OUTPUT=$OUTPUT bash run_6gpu.sh"
echo "手动出图: $PY run.py report --output_dir $OUTPUT"
