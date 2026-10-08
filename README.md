# StreetLearn (Pure Python)

## Overview

Pure Python implementation of the [**StreetLearn**](http://streetlearn.cc) engine and environment for training navigation agents in real-world photographic street environments. Originally developed by Google DeepMind (NeurIPS 2018), this version **removes all C++/Bazel/CLIF dependencies** — the entire engine is implemented in Python using numpy, Pillow, and plyvel.

> 训练栈已切换为 **PyTorch PPO**（原 IMPALA/TensorFlow-Sonnet 的 agents、experiment.py 与 ui/ 调试智能体已删除）。

This environment was used in:
- [1] ["Learning to Navigate in Cities Without a Map"](http://papers.nips.cc/paper/7509-learning-to-navigate-in-cities-without-a-map) (NeurIPS 2018)
- [2] ["Cross-View Policy Learning for Street Navigation"](https://arxiv.org/pdf/1906.05930) (ICCV 2019)
- [3] ["Learning to follow directions in Street View"](https://arxiv.org/pdf/1903.00401) (AAAI 2020)
- [4] ["The StreetLearn Environment and Dataset"](https://arxiv.org/pdf/1903.01292)

### Bibtex

```
@inproceedings{mirowski2018learning,
  title={Learning to Navigate in Cities Without a Map},
  author={Mirowski, Piotr and Grimes, Matthew Koichi and Malinowski, Mateusz and Hermann, Karl Moritz and Anderson, Keith and Teplyashin, Denis and Simonyan, Karen and Kavukcuoglu, Koray and Zisserman, Andrew and Hadsell, Raia},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2018}
}
```

## Quick Start (Linux)

### 1. Install dependencies

```bash
# System dependency for plyvel (LevelDB)
sudo apt-get install libleveldb-dev

# Python dependencies
cd streetlearn-master
pip install -r requirements.txt
```

### 2. Generate protobuf module

```bash
python -m streetlearn.proto.gen_pb2
```

This creates `streetlearn/proto/streetlearn_pb2.py` from `streetlearn.proto`. Uses `protoc` or `grpcio-tools` (auto-installed if needed).

### 3. Verify installation

```bash
python setup_py.py check
```

### 4. Obtain the dataset

Request the StreetLearn dataset from the [project website](https://sites.google.com/view/streetlearn/). Available datasets:
- **manhattan_lowres** (408×204 panoramas, 56k) — recommended for fast experimentation
- **manhattan_highres** (1632×408 panoramas, 56k)
- **pittsburgh_lowres** / **pittsburgh_highres** (58k)
- **touchdown_manhattan_lowres** / **touchdown_manhattan_highres** (29k)

### 5. Run: train / evaluate / plot

Default parameters live in the config sections of [`run.py`](run.py). Any key can be overridden for one run with `--key value`:

```bash
python run.py train                 # PPO 训练（新建 runs/ppo_<GPU数>gpu_<环境数>env_<时间戳>/）
python run.py train --resume auto   # 断点恢复：接着 runs/ 下最近一次同配置、没跑完的训练
python run.py train --resume runs/xxx/ppo_iter_120.pt   # 从指定 checkpoint 续训（写回该目录）
python run.py train --curriculum_pass_rate 1.0          # 每次测试都要 100/100 才算达标
python run.py train --exp B --seed 2                    # 对照实验预设（见下文 Controlled experiments）
python run.py selftest              # 训练前自检（CPU，十几秒）：GAE 数值、串/并联网络、输入遮蔽、存档往返
python run.py eval_all              # Agent + Oracle 两遍（同种子、同一批路线 head-to-head）
python run.py plot_log              # 训练曲线（train_metrics.csv → ppo_curves_1.png）
python run.py plot_curriculum       # 课程测试总览（curriculum_eval.csv → curriculum_eval.png）
python run.py plot_eval             # 评估对比图（eval_results.csv → ppo_eval_triple_1.png）
python run.py traj --ckpt_dir runs/xxx --ckpt_pattern "ppo_best_stage_*.pt"   # 各档轨迹图
python run.py report                # 训练后一键出图：上面四类图全部生成
python run.py demo                  # 引擎+游戏快速自检（无需 checkpoint）
```

`eval` / `traj` / `plot_*` / `report` default to the most recent run under `runs/`
(pass `--output_dir runs/xxx` to pick another). `train` refuses to write into a
directory that already holds results unless `--resume` is given, so an old run's
curves and checkpoints are never overwritten.

#### Six GPUs on one Linux machine

```bash
bash run_6gpu.sh                    # 后台训练；训练结束后自动 run.py report
RESUME=none bash run_6gpu.sh        # 强制新开
OUTPUT=runs/xxx bash run_6gpu.sh    # 接着指定的 run 续训
EXP=B SEED=2 bash run_6gpu.sh       # 对照实验预设 + 种子（输出目录带预设名和种子）
tail -f runs/<run>/train.log
```

Before launching, the script runs `run.py selftest` on the CPU (log in
`runs/selftest_last.log`; `SELFTEST=0` skips it).

`run_6gpu.sh` launches six synchronous PPO workers with PyTorch DDP
(`python run.py train --nproc_per_node 6 --num_envs 32`): 32 CPU environments per
GPU, 192 in total. `total_timesteps` and all curriculum frame counts are **global**
environment frames across all GPUs. By default it resumes the most recent run
whose key settings (curriculum stages, resolution, shaping, `frame_cap`, ...) match
the current configuration and that has not written `ppo_final.pt` yet; otherwise it
starts a new run. Only rank 0 writes logs and checkpoints. Each training worker
needs its own LevelDB directory; by default `ds_cache_dir` is `<project>/tmp_ds`,
and a cache on the same filesystem as the dataset uses hard links instead of full
copies. Torchrun keeps each rank's stdout/stderr under `<output_dir>/torchrun_logs/`.
Collective operations time out after `--ddp_timeout_minutes` (default 120; NCCL's
own default of 10 minutes is too short for the first dataset-copy build).

#### Performance curriculum

Stages default to 200, 300, 500, 800, 1200, 2000, 4000 and 6000 m. Training goals
are drawn from `(min_radius_meters, stage]` (goals already inside the 100 m arrival
radius are excluded). After at least 1 million frames in a stage, the policy is
tested every 500,000 frames on 100 single-goal routes from the current band
`(previous stage, stage]` (the first band starts at the 100 m arrival radius). Two
consecutive tests at >= 95% advance to the next stage; a stage that has not passed
after 10 million frames stops training.

Tests run greedily (argmax) on the training environments themselves: routes are
split across GPUs, and each environment gets a fixed quota of routes decided in
advance, so short (mostly successful) routes cannot crowd out long ones and inflate
the success rate. After a test the environments return to training with fresh
episodes. Each test appends a row to `curriculum_eval.csv`: stage and band,
cumulative training frames (`global_step`), frames in the stage, successes,
`all_success` (100/100), farthest successful route (`max_success_distance`), mean
and maximum steps to reach the goal, pass streak and promotion.
`curriculum_summary.json` keeps the highest passed distance and per-stage history.

Checkpoints: `ppo_best_stage_XX_<m>.pt` (best test of each stage),
`ppo_before_stage_XX_<m>.pt` (weights at promotion, before training on stage XX),
`ppo_final.pt`, and rolling `ppo_iter_<update>.pt` (every `save_interval` updates
and after every test). `checkpoint_latest.json` names the newest one; `--resume`
restores network, optimizer, frame count and curriculum progress, and drops CSV
rows written after that checkpoint. `run_config.json` and `run_config_history.jsonl`
record the effective arguments.

Reward shaping uses an adaptive window instead of a fixed band: for every new
goal the shaping radius is `clamp(shaping_radius_ratio * initial_distance,
min_radius_meters * 1.05, max_radius_meters)`, so the reward level is ~0 when the
agent enters the shaping zone (the reference implementation relies on the same
invariant implicitly, with a fixed 200 m window against a curriculum starting at
500 m). `shaping_radius_ratio <= 0` restores the fixed window. Rewards are
clipped to [-1, 1] before GAE (`--reward_clip`, matching the reference agent's
`reward_clipping=abs_one`). Coin rewards are off by default (`no_coins: True`):
the environment's default puts a +1 coin on half of all panoramas, which after
clipping is worth as much as reaching the goal. `--no_coins false` turns them back on.
The training log starts with the effective reward settings (`[PPO] reward config:
...`, after merging the environment defaults; the same config is saved in every
checkpoint).

Rewards are logged by source so a rising total can be traced: `train_metrics.csv`
has `arrival`, `shaping_reward` and `coin_reward` (raw per-episode sums from the
branch of `CourierGame` that paid them; together they equal `mean_ep_reward`) and
`clipped_ep_reward` (per-step total reward clipped to [-1, 1], then summed: what
PPO actually optimizes). Clipping acts on each step's total, so the clipped sum
cannot be split back into sources. `plot_log` draws one panel per source.

#### Controlled experiments

GAE: `compute_gae` used to read `dones[:, t + 1]` for all but the last step, one
step after the episode actually ended, so a terminal step was credited with the
value of the next episode's first state and the step before it was cut off early.
It now reads `dones[:, t]` (the done returned by the step that took action `t`).
Example: rewards `[0, 1, 0]`, values `[0, 0, 100]`, dones `[F, T, F]`, bootstrap
100: the old code gave advantages `[0, 99.06, -1]`, the fix gives `[0.9405, 1, -1]`.
Checkpoints record `gae_version: 2`; `--resume auto` never continues a run from
before the fix, and an explicit `--resume` of an old checkpoint prints a warning.
To measure what the fix changes, train `--exp base` and `--exp gae_v1` (the old
formula, kept only for this comparison) from scratch with the same seeds.

Presets in `run.py` (`EXPERIMENTS`) change only the listed settings; stack them
with commas (`--exp B,serial`). Train every group from scratch with the same frame
budget and at least 3 seeds (`--seed`); output directories are named
`runs/ppo_<n>gpu_<envs>env_<preset>_seed<seed>_<time>`.

| Preset | Changes | Compare with |
| --- | --- | --- |
| `base` | nothing (GAE fixed, no coins, adaptive shaping window, parallel, no masks) | reference for network/input groups |
| `gae_v1` | pre-fix GAE (`gae_version: 1`) | base: effect of the GAE fix |
| `A` | coins on, fixed 800 m shaping window | common reference for rewards |
| `B` | coins off, fixed 800 m window | A: effect of coins |
| `C` | coins off, fixed 200 m window | B: effect of the shaping radius |
| `serial` | `model_topology: serial` | base: effect of the goal → policy connection |
| `mask_position` / `mask_yaw` / `mask_status_bar` | hide own lat/lng, own heading, or the neighbor-direction bar | base: value of each extra input |
| `mask_all` | all three masks | base |

A, B and C set `shaping_radius_ratio: 0`; changing `max_radius_meters` alone only
caps the adaptive window.

Network: `parallel` (default, every older checkpoint) feeds the goal LSTM and the
policy LSTM side by side and concatenates their outputs for the heads; the policy
LSTM never sees the goal. `serial` follows the reference CityNav connection: the
goal LSTM output passes through a 64-unit tanh bottleneck into the policy LSTM at
every step, and the heads read only the policy LSTM (with ResNet18 the policy input
grows from 582 to 646 dims and the head input shrinks from 1024 to 512). Hidden
size, vision encoder and inputs stay the same, so this is not a full reproduction
of CityNav (256 hidden units there, no dropout here).

Inputs: `mask_yaw` zeros sin/cos of the agent's heading, `mask_position` zeros its
lat/lng, and `mask_status_bar` paints the bottom `status_height` rows of the image
(the neighbor-direction bar) black while the 160×110 projected view stays
pixel-identical. The goal lat/lng is never masked, and tensor shapes do not change.
Masking happens inside `PPOAgent`, so sampling, sequence updates, bootstrap values,
evaluation and trajectory maps all see the same inputs. The auxiliary heading
loss, if enabled, still targets the true heading.

Topology and masks are stored in each checkpoint (`model_topology`, `mask_*`,
`num_parameters`); `eval`, `traj` and `report` rebuild the network from them, and
older checkpoints load as `parallel` without masks. `--resume` refuses a checkpoint
whose network or masks differ from the current settings.

#### Leftover processes

If a previous run died without its environment workers exiting, those workers keep
the LevelDB copies in `ds_cache_dir` locked and every new worker fails with
`IO error: lock .../LOCK: Resource temporarily unavailable`. Before doing anything
else, `run_6gpu.sh` runs `python -m streetlearn.tools.stale_procs`, which lists the
leftover training processes of the current user (with their children) and any
process holding a lock in the cache, together with the output directory they
write to and when that directory was last written. If it finds any, the script
stops without launching anything or creating a run directory:

```bash
KILL_STALE=1 bash run_6gpu.sh     # 结束它们（SIGTERM，20 秒后 SIGKILL）再启动
kill $(cat runs/stale_pids.txt)   # 或者手动结束
```

The trainer itself also skips locked copies (building `_a1`, `_a2`, ... copies
instead) and names the processes holding them. If Chinese output shows up as
mojibake (for example `纭鍙互缁撴潫`), set the SSH client's character encoding to
UTF-8. Environment workers are daemon processes, so a trainer that stops on
an error (a rejected `--resume`, an exception during training) now exits and takes
its workers with it instead of hanging.

#### Evaluation after training

`python run.py report` produces training curves, the curriculum test overview, an
Agent-vs-Oracle evaluation of `ppo_final.pt`, and trajectory maps for every
`ppo_best_stage_*.pt`. Evaluation and trajectory goals default to the checkpoint's
own curriculum band (`--goal_band stage`); use `--goal_band full` for the whole map
or `--goal_band 500,800` for an explicit band in meters. Each environment runs a
fixed number of episodes, so Agent and Oracle face exactly the same routes. An
episode ends when the goal is reached; `success_steps` in `eval_results.csv` is the
mean number of steps of the successful episodes. Read it together with
`fail_rate`: a policy that only solves the easy routes also gets short means.

Alternatively, call the underlying scripts directly:

```bash
# Engine demo (raw engine: build graph, navigate, render)
python -m streetlearn.tools.run_nav --dataset_path=/path/to/manhattan_lowres --mode=engine

# Game demo (full CourierGame environment with goal-directed navigation)
python -m streetlearn.tools.run_nav --dataset_path=/path/to/manhattan_lowres --mode=game
```

## Code Structure

```
streetlearn/
├── engine/python/           # Pure Python engine (replaces C++ engine)
│   ├── streetlearn_engine.py   # Main engine: Create, Clone, BuildGraph, MoveToNextPano, RenderObservation
│   ├── pano_graph.py           # BFS graph building, LRU cache, async prefetch, navigation
│   ├── pano_projection.py      # Equirectangular → perspective projection (numpy + cv2.remap)
│   ├── pano_renderer.py        # First-person view rendering + status bar + stop signs
│   ├── graph_renderer.py       # Top-down map rendering with Bresenham lines
│   ├── metadata_cache.py       # Pano metadata cache (PanoData, PanoMetadata)
│   ├── dataset.py              # LevelDB wrapper (plyvel)
│   ├── pano_calculations.py    # Haversine distance + bearing
│   ├── color.py                # Color class
│   └── math_util.py            # Math helpers
├── proto/                   # Protocol buffers
│   ├── streetlearn.proto       # Pano, StreetLearnGraph message definitions
│   └── gen_pb2.py              # Protobuf generation script
├── python/
│   ├── environment/         # RL environment (OpenAI Gym interface)
│   │   ├── streetlearn.py      # StreetLearn env: reset(), step(), observation()
│   │   ├── game.py             # Base Game class: extended graph, BFS shortest paths
│   │   ├── courier_game.py     # Courier game: goal-directed navigation with reward shaping
│   │   ├── observations.py     # Observation factory: ViewImage, GraphImage, Metadata, etc.
│   │   ├── default_config.py   # Default configuration + game registry
│   │   └── ...                 # Other game variants
│   ├── agents/              # PyTorch PPO agent (ppo_agent.py)
│   ├── ppo_train.py         # PPO training loop (DDP, performance curriculum, resume)
│   ├── ppo_eval.py          # Evaluation (Agent greedy + Oracle rollout, curriculum tests)
│   ├── ppo_traj_viz.py      # Trajectory maps for one or many checkpoints
│   └── checkpoint_utils.py  # Locate the newest checkpoint of a run (resume)
└── tools/
    ├── run_nav.py             # Navigation demo (engine + game modes)
    ├── plot_ppo_log.py        # Training curves from train_metrics.csv
    ├── plot_curriculum.py     # Curriculum test overview from curriculum_eval.csv
    ├── plot_ppo_eval.py       # Agent/Oracle/BFS comparison figure
    └── stale_procs.py         # Find / stop leftover training processes (run_6gpu.sh precheck)
tests/                       # python run.py selftest（CPU）
├── test_gae.py              # GAE vs. a direct per-episode definition, episode boundaries
└── test_ppo_agent.py        # parallel/serial shapes and gradients, done resets, masks, save/load
```

## Engine API

The pure Python `StreetLearnEngine` provides the same 16 methods as the original CLIF binding:

| Method | Description |
|--------|-------------|
| `Create(data_dir, width, height, ...)` | Factory: open LevelDB, build PanoGraph |
| `Clone(...)` | Share dataset+cache for multi-env (IMPALA) |
| `InitEpisode(episode_id, seed)` | Initialize RNG |
| `BuildRandomGraph()` / `BuildGraphWithRoot(id)` / `BuildEntireGraph()` | BFS graph construction |
| `MoveToNextPano()` | Navigate to neighbor by bearing (±30° tolerance) |
| `RotateObserver(yaw, pitch)` | Update agent orientation |
| `RenderObservation(buffer)` | Render first-person view (HWC→CHW planar) |
| `DrawGraph(colors, buffer)` | Render top-down map |
| `SetZoom(zoom)` | Map zoom level (1-32) |
| `GetGraph()` | Returns `{pano_id: [neighbor_ids]}` |
| `GetNeighborOccupancy(resolution)` | Binary traversability vector |
| `GetMetadata(pano_id)` | Pano metadata (GPS, altitude, heading, neighbors) |
| `GetPanoDistance(id1, id2)` | Haversine distance in meters |
| `GetPanoBearing(id1, id2)` | Bearing in degrees |
| `GetYaw()` / `GetPitch()` | Current agent orientation |
| `GetNodeCacheSize()` | LRU cache size |

## Environment Usage

```python
from streetlearn.python.environment import default_config
from streetlearn.python.environment import courier_game
from streetlearn.python.environment import streetlearn as sl_env

config = {
    "dataset_path": "/path/to/manhattan_lowres",
    "width": 320, "height": 240,
    "full_graph": False,
    "min_graph_depth": 50, "max_graph_depth": 100,
    "observations": ["view_image", "graph_image", "yaw", "neighbors"],
}
game = courier_game.CourierGame(config)
env = sl_env.StreetLearn(config["dataset_path"], config, game)

obs = env.reset()
for step in range(100):
    action = env.action_set()[0]  # or your RL policy
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
```

### Actions

| Action | Description |
|--------|-------------|
| Move forward | Move to neighbor within ±30° bearing |
| Rotate left/right | Yaw rotation by `rotation_speed` degrees |
| Rotate up/down | Pitch rotation |
| Zoom in/out | Map zoom |

### Observations

`view_image` (3,H,W), `graph_image` (3,H,W), `view_image_hwc` (H,W,3), `graph_image_hwc`, `yaw`, `pitch`, `metadata` (Pano proto), `target_metadata`, `latlng`, `target_latlng`, `latlng_label`, `target_latlng_label`, `yaw_label`, `neighbors` (binary occupancy), `thumbnails`, `instructions`, `ground_truth_direction`, `prev_reward`, `prev_action`.

### Games

| Game | Description |
|------|-------------|
| `courier_game` | Goal-directed navigation with reward shaping |
| `coin_game` | Collect invisible coins scattered on the map |
| `exploration_game` | Free exploration |
| `curriculum_courier_game` | Courier with curriculum on goal distance |
| `goal_instruction_game` | Navigate following text instructions |
| `incremental_instruction_game` | Instructions revealed incrementally |
| `step_by_step_instruction_game` | One instruction at a time |

## Dependencies

| Package | Purpose | Required? |
|---------|---------|-----------|
| plyvel | LevelDB dataset access | Yes |
| numpy | Projection, rendering, arrays | Yes |
| Pillow | JPEG decoding | Yes |
| protobuf | Protocol buffer runtime | Yes |
| opencv-python | Fast cv2.remap projection | Recommended |
| absl-py | Logging | Yes |
| six | Py2/3 compat | Yes |
| torch / torchvision | PPO training & evaluation | Training |
| matplotlib | Training/eval curves | Plotting |

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Disclaimer

This is not an official Google product. The pure Python engine is a refactor of the original C++ implementation.
