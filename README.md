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

All parameters live in the config sections of [`run.py`](run.py) — edit that file, then:

```bash
python run.py train         # PPO 训练（Windows: run.bat train, Linux: ./run.sh train）
python run.py eval          # Agent 贪心评估
python run.py eval_oracle   # Oracle 实跑评估（论文上界）
python run.py eval_all      # Agent + Oracle 两遍（同种子 head-to-head）
python run.py plot_log      # 训练曲线（train_metrics.csv → ppo_curves.png）
python run.py plot_eval     # 评估对比图（eval_results.csv → ppo_eval_triple.png）
python run.py demo          # 引擎+游戏快速自检（无需 checkpoint）
```

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
│   ├── ppo_train.py         # PPO training loop
│   └── ppo_eval.py          # Evaluation (Agent greedy + Oracle rollout)
└── tools/
    ├── run_nav.py             # Navigation demo (engine + game modes)
    ├── plot_ppo_log.py        # Training curves from train_metrics.csv
    └── plot_ppo_eval.py       # Agent/Oracle/BFS comparison figure
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
| inflection | String utilities | Yes |
| torch / torchvision | PPO training & evaluation | Training |
| matplotlib | Training/eval curves | Plotting |

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Disclaimer

This is not an official Google product. The pure Python engine is a refactor of the original C++ implementation.
