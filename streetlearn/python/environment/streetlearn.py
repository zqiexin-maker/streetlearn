# Copyright 2018 Google LLC.
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
# See the License for the specific language permissions and
# limitations under the License.

"""StreetLearn 的 RL 环境（连接引擎与游戏的"门面"）。

============================================================================
这一层做什么？
============================================================================
- 把底层纯 Python 引擎（streetlearn_engine.py）和具体任务（game.py）组合成
  一个 OpenAI-Gym 风格的接口：reset() / step() / observation() / reward() /
  done() / info()。
- 管理 episode 生命周期：建图、选起点、episode 边界、frame 计数。
- 把离散动作索引翻译成引擎能执行的连续动作向量（见 step 里的点积分解）。
- 它本身不含任务逻辑（奖励/成功判定在 game 里），只负责"调用游戏"。

观测字典（observation()）由 observations.py 中注册的各观测类惰性产生，键名
与 config["observations"] 对应；PPO 训练使用前 6 个。
============================================================================
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import inflection
import numpy as np
import six
import time

from streetlearn.engine.python import color
from streetlearn.engine.python import streetlearn_engine
from streetlearn.python.environment import default_config
from streetlearn.python.environment import observations

_MIN_ZOOM = 1
_MAX_ZOOM = 32


def _action(*entries):
  # 把一个四元动作向量（前进, 水平旋转, 俯仰, 缩放）打包成 float64 数组。
  return np.array(entries, dtype=np.float64)


# 四种基础动作分量（基向量）。每个"动作"是这些分量的线性组合。
ACTIONS = {
    'move_forward': _action(1, 0.0, 0.0, 0.0),
    'horizontal_rotation': _action(0, 1.0, 0.0, 0.0),
    'vertical_rotation': _action(0, 0.0, 1.0, 0.0),
    'map_zoom': _action(0, 0.0, 0.0, 1.0),
}

NUM_ACTIONS = 4  # 动作向量的维数（与上面四个基向量对应）

# 动作集：把一个动作集合名映射成一组离散动作（每个动作是一个四元向量）。
# 例如 "streetlearn_fast_rotate" 给出 5 个动作：前进、左转、左大转、右转、右大转。
# 这正是 PPO 的 num_actions=5 的来源（见 ppo_train.build_config）。
ACTION_SETS = {
    "streetlearn_default": lambda rotation_speed: (
        ACTIONS["move_forward"],
        ACTIONS["horizontal_rotation"] * (-rotation_speed),
        ACTIONS["horizontal_rotation"] * rotation_speed),
    "streetlearn_fast_rotate": lambda rotation_speed: (
        ACTIONS["move_forward"],
        ACTIONS["horizontal_rotation"] * (-rotation_speed),
        ACTIONS["horizontal_rotation"] * (-rotation_speed * 3),
        ACTIONS["horizontal_rotation"] * rotation_speed,
        ACTIONS["horizontal_rotation"] * rotation_speed * 3),
    "streetlearn_tilt": lambda rotation_speed: (
        ACTIONS["move_forward"],
        ACTIONS["horizontal_rotation"] * (-rotation_speed),
        ACTIONS["horizontal_rotation"] * rotation_speed,
        ACTIONS["vertical_rotation"] * rotation_speed,
        ACTIONS["vertical_rotation"] * (-rotation_speed)),
}


def get_action_set(action_spec, rotation_speed):
  """返回给定动作集名对应的离散动作矩阵（num_actions × 4）。"""
  if isinstance(action_spec, six.string_types):
    if action_spec not in ACTION_SETS:
      raise ValueError("Unrecognized action specification %s." % action_spec)
    else:
      return np.array(ACTION_SETS[action_spec](rotation_speed), dtype=np.float64)
  raise ValueError("Action specification %s not a string." % action_spec)


def _log_dictionary(dictionary):
  # 把 config 里的简单值打印到日志（调试用）。
  for k, v in dictionary.items():
    v = dictionary[k]
    if isinstance(v, (int, float, bool, list, str)):
      logging.info(k + ': ' + str(v))


class StreetLearn(object):
  """StreetLearn 环境主体。"""

  def __init__(self, dataset_path, config, game, engine=None):
    """构造环境。

    Args:
      dataset_path: 数据集目录（LevelDB）。
      config: 配置字典，会与 default_config 的默认值合并。
      game: 一个 Game 实例（决定任务与奖励）。
      engine: 已存在的引擎实例（用于 Clone，多环境共享缓存）。
    """
    assert game, "Did not provide game."
    logging.info('dataset_path:')
    logging.info(dataset_path)
    logging.info('config:')
    _log_dictionary(config)
    logging.info('game:')
    logging.info(game)
    self._config = default_config.ApplyDefaults(config)     # 合入默认值，形成完整config字典
    self._seed = self._config["seed"]                       # 随机种子（用于BFS起点选择）
    self._start_pano_id = self._config["start_pano"]        # 可选的固定起点pano_id
    self._zoom = self._config["graph_zoom"]                 # 地图缩放级别（1~32）

    self._black_on_white = self._config["graph_black_on_white"]

    self._frame_cap = self._config["frame_cap"]             # 最大帧数（episode长度上限）
    self._field_of_view = self._config["field_of_view"]     # 第一人称视角FOV（度）
    self._neighbor_resolution = self._config["neighbor_resolution"]  # 邻居方向分箱数
    self._sample_graph_depth = self._config["sample_graph_depth"]    # 是否随机采样图深度 目的：课程学习 动态改变图的大小
    self._min_graph_depth = self._config["min_graph_depth"]          # BFS最小深度
    self._max_graph_depth = self._config["max_graph_depth"]          # BFS最大深度
    self._full_graph = self._config["full_graph"]                    # 是否用整个连通图
    self._color_for_observer = color.Color(*self._config["color_for_observer"])  # 智能体在地图上的颜色

    self._action_spec = self._config["action_spec"]      # 动作集名称（字符串）
    self._rotation_speed = self._config["rotation_speed"]  # 每次旋转的角度（默认22.5°）
    self._auto_reset = self._config["auto_reset"]
    self._action_set = get_action_set(self._action_spec, self._rotation_speed)

    logging.info('Action set:')
    logging.info(self._action_set)
    self._bbox_lat_min = self._config["bbox_lat_min"]
    self._bbox_lat_max = self._config["bbox_lat_max"]
    self._bbox_lng_min = self._config["bbox_lng_min"]
    self._bbox_lng_max = self._config["bbox_lng_max"]

    self._game = game
    self._current_pano_id = None
    self._episode_id = -1
    self._frame_count = 0
    self._prev_reset = time.time()

    if engine: # Clone模式：多环境共享缓存与图结构
      logging.info("Cloning an existing StreetLearnEngine.")
      self._engine = engine.Clone(
          width=self._config["width"],
          height=self._config["height"],
          graph_width=self._config["graph_width"],
          graph_height=self._config["graph_height"],
          status_height=self._config["status_height"],
          field_of_view=self._field_of_view,
          min_graph_depth=self._min_graph_depth,
          max_graph_depth=self._max_graph_depth)
    else:      # 新建模式：首次创建引擎（打开数据集 + 建 PanoGraph）
      logging.info("Creating an new StreetLearnEngine.")
      self._engine = streetlearn_engine.StreetLearnEngine.Create(  # → engine/python/streetlearn_engine.py Create()
          dataset_path,                                  # LevelDB 数据路径
          width=self._config["width"],                   # 第一人称视图分辨率
          height=self._config["height"],
          graph_width=self._config["graph_width"],       # 地图视图分辨率
          graph_height=self._config["graph_height"],
          status_height=self._config["status_height"],   # 底部状态栏高度（导航指示点）
          field_of_view=self._field_of_view,             # FOV
          min_graph_depth=self._min_graph_depth,         # BFS深度范围
          max_graph_depth=self._max_graph_depth,
          max_cache_size=self._config["max_cache_size"]) # LRU缓存容量
    assert self._engine, "Could not initialise engine from %r." % dataset_path

    # 根据 config["observations"] 实例化各观测类（惰性，访问时才取数）。
    self._observations = []  # 跳转至 environment/observations.py
    for name in self._config["observations"]:
      try:
        self._observations.append(observations.Observation.create(name, self))
      except ValueError as e:
        logging.warning(str(e))

    self._reward = 0
    self._prev_reward = 0
    self._prev_action = self._action_set[0]
    self._done = False
    self._info = {}

  @property
  def config(self):
    return self._config

  @property
  def seed(self):
    return self._seed

  @property
  def game(self):
    return self._game

  @property
  def field_of_view(self):
    return self._field_of_view

  @property
  def current_pano_id(self):
    return self._current_pano_id

  @property
  def frame_cap(self):
    return self._frame_cap

  @frame_cap.setter
  def frame_cap(self, value):
    self._frame_cap = value

  @property
  def frame_count(self):
    return self._frame_count

  @property
  def graph(self):
    return self._graph

  @property
  def engine(self):
    return self._engine

  @property
  def neighbor_resolution(self):
    return self._neighbor_resolution

  @property
  def bbox_lat_min(self):
    return self._bbox_lat_min

  @property
  def bbox_lat_max(self):
    return self._bbox_lat_max

  @property
  def bbox_lng_min(self):
    return self._bbox_lng_min

  @property
  def bbox_lng_max(self):
    return self._bbox_lng_max

  @property
  def cache_size(self):
    return self._engine.GetNodeCacheSize()

  def observation_spec(self):
    """返回观测规格字典（观测名 -> 形状），取决于启用的观测格式。"""
    return {observation.name: observation.observation_spec
            for observation in self._observations}

  def action_set(self):
    """返回动作集（把整数动作索引映射到四元动作向量）。"""
    return self._action_set

  def action_spec(self):
    """返回基础动作分量定义。"""
    return ACTIONS

  def reset(self):
    """开始一个新 episode。"""
    reset_time = time.time()                    # 记录时间（用于日志计时）
    logging.info('reset: seed %d, previous episode (%d frames) lasted %f sec',
                 self._seed, self._frame_count, reset_time - self._prev_reset)
    self._prev_reset = reset_time
    self._prev_reward = 0                       # 上一步奖励归零
    self._prev_action = self._action_set[0]     # 上一步动作设为"前进"
    self._frame_count = 0                       # 帧计数归零
    self._episode_id += 1                       # Episode ID递增
    if self._sample_graph_depth:                # 课程学习的核心 动态图深度
      max_depth = np.random.randint(self._min_graph_depth,
                                    self._max_graph_depth + 1)
      self._engine.SetGraphDepth(self._min_graph_depth, max_depth)

    self._engine.InitEpisode(self._episode_id, self._seed)  # 种子初始化（RNG）

    # 若还没有图，则建一个新图；否则在已有图内随机重生。
    if not self._current_pano_id:                                 # 首次reset：需要建新图
      if self._full_graph:                                        # 整个连通图
        self._current_pano_id = self._engine.BuildEntireGraph()
      elif self._start_pano_id:                                   # 从指定根节点BFS
        self._current_pano_id = self._engine.BuildGraphWithRoot(
            self._start_pano_id)
      else:
        self._current_pano_id = self._engine.BuildRandomGraph()   # 随机根节点BFS → pano_graph.cc
      logging.info('seed %d: built new graph with root %s',
                   self._seed, self._current_pano_id)
    # else respawn in current graph.
    elif not self._start_pano_id:                                 # 已有图：在图内随机重生
      self._current_pano_id = np.random.choice(
          list(self._engine.GetGraph().keys()))
      self._engine.SetPosition(self._current_pano_id)
      logging.info('seed %d: reusing existing graph and respawning at %s',
                   self._seed, self._current_pano_id)

    self._graph = self._engine.GetGraph()                         # 获取图结构（pano_id → [邻居ID列表]）
    highlighted_panos = self._game.on_reset(self)                 # ★ 游戏逻辑初始化（选目标、算最短路）
    self._engine.InitGraphRenderer(self._color_for_observer, highlighted_panos,   # 地图渲染器初始化
                                   self._black_on_white)
    self._engine.SetZoom(_MAX_ZOOM)                               # 初始缩放=最大（全局视图）

  def goto(self, pano_id, yaw):
    """跳转到指定全景与朝向（调试/评估用）。"""
    current_pano_id = self._engine.SetPosition(pano_id)
    assert pano_id == current_pano_id
    yaw = (yaw + 180) % 360 - 180
    self._engine.RotateObserver(yaw, 0.0)
    assert yaw == self._engine.GetYaw()
    return self.observation()

  def step(self, action):
    """在环境中执行一步。

    Args:
      action: 一个 1-D 数组（四元动作向量），由离散动作索引经 action_set 查表得到。
    Returns:
      (observation, reward, done, info)
    """
    self._frame_count += 1
    if type(action) != np.ndarray:
      action = np.array(action, dtype=np.float64)
    self._prev_action = action
    assert action.size == NUM_ACTIONS, "Wrong number of actions."

    # ★ 动作向量分解为4个分量（通过点积）。
    # 为什么用点积而不是索引：点积使得同一分量可以被不同动作共享
    #（如"左转"和"左大转"都用了 horizontal_rotation 分量，只是系数不同），
    # 这样能紧凑地表达"旋转不同幅度"等连续语义。
    move_forward = np.dot(action, ACTIONS['move_forward'])                # [1,0,0,0]·action → 是否前进
    horizontal_rotation = np.dot(action, ACTIONS['horizontal_rotation'])  # [0,1,0,0]·action → 旋转角度
    vertical_rotation = np.dot(action, ACTIONS['vertical_rotation'])      # [0,0,1,0]·action → 俯仰角度
    map_zoom = np.dot(action, ACTIONS['map_zoom'])                        # [0,0,0,1]·action → 缩放方向

    if move_forward:
      new_pano_id = self._engine.MoveToNextPano()  # → engine/python/streetlearn_engine.py MoveToNextPano
      # 若移动被阻挡（朝向 tolerance 内无邻居），MoveToNextPano 返回 None，
      # 此时应保持原位，不能把 current_pano_id 置空，否则后续 GetPanoDistance(None, ...) 会崩溃。
      if new_pano_id:
        self._current_pano_id = new_pano_id
    if map_zoom > 0:
      self._zoom = min(self._zoom * 2, _MAX_ZOOM)
    elif map_zoom < 0:
      self._zoom = max(self._zoom / 2, _MIN_ZOOM)

    if horizontal_rotation or vertical_rotation:            # 旋转逻辑
      self._engine.RotateObserver(horizontal_rotation, vertical_rotation) # → engine/python/streetlearn_engine.py RotateObserver

    self._engine.SetZoom(self._zoom)
    self._game.on_step(self)                                # 调用具体游戏的 on_step

    # 更新奖励与 done 标志。因为不知道每个游戏内部逻辑，最安全的做法是在
    # step() 之后立刻取这些值并缓存，供后续 reward()/done()/info() 调用。
    self._prev_reward = self._reward
    self._reward = self._game.get_reward(self)               # ★ 核心：调用游戏的奖励函数 → CourierGame 的 get_reward（courier_game.py）
    self._done = (self._frame_count > self._frame_cap) or self._game.done()
    self._info = self._game.get_info(self)
    if self._auto_reset and self._done:
      self.reset()

    # 返回
    return self.observation(), self.reward(), self.done(), self.info()  # 观测收集 跳转observation()方法

  def observation(self):
    # 渲染流程：LevelDB读JPEG → NodeCache解码为Image3_b
    # → PanoProjection（等矩形→透视投影，加上yaw/pitch旋转）
    # → PanoRenderer（Cairo绘制导航指示点+停止标志）
    # → ConvertRGBPackedToPlanar
    # → CHW格式输出给Python。
    """返回上一步的观测字典。"""
    return {item.name: item.observation for item in self._observations} # 每个观测器的 .observation 属性在被访问时才触发数据获取。跳转observation.py ViewImage

  def reward(self):
    """返回上一步的奖励。"""
    return self._reward

  def done(self):
    """返回 episode 是否结束的标志。"""
    return self._done

  def info(self):
    """返回本步的环境调试信息字典。"""
    return self._info

  def prev_reward(self):
    """返回上一步的奖励。"""
    return self._prev_reward

  def prev_action(self):
    """返回上一步的动作。"""
    return self._prev_action

  def get_metadata(self, pano_id):
    """返回指定全景的元数据（protobuf）。"""
    if hasattr(self, '_graph') and pano_id in self.graph:
      return self._engine.GetMetadata(pano_id)
    else:
      return None

  def render(self):
    """空函数，为了兼容 OpenAI Gym 接口。"""
    pass
