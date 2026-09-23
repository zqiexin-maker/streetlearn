"""StreetLearn 引擎 —— 纯 Python 替换原 C++ streetlearn_engine.cc + CLIF。

============================================================================
这一层是"去 C++ 依赖"的全部工作量所在。
============================================================================
原版 StreetLearn 的引擎是 C++ 实现（streetlearn_engine.cc），通过 CLIF 暴露
给 Python。本文件用纯 Python 重新实现了同样的 API：打开 LevelDB 数据集、
构建全景图（PanoGraph）、渲染街景图与地图图、提供导航与查询接口。

它依赖同目录下的子模块：
  dataset.py        LevelDB 数据集读取
  pano_graph.py      全景节点图（BFS 建图、邻居、距离）
  pano_renderer.py   街景图渲染（透视投影 + Cairo 画指示点）
  graph_renderer.py  地图图渲染
  math_util.py / color.py  数学与颜色工具

关键约定（与 obs/agent 对接）：
  - RenderObservation 把当前视图渲染进一个 planar buffer（长度 3*H*W），
    由 observations.ViewImage 负责 reshape 成 [3,H,W]。
  - DrawGraph 把地图渲染进 buffer，并返回 bool 成功标志（不是图像），
    由 observations.GraphImage 负责 reshape。
  - 旋转/移动等导航操作都在这里转发给 PanoGraph。
============================================================================
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np
import random

from streetlearn.engine.python import math_util
from streetlearn.engine.python.color import Color
from streetlearn.engine.python.dataset import Dataset
from streetlearn.engine.python.graph_renderer import GraphRenderer, Observer
from streetlearn.engine.python.metadata_cache import PanoMetadata
from streetlearn.engine.python.pano_graph import PanoGraph
from streetlearn.engine.python.pano_renderer import PanoRenderer

# 与 C++ streetlearn_engine.cc 对齐的常量
_DEFAULT_PREFETCH_DEPTH = 20   # 建图时预取邻居的层数
_DEFAULT_MAX_NEIGHBOR_DEPTH = 3  # 取邻居方位角时的最大深度
_SHOW_STOP_SIGNS = True        # 是否在地图上画停止标志
_TOLERANCE = 30.0              # 前进时朝向与邻居方位的最大容差（度）
_MAX_YAW = 180.0               # yaw 角度约束范围
_MAX_PITCH = 90.0              # pitch 角度约束范围


def _constrain_angle(constraint, angle):
  """把角度包裹到 [-constraint, constraint]。"""
  while angle < -constraint:
    angle += 2.0 * constraint
  while angle > constraint:
    angle -= 2.0 * constraint
  return angle


class StreetLearnEngine(object):
  """纯 Python 版 StreetLearn 引擎，API 与原 C++ CLIF 版一致。"""

  def __init__(self, dataset, node_cache_or_graph, pano_size, graph_size,
               status_height, field_of_view, min_graph_depth, max_graph_depth):
    """直接构造（对应 C++ 构造函数）。

    Args:
      dataset: Dataset 实例（LevelDB 句柄）。
      node_cache_or_graph: PanoGraph 实例（Clone 时共享，否则新建）。
      pano_size: (width, height) 街景图尺寸。
      graph_size: (graph_width, graph_height) 地图图尺寸。
      status_height: 底部状态栏高度（像素）。
      field_of_view: 视场角（度）。
      min_graph_depth / max_graph_depth: BFS 建图深度范围。
    """
    self._dataset = dataset
    self._pano_size = pano_size
    self._graph_size = graph_size
    self._status_height = status_height
    self._field_of_view = field_of_view
    self._min_graph_depth = min_graph_depth
    self._max_graph_depth = max_graph_depth
    self._show_stop_signs = _SHOW_STOP_SIGNS
    self._rotation_yaw = 0.0    # 观察者相对北向的偏航角（累计）
    self._rotation_pitch = 0.0  # 俯仰角

    self._pano_graph = node_cache_or_graph
    self._pano_renderer = PanoRenderer(
        pano_size[0], pano_size[1], status_height, field_of_view)
    self._graph_renderer = None
    self._observer = Observer()
    # 预分配 planar buffer（长度 3*W*H），RenderObservation/DrawGraph 写入它。
    self._pano_buffer = np.empty(
        3 * pano_size[0] * pano_size[1], dtype=np.uint8)
    self._graph_buffer = np.empty(
        3 * graph_size[0] * graph_size[1], dtype=np.uint8)

  @classmethod
  def Create(cls, data_dir, width=320, height=240, graph_width=320,
             graph_height=240, status_height=10, field_of_view=60,
             min_graph_depth=10, max_graph_depth=15, max_cache_size=1000):
    """工厂方法：打开数据集、创建 PanoGraph、再构造引擎。"""
    dataset = Dataset.create(data_dir)
    if dataset is None:
      logging.error("Failed to create dataset from %s", data_dir)
      return None

    pano_graph = PanoGraph(
        _DEFAULT_PREFETCH_DEPTH, min_graph_depth, max_graph_depth,
        dataset, max_cache_size)
    if not pano_graph.init():
      logging.error("Failed to initialize pano graph")
      return None

    return cls(
        dataset, pano_graph, (width, height), (graph_width, graph_height),
        status_height, field_of_view, min_graph_depth, max_graph_depth)

  def Clone(self, width=320, height=240, graph_width=320, graph_height=240,
            status_height=10, field_of_view=60, min_graph_depth=10,
            max_graph_depth=15):
    """创建一个共享同一数据集与图缓存的克隆（多环境并行时复用，省内存）。"""
    return StreetLearnEngine(
        self._dataset, self._pano_graph, (width, height),
        (graph_width, graph_height), status_height, field_of_view,
        min_graph_depth, max_graph_depth)

  def InitEpisode(self, episode_index, random_seed):
    """为本 episode 初始化随机数发生器（决定 BFS 起点等）。"""
    # 把 episode_index 与 random_seed 组合成种子（匹配 C++ 的 hash 行为）。
    seed = hash((episode_index, random_seed)) & 0xFFFFFFFF
    self._pano_graph.set_random_seed(seed)

  # ---- 建图 ----

  def BuildRandomGraph(self):
    # 从随机根节点做 BFS 建图。
    if not self._pano_graph.build_random_graph():
      return None
    return self._setup_current_graph()

  def BuildGraphWithRoot(self, pano_id):
    # 从指定根节点做 BFS 建图。
    if not self._pano_graph.build_graph_with_root(pano_id):
      return None
    return self._setup_current_graph()

  def BuildEntireGraph(self):
    # 构建整个连通图（full_graph 模式）。
    if not self._pano_graph.build_entire_graph():
      return None
    return self._setup_current_graph()

  def _setup_current_graph(self):
    # 建图完成后重置旋转角，并把当前位置设为图根节点。
    self._rotation_yaw = 0.0
    self._rotation_pitch = 0.0
    return self._pano_graph.root.pano_id

  def SetGraphDepth(self, min_depth, max_depth):
    # 课程学习用：动态调整 BFS 深度范围（改变图大小）。
    self._pano_graph.set_graph_depth(min_depth, max_depth)

  def SetPosition(self, pano_id):
    """把当前位置设为 pano_id（必须在当前图内）。"""
    if pano_id not in self._pano_graph._node_tree:
      logging.error("Pano %s is not in the current graph", pano_id)
      return None
    self._pano_graph._prefetch_graph(pano_id)  # 预取该节点周边数据
    return self._setup_current_graph()

  # ---- 导航 ----

  def MoveToNextPano(self):
    # 沿当前 yaw 方向前进到最近的可行走邻居；若容差内无邻居则失败返回 None。
    if not self._pano_graph.move_to_neighbor(self._rotation_yaw, _TOLERANCE):
      return None
    return self._pano_graph.root.pano_id

  def RotateObserver(self, yaw_deg, pitch_deg):
    # 更新观察者朝向；角度约束到 [-_MAX_YAW, _MAX_YAW] / [-_MAX_PITCH, _MAX_PITCH]。
    # 注意 pitch 用减号：引擎内部约定与 API 符号相反。
    self._rotation_yaw = _constrain_angle(
        _MAX_YAW, self._rotation_yaw + yaw_deg)
    self._rotation_pitch = _constrain_angle(
        _MAX_PITCH, self._rotation_pitch - pitch_deg)

  # ---- 渲染 ----

  def RenderObservation(self, buffer=None):
    """渲染当前视图并写入 buffer（匹配 C++ CLIF API）。

    C++ CLIF 把像素就地写入传入的 numpy buffer。这里复现该行为：
    若提供 buffer，则写入并返回它；否则返回内部 buffer。
    """
    self._render_scene()
    pixels = self._pano_renderer.pixels  # (H, W, 3) HWC 打包

    # HWC 打包 -> CHW planar：把 R/G/B 三个通道分别连续存放。
    h, w = pixels.shape[:2]
    self._pano_buffer[:h * w] = pixels[:, :, 0].ravel()
    self._pano_buffer[h * w:2 * h * w] = pixels[:, :, 1].ravel()
    self._pano_buffer[2 * h * w:] = pixels[:, :, 2].ravel()

    if buffer is not None:
      np.copyto(buffer, self._pano_buffer)
      return buffer
    return self._pano_buffer

  def _render_scene(self):
    # 准备渲染所需的邻居方位、停止标志、当前全景图、全局 yaw，交给 PanoRenderer。
    neighbor_bearings = self._pano_graph.get_neighbor_bearings(
        _DEFAULT_MAX_NEIGHBOR_DEPTH)
    terminal_bearings = {}
    if self._show_stop_signs:
      terminal_bearings = self._pano_graph.terminal_bearings(
          self._pano_graph.root.pano_id)

    root_img = self._pano_graph.root_image()
    global_yaw = self._pano_graph.root.bearing if self._pano_graph.root else 0.0

    self._pano_renderer.render_scene(
        root_img, global_yaw, self._rotation_yaw, self._rotation_pitch,
        _TOLERANCE, neighbor_bearings, terminal_bearings)

  def InitGraphRenderer(self, observer_color, panos_to_highlight,
                        black_on_white):
    # 初始化地图渲染器（记录观察者颜色、需要高亮的目标全景等）。
    self._observer.color = observer_color
    self._graph_renderer = GraphRenderer.create(
        self._pano_graph, self._graph_size, panos_to_highlight, black_on_white)
    return self._graph_renderer is not None

  def DrawGraph(self, pano_id_to_color, buffer):
    # 渲染俯视地图图，写入 buffer；返回 bool 成功标志（不是图像！）。
    # 注意：返回值是成功标志，调用方（observations.GraphImage）必须 reshape buffer。
    if not self._graph_renderer:
      return False
    self._observer.pano_id = self._pano_graph.root.pano_id
    self._observer.yaw_radians = math_util.degrees_to_radians(self._rotation_yaw)
    self._observer.fov_yaw_radians = math_util.degrees_to_radians(
        self._field_of_view)

    if not self._graph_renderer.render_scene(pano_id_to_color, self._observer):
      return False

    self._graph_renderer.get_pixels(buffer)
    return True

  def SetZoom(self, zoom):
    # 设置地图缩放级别（1~32）。无渲染器时返回 False。
    if not self._graph_renderer:
      return False
    return self._graph_renderer.set_zoom(zoom)

  # ---- 查询 ----

  def GetGraph(self):
    # 返回图结构：pano_id -> 邻居 id 列表。
    return self._pano_graph.get_graph()

  def GetNeighborOccupancy(self, resolution):
    # 计算各方向的邻居占用（对应 obs 里的 neighbors 向量）。
    # 把邻居方位角按当前 yaw 重新居中后，按 resolution 个 bin 分箱，
    # 该 bin 置 1 表示此方向可走。
    bearings = self._pano_graph.get_neighbor_bearings(
        _DEFAULT_MAX_NEIGHBOR_DEPTH)
    occupancy = [0] * resolution
    double_offset = 360.0 / resolution / 2.0
    for nb in bearings:
      recentered = nb.bearing - self._rotation_yaw
      offset = recentered + double_offset
      offset -= 360.0 * math.floor(offset / 360.0)
      bin_idx = int(offset / 360.0 * resolution)
      bin_idx = min(max(bin_idx, 0), resolution - 1)  # 截断到合法范围
      occupancy[bin_idx] = 1
    return occupancy

  def GetPano(self):
    """返回当前全景的轻量对象（带 .id）。"""
    meta = self._pano_graph.get_metadata(self._pano_graph.root.pano_id)
    if meta and meta.pano:
      return meta.pano
    return None

  def GetMetadata(self, pano_id):
    # 取指定全景的元数据（protobuf）。
    return self._pano_graph.get_metadata(pano_id)

  def GetPanoDistance(self, pano_id1, pano_id2):
    # 两全景在图中的图距离（步数），用于奖励/成功判定。
    return self._pano_graph.get_pano_distance(pano_id1, pano_id2)

  def GetPanoBearing(self, pano_id1, pano_id2):
    # 从 pano1 指向 pano2 的方位角。
    return self._pano_graph.get_pano_bearing(pano_id1, pano_id2)

  def GetYaw(self):
    # 返回观察者当前 yaw（度）。
    return self._rotation_yaw

  def GetPitch(self):
    # 返回观察者当前 pitch（度）。
    return -self._rotation_pitch

  def GetNodeCacheSize(self):
    # 返回节点缓存大小（调试用）。
    return self._pano_graph.get_node_cache_size()
