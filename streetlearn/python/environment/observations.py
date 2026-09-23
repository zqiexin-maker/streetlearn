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

"""StreetLearn 各类观测（observation）的定义。

============================================================================
观测是什么？
============================================================================
环境每一步返回一个 obs 字典，键是观测名（如 "view_image"），值是对应
Observation 子类的 .observation 属性。StreetLearn 支持很多种观测（图像、
朝向、经纬度、邻居连通性、指令文本……），本文件用"一个类对应一种观测"的
方式组织，并由 Observation.create 工厂按名字实例化。

PPO 训练只用其中 6 种（见 ppo_train.build_config 的 observations 列表）：
  view_image / graph_image / yaw / latlng / target_latlng / neighbors
其余类（Metadata、Instructions、Thumbnails 等）是给别的任务/原版 IMPALA 用的，
保留以兼容。

关键契约（早期 bug 根源）：引擎底层返回的是"扁平 planar buffer"或"bool 成功
标志"，每个观测类必须把原始返回值 reshape / 转换回调用方期望的形状，否则
下游 obs_to_batch 收到的就是错误维度（见下方 ViewImage / GraphImage 注释）。
============================================================================
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six

_METADATA_COUNT = 14
_NUM_HEADING_BINS = 16
_NUM_LAT_BINS = 32
_NUM_LNG_BINS = 32


def reshape_hwc(array, c, w, h):
  """把 planar（按通道平面存储）的 RGB 数组转成交错（HWC）数组。

  Args:
    array: 形如 (c, w*h) 的平面 RGB 字节数组。
    c, w, h: 通道数、宽、高。
  Returns:
    (h, w, c) 的交错数组（每个像素的 RGB 连续存放）。
  用于 *_HWC 变体的观测（通道在最后的布局，符合普通图像处理库习惯）。
  """
  arr = array.reshape(c, w * h)
  return np.ravel((arr[0], arr[1], arr[2]),
                  order='F').reshape(h, w, 3)


class Observation(six.with_metaclass(abc.ABCMeta, object)):
  """所有观测的基类。子类必须实现 observation_spec 与 observation。"""

  def __init__(self, streetlearn):
    self._streetlearn = streetlearn  # 反向持有环境引用，便于取引擎数据

  @abc.abstractproperty
  def observation_spec(self):
    """该观测的形状说明（如 [3, H, W] 或 [2]）。"""
    pass

  @abc.abstractproperty
  def observation(self):
    """观测数据本体。每次访问才真正触发引擎取数（惰性求值）。"""
    pass

  @classmethod
  def create(cls, name, streetlearn):   # 工厂模式：按名字分发对应观测类
    """根据观测名 name 创建对应的 Observation 实例。"""
    observations = [ViewImage, GraphImage, ViewImageHWC, GraphImageHWC, Yaw,
                    Pitch, Metadata, TargetMetadata, LatLng, TargetLatLng,
                    YawLabel, Neighbors, LatLngLabel, TargetLatLngLabel,
                    Thumbnails, Instructions, GroundTruthDirection, PrevReward,
                    PrevAction]
    dispatch = {o.name: o for o in observations}
    try:
      return dispatch[name](streetlearn)
    except KeyError:
      raise ValueError('No Observation named %s found' % name)


class ViewImage(Observation):
  """第一人称街景图的 RGB 像素，形状 (C, H, W)，C=3。"""
  name = 'view_image'
  observation_spec_dtypes = np.uint8

  def __init__(self, streetlearn):
    super(ViewImage, self).__init__(streetlearn)
    self._width = streetlearn.config["width"]
    self._height = streetlearn.config["height"]
    self._depth = 3  # RGB
    # 预分配一个扁平 buffer，引擎 RenderObservation 会写入它（planar 布局）。
    self._buffer = np.empty(self._depth * self._width * self._height,
                            dtype=np.uint8)

  @property
  def observation_spec(self):
    return [self._depth, self._height, self._width]

  @property
  def observation(self):
    # 契约修复点①：RenderObservation 把 planar RGB 像素写入 self._buffer
    # （一个长度 depth*width*height 的 1-D 数组），并直接返回这个 buffer，
    # 它并不是 [3, H, W] 数组。必须在这里 reshape，否则所有消费方
    # （PPO 的 obs_to_batch、原版 IMPALA 的 experiment.py）拿到的是扁平向量
    # 而非预期的 [3, H, W] 图像。
    self._streetlearn.engine.RenderObservation(self._buffer)
    return self._buffer.reshape(self._depth, self._height, self._width)


class ViewImageHWC(ViewImage):
  """第一人称街景图，形状 (H, W, C)，C=3（通道在最后）。"""
  name = 'view_image_hwc'
  observation_spec_dtypes = np.uint8

  @property
  def observation_spec(self):
    return [self._height, self._width, self._depth]

  @property
  def observation(self):
    self._streetlearn.engine.RenderObservation(self._buffer)
    return reshape_hwc(self._buffer, self._depth, self._width, self._height)


class GraphImage(Observation):
  """俯视地图图的 RGB 像素，形状 (C, H, W)，C=3。"""
  name = 'graph_image'
  observation_spec_dtypes = np.uint8

  def __init__(self, streetlearn):
    super(GraphImage, self).__init__(streetlearn)
    self._width = streetlearn.config["graph_width"]
    self._height = streetlearn.config["graph_height"]
    self._depth = 3
    self._buffer = np.empty(self._depth * self._width * self._height,
                            dtype=np.uint8)

  @property
  def observation_spec(self):
    return [self._depth, self._height, self._width]

  @property
  def observation(self):
    # 契约修复点②：DrawGraph 把 planar RGB 像素写入 self._buffer 后，返回的是
    # 一个 bool 成功标志（NOT 图像本身）。如果这里直接返回 DrawGraph 的返回值，
    # graph_image 就会变成标量 bool，下游 obs_to_batch 在 stack 时塌成 1-D。
    # 必须 reshape 这个 buffer 返回 [3, H, W]。
    highlighted_panos = self._streetlearn.game.highlighted_panos()
    self._streetlearn.engine.DrawGraph(highlighted_panos, self._buffer)
    return self._buffer.reshape(self._depth, self._height, self._width)


class GraphImageHWC(GraphImage):
  """俯视地图图，形状 (H, W, C)，C=3。"""
  name = 'graph_image_hwc'
  observation_spec_dtypes = np.uint8

  @property
  def observation_spec(self):
    return [self._height, self._width, self._depth]

  @property
  def observation(self):
    highlighted_panos = self._streetlearn.game.highlighted_panos()
    self._streetlearn.engine.DrawGraph(highlighted_panos, self._buffer)
    return reshape_hwc(self._buffer, self._depth, self._width, self._height)


class Yaw(Observation):
  """智能体当前偏航角 yaw（与当前全景朝向不同）。单位：度。"""
  name = 'yaw'
  observation_spec_dtypes = np.float64

  @property
  def observation_spec(self):
    return [1]

  @property
  def observation(self):
    return np.array(self._streetlearn.engine.GetYaw(), dtype=np.float64)


class Pitch(Observation):
  """智能体当前俯仰角 pitch。单位：度。"""
  name = 'pitch'
  observation_spec_dtypes = np.float64

  @property
  def observation_spec(self):
    return [1]

  @property
  def observation(self):
    return np.array(self._streetlearn.engine.GetPitch(), dtype=np.float64)


class YawLabel(Observation):
  """yaw 的离散分箱标签（把连续角度量化成 _NUM_HEADING_BINS 个 bin）。"""
  name = 'yaw_label'
  observation_spec_dtypes = np.uint8

  @property
  def observation_spec(self):
    return [1]

  @property
  def observation(self):
    yaw = self._streetlearn.engine.GetYaw() % 360.0
    return np.array(yaw * _NUM_HEADING_BINS / 360.0, dtype=np.uint8)


class Metadata(Observation):
  """当前全景的元数据（protobuf 序列化）。"""
  name = 'metadata'
  observation_spec_dtypes = bytearray

  @property
  def observation_spec(self):
    return [_METADATA_COUNT]

  @property
  def observation(self):
    pano_id = self._streetlearn.current_pano_id
    return bytearray(self._streetlearn.engine.GetMetadata(
        pano_id).pano.SerializeToString())


class TargetMetadata(Observation):
  """目标全景的元数据。"""
  name = 'target_metadata'
  observation_spec_dtypes = bytearray

  @property
  def observation_spec(self):
    return [_METADATA_COUNT]

  @property
  def observation(self):
    goal_id = self._streetlearn.game.goal_id
    if goal_id:
      return bytearray(self._streetlearn.engine.GetMetadata(
          goal_id).pano.SerializeToString())
    return bytearray()


class LatLng(Observation):
  """智能体当前经纬度，按 config 的 bbox 边界缩放到 [0,1]。

  原理：原始经纬度是绝对地理坐标，尺度大且不同城市差异大；用数据集的
  bbox（lat/lng 的最小/最大值）做线性归一化，使坐标落在 [0,1]，便于网络学习。
  """
  name = 'latlng'
  observation_spec_dtypes = np.float64

  def _scale_lat(self, lat):
    den = self._streetlearn.bbox_lat_max - self._streetlearn.bbox_lat_min
    return ((lat - self._streetlearn.bbox_lat_min) / den if (den > 0) else 0)

  def _scale_lng(self, lng):
    den = self._streetlearn.bbox_lng_max - self._streetlearn.bbox_lng_min
    return ((lng - self._streetlearn.bbox_lng_min) / den if (den > 0) else 0)

  @property
  def observation_spec(self):
    return [2]

  @property
  def observation(self):
    pano_id = self._streetlearn.current_pano_id
    pano_data = self._streetlearn.engine.GetMetadata(pano_id).pano
    lat_scaled = self._scale_lat(pano_data.coords.lat)
    lng_scaled = self._scale_lng(pano_data.coords.lng)
    return np.array([lat_scaled, lng_scaled], dtype=np.float64)


class LatLngLabel(LatLng):
  """经纬度离散分箱标签（量化成网格索引）。"""
  name = 'latlng_label'
  observation_spec_dtypes = np.int32

  def _latlng_bin(self, pano_id):
    pano_data = self._streetlearn.engine.GetMetadata(pano_id).pano
    lat_bin = np.floor(self._scale_lat(pano_data.coords.lat) * _NUM_LAT_BINS)
    lat_bin = np.max([np.min([lat_bin, _NUM_LAT_BINS-1]), 0])  # 截断到合法范围
    lng_bin = np.floor(self._scale_lng(pano_data.coords.lng) * _NUM_LNG_BINS)
    lng_bin = np.max([np.min([lng_bin, _NUM_LNG_BINS-1]), 0])
    latlng_bin = lat_bin * _NUM_LNG_BINS + lng_bin  # 二维网格展平成一维索引
    return latlng_bin

  @property
  def observation_spec(self):
    return [1]

  @property
  def observation(self):
    pano_id = self._streetlearn.current_pano_id
    return np.array(self._latlng_bin(pano_id), dtype=np.int32)


class TargetLatLng(LatLng):
  """目标点的经纬度（同样缩放到 [0,1]）。"""
  name = 'target_latlng'

  @property
  def observation(self):
    goal_id = self._streetlearn.game.goal_id
    if goal_id:
      pano_data = self._streetlearn.engine.GetMetadata(goal_id).pano
      lat_scaled = self._scale_lat(pano_data.coords.lat)
      lng_scaled = self._scale_lng(pano_data.coords.lng)
      return np.array([lat_scaled, lng_scaled], dtype=np.float64)
    # 没有目标时返回 [0,0]（占位）。
    return np.array([0, 0], dtype=np.float64)


class TargetLatLngLabel(LatLngLabel):
  """目标经纬度的离散分箱标签。"""
  name = 'target_latlng_label'

  @property
  def observation(self):
    goal_id = self._streetlearn.game.goal_id
    if goal_id:
      return np.array(self._latlng_bin(goal_id), dtype=np.int32)
    return np.array(0, dtype=np.int32)


class Thumbnails(Observation):
  """相邻全景的缩略图像素。"""
  name = 'thumbnails'
  observation_spec_dtypes = np.uint8

  @property
  def observation_spec(self):
    return self._streetlearn.game.thumbnails().shape

  @property
  def observation(self):
    return self._streetlearn.game.thumbnails()


class Instructions(Observation):
  """StreetLang 自然语言导航指令（拼接成字符串）。"""
  name = 'instructions'
  observation_spec_dtypes = str

  @property
  def observation_spec(self):
    return [1]

  @property
  def observation(self):
    return ('|'.join(self._streetlearn.game.instructions())).encode('utf-8')


class GroundTruthDirection(Observation):
  """智能体此刻应当前进的真实方向（度）。用于监督/调试。"""
  name = 'ground_truth_direction'
  observation_spec_dtypes = np.float32

  @property
  def observation_spec(self):
    return [1]

  @property
  def observation(self):
    return np.array(self._streetlearn.game.ground_truth_direction(),
                    dtype=np.float32)


class Neighbors(Observation):
  """当前全景各方向的邻居占用情况（连通性标志）。

  返回长度 = neighbor_resolution（默认 8）的 0/1 向量，每个元素表示对应
  方向上是否有可行走邻居。这是 PPO 标量观测里 neighbors(8) 的来源，告诉
  智能体"哪些方向能走"。底层 GetNeighborOccupancy 把邻居方位角按当前 yaw
  重新居中后分箱。
  """
  name = 'neighbors'
  observation_spec_dtypes = np.uint8

  @property
  def observation_spec(self):
    return [self._streetlearn.neighbor_resolution]

  @property
  def observation(self):
    return np.asarray(
        self._streetlearn.engine.GetNeighborOccupancy(
            self._streetlearn.neighbor_resolution),
        dtype=np.uint8)


class PrevReward(Observation):
  """上一步的奖励（供某些策略当作额外输入）。"""
  name = 'prev_reward'
  observation_spec_dtypes = np.float32

  @property
  def observation_spec(self):
    return [1]

  @property
  def observation(self):
    return np.array(self._streetlearn.prev_reward(), dtype=np.float32)


class PrevAction(Observation):
  """上一步的动作（one-hot 风格，长度 4）。"""
  name = 'prev_action'
  observation_spec_dtypes = np.float32

  @property
  def observation_spec(self):
    return [4]

  @property
  def observation(self):
    return np.array(self._streetlearn.prev_action(), dtype=np.float32)
