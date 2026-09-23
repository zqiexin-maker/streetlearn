"""Graph renderer - pure Python replacement for graph_renderer.cc.

Renders a top-down map view of the pano graph, showing pano positions as dots
with configurable colors, the agent's position, and view cone.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from streetlearn.engine.python import math_util
from streetlearn.engine.python import pano_calculations
from streetlearn.engine.python.color import Color


class GraphRenderer(object):
  """Renders a top-down graph image of the streetlearn graph."""

  def __init__(self, pano_graph, graph_size, highlighted_panos,
               black_on_white=False):
    self._pano_graph = pano_graph
    self._width = graph_size[0]
    self._height = graph_size[1]
    self._highlighted = dict(highlighted_panos)
    self._black_on_white = black_on_white
    self._zoom = 1.0
    self._pixels = np.zeros((self._height, self._width, 3), dtype=np.uint8)

    # Compute bounding box of all pano coordinates
    coords = []
    graph = pano_graph.get_graph()
    for pano_id in graph:
      meta = pano_graph.get_metadata(pano_id)
      if meta and meta.pano:
        coords.append((meta.pano.coords.lat, meta.pano.coords.lng))
    if coords:
      self._min_lat = min(c[0] for c in coords)
      self._max_lat = max(c[0] for c in coords)
      self._min_lng = min(c[1] for c in coords)
      self._max_lng = max(c[1] for c in coords)
    else:
      self._min_lat = self._max_lat = 0
      self._min_lng = self._max_lng = 0

  @classmethod
  def create(cls, pano_graph, graph_size, highlighted_panos, black_on_white):
    try:
      return cls(pano_graph, graph_size, highlighted_panos, black_on_white)
    except Exception:
      return None

  def set_zoom(self, zoom):
    self._zoom = max(1.0, min(32.0, zoom))
    return True

  def render_scene(self, pano_id_to_color, observer):
    """Render the graph into self._pixels.

    Args:
      pano_id_to_color: dict {pano_id: Color} for highlighting nodes.
      observer: Observer with pano_id, yaw, fov, color.
    """
    bg = (255, 255, 255) if self._black_on_white else (0, 0, 0)
    self._pixels[:] = bg

    graph = self._pano_graph.get_graph()
    lat_range = max(1e-6, self._max_lat - self._min_lat)
    lng_range = max(1e-6, self._max_lng - self._min_lng)

    # Scale to fit in image, with zoom
    scale = min(self._width / lng_range, self._height / lat_range) * self._zoom
    cx = self._width / 2.0
    cy = self._height / 2.0
    mid_lat = (self._min_lat + self._max_lat) / 2.0
    mid_lng = (self._min_lng + self._max_lng) / 2.0

    # Draw edges
    edge_color = (200, 200, 200) if self._black_on_white else (60, 60, 60)
    for pano_id, neighbors in graph.items():
      meta = self._pano_graph.get_metadata(pano_id)
      if not meta or not meta.pano:
        continue
      x1 = int(cx + (meta.pano.coords.lng - mid_lng) * scale)
      y1 = int(cy - (meta.pano.coords.lat - mid_lat) * scale)
      for nb_id in neighbors:
        nb_meta = self._pano_graph.get_metadata(nb_id)
        if not nb_meta or not nb_meta.pano:
          continue
        x2 = int(cx + (nb_meta.pano.coords.lng - mid_lng) * scale)
        y2 = int(cy - (nb_meta.pano.coords.lat - mid_lat) * scale)
        self._draw_line(x1, y1, x2, y2, edge_color)

    # Draw nodes with passed colors
    for pano_id in graph:
      meta = self._pano_graph.get_metadata(pano_id)
      if not meta or not meta.pano:
        continue
      x = int(cx + (meta.pano.coords.lng - mid_lng) * scale)
      y = int(cy - (meta.pano.coords.lat - mid_lat) * scale)
      color = pano_id_to_color.get(pano_id) if pano_id_to_color else None
      if color:
        c = (int(color.red * 255), int(color.green * 255), int(color.blue * 255))
      else:
        c = (100, 100, 100) if self._black_on_white else (150, 150, 150)
      self._draw_dot(x, y, 2, c)

    # Draw observer position and view cone
    obs_color = (int(observer.color.red * 255),
                 int(observer.color.green * 255),
                 int(observer.color.blue * 255))
    obs_meta = self._pano_graph.get_metadata(observer.pano_id)
    if obs_meta and obs_meta.pano:
      ox = int(cx + (obs_meta.pano.coords.lng - mid_lng) * scale)
      oy = int(cy - (obs_meta.pano.coords.lat - mid_lat) * scale)
      self._draw_dot(ox, oy, 3, obs_color)
      # View cone
      fov_rad = observer.fov_yaw_radians
      yaw_rad = observer.yaw_radians
      cone_len = 15
      for angle_offset in [-fov_rad / 2, fov_rad / 2]:
        ex = int(ox + cone_len * math.sin(yaw_rad + angle_offset))
        ey = int(oy - cone_len * math.cos(yaw_rad + angle_offset))
        self._draw_line(ox, oy, ex, ey, obs_color)

    return True

  def get_pixels(self, buffer):
    """Copy pixels into the provided buffer (1D CHW format)."""
    # Convert HWC packed -> CHW planar
    h, w = self._height, self._width
    packed = self._pixels.reshape(-1)
    planar = np.empty(h * w * 3, dtype=np.uint8)
    planar[:h * w] = packed[0::3]
    planar[h * w:2 * h * w] = packed[1::3]
    planar[2 * h * w:] = packed[2::3]
    buffer[:] = planar
    return True

  def _draw_dot(self, cx, cy, radius, color):
    for dy in range(-radius, radius + 1):
      for dx in range(-radius, radius + 1):
        if dx * dx + dy * dy <= radius * radius:
          px, py = cx + dx, cy + dy
          if 0 <= px < self._width and 0 <= py < self._height:
            self._pixels[py, px] = color

  def _draw_line(self, x1, y1, x2, y2, color):
    """Bresenham line drawing."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    while True:
      if 0 <= x1 < self._width and 0 <= y1 < self._height:
        self._pixels[y1, x1] = color
      if x1 == x2 and y1 == y2:
        break
      e2 = 2 * err
      if e2 > -dy:
        err -= dy
        x1 += sx
      if e2 < dx:
        err += dx
        y1 += sy


class Observer(object):
  """Observer state for graph rendering."""

  def __init__(self):
    self.pano_id = ""
    self.yaw_radians = 0.0
    self.fov_yaw_radians = 0.0
    self.color = Color(0.5, 0.5, 1.0)
