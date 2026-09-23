"""Pano renderer - pure Python replacement for pano_renderer.cc.

Renders the first-person view by:
1. Projecting the equirectangular panorama to perspective (via PanoProjection)
2. Drawing navigation indicator dots on the status bar (green/amber/red)
3. Drawing red stop signs for terminal nodes
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from streetlearn.engine.python import math_util
from streetlearn.engine.python import pano_projection
from streetlearn.engine.python.color import Color

# Status bar colors (matching C++ constants)
_BACK = (0, 0, 0)
_BAR = (128, 128, 128)
_RED = (230, 0, 0)
_AMBER = (255, 125, 0)
_GREEN = (0, 230, 0)
_NO_ENTRY = (230, 0, 0)

_NO_ENTRY_PROPORTION = 0.1
_NO_ENTRY_SIGN_WIDTH_METERS = 3.0
_CONSTRAINT_MULTIPLIER = 2


class PanoRenderer(object):
  """Renders panorama views with status bar and navigation indicators."""

  def __init__(self, screen_width, screen_height, status_height, fov_deg):
    self._width = screen_width
    self._height = screen_height
    self._fov_deg = fov_deg
    self._status_height = status_height
    self._no_entry_width = screen_width * _NO_ENTRY_PROPORTION
    self._degrees_pp = 360.0 / screen_width
    self._image_height = screen_height - status_height
    self._projection = pano_projection.PanoProjection(
        fov_deg, screen_width, self._image_height)
    # Output buffer: HWC packed RGB
    self._pixels = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

  @property
  def pixels(self):
    return self._pixels

  def render_scene(self, pano_image, global_yaw, yaw, pitch, tolerance,
                   bearings, terminal_bearings):
    """Render the complete scene."""
    if self._status_height > 0:
      self._draw_status_bar(tolerance, yaw, bearings)

    # Project panorama (yaw relative to pano heading)
    rel_yaw = self._constrain_angle(360, yaw - global_yaw)
    rel_pitch = self._constrain_angle(90, pitch)
    projected = self._projection.project(pano_image, rel_yaw, rel_pitch)
    self._pixels[:self._image_height] = projected

    # Draw stop signs for terminal nodes
    self._draw_no_entry_signs(global_yaw, yaw, pitch, terminal_bearings)

  def _constrain_angle(self, constraint, angle):
    """Wrap angle to [-constraint, constraint]."""
    value_range = _CONSTRAINT_MULTIPLIER * constraint
    value = angle + constraint
    value -= value_range * math.floor(value / value_range)
    value -= constraint
    return value

  def _draw_status_bar(self, tolerance, current_bearing, bearings):
    """Draw the navigation indicator bar at the bottom."""
    bar = self._pixels[self._image_height:]
    bar[:] = _BACK

    # Find chosen direction (closest to current bearing, within tolerance)
    chosen_dir = None
    min_distance = float("inf")
    for nb in bearings:
      diff = abs(nb.bearing - current_bearing)
      if diff > 180.0:
        diff = 360.0 - diff
      if diff < tolerance and nb.distance < min_distance:
        chosen_dir = nb.bearing
        min_distance = nb.distance

    # Draw all bearings except chosen
    for nb in bearings:
      if nb.bearing == chosen_dir:
        continue
      pos = abs(nb.bearing - current_bearing)
      if pos > 180.0:
        pos = 360.0 - pos
      color = _AMBER if pos < tolerance else _RED
      self._draw_dot(nb.bearing - current_bearing, color)

    if chosen_dir is not None:
      self._draw_dot(chosen_dir - current_bearing, _GREEN)

    # Tolerance bars
    lower_x = int(self._width / 2.0 + self._constrain_angle(180, -tolerance) / self._degrees_pp)
    upper_x = int(self._width / 2.0 + self._constrain_angle(180, tolerance) / self._degrees_pp)
    for x in (lower_x, upper_x):
      if 0 <= x < self._width:
        bar[:, x] = _BAR

  def _draw_dot(self, offset, color):
    """Draw a filled circle on the status bar."""
    pos = self._constrain_angle(180, offset)
    cx = int(self._width / 2.0 + pos / self._degrees_pp)
    cy = self._status_height // 2
    r = max(1, self._status_height // 2 - 1)
    bar = self._pixels[self._image_height:]
    for dy in range(-r, r + 1):
      for dx in range(-r, r + 1):
        if dx * dx + dy * dy <= r * r:
          px, py = cx + dx, cy + dy
          if 0 <= px < self._width:
            bar[py, px] = color

  def _draw_no_entry_signs(self, global_yaw, current_bearing, current_pitch,
                           terminal_bearings):
    """Draw red circles for terminal/stop nodes."""
    for depth in sorted(terminal_bearings.keys(), reverse=True):
      for term in terminal_bearings[depth]:
        sign_width = int(self._no_entry_width *
                         (2.0 * math.atan(_NO_ENTRY_SIGN_WIDTH_METERS / term.distance)) /
                         math_util.degrees_to_radians(self._fov_deg))
        sign_width = max(1, sign_width)
        offset_x = self._constrain_angle(180, term.bearing - current_bearing)
        offset_y = current_pitch
        x_pos = int(self._width * (0.5 + offset_x / self._fov_deg))
        y_pos = int(self._image_height * (0.5 - offset_y * self._width / (self._fov_deg * self._height)))
        self._draw_circle(x_pos, y_pos, sign_width, _NO_ENTRY)

  def _draw_circle(self, cx, cy, radius, color):
    """Draw a filled circle on the main view area."""
    img = self._pixels[:self._image_height]
    for dy in range(-radius, radius + 1):
      for dx in range(-radius, radius + 1):
        if dx * dx + dy * dy <= radius * radius:
          px, py = cx + dx, cy + dy
          if 0 <= px < self._width and 0 <= py < self._image_height:
            img[py, px] = color
