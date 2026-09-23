"""Pano projection - pure Python replacement for pano_projection.cc.

Projects an equirectangular panorama image into a first-person perspective view
given yaw and pitch angles. Uses numpy for the 3D point cloud math and
OpenCV's remap for the final pixel sampling (matching the C++ implementation).
Falls back to numpy-only remap if cv2 is unavailable.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

_PANO_RADIUS = 120.0


class PanoProjection(object):
  """Equirectangular to perspective projection."""

  def __init__(self, fov_deg, proj_width, proj_height):
    self._fov_deg = fov_deg
    self._proj_width = proj_width
    self._proj_height = proj_height
    self._xyz_centered = self._prepare_xyz_point_cloud()

  def _prepare_xyz_point_cloud(self):
    """Build centered 3D point cloud for the projection screen."""
    fov_w = self._fov_deg
    fov_h = self._fov_deg * self._proj_height / self._proj_width

    half_fov_w = math.radians(fov_w / 2.0)
    half_afov_w = math.radians((180.0 - fov_w) / 2.0)
    half_fov_h = math.radians(fov_h / 2.0)
    half_afov_h = math.radians((180.0 - fov_h) / 2.0)

    ratio_w = math.sin(half_fov_w) / math.sin(half_afov_w)
    scale_w = 2.0 * _PANO_RADIUS * ratio_w / (self._proj_width - 1)
    ratio_h = math.sin(half_fov_h) / math.sin(half_afov_h)
    scale_h = 2.0 * _PANO_RADIUS * ratio_h / (self._proj_height - 1)

    c_x = (self._proj_width - 1.0) / 2.0
    c_y = (self._proj_height - 1.0) / 2.0

    # Build coordinate grids
    j_vals = np.arange(self._proj_width, dtype=np.float64)
    i_vals = np.arange(self._proj_height, dtype=np.float64)
    jj, ii = np.meshgrid(j_vals, i_vals)  # shape (H, W)

    x = np.full_like(jj, _PANO_RADIUS)
    y = (jj - c_x) * scale_w
    z = -(ii - c_y) * scale_h
    d = np.sqrt(x * x + y * y + z * z)
    coeff = _PANO_RADIUS / d

    xyz = np.stack([coeff * x, coeff * y, coeff * z], axis=0)  # (3, H, W)
    return xyz.reshape(3, -1)  # (3, H*W)

  def _rotate_xyz(self, yaw_deg, pitch_deg):
    """Rotate point cloud by yaw (around Z) and pitch (around rotated Y)."""
    # Yaw rotation around Z axis
    yaw_rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    # z_rot = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
    z_rot = np.array([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # Pitch: rotate around the Y axis that has been transformed by z_rot.
    # In the C++ code: y_axis_rotated = z_rot @ [0,1,0], then Rodrigues around that.
    # For rotation around Y: [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    pitch_rad = math.radians(-pitch_deg)
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    y_rot_local = np.array([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ], dtype=np.float64)

    # Combined rotation: y_rot_world = z_rot @ y_rot_local @ z_rot^T
    # Then apply to centered cloud: y_rot_world @ z_rot @ xyz
    # Simplify: z_rot @ y_rot_local @ z_rot^T @ z_rot @ xyz = z_rot @ y_rot_local @ xyz
    xyz_rot = z_rot @ y_rot_local @ self._xyz_centered
    return xyz_rot

  def project(self, pano_image, yaw_deg, pitch_deg):
    """Project equirectangular panorama to perspective view.

    Args:
      pano_image: numpy array (H, W, 3) uint8 - equirectangular panorama.
      yaw_deg: agent yaw in degrees.
      pitch_deg: agent pitch in degrees.
    Returns:
      numpy array (proj_height, proj_width, 3) uint8 - perspective view.
    """
    if pano_image is None:
      return np.zeros((self._proj_height, self._proj_width, 3),
                      dtype=np.uint8)

    pano_h, pano_w = pano_image.shape[:2]
    pano_cx = (pano_w - 1.0) / 2.0
    pano_cy = (pano_h - 1.0) / 2.0

    xyz = self._rotate_xyz(yaw_deg, pitch_deg)

    # Project to lat/lon maps
    x = xyz[0, :]
    y = xyz[1, :]
    z = xyz[2, :]

    sin_lat = np.clip(z / _PANO_RADIUS, -1.0, 1.0)
    lat = np.arcsin(sin_lat)
    map_lat = -lat / math.pi * 2.0 * pano_cy + pano_cy

    theta = np.arctan2(y, x)
    # atan2 handles the quadrant logic that the C++ code does manually
    map_lon = theta / math.pi * pano_cx + pano_cx

    map_lon = map_lon.reshape(self._proj_height, self._proj_width).astype(np.float32)
    map_lat = map_lat.reshape(self._proj_height, self._proj_width).astype(np.float32)

    # Use cv2.remap if available (matches C++ exactly)
    try:
      import cv2
      output = cv2.remap(
          pano_image, map_lon, map_lat,
          interpolation=cv2.INTER_CUBIC,
          borderMode=cv2.BORDER_WRAP)
      return output
    except ImportError:
      return self._numpy_remap(pano_image, map_lon, map_lat)

  def _numpy_remap(self, src, map_x, map_y):
    """Fallback remap using numpy indexing (slower, no interpolation)."""
    h, w = src.shape[:2]
    xi = np.clip(map_x.astype(np.int32), 0, w - 1)
    yi = np.clip(map_y.astype(np.int32), 0, h - 1)
    return src[yi, xi]

  def change_fov(self, fov_deg):
    if fov_deg != self._fov_deg:
      self._fov_deg = fov_deg
      self._xyz_centered = self._prepare_xyz_point_cloud()
