"""Pano distance/bearing calculations - replaces pano_calculations.h.

Uses the Haversine formula for distance and spherical trigonometry for bearing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

EARTH_RADIUS_METRES = 6371000.0


def degrees_to_radians(deg):
  return deg * math.pi / 180.0


def bearing_between_coords(lat1, lng1, lat2, lng2):
  """Returns the bearing from (lat1,lng1) to (lat2,lng2) in degrees.

  Bearing is measured clockwise from North, range [-180, 180].
  """
  lat1_r = degrees_to_radians(lat1)
  lon1_r = degrees_to_radians(lng1)
  lat2_r = degrees_to_radians(lat2)
  lon2_r = degrees_to_radians(lng2)
  x = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(lon2_r - lon1_r)
  y = math.sin(lon2_r - lon1_r) * math.cos(lat2_r)
  return 180.0 * math.atan2(y, x) / math.pi


def distance_between_coords(lat1, lng1, lat2, lng2):
  """Haversine distance in metres between two lat/lng points."""
  lat1_r = degrees_to_radians(lat1)
  lon1_r = degrees_to_radians(lng1)
  lat2_r = degrees_to_radians(lat2)
  lon2_r = degrees_to_radians(lng2)
  u = math.sin((lat2_r - lat1_r) / 2.0)
  v = math.sin((lon2_r - lon1_r) / 2.0)
  return 2.0 * EARTH_RADIUS_METRES * math.asin(
      math.sqrt(u * u + math.cos(lat1_r) * math.cos(lat2_r) * v * v))


def bearing_between_panos(pano1, pano2):
  """Bearing between two Pano proto messages."""
  return bearing_between_coords(
      pano1.coords.lat, pano1.coords.lng,
      pano2.coords.lat, pano2.coords.lng)


def distance_between_panos(pano1, pano2):
  """Distance between two Pano proto messages."""
  return distance_between_coords(
      pano1.coords.lat, pano1.coords.lng,
      pano2.coords.lat, pano2.coords.lng)
