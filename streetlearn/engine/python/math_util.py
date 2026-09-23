"""Math utilities - pure Python replacement for C++ math_util.h."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random


def degrees_to_radians(degrees):
  return degrees * math.pi / 180.0


def radians_to_degrees(radians):
  return radians * 180.0 / math.pi


def clamp(low, high, value):
  if value > high:
    return high
  if value < low:
    return low
  return value


def uniform_random_int(rng, max_val):
  """Returns a random integer in [0, max_val]."""
  return rng.randint(0, max_val)
