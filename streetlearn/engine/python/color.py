"""Color class - pure Python replacement for C++ color.h / CLIF color.clif."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Color(object):
  """RGB color with float components in [0, 1]."""

  def __init__(self, r=0.0, g=0.0, b=0.0):
    self.red = float(r)
    self.green = float(g)
    self.blue = float(b)
