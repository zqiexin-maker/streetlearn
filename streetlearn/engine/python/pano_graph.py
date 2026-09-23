"""Pano graph - pure Python replacement for pano_graph.cc.

Handles BFS graph building, async prefetching, navigation (MoveToNeighbor),
neighbor bearing computation, and terminal node detection.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import random
import threading
from collections import OrderedDict, namedtuple
from concurrent.futures import ThreadPoolExecutor

from streetlearn.engine.python import math_util
from streetlearn.engine.python import pano_calculations
from streetlearn.engine.python.metadata_cache import MetadataCache, PanoMetadata

PanoNeighborBearing = namedtuple(
    "PanoNeighborBearing", ["pano_id", "bearing", "distance"])
TerminalGraphNode = namedtuple(
    "TerminalGraphNode", ["distance", "bearing"])

_MIN_GRAPH_DEPTH = 10
_NEIGHBOR_ALTITUDE_THRESHOLD = 2.0
_PREFETCH_DEPTH = 20
_THREAD_COUNT = 16


class PanoGraphNode(object):
  """A node in the pano graph with decoded image data."""

  __slots__ = ("pano_id", "image", "bearing")

  def __init__(self, pano_id, image, bearing):
    self.pano_id = pano_id
    self.image = image  # numpy HWC uint8 or None
    self.bearing = bearing  # pano heading_deg


class PanoGraph(object):
  """Graph of panoramas with BFS building and navigation support."""

  def __init__(self, prefetch_depth, min_graph_depth, max_graph_depth,
               dataset, max_cache_size=1000):
    self._dataset = dataset
    self._min_graph_depth = min_graph_depth
    self._max_graph_depth = max_graph_depth
    self._prefetch_depth = prefetch_depth
    self._metadata_cache = None
    self._node_tree = {}          # pano_id -> PanoMetadata
    self._terminal_nodes = {}     # pano_id -> PanoMetadata
    self._root_node = None        # PanoGraphNode
    self._cache = OrderedDict()   # LRU: pano_id -> PanoGraphNode
    self._max_cache_size = max_cache_size
    self._cache_lock = threading.Lock()
    self._random = random.Random(0)
    self._executor = ThreadPoolExecutor(max_workers=_THREAD_COUNT)
    self._pending = set()

  def init(self):
    """Initialize the metadata cache from the dataset."""
    self._metadata_cache = MetadataCache.create(
        self._dataset, _MIN_GRAPH_DEPTH)
    if self._metadata_cache is None:
      logging.error("Unable to initialize Pano graph")
      return False
    return True

  def set_random_seed(self, seed):
    self._random.seed(seed)

  # ---- Graph building ----

  def build_graph_with_root(self, pano_id):
    if not pano_id:
      logging.error("No pano ID has been specified")
      return False
    if not self._build_graph(pano_id, bounded=True):
      return False
    self._prefetch_graph(pano_id)
    return True

  def build_random_graph(self):
    panos = self._metadata_cache.panos_in_graphs_of_size(self._min_graph_depth)
    if not panos:
      logging.error("No graphs available of depth %d-%d",
                    self._min_graph_depth, self._max_graph_depth)
      return False
    idx = math_util.uniform_random_int(self._random, len(panos) - 1)
    root = panos[idx]
    if not self._build_graph(root, bounded=True):
      return False
    self._prefetch_graph(root)
    return True

  def build_entire_graph(self):
    pano = self._metadata_cache.pano_in_largest_graph()
    if not pano:
      return False
    if not self._build_graph(pano, bounded=False):
      return False
    idx = math_util.uniform_random_int(self._random, len(self._node_tree) - 1)
    root = list(self._node_tree.keys())[idx]
    self._prefetch_graph(root)
    return True

  def _build_graph(self, root_id, bounded=True):
    """BFS from root_id up to max_graph_depth (or unlimited if not bounded)."""
    self._node_tree.clear()
    self._terminal_nodes.clear()

    current_level = [root_id]
    already_in_tree = {root_id}
    max_depth = self._max_graph_depth if bounded else float("inf")

    depth = 0
    while True:
      # For the unbounded ("entire graph") case max_depth is inf, so this
      # comparison is always False and the loop terminates when the frontier
      # is exhausted (BFS completion) instead of via range(int(inf)), which
      # would raise OverflowError.
      if depth >= max_depth:
        break
      if not current_level:
        break
      temp_level = []
      for node_id in current_level:
        metadata = self._metadata_cache.get_pano_metadata(node_id)
        if metadata is None:
          logging.error("Unknown pano: %s", node_id)
          return False
        self._node_tree[node_id] = metadata
        # Single-neighbor nodes are terminal (dead ends)
        if len(metadata.neighbors) == 1:
          self._terminal_nodes[node_id] = metadata
        for neighbor in metadata.neighbors:
          if neighbor.id not in already_in_tree:
            already_in_tree.add(neighbor.id)
            temp_level.append(neighbor.id)
      current_level = temp_level

    # Nodes at the BFS boundary are external terminal nodes
    for node_id in current_level:
      metadata = self._metadata_cache.get_pano_metadata(node_id)
      if metadata:
        self._terminal_nodes[node_id] = metadata
    return True

  # ---- Prefetching ----

  def _prefetch_graph(self, new_root):
    """Prefetch panorama images for the subgraph around new_root."""
    new_tree = []
    current_level = [new_root]
    already = {new_root}
    for _ in range(self._prefetch_depth):
      temp_level = []
      for node_id in current_level:
        metadata = self._node_tree.get(node_id)
        if metadata is None:
          continue
        new_tree.append(node_id)
        for neighbor in metadata.neighbors:
          if neighbor.id not in self._node_tree:
            continue
          if neighbor.id not in already:
            already.add(neighbor.id)
            temp_level.append(neighbor.id)
      current_level = temp_level

    # Fetch root synchronously
    self._root_node = self._fetch_node(new_root)

    # Prefetch others asynchronously
    for node_id in new_tree:
      if node_id != new_root and node_id not in self._cache:
        self._pending.add(node_id)
        self._executor.submit(self._prefetch_node, node_id)

  def _fetch_node(self, pano_id):
    """Synchronously fetch a node from LevelDB, decode JPEG."""
    with self._cache_lock:
      if pano_id in self._cache:
        self._cache.move_to_end(pano_id)
        return self._cache[pano_id]
    node = self._load_node(pano_id)
    with self._cache_lock:
      self._cache[pano_id] = node
      if len(self._cache) > self._max_cache_size:
        self._cache.popitem(last=False)
    return node

  def _prefetch_node(self, pano_id):
    """Asynchronously prefetch a node."""
    try:
      self._fetch_node(pano_id)
    except Exception as e:
      logging.warning("Prefetch error for %s: %s", pano_id, e)
    finally:
      self._pending.discard(pano_id)

  def _load_node(self, pano_id):
    """Load pano from LevelDB, decode JPEG to numpy array."""
    import numpy as np
    from streetlearn.proto import streetlearn_pb2
    from PIL import Image
    import io

    pano_proto = streetlearn_pb2.Pano()
    if not self._dataset.get_pano(pano_id, pano_proto):
      logging.error("Failed to load pano %s", pano_id)
      return PanoGraphNode(pano_id, None, 0.0)

    if pano_proto.compressed_image:
      img = Image.open(io.BytesIO(pano_proto.compressed_image))
      img_array = np.array(img)
    else:
      img_array = None

    return PanoGraphNode(pano_id, img_array, pano_proto.heading_deg)

  # ---- Navigation ----

  def move_to_neighbor(self, current_bearing, tolerance):
    """Move to the neighbor closest to current_bearing within tolerance."""
    if not self._node_tree:
      logging.error("Cannot move until graph has been built!")
      return False

    bearing_rads = math_util.degrees_to_radians(current_bearing)
    tolerance_rads = math_util.degrees_to_radians(tolerance)
    neighbor_bearings = self.get_neighbor_bearings(3)

    node_id = None
    min_distance = float("inf")
    for nb in neighbor_bearings:
      offset = abs(bearing_rads - math_util.degrees_to_radians(nb.bearing))
      if offset > math.pi:
        offset = 2.0 * math.pi - offset
      if offset < tolerance_rads and nb.distance < min_distance:
        min_distance = nb.distance
        node_id = nb.pano_id

    if node_id:
      self._prefetch_graph(node_id)
      return True
    return False

  def get_neighbor_bearings(self, max_depth):
    """BFS to max_depth, returning bearings/distances of reachable neighbors."""
    retval = []
    root_meta = self._node_tree.get(self._root_node.pano_id)
    if root_meta is None:
      return retval

    visited = {self._root_node.pano_id}
    nodes = [n.id for n in root_meta.neighbors]

    for depth in range(max_depth):
      next_nodes = []
      for node_id in nodes:
        if node_id in visited:
          continue
        visited.add(node_id)
        node_meta = self._node_tree.get(node_id)
        if node_meta is None:
          continue
        # Altitude filter
        if depth == 0 or abs(root_meta.pano.alt - node_meta.pano.alt) < _NEIGHBOR_ALTITUDE_THRESHOLD:
          bearing = pano_calculations.bearing_between_panos(
              root_meta.pano, node_meta.pano)
          distance = pano_calculations.distance_between_panos(
              root_meta.pano, node_meta.pano)
          retval.append(PanoNeighborBearing(node_id, bearing, distance))
          for nb in node_meta.neighbors:
            if nb.id not in visited:
              next_nodes.append(nb.id)
      nodes = next_nodes
    return retval

  def terminal_bearings(self, node_id):
    """Get terminal node bearings for stop-sign rendering."""
    node = self._node_tree.get(node_id)
    if node is None:
      return {}

    if node_id in self._terminal_nodes:
      return self._interior_terminal_bearings(node)

    retval = {}
    for neighbor in node.neighbors:
      if neighbor.id in self._terminal_nodes:
        term = self._terminal_nodes[neighbor.id]
        bearing = pano_calculations.bearing_between_panos(node.pano, term.pano)
        distance = pano_calculations.distance_between_panos(node.pano, term.pano)
        retval.setdefault(1, []).append(
            TerminalGraphNode(distance, bearing))
      else:
        nb_meta = self._node_tree.get(neighbor.id)
        if nb_meta is None:
          continue
        for sn in nb_meta.neighbors:
          if sn.id in self._node_tree:
            continue
          if sn.id in self._terminal_nodes:
            term = self._terminal_nodes[sn.id]
            bearing = pano_calculations.bearing_between_panos(
                node.pano, term.pano)
            distance = pano_calculations.distance_between_panos(
                node.pano, term.pano)
            retval.setdefault(2, []).append(
                TerminalGraphNode(distance, bearing))
    return retval

  def _interior_terminal_bearings(self, node):
    retval = {}
    if len(node.neighbors) == 1:
      nb_meta = self._node_tree.get(node.neighbors[0].id)
      if nb_meta:
        bearing = pano_calculations.bearing_between_panos(
            nb_meta.pano, node.pano)
        distance = pano_calculations.distance_between_panos(
            node.pano, nb_meta.pano) / 4
        retval[1] = [TerminalGraphNode(distance, bearing)]
    return retval

  # ---- Queries ----

  def get_graph(self):
    """Return {pano_id: [neighbor_id, ...]} for current graph."""
    retval = {}
    for pano_id, metadata in self._node_tree.items():
      neighbors = []
      for nb in metadata.neighbors:
        if nb.id in self._node_tree:
          neighbors.append(nb.id)
      retval[pano_id] = neighbors
    return retval

  def get_metadata(self, pano_id):
    return self._node_tree.get(pano_id)

  def get_pano_distance(self, id1, id2):
    m1 = self._node_tree.get(id1)
    m2 = self._node_tree.get(id2)
    if m1 and m2:
      return pano_calculations.distance_between_panos(m1.pano, m2.pano)
    return None

  def get_pano_bearing(self, id1, id2):
    m1 = self._node_tree.get(id1)
    m2 = self._node_tree.get(id2)
    if m1 and m2:
      return pano_calculations.bearing_between_panos(m1.pano, m2.pano)
    return None

  @property
  def root(self):
    return self._root_node

  def root_image(self):
    return self._root_node.image if self._root_node else None

  def get_node_cache_size(self):
    with self._cache_lock:
      return len(self._cache)

  def set_graph_depth(self, min_depth, max_depth):
    self._min_graph_depth = min_depth
    self._max_graph_depth = max_depth
