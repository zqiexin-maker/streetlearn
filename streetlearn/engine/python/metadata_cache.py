"""Metadata cache - pure Python replacement for metadata_cache.cc.

Reads the StreetLearnGraph protobuf from the dataset and builds an in-memory
metadata cache mapping pano_id -> PanoMetadata. Only panos whose subgraph
size >= min_graph_depth are included (sufficient connectivity for navigation).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import namedtuple


# Lightweight data classes to avoid protobuf dependency in game logic.
LatLng = namedtuple("LatLng", ["lat", "lng"])
PanoIdAndGpsCoords = namedtuple("PanoIdAndGpsCoords", ["id", "coords"])


class PanoData(object):
  """Holds pano metadata fields (mirrors proto Pano message)."""

  __slots__ = (
      "id", "address_language", "street_name", "full_address",
      "region", "country_code", "coords", "alt", "roll_deg", "pitch_deg",
      "heading_deg", "pano_date", "scene_type", "neighbor", "temporal_neighbor",
      "snapped_coords", "_proto")

  def __init__(self):
    self.id = ""
    self.address_language = ""
    self.street_name = ""
    self.full_address = ""
    self.region = ""
    self.country_code = ""
    self.coords = LatLng(0.0, 0.0)
    self.alt = 0.0
    self.roll_deg = 0.0
    self.pitch_deg = 0.0
    self.heading_deg = 0.0
    self.pano_date = ""
    self.scene_type = 0
    self.neighbor = []
    self.temporal_neighbor = []
    self.snapped_coords = None
    self._proto = None  # keep original proto for SerializeToString

  def SerializeToString(self):
    """Match protobuf API name for compatibility with observations.py."""
    if self._proto is not None:
      return self._proto.SerializeToString()
    return b""

  # Also provide lowercase alias
  serialize_to_string = SerializeToString

  @classmethod
  def from_proto(cls, pano_proto):
    """Build PanoData from a protobuf Pano message."""
    d = cls()
    d.id = pano_proto.id
    d.address_language = pano_proto.address_language
    d.street_name = pano_proto.street_name
    d.full_address = pano_proto.full_address
    d.region = pano_proto.region
    d.country_code = pano_proto.country_code
    d.coords = LatLng(pano_proto.coords.lat, pano_proto.coords.lng)
    d.alt = pano_proto.alt
    d.roll_deg = pano_proto.roll_deg
    d.pitch_deg = pano_proto.pitch_deg
    d.heading_deg = pano_proto.heading_deg
    d.pano_date = pano_proto.pano_date
    d.scene_type = pano_proto.scene_type
    d._proto = pano_proto
    return d


class PanoMetadata(object):
  """Mirrors the C++ PanoMetadata struct."""

  __slots__ = ("neighbors", "pano", "graph_depth")

  def __init__(self):
    self.neighbors = []  # list of PanoIdAndGpsCoords
    self.pano = None     # PanoData
    self.graph_depth = 0


class MetadataCache(object):
  """In-memory cache of panorama metadata loaded from the dataset."""

  def __init__(self, min_graph_depth=10):
    self._pano_metadata = {}
    self.min_lat = -1.0
    self.min_lng = -1.0
    self.max_lat = -1.0
    self.max_lng = -1.0
    self._min_graph_depth = min_graph_depth

  @classmethod
  def create(cls, dataset, min_graph_depth):
    cache = cls(min_graph_depth)
    if not cache.read_data(dataset):
      return None
    return cache

  def read_data(self, dataset):
    """Read the graph protobuf and build the metadata cache."""
    # Lazy import protobuf
    from streetlearn.proto import streetlearn_pb2
    graph = streetlearn_pb2.StreetLearnGraph()
    if not dataset.get_graph(graph):
      logging.error("Cannot read StreetLearn graph!")
      return False

    self._preprocess_graph(graph)

    self.min_lat = graph.min_coords.lat
    self.min_lng = graph.min_coords.lng
    self.max_lat = graph.max_coords.lat
    self.max_lng = graph.max_coords.lng

    # Build pano lookup
    panos = {}
    for pano in graph.pano:
      if pano.id in panos:
        logging.warning("Pano %s has two metadata entries!", pano.id)
      panos[pano.id] = pano

    self._process_neighbors(graph, panos)
    return True

  def _preprocess_pano(self, pano):
    """If neighbor has snapped_coords, use them as coords."""
    for neighbor in pano.neighbor:
      if neighbor.HasField("snapped_coords"):
        neighbor.coords.lat = neighbor.snapped_coords.lat
        neighbor.coords.lng = neighbor.snapped_coords.lng

  def _preprocess_graph(self, graph):
    for pano in graph.pano:
      self._preprocess_pano(pano)

  def _process_neighbors(self, graph, panos):
    for conn in graph.connection:
      if conn.subgraph_size < self._min_graph_depth:
        continue
      pano_id = conn.id
      if pano_id not in panos:
        continue
      pano_proto = panos[pano_id]

      metadata = PanoMetadata()
      metadata.pano = PanoData.from_proto(pano_proto)
      metadata.graph_depth = conn.subgraph_size

      for neighbor_id in conn.neighbor:
        if neighbor_id not in panos:
          logging.warning("Neighbor %s of %s not found in panos",
                          neighbor_id, pano_id)
          continue
        neighbor_pano = panos[neighbor_id]
        coords = LatLng(neighbor_pano.coords.lat, neighbor_pano.coords.lng)
        metadata.neighbors.append(
            PanoIdAndGpsCoords(id=neighbor_id, coords=coords))

      self._pano_metadata[pano_id] = metadata

  def get_pano_metadata(self, pano_id):
    return self._pano_metadata.get(pano_id)

  def panos_in_graphs_of_size(self, min_size):
    """Return pano IDs whose graph depth >= min_size."""
    return [pid for pid, meta in self._pano_metadata.items()
            if meta.graph_depth >= min_size]

  def pano_in_largest_graph(self):
    if not self._pano_metadata:
      return None
    return max(self._pano_metadata.items(),
               key=lambda x: x[1].graph_depth)[1].pano.id
