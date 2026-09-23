"""LevelDB dataset wrapper - pure Python replacement for leveldb_dataset.cc.

Uses plyvel to read the LevelDB database. The database stores:
  - key "panos_connectivity" -> StreetLearnGraph protobuf (graph + metadata)
  - key <pano_id> -> Pano protobuf (includes compressed JPEG image)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

# Lazy import to avoid hard dependency at module load time
_leveldb = None

GRAPH_KEY = b"panos_connectivity"


def _get_leveldb():
  global _leveldb
  if _leveldb is None:
    try:
      import plyvel as _leveldb
    except ImportError:
      try:
        import leveldb as _leveldb
      except ImportError:
        raise ImportError(
            "Neither plyvel nor leveldb is installed. "
            "Install with: pip install plyvel")
  return _leveldb


class Dataset(object):
  """Wrapper around a LevelDB StreetLearn dataset."""

  def __init__(self, db):
    self._db = db

  @classmethod
  def create(cls, dataset_path):
    """Open the LevelDB database at dataset_path.

    NOTE: LevelDB only permits a single process to hold the write lock on a
    given database directory. When several environments are run in parallel
    (e.g. the PPO SubprocVecEnv), each worker must open its OWN copy of the
    dataset, or use a plyvel build that supports read_only=True. See
    streetlearn/python/ppo_train.py for the per-worker copy handling.
    """
    leveldb = _get_leveldb()
    try:
      db = leveldb.DB(dataset_path)
    except Exception as e:
      logging.error("Unable to open dataset at %s: %s", dataset_path, e)
      return None
    return cls(db)

  def get_graph(self, graph_proto):
    """Read the StreetLearnGraph protobuf from the database.

    Args:
      graph_proto: a StreetLearnGraph protobuf message to populate.
    Returns:
      True if successful.
    """
    raw = self._get_value(GRAPH_KEY)
    if raw is None:
      return False
    graph_proto.ParseFromString(raw)
    return True

  def get_pano(self, pano_id, pano_proto):
    """Read a Pano protobuf from the database.

    Args:
      pano_id: string panorama ID.
      pano_proto: a Pano protobuf message to populate.
    Returns:
      True if successful.
    """
    key = pano_id.encode("utf-8") if isinstance(pano_id, str) else pano_id
    raw = self._get_value(key)
    if raw is None:
      return False
    pano_proto.ParseFromString(raw)
    return True

  def close(self):
    """Close the database handle, releasing the LevelDB directory lock."""
    close = getattr(self._db, "close", None)
    if close is not None:
      close()

  def _get_value(self, key):
    """Get a raw byte value from LevelDB."""
    leveldb = _get_leveldb()
    try:
      # plyvel returns bytes or None
      val = self._db.get(key)
      if val is not None:
        return val
    except Exception as e:
      logging.error("LevelDB get error for key %r: %s", key, e)
    return None
