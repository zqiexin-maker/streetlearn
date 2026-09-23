"""Navigation demo for pure Python StreetLearn.

Usage:
  python -m streetlearn.tools.run_nav --dataset_path=/path/to/dataset

Demonstrates:
  1. Engine creation + graph building
  2. Random walk navigation
  3. Observation rendering (view image + graph image)
  4. Metadata query (GPS coords, neighbors, distance/bearing)
  5. Courier game setup with goal-directed navigation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
import time
import numpy as np

from streetlearn.engine.python import streetlearn_engine
from streetlearn.engine.python import color


def run_engine_test(args):
  """Test the raw engine: build graph, navigate, render."""
  print("=" * 60)
  print("StreetLearn Pure Python Engine - Navigation Demo")
  print("=" * 60)

  # 1. Create engine
  print("\n[1] Creating engine from dataset: %s" % args.dataset_path)
  engine = streetlearn_engine.StreetLearnEngine.Create(
      args.dataset_path,
      width=args.width,
      height=args.height,
      graph_width=args.graph_width,
      graph_height=args.graph_height,
      field_of_view=args.field_of_view,
      min_graph_depth=args.min_graph_depth,
      max_graph_depth=args.max_graph_depth,
      max_cache_size=args.max_cache_size)

  if engine is None:
    print("ERROR: Failed to create engine. Check dataset path and dependencies.")
    print("  - Is plyvel installed? (pip install plyvel)")
    print("  - Is streetlearn_pb2.py generated? (python -m streetlearn.proto.gen_pb2)")
    sys.exit(1)

  print("    Engine created.")

  # 2. Init episode + build graph
  print("\n[2] Building graph...")
  engine.InitEpisode(0, 42)
  root_id = engine.BuildRandomGraph()
  if root_id is None:
    print("ERROR: Failed to build graph.")
    sys.exit(1)

  graph = engine.GetGraph()
  print("    Root pano: %s" % root_id)
  print("    Graph size: %d panos" % len(graph))

  # 3. Init graph renderer
  obs_color = color.Color(0.5, 0.5, 1.0)
  engine.InitGraphRenderer(obs_color, {}, False)
  engine.SetZoom(32)

  # 4. Navigation loop
  print("\n[3] Running %d-step navigation..." % args.num_steps)
  obs_buffer = np.empty(3 * args.width * args.height, dtype=np.uint8)
  graph_buffer = np.empty(3 * args.graph_width * args.graph_height, dtype=np.uint8)

  for step in range(args.num_steps):
    t0 = time.time()

    # Alternate actions: forward, rotate left, rotate right
    if step % 3 == 0:
      new_id = engine.MoveToNextPano()
      action = "FORWARD"
      if new_id:
        print("  [%3d] %s -> moved to %s" % (step, action, new_id))
      else:
        print("  [%3d] %s -> blocked (no neighbor in bearing)" % (step, action))
    elif step % 3 == 1:
      engine.RotateObserver(22.5, 0)
      action = "ROTATE +22.5"
      print("  [%3d] %s (yaw=%.1f)" % (step, action, engine.GetYaw()))
    else:
      engine.RotateObserver(-22.5, 0)
      action = "ROTATE -22.5"
      print("  [%3d] %s (yaw=%.1f)" % (step, action, engine.GetYaw()))

    # Render view
    engine.RenderObservation(obs_buffer)
    engine.DrawGraph({}, graph_buffer)

    # Query state
    occupancy = engine.GetNeighborOccupancy(8)
    yaw = engine.GetYaw()
    pitch = engine.GetPitch()

    elapsed = (time.time() - t0) * 1000
    print("         view: min=%d max=%d | neighbors: %s | %.1fms" %
          (obs_buffer.min(), obs_buffer.max(), occupancy, elapsed))

  # 5. Metadata
  print("\n[4] Metadata query...")
  meta = engine.GetMetadata(root_id)
  if meta:
    print("    Pano ID: %s" % meta.pano.id)
    print("    GPS: (%.6f, %.6f)" % (meta.pano.coords.lat, meta.pano.coords.lng))
    print("    Altitude: %.1f m" % meta.pano.alt)
    print("    Heading: %.1f deg" % meta.pano.heading_deg)
    print("    Neighbors: %d" % len(meta.neighbors))
    print("    Graph depth: %d" % meta.graph_depth)

  # 6. Distance/bearing
  if len(graph) >= 2:
    ids = list(graph.keys())[:2]
    dist = engine.GetPanoDistance(ids[0], ids[1])
    bearing = engine.GetPanoBearing(ids[0], ids[1])
    print("\n[5] Distance %s -> %s: %.1f m" % (ids[0], ids[1], dist))
    print("    Bearing %s -> %s: %.1f deg" % (ids[0], ids[1], bearing))

  print("\n" + "=" * 60)
  print("Engine test complete.")
  print("=" * 60)


def run_game_test(args):
  """Test the full game environment (StreetLearn + CourierGame)."""
  print("=" * 60)
  print("StreetLearn Game Environment - Courier Navigation")
  print("=" * 60)

  from streetlearn.python.environment import default_config
  from streetlearn.python.environment import courier_game
  from streetlearn.python.environment import streetlearn as sl_env

  config = {
      "dataset_path": args.dataset_path,
      "width": args.width,
      "height": args.height,
      "graph_width": args.graph_width,
      "graph_height": args.graph_height,
      "field_of_view": args.field_of_view,
      "min_graph_depth": args.min_graph_depth,
      "max_graph_depth": args.max_graph_depth,
      "max_cache_size": args.max_cache_size,
      "full_graph": False,
      "auto_reset": False,
      "frame_cap": 100,
      "observations": ["view_image", "graph_image", "yaw", "pitch",
                       "metadata", "target_metadata", "latlng",
                       "target_latlng", "neighbors"],
  }
  config = default_config.ApplyDefaults(config)

  game = courier_game.CourierGame(config)
  env = sl_env.StreetLearn(args.dataset_path, config, game)

  print("\n[1] Resetting environment...")
  env.reset()
  print("    Current pano: %s" % env.current_pano_id)
  print("    Goal: %s" % game.goal_id)

  print("\n[2] Running %d steps..." % args.num_steps)
  for step in range(args.num_steps):
    action_set = env.action_set()
    action = action_set[np.random.randint(len(action_set))]
    obs, reward, done, info = env.step(action)
    print("  [%3d] pano=%s reward=%.3f done=%s" %
          (step, env.current_pano_id, reward, done))
    if done:
      print("  Episode finished. Resetting...")
      env.reset()

  print("\n" + "=" * 60)
  print("Game test complete.")
  print("=" * 60)


def main():
  parser = argparse.ArgumentParser(
      description="StreetLearn pure Python navigation demo")
  parser.add_argument("--dataset_path", type=str, required=True,
                      help="Path to StreetLearn LevelDB dataset")
  parser.add_argument("--mode", type=str, default="engine",
                      choices=["engine", "game"],
                      help="engine: raw engine test | game: full game env")
  parser.add_argument("--width", type=int, default=320)
  parser.add_argument("--height", type=int, default=240)
  parser.add_argument("--graph_width", type=int, default=320)
  parser.add_argument("--graph_height", type=int, default=240)
  parser.add_argument("--field_of_view", type=int, default=60)
  parser.add_argument("--min_graph_depth", type=int, default=50)
  parser.add_argument("--max_graph_depth", type=int, default=100)
  parser.add_argument("--max_cache_size", type=int, default=500)
  parser.add_argument("--num_steps", type=int, default=20)
  args = parser.parse_args()

  logging.basicConfig(level=logging.WARNING)

  if args.mode == "engine":
    run_engine_test(args)
  else:
    run_game_test(args)


if __name__ == "__main__":
  main()
