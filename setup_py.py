"""Setup helper: install dependencies and generate protobuf.

Usage:
  python setup_py.py install    # Install pip dependencies
  python setup_py.py proto      # Generate protobuf module
  python setup_py.py check      # Check all imports work
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import sys
import os

DEPENDENCIES = [
    "plyvel",       # LevelDB Python binding
    "numpy",        # Numerical computing
    "Pillow",       # JPEG decoding
    "protobuf",     # Protobuf runtime
]

OPTIONAL_DEPENDENCIES = [
    "opencv-python",  # Faster image projection (cv2.remap)
    "absl-py",        # Logging (used by original code)
    "six",            # Python 2/3 compat (used by original code)
    "inflection",     # String utilities (used by original code)
]


def install():
  print("Installing core dependencies...")
  for dep in DEPENDENCIES:
    print("  Installing %s..." % dep)
    subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

  print("\nInstalling optional dependencies...")
  for dep in OPTIONAL_DEPENDENCIES:
    try:
      subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
      print("  Installed %s" % dep)
    except subprocess.CalledProcessError:
      print("  Skipped %s (optional)" % dep)

  print("\nAll dependencies installed.")


def gen_proto():
  """Generate the protobuf Python module."""
  proto_dir = os.path.join(os.path.dirname(__file__), "streetlearn", "proto")
  proto_file = os.path.join(proto_dir, "streetlearn.proto")

  # Try protoc
  try:
    subprocess.check_call([
        "protoc",
        "--proto_path=" + proto_dir,
        "--python_out=" + proto_dir,
        proto_file
    ])
    print("Generated streetlearn_pb2.py via protoc")
    return True
  except (subprocess.CalledProcessError, OSError):
    pass

  # Try grpc_tools.protoc
  try:
    from grpc_tools import protoc as grpc_protoc
    grpc_protoc.main([
        "grpc_tools.protoc",
        "--proto_path=" + proto_dir,
        "--python_out=" + proto_dir,
        proto_file
    ])
    print("Generated streetlearn_pb2.py via grpcio-tools")
    return True
  except ImportError:
    print("grpcio-tools not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "grpcio-tools"])
    from grpc_tools import protoc as grpc_protoc
    grpc_protoc.main([
        "grpc_tools.protoc",
        "--proto_path=" + proto_dir,
        "--python_out=" + proto_dir,
        proto_file
    ])
    print("Generated streetlearn_pb2.py via grpcio-tools")
    return True


def check_imports():
  """Verify all modules can be imported."""
  print("Checking imports...")
  modules = [
      "numpy",
      "PIL",
      "google.protobuf",
  ]
  try:
    import plyvel
    print("  plyvel: OK")
  except ImportError:
    print("  plyvel: MISSING (pip install plyvel)")

  try:
    import cv2
    print("  cv2: OK")
  except ImportError:
    print("  cv2: MISSING (optional, pip install opencv-python)")

  for mod in modules:
    try:
      __import__(mod)
      print("  %s: OK" % mod)
    except ImportError:
      print("  %s: MISSING" % mod)

  # Check protobuf module
  pb2_path = os.path.join(
      os.path.dirname(__file__), "streetlearn", "proto", "streetlearn_pb2.py")
  if os.path.exists(pb2_path):
    print("  streetlearn_pb2.py: EXISTS")
  else:
    print("  streetlearn_pb2.py: MISSING (run: python setup_py.py proto)")

  # Check engine modules
  sys.path.insert(0, os.path.dirname(__file__))
  engine_modules = [
      "streetlearn.engine.python.color",
      "streetlearn.engine.python.math_util",
      "streetlearn.engine.python.pano_calculations",
      "streetlearn.engine.python.dataset",
      "streetlearn.engine.python.metadata_cache",
      "streetlearn.engine.python.pano_graph",
      "streetlearn.engine.python.pano_projection",
      "streetlearn.engine.python.pano_renderer",
      "streetlearn.engine.python.graph_renderer",
      "streetlearn.engine.python.streetlearn_engine",
  ]
  for mod in engine_modules:
    try:
      __import__(mod)
      print("  %s: OK" % mod)
    except Exception as e:
      print("  %s: ERROR (%s)" % (mod, e))


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python setup_py.py [install|proto|check]")
    sys.exit(1)

  cmd = sys.argv[1]
  if cmd == "install":
    install()
  elif cmd == "proto":
    gen_proto()
  elif cmd == "check":
    check_imports()
  else:
    print("Unknown command: %s" % cmd)
    print("Usage: python setup_py.py [install|proto|check]")
