#!/usr/bin/env bash
#
# scripts/run_meshroom.sh — drive the Mac Meshroom build with the right env.
#
# Phase 11 (S42) Meshroom integration: invokes our 12 native Apple Silicon
# Metal aliceVision_* binaries through Meshroom's Python pipeline runner.
#
# Usage:
#   scripts/run_meshroom.sh python bin/meshroom_batch \
#       -i path/to/images -o path/to/out -p photogrammetryLegacy
#
# Anything after the script name is exec'd in the configured environment.
#
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate the Meshroom venv (PySide6 + psutil + pyseq + …).
# shellcheck disable=SC1091
source "$ROOT/meshroom-venv/bin/activate"

# AliceVision runtime — binaries, sensor DB, OCIO config, dylib fallback.
export ALICEVISION_ROOT="$ROOT/build/alicevision_root"
export ALICEVISION_BIN_PATH="$ROOT/build"
export ALICEVISION_SENSOR_DB="$ROOT/build/alicevision_root/share/aliceVision/cameraSensors.db"
export ALICEVISION_OCIO="$ROOT/build/alicevision_root/share/aliceVision/config.ocio"
export ALICEVISION_LIBPATH="/opt/homebrew/lib"
export PATH="$ROOT/build:$PATH"
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib"

# Meshroom plugin discovery — point at our local node-descriptor copy.
# It carries the Darwin patches under patches/alicevision-meshroom/.
export MESHROOM_NODES_PATH="$ROOT/meshroom-mac/nodes"
export MESHROOM_PIPELINE_TEMPLATES_PATH="$ROOT/meshroom-mac/nodes"

# PYTHONPATH:
#   meshroom-mac/         → the Meshroom Python package (patched).
#   src/python_shim/      → pure-Python pyalicevision parallelization shim.
export PYTHONPATH="$ROOT/meshroom-mac:$ROOT/src/python_shim"

# Default cwd inside meshroom-mac so bin/meshroom_batch's relative
# imports work.
cd "$ROOT/meshroom-mac"

exec "$@"
