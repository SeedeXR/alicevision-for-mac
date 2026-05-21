#!/usr/bin/env bash
#
# meshroom-mac/start.sh — launch the Qt UI with the full Apple-Silicon-port
# environment (matches scripts/run_meshroom.sh for the CLI). Without this
# setup, every aliceVision node registers as UnknownNodeType and the UI
# is non-functional.
#
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESHROOM_ROOT="$ROOT/meshroom-mac"

# Activate the Meshroom venv (PySide6 + psutil + pyseq + ...).
# shellcheck disable=SC1091
if [ -f "$ROOT/meshroom-venv/bin/activate" ]; then
    source "$ROOT/meshroom-venv/bin/activate"
fi

# AliceVision runtime — native arm64 Metal binaries, sensor DB, OCIO, dylib paths.
export ALICEVISION_ROOT="$ROOT/build/alicevision_root"
export ALICEVISION_BIN_PATH="$ROOT/build"
export ALICEVISION_SENSOR_DB="$ROOT/build/alicevision_root/share/aliceVision/cameraSensors.db"
export ALICEVISION_OCIO="$ROOT/build/alicevision_root/share/aliceVision/config.ocio"
export ALICEVISION_LIBPATH="/opt/homebrew/lib"
export PATH="$ROOT/build:$PATH"

# macOS needs DYLD_*, not LD_LIBRARY_PATH.
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"

# Meshroom plugin discovery — node descriptors with Darwin patches applied.
export MESHROOM_NODES_PATH="$MESHROOM_ROOT/nodes"
export MESHROOM_PIPELINE_TEMPLATES_PATH="$MESHROOM_ROOT/nodes"

# PYTHONPATH:
#   meshroom-mac/         → patched Meshroom Python package.
#   src/python_shim/      → pyalicevision parallelization shim.
export PYTHONPATH="$MESHROOM_ROOT:$ROOT/src/python_shim:${PYTHONPATH:-}"

# Launch the Qt UI.
exec python3 "$MESHROOM_ROOT/meshroom/ui" "$@"
