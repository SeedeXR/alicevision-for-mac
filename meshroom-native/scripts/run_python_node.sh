#!/bin/bash
#
# run_python_node.sh — generic wrapper that lets the native SwiftUI
# Meshroom executor invoke Python-only nodes (e.g. SegmentationBiRefNet)
# the same way it spawns aliceVision_* binaries: as one Process with argv.
#
# Contract (called by Sources/App/GraphExecutor.swift):
#   run_python_node.sh --nodeType <NodeType> [--<attr> <value> ...]
#
# The first arg pair is the node identifier; the rest are the same
# `--<attrName> <renderedValue>` pairs the Swift executor emits for any
# Spec it finds in NodeBinary.swift.  Outputs (`--output <dir>`) come
# through the same channel.
#
# Layout assumption:
#   <repo-root>/
#     meshroom-native/scripts/run_python_node.sh   <- this file
#     meshroom-mac/                                <- Python Meshroom source
#     meshroom-venv/                               <- bundled Python venv
#     ai-models/                                   <- model cache (U2NET_HOME)
#
# Entry point: `meshroom.bin.node_run` (S52). Parses `--nodeType <T>`
# and the remaining `--<attr> <value>` pairs, imports
# `aliceVision.<T>`, synthesises a chunk, dispatches `processChunk`.

set -e

NODE_TYPE=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodeType)
            NODE_TYPE="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$NODE_TYPE" ]]; then
    echo "run_python_node.sh: missing required --nodeType <name>" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# scripts/ lives inside meshroom-native/, so two `..`s land us at the
# repo root that holds meshroom-mac/, meshroom-venv/, ai-models/.
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV_ACTIVATE="$ROOT/meshroom-venv/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
else
    echo "run_python_node.sh: warning — venv not found at $VENV_ACTIVATE" >&2
fi

export U2NET_HOME="${U2NET_HOME:-$ROOT/ai-models}"
export PYTHONPATH="$ROOT/meshroom-mac:$ROOT/src/python_shim:${PYTHONPATH:-}"

# Entry-point invocation.  When meshroom.bin.node_run lands, this just
# works.  Until then, the call fails with a clear ModuleNotFoundError
# which the Swift executor surfaces to the user.
exec python -m meshroom.bin.node_run --nodeType "$NODE_TYPE" "${ARGS[@]}"
