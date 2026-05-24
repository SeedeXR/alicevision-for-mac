"""
Pytest conftest for the repo-wide Python test suite.

Adds the in-repo `meshroom-mac/`, `src/python_shim/`, and
`plugins/ai-segmentation/python/` directories to `sys.path` so
`import segmentation` (CoreML helpers) and `import meshroom.core.desc`
(Meshroom framework) resolve to the project-local copies.

Also invokes `meshroom.core.initNodes()` once at session start so any
test that deserializes a `.mg` template sees real typed Nodes instead
of `CompatibilityNode` stubs. Without this, descriptor-load regressions
(e.g. the Phase 3 ScenePreview-missing bug) silently pass tests that
only check for "graph loaded" — the graph loads, but every node is a
CompatibilityNode with no typed outputs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MESHROOM_MAC = REPO_ROOT / "meshroom-mac"
AI_SEG_PYTHON = REPO_ROOT / "plugins" / "ai-segmentation" / "python"
AI_SEG_NODES = REPO_ROOT / "plugins" / "ai-segmentation" / "nodes"
MESHROOM_NODES = REPO_ROOT / "meshroom-mac" / "nodes"


def pytest_configure(config) -> None:  # noqa: ARG001 — pytest hook signature
    """Make in-repo packages importable + register Meshroom nodes."""
    for p in (MESHROOM_MAC, REPO_ROOT / "src" / "python_shim", AI_SEG_PYTHON):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)

    # Set MESHROOM_NODES_PATH BEFORE importing meshroom.core so that the
    # plugin manager picks up our node descriptors when initNodes() runs.
    extra_paths = [str(p) for p in (MESHROOM_NODES, AI_SEG_NODES) if p.exists()]
    existing = os.environ.get("MESHROOM_NODES_PATH", "")
    combined = ":".join([p for p in extra_paths + existing.split(":") if p])
    if combined:
        os.environ["MESHROOM_NODES_PATH"] = combined

    # Initialize the node plugin manager. Wrapped in try/except so a
    # broken descriptor in one of the loaded folders doesn't tank the
    # entire test session — the individual tests will surface the issue
    # via their own assertions.
    try:
        import meshroom.core
        meshroom.core.initNodes()
    except Exception as exc:  # noqa: BLE001
        print(f"[conftest] meshroom.core.initNodes() failed: {exc}", file=sys.stderr)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _reset_segmentation_cache():
    """Drop the in-process CoreML session cache between tests so one
    test's monkeypatched `AV_AI_MODELS_DIR` doesn't leak into another."""
    try:
        from segmentation.session import clear_session_cache
        clear_session_cache()
    except Exception:
        pass
    yield
    try:
        from segmentation.session import clear_session_cache
        clear_session_cache()
    except Exception:
        pass
