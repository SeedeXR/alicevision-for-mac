"""
Pytest conftest for the repo-wide Python test suite.

Adds the in-repo `meshroom-mac/`, `src/python_shim/`, and
`plugins/ai-segmentation/python/` directories to `sys.path` so
`import segmentation` (CoreML helpers) and `import meshroom.core.desc`
(Meshroom framework) resolve to the project-local copies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MESHROOM_MAC = REPO_ROOT / "meshroom-mac"
AI_SEG_PYTHON = REPO_ROOT / "plugins" / "ai-segmentation" / "python"


def pytest_configure(config) -> None:  # noqa: ARG001 — pytest hook signature
    """Make in-repo packages importable before any test runs."""
    for p in (MESHROOM_MAC, REPO_ROOT / "src" / "python_shim", AI_SEG_PYTHON):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)


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
