"""
Pytest conftest for the Mac-native segmentation node tests.

Adds the in-repo `meshroom-mac/` directory to sys.path so `import
segmentation` and `import meshroom.core.desc` resolve to the
project-local copies — not anything installed globally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MESHROOM_MAC = REPO_ROOT / "meshroom-mac"


def pytest_configure(config) -> None:  # noqa: ARG001 — pytest hook signature
    """Make in-repo packages importable before any test runs."""
    for p in (MESHROOM_MAC, REPO_ROOT / "src" / "python_shim"):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def isolated_u2net_home(tmp_path, monkeypatch) -> Path:
    """Point U2NET_HOME at a fresh tmp dir for each test."""
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    # Clear the segmentation session cache so tests don't leak state.
    try:
        from segmentation.session import clear_session_cache
        clear_session_cache()
    except Exception:
        pass
    return tmp_path
