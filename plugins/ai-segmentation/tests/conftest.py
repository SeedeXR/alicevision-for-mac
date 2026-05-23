"""Pytest conftest for plugin-local tests.

Adds `plugins/ai-segmentation/python` to sys.path so `import segmentation`
resolves to the in-plugin package. Mirrors the `isolated_u2net_home`
fixture from the repo-wide conftest so segmentation-session tests work
in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_PATH = PLUGIN_ROOT / "python"
REPO_ROOT = PLUGIN_ROOT.parent.parent


def pytest_configure(config) -> None:  # noqa: ARG001 — pytest hook signature
    """Make the plugin's `segmentation` package importable."""
    s = str(PACKAGE_PATH)
    if PACKAGE_PATH.is_dir() and s not in sys.path:
        sys.path.insert(0, s)


@pytest.fixture(autouse=True)
def _reset_session_cache():
    """Clear the in-process model cache between tests so one test's
    `_resolve_package_path` patches don't bleed into another."""
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
