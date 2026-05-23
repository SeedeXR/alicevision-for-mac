"""
Unit tests for the CoreML-only segmentation helpers
(`plugins/ai-segmentation/python/segmentation/`).

The heavy "load + predict on a real BiRefNet mlpackage" path requires
the 90 MB lite or 447 MB general `.mlpackage` to be staged at
`<repo>/ai-models/`, so it is gated behind `RUN_SEG_E2E=1`. Default
`pytest` runs cover only the cheap plumbing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
# Import sanity
# --------------------------------------------------------------------------- #

def test_segmentation_package_importable():
    import segmentation  # noqa: F401
    from segmentation import AI_MODELS_DIR, ensure_models_dir
    from segmentation.session import (
        BiRefNetCoreMLSession, VARIANT_PACKAGES, INPUT_HW,
        get_session, clear_session_cache,
    )
    from segmentation.utils import (
        log_compute_backend, coremltools_version,
    )

    assert AI_MODELS_DIR.name == "ai-models"
    assert callable(ensure_models_dir)
    assert callable(get_session)
    assert callable(clear_session_cache)
    assert callable(log_compute_backend)
    assert callable(coremltools_version)
    assert set(VARIANT_PACKAGES) == {"birefnet-lite", "birefnet-general"}
    assert INPUT_HW == 1024
    assert BiRefNetCoreMLSession is not None


# --------------------------------------------------------------------------- #
# Models-dir plumbing
# --------------------------------------------------------------------------- #

def test_ensure_models_dir_defaults_to_repo_path(monkeypatch, tmp_path):
    """With no override, ensure_models_dir() returns <repo>/ai-models."""
    monkeypatch.delenv("AV_AI_MODELS_DIR", raising=False)
    monkeypatch.delenv("U2NET_HOME", raising=False)
    from segmentation import ensure_models_dir, AI_MODELS_DIR

    home = ensure_models_dir()
    assert home == AI_MODELS_DIR.resolve()


def test_ensure_models_dir_honours_av_override(monkeypatch, tmp_path):
    """AV_AI_MODELS_DIR takes precedence."""
    monkeypatch.setenv("AV_AI_MODELS_DIR", str(tmp_path))
    monkeypatch.delenv("U2NET_HOME", raising=False)
    from segmentation import ensure_models_dir

    home = ensure_models_dir()
    assert home == tmp_path.resolve()


def test_ensure_models_dir_honours_legacy_u2net_home(monkeypatch, tmp_path):
    """U2NET_HOME (legacy rembg variable) is also honoured."""
    monkeypatch.delenv("AV_AI_MODELS_DIR", raising=False)
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    from segmentation import ensure_models_dir

    home = ensure_models_dir()
    assert home == tmp_path.resolve()


# --------------------------------------------------------------------------- #
# coremltools availability — the critical Apple Silicon check
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform != "darwin", reason="CoreML is macOS-only")
def test_coremltools_importable():
    """The whole port assumes coremltools is in the meshroom-venv."""
    from segmentation.utils import coremltools_version
    version = coremltools_version()
    assert version, (
        "coremltools is not importable — install it in meshroom-venv "
        "before running this node."
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="CoreML is macOS-only")
def test_log_compute_backend_reports_coreml(caplog):
    """log_compute_backend must announce CoreML so users can verify dispatch."""
    import logging
    from segmentation.utils import log_compute_backend

    caplog.set_level(logging.INFO)
    info = log_compute_backend(logging.getLogger("seg-test"))
    assert info["platform"] == "Darwin"
    assert info["compute_target"] in {"CoreML", "UNAVAILABLE"}
    if info["compute_target"] == "CoreML":
        assert any(
            "CoreML (CPU + GPU dispatch" in r.message
            for r in caplog.records
        )


# --------------------------------------------------------------------------- #
# Variant -> mlpackage resolution
# --------------------------------------------------------------------------- #

def test_resolve_package_path_unknown_variant_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AV_AI_MODELS_DIR", str(tmp_path))
    from segmentation.session import _resolve_package_path

    with pytest.raises(ValueError, match="unknown BiRefNet variant"):
        _resolve_package_path("birefnet-massive")


def test_resolve_package_path_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AV_AI_MODELS_DIR", str(tmp_path))
    from segmentation.session import _resolve_package_path

    with pytest.raises(FileNotFoundError, match="BiRefNet CoreML package missing"):
        _resolve_package_path("birefnet-lite")


def test_resolve_package_path_finds_staged_package(tmp_path, monkeypatch):
    """A directory with a Manifest.json passes the existence check."""
    monkeypatch.setenv("AV_AI_MODELS_DIR", str(tmp_path))
    fake = tmp_path / "BiRefNet_lite.mlpackage"
    fake.mkdir()
    (fake / "Manifest.json").write_text("{}", encoding="utf-8")

    from segmentation.session import _resolve_package_path

    found = _resolve_package_path("birefnet-lite")
    assert found == fake.resolve()


# --------------------------------------------------------------------------- #
# Optional end-to-end on the Monstree dataset
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    os.environ.get("RUN_SEG_E2E") != "1" or sys.platform != "darwin",
    reason="set RUN_SEG_E2E=1 on macOS to run heavy E2E with staged mlpackage",
)
def test_e2e_birefnet_lite_on_monstree(tmp_path):
    """End-to-end smoke: load lite mlpackage, run on a Monstree image."""
    import numpy as np
    from PIL import Image
    from segmentation.session import get_session, clear_session_cache

    clear_session_cache()  # don't reuse a session leaked from another test

    repo_root = Path(__file__).resolve().parents[3]
    img_dir = repo_root / "dataset_monstree" / "mini3"
    images = sorted(img_dir.glob("*.JPG"))
    assert images, f"no Monstree images at {img_dir}"

    sess = get_session("birefnet-lite")
    src = np.asarray(Image.open(images[0]).convert("RGB"), dtype=np.uint8)
    mask = sess.predict(src)
    assert mask.shape == src.shape[:2]
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0
