"""
Unit tests for the Mac-native segmentation helpers
(`meshroom-mac/segmentation/`).

The heavy end-to-end "run a model on an image" path requires the BiRefNet
ONNX file (~973 MB) and 1-2 GB of RAM, so it is gated behind the
`RUN_SEG_E2E=1` env var. Default `pytest` runs cover only the cheap
plumbing: path resolution, provider selection, CoreML conversion cache
existence checks.
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
    from segmentation import AI_MODELS_DIR, ensure_u2net_home
    from segmentation.session import (
        get_session, clear_session_cache, PROVIDERS_PREFERRED,
    )
    from segmentation.utils import log_compute_backend, available_onnx_providers
    from segmentation.convert_to_coreml import get_coreml_model

    assert PROVIDERS_PREFERRED[0] == "CoreMLExecutionProvider"
    assert AI_MODELS_DIR.name == "ai-models"
    assert callable(get_session)
    assert callable(clear_session_cache)
    assert callable(get_coreml_model)
    assert callable(log_compute_backend)
    assert callable(available_onnx_providers)


# --------------------------------------------------------------------------- #
# U2NET_HOME plumbing
# --------------------------------------------------------------------------- #

def test_ensure_u2net_home_sets_env(isolated_u2net_home, monkeypatch):
    # Clear any prior value
    monkeypatch.delenv("U2NET_HOME", raising=False)
    from segmentation import ensure_u2net_home, AI_MODELS_DIR

    home = ensure_u2net_home()
    assert os.environ["U2NET_HOME"] == str(AI_MODELS_DIR)
    assert home == AI_MODELS_DIR


def test_ensure_u2net_home_honours_existing(isolated_u2net_home):
    # Fixture already set U2NET_HOME to the tmp dir
    from segmentation import ensure_u2net_home

    home = ensure_u2net_home()
    assert str(home) == os.environ["U2NET_HOME"]
    assert home == isolated_u2net_home


# --------------------------------------------------------------------------- #
# CoreML EP availability — the critical Apple Silicon check
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform != "darwin", reason="CoreML is macOS-only")
def test_coreml_ep_available():
    """The whole port assumes the CoreML EP is in the onnxruntime wheel."""
    from segmentation.utils import available_onnx_providers
    providers = available_onnx_providers()
    assert "CoreMLExecutionProvider" in providers, (
        f"CoreML EP missing — wrong onnxruntime wheel. Got: {providers}"
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="CoreML is macOS-only")
def test_log_compute_backend_reports_coreml(caplog):
    """log_compute_backend must announce CoreML so users can verify dispatch."""
    import logging
    from segmentation.utils import log_compute_backend

    caplog.set_level(logging.INFO)
    info = log_compute_backend(logging.getLogger("seg-test"))
    assert info["platform"] == "Darwin"
    assert info["compute_target"] in {"CoreML", "CPU"}
    if info["compute_target"] == "CoreML":
        assert any("CoreML (CPU+GPU+ANE)" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Session loader — CoreML provider selection
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform != "darwin", reason="CoreML is macOS-only")
def test_get_session_selects_coreml(isolated_u2net_home):
    """
    Verify the rembg session is built with the CoreML EP first when
    available. We don't actually load the heavy ONNX file here; we just
    check the provider filtering logic.
    """
    from segmentation.session import _filter_providers, PROVIDERS_PREFERRED

    filtered = _filter_providers(PROVIDERS_PREFERRED)
    assert filtered[0] == "CoreMLExecutionProvider", (
        f"CoreML must be first; got {filtered}"
    )
    assert "CPUExecutionProvider" in filtered


def test_alias_resolution():
    """`birefnet-lite` -> rembg internal `birefnet-general-lite`."""
    from segmentation.session import _resolve_session_name
    assert _resolve_session_name("birefnet-lite") == "birefnet-general-lite"
    assert _resolve_session_name("birefnet-general") == "birefnet-general"
    assert _resolve_session_name("birefnet-dis") == "birefnet-dis"


# --------------------------------------------------------------------------- #
# ONNX_FORCE_CPU escape hatch (S53 follow-up)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform != "darwin", reason="CoreML is macOS-only")
def test_onnx_force_cpu_pins_to_cpu_provider(monkeypatch):
    """`ONNX_FORCE_CPU=1` must short-circuit provider selection to CPU only."""
    from segmentation.session import _filter_providers, PROVIDERS_PREFERRED

    monkeypatch.setenv("ONNX_FORCE_CPU", "1")
    filtered = _filter_providers(PROVIDERS_PREFERRED)
    assert filtered == ["CPUExecutionProvider"], (
        f"ONNX_FORCE_CPU=1 should pin to CPU only; got {filtered}"
    )


def test_onnx_force_cpu_unset_keeps_default(monkeypatch):
    """With ONNX_FORCE_CPU unset/0, provider order stays preference-driven."""
    from segmentation.session import _filter_providers, PROVIDERS_PREFERRED

    monkeypatch.delenv("ONNX_FORCE_CPU", raising=False)
    filtered = _filter_providers(PROVIDERS_PREFERRED)
    # On macOS this should include CoreML; on non-macOS just CPU. Either
    # way the env var should not have collapsed it to CPU-only on macOS.
    if sys.platform == "darwin":
        assert "CoreMLExecutionProvider" in filtered

    monkeypatch.setenv("ONNX_FORCE_CPU", "0")
    filtered2 = _filter_providers(PROVIDERS_PREFERRED)
    if sys.platform == "darwin":
        assert "CoreMLExecutionProvider" in filtered2


# --------------------------------------------------------------------------- #
# CoreML conversion — cache plumbing (does NOT run a real conversion)
# --------------------------------------------------------------------------- #

def test_resolve_onnx_path_missing_raises(isolated_u2net_home):
    from segmentation.convert_to_coreml import _resolve_onnx_path
    with pytest.raises(FileNotFoundError):
        _resolve_onnx_path("definitely-not-a-real-model-name")


def test_resolve_onnx_path_finds_staged_file(isolated_u2net_home):
    fake = isolated_u2net_home / "fake-model.onnx"
    fake.write_bytes(b"\x00" * 16)
    from segmentation.convert_to_coreml import _resolve_onnx_path, _mlpackage_path

    found = _resolve_onnx_path("fake-model")
    assert found == fake.resolve()
    assert _mlpackage_path(found).name == "fake-model.mlpackage"


# --------------------------------------------------------------------------- #
# Optional end-to-end on the Monstree dataset
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    os.environ.get("RUN_SEG_E2E") != "1",
    reason="set RUN_SEG_E2E=1 to run heavy E2E with downloaded model",
)
def test_e2e_birefnet_on_monstree(tmp_path, repo_root):
    """End-to-end smoke: load model, run on a Monstree image, save mask."""
    from segmentation.session import get_session
    from rembg import remove
    from PIL import Image

    img_dir = repo_root / "dataset_monstree" / "mini3"
    images = sorted(img_dir.glob("*.JPG"))
    assert images, f"no Monstree images at {img_dir}"

    sess = get_session("birefnet-general")
    src = Image.open(images[0])
    result = remove(src, session=sess)
    out = tmp_path / "mask.png"
    result.split()[-1].save(out)
    assert out.exists() and out.stat().st_size > 0
