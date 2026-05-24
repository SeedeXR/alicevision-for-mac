"""Regression tests for the native C++ pyalicevision SWIG bindings (Phase 13).

Three modules are built from upstream's .i interface files:
  - pyalicevision.hdr     — bracket detection (LdrToHdr* descriptors)
  - pyalicevision.sfmData — SfMData, Views, ExposureSetting, ...
  - pyalicevision.sfmDataIO — load/save with ESfMData flags

Gating: tests run only when build/pyalicevision_native/ contains the
expected _*.so files. They're skipped (not failed) if AV_BUILD_PYALICEVISION
was OFF at configure time. That way the suite stays green for users who
opt out of the SWIG binding build.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NATIVE_DIR = REPO_ROOT / "build" / "pyalicevision_native"

_HAS_NATIVE = NATIVE_DIR.is_dir() and any(NATIVE_DIR.glob("_*.so"))

skip_if_no_native = pytest.mark.skipif(
    not _HAS_NATIVE,
    reason="native pyalicevision SWIG bindings not built (cmake -DAV_BUILD_PYALICEVISION=ON)",
)


@pytest.fixture(autouse=True)
def _ensure_shim_on_path():
    """Add src/python_shim/ to sys.path for each test, then clean up."""
    shim = str(REPO_ROOT / "src" / "python_shim")
    added = shim not in sys.path
    if added:
        sys.path.insert(0, shim)
    # Drop any cached pyalicevision so __init__ re-runs and re-discovers
    # the native dir for this test.
    for name in list(sys.modules):
        if name.startswith("pyalicevision"):
            del sys.modules[name]
    yield
    if added:
        sys.path.remove(shim)


@skip_if_no_native
def test_hdr_module_is_native_swig_binding():
    """pyalicevision.hdr loads from build/pyalicevision_native/, not the stub."""
    from pyalicevision import hdr
    assert "pyalicevision_native" in hdr.__file__, (
        f"expected native binding, got stub at {hdr.__file__}"
    )
    # The native binding exposes upstream's C++ enums; the stub does not.
    assert hasattr(hdr, "ECalibrationMethod_DEBEVEC")
    assert hasattr(hdr, "ECalibrationMethod_LINEAR")


@skip_if_no_native
def test_hdr_estimategroups_runs_real_cpp_code():
    """vectorli + LuminanceInfo + estimateGroups round-trip via C++."""
    from pyalicevision import hdr
    inputs = hdr.vectorli()
    inputs.append(hdr.LuminanceInfo(1, "/tmp/a.jpg", 0.01))
    inputs.append(hdr.LuminanceInfo(2, "/tmp/b.jpg", 0.02))
    assert len(inputs) == 2
    groups = hdr.estimateGroups(inputs)
    # Output type is a SWIG-wrapped vector< vector<IndexT> >.
    assert type(groups).__name__ == "vvectori"
    # len() must work for the descriptor's `if len(obj) == 0:` branch.
    assert isinstance(len(groups), int)


@skip_if_no_native
def test_sfmdata_module_is_native_and_supports_real_constructors():
    """SfMData + ExposureSetting come from the native binding."""
    from pyalicevision import sfmData
    assert "pyalicevision_native" in sfmData.__file__
    sd = sfmData.SfMData()
    # Empty SfMData has zero views — getViews() returns a SWIG-wrapped
    # std::map that behaves dict-like (supports len() and iteration).
    assert len(sd.getViews()) == 0
    # ExposureSetting math: exposure = shutterSpeed / fnumber^2.
    es = sfmData.ExposureSetting(1.0 / 100, 4.0, 100.0)
    assert es.getExposure() == pytest.approx(0.01 / 16.0)


@skip_if_no_native
def test_sfmdataio_module_is_native_and_exposes_flags():
    """sfmDataIO exposes the ESfMData bitmask the descriptors use."""
    from pyalicevision import sfmDataIO
    assert "pyalicevision_native" in sfmDataIO.__file__
    # ESfMData::ALL is a high bitmask covering all sections; the
    # descriptors call `load(..., sfmDataIO.ALL)`. Exact value 2047
    # matches upstream's enum at this AliceVision version.
    assert sfmDataIO.ALL > 0
    assert callable(sfmDataIO.load)
    assert callable(sfmDataIO.save)


def test_fallback_mode_when_no_native_dir(tmp_path, monkeypatch):
    """When native dir is absent, the stub .py files load instead."""
    # Force the __init__'s _discover_native_dir() to find nothing.
    monkeypatch.setenv("PYALICEVISION_NATIVE_DIR", str(tmp_path / "nope"))
    # Also relocate the package to a tmp dir so the default ../../../build
    # heuristic also fails.
    pkg_src = REPO_ROOT / "src" / "python_shim" / "pyalicevision"
    pkg_dst = tmp_path / "shim_test" / "pyalicevision"
    pkg_dst.parent.mkdir()
    import shutil
    shutil.copytree(pkg_src, pkg_dst)
    # Drop __pycache__ to avoid loading the original location's bytecode.
    for cache in pkg_dst.rglob("__pycache__"):
        shutil.rmtree(cache)
    monkeypatch.syspath_prepend(str(pkg_dst.parent))
    # Wipe import cache.
    for name in list(sys.modules):
        if name.startswith("pyalicevision"):
            del sys.modules[name]
    from pyalicevision import hdr  # noqa: PLC0415
    assert "shim_test" in hdr.__file__, f"expected stub, got {hdr.__file__}"
    # Stub-only attributes
    assert callable(hdr.vectorli)
    assert hdr.vectorli() == []
    # Stub returns empty list (load-only mode)
    assert hdr.estimateGroups([]) == []
