"""
pyalicevision — Apple-Silicon Python shim for the C++-bound `pyalicevision`
module that ships with upstream AliceVision.

The upstream module is a pybind11 wrapper over AliceVision's C++ SfMData,
HDR and image libraries. On the Mac port we have not yet built those
bindings (Phase 13 work item). The Meshroom node descriptors import
`pyalicevision` at parse time purely to obtain chunk-sizing callables
(see `parallelization` submodule), and at template-load time a small
number of HDR descriptors call `update()` hooks that import `hdr` and
`sfmData` to compute initial state.

This shim provides:

  * `parallelization`  — pure-Python re-implementation that parses the
                         SfMData JSON file directly. Sufficient for
                         FeatureExtraction, FeatureMatching, DepthMap,
                         RelativePoseEstimating, ImageProcessing, …

  * `hdr`              — minimal pure-Python stubs (`vectorli`,
                         `LuminanceInfo`, `estimateGroups`) so the HDR
                         descriptors' template-load `update()` hooks
                         complete cleanly. Bracket detection is disabled
                         (returns 0 brackets) until the C++ bindings
                         are built. Affected templates that load again
                         after this shim: `hdrFusion.mg`, `panoramaHdr.mg`,
                         `panoramaFisheyeHdr.mg`.

  * `sfmData`          — `ExposureSetting` dataclass for HDR `getExposure`
                         load-time use. Other attributes (`SfMData`,
                         `Views`, `CameraPose`, ...) raise a clear
                         NotImplementedError on access; those are only
                         reached at compute time by `SfMFilter` etc.

  * `sfmDataIO`        — `load`/`save` raise NotImplementedError; same
                         compute-time-only invariant as `sfmData`.

The shim must remain dependency-free (stdlib only) so it can be
discovered before any C++ extension build.
"""

# Phase 13 — native C++ SWIG bindings, when present, take precedence.
#
# The CMake build stages real bindings (built from upstream's .i files)
# at `<repo>/build/pyalicevision_native/<name>.py` + `_<name>.so`. We
# discover that directory at import time and prepend it to this package's
# __path__, so `from pyalicevision import hdr` resolves to the native
# binding when built and falls back to the pure-Python stubs in this
# directory otherwise.
#
# Honors PYALICEVISION_NATIVE_DIR env var (compile-only artifact path
# override — useful for tests and for the bundled .app where the build
# tree isn't shipped).

import os as _os
from pathlib import Path as _Path

def _discover_native_dir():
    override = _os.environ.get("PYALICEVISION_NATIVE_DIR")
    if override and _Path(override).is_dir():
        return _Path(override)
    # Default: <repo>/build/pyalicevision_native/
    here = _Path(__file__).resolve().parent
    # src/python_shim/pyalicevision/ -> repo root is 3 levels up
    for candidate in (here.parent.parent.parent / "build" / "pyalicevision_native",):
        if candidate.is_dir() and any(candidate.glob("_*.so")):
            return candidate
    return None

_native_dir = _discover_native_dir()
if _native_dir is not None:
    # Prepend so SWIG-generated <name>.py wins over our stub <name>.py.
    __path__ = [str(_native_dir)] + list(__path__)

# Note: do NOT eagerly import sub-modules — keep the import cheap so the
# Meshroom node loader (which imports every descriptor at startup) does
# not pay the cost of building stub objects unless a node actually uses
# them. The submodules (parallelization, hdr, sfmData, sfmDataIO) are
# loaded on demand via `from pyalicevision import <name>`.

__version__ = "0.3.0-mac-shim+native"

__all__ = ["parallelization", "hdr", "sfmData", "sfmDataIO"]
