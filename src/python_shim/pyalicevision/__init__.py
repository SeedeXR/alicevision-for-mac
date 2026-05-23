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

# Note: do NOT eagerly import sub-modules — keep the import cheap so the
# Meshroom node loader (which imports every descriptor at startup) does
# not pay the cost of building stub objects unless a node actually uses
# them. The submodules (parallelization, hdr, sfmData, sfmDataIO) are
# loaded on demand via `from pyalicevision import <name>`.

__version__ = "0.2.0-mac-shim"

__all__ = ["parallelization", "hdr", "sfmData", "sfmDataIO"]
