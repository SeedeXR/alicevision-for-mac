"""
pyalicevision — Apple-Silicon Python shim for the C++-bound `pyalicevision`
module that ships with upstream AliceVision.

The upstream module is a pybind11 wrapper over AliceVision's C++ SfMData,
HDR and image libraries. On the Mac port we have not yet built those
bindings (Phase 13 work item). The Meshroom node descriptors import
`pyalicevision` at parse time purely to obtain chunk-sizing callables
(see `parallelization` submodule), and at runtime a small number of
internal nodes use the C++ SfMData reader.

This shim provides:
  * `parallelization`  — pure-Python re-implementation that parses the
                         SfMData JSON file directly. Sufficient for
                         FeatureExtraction, FeatureMatching, DepthMap,
                         RelativePoseEstimating, ImageProcessing, …
  * `sfmData`/`sfmDataIO`/`hdr`/`image`/`system`/`geometry` — sentinel
                         submodules that raise `NotImplementedError` on
                         attribute access. They only get exercised by
                         specialty nodes (HDR fusion, panorama warping,
                         SfMFilter, etc.) which are not part of the
                         standard photogrammetry pipeline.

The shim must remain dependency-free (stdlib only) so it can be
discovered before any C++ extension build.
"""

# Note: do NOT eagerly import sub-modules — keep the import cheap so the
# Meshroom node loader (which imports every descriptor at startup) does
# not pay the cost of building stub objects unless a node actually uses
# them.

__version__ = "0.1.0-mac-shim"
