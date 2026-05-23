"""
pyalicevision.hdr — Mac-port stub for the upstream C++ HDR bindings.

Upstream `pyalicevision.hdr` is a pybind11 wrapper over AliceVision's C++
HDR library (`aliceVision::hdr::LuminanceInfo`, `estimateGroups`, etc.).
The Mac port has not built those bindings yet, so this stub provides
just enough surface area for the Meshroom node descriptors
(`LdrToHdrSampling`, `LdrToHdrCalibration`, `LdrToHdrMerge`) to complete
their `update(cls, node)` template-load hooks WITHOUT crashing the UI.

The stubbed API is deliberately minimal:

  * `vectorli()`                              → an empty `list`.
  * `LuminanceInfo(viewId, path, exposure)`    → a plain dataclass.
  * `estimateGroups(inputs)`                   → `[]`.

When the user actually tries to COMPUTE an HDR node, the underlying
`aliceVision_LdrToHdr*` C++ binary is what runs — and those binaries
are not built on this port either (Phase 14 work item). The Meshroom UI
will surface a "binary not found" error at compute time, which is the
correct failure mode (a clear, late error rather than an opaque
import-time crash that prevents the user from even seeing the graph).

If/when the C++ pyalicevision bindings are built on macOS, replace this
file with `from _pyalicevision_native.hdr import *` — the API surface
above is a strict subset of what upstream exposes.
"""

from __future__ import annotations

from dataclasses import dataclass


def vectorli() -> list:
    """Stub for `aliceVision::hdr::vectorli` (a typedef for
    `std::vector<LuminanceInfo>`). Returns a plain Python list — supports
    the `.append()` and `len()` calls the descriptors make."""
    return []


@dataclass
class LuminanceInfo:
    """Stub for `aliceVision::hdr::LuminanceInfo`.

    Constructed by the HDR descriptors with (viewId, path, exposure)
    per viewpoint and appended to the `vectorli` list. The Mac port
    only needs the constructor + attribute access to pass the
    template-load step; the values are not consumed past this point
    because `estimateGroups` returns `[]`.
    """

    viewId: int
    path: str
    exposure: float


def estimateGroups(inputs) -> list[list[LuminanceInfo]]:
    """Stub for `aliceVision::hdr::estimateGroups`.

    Returns an empty list so the calling descriptor's `update()` enters
    its early-exit branch (`if len(obj) == 0: node.nbBrackets.value = 0;
    return`) and the template loads cleanly with HDR bracket detection
    disabled. Real bracket detection requires the C++ binding; once it
    lands, this stub disappears.
    """
    return []


__all__ = ["vectorli", "LuminanceInfo", "estimateGroups"]
