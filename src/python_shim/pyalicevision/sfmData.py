"""
pyalicevision.sfmData — Mac-port stub for the upstream C++ SfMData bindings.

Upstream `pyalicevision.sfmData` is a pybind11 wrapper over AliceVision's
C++ `SfMData` class hierarchy. The Mac port has not built those bindings
yet, so this stub provides just enough surface area for descriptors to
import the module without crashing at template-load time.

The descriptors that import this module are:
  * `LdrToHdr{Sampling,Calibration,Merge}.getExposure` — uses
    `ExposureSetting(shutterSpeed, fnumber, iso)` as a value object. Only
    called from the `update()` hook at template load.
  * `SfMChecking`, `SfMFilter`, `SfMPoseFlattening`, `SfMRigApplying` —
    use `SfMData()`, `Views()`, `CameraPose()`, `UndefinedIndexT`, etc.
    These calls are inside `processChunk()` (RUNTIME), not `update()`
    (load time), so a `NotImplementedError` raised on attribute access
    is the correct behavior — the user gets a clear error when they
    actually try to compute the node.

The only API element exercised at template-LOAD time is
`ExposureSetting`, so we implement it as a plain dataclass. Everything
else raises `NotImplementedError` via `__getattr__`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExposureSetting:
    """Stub for `aliceVision::sfmData::ExposureSetting`.

    Used by `LdrToHdr*.getExposure` at template-load time as a value
    object — its `.exposureFactor()` would normally compute (and is
    later consumed) for HDR bracket grouping, but on this port the
    parent `estimateGroups` returns `[]` first so this value is never
    actually consumed downstream.
    """

    shutterSpeed: float
    fnumber: float
    iso: float


def __getattr__(name: str):
    """Anything else (`SfMData`, `Views`, `CameraPose`, `UndefinedIndexT`,
    etc.) raises a clear NotImplementedError when accessed.

    These attributes are only exercised inside `processChunk()` runtime
    paths for `SfMFilter` / `SfMRigApplying` / `SfMPoseFlattening` /
    `SfMChecking`. Compute-time failure with a clear message is the
    correct behavior — the user sees the graph load fine, can wire
    things up, and only hits the error when they press Compute.
    """
    raise NotImplementedError(
        f"pyalicevision.sfmData.{name} is not implemented on the macOS "
        f"port. The C++ pyalicevision bindings have not been built yet "
        f"(Phase 13 work item). The descriptor that needs this attribute "
        f"is invoked only at compute time, not at template load, so the "
        f"graph itself loads — but the node cannot run until the C++ "
        f"binding is provided."
    )


__all__ = ["ExposureSetting"]
