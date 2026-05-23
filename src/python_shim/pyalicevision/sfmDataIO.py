"""
pyalicevision.sfmDataIO — Mac-port stub.

Upstream `pyalicevision.sfmDataIO` exposes `load()`, `save()`, and the
`ALL` / `VIEWS` / `EXTRINSICS` / `INTRINSICS` flag enum that selects
which SfMData sections to (de)serialize. The Mac port has not built
the C++ bindings yet; descriptors that need real serialization
(`SfMChecking`, `SfMFilter`, `SfMPoseFlattening`, `SfMRigApplying`) call
these only at compute time (inside `processChunk`), not at template
load. So this module exists to make `from pyalicevision import
sfmDataIO as avsfmdataio` succeed; first attribute access at compute
time raises a clear NotImplementedError with diagnostic guidance.
"""

from __future__ import annotations


def __getattr__(name: str):
    raise NotImplementedError(
        f"pyalicevision.sfmDataIO.{name} is not implemented on the macOS "
        f"port. The C++ pyalicevision bindings have not been built yet "
        f"(Phase 13 work item). Affected nodes: SfMChecking, SfMFilter, "
        f"SfMPoseFlattening, SfMRigApplying — these load fine in the "
        f"graph but cannot Compute until the C++ binding is provided."
    )


__all__: list[str] = []
