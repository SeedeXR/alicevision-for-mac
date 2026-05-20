"""
pyalicevision.parallelization — pure-Python shim.

Re-implements the four chunk-size callables that Meshroom node
descriptors use to decide how many parallel chunks to split a node into.
The upstream implementation calls into the C++ SfMData reader; here we
parse the SfMData JSON file directly with the stdlib. This is correct
because SfMData files are a stable, fully documented JSON schema:
viewId / poseId / intrinsicId / path / width / height / metadata / …

Used by (legacy photogrammetry pipeline):
  * FeatureExtraction       → DynamicViewsSize
  * FeatureMatching         → DynamicViewsSize
  * DepthMap                → DynamicReconstructedViewsSize
  * RelativePoseEstimating  → DynamicViewsSize     (modern pipeline)
"""

from __future__ import annotations

import json
import os
from typing import Any


def _load_sfm_views(path: str) -> dict[str, Any]:
    """Read and parse an SfMData JSON file, return the parsed dict.

    The on-disk format is the standard AliceVision SfMData JSON. We only
    rely on the `views`, `poses` and `intrinsics` top-level lists.
    """
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"Failed to load file : {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class DynamicViewsSize:
    """Number of parallel chunks = total number of views in the SfMData."""

    def __init__(self, param: str) -> None:
        self._param = param

    def __call__(self, node) -> int:
        param = node.attribute(self._param)
        data = _load_sfm_views(param.value)
        return max(1, len(data.get("views", [])))


class DynamicReconstructedViewsSize:
    """Number of chunks = views that have BOTH a pose and an intrinsic.

    The upstream C++ implementation calls SfMData::isPoseAndIntrinsicDefined.
    A view is "pose-defined" when its `poseId` references an entry in the
    top-level `poses` list, and "intrinsic-defined" when its `intrinsicId`
    references an entry in the top-level `intrinsics` list. Both lists are
    serialised in the SfMData JSON.
    """

    def __init__(self, param: str) -> None:
        self._param = param

    def __call__(self, node) -> int:
        param = node.attribute(self._param)
        data = _load_sfm_views(param.value)
        pose_ids = {p.get("poseId") for p in data.get("poses", [])}
        intrinsic_ids = {i.get("intrinsicId") for i in data.get("intrinsics", [])}
        count = 0
        for view in data.get("views", []):
            if view.get("poseId") in pose_ids and view.get("intrinsicId") in intrinsic_ids:
                count += 1
        return max(1, count)


class DynamicDividedViewsSize:
    """Total views divided by an IntParam (rounded up), with floor 1."""

    def __init__(self, param: str, divider: str) -> None:
        self._param = param
        self._divider = divider

    def __call__(self, node) -> int:
        import math
        from meshroom.core import desc

        param = node.attribute(self._param)
        divider = node.attribute(self._divider)
        if not isinstance(divider.desc, desc.IntParam):
            raise RuntimeError("Divider object is not a number.")
        data = _load_sfm_views(param.value)
        d = max(1, divider.value)
        return max(1, math.ceil(len(data.get("views", [])) / d))


class DynamicDirectorySize:
    """Number of chunks = number of images in the directory (recursive=False).

    The upstream C++ helper uses `image::listImages` which honours a
    library-level allow-list of extensions. We approximate with the same
    extension set; the value only affects parallel chunking granularity
    so an approximation is safe.
    """

    _IMG_EXTS = {
        ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr", ".bmp",
        ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raw", ".raf",
    }

    def __init__(self, param: str) -> None:
        self._param = param

    def __call__(self, node) -> int:
        param = node.attribute(self._param)
        path = param.value
        if not path or not os.path.isdir(path):
            raise RuntimeError(f"Failed to load file : {path}")
        count = 0
        for name in os.listdir(path):
            ext = os.path.splitext(name)[1].lower()
            if ext in self._IMG_EXTS:
                count += 1
        return max(1, count)
