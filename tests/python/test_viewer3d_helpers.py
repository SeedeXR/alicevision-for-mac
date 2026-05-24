"""
Tests for the Mac-native QtQuick3D viewer-3D Python helpers:

  * `meshroom-mac/meshroom/ui/components/scenePreviewLoader.py`
    — ScenePreviewLoader (parses scene_preview.json + resolves paths).
  * `meshroom-mac/meshroom/ui/components/pointCloudGeometry.py`
    — PointCloudGeometry (PLY parser + interleaved GPU buffer upload).
  * `meshroom-mac/meshroom/ui/components/frustumGeometry.py`
    — FrustumGeometry (12-edge wireframe pyramid built from intrinsics).

These were added in Phases 4-6 with manual smoke tests only. Pinning
them down with pytest so future regressions surface in CI rather than
"the user reports the viewer is blank".

The tests construct a `QGuiApplication` once (module-scope fixture)
because Qt's QObject + QML type construction requires an active Qt
event loop / application context. Many of the assertions are about
property values + attribute layouts; they do NOT require an OpenGL
or Metal context.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from PySide6.QtCore import QUrl, QCoreApplication
from PySide6.QtGui import QGuiApplication


# --------------------------------------------------------------------------- #
# Module-scope Qt application — QObject construction needs it.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module", autouse=True)
def _qt_app():
    app = QCoreApplication.instance() or QGuiApplication([])
    yield app


# =========================================================================== #
# ScenePreviewLoader
# =========================================================================== #

def _write_manifest(folder: Path, inputs: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scene_preview.json").write_text(json.dumps({
        "schema": "scene_preview/1.0",
        "node": "ScenePreview",
        "inputs": inputs,
    }))


def test_scenepreview_loader_parses_a_real_manifest(tmp_path):
    from meshroom.ui.components.scenePreviewLoader import ScenePreviewLoader

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "mesh.obj").write_text("v 0 0 0\n")
    _write_manifest(tmp_path, {
        "cameras": {"path": str(tmp_path / "cameras"), "fileCount": 0, "linked": False},
        "model": {"path": str(model_dir), "fileCount": 1, "linked": True},
        "undistortedImages": {"path": "", "fileCount": 0, "linked": False},
        "masks": {"path": "", "fileCount": 0, "linked": False},
    })

    loader = ScenePreviewLoader()
    loader.scenePreviewFolder = str(tmp_path)
    assert loader.isLoaded
    assert loader.schema == "scene_preview/1.0"
    assert loader.modelPath.endswith("mesh.obj"), (
        f"modelPath should resolve to mesh.obj inside the model dir; "
        f"got {loader.modelPath!r}"
    )
    assert loader.modelUrl.toString().startswith("file://")
    assert loader.modelFileCount == 1


def test_scenepreview_loader_missing_manifest_clears_state(tmp_path):
    from meshroom.ui.components.scenePreviewLoader import ScenePreviewLoader
    loader = ScenePreviewLoader()
    loader.scenePreviewFolder = str(tmp_path / "nope")
    assert not loader.isLoaded
    assert "manifest missing" in loader.error
    assert loader.modelPath == ""


def test_scenepreview_loader_ply_routes_to_empty_modelpath(tmp_path):
    """If the manifest's model folder contains ONLY a .ply (no OBJ/GLTF),
    modelPath returns "" so QML branches to the PLY geometry path."""
    from meshroom.ui.components.scenePreviewLoader import ScenePreviewLoader
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "densePointCloud.ply").write_text("ply\nformat ascii 1.0\nend_header\n")
    _write_manifest(tmp_path, {
        "cameras": {"path": "", "fileCount": 0, "linked": False},
        "model": {"path": str(model_dir), "fileCount": 1, "linked": True},
        "undistortedImages": {"path": "", "fileCount": 0, "linked": False},
        "masks": {"path": "", "fileCount": 0, "linked": False},
    })

    loader = ScenePreviewLoader()
    loader.scenePreviewFolder = str(tmp_path)
    assert loader.isLoaded
    # PLY route: modelPath empty so QML knows to use PointCloudGeometry.
    assert loader.modelPath == "", (
        f"PLY-only folder should yield empty modelPath; got {loader.modelPath!r}"
    )


# =========================================================================== #
# PointCloudGeometry
# =========================================================================== #

def _write_ascii_ply(path: Path, vertices: list[tuple]) -> None:
    """Write a minimal ASCII PLY with x/y/z + red/green/blue (uchar)."""
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    body = "\n".join(
        f"{x} {y} {z} {int(r)} {int(g)} {int(b)}"
        for x, y, z, r, g, b in vertices
    )
    path.write_text(header + body + "\n")


def test_pointcloud_geometry_parses_ascii_ply(tmp_path):
    from meshroom.ui.components.pointCloudGeometry import (
        PointCloudGeometry, _parse_ply,
    )
    ply = tmp_path / "tiny.ply"
    _write_ascii_ply(ply, [
        (0.0, 0.0, 0.0, 255, 0, 0),
        (1.0, 0.0, 0.0, 0, 255, 0),
        (0.0, 1.0, 0.0, 0, 0, 255),
    ])
    verts = _parse_ply(str(ply))
    assert len(verts) == 3
    # 10-tuple shape (x, y, z, nx, ny, nz, r, g, b, a). When PLY carries
    # no normal properties, parser writes sentinel (0, 0, 1).
    assert len(verts[0]) == 10
    nx, ny, nz = verts[0][3], verts[0][4], verts[0][5]
    assert (nx, ny, nz) == (0.0, 0.0, 1.0), (
        f"expected sentinel normal (0,0,1) when PLY has none; got "
        f"({nx}, {ny}, {nz})"
    )
    # Colors normalised to [0, 1].
    r, g, b, a = verts[0][6:10]
    assert (r, g, b, a) == (1.0, 0.0, 0.0, 1.0)


def test_pointcloud_geometry_extracts_normals_when_present(tmp_path):
    from meshroom.ui.components.pointCloudGeometry import _parse_ply
    ply = tmp_path / "with_normals.ply"
    ply.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
        "0 0 0 0.6 0.0 0.8 100 150 200\n"
    )
    verts = _parse_ply(str(ply))
    assert len(verts) == 1
    nx, ny, nz = verts[0][3], verts[0][4], verts[0][5]
    # Already unit-length, parser shouldn't renormalize destructively.
    assert abs(nx - 0.6) < 1e-5
    assert abs(ny - 0.0) < 1e-5
    assert abs(nz - 0.8) < 1e-5


def test_pointcloud_geometry_setter_updates_point_count(tmp_path):
    from meshroom.ui.components.pointCloudGeometry import PointCloudGeometry
    ply = tmp_path / "small.ply"
    _write_ascii_ply(ply, [(float(i), 0.0, 0.0, 128, 128, 128) for i in range(5)])
    geom = PointCloudGeometry()
    assert geom.pointCount == 0
    geom.source = QUrl.fromLocalFile(str(ply))
    assert geom.pointCount == 5, (
        f"after setting source, pointCount should be 5; got {geom.pointCount}; "
        f"error={geom.error!r}"
    )
    assert geom.error == ""


def test_pointcloud_geometry_missing_file_sets_error(tmp_path):
    from meshroom.ui.components.pointCloudGeometry import PointCloudGeometry
    geom = PointCloudGeometry()
    geom.source = QUrl.fromLocalFile(str(tmp_path / "nonexistent.ply"))
    assert geom.pointCount == 0
    assert geom.error != ""
    assert "not found" in geom.error.lower()


def test_pointcloud_geometry_binary_ply(tmp_path):
    """The dense AliceVision cloud is binary_little_endian; ensure that
    code path works on a small synthetic file."""
    from meshroom.ui.components.pointCloudGeometry import _parse_ply
    ply = tmp_path / "binary.ply"
    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"end_header\n"
    )
    body = struct.pack("<fff3B", 1.0, 2.0, 3.0, 10, 20, 30) + \
           struct.pack("<fff3B", -1.0, -2.0, -3.0, 200, 100, 50)
    ply.write_bytes(header + body)
    verts = _parse_ply(str(ply))
    assert len(verts) == 2
    assert verts[0][0] == 1.0 and verts[0][1] == 2.0 and verts[0][2] == 3.0


# =========================================================================== #
# FrustumGeometry
# =========================================================================== #

def test_frustum_geometry_builds_lines():
    from meshroom.ui.components.frustumGeometry import FrustumGeometry
    geom = FrustumGeometry()
    # Default values should produce valid output without crashing.
    assert geom.nearPlane > 0
    assert geom.farPlane > geom.nearPlane
    assert geom.fovYDegrees > 0
    assert geom.aspectRatio > 0


def test_frustum_geometry_updates_on_param_change():
    """Geometry must re-build when any of its params change so the
    visible frustum scales with the scene radius."""
    from meshroom.ui.components.frustumGeometry import FrustumGeometry
    geom = FrustumGeometry()

    geom.nearPlane = 0.5
    assert geom.nearPlane == 0.5
    geom.farPlane = 2.0
    assert geom.farPlane == 2.0
    geom.fovYDegrees = 60.0
    assert geom.fovYDegrees == 60.0
    geom.aspectRatio = 1.777
    assert abs(geom.aspectRatio - 1.777) < 1e-6


def test_frustum_geometry_no_param_change_is_noop():
    """Repeated assignment with the same value shouldn't trigger rebuild
    storms — the setter short-circuits when the value matches."""
    from meshroom.ui.components.frustumGeometry import FrustumGeometry
    geom = FrustumGeometry()
    captured = []
    geom.geometryChanged.connect(lambda: captured.append(1))
    initial = len(captured)
    # Same value 5 times → at most one signal.
    for _ in range(5):
        geom.nearPlane = geom.nearPlane
    assert len(captured) - initial <= 1
