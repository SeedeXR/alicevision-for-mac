"""
frustumGeometry — QQuick3DGeometry that renders the 12 edges of a
camera-frustum pyramid as a single line list. Used by the Mac-native
MetalScenePreview viewer to visualize per-view camera poses.

A camera frustum is the pyramidal volume between the camera's near and
far clip planes. With principal point at the image center (the common
case for AliceVision SfMData), the eight corners are:

    near plane (at z = -near, in camera-local coords):
        (-near*ax, -near*ay, -near)   bottom-left
        (+near*ax, -near*ay, -near)   bottom-right
        (+near*ax, +near*ay, -near)   top-right
        (-near*ax, +near*ay, -near)   top-left
    far plane (at z = -far):
        (... × (far/near) ...)         same x/y signs, scaled

where `ax = (W/2) / fx` and `ay = (H/2) / fy` are the half-angle ratios.

We render 12 line segments connecting these corners (4 near edges + 4 far
edges + 4 connecting edges), interleaved as (x, y, z) float32 vertices in
a single buffer. The Model that owns this Geometry transforms it into
world space via the camera pose (translation + rotation).

This is the philosophy-compliant path (§3 Metal GPU, §5 minimal-copy):
the vertex buffer is uploaded once and reused for every render frame;
the per-frustum transform happens in the vertex shader via the Model's
worldTransform.
"""

from __future__ import annotations

import struct

from PySide6.QtCore import Property, Signal
from PySide6.QtGui import QVector3D
from PySide6.QtQuick3D import QQuick3DGeometry


_VERTEX_STRIDE = 12  # 3 × float32 = 12 bytes per vertex; no colors.


# Twelve line segments × 2 endpoints = 24 vertices.
# Indices into the 8 corner array:
#   0..3 = near (BL, BR, TR, TL)
#   4..7 = far  (BL, BR, TR, TL)
_FRUSTUM_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # near plane
    (4, 5), (5, 6), (6, 7), (7, 4),  # far plane
    (0, 4), (1, 5), (2, 6), (3, 7),  # near-to-far edges
)


class FrustumGeometry(QQuick3DGeometry):
    """A QQuick3DGeometry that renders a single camera frustum as line segments.

    Properties:
        nearPlane (float): distance to the near clip plane, in scene units.
        farPlane  (float): distance to the far clip plane.
        fovYDegrees (float): vertical field of view, degrees.
        aspectRatio (float): width / height.
    """

    geometryChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._near: float = 0.02
        self._far: float = 0.15
        self._fov_y: float = 45.0
        self._aspect: float = 1.5
        self._rebuild()

    # --------- properties (any change triggers rebuild) ---------

    @Property(float, notify=geometryChanged)
    def nearPlane(self) -> float:
        return self._near

    @nearPlane.setter
    def nearPlane(self, v: float) -> None:
        if abs(v - self._near) < 1e-9: return
        self._near = float(v); self._rebuild()

    @Property(float, notify=geometryChanged)
    def farPlane(self) -> float:
        return self._far

    @farPlane.setter
    def farPlane(self, v: float) -> None:
        if abs(v - self._far) < 1e-9: return
        self._far = float(v); self._rebuild()

    @Property(float, notify=geometryChanged)
    def fovYDegrees(self) -> float:
        return self._fov_y

    @fovYDegrees.setter
    def fovYDegrees(self, v: float) -> None:
        if abs(v - self._fov_y) < 1e-6: return
        self._fov_y = float(v); self._rebuild()

    @Property(float, notify=geometryChanged)
    def aspectRatio(self) -> float:
        return self._aspect

    @aspectRatio.setter
    def aspectRatio(self, v: float) -> None:
        if abs(v - self._aspect) < 1e-6: return
        self._aspect = float(v); self._rebuild()

    # --------- core ---------

    def _rebuild(self) -> None:
        import math
        half_fy = math.radians(self._fov_y) * 0.5
        ay = math.tan(half_fy)
        ax = ay * self._aspect
        n, f = self._near, self._far
        # 8 corners, camera-local, OpenGL/Metal forward = -Z.
        corners = [
            (-n * ax, -n * ay, -n),  # 0 near BL
            ( n * ax, -n * ay, -n),  # 1 near BR
            ( n * ax,  n * ay, -n),  # 2 near TR
            (-n * ax,  n * ay, -n),  # 3 near TL
            (-f * ax, -f * ay, -f),  # 4 far BL
            ( f * ax, -f * ay, -f),  # 5 far BR
            ( f * ax,  f * ay, -f),  # 6 far TR
            (-f * ax,  f * ay, -f),  # 7 far TL
        ]
        # Also add a small "look-at" stub from the origin to the centre
        # of the near plane so the user can tell which way the camera
        # points even when the frustum is tiny on-screen.
        corners.append((0.0, 0.0, 0.0))         # 8 = camera origin
        corners.append((0.0, 0.0, -n))           # 9 = near plane centre

        edges = list(_FRUSTUM_EDGES) + [(8, 9)]

        from PySide6.QtCore import QByteArray
        n_verts = len(edges) * 2
        buf = bytearray(n_verts * _VERTEX_STRIDE)
        offset = 0
        for a, b in edges:
            ax_, ay_, az_ = corners[a]
            bx_, by_, bz_ = corners[b]
            struct.pack_into("<3f", buf, offset, ax_, ay_, az_); offset += _VERTEX_STRIDE
            struct.pack_into("<3f", buf, offset, bx_, by_, bz_); offset += _VERTEX_STRIDE

        self.clear()
        self.setVertexData(QByteArray(bytes(buf)))
        self.setStride(_VERTEX_STRIDE)
        self.setPrimitiveType(QQuick3DGeometry.PrimitiveType.Lines)
        self.addAttribute(
            QQuick3DGeometry.Attribute.PositionSemantic, 0,
            QQuick3DGeometry.Attribute.F32Type,
        )
        bound = max(self._far, 1e-3) * max(ax, ay, 1.0)
        self.setBounds(QVector3D(-bound, -bound, -self._far),
                       QVector3D(bound, bound, 0.0))
        self.update()
        self.geometryChanged.emit()


__all__ = ["FrustumGeometry"]
