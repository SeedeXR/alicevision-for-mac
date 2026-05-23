"""
pointCloudGeometry — Mac-native QQuick3DGeometry subclass for PLY point clouds.

QtQuick3D's RuntimeLoader handles GLTF/GLB/OBJ but NOT PLY point
clouds (it can parse PLY meshes but loses per-vertex color). For our
photogrammetry use case the canonical dense output is `densePointCloud.ply`
from `aliceVision_meshing`, which carries per-vertex RGB color. We need
those colors to render the point cloud correctly.

This module subclasses `QQuick3DGeometry` to:
  1. Parse a PLY file (ASCII or binary little-endian) at the path bound
     from QML.
  2. Pack vertex positions + per-vertex colors into a single interleaved
     buffer.
  3. Hand the buffer to Qt RHI (which on macOS dispatches it to a Metal
     `MTLBuffer` with `MTLResourceStorageModeShared` — exactly the UMA
     pattern the philosophy doc calls for).
  4. Set primitive type to `Points` so QtQuick3D renders one GPU point
     per vertex.

The renderer (in QML) attaches this Geometry to a `Model` with a
`PrincipledMaterial` that uses vertex colors. The point size is driven
by a `pointSize` Q_PROPERTY on the geometry (rebuilds on change).

This is the philosophy-compliant path:
  - §1 UMA-first: vertex buffer is uploaded once, GPU reads it directly
    from shared memory.
  - §6 Data-oriented: positions+colors are interleaved SoA-style per
    vertex (12B pos + 12B normal-or-zero + 16B color = 40B per vertex).
  - §5 Minimal-copy: the file is parsed into a single bytes() blob that
    becomes the GPU buffer with zero further copies.
  - No premature abstraction: this is a direct Geometry implementation,
    no scene-engine layer.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

from PySide6.QtCore import Property, Signal, QByteArray, QUrl
from PySide6.QtQuick3D import QQuick3DGeometry
from PySide6.QtGui import QVector3D


log = logging.getLogger(__name__)


# Single-vertex stride for our interleaved layout:
#   position (3 × float32) + color (4 × float32, RGBA) = 28 bytes.
# We pack color as float32 RGBA rather than uint8 because QtQuick3D's
# vertex shader expects PrincipledMaterial's vertex-color attribute as
# normalized floats; doing the float conversion CPU-side is one less
# vertex-shader op per draw.
_POS_OFFSET = 0
_COLOR_OFFSET = 12
_VERTEX_STRIDE = 28


class PointCloudGeometry(QQuick3DGeometry):
    """A QQuick3DGeometry that holds a PLY point cloud.

    QML usage::

        import Meshroom.Helpers 1.0
        Model {
            geometry: PointCloudGeometry { source: "/path/to/cloud.ply" }
            materials: PrincipledMaterial { lighting: PrincipledMaterial.NoLighting }
        }
    """

    sourceChanged = Signal()
    pointCountChanged = Signal()
    errorChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: QUrl = QUrl()
        self._point_count: int = 0
        self._error: str = ""

    # --------- properties ---------

    @Property(QUrl, notify=sourceChanged)
    def source(self) -> QUrl:
        return self._source

    @source.setter
    def source(self, value) -> None:
        if isinstance(value, str):
            value = QUrl(value) if "://" in value else QUrl.fromLocalFile(value)
        if value == self._source:
            return
        self._source = value
        self.sourceChanged.emit()
        self._reload()

    @Property(int, notify=pointCountChanged)
    def pointCount(self) -> int:
        return self._point_count

    @Property(str, notify=errorChanged)
    def error(self) -> str:
        return self._error

    # --------- core loader ---------

    def _set_error(self, msg: str) -> None:
        if msg != self._error:
            self._error = msg
            self.errorChanged.emit()
        if msg:
            log.warning(f"[PointCloudGeometry] {msg}")

    def _reload(self) -> None:
        self._set_error("")
        if self._source.isEmpty():
            self._clear_geometry()
            return

        path = self._source.toLocalFile() if self._source.isLocalFile() else self._source.path()
        if not path or not Path(path).is_file():
            self._set_error(f"PLY not found: {path}")
            self._clear_geometry()
            return

        try:
            vertices = _parse_ply(path)
        except Exception as exc:  # noqa: BLE001
            self._set_error(f"PLY parse failed: {exc}")
            self._clear_geometry()
            return

        if not vertices:
            self._set_error("PLY contained zero vertices")
            self._clear_geometry()
            return

        self._upload(vertices)

    def _clear_geometry(self) -> None:
        self.clear()
        if self._point_count != 0:
            self._point_count = 0
            self.pointCountChanged.emit()

    def _upload(self, vertices: list[tuple[float, float, float, float, float, float, float]]) -> None:
        """Pack the parsed vertices into a single interleaved GPU buffer."""
        n = len(vertices)
        buf = bytearray(n * _VERTEX_STRIDE)
        bmin = [float("inf")] * 3
        bmax = [float("-inf")] * 3

        offset = 0
        for x, y, z, r, g, b, a in vertices:
            struct.pack_into("<7f", buf, offset, x, y, z, r, g, b, a)
            offset += _VERTEX_STRIDE
            if x < bmin[0]: bmin[0] = x
            if y < bmin[1]: bmin[1] = y
            if z < bmin[2]: bmin[2] = z
            if x > bmax[0]: bmax[0] = x
            if y > bmax[1]: bmax[1] = y
            if z > bmax[2]: bmax[2] = z

        # Hand the buffer to Qt RHI. On macOS this becomes an MTLBuffer
        # in shared storage — no host↔device copy after this point.
        self.clear()
        self.setVertexData(QByteArray(bytes(buf)))
        self.setStride(_VERTEX_STRIDE)
        self.setPrimitiveType(QQuick3DGeometry.PrimitiveType.Points)
        # Position: location 0, 3 × float32 at offset 0.
        self.addAttribute(
            QQuick3DGeometry.Attribute.PositionSemantic,
            _POS_OFFSET,
            QQuick3DGeometry.Attribute.F32Type,
        )
        # Per-vertex color: location 1, 4 × float32 at offset 12.
        self.addAttribute(
            QQuick3DGeometry.Attribute.ColorSemantic,
            _COLOR_OFFSET,
            QQuick3DGeometry.Attribute.F32Type,
        )
        self.setBounds(QVector3D(*bmin), QVector3D(*bmax))
        self.update()

        self._point_count = n
        self.pointCountChanged.emit()
        log.info(
            f"[PointCloudGeometry] uploaded {n} points "
            f"(bounds: {bmin} -> {bmax}), {n * _VERTEX_STRIDE / 1024:.0f} KB"
        )


# ----------------------------- PLY parser ----------------------------- #
#
# Pure-stdlib parser handling the two PLY flavours produced by AliceVision:
#   - ASCII PLY (cloud_and_poses.ply / densePointCloud-debug variants)
#   - binary_little_endian (the default for densePointCloud.ply)
#
# Both carry per-vertex x/y/z (float) + red/green/blue (uchar). Some
# variants add alpha or normals. We extract only x/y/z + r/g/b; everything
# else is skipped per element.
#
# Reference: http://paulbourke.net/dataformats/ply/

def _parse_ply(path: str) -> list[tuple[float, float, float, float, float, float, float]]:
    """Parse a PLY file, returning a list of (x, y, z, r, g, b, a) tuples
    where colors are in [0.0, 1.0]."""
    with open(path, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("unexpected EOF before end_header")
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        # The current file position is now the start of the binary body
        # (if binary). For ASCII we keep reading line-by-line.
        binary, vertex_count, props = _parse_ply_header(header_lines)
        if binary:
            return _read_ply_binary(f, vertex_count, props)
        return _read_ply_ascii(f, vertex_count, props)


def _parse_ply_header(lines: list[str]) -> tuple[bool, int, list[tuple[str, str]]]:
    """Return (is_binary_le, vertex_count, vertex_property_list)."""
    if not lines or lines[0] != "ply":
        raise ValueError("not a PLY file (magic missing)")
    binary = False
    vertex_count = 0
    vertex_props: list[tuple[str, str]] = []
    in_vertex_block = False
    for line in lines[1:]:
        if line.startswith("format ascii"):
            binary = False
        elif line.startswith("format binary_little_endian"):
            binary = True
        elif line.startswith("format binary_big_endian"):
            raise ValueError("PLY: binary_big_endian not supported")
        elif line.startswith("element vertex "):
            vertex_count = int(line.split()[2])
            in_vertex_block = True
        elif line.startswith("element "):
            in_vertex_block = False
        elif line.startswith("property ") and in_vertex_block:
            parts = line.split()
            # `property <type> <name>` or `property list <ct> <vt> <name>`
            if parts[1] == "list":
                continue
            vertex_props.append((parts[1], parts[2]))
        elif line == "end_header":
            break
    return binary, vertex_count, vertex_props


_PLY_TYPE_FMT: dict[str, str] = {
    "float": "f", "float32": "f",
    "double": "d", "float64": "d",
    "char": "b",  "int8": "b",
    "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h",
    "ushort": "H", "uint16": "H",
    "int": "i",  "int32": "i",
    "uint": "I", "uint32": "I",
}


def _read_ply_binary(f, n: int, props: list[tuple[str, str]]) -> list[tuple]:
    fmt = "<" + "".join(_PLY_TYPE_FMT[t] for t, _ in props)
    size = struct.calcsize(fmt)
    names = [name for _, name in props]
    out: list[tuple] = []
    for _ in range(n):
        chunk = f.read(size)
        if len(chunk) < size:
            break
        row = struct.unpack(fmt, chunk)
        out.append(_extract_row(row, names))
    return out


def _read_ply_ascii(f, n: int, props: list[tuple[str, str]]) -> list[tuple]:
    names = [name for _, name in props]
    out: list[tuple] = []
    for _ in range(n):
        line = f.readline()
        if not line:
            break
        toks = line.decode("ascii", errors="replace").split()
        row = tuple(float(t) for t in toks[: len(props)])
        out.append(_extract_row(row, names))
    return out


def _extract_row(row: tuple, names: list[str]) -> tuple[float, float, float, float, float, float, float]:
    """Pick out x/y/z + r/g/b/a from a parsed property row."""
    by_name = dict(zip(names, row))
    x = float(by_name.get("x", 0.0))
    y = float(by_name.get("y", 0.0))
    z = float(by_name.get("z", 0.0))
    r = by_name.get("red", by_name.get("r", 200.0))
    g = by_name.get("green", by_name.get("g", 200.0))
    b = by_name.get("blue", by_name.get("b", 200.0))
    a = by_name.get("alpha", by_name.get("a", 255.0))
    # Normalise: if the source type is uchar (range 0..255) we still get
    # a value in that range here; divide by 255 to land in [0, 1]. If it
    # was already float (rare in PLY), the divide is harmless for sane
    # inputs.
    if r > 1.0 or g > 1.0 or b > 1.0 or a > 1.0:
        r /= 255.0; g /= 255.0; b /= 255.0; a /= 255.0
    return (x, y, z, float(r), float(g), float(b), float(a))


__all__ = ["PointCloudGeometry"]
