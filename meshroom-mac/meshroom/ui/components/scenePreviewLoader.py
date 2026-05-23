"""
scenePreviewLoader — Python-side helper for the Mac-native MetalScenePreview
QML viewer.

Reads the `scene_preview.json` manifest produced by the ScenePreview
node (see `nodes/aliceVision/ScenePreview.py`) and exposes the resolved
paths to QML as Q_PROPERTYs. The QML viewer binds to these properties to
locate the mesh, point cloud, cameras file, and per-view masks folder.

The manifest schema (also documented inside the ScenePreview node):

    {
      "schema": "scene_preview/1.0",
      "node": "ScenePreview",
      "inputs": {
        "cameras":            {"path": "...", "fileCount": N, "linked": true},
        "model":              {"path": "...", "fileCount": N, "linked": true},
        "undistortedImages":  {"path": "...", "fileCount": N, "linked": true},
        "masks":              {"path": "...", "fileCount": N, "linked": true}
      }
    }

We deliberately keep this class small: no asset parsing, no PLY/OBJ
decoding, no QML rendering logic. Loading 3D assets is the QtQuick3D
RuntimeLoader's job (it handles GLTF/OBJ via Metal RHI). Camera frustums
are built by a separate QQuick3DGeometry subclass.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from PySide6.QtCore import QObject, Property, Signal, Slot, QUrl


class ScenePreviewLoader(QObject):
    """Parses a ScenePreview node's output folder and exposes paths to QML.

    Usage from QML:
        ScenePreviewLoader {
            id: previewLoader
            scenePreviewFolder: someNode.outputAttribute.value
            onLoaded: viewer.refresh()
        }
        Model { source: previewLoader.modelUrl }
    """

    folderChanged = Signal()
    loadedChanged = Signal()
    schemaChanged = Signal()
    modelPathChanged = Signal()
    camerasPathChanged = Signal()
    imagesPathChanged = Signal()
    masksPathChanged = Signal()
    modelFileCountChanged = Signal()
    masksFileCountChanged = Signal()
    errorChanged = Signal()

    loaded = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._folder: str = ""
        self._is_loaded: bool = False
        self._schema: str = ""
        self._model_path: str = ""
        self._cameras_path: str = ""
        self._images_path: str = ""
        self._masks_path: str = ""
        self._model_file_count: int = 0
        self._masks_file_count: int = 0
        self._error: str = ""

    # ------------------------------------------------------------------ #
    # scenePreviewFolder property — the manifest's parent folder
    # ------------------------------------------------------------------ #

    @Property(str, notify=folderChanged)
    def scenePreviewFolder(self) -> str:
        return self._folder

    @scenePreviewFolder.setter
    def scenePreviewFolder(self, value: str) -> None:
        # Strip a `file://` prefix if the QML side passed a Url-as-string.
        if value.startswith("file://"):
            value = QUrl(value).toLocalFile()
        if value == self._folder:
            return
        self._folder = value
        self.folderChanged.emit()
        self._reload()

    # ------------------------------------------------------------------ #
    # Derived read-only properties bound by QML
    # ------------------------------------------------------------------ #

    @Property(bool, notify=loadedChanged)
    def isLoaded(self) -> bool:
        return self._is_loaded

    @Property(str, notify=schemaChanged)
    def schema(self) -> str:
        return self._schema

    @Property(str, notify=modelPathChanged)
    def modelPath(self) -> str:
        return self._model_path

    @Property(QUrl, notify=modelPathChanged)
    def modelUrl(self) -> QUrl:
        """QtQuick3D's RuntimeLoader / Model.source expects a QUrl."""
        return QUrl.fromLocalFile(self._model_path) if self._model_path else QUrl()

    @Property(str, notify=camerasPathChanged)
    def camerasPath(self) -> str:
        return self._cameras_path

    @Property(str, notify=imagesPathChanged)
    def imagesPath(self) -> str:
        return self._images_path

    @Property(str, notify=masksPathChanged)
    def masksPath(self) -> str:
        return self._masks_path

    @Property(int, notify=modelFileCountChanged)
    def modelFileCount(self) -> int:
        return self._model_file_count

    @Property(int, notify=masksFileCountChanged)
    def masksFileCount(self) -> int:
        return self._masks_file_count

    @Property(str, notify=errorChanged)
    def error(self) -> str:
        return self._error

    # ------------------------------------------------------------------ #
    # Loader logic
    # ------------------------------------------------------------------ #

    @Slot()
    def reload(self) -> None:
        """Explicit refresh entry point. QML can call this after the
        ScenePreview node finishes computing."""
        self._reload()

    def _reload(self) -> None:
        prev_loaded = self._is_loaded
        self._set_error("")
        self._is_loaded = False

        folder = self._folder
        if not folder:
            self._clear()
            self.loadedChanged.emit() if prev_loaded else None
            return

        manifest_path = Path(folder) / "scene_preview.json"
        if not manifest_path.is_file():
            self._set_error(f"manifest missing: {manifest_path}")
            self._clear()
            self.loadedChanged.emit() if prev_loaded else None
            return

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            self._set_error(f"manifest parse failed: {exc}")
            self._clear()
            self.loadedChanged.emit() if prev_loaded else None
            return

        schema = str(data.get("schema", ""))
        inputs = data.get("inputs", {}) or {}

        # The ScenePreview node creates symlinks at fixed names; honour
        # those first (predictable), fall back to the absolute path from
        # the manifest if the symlink wasn't created.
        model_link = Path(folder) / "model"
        cameras_link = Path(folder) / "cameras"
        images_link = Path(folder) / "images"
        masks_link = Path(folder) / "masks"

        def _resolve(link: Path, manifest_entry: dict) -> str:
            if link.exists():
                try:
                    return str(link.resolve())
                except OSError:
                    pass
            return str(manifest_entry.get("path") or "")

        model = inputs.get("model", {}) or {}
        cameras = inputs.get("cameras", {}) or {}
        images = inputs.get("undistortedImages", {}) or {}
        masks = inputs.get("masks", {}) or {}

        new_model = _resolve(model_link, model)
        new_cameras = _resolve(cameras_link, cameras)
        new_images = _resolve(images_link, images)
        new_masks = _resolve(masks_link, masks)

        # The model path may be a folder (containing mesh.obj) or a
        # direct .obj/.gltf/.ply file. Resolve the actual renderable.
        new_model_render = self._resolve_renderable_model(new_model)

        # Update + emit.
        if schema != self._schema:
            self._schema = schema
            self.schemaChanged.emit()
        if new_model_render != self._model_path:
            self._model_path = new_model_render
            self.modelPathChanged.emit()
        if new_cameras != self._cameras_path:
            self._cameras_path = new_cameras
            self.camerasPathChanged.emit()
        if new_images != self._images_path:
            self._images_path = new_images
            self.imagesPathChanged.emit()
        if new_masks != self._masks_path:
            self._masks_path = new_masks
            self.masksPathChanged.emit()

        mfc = int(model.get("fileCount", 0))
        if mfc != self._model_file_count:
            self._model_file_count = mfc
            self.modelFileCountChanged.emit()
        kfc = int(masks.get("fileCount", 0))
        if kfc != self._masks_file_count:
            self._masks_file_count = kfc
            self.masksFileCountChanged.emit()

        self._is_loaded = True
        self.loadedChanged.emit()
        self.loaded.emit()
        logging.info(
            f"[ScenePreviewLoader] manifest loaded: schema={schema}, "
            f"model={new_model_render or '(none)'}, "
            f"cameras={new_cameras or '(none)'}, "
            f"masks={kfc} files"
        )

    @staticmethod
    def _resolve_renderable_model(model: str) -> str:
        """If `model` is a folder, find the canonical renderable file.

        QtQuick3D RuntimeLoader prefers GLTF/GLB (built-in PBR), then OBJ
        (built-in), then PLY (point cloud — handled by separate
        QQuick3DGeometry path, return empty so the QML side can branch).
        """
        if not model:
            return ""
        p = Path(model)
        if p.is_file():
            return str(p)
        if p.is_dir():
            for ext in ("gltf", "glb", "obj"):
                hits = list(p.glob(f"*.{ext}"))
                if hits:
                    return str(hits[0])
            # Fall through: PLY point cloud. Return empty — QML branches
            # on (modelPath == "" && modelFileCount > 0) to invoke the
            # custom PLY loader.
            ply_hits = list(p.glob("*.ply"))
            if ply_hits:
                return ""
        return ""

    def _clear(self) -> None:
        for attr_name, signal_name in (
            ("_schema", "schemaChanged"),
            ("_model_path", "modelPathChanged"),
            ("_cameras_path", "camerasPathChanged"),
            ("_images_path", "imagesPathChanged"),
            ("_masks_path", "masksPathChanged"),
        ):
            if getattr(self, attr_name) != "":
                setattr(self, attr_name, "")
                getattr(self, signal_name).emit()
        if self._model_file_count != 0:
            self._model_file_count = 0
            self.modelFileCountChanged.emit()
        if self._masks_file_count != 0:
            self._masks_file_count = 0
            self.masksFileCountChanged.emit()

    def _set_error(self, msg: str) -> None:
        if msg != self._error:
            self._error = msg
            self.errorChanged.emit()
            if msg:
                logging.warning(f"[ScenePreviewLoader] {msg}")


__all__ = ["ScenePreviewLoader"]
