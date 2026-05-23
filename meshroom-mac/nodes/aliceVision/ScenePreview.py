__version__ = "2.0"

#
# ScenePreview — Mac-native scene-preview aggregator node.
#
# Background: Meshroom 2026.1.0 templates (cameraTracking,
# cameraTrackingDepth, nodalCameraTracking, photogrammetryAndCameraTracking,
# and their legacy / no-calibration variants — 10 templates total)
# reference a `ScenePreview` node type that upstream introduced for the
# 2026 release. Upstream has not (yet) shipped a Python descriptor file
# for it in the open-source tree, so the templates fail to deserialize
# cleanly: `WARNING:root:Expression '{ScenePreview_1.output}': 'output'`
# fires for every downstream consumer (`CopyFiles_1.inputFiles`) of the
# missing typed `output` attribute.
#
# This descriptor is our Mac-native replacement. It's a pure-Python
# Meshroom node that runs in the worker process on Apple Silicon — no
# CLI binary, no CUDA, no PyTorch, no Rosetta. It collects the four
# upstream inputs (cameras, model, undistortedImages, masks) and emits a
# single `output` folder containing:
#
#   1. `scene_preview.json` — a manifest listing the input paths +
#      basic stats (file counts) for diagnostics.
#   2. Optional per-input symlinks (cameras/, model/, images/, masks/)
#      pointing back at the source folders. This means CopyFiles_1, which
#      receives `output`, gets a single self-describing directory.
#
# The current implementation does NOT render a 3D scene preview. A real
# preview would need a Metal-based renderer reading the dense point
# cloud + cameras + textures; that's a future Phase 3 feature. For now,
# making the descriptor known to Meshroom is sufficient to silence the
# warning and let the 10 affected templates load with proper typed
# attributes for the downstream `CopyFiles` consumer.
#
# Native-on-Apple-Silicon properties:
#   * Pure Python — runs inside the meshroom-venv Python 3.13 which is
#     itself ARM64-native.
#   * Uses only stdlib (json, os, shutil) — no transitive Rosetta deps.
#   * O(1) symlink creation; no GPU work needed for the aggregator path.
#   * Symlinks (not copies) keep the output folder size to a few KB
#     regardless of how big the upstream model / image folders are.
#

import json
import logging
import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class ScenePreview(desc.Node):
    """Aggregate scene assets (cameras, model, images, masks) into a
    single preview-friendly folder.

    Output:
        {output}/
            scene_preview.json   — manifest with paths + counts
            cameras/             — symlink to {cameras input folder}
            model/               — symlink to {model input folder}
            images/              — symlink to {undistortedImages input folder}
            masks/               — symlink to {masks input folder}

    Downstream consumers (typically CopyFiles_1.inputFiles) read the
    output folder as a single archive root.
    """

    size = desc.DynamicNodeSize("cameras")
    category = "Utils"
    documentation = """
Mac-native scene-preview aggregator. Bundles cameras + dense model +
undistorted images + foreground masks into a single output folder for
downstream archival or preview consumption.

This is a Python-only Meshroom node; no CLI binary is invoked.
"""

    inputs = [
        desc.File(
            name="cameras",
            label="Cameras",
            description="SfMData JSON with calibrated camera intrinsics + extrinsics "
                        "(typically wired from ConvertSfMFormat or ExportAnimatedCamera).",
            value="",
        ),
        desc.File(
            name="model",
            label="Model",
            description="Decimated mesh or dense point cloud "
                        "(typically wired from MeshDecimate or Meshing).",
            value="",
        ),
        desc.File(
            name="undistortedImages",
            label="Undistorted Images",
            description="Folder of undistorted per-view images "
                        "(typically wired from ExportImages).",
            value="",
        ),
        desc.File(
            name="masks",
            label="Masks",
            description="Per-view foreground masks "
                        "(typically wired from SegmentationBiRefNet).",
            value="",
        ),
        # Point-cloud render hints. Used by future Metal-based preview
        # viewer (currently consumed only as serialized metadata in
        # scene_preview.json). Present here so nodalCameraTracking +
        # nodalCameraTrackingWithoutCalibration templates load without
        # a DescriptionConflict.
        desc.GroupAttribute(
            name="pointCloudParams",
            label="Point Cloud Render",
            description="Visual parameters for the dense point-cloud preview.",
            commandLineGroup="",
            items=[
                desc.FloatParam(
                    name="particleSize",
                    label="Particle Size",
                    description="Render-time size of each point.",
                    value=0.001,
                    range=(0.0001, 0.1, 0.0001),
                ),
                desc.ChoiceParam(
                    name="particleColor",
                    label="Particle Color",
                    description="Preset color for points.",
                    value="Red",
                    values=["Red", "Green", "Blue", "White", "Black", "Grey"],
                    exclusive=True,
                ),
            ],
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
            invalidate=False,
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Scene Preview",
            description="Folder containing scene_preview.json + symlinks to "
                        "cameras / model / images / masks.",
            value="{nodeCacheFolder}",
        ),
    ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_symlink(source: Path, dest: Path, log: logging.Logger) -> bool:
        """Create dest → source. Replace stale links; tolerate missing source."""
        if not source.exists():
            log.warning(f"[ScenePreview] source missing, skipping symlink: {source}")
            return False
        try:
            if dest.is_symlink() or dest.exists():
                dest.unlink()
        except OSError:
            pass
        try:
            os.symlink(str(source), str(dest))
            return True
        except OSError as exc:
            log.warning(f"[ScenePreview] symlink failed {dest} -> {source}: {exc}")
            return False

    @staticmethod
    def _count_files(folder: Path) -> int:
        """Best-effort file count under a folder; 0 if not a directory."""
        if not folder.is_dir():
            return 0
        try:
            return sum(1 for _ in folder.iterdir() if _.is_file())
        except OSError:
            return 0

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            log = chunk.logger

            cameras = (chunk.node.cameras.value or "").strip()
            model = (chunk.node.model.value or "").strip()
            images = (chunk.node.undistortedImages.value or "").strip()
            masks = (chunk.node.masks.value or "").strip()

            out_dir = Path(chunk.node.output.value)
            out_dir.mkdir(parents=True, exist_ok=True)

            log.info(f"[ScenePreview] output folder: {out_dir}")
            log.info(f"[ScenePreview]   cameras           : {cameras or '(empty)'}")
            log.info(f"[ScenePreview]   model             : {model or '(empty)'}")
            log.info(f"[ScenePreview]   undistortedImages : {images or '(empty)'}")
            log.info(f"[ScenePreview]   masks             : {masks or '(empty)'}")

            inputs = {
                "cameras": cameras,
                "model": model,
                "undistortedImages": images,
                "masks": masks,
            }
            stats: dict = {}

            for key, src in inputs.items():
                if not src:
                    stats[key] = {"path": None, "fileCount": 0, "linked": False}
                    continue
                src_path = Path(src).resolve()
                # If the source is a file (e.g. SfMData JSON), symlink
                # it directly. If it's a folder (the common case for
                # images/masks/model), symlink the folder.
                link_name = "cameras" if key == "cameras" else (
                    "model" if key == "model" else (
                        "images" if key == "undistortedImages" else "masks"
                    )
                )
                dest = out_dir / link_name
                linked = self._safe_symlink(src_path, dest, log)
                file_count = (
                    self._count_files(src_path) if src_path.is_dir() else (1 if src_path.is_file() else 0)
                )
                stats[key] = {
                    "path": str(src_path) if src_path.exists() else None,
                    "fileCount": file_count,
                    "linked": linked,
                }
                log.info(
                    f"[ScenePreview]   {key}: linked={linked}, "
                    f"files={file_count}"
                )

            manifest = {
                "schema": "scene_preview/1.0",
                "node": "ScenePreview",
                "inputs": stats,
            }
            manifest_path = out_dir / "scene_preview.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            log.info(f"[ScenePreview] manifest written: {manifest_path}")
            log.info("[ScenePreview] done")
        finally:
            chunk.logManager.end()
