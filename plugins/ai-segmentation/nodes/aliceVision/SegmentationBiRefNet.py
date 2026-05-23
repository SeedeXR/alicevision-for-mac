__version__ = "2.0"

import json
import os
import sys
import time
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class SegmentationBiRefNet(desc.Node):
    """
Generate a binary foreground mask per view using BiRefNet on CoreML.

Mac-native, in-process AI segmentation node. Loads a pre-converted FP16
`.mlpackage` from `<repo>/ai-models/` and runs it on the Apple Silicon
GPU via `MLComputeUnits.cpuAndGPU`. There is no CUDA, no PyTorch, no
ONNX Runtime, no rembg, no subprocess invocation: inference runs in
the Meshroom worker process.

The Apple Neural Engine is intentionally NOT targeted. BiRefNet's
`ASPPDeformable` decoder lowers to `grid_sample`, which the ANE compiler
cannot plan; using `.all` or `.cpuAndNeuralEngine` hangs the model
load. See `models/production_note.md` for the full analysis.

Pre-flight:
    1. Activate `meshroom-venv/`.
    2. Stage the .mlpackage files at `<repo>/ai-models/`:
         ai-models/BiRefNet_lite.mlpackage  (default, ~350 ms / frame)
         ai-models/BiRefNet.mlpackage       (higher accuracy, ~980 ms / frame)
       Conversion recipe: `ai-models/README.md`.

Outputs masks named `{imageStem}_mask.png` (PNG default) so they match
the AliceVision/Meshroom convention consumed by FeatureExtraction and
DepthMap when those nodes are pointed at `masksFolder=<this node>/`.
"""

    size = desc.DynamicNodeSize("input")
    category = "Utils"

    inputs = [
        desc.File(
            name="input",
            label="SfMData",
            description="Input SfMData JSON file. One mask is produced per view.",
            value="",
        ),
        desc.ChoiceParam(
            name="modelVariant",
            label="Model Variant",
            description=(
                "BiRefNet CoreML mlpackage to load from <repo>/ai-models/.\n\n"
                " - birefnet-lite (90 MB, swin_v1_t): default. ~350 ms/frame on "
                "M-series GPU. Recommended for everyday photogrammetry, large "
                "batch jobs, and M1/M2 base (8 GB UMA).\n"
                " - birefnet-general (447 MB, swin_v1_l): higher accuracy on "
                "hair / fine edges. ~980 ms/frame on M-series GPU. Use when "
                "you need the very best mask quality.\n\n"
                "Switching variants invalidates the per-node Meshroom cache "
                "(UID changes) — masks regenerate on next compute. The "
                "in-process session cache keeps the prior model in RAM until "
                "the worker exits."
            ),
            value="birefnet-lite",
            values=["birefnet-lite", "birefnet-general"],
            exposed=True,
        ),
        desc.ChoiceParam(
            name="maskFormat",
            label="Mask Format",
            description="Output mask format. PNG matches FeatureExtraction's "
                        "default; EXR matches AliceVision ImageSegmentation.",
            value="png",
            values=["png", "exr"],
        ),
        desc.BoolParam(
            name="keepFilename",
            label="Keep Filename",
            description="Use original image stem in mask filename. If False, "
                        "use the SfMData viewId (matches AliceVision conventions).",
            value=True,
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
            label="Masks Folder",
            description="Output folder containing one mask per view.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="masks",
            label="Masks",
            description="Generated foreground masks.",
            semantic="image",
            value=lambda attr: (
                "{nodeCacheFolder}/<FILESTEM>_mask." + attr.node.maskFormat.value
                if attr.node.keepFilename.value
                else "{nodeCacheFolder}/<VIEW_ID>." + attr.node.maskFormat.value
            ),
            commandLineGroup="",  # not a CLI parameter
        ),
    ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ensure_segmentation_on_path(self):
        """Make the in-repo `segmentation` package importable.

        The node descriptor lives at
        `plugins/ai-segmentation/nodes/aliceVision/SegmentationBiRefNet.py`
        and the helpers at `plugins/ai-segmentation/python/segmentation/`.
        We add the plugin's `python/` dir to sys.path so `import segmentation`
        works regardless of how Meshroom or the run script was launched.
        """
        here = Path(__file__).resolve().parent  # .../nodes/aliceVision
        plugin_root = here.parent.parent
        candidate = str(plugin_root / "python")
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    def _read_views(self, sfm_data_path: str, chunk) -> list[tuple[str, str]]:
        """Parse SfMData JSON and return [(viewId, imagePath), ...]."""
        if not sfm_data_path or not os.path.isfile(sfm_data_path):
            raise RuntimeError(f"SfMData file not found: {sfm_data_path}")
        with open(sfm_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results: list[tuple[str, str]] = []
        for v in data.get("views", []):
            view_id = str(v.get("viewId", ""))
            path = v.get("path", "")
            if not view_id or not path:
                chunk.logger.warning(f"Skipping malformed view entry: {v!r}")
                continue
            results.append((view_id, path))
        return results

    def _mask_path(self, out_dir: Path, view_id: str, image_path: str,
                   keep_filename: bool, ext: str) -> Path:
        if keep_filename:
            stem = Path(image_path).stem
            return out_dir / f"{stem}_mask.{ext}"
        return out_dir / f"{view_id}.{ext}"

    def _save_mask(self, mask: "np.ndarray", dest: Path, ext: str) -> None:
        """Save a `[H, W]` float32 mask in [0, 1] as PNG or EXR."""
        import numpy as np
        from PIL import Image

        dest.parent.mkdir(parents=True, exist_ok=True)
        if ext == "png":
            img8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(img8, mode="L").save(dest, format="PNG", optimize=True)
            return
        if ext == "exr":
            import imageio.v3 as iio
            iio.imwrite(str(dest), mask.astype(np.float32), extension=".exr")
            return
        raise RuntimeError(f"Unsupported mask format: {ext}")

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            log = chunk.logger

            # 1. Bring our helper package onto sys.path.
            self._ensure_segmentation_on_path()

            from segmentation import ensure_models_dir
            from segmentation.session import get_session
            from segmentation.utils import log_compute_backend

            # 2. Resolve the models dir and announce compute target.
            home = ensure_models_dir()
            log.info(f"[SegmentationBiRefNet] ai-models dir: {home}")
            log_compute_backend(log)

            # 3. Validate inputs.
            sfm_data = chunk.node.input.value
            views = self._read_views(sfm_data, chunk)
            log.info(f"[SegmentationBiRefNet] {len(views)} views to process")

            out_dir = Path(chunk.node.output.value)
            out_dir.mkdir(parents=True, exist_ok=True)

            keep_filename = bool(chunk.node.keepFilename.value)
            mask_ext = chunk.node.maskFormat.value
            model_variant = chunk.node.modelVariant.value

            # 4. Load CoreML session (cached across chunk calls).
            t0 = time.time()
            sess = get_session(model_variant)
            log.info(
                f"[SegmentationBiRefNet] session ready in "
                f"{time.time() - t0:.2f}s (variant={model_variant})"
            )

            # 5. Iterate views.
            import numpy as np
            from PIL import Image

            processed = 0
            skipped = 0
            failures: list[str] = []

            for idx, (view_id, image_path) in enumerate(views, start=1):
                dest = self._mask_path(
                    out_dir, view_id, image_path, keep_filename, mask_ext
                )

                if dest.exists() and dest.stat().st_size > 0:
                    skipped += 1
                    log.debug(f"  [{idx}/{len(views)}] skip (cached): {dest.name}")
                    continue

                if not os.path.isfile(image_path):
                    log.warning(
                        f"  [{idx}/{len(views)}] missing image: {image_path}"
                    )
                    failures.append(image_path)
                    continue

                try:
                    t_img = time.time()
                    src = Image.open(image_path).convert("RGB")
                    src_arr = np.asarray(src, dtype=np.uint8)
                    mask = sess.predict(src_arr)
                    self._save_mask(mask, dest, mask_ext)
                    processed += 1
                    log.info(
                        f"  [{idx}/{len(views)}] {Path(image_path).name} -> "
                        f"{dest.name} ({time.time() - t_img:.2f}s)"
                    )
                except Exception as exc:
                    log.error(
                        f"  [{idx}/{len(views)}] failed: {image_path}: {exc}"
                    )
                    failures.append(image_path)

            log.info(
                f"[SegmentationBiRefNet] done: processed={processed}, "
                f"skipped={skipped}, failed={len(failures)}, "
                f"total={len(views)}"
            )
            if failures and processed == 0:
                raise RuntimeError(
                    f"SegmentationBiRefNet produced no masks "
                    f"({len(failures)} failures)."
                )
        finally:
            chunk.logManager.end()
