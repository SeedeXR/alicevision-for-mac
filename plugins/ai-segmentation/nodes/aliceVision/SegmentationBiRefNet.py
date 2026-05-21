__version__ = "1.0"

import json
import logging
import os
import sys
import time
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class SegmentationBiRefNet(desc.Node):
    """
Generate a binary foreground mask per view using BiRefNet via rembg.

This is a Mac-native, in-process AI segmentation node that runs entirely
through onnxruntime's CoreML Execution Provider — dispatching to the
Apple Silicon GPU and Neural Engine. There is no CUDA, no PyTorch, and
no subprocess invocation: inference runs in the Meshroom worker process.

Pre-flight:
    1. Activate `meshroom-venv/`.
    2. Run `python scripts/download_models.py` to stage the ONNX weights
       under `<repo>/ai-models/`.

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
                "BiRefNet ONNX checkpoint. Swap freely between runs — the "
                "per-model ANE bundle is cached after the first compile.\n\n"
                " - birefnet-general (~927 MB, swin_v1_large): default for "
                "everyday photogrammetry. Best general accuracy/speed ratio "
                "on rigid subjects, turntable, mixed indoor/outdoor.\n"
                " - birefnet-dis (~929 MB, swin_v1_large DIS-trained): "
                "specialized for hair, fur, fine foliage, transparent or "
                "mesh-like edges. ~2-5% slower than 'general' for noticeably "
                "better complex-edge masks.\n"
                " - birefnet-lite (~213 MB, swin_v1_tiny): 4x smaller, 2-4x "
                "faster on CoreML+ANE. Recommended for M1/M2 base (8 GB UMA) "
                "or large batch jobs where wall-clock matters more than the "
                "last few percent of mask precision.\n\n"
                "Switching variants mid-pipeline invalidates the per-node "
                "Meshroom cache (UID changes) — masks regenerate on next "
                "compute. The Python session cache keeps the prior model in "
                "RAM until the worker exits."
            ),
            value="birefnet-general",
            values=["birefnet-general", "birefnet-dis", "birefnet-lite"],
            exposed=True,
        ),
        desc.ChoiceParam(
            name="outputResolution",
            label="Output Resolution",
            description="Mask output longest-edge resolution. Higher = sharper "
                        "edges but more memory.",
            value="1024",
            values=["512", "1024", "2048"],
        ),
        desc.BoolParam(
            name="alphaMatting",
            label="Alpha Matting",
            description="Run rembg's alpha-matting post-pass for refined edges. "
                        "Slower; useful for hair/fur/foliage.",
            value=False,
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

    def _ensure_segmentation_on_path(self, chunk):
        """Make the in-repo `segmentation` package importable.

        After the S53 plugin refactor the node descriptor lives at
        `plugins/ai-segmentation/nodes/aliceVision/SegmentationBiRefNet.py`
        and the helpers at `plugins/ai-segmentation/python/segmentation/`.
        We add the plugin's `python/` dir to sys.path so
        `import segmentation` works regardless of how Meshroom or the
        `run_python_node.sh` wrapper was launched.

        A back-compat symlink at `meshroom-mac/nodes/aliceVision/...`
        resolves through `Path.resolve()` so legacy import paths still
        hit the same plugin tree.
        """
        here = Path(__file__).resolve().parent  # .../nodes/aliceVision
        # nodes/aliceVision -> nodes -> ai-segmentation (plugin root)
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

        views = data.get("views", [])
        results: list[tuple[str, str]] = []
        for v in views:
            view_id = str(v.get("viewId", ""))
            path = v.get("path", "")
            if not view_id or not path:
                chunk.logger.warning(
                    f"Skipping malformed view entry: {v!r}"
                )
                continue
            results.append((view_id, path))
        return results

    def _mask_path(self, out_dir: Path, view_id: str, image_path: str,
                   keep_filename: bool, ext: str) -> Path:
        if keep_filename:
            stem = Path(image_path).stem
            return out_dir / f"{stem}_mask.{ext}"
        return out_dir / f"{view_id}.{ext}"

    def _save_mask(self, rgba_image, dest: Path, ext: str) -> None:
        """Extract alpha channel and save as PNG or EXR."""
        from PIL import Image

        if rgba_image.mode != "RGBA":
            rgba_image = rgba_image.convert("RGBA")
        alpha = rgba_image.split()[-1]  # 'L' mode, 0..255 mask

        dest.parent.mkdir(parents=True, exist_ok=True)
        if ext == "png":
            alpha.save(dest, format="PNG", optimize=True)
            return
        if ext == "exr":
            # EXR via imageio (already a rembg dep). Normalise to 0..1 float32.
            import numpy as np
            import imageio.v3 as iio
            arr = np.asarray(alpha, dtype=np.float32) / 255.0
            iio.imwrite(str(dest), arr, extension=".exr")
            return
        raise RuntimeError(f"Unsupported mask format: {ext}")

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def processChunk(self, chunk):
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            log = chunk.logger

            # 1. Bring our helper package onto sys.path
            self._ensure_segmentation_on_path(chunk)

            from segmentation import ensure_u2net_home
            from segmentation.session import get_session
            from segmentation.utils import log_compute_backend

            # 2. Set U2NET_HOME and log compute backend (Metal/ANE check)
            ensure_u2net_home()
            backend = log_compute_backend(log)
            log.info(f"[SegmentationBiRefNet] backend={backend['compute_target']}, "
                     f"providers={backend['providers']}")

            # 3. Validate inputs
            sfm_data = chunk.node.input.value
            views = self._read_views(sfm_data, chunk)
            log.info(f"[SegmentationBiRefNet] {len(views)} views to process")

            out_dir = Path(chunk.node.output.value)
            out_dir.mkdir(parents=True, exist_ok=True)

            keep_filename = bool(chunk.node.keepFilename.value)
            mask_ext = chunk.node.maskFormat.value
            alpha_matting = bool(chunk.node.alphaMatting.value)
            try:
                out_res = int(chunk.node.outputResolution.value)
            except (TypeError, ValueError):
                out_res = 1024

            # 4. Load session (cached across chunk calls in same process)
            model_variant = chunk.node.modelVariant.value
            t0 = time.time()
            sess = get_session(model_variant)
            log.info(f"[SegmentationBiRefNet] session ready in "
                     f"{time.time() - t0:.2f}s")

            # 5. Iterate views
            from PIL import Image
            from rembg import remove

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
                    log.warning(f"  [{idx}/{len(views)}] missing image: "
                                f"{image_path}")
                    failures.append(image_path)
                    continue

                try:
                    t_img = time.time()
                    src = Image.open(image_path)
                    # Optionally resize the longest edge before inference to
                    # control memory; BiRefNet handles arbitrary input but
                    # rembg upscales output back to source size.
                    if out_res > 0 and max(src.size) > out_res:
                        ratio = out_res / float(max(src.size))
                        new_size = (
                            int(src.size[0] * ratio),
                            int(src.size[1] * ratio),
                        )
                        src_small = src.resize(new_size, Image.Resampling.LANCZOS)
                    else:
                        src_small = src

                    result = remove(
                        src_small,
                        session=sess,
                        alpha_matting=alpha_matting,
                    )
                    # Re-scale alpha back to original image dims so the mask
                    # aligns 1:1 with the source pixels (FeatureExtraction
                    # requires identical dimensions).
                    if result.size != src.size:
                        result = result.resize(src.size, Image.Resampling.LANCZOS)

                    self._save_mask(result, dest, mask_ext)
                    processed += 1
                    log.info(
                        f"  [{idx}/{len(views)}] {Path(image_path).name} -> "
                        f"{dest.name} ({time.time() - t_img:.2f}s)"
                    )
                except Exception as exc:
                    log.error(f"  [{idx}/{len(views)}] failed: "
                              f"{image_path}: {exc}")
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
