"""
segmentation.session

CoreML-only BiRefNet session loader for `SegmentationBiRefNet`.

The session opens one of the pre-converted `.mlpackage` files in
`<repo>/ai-models/` and returns a thin wrapper exposing a single
`predict(rgb_uint8_HxWx3)` method that returns a `[H, W]` float32 mask
in `[0, 1]`.

Why CoreML-only:
  * The rembg + ONNX Runtime path was removed 2026-05-23. On Apple
    Silicon it was ~10–35× slower than the FP16 mlpackage path (Metal
    command-buffer thrashing in ONNX Runtime's CoreML EP for swin-v1
    graphs), and its CPU-only fallback ran at 6–10 s / frame.
  * The pre-converted mlpackage models load at `cpuAndGPU` in 3–5 s and
    predict in 350–980 ms per 1024² frame.

Hard rule — load with `MLComputeUnits.cpuAndGPU`. Do NOT pass `.all`
or `.cpuAndNeuralEngine`: BiRefNet's `ASPPDeformable` decoder lowers
via `grid_sample`, which the ANE compiler cannot plan; the load call
hangs in `com.apple.anef.p3` forever. See `models/production_note.md`
for the full diagnosis.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import ensure_models_dir

log = logging.getLogger(__name__)


# Module-level cache. Survives across Meshroom chunk calls in the same
# process — critical because loading the 447 MB general mlpackage takes
# ~5 s and JIT-compiling the Metal pipelines on first predict adds
# another second on top.
_SESSION_CACHE: dict[str, "BiRefNetCoreMLSession"] = {}


# User-facing variant id -> mlpackage filename under ai-models/.
VARIANT_PACKAGES: dict[str, str] = {
    "birefnet-lite": "BiRefNet_lite.mlpackage",
    "birefnet-general": "BiRefNet.mlpackage",
}

# ImageNet preprocessing constants — fixed by the BiRefNet checkpoints.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fixed input resolution baked into the mlpackage during conversion.
INPUT_HW = 1024


class BiRefNetCoreMLSession:
    """Thin runtime wrapper around a CoreML `MLModel`.

    Exposes `predict(rgb_uint8)` which:
      1. Resizes to `INPUT_HW × INPUT_HW` via bilinear (PIL).
      2. Normalizes with ImageNet stats, produces `[1, 3, 1024, 1024]` float32.
      3. Calls `MLModel.predict` (CPU + Metal GPU).
      4. Returns the sigmoid mask resized back to the source `(H, W)`.

    The caller does anything else (alpha compositing, PNG/EXR serialization).
    """

    def __init__(self, variant: str, mlmodel: Any, package_path: Path):
        self.variant = variant
        self.mlmodel = mlmodel
        self.package_path = package_path

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run a single image through the model.

        Args:
            image_rgb: `[H, W, 3]` uint8 RGB.

        Returns:
            `[H, W]` float32 mask in `[0, 1]` at the *source* resolution.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"expected HxWx3 RGB, got shape {image_rgb.shape}"
            )
        if image_rgb.dtype != np.uint8:
            raise ValueError(f"expected uint8, got dtype {image_rgb.dtype}")

        src_h, src_w = image_rgb.shape[:2]

        # Resize source to the fixed model input. PIL bilinear here to keep
        # this stdlib-light; we only need numpy + Pillow.
        from PIL import Image

        pil = Image.fromarray(image_rgb, mode="RGB")
        if pil.size != (INPUT_HW, INPUT_HW):
            pil_resized = pil.resize(
                (INPUT_HW, INPUT_HW), Image.Resampling.BILINEAR
            )
        else:
            pil_resized = pil

        # HWC uint8 -> CHW float32, ImageNet-normalized.
        arr = np.asarray(pil_resized, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        arr = np.transpose(arr, (2, 0, 1))[None, ...]  # [1, 3, 1024, 1024]
        arr = np.ascontiguousarray(arr, dtype=np.float32)

        out = self.mlmodel.predict({"input": arr})
        mask_chw = out["mask"]  # [1, 1, 1024, 1024] float32 in [0,1]
        mask = np.asarray(mask_chw, dtype=np.float32).squeeze()  # [1024, 1024]
        mask = np.clip(mask, 0.0, 1.0)

        # Resize back to source resolution. Use PIL bilinear on a single-
        # channel image to match the upscaling used by training-time
        # rembg pipelines.
        mask_pil = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L")
        if mask_pil.size != (src_w, src_h):
            mask_pil = mask_pil.resize(
                (src_w, src_h), Image.Resampling.BILINEAR
            )
        return np.asarray(mask_pil, dtype=np.float32) / 255.0


def _resolve_package_path(variant: str) -> Path:
    """Return the absolute `.mlpackage` path for a variant, or raise."""
    if variant not in VARIANT_PACKAGES:
        raise ValueError(
            f"unknown BiRefNet variant {variant!r}; "
            f"expected one of {sorted(VARIANT_PACKAGES)}"
        )
    home = ensure_models_dir()
    path = (home / VARIANT_PACKAGES[variant]).resolve()
    if not path.is_dir() or not (path / "Manifest.json").is_file():
        raise FileNotFoundError(
            f"BiRefNet CoreML package missing at {path}. "
            f"Convert it with `python models/convert/convert_to_coreml.py "
            f"{variant.replace('birefnet-', '')}` (see ai-models/README.md)."
        )
    return path


def get_session(variant: str = "birefnet-lite") -> BiRefNetCoreMLSession:
    """Return a cached CoreML session, creating it if needed.

    Loads the model with `MLComputeUnits.cpuAndGPU`. Issues one warm-up
    predict on a synthetic input so the Metal pipelines are JIT-compiled
    before the first real frame.
    """
    if variant in _SESSION_CACHE:
        return _SESSION_CACHE[variant]

    pkg = _resolve_package_path(variant)
    log.info(
        f"[SegmentationBiRefNet] Loading {pkg.name} (cpuAndGPU)"
    )

    # Import lazily so this module is cheap to import in tests that don't
    # actually load a model.
    import coremltools as ct

    mlmodel = ct.models.MLModel(
        str(pkg),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    # Warm-up. The first predict is much slower than steady state because
    # CoreML JIT-compiles the Metal pipelines lazily.
    warmup = np.zeros((1, 3, INPUT_HW, INPUT_HW), dtype=np.float32)
    try:
        mlmodel.predict({"input": warmup})
    except Exception as exc:  # noqa: BLE001 — keep going; first predict can fail on garbage input
        log.warning(f"[SegmentationBiRefNet] warmup predict failed (ignored): {exc}")

    sess = BiRefNetCoreMLSession(variant, mlmodel, pkg)
    _SESSION_CACHE[variant] = sess
    log.info(f"[SegmentationBiRefNet] Session ready for variant={variant}")
    return sess


def clear_session_cache() -> None:
    """Drop all cached sessions — primarily a hook for tests."""
    _SESSION_CACHE.clear()


__all__ = [
    "BiRefNetCoreMLSession",
    "VARIANT_PACKAGES",
    "INPUT_HW",
    "get_session",
    "clear_session_cache",
]
