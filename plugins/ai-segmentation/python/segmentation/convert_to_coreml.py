"""
segmentation.convert_to_coreml

Optional: convert a downloaded BiRefNet ONNX file to CoreML .mlpackage
format and cache it beside the .onnx. The primary inference path uses
onnxruntime's CoreML EP directly — this conversion is an extra
optimisation that exposes the model as a first-class CoreML graph and is
useful for native Swift integration (S52 task #97).

The conversion is run lazily on demand. If `coremltools.convert()` fails
to lower a particular op onto ANE, we retry with `CPU_AND_GPU` (Metal
only). We NEVER silently fall back to CPU-only — that is logged as an
ERROR so the user knows acceleration is degraded.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

from . import AI_MODELS_DIR

log = logging.getLogger(__name__)


def _resolve_onnx_path(model_name: str) -> Path:
    """Look up the on-disk ONNX path for a given rembg model name."""
    candidates = [
        AI_MODELS_DIR / f"{model_name}.onnx",
        Path(os.environ.get("U2NET_HOME", AI_MODELS_DIR)) / f"{model_name}.onnx",
        Path.home() / ".u2net" / f"{model_name}.onnx",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"ONNX model '{model_name}' not found. Searched: "
        + ", ".join(str(c) for c in candidates)
        + ". Run scripts/download_models.py first."
    )


def _mlpackage_path(onnx_path: Path) -> Path:
    """`.../foo.onnx` -> `.../foo.mlpackage` (beside the source ONNX)."""
    return onnx_path.with_suffix(".mlpackage")


def get_coreml_model(model_name: str = "birefnet-general",
                     force_rebuild: bool = False):
    """
    Lazily convert ONNX -> CoreML .mlpackage, cache it, and return the
    loaded CoreML model.

    Args:
        model_name: rembg-internal session name (e.g. "birefnet-general",
                    "birefnet-general-lite", "birefnet-dis").
        force_rebuild: if True, delete and re-build the .mlpackage.

    Returns:
        A `coremltools.models.MLModel` instance, loaded with the best
        available compute units (ALL -> CPU+GPU+ANE; CPU_AND_GPU
        fallback; never CPU-only without an explicit ERROR log).

    Raises:
        FileNotFoundError if the ONNX source is not staged.
        Whatever coremltools raises if both conversion attempts fail.
    """
    import coremltools as ct  # heavy import — keep local

    onnx_path = _resolve_onnx_path(model_name)
    cml_path = _mlpackage_path(onnx_path)

    if force_rebuild and cml_path.exists():
        log.info(f"[CoreML] removing stale cache at {cml_path}")
        # .mlpackage is a directory — handle both dir and file safely
        if cml_path.is_dir():
            import shutil
            shutil.rmtree(cml_path)
        else:
            cml_path.unlink()

    if not cml_path.exists():
        log.info(f"[CoreML] converting {onnx_path.name} -> {cml_path.name}")
        t0 = time.time()
        compute_units = ct.ComputeUnit.ALL
        try:
            mlmodel = ct.convert(
                str(onnx_path),
                convert_to="mlprogram",
                compute_units=compute_units,
            )
        except Exception as exc_all:
            log.warning(
                f"[CoreML] ComputeUnit.ALL conversion failed ({exc_all}); "
                "retrying with CPU_AND_GPU (Metal only, no ANE)."
            )
            compute_units = ct.ComputeUnit.CPU_AND_GPU
            try:
                mlmodel = ct.convert(
                    str(onnx_path),
                    convert_to="mlprogram",
                    compute_units=compute_units,
                )
            except Exception as exc_gpu:
                log.error(
                    f"[CoreML] Metal-only conversion ALSO failed: {exc_gpu}. "
                    "Refusing to silently downgrade to CPU-only. "
                    "Use the onnxruntime CoreML EP path via "
                    "segmentation.session.get_session() instead."
                )
                raise
        mlmodel.save(str(cml_path))
        log.info(
            f"[CoreML] saved {cml_path} in {time.time() - t0:.1f}s "
            f"(compute_units={compute_units!r})"
        )

    # Re-open with the best compute units available. Loading is idempotent
    # and cheap — coremltools mmaps the underlying mlmodelc.
    return ct.models.MLModel(str(cml_path), compute_units=ct.ComputeUnit.ALL)


__all__ = ["get_coreml_model"]
