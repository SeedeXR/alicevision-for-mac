"""
ai-segmentation plugin: segmentation package.

CoreML-only BiRefNet inference helpers for the `SegmentationBiRefNet`
Meshroom node:

  * session.py   CoreML `.mlpackage` loader + per-process cache
  * utils.py     compute-backend logging helpers

The session loader reads the pre-converted `.mlpackage` files from
`<repo>/ai-models/`. The legacy rembg/ONNX backend was removed
2026-05-23 — see `models/production_note.md` for the perf rationale.
"""

from __future__ import annotations

import os
from pathlib import Path

# Module-level path constants. The package lives at
# `<repo>/plugins/ai-segmentation/python/segmentation/__init__.py`.
# Walk up FOUR parents to reach the repo root:
#   segmentation -> python -> ai-segmentation -> plugins -> <repo>
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[3]
AI_MODELS_DIR = REPO_ROOT / "ai-models"


def ensure_models_dir() -> Path:
    """Return the resolved AI-models directory, honouring an env override.

    `AV_AI_MODELS_DIR` overrides the default `<repo>/ai-models/`. `U2NET_HOME`
    is also honoured for backwards-compat with operators who set it during
    the rembg era — but new deployments should prefer `AV_AI_MODELS_DIR`.
    """
    override = os.environ.get("AV_AI_MODELS_DIR") or os.environ.get("U2NET_HOME")
    home = Path(override).expanduser() if override else AI_MODELS_DIR
    home.mkdir(parents=True, exist_ok=True)
    return home.resolve()


__all__ = ["REPO_ROOT", "AI_MODELS_DIR", "ensure_models_dir"]
