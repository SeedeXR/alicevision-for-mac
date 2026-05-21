"""
ai-segmentation plugin: segmentation package.

Helpers for the in-process AI segmentation node:
  * session.py            CoreML-first rembg session loader
  * convert_to_coreml.py  ONNX -> CoreML .mlpackage caching
  * utils.py              compute-backend logging helpers

Models are staged under <repo_root>/ai-models/ — pre-download with
`plugins/ai-segmentation/scripts/download_models.py`. The session loader
honours `U2NET_HOME` which rembg also consults; we default it to the
project ai-models/ dir so installs do not pollute ~/.u2net/.
"""

from __future__ import annotations

import os
from pathlib import Path

# Module-level path constants. After the S53 plugin refactor the package
# lives at `<repo>/plugins/ai-segmentation/python/segmentation/__init__.py`.
# Walk up FOUR parents to reach the repo root:
#   segmentation -> python -> ai-segmentation -> plugins -> <repo>
# A back-compat symlink at `<repo>/meshroom-mac/segmentation` resolves
# through `Path.resolve()` to the canonical plugin location, so this works
# whether the package is imported via the symlink or directly.
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parents[3]
AI_MODELS_DIR = REPO_ROOT / "ai-models"


def ensure_u2net_home() -> Path:
    """Set `U2NET_HOME` to the project ai-models/ dir if not already set.

    Safe to call multiple times. Returns the resolved cache path.
    """
    home = os.environ.get("U2NET_HOME")
    if not home:
        os.environ["U2NET_HOME"] = str(AI_MODELS_DIR)
        home = str(AI_MODELS_DIR)
    Path(home).mkdir(parents=True, exist_ok=True)
    return Path(home)


__all__ = ["REPO_ROOT", "AI_MODELS_DIR", "ensure_u2net_home"]
