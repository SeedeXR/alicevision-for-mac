"""
segmentation.utils

Small helpers shared by the session loader, the CoreML converter, and the
Meshroom node. Stdlib-only by design — these run during node startup
before any heavyweight imports happen.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path


# --------------------------------------------------------------------------- #
# Path / env
# --------------------------------------------------------------------------- #

def expand(path: str | os.PathLike) -> Path:
    """Expand `~`, `$VARS`, and resolve symlinks/absolutes."""
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


# --------------------------------------------------------------------------- #
# Compute backend introspection
# --------------------------------------------------------------------------- #

def _chip_brand() -> str:
    """Return the human-readable CPU brand string on macOS, or '' elsewhere."""
    if platform.system() != "Darwin":
        return ""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def available_onnx_providers() -> list[str]:
    """Return the list of ONNX Runtime execution providers, or [] if missing."""
    try:
        import onnxruntime as ort  # type: ignore
        return list(ort.get_available_providers())
    except Exception:
        return []


def log_compute_backend(log: logging.Logger | None = None,
                        prefix: str = "[SegmentationBiRefNet]") -> dict:
    """
    Emit a one-shot summary of the active compute target so users can verify
    Metal/ANE dispatch from the Meshroom log.

    Returns a dict with the introspected values (useful for tests).
    """
    log = log or logging.getLogger("segmentation")
    info: dict = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "chip": _chip_brand(),
        "providers": available_onnx_providers(),
    }

    if info["platform"] != "Darwin":
        log.warning(f"{prefix} Not running on macOS — CoreML unavailable.")
        info["compute_target"] = "CPU"
        return info

    if info["chip"]:
        log.info(f"{prefix} Host chip: {info['chip']}")

    if "CoreMLExecutionProvider" in info["providers"]:
        log.info(f"{prefix} Compute target: CoreML (CPU+GPU+ANE)")
        info["compute_target"] = "CoreML"
    else:
        log.warning(
            f"{prefix} CoreMLExecutionProvider NOT available — "
            f"falling back to CPU. Providers: {info['providers']}"
        )
        info["compute_target"] = "CPU"

    return info


__all__ = ["expand", "available_onnx_providers", "log_compute_backend"]
