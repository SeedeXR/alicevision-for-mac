"""
segmentation.utils

Small helpers shared by the session loader and the Meshroom node.
Stdlib + coremltools only — these run during node startup before any
heavyweight imports happen.
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


def coremltools_version() -> str:
    """Return the installed coremltools version, or '' if unavailable."""
    try:
        import coremltools as ct  # type: ignore
        return str(ct.__version__)
    except Exception:
        return ""


def log_compute_backend(log: logging.Logger | None = None,
                        prefix: str = "[SegmentationBiRefNet]") -> dict:
    """
    Emit a one-shot summary of the active compute target so users can verify
    Metal dispatch from the Meshroom log.

    The node ALWAYS targets `MLComputeUnits.cpuAndGPU`. The Apple Neural
    Engine is not viable for BiRefNet's `ASPPDeformable` decoder — see
    `models/production_note.md`. If you see the ANE in the log here, the
    log line is lying.

    Returns a dict with the introspected values (useful for tests).
    """
    log = log or logging.getLogger("segmentation")
    info: dict = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "chip": _chip_brand(),
        "coremltools": coremltools_version(),
    }

    if info["platform"] != "Darwin":
        log.warning(f"{prefix} Not running on macOS — CoreML unavailable.")
        info["compute_target"] = "UNAVAILABLE"
        return info

    if info["chip"]:
        log.info(f"{prefix} Host chip: {info['chip']}")

    if not info["coremltools"]:
        log.error(
            f"{prefix} coremltools is not importable — install it in the "
            f"meshroom-venv before running this node."
        )
        info["compute_target"] = "UNAVAILABLE"
        return info

    log.info(
        f"{prefix} Compute target: CoreML (CPU + GPU dispatch, "
        f"coremltools {info['coremltools']})"
    )
    info["compute_target"] = "CoreML"
    return info


__all__ = ["expand", "coremltools_version", "log_compute_backend"]
