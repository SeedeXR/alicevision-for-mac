"""
segmentation.session

CoreML-first rembg session loader.

Strategy:
  1. Force `U2NET_HOME` to <repo>/ai-models/ unless the user overrode it.
  2. Build a `rembg.new_session(model_name)` with the provider order
     [CoreMLExecutionProvider, CPUExecutionProvider] so Apple Silicon
     dispatches to the GPU + Neural Engine via the CoreML EP.
  3. Cache the loaded session for the lifetime of the Python process so
     Meshroom chunk re-entry does not re-pay the model-load cost.
  4. Gracefully fall back to the default rembg provider list (CPU) if
     onnxruntime is built without the CoreML EP.

The CoreML EP autopath is the canonical way to use Metal/ANE from ONNX
Runtime on macOS — we do NOT pre-convert to .mlpackage here; that path
exists as an optional optimisation in `convert_to_coreml.py`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from . import ensure_u2net_home

log = logging.getLogger(__name__)


# Module-level cache. Survives across Meshroom chunk calls in the same
# process — critical because loading a 973 MB BiRefNet ONNX takes 5-10s.
_SESSION_CACHE: dict[str, Any] = {}


# Map of friendly CLI aliases -> rembg internal session names. Mirrors the
# alias map in `scripts/download_models.py`. Users may pass either form.
_ALIASES: dict[str, str] = {
    "birefnet-lite": "birefnet-general-lite",
}


# Provider preference order: CoreML first (CPU+GPU+ANE), then CPU.
PROVIDERS_PREFERRED: list[str] = [
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]


def _force_cpu_requested() -> bool:
    """Return True if `ONNX_FORCE_CPU=1` is set in the environment.

    Escape hatch added in S53 follow-up: on macOS 26.5 + onnxruntime 1.26
    + M-series, the CPU EP is the empirically-fastest steady-state path
    for swin-transformer BiRefNet (~6.7s/image lite, ~10s general/dis).
    CoreML EP with `CPUAndGPU` is ~35× slower (Metal command-buffer
    thrashing); `MLComputeUnits=ALL` requires a 15-25 min ANE compile.
    Setting `ONNX_FORCE_CPU=1` bypasses the CoreML preference entirely
    and pins the session to `CPUExecutionProvider`. See
    `memory/perf_segmentation_s52.md` for measurements.
    """
    val = os.environ.get("ONNX_FORCE_CPU", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _resolve_session_name(model_name: str) -> str:
    """Translate user-facing aliases to rembg's internal session names."""
    return _ALIASES.get(model_name, model_name)


def _available_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception as exc:
        log.warning(f"onnxruntime unavailable: {exc}")
        return []


def _filter_providers(preferred: list[str]) -> list[str]:
    # `ONNX_FORCE_CPU=1` short-circuits to CPU-only regardless of preference.
    if _force_cpu_requested():
        log.info(
            "[SegmentationBiRefNet] ONNX_FORCE_CPU=1 — pinning to "
            "CPUExecutionProvider (skipping CoreML EP)."
        )
        return ["CPUExecutionProvider"]
    available = set(_available_providers())
    filtered = [p for p in preferred if p in available]
    if not filtered:
        filtered = ["CPUExecutionProvider"]
    return filtered


def get_session(model_name: str = "birefnet-general"):
    """Return a cached rembg session, creating it if needed.

    Honours `U2NET_HOME`. Tries CoreML EP first, falls back to CPU.
    Raises only if even the CPU fallback fails (which usually means the
    ONNX file is missing — run `scripts/download_models.py` first).
    """
    session_name = _resolve_session_name(model_name)
    if session_name in _SESSION_CACHE:
        return _SESSION_CACHE[session_name]

    ensure_u2net_home()
    home = os.environ["U2NET_HOME"]
    log.info(f"[SegmentationBiRefNet] U2NET_HOME={home}")

    # Import lazily — rembg pulls in scikit-image / numba which add ~2s
    # to startup if loaded eagerly at module import time.
    from rembg import new_session

    providers = _filter_providers(PROVIDERS_PREFERRED)
    log.info(
        f"[SegmentationBiRefNet] Loading model '{session_name}' "
        f"with providers={providers}"
    )

    try:
        sess = new_session(session_name, providers=providers)
        if "CoreMLExecutionProvider" in providers:
            log.info("[SegmentationBiRefNet] CoreML EP active "
                     "(CPU + GPU + ANE dispatch)")
        else:
            log.warning("[SegmentationBiRefNet] CoreML EP unavailable — "
                        "running CPU-only.")
    except Exception as exc:
        log.warning(
            f"[SegmentationBiRefNet] CoreML session load failed ({exc}); "
            "retrying with default (CPU) provider list."
        )
        sess = new_session(session_name)

    _SESSION_CACHE[session_name] = sess
    return sess


def clear_session_cache() -> None:
    """Drop all cached sessions — primarily a hook for tests."""
    _SESSION_CACHE.clear()


# ---------------------------------------------------------------------------
# Profiling-only helper: bypass rembg's BaseSession.__init__ so we can pass
# `provider_options` (e.g. `MLComputeUnits=CPUAndGPU`) to the CoreML EP.
#
# rembg's `BaseSession.__init__` only forwards `providers=...` to
# `ort.InferenceSession(...)`. To pin the CoreML EP to CPU+GPU (skipping the
# 15-25 min `ANECompilerService` step) we must construct the InferenceSession
# ourselves and graft it onto a duck-typed rembg session shell.
#
# This is NOT used in the production node path — it exists for
# `scripts/profile_segmentation_light.py` only.
# ---------------------------------------------------------------------------

_REMBG_SESSION_CLASSES: dict[str, str] = {
    # variant alias -> (module, class_name)
    "birefnet-general": "rembg.sessions.birefnet_general:BiRefNetSessionGeneral",
    "birefnet-general-lite": "rembg.sessions.birefnet_general_lite:BiRefNetSessionGeneralLite",
    "birefnet-lite": "rembg.sessions.birefnet_general_lite:BiRefNetSessionGeneralLite",
    "birefnet-dis": "rembg.sessions.birefnet_dis:BiRefNetSessionDIS",
}


def _onnx_path_for_variant(model_name: str) -> str:
    """Return the on-disk ONNX path for a variant, using U2NET_HOME."""
    rembg_name = _resolve_session_name(model_name)
    home = os.environ.get("U2NET_HOME") or str(ensure_u2net_home())
    return os.path.join(home, f"{rembg_name}.onnx")


def get_session_with_compute_units(
    model_name: str = "birefnet-general",
    compute_units: str = "CPUAndGPU",
    use_cache: bool = False,
):
    """Build a rembg-compatible session with an explicit CoreML EP MLComputeUnits.

    `compute_units` is forwarded to the CoreML EP via `provider_options` and
    must be one of: `CPUOnly`, `CPUAndGPU`, `CPUAndNeuralEngine`, `ALL`.
    Using `CPUAndGPU` is the documented way to keep inference on Metal GPU
    while bypassing the ANE compile step.

    The returned object exposes `.predict(img)` like a rembg session, so it
    can be passed to `rembg.remove(img, session=...)`. By default the
    profile cache is bypassed — pass `use_cache=True` to share across calls.
    """
    import importlib

    import onnxruntime as ort

    ensure_u2net_home()
    rembg_name = _resolve_session_name(model_name)

    cache_key = f"{rembg_name}::{compute_units}"
    if use_cache and cache_key in _SESSION_CACHE:
        return _SESSION_CACHE[cache_key]

    spec = _REMBG_SESSION_CLASSES.get(model_name) or _REMBG_SESSION_CLASSES.get(rembg_name)
    if spec is None:
        raise ValueError(
            f"Unknown BiRefNet variant {model_name!r}; "
            f"expected one of {sorted(_REMBG_SESSION_CLASSES)}"
        )
    module_name, class_name = spec.split(":")
    mod = importlib.import_module(module_name)
    SessCls = getattr(mod, class_name)

    onnx_path = _onnx_path_for_variant(model_name)
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(
            f"ONNX not found at {onnx_path}; run scripts/download_models.py"
        )

    available = set(_available_providers())
    providers: list[str] = []
    provider_options: list[dict] = []
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
        provider_options.append({"MLComputeUnits": compute_units})
    providers.append("CPUExecutionProvider")
    provider_options.append({})

    log.info(
        f"[SegmentationBiRefNet] (profile) Loading '{rembg_name}' with "
        f"providers={providers} provider_options={provider_options}"
    )

    sess_opts = ort.SessionOptions()
    inner = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=providers,
        provider_options=provider_options,
    )

    # Build a rembg-shaped wrapper. We bypass BaseSession.__init__ on purpose
    # so it does not try to re-create the inner_session.
    wrapper = SessCls.__new__(SessCls)
    wrapper.model_name = rembg_name
    wrapper.inner_session = inner

    if use_cache:
        _SESSION_CACHE[cache_key] = wrapper
    return wrapper


__all__ = [
    "get_session",
    "get_session_with_compute_units",
    "clear_session_cache",
    "PROVIDERS_PREFERRED",
]
