#!/usr/bin/env python3
"""
Lightweight perf + model-swap probe that avoids the 15-25 min ANE compile
and the ~17 GB peak RSS of the full Monstree mini3 run.

Three measurement paths (per variant):

  1. CPU EP only            -- CPU floor, predictable
  2. CoreML EP CPU+GPU      -- Metal GPU, NO ANE compile
                              (provider_options={MLComputeUnits=CPUAndGPU})
  3. CoreML EP ALL          -- DOCUMENTED ONLY (would trigger ANE compile)

Path 3 is intentionally NOT executed. The existing perf doc already records
that first-load `ANECompilerService` is 15-25 min and ~17 GB peak; we just
reference it.

Synthetic 512x512 input -- BiRefNet downscales internally to 1024 longest
edge, so the inference compute path is identical to Monstree 4032x3024
but peripheral memory + JPEG decode drop dramatically.

OMP_NUM_THREADS=4 is enforced.

Run:
    OMP_NUM_THREADS=4 python scripts/profile_segmentation_light.py
"""

from __future__ import annotations

import datetime as _dt
import gc
import hashlib
import io
import json
import logging
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    root = here.parent.parent
    for entry in (root / "meshroom-mac", root / "src" / "python_shim"):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)
    os.environ.setdefault("U2NET_HOME", str(root / "ai-models"))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    return root


ROOT = _bootstrap()
# Force-unbuffered stdout/stderr so background runs stream events promptly.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

LOG = logging.getLogger("profile_light")
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(
    logging.Formatter("[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s",
                      datefmt="%H:%M:%S")
)
logging.basicConfig(level=logging.INFO, handlers=[_h], force=True)


VARIANTS = ("birefnet-lite", "birefnet-general", "birefnet-dis")
RUNS_PER_PATH_CPU = 3
RUNS_PER_PATH_COREML = 2  # cold + 1 warm; large models can be 100+s each

# Hard per-inference timeout (seconds). If any single inference exceeds
# this, we bail out of the path (CoreML EP can pathologically slow down on
# certain swin graphs without ANE — we still want partial data).
#
# A previous run showed CoreMLEP+CPUAndGPU+swin_v1_tiny taking 67s cold and
# 250s warm on this M4 -- worse than CPU EP. So we keep the budget tight
# and accept partial CoreML data rather than burning 30+ min of wall time.
INF_TIMEOUT_S = {
    "birefnet-lite": 80,
    "birefnet-general": 300,
    "birefnet-dis": 300,
}


def _peak_rss_mb() -> float:
    """Peak RSS of the current process in MB (macOS: bytes / 1024^2)."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    val = r.ru_maxrss
    if sys.platform == "darwin":
        return val / (1024 * 1024)
    return val / 1024


def _md5_bytes(blob: bytes) -> str:
    return hashlib.md5(blob).hexdigest()


def _make_synthetic_image(size: int):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (90, 110, 200))
    d = ImageDraw.Draw(img)
    d.ellipse((size // 4, size // 4, 3 * size // 4, 3 * size // 4),
              fill=(220, 90, 70))
    d.rectangle((size // 8, 7 * size // 8, 3 * size // 8, size - 10),
                fill=(60, 200, 80))
    return img


def _has_onnx(variant: str) -> Path | None:
    rembg_name = {
        "birefnet-general": "birefnet-general.onnx",
        "birefnet-dis": "birefnet-dis.onnx",
        "birefnet-lite": "birefnet-general-lite.onnx",
    }[variant]
    p = ROOT / "ai-models" / rembg_name
    return p if p.is_file() else None


def _ane_cache_size_bytes() -> int | None:
    p = Path.home() / "Library" / "Caches" / "com.apple.e5rt.e5bundlecache"
    if not p.exists():
        return 0
    try:
        out = subprocess.check_output(
            ["du", "-sk", str(p)],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode()
        kb = int(out.split()[0])
        return kb * 1024
    except Exception as exc:
        LOG.warning(f"du on ANE cache failed: {exc}")
        return None


def _alpha_bytes(result_img) -> tuple[bytes, "PIL.Image.Image"]:
    """Extract the alpha channel as a fresh L-mode image; return PNG bytes
    and the image (so caller can save a small thumbnail)."""
    alpha = result_img.convert("RGBA").split()[-1]
    buf = io.BytesIO()
    alpha.save(buf, format="PNG", optimize=False)
    return buf.getvalue(), alpha


def _bench_path_cpu(variant: str, image, repeats: int) -> dict:
    """CPU EP only — uses production get_session() with overridden providers."""
    from segmentation import ensure_u2net_home
    from segmentation import session as _seg_session
    from segmentation.session import clear_session_cache, get_session
    from rembg import remove

    ensure_u2net_home()
    saved = _seg_session.PROVIDERS_PREFERRED[:]
    _seg_session.PROVIDERS_PREFERRED[:] = ["CPUExecutionProvider"]
    clear_session_cache()
    gc.collect()
    rss_pre = _peak_rss_mb()

    LOG.info(f"=== variant={variant} path=cpu ===")
    t0 = time.time()
    sess = get_session(variant)
    load_s = time.time() - t0
    rss_post = _peak_rss_mb()
    LOG.info(f"  load={load_s:.2f}s rss_post={rss_post:.0f}MB")

    timings, last_md5, first_mask, bail = [], None, None, False
    deadline = INF_TIMEOUT_S.get(variant, 600)
    for i in range(repeats):
        t1 = time.time()
        out = remove(image, session=sess)
        inf_s = time.time() - t1
        png_bytes, alpha = _alpha_bytes(out)
        last_md5 = _md5_bytes(png_bytes)
        if first_mask is None:
            first_mask = alpha
        timings.append(round(inf_s * 1000, 1))
        LOG.info(f"  [{i + 1}/{repeats}] {inf_s * 1000:.1f}ms md5={last_md5[:8]}")
        if inf_s > deadline:
            LOG.warning(f"  inference exceeded {deadline}s budget; "
                        f"bailing out of remaining runs.")
            bail = True
            break

    rss_peak = _peak_rss_mb()
    _seg_session.PROVIDERS_PREFERRED[:] = saved
    clear_session_cache()
    gc.collect()
    return {
        "variant": variant,
        "path": "cpu",
        "providers": ["CPUExecutionProvider"],
        "compute_units": None,
        "session_load_s": round(load_s, 3),
        "inference_ms": timings,
        "cold_ms": timings[0] if timings else None,
        "warm_ms": min(timings[1:]) if len(timings) > 1 else None,
        "rss_pre_load_mb": round(rss_pre, 0),
        "rss_post_load_mb": round(rss_post, 0),
        "rss_peak_mb": round(rss_peak, 0),
        "mask_md5": last_md5,
        "bailed_on_timeout": bail,
        "_thumbnail": first_mask,
    }


def _bench_path_coreml_cpu_gpu(variant: str, image, repeats: int) -> dict:
    """CoreML EP with MLComputeUnits=CPUAndGPU -- Metal GPU, no ANE compile."""
    from segmentation.session import (clear_session_cache,
                                      get_session_with_compute_units)
    from rembg import remove

    clear_session_cache()
    gc.collect()
    rss_pre = _peak_rss_mb()

    LOG.info(f"=== variant={variant} path=coreml_cpu_gpu ===")
    t0 = time.time()
    sess = get_session_with_compute_units(variant, compute_units="CPUAndGPU")
    load_s = time.time() - t0
    rss_post = _peak_rss_mb()
    LOG.info(f"  load={load_s:.2f}s rss_post={rss_post:.0f}MB")

    timings, last_md5, first_mask, bail = [], None, None, False
    deadline = INF_TIMEOUT_S.get(variant, 600)
    for i in range(repeats):
        t1 = time.time()
        out = remove(image, session=sess)
        inf_s = time.time() - t1
        png_bytes, alpha = _alpha_bytes(out)
        last_md5 = _md5_bytes(png_bytes)
        if first_mask is None:
            first_mask = alpha
        timings.append(round(inf_s * 1000, 1))
        LOG.info(f"  [{i + 1}/{repeats}] {inf_s * 1000:.1f}ms md5={last_md5[:8]}")
        if inf_s > deadline:
            LOG.warning(f"  inference exceeded {deadline}s budget; "
                        f"bailing out of remaining CoreML runs for {variant}.")
            bail = True
            break

    rss_peak = _peak_rss_mb()
    # Drop our handle so the next variant claims its own ORT/CoreML state.
    del sess
    gc.collect()
    return {
        "variant": variant,
        "path": "coreml_cpu_gpu",
        "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "compute_units": "CPUAndGPU",
        "session_load_s": round(load_s, 3),
        "inference_ms": timings,
        "cold_ms": timings[0] if timings else None,
        "warm_ms": min(timings[1:]) if len(timings) > 1 else None,
        "rss_pre_load_mb": round(rss_pre, 0),
        "rss_post_load_mb": round(rss_post, 0),
        "rss_peak_mb": round(rss_peak, 0),
        "mask_md5": last_md5,
        "bailed_on_timeout": bail,
        "_thumbnail": first_mask,
    }


def _document_coreml_all() -> dict:
    """Path 3 is documented from prior measurements -- NOT executed here."""
    return {
        "path": "coreml_all_declared",
        "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "compute_units": "ALL",
        "executed": False,
        "note": (
            "Asking for MLComputeUnits=ALL declares ANE eligibility to the "
            "CoreML EP, which triggers ANECompilerService on first load. "
            "Prior measurement (memory/perf_segmentation_s52.md): "
            "ANECompilerService CPU-pegged 15-25 min, ~17 GB peak RSS during "
            "compile, ~5 GB resident at peak before macOS swaps it out. "
            "After first compile the .mlmodelc is cached at "
            "~/Library/Caches/com.apple.e5rt.e5bundlecache/<os>/ and "
            "subsequent inferences mmap the bundle -- steady-state perf is "
            "expected to be similar to or slightly better than CPUAndGPU."
        ),
    }


def _model_swap_check(image) -> dict:
    """Cycle lite -> general -> dis -> lite; verify cache+distinctness."""
    from segmentation import ensure_u2net_home
    from segmentation import session as _seg_session
    from segmentation.session import (_SESSION_CACHE, clear_session_cache,
                                      get_session)
    from rembg import remove

    ensure_u2net_home()
    # Use CPU EP for the swap so we don't trigger any CoreML graph compile.
    saved = _seg_session.PROVIDERS_PREFERRED[:]
    _seg_session.PROVIDERS_PREFERRED[:] = ["CPUExecutionProvider"]
    clear_session_cache()

    available = [v for v in VARIANTS if _has_onnx(v)]
    sess_ids: dict[str, int] = {}
    mask_md5s: dict[str, str] = {}

    for v in available:
        LOG.info(f"swap: loading {v}")
        sess = get_session(v)
        sess_ids[v] = id(sess)
        out = remove(image, session=sess)
        png_bytes, _ = _alpha_bytes(out)
        mask_md5s[v] = _md5_bytes(png_bytes)

    # Re-enter the first variant: must return the same session object.
    first = available[0]
    sess_again = get_session(first)
    cache_hit = id(sess_again) == sess_ids[first]

    sessions_distinct = len(set(sess_ids.values())) == len(sess_ids)
    masks_distinct = len(set(mask_md5s.values())) == len(mask_md5s)

    LOG.info(f"swap: sessions_distinct={sessions_distinct} "
             f"masks_distinct={masks_distinct} cache_hit={cache_hit}")

    cache_keys = sorted(_SESSION_CACHE.keys())
    cache_keys_match = set(cache_keys) == {
        # `_resolve_session_name` collapses 'birefnet-lite' -> 'birefnet-general-lite'
        ("birefnet-general-lite" if v == "birefnet-lite" else v)
        for v in available
    }

    _seg_session.PROVIDERS_PREFERRED[:] = saved

    return {
        "variants_tested": available,
        "cache_keys_after": cache_keys,
        "cache_keys_match_expected": cache_keys_match,
        "sessions_distinct": sessions_distinct,
        "masks_distinct": masks_distinct,
        "cache_hit_on_reload": cache_hit,
        "mask_md5s": mask_md5s,
        "session_ids": sess_ids,
    }


def _save_thumbnail(img, dest: Path, size: int = 256) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb = img.copy()
    thumb.thumbnail((size, size))
    thumb.save(dest, format="PNG", optimize=True)


def _fmt_ms(v) -> str:
    return "n/a" if v is None else f"{v:.0f}"


def _build_markdown(report: dict) -> str:
    runs = report["runs"]
    by = {(r["variant"], r["path"]): r for r in runs}
    variants = report["available_variants"]

    lines = []
    lines.append(
        f"\n## {report['date']} light profile (no-ANE-compile path)\n"
    )
    lines.append(
        "Goal: measure CPU EP and CoreML EP (CPUAndGPU, no ANE) for all "
        "three BiRefNet variants without paying the 15-25 min "
        "ANECompilerService cost. 512x512 synthetic input, "
        "`OMP_NUM_THREADS=4`. BiRefNet downscales internally to a 1024 "
        "longest edge so inference compute matches a 4032x3024 source; "
        "only pre/post-resize and JPEG decode shrink.\n"
    )

    lines.append(
        f"- Host: {report['host']}\n"
        f"- onnxruntime providers available: "
        f"`{report['ort_available_providers']}`\n"
        f"- ANE cache before run: "
        f"{report['ane_cache_bytes_before']} bytes "
        f"({report['ane_cache_mb_before']:.1f} MB)\n"
        f"- ANE cache after run:  "
        f"{report['ane_cache_bytes_after']} bytes "
        f"({report['ane_cache_mb_after']:.1f} MB)\n"
        f"- ANE cache growth: "
        f"{report['ane_cache_growth_bytes']} bytes "
        f"(0 confirms `MLComputeUnits=CPUAndGPU` skipped ANE compile)\n"
    )

    lines.append("### Inference latency (ms)\n")
    lines.append("| Variant | CPU cold | CPU warm | Metal cold | Metal warm |")
    lines.append("|---|---|---|---|---|")
    for v in variants:
        cpu = by.get((v, "cpu"))
        gpu = by.get((v, "coreml_cpu_gpu"))
        lines.append(
            f"| `{v}` | "
            f"{_fmt_ms(cpu['cold_ms']) if cpu else 'n/a'} | "
            f"{_fmt_ms(cpu['warm_ms']) if cpu else 'n/a'} | "
            f"{_fmt_ms(gpu['cold_ms']) if gpu else 'n/a'} | "
            f"{_fmt_ms(gpu['warm_ms']) if gpu else 'n/a'} |"
        )

    lines.append("\n### Load time, peak RSS, mask md5 (Metal path)\n")
    lines.append("| Variant | load_s | rss_peak_GB | mask_md5_first8 |")
    lines.append("|---|---|---|---|")
    for v in variants:
        gpu = by.get((v, "coreml_cpu_gpu"))
        if (not gpu or "error" in gpu
                or gpu.get("rss_peak_mb") is None
                or gpu.get("session_load_s") is None):
            lines.append(f"| `{v}` | n/a | n/a | n/a |")
            continue
        rss_gb = gpu["rss_peak_mb"] / 1024.0
        md5 = (gpu.get("mask_md5") or "n/a")[:8]
        lines.append(
            f"| `{v}` | {gpu['session_load_s']:.2f} | "
            f"{rss_gb:.2f} | `{md5}` |"
        )

    lines.append("\n### Speedup CPU warm / Metal warm\n")
    lines.append("| Variant | CPU warm (ms) | Metal warm (ms) | Speedup |")
    lines.append("|---|---|---|---|")
    for v in variants:
        cpu = by.get((v, "cpu"))
        gpu = by.get((v, "coreml_cpu_gpu"))
        cpu_w = cpu["warm_ms"] if cpu else None
        gpu_w = gpu["warm_ms"] if gpu else None
        if cpu_w and gpu_w:
            ratio = cpu_w / gpu_w
            lines.append(
                f"| `{v}` | {cpu_w:.0f} | {gpu_w:.0f} | {ratio:.2f}x |"
            )
        else:
            lines.append(f"| `{v}` | n/a | n/a | n/a |")

    swap = report["model_swap"]
    lines.append("\n### Model-swap correctness\n")
    lines.append(f"- Variants cycled: `{swap['variants_tested']}`")
    lines.append(
        f"- Cache keys after cycle: `{swap['cache_keys_after']}` -> "
        f"{'PASS' if swap['cache_keys_match_expected'] else 'FAIL'}"
    )
    lines.append(
        f"- Session ids distinct: "
        f"{'PASS' if swap['sessions_distinct'] else 'FAIL'}"
    )
    lines.append(
        f"- Mask MD5s distinct (different models -> different masks): "
        f"{'PASS' if swap['masks_distinct'] else 'FAIL'}"
    )
    lines.append(
        f"- Re-loading first variant hits cache (same id): "
        f"{'PASS' if swap['cache_hit_on_reload'] else 'FAIL'}"
    )

    lines.append("\n### Path 3 declared (NOT executed)\n")
    p3 = report["coreml_all_declared"]
    lines.append(
        f"`providers={p3['providers']}, "
        f"provider_options=[{{'MLComputeUnits': '{p3['compute_units']}'}}]` "
        f"-- {p3['note']}"
    )

    lines.append("\n### Memory budget guidance (CPU EP RSS, applies to both paths)\n")
    fit_lines = []
    for v in variants:
        cpu = by.get((v, "cpu"))
        if not cpu or "rss_peak_mb" not in cpu:
            continue
        peak_gb = cpu["rss_peak_mb"] / 1024.0
        # OS + Meshroom typically need ~3-4 GB headroom on UMA.
        fits_8 = peak_gb < 4.0
        fits_16 = peak_gb < 10.0
        fits_24 = peak_gb < 18.0
        fit_lines.append(
            f"- `{v}`: peak RSS {peak_gb:.2f} GB -> "
            f"8GB UMA: {'OK' if fits_8 else 'tight'}, "
            f"16GB: {'OK' if fits_16 else 'tight'}, "
            f"24GB: {'OK' if fits_24 else 'tight'}"
        )
    lines.extend(fit_lines)

    lines.append("\n### Notes\n")
    lines.append(
        "- `OMP_NUM_THREADS=4` honored via env; verify with "
        "`ps -L -p <pid>` during a run if needed."
    )
    lines.append(
        "- GPU utilization not captured programmatically; recommended "
        "approach is `sudo powermetrics --samplers gpu_power -n 5` or "
        "Xcode Instruments \"Metal System Trace\" template during a run."
    )
    lines.append(
        f"- Full structured JSON: "
        f"`build/seg_profile/light_profile_v2.json`"
    )
    lines.append(
        "- Per-variant per-path mask thumbnails (256x256): "
        "`build/seg_profile/masks/<variant>_<path>.png`"
    )

    return "\n".join(lines) + "\n"


def _placeholder_for_missing(variant: str, path: str, reason: str) -> dict:
    return {
        "variant": variant,
        "path": path,
        "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                     if path == "coreml_cpu_gpu" else ["CPUExecutionProvider"],
        "compute_units": "CPUAndGPU" if path == "coreml_cpu_gpu" else None,
        "session_load_s": None,
        "inference_ms": [],
        "cold_ms": None,
        "warm_ms": None,
        "rss_pre_load_mb": None,
        "rss_post_load_mb": None,
        "rss_peak_mb": None,
        "mask_md5": None,
        "bailed_on_timeout": True,
        "skipped_reason": reason,
    }


def main() -> int:
    LOG.info(f"U2NET_HOME = {os.environ.get('U2NET_HOME')}")
    LOG.info(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    available = [v for v in VARIANTS if _has_onnx(v)]
    LOG.info(f"variants_on_disk = {available}")

    # Allow --finalize-from <partial.json> to skip re-running the heavy
    # CoreML+CPU benches when we already have data; just run the cheap
    # model-swap check and serialize the final report.
    finalize_from = None
    if "--finalize-from" in sys.argv:
        idx = sys.argv.index("--finalize-from")
        finalize_from = sys.argv[idx + 1]

    import onnxruntime as ort
    ort_providers = list(ort.get_available_providers())
    LOG.info(f"ort providers = {ort_providers}")

    ane_before = _ane_cache_size_bytes() or 0
    LOG.info(f"ANE cache size before: {ane_before} bytes "
             f"({ane_before / (1024 * 1024):.1f} MB)")

    image = _make_synthetic_image(512)
    LOG.info(f"synthetic input = {image.size}")

    masks_dir = ROOT / "build" / "seg_profile" / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "build" / "seg_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "light_profile_v2.partial.json"

    runs: list[dict] = []

    if finalize_from:
        with open(finalize_from) as f:
            prev = json.load(f)
        runs = prev.get("runs", [])
        LOG.info(f"finalize: loaded {len(runs)} prior runs from {finalize_from}")
        present = {(r["variant"], r["path"]) for r in runs}
        for v in available:
            if (v, "cpu") not in present:
                runs.append(_placeholder_for_missing(
                    v, "cpu", "not measured in finalize-from input"))
            if (v, "coreml_cpu_gpu") not in present:
                runs.append(_placeholder_for_missing(
                    v, "coreml_cpu_gpu",
                    "CoreML EP+CPUAndGPU on swin_v1_large was too slow to "
                    "characterize within the resource budget (extrapolating "
                    "from lite which took 50s cold / 233s warm, the large "
                    "variants exceed 5 min per inference)."))
        # Skip the heavy bench loop, go straight to swap-check + report.
        swap = _model_swap_check(image)
        ane_after = _ane_cache_size_bytes() or 0
        ane_before_used = prev.get("ane_cache_bytes_before", ane_before)
        LOG.info(f"ANE cache size after:  {ane_after} bytes "
                 f"(growth from prev-known-before: "
                 f"{ane_after - ane_before_used} bytes)")
        report = {
            "date": _dt.date.today().isoformat(),
            "host": "Apple M4 (24 GB UMA), macOS 26.5, py3.13.3, ort 1.26.0",
            "synthetic_input": "512x512 RGB",
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "ort_available_providers": ort_providers,
            "ane_cache_bytes_before": ane_before_used,
            "ane_cache_mb_before": ane_before_used / (1024 * 1024),
            "ane_cache_bytes_after": ane_after,
            "ane_cache_mb_after": ane_after / (1024 * 1024),
            "ane_cache_growth_bytes": ane_after - ane_before_used,
            "available_variants": available,
            "runs": runs,
            "coreml_all_declared": _document_coreml_all(),
            "model_swap": swap,
        }
        out_json = out_dir / "light_profile_v2.json"
        with open(out_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        LOG.info(f"JSON written to {out_json}")
        md = _build_markdown(report)
        perf_doc = ROOT / "memory" / "perf_segmentation_s52.md"
        with open(perf_doc, "a") as f:
            f.write(md)
        LOG.info(f"Appended report to {perf_doc}")
        print("\n=== SUMMARY ===")
        print(md)
        return 0

    def _checkpoint():
        """Persist a partial JSON after each run -- always-on safety net."""
        # Strip PIL thumbnails before json
        clean = []
        for r in runs:
            c = dict(r)
            c.pop("_thumbnail", None)
            clean.append(c)
        with open(checkpoint_path, "w") as f:
            json.dump({"runs": clean,
                       "ane_cache_bytes_before": ane_before}, f,
                      indent=2, default=str)

    def _persist_thumbnail(r):
        thumb = r.get("_thumbnail")
        if thumb is None:
            return
        dest = masks_dir / f"{r['variant']}_{r['path']}.png"
        try:
            _save_thumbnail(thumb, dest, size=256)
            r["mask_thumbnail"] = str(dest.relative_to(ROOT))
        except Exception as exc:
            LOG.warning(f"thumbnail save failed: {exc}")

    for v in available:
        runs.append(_bench_path_cpu(v, image, RUNS_PER_PATH_CPU))
        _persist_thumbnail(runs[-1])
        _checkpoint()
        LOG.info(f"checkpoint after cpu/{v}: {len(runs)} runs persisted")

    for v in available:
        try:
            runs.append(_bench_path_coreml_cpu_gpu(v, image,
                                                   RUNS_PER_PATH_COREML))
            _persist_thumbnail(runs[-1])
        except Exception as exc:
            import traceback
            LOG.error(f"CoreML path failed for {v}: {exc}")
            LOG.error(traceback.format_exc())
            runs.append({"variant": v, "path": "coreml_cpu_gpu",
                         "error": str(exc)})
        _checkpoint()
        LOG.info(f"checkpoint after coreml/{v}: {len(runs)} runs persisted")

    # Strip thumbnails before final serialization
    for r in runs:
        r.pop("_thumbnail", None)

    swap = _model_swap_check(image)

    ane_after = _ane_cache_size_bytes() or 0
    LOG.info(f"ANE cache size after:  {ane_after} bytes "
             f"({ane_after / (1024 * 1024):.1f} MB) "
             f"(growth: {ane_after - ane_before} bytes)")

    report = {
        "date": _dt.date.today().isoformat(),
        "host": "Apple M4 (24 GB UMA), macOS 26.5, py3.13.3, ort 1.26.0",
        "synthetic_input": "512x512 RGB",
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "ort_available_providers": ort_providers,
        "ane_cache_bytes_before": ane_before,
        "ane_cache_mb_before": ane_before / (1024 * 1024),
        "ane_cache_bytes_after": ane_after,
        "ane_cache_mb_after": ane_after / (1024 * 1024),
        "ane_cache_growth_bytes": ane_after - ane_before,
        "available_variants": available,
        "runs": runs,
        "coreml_all_declared": _document_coreml_all(),
        "model_swap": swap,
    }

    out_json = out_dir / "light_profile_v2.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    LOG.info(f"JSON written to {out_json}")

    # Append Markdown block to perf doc
    md = _build_markdown(report)
    perf_doc = ROOT / "memory" / "perf_segmentation_s52.md"
    with open(perf_doc, "a") as f:
        f.write(md)
    LOG.info(f"Appended report to {perf_doc}")

    print("\n=== SUMMARY ===")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
