#!/usr/bin/env python3
"""
profile_segmentation.py — end-to-end benchmark + model-swap verifier
for SegmentationBiRefNet.

Exercises:

  - Session-load timing per model variant
  - Inference timing per image, cold (first call) vs warm (subsequent)
  - Per-provider comparison: CoreML+ANE vs CPU-only
  - Model-swap correctness: load all 3 variants, ensure each has its
    own ONNX session in `_SESSION_CACHE`, ensure mask outputs differ
    bit-for-bit when expected
  - Output mask integrity (file size, MD5)

Run:
    source meshroom-venv/bin/activate
    PYTHONPATH=meshroom-mac:src/python_shim \\
        python scripts/profile_segmentation.py [--variants ...] [--images N]

Outputs a JSON report to stdout and writes a human-readable Markdown
summary to memory/perf_segmentation_s52.md (replacing the existing
"Files produced" + "How the node satisfies the contract" tail; the rest
is preserved).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path


def _bootstrap_path() -> Path:
    here = Path(__file__).resolve()
    root = here.parent.parent
    for entry in (root / "meshroom-mac", root / "src" / "python_shim"):
        s = str(entry)
        if s not in sys.path:
            sys.path.insert(0, s)
    os.environ.setdefault("U2NET_HOME", str(root / "ai-models"))
    return root


ROOT = _bootstrap_path()
log = logging.getLogger("profile")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_onnx(variant: str) -> Path | None:
    """Map a user-facing variant name to an on-disk ONNX path."""
    candidates = {
        "birefnet-general": ["birefnet-general.onnx"],
        "birefnet-dis": ["birefnet-dis.onnx"],
        "birefnet-lite": [
            "birefnet-general-lite.onnx",
            "birefnet-lite.onnx",
        ],
    }
    for name in candidates.get(variant, [f"{variant}.onnx"]):
        p = ROOT / "ai-models" / name
        if p.is_file():
            return p
    return None


def _gather_images(images_arg: int) -> list[Path]:
    src = ROOT / "dataset_monstree" / "mini3"
    files = sorted(src.glob("*.JPG"))[: max(1, int(images_arg))]
    if not files:
        raise SystemExit(f"No images found in {src}")
    return files


def _benchmark_variant(
    variant: str,
    images: list[Path],
    out_dir: Path,
    providers_pref: list[str],
    *,
    tag: str,
) -> dict:
    """Run inference on the given images and capture timings."""
    from segmentation import ensure_u2net_home
    from segmentation.session import clear_session_cache
    from segmentation.session import _filter_providers, get_session
    from PIL import Image
    from rembg import remove

    ensure_u2net_home()

    # Inject the requested provider list by temporarily swapping the
    # session.PROVIDERS_PREFERRED list.
    from segmentation import session as _seg_session

    saved = _seg_session.PROVIDERS_PREFERRED[:]
    _seg_session.PROVIDERS_PREFERRED[:] = providers_pref
    clear_session_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"--- variant={variant} tag={tag} providers={providers_pref} ---")

    t0 = time.time()
    sess = get_session(variant)
    load_s = time.time() - t0
    active = _filter_providers(providers_pref)
    log.info(f"  session loaded in {load_s:.2f}s (active providers={active})")

    per_image: list[dict] = []
    for idx, img_path in enumerate(images):
        with Image.open(img_path) as src:
            src.load()
        dest = out_dir / f"{img_path.stem}_{variant}_{tag}.png"
        t1 = time.time()
        result = remove(src, session=sess)
        inf_s = time.time() - t1

        t_save = time.time()
        alpha = result.convert("RGBA").split()[-1]
        alpha.save(dest, format="PNG", optimize=False)
        save_s = time.time() - t_save

        per_image.append({
            "image": img_path.name,
            "inference_s": round(inf_s, 3),
            "save_s": round(save_s, 3),
            "mask_md5": _md5(dest),
            "mask_size_bytes": dest.stat().st_size,
            "width": src.size[0],
            "height": src.size[1],
        })
        log.info(
            f"  [{idx+1}/{len(images)}] {img_path.name} "
            f"inf={inf_s:.2f}s save={save_s:.2f}s -> {dest.name}"
        )

    # Restore preferred providers
    _seg_session.PROVIDERS_PREFERRED[:] = saved
    clear_session_cache()

    return {
        "variant": variant,
        "tag": tag,
        "providers_pref": providers_pref,
        "providers_active": active,
        "session_load_s": round(load_s, 3),
        "images": per_image,
    }


def _verify_model_swap(images: list[Path], out_dir: Path) -> dict:
    """Load all 3 variants in sequence and assert session cache + outputs
    are correctly distinct."""
    from segmentation import ensure_u2net_home
    from segmentation.session import clear_session_cache
    from segmentation.session import _SESSION_CACHE, _resolve_session_name, get_session
    from PIL import Image
    from rembg import remove

    ensure_u2net_home()
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_session_cache()

    img = images[0]
    masks: dict[str, str] = {}
    session_ids: dict[str, int] = {}

    variants_present = [v for v in ("birefnet-lite", "birefnet-general", "birefnet-dis") if _has_onnx(v)]

    for variant in variants_present:
        sess = get_session(variant)
        with Image.open(img) as src:
            src.load()
            result = remove(src, session=sess)
            alpha = result.convert("RGBA").split()[-1]
            dest = out_dir / f"{img.stem}_swap_{variant}.png"
            alpha.save(dest, format="PNG", optimize=False)
            masks[variant] = _md5(dest)
            session_ids[variant] = id(sess)

    # Assertions
    distinct_sessions = len(set(session_ids.values())) == len(session_ids)
    distinct_masks = len(set(masks.values())) == len(masks)

    # Reload one and confirm session cache hit
    if variants_present:
        first = variants_present[0]
        sess_again = get_session(first)
        cache_hit = id(sess_again) == session_ids[first]
    else:
        cache_hit = False

    return {
        "variants_tested": variants_present,
        "session_cache_keys": sorted(_SESSION_CACHE.keys()),
        "session_ids_distinct": distinct_sessions,
        "mask_md5s_distinct": distinct_masks,
        "session_cache_hit_on_reload": cache_hit,
        "mask_md5s": masks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="*",
                    default=["birefnet-lite", "birefnet-general", "birefnet-dis"],
                    help="Variants to benchmark.")
    ap.add_argument("--images", type=int, default=3,
                    help="Monstree mini3 images to use (max 3).")
    ap.add_argument("--cpu-only", action="store_true",
                    help="Force CPU EP only (skip CoreML).")
    ap.add_argument("--coreml-only", action="store_true",
                    help="Skip CPU baseline, only run CoreML.")
    ap.add_argument("--report-json", type=str,
                    default=str(ROOT / "memory" / "perf_segmentation_s52_results.json"))
    args = ap.parse_args()

    out_dir = ROOT / "build" / "seg_profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _gather_images(args.images)
    log.info(f"Profiling images: {[p.name for p in images]}")

    available = []
    for v in args.variants:
        p = _has_onnx(v)
        if p is None:
            log.warning(f"Variant {v}: ONNX file not on disk — skipping.")
            continue
        available.append(v)
        log.info(f"Variant {v}: {p.name} ({p.stat().st_size // (1024*1024)} MB)")

    runs: list[dict] = []

    # CPU baseline
    if not args.coreml_only:
        for v in available:
            runs.append(_benchmark_variant(
                v, images, out_dir,
                providers_pref=["CPUExecutionProvider"],
                tag="cpu",
            ))

    # CoreML run
    if not args.cpu_only:
        for v in available:
            runs.append(_benchmark_variant(
                v, images, out_dir,
                providers_pref=["CoreMLExecutionProvider", "CPUExecutionProvider"],
                tag="coreml",
            ))

    # Model swap
    swap = _verify_model_swap(images, out_dir)

    report = {
        "host": {
            "platform": sys.platform,
            "u2net_home": os.environ.get("U2NET_HOME", ""),
            "available_variants": available,
            "image_count": len(images),
            "image_dims": f"{images[0].stat().st_size / 1024:.0f} KB JPEG",
        },
        "runs": runs,
        "model_swap": swap,
    }

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"JSON report -> {args.report_json}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
