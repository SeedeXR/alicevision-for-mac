#!/usr/bin/env python3
"""
Download BiRefNet ONNX weights into the project's ai-models/ folder.

This script is the single-source-of-truth for staging the segmentation
weights used by `meshroom-mac/nodes/aliceVision/SegmentationBiRefNet.py`.

CRITICAL DEVIATION from upstream rembg conventions:
  rembg's default cache is ~/.u2net/. This script instead stages weights
  into <project_root>/ai-models/ so the user can keep model artifacts
  tracked alongside the project. The session loader honours the
  `U2NET_HOME` environment variable, which is what rembg internally
  consults before falling back to ~/.u2net/. We export U2NET_HOME here
  for users who source this script's helper.

Usage:
    python scripts/download_models.py                   # default variant
    python scripts/download_models.py birefnet-general  # explicit variant
    python scripts/download_models.py --variant birefnet-lite
    python scripts/download_models.py --all             # grab everything

Variants:
    birefnet-general   (default; general photogrammetry, ~973 MB)
    birefnet-dis       (high-detail / complex edges)
    birefnet-lite      (faster, lower RAM; M1/M2 friendly)
"""
from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

# Python.org's macOS installer ships with no system trust store wired in,
# so stock urllib SSL verification fails out of the box. Prefer certifi
# if installed (it's already a transitive dep via requests/huggingface_hub),
# otherwise fall back to the OpenSSL default paths.
try:
    import certifi  # type: ignore
    _SSL_CONTEXT: ssl.SSLContext | None = ssl.create_default_context(
        cafile=certifi.where()
    )
except Exception:
    _SSL_CONTEXT = None


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# After the S53 plugin refactor this file lives at
# `<repo>/plugins/ai-segmentation/scripts/download_models.py` — walk up
# THREE levels (scripts -> ai-segmentation -> plugins -> <repo>) to reach
# the repo root. A back-compat symlink at `<repo>/scripts/download_models.py`
# resolves through `Path(__file__).resolve()` to this canonical location.
REPO_ROOT = Path(__file__).resolve().parents[3]
AI_MODELS_DIR = REPO_ROOT / "ai-models"

# Export U2NET_HOME so rembg / downstream tools pick up our local cache
os.environ.setdefault("U2NET_HOME", str(AI_MODELS_DIR))


# --------------------------------------------------------------------------- #
# Model catalogue
# --------------------------------------------------------------------------- #

# Map of rembg-compatible model name -> (primary URL, fallback URL, basename).
#
# Filenames MUST match rembg's expected `<sessionName>.onnx` convention so
# `rembg.new_session(name)` finds the pre-staged weights (rembg looks up
# files via `U2NET_HOME / <sessionName>.onnx` before downloading).
#
# Primary URLs point at rembg's own GitHub releases — these match the MD5
# checksums hard-coded into rembg's session classes. Fallback URLs point at
# the upstream ZhengPeng7/BiRefNet HuggingFace repo (slightly different
# builds; rembg will refuse to load them unless `MODEL_CHECKSUM_DISABLED=1`).
#
# The "birefnet-lite" CLI alias corresponds to rembg's internal name
# "birefnet-general-lite" — we map both for user friendliness.
MODELS: dict[str, tuple[str, str, str]] = {
    "birefnet-general": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "BiRefNet-general-epoch_244.onnx",
        "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/onnx/"
        "birefnet-general.onnx",
        "birefnet-general.onnx",
    ),
    "birefnet-dis": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "BiRefNet-DIS-epoch_590.onnx",
        "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/onnx/"
        "birefnet-dis.onnx",
        "birefnet-dis.onnx",
    ),
    "birefnet-lite": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        "https://huggingface.co/ZhengPeng7/BiRefNet-lite/resolve/main/onnx/"
        "birefnet-general-lite.onnx",
        # rembg internal name is "birefnet-general-lite" — we stage under
        # that filename so new_session("birefnet-general-lite") works.
        "birefnet-general-lite.onnx",
    ),
}

DEFAULT_VARIANT = "birefnet-general"

# Map from CLI alias -> rembg session name (the string passed to
# rembg.new_session()). For most variants the names are identical; the
# "lite" alias is the one exception.
SESSION_NAMES: dict[str, str] = {
    "birefnet-general": "birefnet-general",
    "birefnet-dis": "birefnet-dis",
    "birefnet-lite": "birefnet-general-lite",
}


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #

def _format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:6.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:6.1f} TB"


def _progress_factory():
    start = time.time()

    def _progress(count: int, block_size: int, total_size: int) -> None:
        downloaded = count * block_size
        if total_size <= 0:
            print(f"\r  downloaded {_format_bytes(downloaded)}", end="", flush=True)
            return
        pct = min(downloaded * 100 // total_size, 100)
        elapsed = max(time.time() - start, 1e-6)
        speed = downloaded / elapsed
        bar_len = 30
        filled = bar_len * pct // 100
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"\r  [{bar}] {pct:3d}%  "
            f"{_format_bytes(downloaded)} / {_format_bytes(total_size)}  "
            f"({_format_bytes(speed)}/s)",
            end="",
            flush=True,
        )

    return _progress


def _try_download(url: str, dest: Path) -> None:
    """Stream `url` into `dest` with a progress bar.

    Uses an explicit SSL context backed by certifi when available — works
    around the Python.org macOS installer's missing system trust store.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "alicevision-for-mac/segmentation"}
        )
        opener_kwargs = {}
        if _SSL_CONTEXT is not None:
            opener_kwargs["context"] = _SSL_CONTEXT
        with urllib.request.urlopen(req, **opener_kwargs) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            block_size = 1 << 16  # 64 KiB
            progress = _progress_factory()
            downloaded = 0
            with open(tmp, "wb") as out:
                count = 0
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    count += 1
                    downloaded += len(chunk)
                    # urlretrieve compatibility shape: (count, block, total)
                    progress(count, block_size, total)
        print()  # newline after progress bar
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def download(name: str, primary_url: str, fallback_url: str,
             dest_basename: str) -> Path:
    """Idempotently download `name` into AI_MODELS_DIR / dest_basename.

    Tries `primary_url` first (rembg-checksummed releases). Falls back to
    `fallback_url` (HuggingFace) if the primary fails. Note that fallback
    weights will only load if `MODEL_CHECKSUM_DISABLED=1` is set.
    """
    AI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = AI_MODELS_DIR / dest_basename

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {name}: already present at {dest} "
              f"({_format_bytes(dest.stat().st_size)})")
        return dest

    print(f"[download] {name}")
    print(f"  dst: {dest}")

    last_exc: Exception | None = None
    for label, url in (("primary", primary_url), ("fallback", fallback_url)):
        print(f"  trying {label}: {url}")
        try:
            _try_download(url, dest)
            print(f"[done]  {name} -> {dest} "
                  f"({_format_bytes(dest.stat().st_size)})")
            if label == "fallback":
                print("[warn]  used HuggingFace fallback — checksums may "
                      "differ from rembg's expected MD5. Export "
                      "MODEL_CHECKSUM_DISABLED=1 before running the node.",
                      file=sys.stderr)
            return dest
        except Exception as exc:
            last_exc = exc
            print(f"\n  [{label} failed] {exc}", file=sys.stderr)

    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download BiRefNet ONNX weights into ai-models/."
    )
    parser.add_argument(
        "variant",
        nargs="?",
        default=None,
        help=f"Model variant to download (default: {DEFAULT_VARIANT}).",
    )
    parser.add_argument(
        "--variant",
        dest="variant_opt",
        default=None,
        help="Same as the positional argument; explicit form.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every catalogued variant.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available variants and exit.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="After downloading, pre-compile the ANE bundle by loading "
             "the model through onnxruntime's CoreML EP once. First-run "
             "ANECompilerService takes 15-25 min on M-series; running it "
             "now means production pipelines aren't blocked on it.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available variants:")
        for name, (primary, fallback, basename) in MODELS.items():
            session = SESSION_NAMES[name]
            print(f"  {name:20s} (session={session}) -> {basename}")
            print(f"  {'':20s}    primary:  {primary}")
            print(f"  {'':20s}    fallback: {fallback}")
        return 0

    print(f"AI_MODELS_DIR = {AI_MODELS_DIR}")
    print(f"U2NET_HOME    = {os.environ.get('U2NET_HOME')}")
    print()

    if args.all:
        targets = list(MODELS.keys())
    else:
        variant = args.variant or args.variant_opt or DEFAULT_VARIANT
        if variant not in MODELS:
            print(f"[error] unknown variant: {variant}", file=sys.stderr)
            print(f"        known: {', '.join(MODELS.keys())}", file=sys.stderr)
            return 2
        targets = [variant]

    failures = []
    for name in targets:
        primary, fallback, basename = MODELS[name]
        try:
            download(name, primary, fallback, basename)
        except Exception:
            failures.append(name)

    print()
    if failures:
        print(f"[summary] {len(failures)} failure(s): {', '.join(failures)}",
              file=sys.stderr)
        return 1
    print("[summary] all requested variants present in ai-models/.")
    print("CoreML conversion runs lazily on first node execution.")

    if args.warmup:
        print()
        print("[warmup] pre-compiling ANE bundle(s) via onnxruntime CoreML EP.")
        print("[warmup] This is a one-time cost per (machine, ONNX file).")
        print("[warmup] Expect 15-25 min per variant on M-series.")
        try:
            # The segmentation package lives in this plugin's `python/` dir.
            # Adding that on sys.path lets `import segmentation` resolve.
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
            from segmentation.session import get_session, clear_session_cache
            from rembg import remove
            from PIL import Image
            warmup_img = Image.new("RGB", (256, 256), (128, 128, 128))
            for name in targets:
                t0 = time.time()
                print(f"[warmup] loading session for '{name}' ...")
                sess = get_session(name)
                print(f"[warmup] '{name}' session ready in {time.time()-t0:.1f}s")
                t1 = time.time()
                _ = remove(warmup_img, session=sess)
                print(f"[warmup] '{name}' first inference {time.time()-t1:.1f}s "
                      f"-> ANE bundle now cached.")
                clear_session_cache()  # free RAM between variants
        except Exception as exc:
            print(f"[warmup] failed: {exc}", file=sys.stderr)
            return 3
    return 0


def _repo_root():
    """Locate the repo root from this script's path.

    After the S53 plugin refactor, this file is at
    `<repo>/plugins/ai-segmentation/scripts/download_models.py`, so the
    repo root is `parents[3]`.
    """
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    raise SystemExit(main())
