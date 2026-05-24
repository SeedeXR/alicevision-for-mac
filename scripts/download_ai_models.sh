#!/usr/bin/env bash
#
# scripts/download_ai_models.sh — fetch the 4 pre-converted CoreML
# .mlpackages used by alicevision-for-mac, from the project's Google
# Drive archive, into ./ai-models/.
#
# Two paths to get the models, depending on how much you want to do
# yourself:
#
#   1. PRE-CONVERTED, just-want-to-run  ← this script
#      Downloads one Google Drive archive that bundles all four
#      .mlpackages, extracts it into ai-models/. Recommended path for
#      everyone except contributors who are tuning model conversion.
#
#   2. CONVERT FROM SCRATCH (companion repo)
#      https://github.com/SeedeXR/alicevision-for-mac-models
#      That repo contains the conversion scripts + recipe for each
#      model (BiRefNet, YOLOv8n, MoGe-2, TinyRoMa). Use it when you
#      want to re-export with different input shape, FP32 precision,
#      etc. Not required for normal usage.
#
# Models fetched (~750 MB total, ~3-5 min on a fast connection):
#   BiRefNet_lite.mlpackage          (~90 MB)   foreground segmentation (default)
#   BiRefNet.mlpackage               (~447 MB)  foreground segmentation (high accuracy)
#   yolov8n.mlpackage                (~13 MB)   sphere detection (aliceVision_sphereDetection)
#   moge2_504x672_t1728.mlpackage    (~187 MB)  mono-depth (aliceVision_moGe)
#   tiny_roma_v1_480x640.mlpackage   (~5.5 MB)  dense matcher (aliceVision_matchMasking)
#
# Usage:
#   scripts/download_ai_models.sh                 # download + extract into ./ai-models/
#   scripts/download_ai_models.sh --dest /path    # write to a different directory
#   scripts/download_ai_models.sh --force         # re-download even if all 5 are present
#   scripts/download_ai_models.sh --keep-archive  # don't delete the .tar.gz after extract
#   scripts/download_ai_models.sh --archive /local/path.tar.gz   # skip download, just extract
#   scripts/download_ai_models.sh --url URL       # override the Google Drive URL
#
# Requires: python3 + `gdown` (auto-installs into ~/.local if missing).
# Google Drive's large-file flow uses a confirm-token dance that
# changes periodically; `gdown` is the maintained Python tool that
# tracks Google's API and is the only reliable way to script this.
#
set -euo pipefail

# ----------------------------------------------------------------------------
# Defaults — Google Drive archive containing all four .mlpackages
# ----------------------------------------------------------------------------
ARCHIVE_URL_DEFAULT="https://drive.google.com/file/d/12jt788_0Wab_nahVa7zP2lHfDSAYky5z/view?usp=sharing"

# The four .mlpackage directories the archive must produce in $DEST.
# Used for the "skip if already present" check and the post-extract
# verification.
EXPECTED_MLPACKAGES=(
    "BiRefNet_lite.mlpackage"
    "BiRefNet.mlpackage"
    "yolov8n.mlpackage"
    "moge2_504x672_t1728.mlpackage"
    "tiny_roma_v1_480x640.mlpackage"
)

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/ai-models"
ARCHIVE_URL="$ARCHIVE_URL_DEFAULT"
LOCAL_ARCHIVE=""
FORCE=0
KEEP_ARCHIVE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dest)          DEST="$2"; shift 2 ;;
        --url)           ARCHIVE_URL="$2"; shift 2 ;;
        --archive)       LOCAL_ARCHIVE="$2"; shift 2 ;;
        --force)         FORCE=1; shift ;;
        --keep-archive)  KEEP_ARCHIVE=1; shift ;;
        -h|--help)       sed -n '2,/^set/p' "$0" | sed -n '/^# /p' | sed 's/^# //'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
if [ -t 1 ]; then
    BOLD="$(tput bold 2>/dev/null || true)"
    RED="$(tput setaf 1 2>/dev/null || true)"
    GREEN="$(tput setaf 2 2>/dev/null || true)"
    YELLOW="$(tput setaf 3 2>/dev/null || true)"
    BLUE="$(tput setaf 4 2>/dev/null || true)"
    RESET="$(tput sgr0 2>/dev/null || true)"
else
    BOLD=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; RESET=""
fi
log()      { printf "%s[%s]%s %s\n" "$BLUE"   "$(date +%H:%M:%S)" "$RESET" "$*"; }
log_ok()   { printf "%s[%s] ✓ %s%s\n" "$GREEN" "$(date +%H:%M:%S)" "$*" "$RESET"; }
log_warn() { printf "%s[%s] ⚠ %s%s\n" "$YELLOW" "$(date +%H:%M:%S)" "$*" "$RESET" >&2; }
log_err()  { printf "%s[%s] ✗ %s%s\n" "$RED"   "$(date +%H:%M:%S)" "$*" "$RESET" >&2; }

# Try to find an existing gdown installation. Set GDOWN to the cmd to run.
detect_gdown() {
    # Prefer `python3 -m gdown` so we don't depend on PATH/shim weirdness.
    if python3 -c "import gdown" >/dev/null 2>&1; then
        GDOWN="python3 -m gdown"
        return 0
    fi
    # Fall back to the binary if pip installed it on PATH.
    if command -v gdown >/dev/null 2>&1; then
        GDOWN="gdown"
        return 0
    fi
    return 1
}

# Install gdown into the user's site-packages (Apple Silicon-friendly,
# no sudo, no homebrew Python lock-out).
install_gdown() {
    log "Installing gdown into the user site-packages (one-time)…"
    if python3 -m pip install --user gdown >/dev/null 2>&1; then
        return 0
    fi
    # Apple's system Python sometimes rejects --user without
    # --break-system-packages. Retry with that.
    if python3 -m pip install --user --break-system-packages gdown >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------
command -v python3 >/dev/null || { log_err "python3 is required (install via brew or Apple)."; exit 2; }
command -v tar >/dev/null     || { log_err "tar is required."; exit 2; }
mkdir -p "$DEST"

log "Dest:    $DEST"
log "Source:  $([ -n "$LOCAL_ARCHIVE" ] && echo "local: $LOCAL_ARCHIVE" || echo "$ARCHIVE_URL")"

# Skip-if-present: all 5 .mlpackages already on disk and --force not set.
already_complete=1
for pkg in "${EXPECTED_MLPACKAGES[@]}"; do
    if [ ! -d "$DEST/$pkg" ]; then
        already_complete=0
        break
    fi
done
if [ "$already_complete" -eq 1 ] && [ "$FORCE" -eq 0 ]; then
    log_ok "All 5 .mlpackages already present in $DEST."
    log "Use --force to re-download anyway."
    exit 0
fi

# ----------------------------------------------------------------------------
# Acquire the archive (download from Drive, or accept a local path)
# ----------------------------------------------------------------------------
TMPDIR="$(mktemp -d -t avmodels)"
trap '[ "$KEEP_ARCHIVE" -eq 0 ] && rm -rf "$TMPDIR" || log "Archive kept at $TMPDIR"' EXIT

if [ -n "$LOCAL_ARCHIVE" ]; then
    [ -f "$LOCAL_ARCHIVE" ] || { log_err "--archive path doesn't exist: $LOCAL_ARCHIVE"; exit 2; }
    archive_path="$LOCAL_ARCHIVE"
    log_ok "Using local archive (skipping download)."
else
    if ! detect_gdown; then
        log_warn "gdown not found — attempting auto-install."
        if ! install_gdown; then
            log_err "Auto-install of gdown failed. Install it manually:"
            log_err "    pip3 install --user gdown"
            log_err "  or:"
            log_err "    pip3 install --user --break-system-packages gdown"
            log_err "  then re-run this script."
            log_err ""
            log_err "Alternative: download the archive manually from"
            log_err "    $ARCHIVE_URL"
            log_err "  and re-run with: scripts/download_ai_models.sh --archive /path/to/file.tar.gz"
            exit 1
        fi
        detect_gdown || { log_err "gdown installed but not detectable — check PYTHONPATH."; exit 1; }
    fi
    log "Using gdown via: $GDOWN"

    archive_path="$TMPDIR/ai-models.tar.gz"
    log "Downloading from Google Drive (~750 MB) — this can take 3-5 min on a fast connection."
    # `gdown --fuzzy` accepts the file/d/<ID>/view URL form directly,
    # so the user-facing $ARCHIVE_URL can be the Drive "share" link as
    # copy-pasted from the browser (no need to extract the file ID).
    if ! $GDOWN --fuzzy "$ARCHIVE_URL" -O "$archive_path"; then
        log_err "gdown failed. The most common cause is that the share permission"
        log_err "on the Drive file is not 'Anyone with the link can view'."
        log_err "Manual fallback: download from $ARCHIVE_URL and re-run with --archive."
        exit 1
    fi
    log_ok "Download complete: $archive_path"
fi

# ----------------------------------------------------------------------------
# Extract
# ----------------------------------------------------------------------------
log "Extracting into $DEST"
# The archive can be a .tar.gz OR a .zip; detect by magic bytes.
magic="$(head -c 4 "$archive_path" | xxd -p | head -c 8)"
case "$magic" in
    1f8b*)        # gzip
        tar -xzf "$archive_path" -C "$DEST" ;;
    504b0304|504b0506|504b0708)  # zip
        if command -v unzip >/dev/null; then
            unzip -oq "$archive_path" -d "$DEST"
        else
            log_err "archive is a .zip but unzip is not on PATH"
            exit 1
        fi ;;
    *)
        # Try tar autodetect (could be plain tar, .tar.bz2, etc.)
        tar -xf "$archive_path" -C "$DEST" ;;
esac

# ----------------------------------------------------------------------------
# Verify
# ----------------------------------------------------------------------------
log "Verifying installation"
missing=0
for pkg in "${EXPECTED_MLPACKAGES[@]}"; do
    if [ -d "$DEST/$pkg" ] && [ -f "$DEST/$pkg/Manifest.json" ]; then
        sz="$(du -sh "$DEST/$pkg" 2>/dev/null | awk '{print $1}')"
        log_ok "  $pkg  ($sz)"
    else
        log_err "  $pkg  MISSING"
        missing=$((missing + 1))
    fi
done

if [ "$missing" -gt 0 ]; then
    log_err "$missing of ${#EXPECTED_MLPACKAGES[@]} .mlpackages missing after extract."
    log_err "The archive layout may have changed — inspect $archive_path or $DEST."
    exit 1
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
total_size="$(du -sh "$DEST" 2>/dev/null | awk '{print $1}')"
log_ok "All ${#EXPECTED_MLPACKAGES[@]} .mlpackages installed in $DEST ($total_size total)."
log ""
log "Next steps:"
log "  • Native binaries auto-discover models in ai-models/ (or via"
log "    ALICEVISION_MOGE_MLPACKAGE / ALICEVISION_ROMA_MLPACKAGE env vars)."
log "  • Conversion / training scripts: https://github.com/SeedeXR/alicevision-for-mac-models"
log "  • Per-model details: ai-models/README.md"
