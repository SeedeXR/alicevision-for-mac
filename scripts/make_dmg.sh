#!/usr/bin/env bash
#
# scripts/make_dmg.sh — package Meshroom.app into a distributable DMG.
#
# Produces `build/release/Meshroom-<version>-arm64.dmg` ready to upload
# to a release page. The DMG contains:
#   - Meshroom.app (signed by scripts/codesign_macos_app.sh)
#   - A symlink to /Applications (so the user drags + drops to install)
#   - README.txt with launch instructions
#
# Compression
# -----------
# Defaults to ULMO (UDIF LZMA, macOS 10.15+). On our benchmark fixture
# (186 MB of Mach-Os + Homebrew dylibs) ULMO produced a DMG 34 % smaller
# than the legacy UDZO baseline at 4× the wall-time. The Mac port targets
# macOS 14+ so the 10.15+ requirement is irrelevant. Other --compression
# values you may want:
#
#   udzo      — UDIF zlib level 1 (legacy default, fast, broad compat,
#               works back to macOS 10.0). Baseline; +0%, ~baseline time.
#   udzo-max  — UDIF zlib level 9 (~10 % smaller than udzo, ~2× slower).
#   ulfo      — UDIF lzfse, macOS 10.11+ (~11 % smaller, same speed as udzo).
#   ulmo      — UDIF lzma,  macOS 10.15+ (~34 % smaller, ~4× slower).  [DEFAULT]
#
# Safety
# ------
# After building, the script always runs `hdiutil verify <dmg>` to
# validate the image's checksums + that it can be mounted. If the chosen
# compression format FAILS to produce a verifiable DMG for any reason,
# the script automatically retries with UDZO (the legacy-safe default)
# and prints a warning to stderr rather than leaving the operator empty
# handed.
#
# Notarization (Apple Developer ID required)
# ------------------------------------------
# After running this, notarize the DMG itself (notarytool wants the DMG
# rather than the .app for the upload). All compression formats above
# are accepted by Apple's notary service.
#
#     xcrun notarytool submit Meshroom-<v>-arm64.dmg \
#         --apple-id you@example.com --team-id TEAMID \
#         --password APP-SPECIFIC-PWD --wait
#     xcrun stapler staple Meshroom-<v>-arm64.dmg
#
# Usage
# -----
#   scripts/make_dmg.sh [--app PATH] [--out PATH] [--compression FORMAT]
#                       [--no-verify]
#
set -euo pipefail

APP=""
OUT_DMG=""
COMPRESSION="ulmo"   # ulmo | udzo | udzo-max | ulfo
DO_VERIFY=1

while [ $# -gt 0 ]; do
    case "$1" in
        --app) APP="$2"; shift 2 ;;
        --out) OUT_DMG="$2"; shift 2 ;;
        --compression) COMPRESSION="$2"; shift 2 ;;
        --no-verify) DO_VERIFY=0; shift ;;
        -h|--help) sed -n '2,/^set/p' "$0" | sed -n '/^# /p' | sed 's/^# //'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Validate compression choice up-front so we don't fail mid-build.
case "$COMPRESSION" in
    udzo|udzo-max|ulfo|ulmo) ;;
    *)
        echo "unknown --compression value: $COMPRESSION" >&2
        echo "  valid: udzo, udzo-max, ulfo, ulmo" >&2
        exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -z "$APP" ] && APP="$ROOT/build/release/Meshroom.app"
[ -d "$APP" ] || { echo "no .app at $APP" >&2; exit 2; }

# Version pull from CHANGELOG header (matches package_macos_app.sh).
VERSION="$(grep -E '^## \[' "$ROOT/CHANGELOG.md" 2>/dev/null | sed -nE '2s/.*\[([0-9.]+)\].*/\1/p')"
[ -z "$VERSION" ] && VERSION="0.1.0-dev"

[ -z "$OUT_DMG" ] && OUT_DMG="$ROOT/build/release/Meshroom-$VERSION-arm64.dmg"
mkdir -p "$(dirname "$OUT_DMG")"

# Staging dir for the DMG layout (deleted at end).
STAGE="$(mktemp -d -t mr_dmg)"
trap 'rm -rf "$STAGE"' EXIT

echo "[1/4] Staging DMG layout"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/README.txt" <<EOF
Meshroom (AliceVision for Mac) $VERSION — Apple Silicon

To install:
  1. Drag Meshroom.app into the Applications folder shortcut on the right.
  2. On first launch, right-click Meshroom.app → Open (Gatekeeper bypass).

Built-in features:
  - 60 native ARM64 photogrammetry binaries (full Meshroom 2026.1.0 template set)
  - 4 CoreML models: BiRefNet (segmentation), YOLOv8n (sphere detect),
    MoGe-2 (mono-depth), TinyRoMa (dense matching)
  - Native pyalicevision SWIG bindings (hdr, sfmData, sfmDataIO)
  - Mac-native QtQuick3D scene preview viewer

Bug reports: https://github.com/SeedeXR/alicevision-for-mac/issues
EOF

# Map our friendly compression names to the hdiutil flags they imply.
hdiutil_args_for_compression() {
    case "$1" in
        udzo)     printf '%s\0' "-format" "UDZO" ;;
        udzo-max) printf '%s\0' "-format" "UDZO" "-imagekey" "zlib-level=9" ;;
        ulfo)     printf '%s\0' "-format" "ULFO" ;;
        ulmo)     printf '%s\0' "-format" "ULMO" ;;
    esac
}

# create_dmg <compression-name> <output-path>
# Returns the hdiutil exit code.
create_dmg() {
    local fmt="$1" out="$2"
    local args=()
    while IFS= read -r -d '' arg; do args+=("$arg"); done \
        < <(hdiutil_args_for_compression "$fmt")
    rm -f "$out"
    hdiutil create -volname "Meshroom $VERSION" \
                   -srcfolder "$STAGE" \
                   -ov \
                   "${args[@]}" \
                   "$out"
}

# verify_dmg <path> — returns 0 if hdiutil considers the image valid.
verify_dmg() {
    [ "$DO_VERIFY" -eq 0 ] && return 0
    hdiutil verify "$1" > /dev/null 2>&1
}

echo "[2/4] Creating DMG (compression: $COMPRESSION)"
t0=$(date +%s)
fallback_used=0
if ! create_dmg "$COMPRESSION" "$OUT_DMG"; then
    echo "WARNING: hdiutil create with --compression=$COMPRESSION failed." >&2
    if [ "$COMPRESSION" != "udzo" ]; then
        echo "         Falling back to udzo (legacy-safe)." >&2
        if ! create_dmg "udzo" "$OUT_DMG"; then
            echo "ERROR: udzo fallback also failed; check disk space and Gatekeeper." >&2
            exit 1
        fi
        fallback_used=1
        COMPRESSION="udzo"
    else
        echo "ERROR: udzo failed; check disk space and hdiutil." >&2
        exit 1
    fi
fi
t_create=$(($(date +%s) - t0))
echo "       create elapsed: ${t_create}s"

echo "[3/4] Verifying DMG"
if verify_dmg "$OUT_DMG"; then
    echo "       hdiutil verify: OK"
else
    if [ "$COMPRESSION" != "udzo" ] && [ "$fallback_used" -eq 0 ]; then
        echo "WARNING: hdiutil verify failed for $COMPRESSION DMG." >&2
        echo "         Falling back to udzo." >&2
        if ! create_dmg "udzo" "$OUT_DMG" || ! verify_dmg "$OUT_DMG"; then
            echo "ERROR: udzo fallback verify also failed." >&2
            exit 1
        fi
        fallback_used=1
        COMPRESSION="udzo"
        echo "       udzo verify: OK"
    else
        echo "ERROR: hdiutil verify failed even on udzo." >&2
        exit 1
    fi
fi

echo "[4/4] Done"
DMG_SIZE="$(du -sh "$OUT_DMG" | awk '{print $1}')"
echo "  $OUT_DMG"
echo "  size:         $DMG_SIZE"
echo "  compression:  $COMPRESSION$([ "$fallback_used" -eq 1 ] && echo ' (fallback)')"
echo "  create time:  ${t_create}s"
echo
echo "Next: notarize (Apple Developer ID required):"
echo "  xcrun notarytool submit \"$OUT_DMG\" \\"
echo "      --apple-id you@example.com --team-id TEAMID \\"
echo "      --password APP-SPECIFIC-PWD --wait"
echo "  xcrun stapler staple \"$OUT_DMG\""
