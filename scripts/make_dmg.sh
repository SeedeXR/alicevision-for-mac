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
# After running this, notarize the DMG itself (notarytool wants the DMG
# rather than the .app for the upload):
#
#     xcrun notarytool submit Meshroom-<v>-arm64.dmg \
#         --apple-id you@example.com --team-id TEAMID \
#         --password APP-SPECIFIC-PWD --wait
#     xcrun stapler staple Meshroom-<v>-arm64.dmg
#
# Usage:
#   scripts/make_dmg.sh [--app /path/to/Meshroom.app] [--out /path/to/output.dmg]
#
set -euo pipefail

APP=""
OUT_DMG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --app) APP="$2"; shift 2 ;;
        --out) OUT_DMG="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^set/p' "$0" | sed -n '/^# /p' | sed 's/^# //'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

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

echo "[1/3] Staging DMG layout"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/README.txt" <<EOF
Meshroom (AliceVision for Mac) $VERSION — Apple Silicon

To install:
  1. Drag Meshroom.app into the Applications folder shortcut on the right.
  2. On first launch, right-click Meshroom.app → Open (Gatekeeper bypass).

Built-in features:
  - 12 native ARM64 Metal photogrammetry binaries
  - AI foreground segmentation via BiRefNet CoreML (CPU + Metal GPU)
  - Mac-native QtQuick3D scene preview viewer

Limitations:
  - HDR / panorama compute: gated on Phase 13–14 binary builds
  - Modern SfM templates (cameraTracking*) not yet runnable

Bug reports: https://github.com/<placeholder>/alicevision-for-mac/issues
EOF

echo "[2/3] Creating compressed DMG"
rm -f "$OUT_DMG"
# UDZO = compressed; ADC is slightly smaller but takes longer.
hdiutil create -volname "Meshroom $VERSION" \
               -srcfolder "$STAGE" \
               -ov \
               -format UDZO \
               "$OUT_DMG"

echo "[3/3] Done"
DMG_SIZE="$(du -sh "$OUT_DMG" | awk '{print $1}')"
echo "  $OUT_DMG  ($DMG_SIZE)"
echo
echo "Next: notarize (Apple Developer ID required):"
echo "  xcrun notarytool submit \"$OUT_DMG\" \\"
echo "      --apple-id you@example.com --team-id TEAMID \\"
echo "      --password APP-SPECIFIC-PWD --wait"
echo "  xcrun stapler staple \"$OUT_DMG\""
