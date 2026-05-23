#!/usr/bin/env bash
#
# scripts/package_macos_app.sh — assemble Meshroom.app from the repo.
#
# **Scaffold, not a finished product.** Produces a runnable .app that
# launches the bundled Python Meshroom on the build machine, but is NOT
# yet self-contained — Homebrew dylibs (boost, ceres-solver, openimageio,
# etc.) are still consumed via `/opt/homebrew/lib`. The launcher falls
# back to that path automatically when `Contents/Resources/lib/` is
# absent, so a dev-mode .app works on the original build machine without
# the dylib-bundling step.
#
# What this script DOES today:
#   - Validates prerequisites (cmake build complete, venv exists, models present)
#   - Mints the .app skeleton (Contents/, MacOS/, Resources/)
#   - Copies in: 12 aliceVision_* + default.metallib + share/aliceVision/
#                ai-models/ (whichever .mlpackages are present)
#                meshroom-mac/ (Meshroom Python package)
#                meshroom-venv/ (PySide6 + numpy + Pillow + coremltools + …)
#                plugins/ai-segmentation/
#                src/python_shim/
#   - Renders Info.plist + launcher.sh from the .tmpl files
#   - Sets the launcher executable
#   - Writes a README.txt inside the .app explaining "dev-mode" caveats
#
# What this script DOES NOT do yet (Phase 2 work — see
# memory/macos_app_packaging.md):
#   - dylibbundler to copy + rewrite Homebrew dylibs (~500 MB delta)
#   - Apple Developer ID codesigning
#   - notarization
#   - DMG / installer pkg generation
#
# Usage:
#   scripts/package_macos_app.sh [<output-dir>]
#
#   Default output: build/release/Meshroom.app
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/build/release}"
APP="$OUT_DIR/Meshroom.app"
TPL_DIR="$ROOT/scripts/macos_app"

# Bundle version — pulled from the most recent release tarball if we can
# find one, else hard-coded.
VERSION="$(grep -E '^## \[' "$ROOT/CHANGELOG.md" 2>/dev/null | sed -nE '2s/.*\[([0-9.]+)\].*/\1/p')"
[ -z "$VERSION" ] && VERSION="0.1.0-dev"

# -------------------------------------------------------------------- #
# 0. Preflight
# -------------------------------------------------------------------- #

echo "[1/8] Preflight"
for dir in "$ROOT/build" "$ROOT/meshroom-mac" "$ROOT/meshroom-venv" \
           "$ROOT/plugins/ai-segmentation" "$ROOT/src/python_shim"; do
    [ -d "$dir" ] || { echo "  ! missing $dir — run cmake build / create venv first" ; exit 2; }
done
for binary in aliceVision_cameraInit aliceVision_featureExtraction \
              aliceVision_featureMatching aliceVision_meshing \
              aliceVision_texturing; do
    [ -x "$ROOT/build/$binary" ] || { echo "  ! missing $ROOT/build/$binary"; exit 2; }
done
[ -f "$ROOT/build/default.metallib" ] || { echo "  ! missing default.metallib"; exit 2; }

# -------------------------------------------------------------------- #
# 1. Skeleton
# -------------------------------------------------------------------- #

echo "[2/8] Skeleton — $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# -------------------------------------------------------------------- #
# 2. AliceVision binaries + metallib + share/
# -------------------------------------------------------------------- #

echo "[3/8] AliceVision binaries"
mkdir -p "$APP/Contents/Resources/alicevision"
for bin in "$ROOT/build"/aliceVision_*; do
    [ -f "$bin" ] && [ -x "$bin" ] || continue
    cp "$bin" "$APP/Contents/Resources/alicevision/"
done
cp "$ROOT/build/default.metallib" "$APP/Contents/Resources/alicevision/"
if [ -d "$ROOT/build/alicevision_root/share" ]; then
    cp -R "$ROOT/build/alicevision_root/share" "$APP/Contents/Resources/alicevision/"
fi

# -------------------------------------------------------------------- #
# 3. AI segmentation models
# -------------------------------------------------------------------- #

echo "[4/8] BiRefNet CoreML models"
mkdir -p "$APP/Contents/Resources/ai-models"
[ -f "$ROOT/ai-models/README.md" ] && cp "$ROOT/ai-models/README.md" "$APP/Contents/Resources/ai-models/"
for pkg in "$ROOT/ai-models/BiRefNet_lite.mlpackage" "$ROOT/ai-models/BiRefNet.mlpackage"; do
    [ -d "$pkg" ] || continue
    cp -R "$pkg" "$APP/Contents/Resources/ai-models/"
done

# -------------------------------------------------------------------- #
# 4. Meshroom Python package + plugins + shim
# -------------------------------------------------------------------- #

echo "[5/8] Python source trees"
# Excludes: __pycache__/, .pytest_cache/, stale per-checkout venvs, build
# artefacts, docs, dev-only tests, .git noise. The .meshroom-venv in
# particular is 1.4 GB of redundant PySide6 wheels left over from an
# earlier dev iteration; it MUST be excluded or the bundle balloons.
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
      --exclude='install/' --exclude='build/' --exclude='dist/' \
      --exclude='meshroom-mac-out/' --exclude='.meshroom-venv/' \
      --exclude='.git/' --exclude='.github/' --exclude='.tx/' \
      --exclude='docs/' --exclude='tests/' --exclude='docker/' \
      --exclude='*.md' --exclude='.DS_Store' \
      "$ROOT/meshroom-mac/" "$APP/Contents/Resources/meshroom-mac/"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
      "$ROOT/plugins/" "$APP/Contents/Resources/plugins/"
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
      "$ROOT/src/python_shim/" "$APP/Contents/Resources/python_shim/"

# -------------------------------------------------------------------- #
# 5. Bundled venv
# -------------------------------------------------------------------- #

echo "[6/8] Python venv (this is the biggest chunk — ~300 MB)"
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
      "$ROOT/meshroom-venv/" "$APP/Contents/Resources/meshroom-venv/"
# Rewrite the shebang lines of any installed console scripts so they
# point at the bundled python, not the original venv path. This is a
# partial fix; for a fully relocatable venv, replace this with
# `python -m venv --copies` + a post-process step.
VENV_PY="$APP/Contents/Resources/meshroom-venv/bin/python3"
if [ -x "$VENV_PY" ]; then
    # The python3 symlink itself is usually fine; the wrapper scripts
    # (e.g. pip, meshroom_compute) may carry an absolute shebang. Patch
    # them to use a relative shebang via /usr/bin/env python3.
    find "$APP/Contents/Resources/meshroom-venv/bin" -type f -maxdepth 1 \
        -exec sed -i '' '1{s|^#!.*python.*|#!/usr/bin/env python3|;}' {} \;
fi

# -------------------------------------------------------------------- #
# 6. Info.plist + launcher
# -------------------------------------------------------------------- #

echo "[7/8] Info.plist + launcher"
sed "s/__VERSION__/$VERSION/g" "$TPL_DIR/Info.plist.tmpl" > "$APP/Contents/Info.plist"
cp "$TPL_DIR/launcher.sh.tmpl" "$APP/Contents/MacOS/meshroom"
chmod +x "$APP/Contents/MacOS/meshroom"

# -------------------------------------------------------------------- #
# 7. Dev-mode README
# -------------------------------------------------------------------- #

cat > "$APP/Contents/Resources/PACKAGING_README.txt" <<EOF
Meshroom.app — dev-mode build ($VERSION)

This bundle is NOT fully self-contained. The launcher script falls back
to /opt/homebrew/lib for the Homebrew-managed dylib dependencies
(boost, ceres-solver, openimageio, openexr, alembic, etc.) and the
system python3. On a machine without Homebrew + the documented brew
deps installed, this .app will fail to launch.

To distribute to end-users, the following Phase 2 work is required (see
memory/macos_app_packaging.md):

1. dylibbundler — copy + relocate Homebrew dylibs into Contents/Resources/lib/.
2. install_name_tool — rewrite @rpath in the 12 aliceVision_* binaries.
3. Apple Developer ID codesigning — codesign --deep --options runtime --sign "Developer ID Application: ..." Meshroom.app
4. notarization — notarytool submit + staple.
5. DMG generation — create-dmg or hdiutil for the consumer download.

Until those steps land, this bundle is for in-house testing only.
EOF

# -------------------------------------------------------------------- #
# 8. Summary
# -------------------------------------------------------------------- #

echo "[8/8] Done"
du -sh "$APP" 2>/dev/null || true
echo "Bundle: $APP"
echo "Launch with: open '$APP'"
echo
echo "Phase 2 (codesign / dylibbundler / notarize / DMG) not yet wired."
echo "See memory/macos_app_packaging.md."
