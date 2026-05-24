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
# 3b. Bundle Homebrew dylibs (the dylibbundler step)
# -------------------------------------------------------------------- #
# Walks every aliceVision_* binary + recursively every copied dylib,
# rewriting `/opt/homebrew/...` references to
# `@executable_path/../lib/<name>`. After this step the .app
# is self-contained: it runs on any macOS Apple Silicon machine even
# without Homebrew installed.
#
# This is a self-contained mini-dylibbundler in bash so we don't add a
# `brew install dylibbundler` dependency to the build process.

echo "[3b] Bundling Homebrew dylibs"

LIB_DIR="$APP/Contents/Resources/lib"
mkdir -p "$LIB_DIR"

# Track already-processed dylib basenames so we don't loop on circular
# inter-dylib references (e.g. boost_graph → boost_iostreams → boost_*).
PROCESSED_LIBS=()

_is_processed() {
    local name="$1"
    # The `${arr[@]+"${arr[@]}"}` idiom expands to empty (not "unbound")
    # when the array has zero elements — required under `set -u`.
    for p in ${PROCESSED_LIBS[@]+"${PROCESSED_LIBS[@]}"}; do
        [ "$p" = "$name" ] && return 0
    done
    return 1
}

# Extract Homebrew + non-system dylib references from a Mach-O. We keep
# /usr/lib/* and /System/* (those are guaranteed on every Mac) and bundle
# everything else. Every step is `|| true` so a grep-no-match (= exit 1)
# under `set -e` doesn't kill the whole pipeline.
_external_deps() {
    local target="$1"
    local result
    result="$(otool -L "$target" 2>/dev/null \
        | tail -n +2 \
        | awk '{print $1}' \
        | { grep -vE '^(/usr/lib/|/System/|@executable_path|@rpath|@loader_path)' || true; } \
        | { grep -v "^$(basename "$target")\$" || true; } \
        | sort -u)"
    printf '%s\n' "$result"
}

_bundle_target() {
    local target="$1"
    local deps
    deps="$(_external_deps "$target")"
    [ -z "$deps" ] && return 0
    # IMPORTANT: every loop variable that survives across a recursive
    # _bundle_target call MUST be declared `local`. The `while read -r
    # VAR` syntax does NOT auto-localize VAR — the recursive call's own
    # `while read -r dep` would clobber the outer caller's `dep`, so
    # after recursion returns, install_name_tool -change runs with an
    # empty `dep` argument and silently no-ops (returning rc=0 with no
    # error). That bug burned us 2026-05-23 — 7/21 references in
    # aliceVision_cameraInit silently failed to rewrite, leaving
    # `/opt/homebrew/...` leaks the user only saw after distribution.
    local dep dep_name dest
    while IFS= read -r dep; do
        [ -z "$dep" ] && continue
        dep_name="$(basename "$dep")"
        dest="$LIB_DIR/$dep_name"
        # First time we see this dep: copy + rewrite its own ID + recurse.
        if [ ! -f "$dest" ]; then
            if [ ! -f "$dep" ]; then
                echo "  ! missing dep on host: $dep (referenced by $target)" >&2
                continue
            fi
            cp "$dep" "$dest"
            chmod u+w "$dest"
            # Rewrite the lib's own install_name so dyld matches references.
            install_name_tool -id "@executable_path/../lib/$dep_name" "$dest" 2>/dev/null || true
            # Recurse: any non-system dep this lib depends on must also
            # land in lib/ + get its references rewritten.
            if ! _is_processed "$dep_name"; then
                PROCESSED_LIBS+=("$dep_name")
                _bundle_target "$dest"
            fi
        fi
        # Rewrite the original reference in `target`. -change is idempotent.
        install_name_tool -change "$dep" "@executable_path/../lib/$dep_name" "$target" 2>/dev/null || true
    done <<< "$deps"
}

bundle_count=0
for bin in "$APP/Contents/Resources/alicevision"/aliceVision_*; do
    [ -f "$bin" ] && [ -x "$bin" ] || continue
    _bundle_target "$bin"
    bundle_count=$((bundle_count + 1))
done

n_libs=$(ls -1 "$LIB_DIR" 2>/dev/null | wc -l | tr -d ' ')
echo "  bundled $n_libs dylibs across $bundle_count binaries"
echo "  Resources/lib size: $(du -sh "$LIB_DIR" 2>/dev/null | awk '{print $1}')"

# Re-sign every modified Mach-O with an ad-hoc signature. install_name_tool
# invalidates the original codesignature, so macOS Gatekeeper SIGKILLs
# the binary at launch (rc=137) unless we re-sign. Ad-hoc (-s -) is free
# + works for local distribution; a real Developer ID identity is wired
# in `scripts/codesign_macos_app.sh` for shippable .app bundles.
echo "[3c] Ad-hoc resign of modified binaries + dylibs"
codesign_count=0
for f in "$APP/Contents/Resources/alicevision"/aliceVision_* \
         "$LIB_DIR"/*.dylib; do
    [ -f "$f" ] || continue
    codesign --force --sign - "$f" 2>/dev/null || true
    codesign_count=$((codesign_count + 1))
done
echo "  ad-hoc signed $codesign_count Mach-O files"

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
