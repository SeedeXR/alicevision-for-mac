#!/bin/bash
# Phase 12 smoke-test: reconfigure, install into a scratch prefix, and
# poke at the layout. Validates the install() rules + ad-hoc codesign
# block added in CMakeLists.txt for Homebrew packaging.
set -e

SRC=/Users/alexmkwizu/Documents/SoftwareProjects/alicevision-mac/alicevision-for-mac
BLD="$SRC/build"
PREFIX=/tmp/av-install

rm -rf "$PREFIX"

echo "=== Reconfigure ==="
cmake -B "$BLD" -S "$SRC"

echo
echo "=== Install ==="
cmake --install "$BLD" --prefix "$PREFIX"

echo
echo "=== bin/ ==="
ls -la "$PREFIX/bin/"

echo
echo "=== share/aliceVision/ ==="
ls -la "$PREFIX/share/aliceVision/"

echo
echo "=== aliceVision_depthMapEstimation --help (first 20 lines) ==="
ALICEVISION_ROOT="$PREFIX" "$PREFIX/bin/aliceVision_depthMapEstimation" --help 2>&1 | head -20 || true

echo
echo "=== codesign verification (first 3 binaries) ==="
for b in aliceVision_cameraInit aliceVision_depthMapEstimation aliceVision_texturing; do
  codesign -dv "$PREFIX/bin/$b" 2>&1 | head -3
  echo "---"
done
