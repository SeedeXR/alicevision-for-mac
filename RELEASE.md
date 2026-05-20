# Release process — alicevision-for-mac

This document describes how to cut a redistributable binary tarball for
the Apple Silicon Metal port of AliceVision. The output is a tarball
containing the twelve `aliceVision_*` pipeline CLIs, the
`default.metallib` Metal shader archive, and the `share/aliceVision/`
runtime data (OCIO config, LUTs, camera-sensor database).

The tarball is **not fully vendored** — runtime dynamic-library
dependencies (alembic, boost, ceres-solver, eigen, geogram, imath,
lemon, libomp, nanoflann, onnxruntime, open-mesh, openexr,
openimageio) are expected to be present via Homebrew on the consumer
machine. See `Formula/alicevision-for-mac.rb` for the canonical
dependency set.

---

## 1. Prerequisites

### Toolchain

- macOS 14.0 (Sonoma) or newer on Apple Silicon (`arm64`).
- Xcode 26 (or newer) — the build uses Metal 4 / metal-cpp headers
  from the Xcode-bundled SDK.
- CMake ≥ 3.30 (`brew install cmake`).
- Ninja (`brew install ninja`).
- `pkgconf` (`brew install pkgconf`).

### Runtime + link-time deps (Homebrew)

```bash
brew install \
    alembic assimp boost ceres-solver eigen geogram imath lemon \
    libomp nanoflann onnxruntime open-mesh openexr openimageio
```

These match the `depends_on` block in
`Formula/alicevision-for-mac.rb`. The bundled tarball is **not**
self-contained for runtime dylibs — users must have these installed.

---

## 2. Build

From the repository root:

```bash
cmake -S . -B build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DAV_BUILD_UPSTREAM=ON \
      -DAV_BUILD_UPSTREAM_DEPTHMAP=ON \
      -DAV_USE_HOMEBREW_DEPS=ON \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
      -DCMAKE_OSX_ARCHITECTURES=arm64

cmake --build build
```

The build target produces twelve pipeline binaries plus
`default.metallib` under `build/`.

### Optional: sanity-check the test suite

```bash
ctest --test-dir build --output-on-failure
```

Expected: `37/37` pass on a clean Apple-Silicon build (recorded
result as of the S42 checkpoint).

---

## 3. Install (manual)

```bash
cmake --install build --prefix /tmp/av-install-release
```

This stages a clean layout:

```
/tmp/av-install-release/
├── bin/
│   ├── aliceVision_cameraInit
│   ├── aliceVision_featureExtraction
│   ├── aliceVision_imageMatching
│   ├── aliceVision_featureMatching
│   ├── aliceVision_incrementalSfM
│   ├── aliceVision_prepareDenseScene
│   ├── aliceVision_depthMapEstimation
│   ├── aliceVision_depthMapFiltering
│   ├── aliceVision_meshing
│   ├── aliceVision_meshFiltering
│   ├── aliceVision_texturing
│   ├── aliceVision_importMiddlebury
│   └── default.metallib
└── share/
    └── aliceVision/
        ├── cameraSensors.db
        ├── config.ocio
        └── luts/
            ├── InvRRT.Rec.709.Log2_48_nits_Shaper.spi3d
            ├── Log2_48_nits_Shaper.RRT.Rec.709.spi3d
            └── Log2_48_nits_Shaper_to_linear.spi1d
```

The install step also ad-hoc codesigns every binary via
`install(CODE ...)` blocks in the root `CMakeLists.txt` — codesigning
must happen **after** CMake fixes RPATHs, otherwise the signature is
invalidated.

---

## 4. Build the tarball

### Recommended: the `package` target

```bash
cmake --build build --target package
# (or, with the Ninja generator)
ninja -C build package
```

This stages a fresh install tree under
`build/release/alicevision-for-mac-<VERSION>/` and produces
`build/release/alicevision-for-mac-<VERSION>-arm64.tar.gz`. The
target wipes any previous staging directory so re-running it always
ships a clean tree.

### Manual fallback

If you need to drive the tarball step from a script or CI outside
CMake:

```bash
cd /tmp
rm -rf /tmp/av-install-release
cmake --install <path-to-build> --prefix /tmp/av-install-release
mv /tmp/av-install-release /tmp/alicevision-for-mac-0.1.0
tar czf /tmp/alicevision-for-mac-0.1.0-arm64.tar.gz \
    -C /tmp \
    alicevision-for-mac-0.1.0
```

(macOS BSD-tar accepts `-s '|^old|new|'` for in-flight path rewriting
if you prefer not to rename the staging directory; GNU tar uses
`--transform 's|^old|new|'`. The recommended `package` target sidesteps
this by staging directly under the final name.)

---

## 5. Smoke-test the tarball

Extract to a clean path and confirm the binaries run:

```bash
mkdir -p /tmp/extracted-release
cd /tmp/extracted-release
tar xzf /tmp/alicevision-for-mac-0.1.0-arm64.tar.gz

ls alicevision-for-mac-0.1.0/bin   # 12 binaries + default.metallib
ls alicevision-for-mac-0.1.0/share/aliceVision  # cameraSensors.db, config.ocio, luts/

# Run one of the binaries — the "unrecognised option" error is
# expected and confirms the binary loaded + parsed CLI:
ALICEVISION_ROOT=/tmp/extracted-release/alicevision-for-mac-0.1.0 \
    /tmp/extracted-release/alicevision-for-mac-0.1.0/bin/aliceVision_depthMapEstimation 2>&1 | head -5

# Verify default.metallib is alongside the binary (loaded via
# @executable_path in Device::load_library({})):
ls /tmp/extracted-release/alicevision-for-mac-0.1.0/bin/default.metallib
```

If a binary fails with `dyld: Library not loaded: ...`, the missing
dylib is a Homebrew runtime dep that the consumer machine is missing
— `brew install` it (see §1). The tarball is intentionally
non-vendored.

---

## 6. Codesign verification

Every binary in the install tree is ad-hoc signed at install time.
Verify with:

```bash
for bin in /tmp/extracted-release/alicevision-for-mac-0.1.0/bin/aliceVision_*; do
    codesign -dv "$bin" 2>&1 | grep -E '^(Executable|Signature|TeamIdentifier)' \
        | sed "s|^|$(basename "$bin"): |"
done
```

Expected output per binary:

```
aliceVision_<name>: Executable=/.../bin/aliceVision_<name>
aliceVision_<name>: Signature=adhoc
aliceVision_<name>: TeamIdentifier=not set
```

`Signature=adhoc` is the expected current state. This is sufficient
for local execution and for users who explicitly trust the binaries
(e.g. via `xattr -d com.apple.quarantine`), but is **not** enough to
pass Gatekeeper on machines where the tarball is downloaded from the
internet — see §8.

---

## 7. RPATH / linkage check

The pipeline binaries link against Homebrew dylibs (boost, openimageio,
openexr, ...) and pick them up via the install-time RPATH that CMake
embeds. Sanity-check with:

```bash
otool -L /tmp/extracted-release/alicevision-for-mac-0.1.0/bin/aliceVision_depthMapEstimation | head -30
otool -l /tmp/extracted-release/alicevision-for-mac-0.1.0/bin/aliceVision_depthMapEstimation \
    | grep -A2 LC_RPATH
```

You should see absolute paths into `/opt/homebrew/...` for the
runtime deps, and `@executable_path` (or `@loader_path`) in the
RPATH section. The Metal shader archive is loaded via
`@executable_path/default.metallib` so it stays next to whichever
binary is invoked.

---

## 8. Known limitations

- **Ad-hoc signed, not Developer-ID + notarized.** A user who
  downloads the tarball over the network will hit Gatekeeper. Until
  Developer-ID signing + `notarytool` integration lands (Phase 12
  follow-up), workarounds are:
  - `xattr -dr com.apple.quarantine /path/to/alicevision-for-mac-0.1.0/`
  - Or distribute via Homebrew (the formula builds from source on the
    user's machine, so Gatekeeper doesn't apply).
- **Not fully vendored.** Homebrew runtime dylibs are required on
  the host. A future bundling step using `dylibbundler` /
  `install_name_tool` could produce a fully-portable `.app` /
  `.tar.gz`, but that is out of scope for the current release.
- **Apple Silicon only.** The build refuses to configure on
  `x86_64`; the tarball name reflects this (`-arm64.tar.gz`).
- **Upstream source not embedded.** The Homebrew formula expects the
  release tarball to also carry `upstream/` (the AliceVision source
  tree) so the formula's `cmake --build` step has something to compile.
  Either add a `resource "upstream"` block to the formula or include
  `upstream/` in the source tarball — see the comment in
  `Formula/alicevision-for-mac.rb`.

---

## 9. Quick reference

```bash
# Full release flow, top-to-bottom:
cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DAV_BUILD_UPSTREAM=ON \
      -DAV_BUILD_UPSTREAM_DEPTHMAP=ON \
      -DAV_USE_HOMEBREW_DEPS=ON
cmake --build build
ctest --test-dir build --output-on-failure      # 37/37
cmake --build build --target package            # → build/release/*.tar.gz
```
