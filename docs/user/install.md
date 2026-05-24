# Installation

User-facing install guide for the **Apple Silicon port** of AliceVision.
This page distills `INSTALL_macOS.md` from the repo root.

!!! warning "Honest status (2026-05-20)"
    The full photogrammetry pipeline runs end-to-end on Apple Silicon as of
    S41, and the release tarball (`build/release/*.tar.gz`) bundles 12
    `aliceVision_*` binaries plus `default.metallib` plus the runtime data
    layout. A Homebrew formula is on disk at `Formula/alicevision-for-mac.rb`
    but the `url` / `sha256` placeholders still need a tagged GitHub release.

## 1. System requirements

| Requirement | Value | Why |
| --- | --- | --- |
| macOS | 14.0 (Sonoma) or newer | `CMAKE_OSX_DEPLOYMENT_TARGET = 14.0` pinned in `CMakeLists.txt`. |
| CPU | Apple Silicon (M1 / M2 / M3 / M4) | `CMAKE_OSX_ARCHITECTURES = arm64` forced; build warns on `x86_64`. |
| Xcode | 26 or newer (clang 21, Metal 4 toolchain) | `cmake/Metal.cmake` shells out to `xcrun metal` and `xcrun metallib`. |
| CMake | ≥ 3.30 | Set by the root `cmake_minimum_required`. |
| Homebrew | Required for the default build | `AV_USE_HOMEBREW_DEPS=ON` prepends `brew --prefix` to `CMAKE_PREFIX_PATH`. |

**Out of scope for milestone 1**: Intel (x86_64) Macs, Rosetta, macOS older
than 14.0.

## 2. Install via Homebrew

=== "Tap (not yet published)"

    ```bash
    # [TODO] No public tap yet; the formula at Formula/alicevision-for-mac.rb
    # has placeholder url + sha256. Once the first GitHub release is tagged,
    # the canonical invocation will be:
    brew tap placeholder/alicevision-for-mac
    brew install alicevision-for-mac
    ```

=== "Local formula (today)"

    ```bash
    # Install the runtime + link-time deps first:
    brew install \
        alembic assimp boost ceres-solver eigen geogram imath lemon \
        libomp nanoflann onnxruntime open-mesh openexr openimageio cmake \
        ninja pkgconf

    # Install from the local formula in this repo (build from source):
    brew install --build-from-source ./Formula/alicevision-for-mac.rb
    ```

The formula declares `depends_on arch: :arm64` and `depends_on macos: :sonoma`
to match the CMake-level guards.

## 3. Install from source

The short version:

```bash
git clone https://github.com/SeedeXR/alicevision-for-mac
cd alicevision-for-mac
brew install cmake ninja swig eigen opencv \
    alembic assimp boost ceres-solver geogram imath lemon \
    libomp nanoflann onnxruntime open-mesh openexr openimageio pkgconf

# Fetch the 4 CoreML models (~750 MB) into ai-models/
bash scripts/download_ai_models.sh

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAV_BUILD_UPSTREAM=ON \
    -DAV_BUILD_UPSTREAM_DEPTHMAP=ON \
    -DAV_BUILD_PYALICEVISION=ON

cmake --build build
ctest --test-dir build              # 37/37 expected
```

For the kernel-only minimal build (no upstream tree, no pipeline binaries)
see the [Developer build guide](../dev/build.md).

## 4. Download AI models

The Mac port ships four CoreML `.mlpackage` files (~750 MB total) that
are NOT committed to git — `ai-models/` is gitignored. Native binaries
(`aliceVision_sphereDetection`, `aliceVision_moGe`,
`aliceVision_matchMasking`) and the `SegmentationBiRefNet` Meshroom
node look for them at `<repo>/ai-models/<name>.mlpackage` at runtime.

### Option A: pre-converted bundle (recommended)

```bash
bash scripts/download_ai_models.sh
```

Fetches a single Google Drive archive
([link](https://drive.google.com/file/d/12jt788_0Wab_nahVa7zP2lHfDSAYky5z/view?usp=sharing))
containing all four `.mlpackages` and extracts them into `ai-models/`.

- Skips models already present; pass `--force` to re-download.
- Auto-installs the `gdown` Python tool if missing.
- Pass `--archive /path/to/file.tar.gz` if you've downloaded the
  archive manually (browser → "Download" on the Drive URL above).
- `bash scripts/download_ai_models.sh -h` prints the full flag set.

### Option B: convert from upstream sources

The conversion scripts + per-model recipes are in a companion repo:
<https://github.com/SeedeXR/alicevision-for-mac-models>.

Use this path when you want a different input shape, FP32 precision,
a different backbone (BiRefNet `general` vs `lite`), or are tuning the
conversion for a different distribution model. Not required for normal
usage.

The four models the binaries expect (any names other than these will
not be auto-discovered):

| Filename | Native binary / consumer | Size |
|---|---|---|
| `BiRefNet_lite.mlpackage` | `SegmentationBiRefNet` Meshroom node (default) | ~90 MB |
| `BiRefNet.mlpackage` | `SegmentationBiRefNet` (high-accuracy variant) | ~447 MB |
| `yolov8n.mlpackage` | `aliceVision_sphereDetection` | ~13 MB |
| `moge2_504x672_t1728.mlpackage` | `aliceVision_moGe` | ~187 MB |
| `tiny_roma_v1_480x640.mlpackage` | `aliceVision_matchMasking` | ~5.5 MB |

See [ai-models/README.md](https://github.com/SeedeXR/alicevision-for-mac/blob/main/ai-models/README.md)
for the full per-model integration recipe, ANE outcome matrix, and
runtime discovery rules (`ALICEVISION_MOGE_MLPACKAGE`,
`ALICEVISION_ROMA_MLPACKAGE`, `ALICEVISION_ROOT` env vars).

## 5. Install the release tarball

```bash
cmake --build build --target package
# → build/release/alicevision-for-mac-<VERSION>-arm64.tar.gz
```

Extract and run:

```bash
tar xzf alicevision-for-mac-0.1.0-arm64.tar.gz -C /opt
export ALICEVISION_ROOT=/opt/alicevision-for-mac-0.1.0
/opt/alicevision-for-mac-0.1.0/bin/aliceVision_cameraInit --help
```

Resulting install layout:

```
<prefix>/
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
└── share/aliceVision/
    ├── cameraSensors.db
    ├── config.ocio
    └── luts/
```

The Metal shader archive is loaded via `@executable_path/default.metallib` and
**must remain alongside the binary**. The CMake install rule
(`av_install_metallib`) stages it there.

## 6. Verify

```bash
# Confirm the binary is arm64 + adhoc-signed:
file /opt/alicevision-for-mac-0.1.0/bin/aliceVision_cameraInit
codesign -dv /opt/alicevision-for-mac-0.1.0/bin/aliceVision_cameraInit

# Confirm --help runs (covers most of the dynamic-link surface):
/opt/alicevision-for-mac-0.1.0/bin/aliceVision_cameraInit --help | head -20
```

Expected: `Mach-O 64-bit executable arm64` and `Signature=adhoc`.

## 7. Sample run

The fastest sample is `dataset_monstree/mini3/` (3 JPGs at 4032×3024). Drive
it through the Meshroom wrapper to get a textured mesh in ~1 minute on an M4:

```bash
./scripts/run_meshroom.sh python meshroom-mac/bin/meshroom_batch \
    -i dataset_monstree/mini3 \
    -o /tmp/monstree-out \
    -p photogrammetryLegacy
```

See [Running the pipeline](pipeline.md) for the per-binary breakdown and
[Meshroom integration](meshroom.md) for the wrapper-script internals.

## 8. Known limitations

| Limitation | Tracking |
| --- | --- |
| No public Homebrew tap; formula has placeholder url/sha256. | Phase 12 |
| Ad-hoc codesigned only; downloaded tarballs hit Gatekeeper. Workaround: `xattr -dr com.apple.quarantine`. | Phase 12 |
| Not vendored — Homebrew runtime dylibs must be present on the host. | Phase 12 (`dylibbundler`) |
| Apple Silicon only; build refuses on `x86_64`. | By design |
| `aliceVision_meshing --output foo.abc` errors — use `.sfm` or `.ply`; Alembic is not linked on macOS. | Out of scope |

For full troubleshooting see [Troubleshooting](troubleshooting.md).
