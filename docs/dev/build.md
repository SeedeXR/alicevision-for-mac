# Building from source

How to build the Apple Silicon port from source. Distilled from the
repo-root `BUILD.md`.

For the upfront [installation](../user/install.md) story see the User Guide.

## Prerequisites

### System

- **macOS 14.0+** (Sonoma or newer). `CMAKE_OSX_DEPLOYMENT_TARGET = 14.0` is
  pinned in `CMakeLists.txt`.
- **Xcode 26+** with the Metal toolchain. Verify with
  `xcrun metal --version` and `xcrun metallib --version`.
- **Apple Silicon arm64.** `CMAKE_OSX_ARCHITECTURES = arm64` is forced; the
  build warns on `x86_64`.
- **CMake ≥ 3.30** (root `cmake_minimum_required`).
- **Homebrew.** `AV_USE_HOMEBREW_DEPS=ON` (default) shells out to
  `brew --prefix` and prepends it to `CMAKE_PREFIX_PATH`.

### Homebrew packages

For the **kernel-only** build (no upstream tree, no pipeline binaries) you
need only Eigen3:

```bash
brew install cmake ninja eigen
```

For the **full pipeline** build (`-DAV_BUILD_UPSTREAM=ON
-DAV_BUILD_UPSTREAM_DEPTHMAP=ON`) the upstream dependency tree pulls in:

```bash
brew install \
    boost ceres-solver openimageio openexr imath zlib \
    libomp pkgconf alembic assimp geogram imath lemon \
    nanoflann onnxruntime open-mesh
```

(The Homebrew `lemon` package is the SQLite parser-generator — *not* the
COIN-OR graph library — so we vendor LEMON 1.3.1 under `third_party/lemon/`
and patch it for C++17. See `memory/mental_note.md` §8a.)

## One-shot build

=== "Kernel only (smoke tests)"

    ```bash
    cmake -S . -B build -G Ninja
    cmake --build build
    cd build && ctest                  # 37/37
    ```

=== "Full pipeline (12 binaries)"

    ```bash
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DAV_BUILD_UPSTREAM=ON \
        -DAV_BUILD_UPSTREAM_DEPTHMAP=ON \
        -DAV_USE_HOMEBREW_DEPS=ON
    cmake --build build
    cd build && ctest                  # 37/37
    ```

=== "Release tarball"

    ```bash
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DAV_BUILD_UPSTREAM=ON \
        -DAV_BUILD_UPSTREAM_DEPTHMAP=ON \
        -DAV_USE_HOMEBREW_DEPS=ON
    cmake --build build
    cmake --build build --target package
    # → build/release/alicevision-for-mac-<VERSION>-arm64.tar.gz
    ```

## Build artifacts

```
build/
├── src/av_gpu/libav_gpu.a               Layer 1 (Metal abstraction)
├── src/depth_map_metal/libav_depth_map_metal.a   Layer 2
├── src/shaders/default.metallib         35 kernel entries
├── tests/test_*                         37 test executables
└── (with AV_BUILD_UPSTREAM=ON)
    aliceVision_cameraInit
    aliceVision_featureExtraction
    aliceVision_imageMatching
    aliceVision_featureMatching
    aliceVision_incrementalSfM
    aliceVision_prepareDenseScene
    aliceVision_depthMapEstimation
    aliceVision_depthMapFiltering
    aliceVision_meshing
    aliceVision_meshFiltering
    aliceVision_texturing
    aliceVision_importMiddlebury
    default.metallib                     (staged here too)
```

## Build options

All options are at the top of the root `CMakeLists.txt`. Full reference
at [CMake options](../reference/cmake-options.md).

| Option                       | Default | When to flip |
| ---------------------------- | ------- | ------------ |
| `AV_USE_METAL`               | `ON`    | Always ON on Apple. |
| `AV_USE_CUDA`                | `OFF`   | `CMakeLists.txt` aborts if you flip it. |
| `AV_BUILD_TESTS`             | `ON`    | OFF for distribution/CI. |
| `AV_BUILD_HELLO_METAL`       | `ON`    | OFF to skip `test_metal_hello` + `test_texture_smoke`. |
| `AV_BUILD_UPSTREAM`          | `OFF`   | ON to build the 12 pipeline binaries. |
| `AV_BUILD_UPSTREAM_DEPTHMAP` | `OFF`   | ON to build the 12-module upstream dep subset. |
| `AV_USE_HOMEBREW_DEPS`       | `ON`    | OFF only with pre-populated `CMAKE_PREFIX_PATH`. |
| `AV_PROFILE_ADAPTER`         | `OFF`   | ON to enable per-forwarder timing (see [Performance profiling](perf.md)). |

## Running tests

```bash
cd build
ctest                                  # 37/37 expected
ctest -j1                              # serialize (use this if a test is flaky on -j8)
ctest -R test_depth_pipeline --output-on-failure -V  # specific test
```

Notable end-to-end tests:

| Test                          | What it validates |
| ----------------------------- | ----------------- |
| `test_metal_hello`            | metal-cpp wiring + SAXPY smoke. |
| `test_texture_smoke`          | RAII textures, bilinear, mipmap cascade. |
| `test_sgm_pipeline`           | `init_sim → compute_similarity → optimize → retrieve_best_depth`. |
| `test_refine_pipeline`        | `init_refine → refine_similarity → refine_best_depth`. |
| `test_depth_pipeline`         | Full SGM → Bridge → Refine → Optimize chain. |
| `test_multi_t_aggregation`    | WTA + FP16 additive across multiple T cameras. |
| `test_device_mipmap_image`    | `DeviceMipmapImage` end-to-end. |
| `test_device_cache`           | `LRUCache` + `DeviceCache` eviction. |
| `test_device_stream_manager`  | Multi-queue parallel dispatch. |
| `test_volume_optimize_adaptive_p2` | S31 adaptive-P2 path. |
| `test_upstream_adapter`       | Adapter forwarder smoke. |

## Path C (`AV_BUILD_UPSTREAM_DEPTHMAP=ON`)

This builds the **32 upstream module subdirectories** that the depthMap host
code and the surrounding photogrammetry pipeline depend on. `Path C` lives
in `cmake/UpstreamShim.cmake` and provides shim implementations of
upstream's CMake macros (`alicevision_add_library`, `alicevision_add_test`,
`alicevision_add_interface`, `alicevision_add_software`,
`alicevision_swig_add_library`) so each per-module `CMakeLists.txt` composes
without pulling in upstream's install rules, SOVERSION dance, or Windows
.rc generation.

The upstream tree itself is **never edited on disk**. Quirks accommodated at
CMake time:

- `Boost::system` stubbed as `INTERFACE IMPORTED` (modern Boost made it
  header-only; Homebrew dropped the separate config).
- `Coin::Clp` / `Coin::CoinUtils` / `Coin::Osi` stubbed as empty INTERFACE
  targets (Coin-OR not on Homebrew; `linearProgramming` is INTERFACE-only
  anyway and depthMap doesn't solve LPs at runtime).
- `ALICEVISION_ROTATION_AVERAGING_WITH_BOOST` defined globally so
  `multiview/rotationAveraging` uses Boost.Graph instead of LEMON.
- `multiview/rotationAveraging/l1.cpp` is patched at configure time via
  `file(READ … REPLACE … WRITE …)` to drop `const` from three default-
  initialized `Eigen::Matrix` declarations (clang 21 enforces the
  [dcl.init] rule that const non-user-defined-default-ctor class objects
  must have an initializer). Patched copy → `build/upstream-patched/`.

Full rationale in `PORTING_NOTES.md` §10.

## Troubleshooting

### Boost.System missing

Modern Boost (≥1.86) made it header-only; the `Boost::system` interface stub
in the root `CMakeLists.txt` handles this for `AV_BUILD_UPSTREAM_DEPTHMAP=ON`.
If you see it elsewhere, mirror the pattern:

```cmake
if(NOT TARGET Boost::system)
    add_library(Boost::system INTERFACE IMPORTED)
    if(TARGET Boost::headers)
        set_target_properties(Boost::system PROPERTIES
            INTERFACE_LINK_LIBRARIES Boost::headers)
    endif()
endif()
```

### Metal license not accepted

```bash
sudo xcodebuild -license accept
```

### `default.metallib` fails to load at test runtime

Check:

1. `build/src/shaders/default.metallib` exists.
2. `build/tests/default.metallib` exists (staged by `av_install_metallib`).
3. `xcrun metallib --version` runs without licence prompts.

For ad-hoc debug, load by absolute path:
`MTL::Device::default_device().load_library("/path/to/default.metallib")`.

## Clean rebuild

```bash
rm -rf build
cmake -S . -B build -G Ninja
cmake --build build
```

There are no generated files outside `build/`. Even the `l1.cpp` patch lands
in `build/upstream-patched/`, not in source. `upstream/` stays read-only.
