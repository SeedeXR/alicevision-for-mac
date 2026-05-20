# CMake options

Every `option(AV_*)` declared in the root `CMakeLists.txt`. Flip with
`-D<NAME>=ON|OFF` on the `cmake -S . -B build` line.

| Option | Default | Effect |
|---|---|---|
| `AV_USE_METAL` | `ON` | Reserved-for-future gate on the Metal backend. Currently the subdirectories `src/av_gpu`, `src/depth_map_metal`, `src/shaders` are added unconditionally because there is no CPU-only fallback. Flipping OFF will `FATAL_ERROR` until a CPU path is wired in. |
| `AV_USE_CUDA` | `OFF` | `CMakeLists.txt` aborts if ON. CUDA is unavailable on macOS. |
| `AV_BUILD_TESTS` | `ON` | Build the 37 ctest executables. Set OFF for distribution / CI builds that don't need them. |
| `AV_BUILD_HELLO_METAL` | `ON` | Build `test_metal_hello` + `test_texture_smoke` (the lightweight Metal smoke tests). |
| `AV_BUILD_UPSTREAM` | `OFF` | Configure the upstream AliceVision tree (Phase 2 work; enables the 12 pipeline binaries when combined with `AV_BUILD_UPSTREAM_DEPTHMAP=ON`). |
| `AV_BUILD_UPSTREAM_DEPTHMAP` | `OFF` | Build the 12-module upstream subset depthMap depends on. Combined with `AV_BUILD_UPSTREAM=ON` produces the 12 `aliceVision_*` binaries. See [Project overview](../dev/overview.md) for the "Path C" rationale. |
| `AV_USE_HOMEBREW_DEPS` | `ON` | Shell out to `brew --prefix` and prepend it to `CMAKE_PREFIX_PATH`. Set OFF only with a pre-populated `CMAKE_PREFIX_PATH`. |
| `AV_PROFILE_ADAPTER` | `OFF` | Enable the per-forwarder timing accumulator (S43). When ON, every `cuda_*` adapter records RAII timings and prints a sorted table on `std::atexit`. Zero cost when OFF (macro is `do{}while(0)`). See [Performance profiling](../dev/perf.md). |

## Additional CMake-level toggles

| Flag | Notes |
|---|---|
| `-DCMAKE_BUILD_TYPE=Release` | Default is `RelWithDebInfo`. |
| `-DCMAKE_BUILD_TYPE=Debug` | Useful when chasing kernel bugs; tests still pass but are slower. |
| `-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0` | Raise the minimum macOS above the 14.0 default. |
| `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` | For clangd / IDE integration. The repo ships `build/compile_commands.json` after the first build. |
| `-DCMAKE_OSX_ARCHITECTURES=arm64` | Forced to `arm64` if unset. Building `x86_64` emits a warning and is out of scope. |

## Recommended combinations

=== "Kernel development"

    ```bash
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DAV_BUILD_TESTS=ON
    ```

=== "Pipeline integration"

    ```bash
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DAV_BUILD_UPSTREAM=ON \
        -DAV_BUILD_UPSTREAM_DEPTHMAP=ON
    ```

=== "Profiling"

    ```bash
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DAV_BUILD_UPSTREAM=ON \
        -DAV_BUILD_UPSTREAM_DEPTHMAP=ON \
        -DAV_PROFILE_ADAPTER=ON
    ```

=== "Release tarball"

    See [Building from source → Release tarball](../dev/build.md#one-shot-build).

## Pinned values you can't change

| Variable | Value | Why |
|---|---|---|
| `CMAKE_CXX_STANDARD` | `20` | Apple Clang 21; `<format>`, `<bit>` headers used. |
| `CMAKE_CXX_STANDARD_REQUIRED` | `ON` | No silent downgrade. |
| `CMAKE_CXX_EXTENSIONS` | `OFF` | No `-std=gnu++20`. |
| `CMAKE_OSX_DEPLOYMENT_TARGET` | `14.0` | Sonoma is the baseline. |
| `CMAKE_OSX_ARCHITECTURES` | `arm64` | Forced. `x86_64` warns. |
| `cmake_minimum_required` | `3.30` | Set in the root CMakeLists. |
| `CMAKE_INSTALL_RPATH` | `@executable_path/../lib;@loader_path/../lib` | Installed binaries find their dylibs by `@executable_path`. |
| `ALICEVISION_ROTATION_AVERAGING_WITH_BOOST` | defined globally | Forces Boost.Graph instead of LEMON for rotation averaging (see PORTING_NOTES §9). |

## Where the options live

`CMakeLists.txt` lines 49-69:

```cmake
option(AV_USE_METAL              "Enable Metal backend"                       ON)
option(AV_USE_CUDA               "Enable CUDA (disabled on Apple)"            OFF)
option(AV_BUILD_TESTS            "Build unit + integration tests"             ON)
option(AV_BUILD_HELLO_METAL      "Build the Metal hello-kernel smoke test"    ON)
option(AV_BUILD_UPSTREAM         "Configure upstream/AliceVision (separate)"  OFF)
option(AV_BUILD_UPSTREAM_DEPTHMAP "Build the upstream depthMap-only dep tree" OFF)
option(AV_USE_HOMEBREW_DEPS      "Resolve dependencies via Homebrew prefix"   ON)
option(AV_PROFILE_ADAPTER        "Profile cuda_* adapter forwarders (S43)"    OFF)
```
