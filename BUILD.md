# BUILD.md — alicevision-for-mac

How to build the Apple Silicon port of AliceVision from source.

This repository is an out-of-tree overlay: the `upstream/` directory points at
a read-only reference clone of AliceVision (Windows/Linux/CUDA source), and
this tree adds the Metal backend, the `av::gpu` abstraction layer, and the
macOS-native build glue on top.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the codebase tour and
[PORTING_NOTES.md](PORTING_NOTES.md) for the CUDA → Metal decisions.

---

## 1. Prerequisites

### System

- **macOS 14.0+** (Sonoma or newer). The root `CMakeLists.txt` pins
  `CMAKE_OSX_DEPLOYMENT_TARGET = 14.0` and `find_package` will succeed only
  against SDKs that ship with that floor.
- **Xcode 26+** with the Metal toolchain. Verify with
  `xcrun metal --version` and `xcrun metallib --version`. The
  `cmake/Metal.cmake` module shells out to both.
- **Apple Silicon arm64**. The root `CMakeLists.txt` forces
  `CMAKE_OSX_ARCHITECTURES = arm64` and warns on `x86_64`. Universal
  binaries are explicitly out of scope for milestone 1.
- **CMake ≥ 3.30** (set by `cmake_minimum_required` at the top of
  `CMakeLists.txt`).
- **Homebrew**. `AV_USE_HOMEBREW_DEPS=ON` (default) shells out to
  `brew --prefix` and prepends it to `CMAKE_PREFIX_PATH` so the `find_package`
  calls below resolve from `/opt/homebrew` (Apple Silicon) without extra
  flags.

### Homebrew packages (minimum required)

The default build (no `AV_BUILD_UPSTREAM_DEPTHMAP`) needs only **Eigen3**.
Everything else — `metal-cpp`, `MTL::Device`, the Metal toolchain — comes
from the Xcode SDK or the vendored `third_party/metal-cpp/`.

```bash
brew install cmake ninja eigen
```

If you intend to enable `-DAV_BUILD_UPSTREAM_DEPTHMAP=ON` (see §5) the
upstream depthMap dependency tree pulls in:

```bash
brew install \
  boost ceres-solver openimageio openexr imath zlib \
  libomp pkg-config
```

(The full Phase-0 install list, including packages reserved for future
phases such as Alembic / Geogram / nanoflann / TBB / OpenMesh, lives in
`memory/todo.md` under "Phase 1".)

---

## 2. One-shot build

From the project root:

```bash
cmake -S . -B build
cmake --build build
```

That configures with the defaults below (Metal backend ON, CUDA OFF,
tests ON, hello-Metal smoke test ON, upstream tree OFF), compiles the
`av::gpu` and `av::depth_map_metal` static libraries, builds the
`default.metallib` from `src/shaders/depth_map/*.metal` via
`xcrun metal`/`metallib`, and links the test executables.

To use Ninja instead of the default Xcode/Makefiles generator:

```bash
cmake -G Ninja -S . -B build
cmake --build build
```

Build artifacts land under `build/`:

- `build/src/av_gpu/libav_gpu.a` — the Metal abstraction layer.
- `build/src/depth_map_metal/libav_depth_map_metal.a` — the
  AliceVision-shaped Metal port.
- `build/src/shaders/default.metallib` — staged next to every test exe by
  `av_install_metallib` (`cmake/Metal.cmake`).
- `build/tests/test_*` — the 33 test executables.

---

## 3. Running tests

The build tree is the working directory for ctest:

```bash
cd build
ctest
```

Expected result on a clean tree at HEAD (verified 2026-05-19):

```
100% tests passed, 0 tests failed out of 33
```

Verbose output for a specific test:

```bash
ctest -R test_depth_pipeline --output-on-failure -V
```

Individual binaries can also be invoked directly:

```bash
./build/tests/test_metal_hello
./build/tests/test_depth_pipeline
```

The full test inventory is in `tests/CMakeLists.txt`. Notable
end-to-end tests:

| Test                          | What it validates                                    |
| ----------------------------- | ---------------------------------------------------- |
| `test_metal_hello`            | metal-cpp wiring + SAXPY/SIMD-reduction smoke.       |
| `test_texture_smoke`          | RAII textures, bilinear sampling, mipmap cascade.    |
| `test_sgm_pipeline`           | `init_sim → compute_similarity → optimize → retrieve_best_depth` on a synthetic scene. |
| `test_refine_pipeline`        | `init_refine → refine_similarity → refine_best_depth`. |
| `test_depth_pipeline`         | The full SGM → Bridge → Refine → Optimize chain.     |
| `test_multi_t_aggregation`    | WTA + FP16 additive aggregation across multiple T cameras. |
| `test_device_mipmap_image`    | `DeviceMipmapImage` end-to-end (download + Lab + mipmap). |
| `test_device_cache`           | `LRUCache<T>` invariants + `DeviceCache` eviction.   |
| `test_device_stream_manager`  | Multi-`MTLCommandQueue` parallel dispatch.           |

---

## 4. Build options

All options live near the top of the root `CMakeLists.txt`. Flip with
`-D<NAME>=ON|OFF` on the `cmake -S . -B build` line.

| Option                       | Default | When to flip                                              |
| ---------------------------- | ------- | --------------------------------------------------------- |
| `AV_USE_METAL`               | `ON`    | Always ON on Apple. The whole point of the port.          |
| `AV_USE_CUDA`                | `OFF`   | Never flip ON — `CMakeLists.txt` aborts if you do. CUDA is not available on macOS. |
| `AV_BUILD_TESTS`             | `ON`    | Set OFF for distribution/CI builds that don't run tests. |
| `AV_BUILD_HELLO_METAL`       | `ON`    | OFF to skip the two smoke tests (`test_metal_hello`, `test_texture_smoke`). |
| `AV_BUILD_UPSTREAM`          | `OFF`   | Reserved for Phase 2 (full upstream AliceVision). Not currently active. |
| `AV_BUILD_UPSTREAM_DEPTHMAP` | `OFF`   | ON to build the 12-module upstream subset depthMap depends on. See §5. |
| `AV_USE_HOMEBREW_DEPS`       | `ON`    | OFF only if you've installed dependencies outside Homebrew and pre-populated `CMAKE_PREFIX_PATH`. |

Additional CMake-level toggles you may want:

- `-DCMAKE_BUILD_TYPE=Release` (default is `RelWithDebInfo`).
- `-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0` to raise the minimum macOS.
- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` for clangd/IDE integration.

---

## 5. Optional: upstream depthMap dependency tree

`-DAV_BUILD_UPSTREAM_DEPTHMAP=ON` builds the 12 upstream AliceVision
modules that depthMap's host-side code depends on (`system`, `numeric`,
`image`, `stl`, `linearProgramming`, `geometry`, `camera`,
`robustEstimation`, `multiview`, `mvsData`, `mvsUtils`, `gpu`). This is
"Path C" of the build strategy (`memory/todo.md` Phase 2). Useful when
you're about to wire `Sgm.cpp` / `Refine.cpp` / `DepthMapEstimator.cpp`
against our `av::depth_map::*` types — not needed for kernel work in
isolation.

```bash
cmake -S . -B build -DAV_BUILD_UPSTREAM_DEPTHMAP=ON
cmake --build build
```

The upstream tree itself is untouched. `cmake/UpstreamShim.cmake`
provides shim implementations of upstream's `alicevision_add_library`,
`alicevision_add_test`, `alicevision_add_interface`, and
`alicevision_swig_add_library` macros so the per-module `CMakeLists.txt`
files compose without pulling in upstream's install rules, SOVERSION
dance, or Windows .rc generation.

Several upstream quirks are accommodated at CMake time without editing
`upstream/`:

- `Boost::system` is stubbed as an `INTERFACE IMPORTED` target (modern
  Boost made it header-only and Homebrew dropped the separately-packaged
  CMake config).
- `Coin::Clp`, `Coin::CoinUtils`, `Coin::Osi` are stubbed as empty
  interface targets (Coin-OR is not on Homebrew; the `linearProgramming`
  module is INTERFACE-only and depthMap does not solve LPs anyway).
- `ALICEVISION_ROTATION_AVERAGING_WITH_BOOST` is defined globally so
  `multiview/rotationAveraging` uses Boost.Graph instead of LEMON
  (Homebrew's `lemon` is the parser generator, not the COIN-OR graph
  library).
- `multiview/rotationAveraging/l1.cpp` is patched at configure time
  via `file(READ/REPLACE/WRITE)` to drop `const` from three
  default-initialized `Eigen::Matrix` declarations (clang 21 enforces
  the [dcl.init] rule that const non-user-defined-default-ctor class
  objects must have an initializer). The patched copy lands in
  `build/upstream-patched/`; the original on disk is untouched.

See [PORTING_NOTES.md §10](PORTING_NOTES.md) for the rationale.

---

## 6. Troubleshooting

### "Boost.System missing" or "Could not find Boost::system"

Modern Boost (≥1.86) made `Boost.System` header-only and Homebrew
stopped installing the separate `boost_system-config.cmake`. The root
`CMakeLists.txt` already handles this for the `AV_BUILD_UPSTREAM_DEPTHMAP=ON`
branch by **stubbing** `Boost::system` as an interface target that
forwards to `Boost::headers`. If you see this error against another
target, replicate that pattern:

```cmake
if(NOT TARGET Boost::system)
    add_library(Boost::system INTERFACE IMPORTED)
    if(TARGET Boost::headers)
        set_target_properties(Boost::system PROPERTIES
            INTERFACE_LINK_LIBRARIES Boost::headers)
    endif()
endif()
```

### "Coin::Clp not found" or "Coin::CoinUtils not found"

Coin-OR is not on Homebrew. The default build does not need it; the
`AV_BUILD_UPSTREAM_DEPTHMAP=ON` build stubs `Coin::Clp`,
`Coin::CoinUtils`, and `Coin::Osi` as empty interface targets because
the upstream `linearProgramming` module is INTERFACE-only and depthMap
does not solve LP problems at runtime. If you're building a different
upstream module that *does* link Coin-OR symbolically, you'll see a
clean link error and know to either port that call site or build
Coin-OR from source (Path A in `memory/todo.md`).

### Metal library fails to load at test runtime

The `cmake/Metal.cmake` module compiles `.metal` → `.air` → `default.metallib`,
and `av_install_metallib` copies it next to every test executable
(`build/tests/default.metallib`). If a test reports
`Failed to load default.metallib`, check:

1. `build/src/shaders/default.metallib` exists.
2. `build/tests/default.metallib` exists (or is a symlink).
3. `xcrun metallib --version` works without prompting for licence
   acceptance (`sudo xcodebuild -license accept`).

For ad-hoc debugging,
`MTL::Device::default_device().load_library("/explicit/path/default.metallib")`
takes an absolute path.

---

## 7. Clean rebuild

```bash
rm -rf build
cmake -S . -B build
cmake --build build
cd build && ctest
```

There are no generated files outside `build/`. The `upstream/` directory
is treated as read-only reference; even the `l1.cpp` patch lands in
`build/upstream-patched/`, not in source.

---

Related: [ARCHITECTURE.md](ARCHITECTURE.md) · [PORTING_NOTES.md](PORTING_NOTES.md)
