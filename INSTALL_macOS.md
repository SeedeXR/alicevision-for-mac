# INSTALL_macOS.md — alicevision-for-mac

User-facing install guide for the **Apple Silicon port** of AliceVision.

This document is for early adopters and contributors who want to install
and run the macOS port. If you are looking to build the project from
source, see [BUILD.md](BUILD.md) instead. For the CUDA → Metal design
decisions see [PORTING_NOTES.md](PORTING_NOTES.md); for a code tour see
[ARCHITECTURE.md](ARCHITECTURE.md).

> **Honest status (2026-05-20).** This port is **work-in-progress**.
> The depthMap Metal kernels are complete and validated end-to-end
> (`ctest`: 33/33), but the upstream `aliceVision_depthMapEstimation`
> command-line binary is **not yet runnable** on macOS — the host-side
> adapter that wires our `av::depth_map` kernels into upstream's
> `Sgm.cpp` / `Refine.cpp` / `DepthMapEstimator.cpp` is Phase 8 work in
> progress (see [memory/todo.md](memory/todo.md) Phase 8 and
> [memory/phase8_adapter_map.md](memory/phase8_adapter_map.md)). What
> you *can* do today is build the tree from source and run the 33
> validation tests. What you *cannot* do today is point the binary at a
> folder of photographs and get a mesh out. The "Known limitations"
> section below is the canonical list of what is and is not finished.

---

## 1. What's new on macOS

Upstream's macOS install guide (`upstream/INSTALL_macOS.md`) is explicit
that "the DepthMap library from AliceVision (and therefore all its
dependants) are unavailable due to them being implemented only in
CUDA". This port lifts that restriction:

- **depthMap is available on Apple Silicon via Metal.** The CUDA
  kernels under `upstream/src/aliceVision/depthMap/cuda/` have been
  re-implemented as MSL (Metal Shading Language) kernels in
  `src/shaders/depth_map/*.metal`, orchestrated by the `av::gpu` and
  `av::depth_map` host layers in `src/`. **35 distinct kernel entry
  points** across 15 `.metal` files cover the full pipeline.
  ([ARCHITECTURE.md §5](ARCHITECTURE.md))
- **SGM core** — `init_sim → compute_similarity → optimize →
  retrieve_best_depth` ported and validated end-to-end on a synthetic
  plane-induced-homography scene
  ([ARCHITECTURE.md §1](ARCHITECTURE.md)).
- **Refine pass** — `init_refine → refine_similarity →
  refine_best_depth` ported with the FP16 cost-volume + sub-pixel
  depth movement, validated end-to-end on the same scene
  ([ARCHITECTURE.md §1](ARCHITECTURE.md)).
- **Optimize stage** — `optimize_depth_sim_map` gradient-descent
  fusion of the SGM rough + Refine fine maps via chained-sigmoid
  blend ([PORTING_NOTES.md §5](PORTING_NOTES.md)).
- **Multi-T-camera aggregation** — both WTA (cost-similarity) and
  FP16-additive (refine-similarity) patterns validated on a
  3-T-camera plane scene
  ([ARCHITECTURE.md §1](ARCHITECTURE.md)).

Every CUDA → Metal architectural decision (FP64 → FP32, Lab × 2.55
preservation, `-ffast-math` sigmoid drift, multi-mip texture
workarounds, etc.) is documented case-by-case in
[PORTING_NOTES.md](PORTING_NOTES.md).

---

## 2. System requirements

| Requirement | Value | Why |
| --- | --- | --- |
| macOS | 14.0 (Sonoma) or newer | `CMAKE_OSX_DEPLOYMENT_TARGET = 14.0` is pinned in the root `CMakeLists.txt`. |
| CPU | Apple Silicon (M1 / M2 / M3 / M4) | `CMAKE_OSX_ARCHITECTURES = arm64` is forced; the build warns on `x86_64`. |
| Xcode | 26 or newer (clang 21, Metal toolchain) | `cmake/Metal.cmake` shells out to `xcrun metal` and `xcrun metallib`. |
| CMake | ≥ 3.30 | Set by the root `cmake_minimum_required`. |
| Homebrew | Required (default build) | `AV_USE_HOMEBREW_DEPS=ON` prepends `brew --prefix` to `CMAKE_PREFIX_PATH`. |

**Out of scope for milestone 1:**

- **Intel (x86_64) Macs.** Universal binaries are explicitly disabled.
- **Rosetta.**
- **Older macOS than 14.0.**

See [BUILD.md §1](BUILD.md) for the verification commands (`xcrun metal
--version`, etc.).

---

## 3. Install via Homebrew

`[TODO: Phase 12]` — A Homebrew formula is **not yet available**.
Packaging, code-signing (ad-hoc for dev / Developer ID for
distribution), notarization, and the `brew tap` are all listed under
[memory/todo.md](memory/todo.md) **Phase 12** ("Packaging &
distribution") and have not been started.

For now, build from source (next section).

---

## 4. Install from source

The short version:

```bash
git clone <this-repo> alicevision-for-mac
cd alicevision-for-mac
brew install cmake ninja eigen
cmake -G Ninja -S . -B build
cmake --build build
cd build && ctest
```

Expected result: `100% tests passed, 0 tests failed out of 33`.

For prerequisites, build options (`AV_BUILD_UPSTREAM_DEPTHMAP=ON` etc.),
the full troubleshooting list, and a clean-rebuild recipe see
[BUILD.md](BUILD.md).

---

## 5. Running the depthMap step

`[TODO: Phase 8 + Phase 3]` — **There is currently no end-user CLI to
run depthMap on a folder of photographs.** Two things are missing:

1. **Phase 8 adapter shim.** Upstream's `aliceVision_depthMapEstimation`
   binary is implemented in `upstream/src/software/pipeline/main_depthMapEstimation.cpp`
   and links against the upstream `aliceVision_depthMap` library, which
   in turn references the CUDA host classes in
   `upstream/src/aliceVision/depthMap/cuda/host/`. Our Metal kernels
   live behind a parallel `av::depth_map::*` API; the **12 unique
   `cuda_*` symbols** that bridge upstream's host code to the GPU need
   forwarding shims. The full adapter map (every symbol, every call
   site, type-translation rules) is in
   [memory/phase8_adapter_map.md](memory/phase8_adapter_map.md);
   progress is tracked under [memory/todo.md](memory/todo.md) Phase 8.
2. **Phase 3 unified CLI.** The project's stated milestone is a single
   `aliceVision` binary with subcommand dispatch (with symlinks for
   the legacy per-subcommand names). That work has not started — see
   [memory/todo.md](memory/todo.md) Phase 3.

**What you can do today** is run the validation tests under `build/tests/`.
The most illustrative end-to-end test is:

```bash
./build/tests/test_depth_pipeline
```

which wires the full SGM → Bridge → Refine → Optimize chain on a
synthetic 128×96 plane-induced-homography scene. The test inventory
and what each test validates is in [BUILD.md §3](BUILD.md).

**Once Phase 8 lands**, the canonical invocation will mirror upstream's
existing command (preserved verbatim because we are not changing the
on-disk CLI surface in that phase):

```bash
# [TODO: Phase 8] — does not yet work on macOS
aliceVision_depthMapEstimation \
  --input <sfm.abc> \
  --output <depth_dir> \
  --imagesFolder <images_dir>
```

The flag set is identical to the upstream CUDA build (`--help` is
defined in `main_depthMapEstimation.cpp`); we do not invent new flags.

---

## 6. Meshroom integration

`[TODO: Phase 11]` — Source-tree fixes for Meshroom on macOS exist as
**git patches**, not yet merged upstream and not yet validated
end-to-end. See [`patches/meshroom/README.md`](patches/meshroom/README.md)
for the apply procedure. The four patches address:

- **`01-init-darwin-libpath.patch`** — adds a Darwin branch to
  `meshroom/__init__.py`'s `setupEnvironment()` setting
  `DYLD_FALLBACK_LIBRARY_PATH` from `ALICEVISION_LIBPATH`.
- **`02-stats-darwin-gpu.patch`** — replaces the `nvidia-smi` GPU
  stats path with `system_profiler SPDisplaysDataType`.
- **`03-cgroup-darwin-sysctl.patch`** — short-circuits the Linux
  `/proc/cgroup` probes with `sysctl hw.memsize` / `sysctl hw.ncpu`.
- **`04-startsh-readlink-portable.patch`** — replaces GNU-only
  `readlink -f` in `start.sh` with a portable `python3 -c
  'os.path.realpath(...)'`.

All four pass `git apply --check` against the pinned upstream commit
(`0ab90c0b`); patches 1, 3, and 4 are flagged as **ready for upstream
PR**, patch 2 needs maintainer review (see the README's "Upstream-PR
readiness" table).

End-to-end Meshroom run is **gated on the Phase 8 adapter** — Meshroom
will only render a graph if the underlying `aliceVision_*` binaries
actually produce output, which today they do not on macOS.

---

## 7. Known limitations

What does **not** work on this port today:

| Limitation | Phase / tracking |
| --- | --- |
| Structure-from-motion (SfM) — only depthMap kernels are ported | not in current milestone scope |
| No GUI — Meshroom integration is patches-only and not validated end-to-end | Phase 11 ([memory/todo.md](memory/todo.md)) |
| No unified `aliceVision` CLI binary | Phase 3 ([memory/todo.md](memory/todo.md)) |
| `aliceVision_depthMapEstimation` (and dependants) not yet linkable / runnable | Phase 8 adapter shim ([memory/phase8_adapter_map.md](memory/phase8_adapter_map.md)) |
| No Homebrew formula, no code-signed bundle, no notarization | Phase 12 ([memory/todo.md](memory/todo.md)) |
| Image loading / SfMData I/O not yet wired into our `DeviceMipmapImage` | [ARCHITECTURE.md §1](ARCHITECTURE.md) "Stages still TODO" |
| Mesh reconstruction (meshing, texturing, etc.) — none of it yet | Phase 11+ |
| `x86_64` Mac, universal binaries, Rosetta — explicitly out of scope for milestone 1 | not planned |
| `deviceMipmappedArray.cu` custom mipmap cascade — cosmetic, deferred | Phase 6 (~95% done) |

**Reference of record.** The CUDA build on Linux / Windows remains the
parity reference for output. When the Phase 8 adapter lands, we will
compare per-pixel depth maps from the Metal pipeline against the CUDA
pipeline on the same input scene to bound the numerical drift.

---

## 8. Troubleshooting (runtime / install)

For **build-time** issues (CMake configure errors, missing Boost,
metallib failing to compile, etc.) see [BUILD.md §6](BUILD.md). The
items below cover **runtime** issues an installer will hit after a
successful build.

### "`xcrun metal` reports licence not accepted"

The Metal toolchain refuses to compile until the Xcode licence is
accepted:

```bash
sudo xcodebuild -license accept
```

### `default.metallib` not found at runtime

Every test executable expects `default.metallib` next to itself
(staged there by `av_install_metallib` in `cmake/Metal.cmake`). If a
binary reports `Failed to load default.metallib`:

1. Verify `build/src/shaders/default.metallib` exists.
2. Verify `build/tests/default.metallib` exists (it is copied from
   the source path by the install rule).
3. For ad-hoc debugging,
   `MTL::Device::default_device().load_library("/absolute/path/default.metallib")`
   accepts an absolute path.

See [BUILD.md §6](BUILD.md) for the full diagnostic.

### Slow first launch, "XprotectService" in Activity Monitor

This is the standard Gatekeeper / `dyld` malware scan on first launch
of any unsigned Mach-O. Upstream documents the same behaviour. Wait
it out — subsequent launches are near-instant. Once Phase 12 ships
code-signed and notarized binaries, this delay disappears.

### `dyld: Library not loaded: ...` after moving the build directory

Our build uses `@rpath` install names (mirroring upstream's
`ALICEVISION_USE_RPATH=ON` default). The `metallib` and the test
binaries assume their relative layout under `build/`. If you move
artefacts out of the build tree, either re-resolve rpaths with
`install_name_tool` or run from inside `build/`.

### Apple-Silicon-only — the binary won't run on Intel

By design. `CMAKE_OSX_ARCHITECTURES = arm64` is forced; the build
emits a warning on `x86_64`. There is no Rosetta path because the
Metal kernels target `apple-m1` and later. See "Known limitations"
above.

### "I want to file a bug"

The port is pre-release. Open an issue against this repository (not
upstream `alicevision/AliceVision`) with:

- macOS version (`sw_vers`).
- Apple Silicon generation (`sysctl -n machdep.cpu.brand_string`).
- Xcode version (`xcodebuild -version`).
- The full output of `ctest --output-on-failure` from the `build/`
  directory.

---

Related: [BUILD.md](BUILD.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
[PORTING_NOTES.md](PORTING_NOTES.md) ·
[patches/meshroom/README.md](patches/meshroom/README.md) ·
[memory/todo.md](memory/todo.md) ·
[memory/phase8_adapter_map.md](memory/phase8_adapter_map.md)
