# Mental notes (dev book)

Cleaned-up version of `memory/mental_note.md` — provisional learnings from
sessions S38-S47. Each entry surfaced from a concrete bug or design
decision; entries that have been verified across sessions are promoted into
the architecture / porting-notes docs.

These notes are deliberately terse. If you're new to the codebase, read
[Project overview](../dev/overview.md) first.

## libc++ vs libstdc++ — `vector<T>::reserve` instantiation

`std::vector<T>::reserve(N)` under libc++ eagerly instantiates the
relocation template chain (`__swap_out_circular_buffer` →
`move_if_noexcept` → `__construct_at`) at compile time, regardless of `N`.
If `T` is neither copyable nor noexcept-movable, the call fails to compile
**even for `reserve(0)`**. libstdc++ is lazier.

Concrete trigger (S38): `DepthMapEstimator.cpp:281` —
`refinePerStream.reserve(_depthMapParams.useRefine ? nbStreams : 0)` where
`Refine` has `~Refine() = default;` explicitly, which suppresses the
implicit move constructor. Combined with non-copyable
`CudaDeviceMemoryPitched<>` members, the class is neither movable nor
copyable.

Fix: CMake-time patch adds `Refine(Refine&&) noexcept = default;` (same for
`Sgm`). Reference members make the move assignment ill-formed, but the
move ctor itself is well-formed.

## `~T() = default;` is NOT semantically equivalent to "no destructor"

`= default;` still counts as a user-declared special member for purposes
of the "no user-declared X" rule ([class.copy.ctor]/8). Result: a class
with `~T() = default;` and no other explicit special-member declarations
has **no implicit move constructor**.

Pattern to apply when porting libstdc++ code to libc++: audit
`~T() = default;` declarations; add `T(T&&) noexcept = default;` if `T`
is used in any `vector<T>` operation.

## CUDA host-API surface needed to compile upstream depthMap host code

| CUDA symbol | Shim approach |
|---|---|
| `float2`, `float3`, `float4`, `uchar4` | POD structs in `memory.hpp` shim |
| `make_float2/3/4`, `make_uchar4` | Trivial inline factories |
| `cudaStream_t` | `typedef void*` |
| `cudaError_t` | `typedef int` |
| `cudaDeviceSynchronize()` | No-op returning 0 (we already `commit_and_wait`) |
| `__half` (cuda_fp16.h) | `typedef _Float16 __half;` on arm64 Clang (S38 §7b mangling fix) |
| `__float2half`, `__half2float` | Inline using `_Float16` conversion |
| `__constant__` | Drop the declaration (shim the header that uses it) |
| `cudaMallocHost`, `cudaFreeHost` | Handled in `DeviceCache` shim (UMA → plain malloc) |
| `cudaMemcpyToSymbol` | Handled in patchPattern shim (pass via `set_bytes`) |

## Header-only shim path must precede `upstream/src` in `include_directories`

So that `#include <aliceVision/depthMap/cuda/host/memory.hpp>` resolves to
our shim, not upstream's CUDA-heavy version, and likewise for
`DeviceCameraParams.hpp`.

Also `upstream-patched/` must precede `upstream/src/` so the patched
`Sgm.hpp` / `Refine.hpp` win.

## UMA changes the meaning of `cudaDeviceSynchronize`

On a discrete GPU, `cudaDeviceSynchronize()` blocks the host until device
work completes. On Apple Silicon UMA with `commit_and_wait`-per-dispatch,
every adapter call already drained the GPU before returning. The
synchronize becomes a no-op.

If/when we introduce async dispatch (Phase 14 perf), this assumption
becomes load-bearing — re-audit.

## Shim-discovery heuristic for upstream `.hpp`/`.cpp` pairs

When upstream's host code includes a CUDA-using header:

1. **Header-only?** Shim by writing a replacement header at
   `cmake/shims/aliceVision-includes/<same-path>`. Strip CUDA-specific
   declarations (`__constant__`, kernel decls); keep value types.
2. **Header + .cpp pair?** Shim both. Provide a replacement `.cpp` under
   `src/depth_map_metal/src/cuda_host_shim/` that wraps our native
   `av::depth_map::*` types. The original upstream `.cpp` is NOT compiled.

S38 had four such pairs (`DeviceCache`, `DeviceStreamManager`,
`patchPattern`, `utils`) and one header-only (`DeviceCameraParams`).

## `_Float16` vs `__fp16` mangling divergence

Apple Clang treats `_Float16` (C++23) and `__fp16` (ARM ACLE storage-only)
as distinct types with distinct mangled names: `_Float16` → `DF16_`,
`__fp16` → `Dh` (demangles to "half").

Concrete bite: adapter header uses `using TSimRefine = _Float16;` while
the `cuda_fp16.h` shim originally said `typedef __fp16 __half;`. Upstream's
`Refine.cpp` compiled against `__half = __fp16` generates external refs to
`cuda_volumeRefineSimilarity(..., half, ...)`. The adapter, compiled against
`_Float16`, exports the symbol with `DF16_` mangling. Link error.

Fix: change the shim to `typedef _Float16 __half;`. Both sides now use the
same type.

Pattern: when a shim aliases a CUDA type onto a host equivalent, audit
that host type against any sibling adapter's declared type. Mangling
divergence is silent at compile time and surfaces only at link.

## ALICEVISION_ROOT layout

`<root>/share/aliceVision/` contains `config.ocio` + `luts/` +
`cameraSensors.db`. **NOT a path to the source tree.** It's a Unix-like
install prefix.

Either:

1. Set `ALICEVISION_ROOT` to a directory with that structure.
2. Or don't set it — binary falls back to an embedded path. Works for
   simple ops but pipeline ops that re-resolve OCIO later in the run
   still crash.

Phase 12 packaging populates this properly.

## Auto-init for adapter-device + metallib

Pipeline binaries don't call `set_adapter_device()` or `load_library()`
explicitly. Solution: lazy function-local-static init inside
`require_adapter_device()`:

```cpp
static av::gpu::Device s_default = []() {
    auto d = av::gpu::Device::default_device();
    d.load_library({});  // looks for default.metallib at @executable_path
    return d;
}();
```

So `default.metallib` must be staged next to every pipeline binary via
`av_install_metallib(FROM av_shaders EXECUTABLE <bin>)`.

## UMA "Device memory available" reporting

`getDeviceMemoryInfo` reports system RAM via Mach `host_statistics64`. On
real M4 with 16 GB the binary correctly reports
`16384.0 MB total, 4280 MB available, 12100 MB used`. The "used" number is
system-wide (NOT this process's allocations).

For pipeline scheduling that's the right metric — AliceVision uses
"available" to decide tile count + simultaneous-depth-map parallelism.

## Middlebury cameras-only ≠ usable depth output

`aliceVision_importMiddlebury` imports cameras + poses but NOT landmarks
(it can't — Middlebury `.par` files have no 3D structure).
`depthMapEstimation` needs landmarks to select T-camera neighbours by
geometric + observation co-visibility. Without landmarks: "0/10 nearest
cameras" → all-`(-1)` / all-`(-2)` depth maps.

A real end-to-end test on Middlebury data requires either the full SfM
cascade or a `DepthMapEstimator::computeTCamsList` patch that falls back
to geometric nearest-camera selection.

## Two-bug cascade in `cuda_volumeRetrieveBestDepth` (S40)

Documented in detail in the [Adapter pattern](../dev/adapter.md) page —
the canonical example of parameter-scaling mismatch:

1. **`maxSimilarity` not scaled by 254** — uchar range `[0, 254]` rejected
   nearly every valid voxel under default `sgmParams.maxSimilarity = 1.0`.
2. **`thicknessMultFactor` missing `1.f +`** — default
   `depthThicknessInflate = 0` zeroed thickness → pixSize 0 → NaN cascade
   → blank EXR output.

Result after fix: Monstree mini3 produces real depth maps with
Min=-2, Max≈20-22, Avg≈3-4, on the SfM-reported depth range.

Pattern: every parameter passed to a Metal kernel that mirrors a CUDA call
site must mirror the EXACT pre-processing the CUDA caller does. Audit
each parameter against `grep -A 3 'cuda_*' upstream/.../*.cu` before
declaring the adapter complete.

## LEMON graph library — vendored, not Homebrew

`aliceVision_track` needs `lemon::ListDigraph` + `lemon::UnionFindEnum`
with no fallback. Homebrew's `lemon` package is the SQLite
parser-generator (unrelated). The COIN-OR LEMON 1.x graph library is not
on Homebrew at all.

We vendor LEMON 1.3.1 under `third_party/lemon/`. Build via a small
`CMakeLists` that compiles only `base.cc`, `random.cc`, `color.cc`,
`arg_parser.cc` (skip LP backends).

C++17 patches against Apple Clang 21:

- `lemon/random.h`: drop `register` keyword.
- `lemon/bits/array_map.h`: replace `allocator.construct(p, v)` /
  `allocator.destroy(p)` with `std::allocator_traits<…>::construct` /
  `std::allocator_traits<…>::destroy` (removed from `std::allocator` in
  C++20).

## `lInfinityCV` depends on Coin-OR Clp — shimmed out

`aliceVision_lInftyComputerVision` instantiates `OSI_CISolverWrapper`
requiring Coin-OR Clp headers we don't ship.

Resolution: shim header at
`cmake/shims/aliceVision-includes/aliceVision/linearProgramming/OSIXSolver.hpp`
that:

1. Forward-declares `OsiClpSolverInterface` (never defined; never
   dereferenced because we provide inline stub bodies).
2. Provides `OSIXSolver<T>` with inline no-op bodies
   (`setup() → true`, `solve() → true`, `getSolution() → all zeros`).

`solve() → true` means `geometry::halfPlane::isNotEmpty()` always reports
"intersection found". This is a safe over-approximation: extra image
pairs are kept for matching; downstream RANSAC filters by inlier count.

## CMake target-name collision: `aliceVision_imageMatching` lib vs exe

Both upstream's library and the executable declare target name
`aliceVision_imageMatching`. CMake forbids this.

Resolution: name the executable target `aliceVision_imageMatching_bin` and
use `set_target_properties(... OUTPUT_NAME aliceVision_imageMatching)` so
the on-disk filename matches Meshroom's expectation.

## `ALICEVISION_BUILD_LIDAR` inverted check

`dataio/CMakeLists.txt` builds `E57Reader.cpp` when
`NOT ALICEVISION_BUILD_LIDAR STREQUAL "OFF"`. If the variable is unset →
TRUE → E57Reader → looks for `<E57SimpleData.h>` (not packaged).

Must explicitly `set(ALICEVISION_BUILD_LIDAR OFF)` before
`add_subdirectory(dataio)`.

## Meshing / texturing flag gotchas

| Flag | Gotcha |
|---|---|
| `aliceVision_meshing --output foo.abc` | Fatals: *"AliceVision is built without Alembic support"*. We don't link Alembic. Use `.sfm` (dense SfMData JSON) or `.ply`. |
| `aliceVision_texturing --colorMappingFileType=none` (default) | Skips actual texture-atlas baking. Pass `png` / `jpg` / `tif` / `exr` to bake. |
| `aliceVision_texturing -i foo.ply` | Wrong — input must be the dense `.sfm` (`Found 0 image dimension(s)` warning otherwise). |

## Defensive allocate in adapter forwarders

Upstream's `Sgm.cpp:58` does
`if (_computeDepthSimMap) _depthSimMap_dmp.allocate(mapDim);` but still
passes the buffer unconditionally to `cuda_volumeRetrieveBestDepth(...)`.
CUDA tolerates writes to a null device pointer; our Metal shim
dereferences a null `unique_ptr` → silent UB → hang.

Fix: every adapter forwarder that takes a non-const
`CudaDeviceMemoryPitched&` must check
`out.getBytesPadded() == 0` and lazy-allocate.

## Lines of upstream code we don't compile

After S38, total upstream depthMap LOC compiled = 9 host `.cpp`s. Upstream
CUDA source LOC we replaced: all of `cuda/device/*.cu`,
`cuda/imageProcessing/*.cu`, `cuda/planeSweeping/*.cu`, plus 4
`cuda/host/*.cpp` we shimmed. Rough delta: **~6000 LOC of CUDA C++/.cu →
~2000 LOC of MSL + adapter forwarders + shims**.

The right ratio: CUDA-host-side glue is high-overhead per kernel; Metal's
binding model is leaner.

---

For session-specific decisions not promoted here see the raw
`memory/mental_note.md` and the per-session perf docs
(`memory/perf_profile_s43.md`, `memory/perf_optimization_s44.md`,
`memory/perf_optimization_s45.md`).
