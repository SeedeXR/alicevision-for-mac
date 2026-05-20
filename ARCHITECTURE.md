# ARCHITECTURE.md — alicevision-for-mac

A reader-friendly tour of the codebase. Read [BUILD.md](BUILD.md) first if
you want to actually build it; read [PORTING_NOTES.md](PORTING_NOTES.md) for
the CUDA → Metal decision log.

The port is an **out-of-tree overlay**: the upstream AliceVision source
lives under `upstream/` as a read-only reference clone (the Windows/Linux
CUDA tree), and everything in `src/` is new Apple Silicon code that
overlays on top. The Metal kernels and host orchestration that
collectively replace `upstream/src/aliceVision/depthMap/cuda/` live in
two layers: a generic `av::gpu` Metal abstraction, and a depthMap-shaped
`av::depth_map` layer that consumes it.

---

## 1. High-level pipeline

The end goal is the AliceVision photogrammetry pipeline:

```
   SfMData (cameras + sparse points)
              │
              ▼
       image loading                ◄── upstream aliceVision_image (built S30)
              │
              ▼
   ┌──────────────────────┐
   │  depthMap pipeline   │         ◄── this is what we port to Metal
   │                      │
   │   SGM                │           Phase 7: kernels ported (Session 10-14)
   │     ↓                │           Phase 7: end-to-end validated (S14, S15)
   │   Bridge             │           Phase 7: ported (S21)
   │     ↓                │
   │   Refine             │           Phase 7: ported (S16-S18)
   │     ↓                │
   │   Optimize           │           Phase 7: ported (S23, S24)
   └──────────┬───────────┘
              ▼
       per-view depth maps           ◄── 33/33 tests pass, including
              │                          end-to-end accuracy (S15, S24)
              ▼
       mesh reconstruction           ◄── TODO (Phase 11+)
              │
              ▼
            output mesh
```

Stages **ported and end-to-end-validated** on Apple Silicon as of
2026-05-19 (`memory/session_start.md` "Latest milestone"):

- `init_sim → compute_similarity → optimize → retrieve_best_depth` (SGM core).
- `compute_sgm_upscaled_depth_pix_size_map` (SGM→Refine resolution bridge).
- `init_refine → refine_similarity → refine_best_depth` (Refine).
- `optimize_depth_sim_map` (gradient-descent fusion).
- Multi-T-camera WTA + FP16-additive aggregation (S25).

Stages **still TODO**:

- Image loading and SfMData I/O — not yet wired; the upstream
  `aliceVision_image` library builds (S30) but no host code routes
  imagery into our `DeviceMipmapImage`.
- Mesh reconstruction (Phase 11+ in `memory/todo.md`).
- Meshroom integration (Phase 11).

---

## 2. Repository layout

```
alicevision-for-mac/
├── CMakeLists.txt                  (root build; options at lines 49-55)
├── cmake/
│   ├── Metal.cmake                 (.metal → .air → .metallib + staging)
│   ├── Warnings.cmake
│   ├── UpstreamShim.cmake          (Path C; alicevision_add_library shim)
│   └── shims/eigen3/               (promiscuous Eigen3 5.x → 3.3 bridge)
├── src/
│   ├── av_gpu/                     (Layer 1 — generic Metal abstraction)
│   │   ├── include/av/gpu/         (public headers)
│   │   └── src/                    (metal-cpp impls)
│   ├── depth_map_metal/            (Layer 2 — depthMap-shaped port)
│   │   ├── include/av/depth_map/   (public headers)
│   │   └── src/                    (host driver classes)
│   └── shaders/depth_map/          (MSL kernels + .h shared with host)
├── tests/                          (33 ctest executables)
├── third_party/
│   └── metal-cpp/                  (vendored Apple metal-cpp headers)
├── upstream/                       (read-only symlink to alicevision-windows/)
├── memory/                         (engineering memory; session log)
├── instructions/                   (engineering handbook)
└── build/                          (generated)
```

---

## 3. Layer 1 — `av::gpu` (generic Metal abstraction)

Lives in `src/av_gpu/`. Pure metal-cpp; no Objective-C++ (`.cpp` files
throughout). Compiles to `libav_gpu.a` (STATIC). The seven public
classes:

### `Device` (`include/av/gpu/Device.hpp`)
Thin RAII handle on a Metal device plus its default command queue.
`Device::default_device()` acquires the system GPU (throws `GpuError` if
none); `load_library()` mmaps a `.metallib`; `make_pipeline(name)`
builds a `Pipeline` from a MSL function symbol. Reports
`recommended_working_set()` and `has_unified_memory()` (always `true` on
Apple Silicon).

### `Queue` (`include/av/gpu/Queue.hpp`, added S28)
Move-only RAII wrapper around `MTL::CommandQueue`. Exists separately
from `Device` so callers can fan out work across multiple queues
(used by `DeviceStreamManager`). `wait_until_completed()` uses the
canonical no-op-command-buffer-plus-`waitUntilCompleted` idiom to drain
the queue (Metal has no direct "wait until idle" API).

### `Buffer` (`include/av/gpu/Buffer.hpp`)
RAII handle on an `MTL::Buffer`. `Storage::Shared` (default) maps the
same physical pages to CPU and GPU — UMA, no `cudaMemcpy`. Exposes
`data()` plus typed `as_span<T>()` and `upload<T>()` for trivially-copyable
types. `Storage::Private` is supported but rarely useful on Apple
Silicon.

### `Texture` (`include/av/gpu/Texture.hpp`)
RAII multi-mip `MTL::Texture` with a descriptor builder, `upload_level()`,
`download_level()`, and `generate_mipmaps()` via blit encoder (used by
`DeviceMipmapImage`). The smoke test in `tests/test_texture_smoke.cpp`
validates bilinear sampling and the mip cascade on a 64×64 ramp.

### `Pipeline` (`include/av/gpu/Pipeline.hpp`)
Move-only RAII handle on an `MTLComputePipelineState`. Built exclusively
by `Device::make_pipeline(function_name)` — name lookup happens against
the currently loaded library.

### `CommandBuffer` (`include/av/gpu/CommandBuffer.hpp`)
Scoped compute encoder. Two constructors: `CommandBuffer(const Device&)`
(default queue) and `CommandBuffer(const Queue&)` (specific queue, S28).
Setter API: `set_pipeline`, `set_buffer`, `set_bytes`, `set_texture`,
`dispatch`, `dispatch_1d`. Commit modes: `commit_and_wait()` for sync
testing, `commit_async()` with a completion handler for steady-state.

### `Errors` (`include/av/gpu/Errors.hpp`)
`GpuError` exception. The single TU `src/metal_cpp_impl.cpp` emits the
`NS::Private::Class::*` and `MTL::Private::*` private-implementation
symbols that metal-cpp's header-only API requires.

---

## 4. Layer 2 — `av::depth_map` (depthMap-shaped port)

Lives in `src/depth_map_metal/`. Consumes `av::gpu`, exposes
AliceVision-flavoured host driver classes. Compiles to
`libav_depth_map_metal.a`. The 16 public headers in
`include/av/depth_map/`:

### Numerical primitives

- **`Eig33`** — Householder + symmetric tridiagonal QL on 3×3 symmetric
  matrices. Single MSL kernel `av_eig33_decompose`. The header
  `eig33.h` is re-includable from other `.metal` translation units
  (extracted in S22 so `depth_sim_map.metal` can call it for PCA without
  duplicating ~150 LOC).
- **`MatrixOps`** — column-major matrix multiplies, projections, outer
  product, geometric primitives, `sigmoid`. Validation kernel
  `av_matrix_validate`.
- **`PatchOps`** — host mirror of the MSL `Patch` + `DeviceCameraParams`
  structs (`packed_float3` keeps them binary-compatible). Validation
  kernel `av_patch_validate`.
- **`ColorOps`** — sRGB EOTF, sRGB→XYZ, XYZ→Lab (with the
  upstream-derived `× 2.55` scaling — see PORTING_NOTES.md §3), HSL,
  Yoon-Kweon adaptive support weight. Validation kernel
  `av_color_validate`.
- **`SimStatOps`** — weighted moments accumulator + `computeWSim` NCC
  similarity. Validation kernel `av_simstat_validate`.

### NCC kernels

- **`CompNCC`** — `compNCCby3DptsYK<TInvertAndFilter>` and the S31
  `compNCCby3DptsYK_customPatchPattern<TInvertAndFilter>`. Four PSO
  variants total (filter × pattern flavour).
- **`DevicePatchPattern`** (S31) — host + MSL mirror of upstream's
  layout-stable patch-pattern struct (216 B × 4 subparts + 4 B count =
  868 B). Bound via `set_bytes`.

### Image-processing kernels

- **`ImageColorConversion`** — in-place `rgb2lab_kernel`
  (`access::read_write` on RGBA32Float). Host driver added
  `Texture::download_level()` so tests can read back results (S8).
- **`GaussianTable`** — host helper that uploads
  `(weights, offsets)` LUTs matching upstream's
  `cuda_createConstantGaussianArray`.
- **`GaussianFilter`** — `downscaleWithGaussianBlur`, `medianFilter3`,
  plus the S31 volume-Gaussian kernels `gaussian_blur_volume_z` and
  `gaussian_blur_volume_xyz`.

### Volume kernels

- **`Volume`** — Phase-7 SGM/Refine hot path. Owns 11+ pipelines for
  `init_sim`, `init_refine`, `add_refine`, `update_uninitialized`,
  `compute_similarity`, `retrieve_best_depth`, `refine_similarity`,
  `refine_best_depth`, and the 4 `volume_optimize` sub-kernels (init Y
  slice, get XZ slice, compute best Z, aggregate cost). The
  `optimize()` method takes an optional `const av::gpu::Texture*
  rc_mipmap` for the adaptive-P2 branch (S31).

### DepthSimMap kernels

- **`DepthSimMap`** — 9 SGM/Refine post-processing kernels:
  `copy_depth_only`, `normal_map_upscale`, `smooth_thickness`,
  `compute_sgm_upscaled_depth_pix_size_map` (nearest + bilinear),
  `compute_normal`, `optimize_var_l_of_lab_to_w`,
  `optimize_get_opt_depth_map`, `optimize_depth_sim_map`.

### Host orchestration (Phase 8)

- **`DeviceMipmapImage`** (S26) — wraps a multi-mip `Texture`; `fill()`
  uploads, optionally Gaussian-downscales, runs `rgb2lab`, copies via
  shared-storage memcpy into level 0 of the destination, and
  `generate_mipmaps()`. The working-texture indirection is a deliberate
  workaround for MSL's `access::read_write` semantics on multi-mip
  textures (see PORTING_NOTES.md §6).
- **`LRUCache<T>` + `CameraPair`** (S27) — header-only template port of
  upstream's slot-stable LRU (`include/av/depth_map/LRUCache.hpp`).
- **`DeviceCache`** (S27) — two `LRUCache` pools: mipmap-images keyed by
  `camera_id`, camera-params keyed by `(camera_id, downscale)`. Three
  deliberate API simplifications vs upstream (no singleton, no
  `MultiViewParams`, no ID indirection — see PORTING_NOTES.md context).
- **`DeviceStreamManager`** (S28) — pool of `av::gpu::Queue` instances
  with modular indexing including negative-index wrap.

---

## 5. MSL kernels (`src/shaders/depth_map/`)

The `.metal` files are compiled by `cmake/Metal.cmake` into a single
`default.metallib` per build, then staged next to every test
executable. The kernels are grouped roughly by phase. Counts below
match the kernel inventory in `src/shaders/depth_map/*.metal`.

### Phase 5 — utility kernels (validation harnesses)

| File                       | Kernels |
| -------------------------- | ------- |
| `eig33.metal`              | `av_eig33_decompose` |
| `matrix_kernels.metal`     | `av_matrix_validate` |
| `patch_kernels.metal`      | `av_patch_validate` |
| `color_kernels.metal`      | `av_color_validate` |
| `simstat_kernels.metal`    | `av_simstat_validate` |
| `comp_ncc.metal`           | `av_compNCC_validate_no_filter`, `av_compNCC_validate_filter`, `av_compNCC_customPattern_no_filter`, `av_compNCC_customPattern_filter` |
| `texture_smoke.metal`      | `av_texture_sample` |

Headers shared across `.metal` translation units (no kernels, included
both from `.metal` and from the host C++ that builds the matching
host-side reference): `operators.h`, `matrix.h`, `Patch.h`, `color.h`,
`SimStat.h`, `DevicePatchPattern.h`, `eig33.h`, `volume_helpers.h`.

### Phase 6 — image processing

| File                       | Kernels |
| -------------------------- | ------- |
| `color_conversion.metal`   | `av_rgb2lab` |
| `gaussian_filter.metal`    | `av_downscale_with_gaussian_blur`, `av_median_filter_3`, `av_gaussian_blur_volume_z`, `av_gaussian_blur_volume_xyz` |

### Phase 7 — SGM / Refine / Optimize / DepthSimMap

| File                              | Kernels |
| --------------------------------- | ------- |
| `volume_kernels.metal`            | `av_volume_init_uchar`, `av_volume_init_half`, `av_volume_add_half`, `av_volume_update_uninitialized_uchar`, `av_volume_retrieve_best_depth` |
| `volume_compute_similarity.metal` | `av_volume_compute_similarity` |
| `volume_optimize.metal`           | `av_volume_init_y_slice_uchar`, `av_volume_get_xz_slice_uchar_to_uint`, `av_volume_compute_best_z_in_slice`, `av_volume_aggregate_cost_at_x` |
| `volume_refine_similarity.metal`  | `av_volume_refine_similarity` |
| `volume_refine_best_depth.metal`  | `av_volume_refine_best_depth` |
| `depth_sim_map.metal`             | `av_depth_sim_map_copy_depth_only`, `av_map_upscale_float3`, `av_depth_thickness_smooth_thickness`, `av_compute_sgm_upscaled_depth_pix_size_map_nearest`, `av_compute_sgm_upscaled_depth_pix_size_map_bilinear`, `av_depth_sim_map_compute_normal`, `av_optimize_var_l_of_lab_to_w`, `av_optimize_get_opt_depth_map`, `av_optimize_depth_sim_map` |

Total: **35 distinct kernel entry points** spread across 15 `.metal`
files.

---

## 6. Pipeline orchestration

The canonical end-to-end example is `tests/test_depth_pipeline.cpp`
(Session 24, "first fused depth map produced by the full Apple Silicon
depth pipeline"). It wires all four stages on a synthetic
plane-induced-homography scene (128 × 96 image, SGM ROI 16 × 12 / 11
depth planes, Refine ROI 32 × 24 / 9 z-slices, Optimize 20 iter):

```
SGM
├── Volume::init_sim                  (allocate cost + 2nd-best volume)
├── Volume::compute_similarity        (per-voxel NCC, WTA across T-cameras)
├── Volume::optimize                  (4-direction DP aggregation)
└── Volume::retrieve_best_depth       (WTA + thickness + sim normalization)
        │
        ▼ (out: depth_thickness_map @ SGM resolution)
Bridge
└── DepthSimMap::compute_sgm_upscaled_depth_pix_size_map
        │
        ▼ (out: depth_pixsize_map @ Refine resolution)
Refine
├── Volume::init_refine               (FP16 half volume)
├── Volume::refine_similarity         (NCC<true> + sigmoid invert-and-filter,
│                                     additive promote-add-demote)
└── Volume::refine_best_depth         (Gaussian-weighted convolution → depth)
        │
        ▼ (out: refined_depth_sim_map @ Refine resolution)
Optimize
├── DepthSimMap::optimize_depth_sim_map (N-iter gradient-descent fusion of
                                         SGM rough + Refine fine via chained
                                         sigmoid blend)
```

Result on 768 pixels in S24: 87 % valid, lock-on rate 94.0 % within 1.5
SGM depth-plane steps, median |Δ truth| = 0.1010.

For an integration test of only the SGM half, see
`tests/test_sgm_pipeline.cpp` (S14, S15). For the Refine half see
`tests/test_refine_pipeline.cpp` (S18). For multi-T-camera aggregation
patterns specifically, `tests/test_multi_t_aggregation.cpp` (S25).

---

## 7. Apple Silicon specifics

Decisions that are load-bearing for porting reviewers:

- **UMA storage**. Default `MTLResourceStorageModeShared`. No
  `cudaMemcpy` mental model. `Buffer::data()` returns the same pointer
  the GPU dereferences. Synchronization happens at the command-buffer
  boundary (`commit_and_wait()` or `addCompletedHandler:`). Read-after-
  write without a barrier is undefined even when the memory is "visible."
- **MTLCommandQueue per stream**. `DeviceStreamManager` owns a vector
  of `av::gpu::Queue` instances; each queue is one independent FIFO of
  command buffers. `wait_until_completed()` submits a no-op buffer
  and waits on it (Metal has no direct "queue drain" API).
- **`packed_float3`** for any GPU struct that the host C++ also
  defines. Plain `float3` in MSL is 16-byte-aligned (vec4-padded);
  `packed_float3` is 12 bytes. `DeviceCameraParams` (276 B) and
  `DevicePatchPattern` (868 B) are byte-compatible across MSL and the
  host mirror in `av/depth_map/PatchOps.hpp` / `DevicePatchPattern.hpp`.
- **FP32-only kernels**. Apple GPUs have no FP64. Everything that
  upstream computed in `double` (Eig33, stat3d, several patch-geometry
  helpers) was ported to `float`. The numerical impact is documented
  per-kernel in [PORTING_NOTES.md](PORTING_NOTES.md). FP16 (`half`)
  appears as the storage type for the Refine cost volume (`TSimRefine`),
  promoted to `float` for arithmetic per upstream's
  promote-add-demote pattern.
- **`access::read_write` on multi-mip textures requires explicit-LOD
  reads/writes** — our `rgb2lab` kernel uses the implicit form, which
  is why `DeviceMipmapImage::fill` indirects through a single-mip
  working texture (S26 discovery; PORTING_NOTES.md §6).

---

Related: [BUILD.md](BUILD.md) · [PORTING_NOTES.md](PORTING_NOTES.md) ·
`memory/philosophy.md` · `memory/mindmap.md`
