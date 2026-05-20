# Changelog

Session-by-session summary of the port. The session numbers (S0, S3, …)
match `memory/handover_session.md` / `memory/mental_note.md` entries in
the repo. Dates are approximate (the port has run through May 2026).

Phases below are coarse-grained groupings of related sessions.

## Phase 1 — Bootstrap (S0–S2)

- S0: Repository layout, CMake skeleton, `upstream/` read-only symlink to
  `alicevision-windows/AliceVision`.
- S1: `cmake/Metal.cmake` — `.metal → .air → .metallib` pipeline.
- S2: `av::gpu` Layer 1 — `Device`, `Buffer`, `Texture`, `Pipeline`,
  `CommandBuffer`, `Errors`. Pure metal-cpp; no Objective-C++.

## Phase 5 — Numerical primitives (S3–S8)

- S3: `Eig33` — Householder + QL on 3×3 symmetric matrices. FP32 port of
  upstream FP64 algorithm. Worst eigenvalue rel err 2.71e-6.
- S4-S7: `MatrixOps`, `PatchOps`, `ColorOps`, `SimStatOps`, `CompNCC`. All
  validated against CPU references with `1e-5` rel budget.
- S8: `ImageColorConversion` (sRGB → Lab × 2.55 preservation; the
  classical-vs-upstream-Lab gotcha lands here).

## Phase 6 — Image processing (S9)

- S9: `GaussianFilter` (`downscaleWithGaussianBlur`, `medianFilter3`).
  Custom mipmap cascade (`deviceMipmappedArray.cu`) **deliberately
  deferred** — Metal's built-in `generate_mipmaps()` substitutes.

## Phase 7 — SGM / Refine / Optimize (S10–S25)

- S10-S14: SGM core kernels — `init_sim`, `compute_similarity`, `optimize`
  (4-direction DP), `retrieve_best_depth`.
- S15: `test_sgm_pipeline` — first end-to-end SGM run on a synthetic
  plane-induced-homography scene.
- S16-S18: Refine kernels — `init_refine`, `refine_similarity` (FP16),
  `refine_best_depth`. End-to-end `test_refine_pipeline` in S18.
- S20: `smooth_thickness` — first sub-FP32-ULP drift case (`-ffast-math`
  fuses `fmax/fmin` into `clamp` intrinsic).
- S21: SGM→Refine bridge (`compute_sgm_upscaled_depth_pix_size_map`).
- S22: `compute_normal` — `Stat3d` FP32 PCA accumulators (replacing
  upstream's FP64). Worst cos deviation 1.81e-5.
- S23, S24: `optimize_depth_sim_map` — chained-sigmoid blend. First case
  needing a relaxed budget (`1e-3` sim) under `-ffast-math`. S24
  `test_depth_pipeline`: full SGM → Bridge → Refine → Optimize on a
  128×96 fused scene. 87 % valid, 94 % lock-on within 1.5 SGM steps.
- S25: Multi-T-camera aggregation (WTA cost + FP16 additive refine).

## Phase 8 — Host orchestration (S26–S31)

- S26: `DeviceMipmapImage` — working-texture indirection for
  multi-mip `access::read_write` semantics.
- S27: `LRUCache<T>` + `DeviceCache`. Header-only template port of
  upstream's slot-stable LRU.
- S28: `DeviceStreamManager` — multi-`MTLCommandQueue` pool with negative-
  index wrap.
- S30: First upstream module compiled (`aliceVision_image`).
- S31: Adaptive-P2 SGM path (4 `_fc` variants of `volume_optimize`
  kernels). `DevicePatchPattern` byte-identical (868 B) across MSL + host.

## Phase 8b — Adapter forwarders (S33–S38)

- S33-S35: `upstream_adapter.hpp` skeleton; 12 (later 15) `cuda_*`
  symbols declared.
- S36-S37: Type-shim layer for `CudaDeviceMemoryPitched<T, N>`,
  `CudaSize<N>`, `float2/3/4`, `__half`, `cudaStream_t`.
- S38: First `aliceVision_depthMapEstimation` binary builds + links.
  Eight pipeline binaries built: `importMiddlebury`, `cameraInit`,
  `featureExtraction`, `imageMatching` (target `_bin`),
  `featureMatching`, `incrementalSfM`, `prepareDenseScene`,
  `depthMapEstimation`. 19 upstream modules built. libc++ /
  `vector<T>::reserve` instantiation patch lands.

## Phase 9 — End-to-end pipeline (S39–S40)

- S39: Full SfM cascade runs on Monstree mini3 — `cameraInit` through
  `prepareDenseScene` produces 7,430 landmarks. depthMap hangs at
  "Retrieve best depth in volume" due to conditionally-allocated null
  pointer.
- S40: Two-bug cascade fix in `cuda_volumeRetrieveBestDepth`:
  `maxSimilarity × 254` + `1.f + depthThicknessInflate`. **Monstree mini3
  now produces real depth maps** (Min=-2, Max=20-22, Avg≈3-4 on the
  SfM-reported range). 8 pipeline binaries shipped.

## Phase 10 — Mesh reconstruction (S41)

- S41: Meshing pipeline lands clean — `aliceVision_depthMapFiltering`,
  `aliceVision_meshing`, `aliceVision_meshFiltering`,
  `aliceVision_texturing`. ZERO CMake-time patches needed (everything
  compiled against the existing shim path). 12 pipeline binaries
  shipped. 32 upstream modules. Adapter audit
  (`memory/adapter_audit_s41.md`).
- **End-to-end deliverable**: Monstree mini3 → texturedMesh.obj + .mtl +
  8192² PNG atlas. Total wall-clock ~1 min on M4.

## Phase 11 — Meshroom integration (S42)

- S42: 4 Darwin patches against upstream Meshroom land
  (`init-darwin-libpath`, `stats-darwin-gpu`, `cgroup-darwin-sysctl`,
  `startsh-readlink-portable`). 2 patches against AliceVision node
  descriptors (ABC → SfM / PLY for the no-Alembic build).
  `scripts/run_meshroom.sh` wraps the env vars Meshroom expects.
  `meshroom-mac/` working copy + `meshroom-venv/` ship in-tree.

## Phase 12 — Release engineering (S46)

- S46: `Formula/alicevision-for-mac.rb` Homebrew formula. CMake
  `package` target produces `build/release/*.tar.gz`. Ad-hoc codesign
  via `install(CODE ...)` blocks. `phase12_install_smoke.sh` exercises
  the install + tarball path. Tarball is not fully vendored —
  Homebrew runtime dylibs expected on consumer machine.

## Phase 14 — Performance profiling + optimization (S43–S45)

- S43: `AV_PROFILE_ADAPTER` option lands. Per-forwarder timing
  accumulator (RAII `ScopeTimer` + atexit-flushed sorted table).
  Baseline profile on Monstree mini3 view 0: adapter total 14.5 s,
  `cuda_volumeOptimize` 63 %.
- S44: `cuda_volumeOptimize` — batch all dispatches per SGM path onto
  one command buffer + encoder. **-49.6 %** (9.16 s → 4.62 s).
- S45: `cuda_volumeComputeSimilarity` — threadgroup reshape from
  `{16, 4, 1}` to `{4, 2, 8}` (Z-coherent texture cache hits).
  **-65.0 %** (3.26 s → 1.14 s). Adapter total 12.4 s.

## Phase 15 — Native SwiftUI app (M1–M6)

- M1: `ProjectModel` — `.mg` round-trip + template-reference parser.
  Foundation-only.
- M2: App scaffold; `MeshroomNativeApp` SwiftUI executable; open `.mg`.
- M3: Schema-driven parameter forms (Stepper / Slider / Toggle / Picker
  / List / GroupAttribute).
- M4: Inspector pane; parameter editing with `UndoManager`.
- M5: `GraphExecutor` — topo-sort, `Process`-per-node, AsyncStream
  events. Cmd-R to run; Cmd-period to stop.
- M6: Graph canvas with drag-to-connect edges. 115 Swift tests pass.

## Current state (S47, 2026-05-20)

- 12 pipeline binaries, ARM64 native, codesigned.
- 32 upstream modules compiled.
- 41 MSL kernel entry points.
- 15 `cuda_*` adapter forwarders, audited.
- 37/37 C++ tests pass under `ctest -j8`.
- 115/115 Swift tests pass under `swift test`.
- End-to-end Monstree mini3 → textured 3D mesh in ~1 min on M4.
- Release tarball + Homebrew formula in place (formula needs published
  GitHub release URL).

## Roadmap

Open in `memory/todo.md`:

- **S48 (current)**: docs site (this), native UI M7/M8/M9, `llms.txt`
  generation, Phase 14 R4 (refine_similarity threadgroup reshape), ctest
  flake investigation.
- **Phase 12 finish**: Developer-ID signing, `notarytool` integration,
  `dylibbundler` for a fully-vendored tarball.
- **Phase 3**: Unified `aliceVision` binary with subcommand dispatch.
- **Native UI M7-M9**: cache UI, 3D viewer (Metal), asset library.
- **Custom mipmap cascade**: bit-match the upstream
  `deviceMipmappedArray.cu` for future numerical-parity work.
