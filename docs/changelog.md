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

## Phase 15 — Native SwiftUI app (decommissioned 2026-05-23)

An earlier track shipped a parallel native SwiftUI Meshroom frontend at
`meshroom-native/` (M1–M6 milestones, 115 Swift tests). It was retired on
2026-05-23 to consolidate work on the upstream-compatible PySide6
Meshroom: maintaining two graph editors against an evolving upstream node
schema doubled review surface for marginal user benefit. The Swift code,
tests, scripts, and per-page documentation have been removed.

## S52–S54 — AI segmentation & macOS Qt UI fixes

- S52: `SegmentationBiRefNet` Meshroom node landed (rembg + ONNX Runtime
  CoreML EP, with `ONNX_FORCE_CPU=1` escape hatch for the broken
  `CPUAndGPU` path on `swin_v1`).
- S53: native plugin system at `plugins/` — `plugin.json` manifests
  consumed via `MESHROOM_NODES_PATH`.
- S54: macOS Qt UI fixes for meshroom-mac (semaphore leak via
  `ThreadPoolExecutor`, RHI backend pinned to Metal, Qt3D deferred behind
  `MESHROOM_ENABLE_VIEWER3D=1`, `PrepareDenseScene.size` uses
  `DynamicViewsSize`).
- 2026-05-23: pre-converted BiRefNet `.mlpackage` models added at
  `ai-models/`; conversion pipeline at `models/`. ANE found to be not
  viable for this graph (deformable conv v2 lowered via `grid_sample`
  cannot be planned by the ANE compiler); `MLComputeUnits.cpuAndGPU` is
  the production target on Apple Silicon.

## Phase 13 + 14 — Feature parity sprint (2026-05-23 → 2026-05-24)

Brought the port from 12 to **60 binaries** and from **4 of 25 to 25 of 25
covered** Meshroom templates in nine focused phases. Full per-phase notes are
in the top-level [CHANGELOG.md](https://github.com/SeedeXR/alicevision-for-mac/blob/main/CHANGELOG.md);
high-level recap:

- **Phase 14.1** — 6 modern-SfM binaries (`sfmBootstrapping`, `sfmExpanding`,
  `relativePoseEstimating`, `sfmTriangulation`, `tracksBuilding`, `meshDecimate`)
  + a CMake-time inline patch for `relativePoses.hpp` (ODR fix).
- **Phase 14.2** — 3 HDR binaries (`LdrToHdrSampling`, `LdrToHdrCalibration`,
  `LdrToHdrMerge`). `hdrFusion` template covered.
- **Phase 14.3** — 8 panorama binaries + bonus `sfmTransform`. `panoramaHdr`
  + `panoramaFisheyeHdr` covered.
- **Phase 14.4a** — `lightingCalibration` + `photometricStereo` binaries + 2
  photometric sublibs. OpenCV `find_package` wired in.
- **Phase 14.4b** — **CoreML port of `sphereDetection`** (replaces upstream's
  ONNX Runtime). New `src/sphere_detection/` module wrapping
  `ai-models/yolov8n.mlpackage`. Runs on ANE (3× faster than GPU).
- **Phase 14.5** — 21 utility + lidar binaries via parallel agent dispatch.
  6 templates flip to covered (`colorCalibration`, `distortionCalibration`,
  `lidarMeshing`, `photogrammetry`, `photogrammetryObjectTwoSides`,
  `rawImageConversion`).
- **Phase 14.6** — Alembic enabled in the existing `sfmDataIO` sublib by
  setting `ALICEVISION_HAVE_ALEMBIC=1` BEFORE `add_subdirectory`. 2 binaries
  (`exportAlembic`, `exportAnimatedCamera`); 8 cameraTracking-family templates
  flip to covered.
- **Phase 14.7** — 3 Mac-port-native binaries at `src/native_binaries/` for
  2026.1.0 features upstream defers to Python pipelines: `starListing`
  (algorithmic), `matchMasking` + `moGe` (honest stubs). Coverage 25/25 in
  the matrix sense.
- **Phase 14.8** — **CoreML port of `moGe`** (replaces Phase 14.7 stub). New
  `src/moge/` module wrapping `ai-models/moge2_504x672_t1728.mlpackage`
  (DINOv2 ViT-B/14). Real per-view depth + normals at 504×672. Runs partially
  on ANE (~228 ms vs ~384 ms CPU).
- **Phase 14.9** — **CoreML port of `matchMasking`** (TinyRoMa, replaces Phase
  14.7 stub). New `src/roma/` module wrapping
  `ai-models/tiny_roma_v1_480x640.mlpackage`. CRITICAL: uses
  `MLComputeUnitsCPUAndGPU` (not `.all`) — ANE is **4× slower** here due to
  `grid_sample` handoffs. **Zero honest stubs remain.**
- **Phase 13** — Native `pyalicevision` SWIG bindings (`hdr`, `sfmData`,
  `sfmDataIO`) replace the pure-Python stubs when `AV_BUILD_PYALICEVISION=ON`.
  Auto-discovery via `__path__` manipulation; falls back to stubs when off.
  Critical macOS gotcha: must pass `-DLINUXPLATFORM` to SWIG to avoid a
  `size_t` typedef mismatch with `__darwin_size_t`.

## Current state (2026-05-24)

- **60 pipeline binaries**, ARM64 native, codesigned.
- 32 upstream modules + 7 new sublibs compiled (panorama, hdr,
  photometricStereo, lightingEstimation, calibration, keyframe,
  imageProcessing).
- **3 native ML wrapper modules** (sphere_detection, moge, roma) + the
  segmentation Python plugin.
- 41 MSL kernel entry points.
- 15 `cuda_*` adapter forwarders, audited.
- 37/37 C++ tests pass under `ctest -j8`.
- **73 passed / 25 skipped** in `pytest tests/python` (gated heavy E2E
  behind `RUN_*=1` env vars).
- **4 CoreML models** at `ai-models/`:
  BiRefNet × 2 (`cpuAndGPU`), YOLOv8n (`.all` / ANE),
  MoGe-2 (`.all` / partial ANE), TinyRoMa (`cpuAndGPU` — ANE is a regression).
- **25 / 25 Meshroom templates covered**.
- 4 templates E2E-verified end-to-end on Monstree mini3 in ~1 min on M4.
- Self-contained `.app` bundling (155 MB dylibs) + DMG packaging shipped.

## Roadmap

For the remaining 21 covered-but-load-only templates to be E2E-verified,
fixture datasets are needed (HDR brackets, calibration spheres, LIDAR
`.e57`s, checkerboards, RAW images, calibrated video footage, etc.).
That's data collection, not code.

Code-side items still open:

- Developer-ID notarization (`scripts/codesign_macos_app.sh --identity` path).
- Unified `aliceVision` binary with subcommand dispatch.
- Custom mipmap cascade to bit-match upstream `deviceMipmappedArray.cu`.
- Roma at FP32 for sub-1% argmax-sensitivity diff (currently FP16, 2.3% rel max).
