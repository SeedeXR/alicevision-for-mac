# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIT LICENSE (overlay code) with third-party-notices section covering upstream MPL-2.0 / Boost-1.0 / Apache-2.0 components.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub issue / PR templates.
- `docs/dev/codebase-navigation.md` — one-page repo-layout guide.
- AI segmentation node `SegmentationBiRefNet` using rembg + BiRefNet ONNX → CoreML; pre-flight model downloader; Native UI integration.
- Native plugin system: `plugins/ai-segmentation/` self-contained plugin with `plugin.json` manifest + Swift manifest loader. Third-party plugin contract documented at `docs/dev/plugin-system.md`. AI segmentation is the first plugin.
- `docs/dev/apple-silicon-optimization.md` deep-dive: UMA, Metal, ANE, CPU performance characteristics with measured numbers.
- `ONNX_FORCE_CPU=1` env var to pin segmentation to CPU EP (recommended default until ANE warmup is automated).

### Changed
- `.gitignore` consolidated to single root file (removed redundancies, added explicit reasons per section).
- **PySide6 upgraded 6.8.3 → 6.11.1** in `meshroom-venv` to ship `Qt3DQuickScene3D.framework` (missing from 6.8.x wheels — broke Viewer3D with `QQmlComponent: Component is not ready`).
- `meshroom-mac/start.sh` now sources the same `ALICEVISION_ROOT` / `MESHROOM_NODES_PATH` / `DYLD_FALLBACK_LIBRARY_PATH` env as `scripts/run_meshroom.sh`. Without this, the Qt UI registered every aliceVision node as `UnknownNodeType`.
- Three `multiprocessing.pool.ThreadPool` instances in `meshroom-mac` (ThumbnailCache.workerThreads, FilesModTimePollerThread._threadPool, Scene._workerThreads) replaced with `concurrent.futures.ThreadPoolExecutor`. The MP pool's internal SimpleQueue+Lock pair stayed registered with `multiprocessing.resource_tracker` under Qt's QObject parent hierarchy, producing a 6-semaphore leak warning at every shutdown. Executor has no such tracker registration.
- `meshroom-mac/meshroom/ui/__main__.py` SIGINT handler now routes to `QApplication.quit()` so `aboutToQuit` fires on Ctrl-C (previously `signal.SIG_DFL` killed the process at the C level, bypassing Qt teardown).

### Fixed
- `meshroom-mac/meshroom/core/graph.py` + `node.py` — guard `node.nodeDesc` accesses against `None` (CompatibilityNode case). Default templates with unknown node types (`ImageSegmentationSam3`, `ScenePreview`, `MoGe`, etc.) no longer crash graph load with `AttributeError: 'NoneType' object has no attribute 'hasPreprocess'`.
- `meshroom-mac/nodes/aliceVision/ExtractMetadata.py` — removed dead `import distutils.dir_util` (broken on Python 3.13, never used).
- **Render-loop segfault on macOS 26.5 Apple Silicon**: Apple's OpenGL drivers crashed inside `glDrawElements_ACC_Exec` / `glDrawElements_GL3Exec` during `QRhi::endFrame`, exiting the app a few seconds after launch and on every click of "New Project" / pipeline buttons. Forced Metal RHI backend via `QQuickWindow.setGraphicsApi(QSGRendererInterface.GraphicsApi.Metal)` at app start. (`QSG_RHI_BACKEND=metal` env var is silently ignored on Qt 6.11.1 / macOS 26.5; only the API call works.)
- **Qt3D Python imports force OpenGL backend at process load**: `meshroom/ui/components/scene3D.py` did `from PySide6.Qt3DCore import Qt3DCore` at module top, which registered Qt3D with the QML engine and pinned the RHI backend to OpenGL even with Viewer3D disabled. Deferred import behind `MESHROOM_ENABLE_VIEWER3D=1` env var.
- **`viewer3DVisibilityCB` defaults `checked: false`** so the Loader does not instantiate Qt3D Scene3D at startup. Qt3D's Scene3D embedding cannot share a Metal context with QtQuick on macOS; the native Metal 3D viewer arrives in Track B B4. Users can toggle the menu item back on at their own risk.
- **`_viewer3DAvailable` context property** gates the panel3dViewer Loader's `active` (AND'd with the QML `Settings`-persisted menu toggle). Without this, QML's `Settings` cache remembers a user's previous `showViewer3D=true` and re-instantiates Qt3D — triggering `Qt3D.Renderer.RHI.Backend: Failed to build graphics pipeline: vertex attribute vertexNormal(1) missing` and another segfault on every "New Project" click.
- **`PrepareDenseScene.size` switched from `desc.DynamicNodeSize` to `avpar.DynamicViewsSize`** to fix a silent skip in `meshroom_batch`. The framework's `DynamicNodeSize` reads `inputLink.node.size` — a stored value captured at template-load time (before `CameraInit.initialize` populates viewpoints), which on this Mac port returns 0 → 0 chunks → step `[6/12]` silently absent from the log → `DepthMap` then fails with "Cannot find image file corresponding to the view '<viewId>'". `DynamicViewsSize` (our `pyalicevision/parallelization.py` shim) parses the upstream SfMData JSON directly and always returns ≥ 1, bypassing the stale-size chain. End-to-end `photogrammetryLegacy` now completes 12/12 nodes on Monstree mini3 and produces a textured mesh of ~4756 vertices (Phase 11 documented 4790 — matches within ±1%).

### Tests
- `tests/python/test_meshroom_mac_ui.py` — 13 tests (8 source-check unit, 5 UI smoke gated behind `RUN_MESHROOM_UI=1`) guarding Track A fixes.
- `tests/python/test_pipeline_integration.py` — 31 tests (6 always-on coverage assertions, 2 E2E `photogrammetryDraft`/`photogrammetryLegacy` runs on Monstree mini3 gated behind `RUN_PIPELINE_E2E=1`, 23 uncovered-pipeline diagnostic tests).
- `scripts/pipeline_coverage.py` — coverage-matrix tool: maps each `.mg` template to required binaries vs shipped binaries; classifies as covered/uncovered with specific missing pieces named.
- Pipeline test results on Apple M4: `photogrammetryDraft` (10 nodes, ~30s) and `photogrammetryLegacy` (12 nodes, ~71s incl warm cache) both pass; produce textured meshes (~4756 verts for Monstree mini3, matches Phase 11's 4790 ± noise).

### Pipeline-coverage triage
Surveyed all 25 Meshroom templates against the 12 native arm64 Metal binaries we ship today. **2 covered** (`photogrammetryDraft`, `photogrammetryLegacy`); **23 uncovered** because they require binaries we have not built yet (modern SfM: `aliceVision_relativePoseEstimating` / `sfmBootStrapping` / `sfmExpanding` / `sfmColorizing` / `intrinsicsTransforming` / `tracksBuilding`; camera tracking: `aliceVision_keyframeSelection` / `applyCalibration` / `convertSfMFormat` / `exportAnimatedCamera` / `exportAlembic` etc.; HDR: 3 `LdrToHdr*` binaries; panorama: 8 `panorama*` binaries; photometric stereo: `lightingCalibration` / `photometricStereo` / `sphereDetection`; lidar: 3 `lidar*` binaries) and/or 2026-upstream descriptors we haven't ported (`ImageDetectionPrompt`, `ImageSegmentationBox`, `ScenePreview`, `SfMBootStrapping`, `SfMExpanding`, `MoGe`, `StarListing`, `RomaMatcher`/`Sampler`/`Reducer`, `MatchMasking`). Full matrix in `scripts/pipeline_coverage.py --json`.

## [0.1.0] — 2026-05-20 (S0-S49)

The S0-S49 development log lives in `docs/changelog.md` (per-session
deliverables) and `docs/perf-history.md` (kernel-by-kernel perf timeline).
The summary below is the rollup.

### Added
- 12 native Apple Silicon `aliceVision_*` pipeline binaries (cameraInit, featureExtraction, imageMatching, featureMatching, incrementalSfM, prepareDenseScene, depthMapEstimation, depthMapFiltering, meshing, meshFiltering, texturing, importMiddlebury).
- Native macOS SwiftUI Meshroom replacement (`meshroom-native/`) with viewer, drag-to-move, parameter editing, graph execution, multi-output pins, type-checked connections, node-creation palette.
- Meshroom integration via `meshroom-mac/` + `patches/meshroom/` + `patches/alicevision-meshroom/`. End-to-end Monstree run produces textured 3D mesh.
- MkDocs Material documentation site (19 pages, 16 Mermaid diagrams) under `docs/`.
- `llms.txt` for AI-agent consumption of the project digest.
- Homebrew formula `Formula/alicevision-for-mac.rb`.
- Dual-tarball release: `alicevision-for-mac-0.1.0-arm64.tar.gz` (15 MB binary) + `alicevision-for-mac-0.1.0-arm64-dSYM.tar.gz` (161 MB symbols).
- Comprehensive Metal kernel ports (~41 MSL kernels in `src/shaders/`).
- `AV_PROFILE_ADAPTER` CMake option for per-forwarder timing.

### Performance
Cumulative S43→S48 kernel optimizations on Monstree mini3 view 0:

| Kernel | Baseline | Final | Δ |
|---|---|---|---|
| `cuda_volumeOptimize` | 9161 ms | 4577 ms | **-50%** |
| `cuda_volumeComputeSimilarity` | 3260 ms | 1111 ms | **-65.9%** |
| `cuda_volumeRefineSimilarity` | 789 ms | 612 ms | **-22.4%** |
| Adapter grand total | 14504 ms | 11870 ms | **-18.2%** |

### Fixed
- S40: `cuda_volumeRetrieveBestDepth` scaling bugs (maxSimilarity needs ×254, thicknessMultFactor needs 1.f+). Produces real depth values on real photogrammetric input.
- S46: `ninja package` cd-before-mkdir bug.
- S48: `test_upscale_depth_pixsize` OOB-read flake under `ctest -j8`.

### Validation
- `ctest -j8`: 37/37 pass reliably.
- `swift test`: 151/151 pass.
- End-to-end on Monstree mini3 (3 photos): produces textured `texturedMesh.obj` (8,431 vertices, 8192² texture) in ~12 minutes wall-clock.

### Known limitations
- Notarization deferred — needs Apple Developer ID (user-gated).
- CUDA wall-clock perf comparison deferred — needs NVIDIA hardware.
- Segmentation pipeline not built — gated on adding upstream's `aliceVision_segmentation` (out of scope for 0.1.0).
- Monstree-full E2E timing deferred — needs user-run of `bash scripts/run_meshroom.sh full` (41 JPGs).

[Unreleased]: https://github.com/<placeholder>/alicevision-for-mac/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/<placeholder>/alicevision-for-mac/releases/tag/v0.1.0
