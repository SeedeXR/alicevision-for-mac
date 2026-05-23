# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (2026-05-23 Phase 5 — PLY point clouds + wireframe frustums + UX polish)
- **PLY point-cloud rendering** in `MetalScenePreview`. A new `PointCloudGeometry` (`meshroom-mac/meshroom/ui/components/pointCloudGeometry.py`) subclasses `QQuick3DGeometry`, parses PLY files (ASCII + binary_little_endian) with per-vertex RGB color, and uploads positions+colors interleaved into a single Qt RHI Metal shared-storage buffer. Verified end-to-end on real Monstree `cloud_and_poses.ply` (7,433 points) and `densePointCloud.ply` (7,810 points). Activates automatically when the ScenePreview manifest's model path resolves to a `.ply` (no GLTF/OBJ); routed via the empty-`modelPath` branch the Python `ScenePreviewLoader` was already returning.
- **True wireframe camera frustums** in `CameraFrustumGroup`. New `FrustumGeometry` (`meshroom-mac/meshroom/ui/components/frustumGeometry.py`) builds 12 line segments per frustum from `nearPlane × farPlane × fovYDegrees × aspectRatio` + a look-at stub from the origin to the near plane centre. One shared geometry, N draw calls per scene (transformed per-instance by the Model's worldTransform). Replaces the Phase 4 cube-marker placeholders. Pose conversion: `SfMData.poses[].transform.rotation` is a 3×3 world-to-camera matrix; we transpose to camera-to-world + convert to Euler XYZ degrees for QtQuick3D's `eulerRotation` property.
- **Real `fitCameraToBounds`** — reads `loadedNode.bounds` from QtQuick3D's RuntimeLoader for meshes, and `geometry.boundsMin/boundsMax` for the PLY path. Auto-scales the frustum near/far planes proportional to scene radius so per-view markers stay visible at any zoom.

### Fixed (2026-05-23 Phase 5 — recurring user-reported issues)
- **`qt.qpa.fonts: missing font family "Monospace, Consolas, Monaco"`**: replaced the comma-separated CSS-style fallback list with `"Menlo"` (native macOS monospace) at `meshroom-mac/meshroom/ui/qml/Controls/TextFileViewer.qml:192,296`. Qt's font system doesn't honour CSS-style fallback lists; the alias-resolution attempt added a 39 ms startup cost on every Meshroom launch.
- **`qml: Missing plugin qtAliceVision`**: demoted the noisy `console.warn` in `meshroom-mac/meshroom/ui/qml/Viewer/Viewer2D.qml:91` to a `console.info` with explanatory message. The qtAliceVision C++ plugin (HDR / panorama / lens-distortion / feature-overlay viewers) is not built on the macOS port; the codebase already gracefully falls back to Qt's standard `Image` element. Full porting plan in `memory/qtalicevision_status.md` (5–8 day estimate, deferred until justified).
- **MetalScenePreview viewer didn't fill its panel**: switched `View3D.renderMode` from `Offscreen` to `Underlay` (renders directly into QtQuick's Metal backing store; the Item's size tracks every frame via Qt RHI scissor/viewport). Added `implicitWidth: parent.width` / `implicitHeight: parent.height` fallback on the root Item, and `anchors.fill: parent` + `Layout.fillWidth: true` + `Layout.fillHeight: true` on the Panel content. Defensive layering against macOS Qt 6.11.1 layout-cycle ordering issues.

### Added (2026-05-23 Phase 4 — Mac-native QtQuick3D scene preview viewer)
- **MetalScenePreview** at `meshroom-mac/meshroom/ui/qml/MetalViewer3D/MetalScenePreview.qml` — a Mac-native QtQuick3D-based 3D viewer that renders the reconstructed mesh + camera frustums from a ScenePreview node's output folder. Uses Qt RHI's Metal backend directly (the same Metal RHI the rest of the app forces via `QSGRendererInterface.GraphicsApi.Metal` in `app.py:44`). **Does NOT pull Qt3D into the QML engine** — coexists cleanly with QtQuick on macOS 26.5 Apple Silicon, unlike the legacy Qt3D Scene3D viewer which is still gated off via `MESHROOM_ENABLE_VIEWER3D=0`.
- **CameraFrustumGroup** — per-viewpoint camera marker rendering. Reads AliceVision SfMData JSON, places a small marker at each view's world-space camera centre. Up to 200 cameras by default. (First cut: cube markers; QQuick3DGeometry-based true wireframe frustums deferred to a follow-up.)
- **ScenePreviewLoader** Python helper at `meshroom-mac/meshroom/ui/components/scenePreviewLoader.py` — parses `scene_preview.json` and exposes resolved file paths (modelPath, modelUrl, camerasPath, masksPath) to QML as `Q_PROPERTY`s. Honours the symlink convention from the ScenePreview node + falls back to the manifest's absolute paths if symlinks are missing. Registered as a QML type at `Meshroom.Helpers/ScenePreviewLoader`.
- **WorkspaceView integration** — added a second `panel3dMetalViewerLoader` next to the existing `panel3dViewerLoader`. Both reuse the user's `Show 3D Viewer` toggle; routing is automatic:
  - Default: `_metalViewer3DAvailable=true` (via `MESHROOM_ENABLE_METAL_VIEWER3D` env var, default `"1"`) → Metal viewer renders.
  - Diagnostic: `MESHROOM_ENABLE_VIEWER3D=1` → legacy Qt3D viewer takes over (Metal viewer steps aside).
- **End-to-end verified**: real Monstree `texturedMesh.obj` loads via the QtQuick3D `RuntimeLoader` in 2.5 s wall-clock (Apple M-series). Scene state reports `loaded=True, modelLoaded=True, modelStatus='loaded'`. Zero engine warnings during normal operation.
- **Design rationale** in `memory/phase4_2026_05_23.md`: UMA-first (Metal RHI uses `MTLResourceStorageModeShared` by default); direct Metal use first (QtQuick3D dispatches to Metal RHI without intermediate OpenGL emulation); no premature abstraction (build on QtQuick3D primitives, not a custom scene engine). The Mac-native viewer is the path forward; Qt3D Scene3D is deprecated by Qt and not maintainable on macOS 26.5+.

### Fixed (2026-05-23 Phase 3 — recurring user-reported issues)
- **`WARNING:root:Expression '{ScenePreview_1.output}': 'output'` eliminated across 10 templates.** Root cause: `ScenePreview` is a 2026-upstream Meshroom node with no published Python descriptor — every template referencing it fell back to a `CompatibilityNode` stub with untyped attributes, so any downstream `{ScenePreview_1.output}` expression failed to resolve. Fix: wrote a native Mac descriptor at `meshroom-mac/nodes/aliceVision/ScenePreview.py` — pure-Python Meshroom node (no CLI binary, no Rosetta, ARM64-native via meshroom-venv Python 3.13) that aggregates `cameras + model + undistortedImages + masks` into a single output folder via symlinks + a `scene_preview.json` manifest. Includes the `pointCloudParams` group (particleSize + particleColor) that `nodalCameraTracking*` templates require. After this fix, all 10 affected templates (`cameraTracking`, `cameraTrackingDepth`, `cameraTrackingLegacy`, `cameraTrackingRoma`, `cameraTrackingWithoutCalibration`, `cameraTrackingWithoutCalibrationLegacy`, `nodalCameraTracking`, `nodalCameraTrackingWithoutCalibration`, `photogrammetryAndCameraTracking`, `photogrammetryAndCameraTrackingLegacy`) deserialize `ScenePreview_1` as a real typed `Node`, not a `CompatibilityNode`. A future Metal-based 3D viewer can read `scene_preview.json` to render the preview — current node just stages the assets.
- **Top toolbar fix v2** — Phase 1's `implicitHeight: 36` on the bare `RowLayout` was insufficient: on macOS 26.5 + Qt 6.11.1 + Fusion, the inner `MenuBar` collapses to height 0 (native menu-bar routing), which forces the RowLayout to ignore the implicit-height hint for its children. Replaced the bare `RowLayout` header with a `ToolBar { RowLayout { anchors.fill: parent ... } }` (matches the existing `footer: ToolBar { ... }` pattern in the same file at line 1200). Also added `Layout.fillHeight: true` to the `homeButton` and `Layout.fillHeight: true` + `Layout.preferredHeight: 36` to the `MenuBar` so it participates in the ToolBar's vertical sizing. Added an explicit `background: Rectangle` to the ToolBar so the bar is visually anchored even if a child collapses. qmllint reports zero new warnings in the changed lines.

### Added (2026-05-23 Phase 2)
- **macOS `.app` packaging scaffold** at `scripts/package_macos_app.sh` + `scripts/macos_app/{Info.plist,launcher.sh}.tmpl`. Produces a runnable `Meshroom.app` bundle (~2.6 GB) wrapping the 12 `aliceVision_*` binaries, `default.metallib`, BiRefNet `.mlpackage` models, Meshroom Python package, plugins, and `meshroom-venv`. Verified end-to-end on `photogrammetryDraft` against Monstree mini3 — `Meshroom.app/Contents/MacOS/meshroom` produced `texturedMesh.obj` using only bundled binaries + bundled venv. Dev-mode bundle: still falls back to `/opt/homebrew/lib` for Homebrew dylibs. Phase 3 (dylib bundling via `dylibbundler`, codesign, notarize, DMG) deferred — see `memory/macos_app_packaging.md`.
- **Trackpad pinch-zoom** in `GraphEditor.qml` via Qt 6 `PinchHandler` — coexists with the existing `MouseArea` and applies incremental scale around the pinch centroid (mirrors wheel-zoom math).
- **`PIPELINE_DATASET` env var** in `tests/python/test_pipeline_integration.py` to switch the E2E suite between `mini3` (default), `mini6`, and `full` (41 photos, battle test). Example: `RUN_PIPELINE_E2E=1 PIPELINE_DATASET=full python -m pytest tests/python -k test_pipeline_runs_end_to_end`.
- `photogrammetryObject` and `photogrammetryObjectTurntable` promoted to `EXPECTED_COVERED` after end-to-end verification on Monstree mini3 (each: 13 nodes, 3 BiRefNet masks ~400 ms/view, textured mesh).
- **Monstree-full battle test passing**: `photogrammetryDraft` (~10 min) and `photogrammetryLegacy` (~17 min) both produce a textured mesh on the 41-photo dataset. Subprocess timeout in `_run_pipeline` now scales as `max(900, 60 * n_photos)` instead of the previous hard-coded 900 s — Legacy on full would have just missed the old ceiling.

### Fixed (2026-05-23 bug-sweep — user-reported UI + pipeline regressions)
- **HDR pipelines no longer crash on load.** `hdrFusion.mg`, `panoramaHdr.mg`, and `panoramaFisheyeHdr.mg` were failing during deserialization with `ImportError: cannot import name 'hdr' from 'pyalicevision'`. Root cause: `src/python_shim/pyalicevision/__init__.py` advertised sentinel submodules (`hdr`, `sfmData`, `sfmDataIO`) in its docstring but never created them. Fix: implemented `pyalicevision/hdr.py` (pure-Python stubs for `vectorli`, `LuminanceInfo`, `estimateGroups`), `pyalicevision/sfmData.py` (`ExposureSetting` dataclass + `__getattr__` raising clean `NotImplementedError`), `pyalicevision/sfmDataIO.py` (same). `update()` template-load hooks complete cleanly; bracket detection is disabled (`nbBrackets=0`) until C++ bindings are built; compute-time failures from `SfMFilter` / `SfMRigApplying` / `SfMPoseFlattening` / `SfMChecking` now surface a clear NotImplementedError instead of an opaque ImportError at template load.
- **Top toolbar missing on macOS** — `meshroom-mac/meshroom/ui/qml/Application.qml:667` had `header: RowLayout { ... }` without an explicit `implicitHeight`. On Qt 6.11.1 + Fusion style + macOS 26.5, the Page's header delegate sized the slot to 0 px, clipping all toolbar contents (home button, MenuBar, process buttons). Fix: pinned `implicitHeight: 36` matching the tallest child (MaterialToolButton at `font.pointSize: 18`).
- **Graph editor pan/zoom UX** — three changes to `meshroom-mac/meshroom/ui/qml/GraphEditor/GraphEditor.qml`:
  - Bumped wheel zoom factor `1.15` → `1.2` to match Viewer2D's snappier feel (the user's "intuitive" reference UX).
  - Added trackpad-pan detection — wheel events with `pixelDelta` populated and no Ctrl modifier now pan the canvas instead of zooming, matching macOS-native expectations.
  - Added double-click-on-empty-canvas → `fit()` (the existing F-key shortcut is now discoverable without keyboard knowledge).
- **SAM-era node types purged from 13 templates.** The Windows-version templates referenced `ImageSegmentationSam3` / `ImageSegmentationBox` / `ImageDetectionPrompt` (SAM/GroundingDINO segmentation nodes the Mac port doesn't have). The "Compatibility issues detected" dialog shown on every camera-tracking / object-photogrammetry template load was the symptom. Fix: replaced SAM segmentation nodes with `SegmentationBiRefNet` (the CoreML BiRefNet plugin) in all 13 templates (`cameraTracking.mg`, `cameraTrackingDepth.mg`, `cameraTrackingLegacy.mg`, `cameraTrackingRoma.mg`, `cameraTrackingWithoutCalibration.mg`, `cameraTrackingWithoutCalibrationLegacy.mg`, `nodalCameraTracking.mg`, `nodalCameraTrackingWithoutCalibration.mg`, `photogrammetryAndCameraTracking.mg`, `photogrammetryAndCameraTrackingLegacy.mg`, `photogrammetryObject.mg`, `photogrammetryObjectTurntable.mg`, `photogrammetryObjectTwoSides.mg`); dropped `ImageDetectionPrompt` nodes (BiRefNet is salient-object, no prompts needed); rewired all downstream `masksFolders` / `masks` references; set `maskFormat: "exr"` on the 7 templates whose consumers (`ExportImages`, `FeatureExtraction`) expected `.exr` masks. **Known caveats**: 3 legacy templates dropped `maskInvert: true` from the old SAM-Box config — BiRefNet outputs foreground masks directly with no invert option; if downstream visual results look polarity-flipped on those pipelines, that's the reason.
- Updated `tests/python/test_pipeline_integration.py::EXPECTED_UNCOVERED` — `photogrammetryObject` and `photogrammetryObjectTurntable` flipped from uncovered → covered after the SAM purge; left in transitional state until an actual E2E run validates them on Monstree.

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

### Removed
- **rembg + ONNX Runtime backend removed from `SegmentationBiRefNet` (2026-05-23).**
  The plugin is now CoreML-only — it loads pre-converted `.mlpackage`
  models from `ai-models/` via `coremltools` at `MLComputeUnits.cpuAndGPU`.
  Reason: on Apple Silicon the rembg path ran 6–10 s per frame on the
  CPU EP and ~233 s per frame on the CoreML EP `CPUAndGPU` mode
  (Metal command-buffer thrashing in ORT for swin-v1 graphs). The
  mlpackage path is 350–980 ms per frame depending on variant — 10–35×
  faster with 5× less memory. The following were deleted:
  `plugins/ai-segmentation/python/segmentation/convert_to_coreml.py`,
  `plugins/ai-segmentation/scripts/` (entire dir, including
  `download_models.py`). `session.py` was rewritten as a thin
  `coremltools.models.MLModel` wrapper. The node descriptor's
  `alphaMatting` and `outputResolution` parameters were dropped
  (rembg-specific and mlpackage-shape-locked, respectively). The
  `modelVariant` choices are now `birefnet-lite` (default) +
  `birefnet-general`; `birefnet-dis` was rembg-only. `ONNX_FORCE_CPU=1`
  is no longer honoured; `AV_AI_MODELS_DIR` is the new env override for
  the models path.
- **`meshroom-native/` (Swift / SwiftUI) decommissioned (2026-05-23).** The
  native SwiftUI Meshroom prototype that shipped in 0.1.0 has been retired.
  Reason: maintaining two graph-editor frontends (SwiftUI + upstream PySide6)
  diverged from upstream's evolving node schema and doubled review surface.
  Going forward there is **one** Meshroom frontend on macOS — upstream's
  PySide6 Meshroom run via `meshroom-mac/` + `scripts/run_meshroom.sh`. The
  Swift sources, the `swift test` suite, the `meshroom-native/` documentation,
  and all references in `docs/`, `README`, `CONTRIBUTING`, `SECURITY` have
  been removed in this release.

### Added — AI segmentation: BiRefNet CoreML models
- **Pre-converted BiRefNet CoreML models shipped at `ai-models/`** —
  `BiRefNet_lite.mlpackage` (90 MB, swin_v1_t backbone) and
  `BiRefNet.mlpackage` (447 MB, swin_v1_l backbone), both FP16 mlprogram
  at fixed 1024×1024 input, targeting macOS 14+.
- **Reproducible CoreML conversion pipeline** at `models/` with conversion
  scripts (`models/convert/`), HF checkpoint loaders, deformable-conv
  `grid_sample` patch, validation + benchmark harness. Full how-to at
  `ai-models/README.md`.
- **Production note** at `models/production_note.md` capturing the
  ANE-not-viable finding (BiRefNet's `ASPPDeformable` decoder uses
  deformable conv v2, which CoreML lowers via `grid_sample`, which the ANE
  compiler cannot plan — `MLComputeUnits.cpuAndGPU` is mandatory; `.all`
  hangs in `com.apple.anef.p3`). Production GPU latency: ~350 ms/frame
  (lite) / ~980 ms/frame (general) at 1024² on Apple Silicon.

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
