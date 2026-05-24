# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (2026-05-23 Phase 14.3 — 9 more aliceVision_* binaries; panorama templates unlocked)
- **9 new pipeline binaries** built on Apple Silicon, bringing the total from 21 to **30**:
  - **8 panorama binaries**: `aliceVision_panoramaInit`, `aliceVision_panoramaPrepareImages`, `aliceVision_panoramaWarping`, `aliceVision_panoramaEstimation`, `aliceVision_panoramaCompositing`, `aliceVision_panoramaMerging`, `aliceVision_panoramaSeams`, `aliceVision_panoramaPostProcessing`. Required adding `add_subdirectory(upstream/src/aliceVision/panorama)` for the `aliceVision_panorama` static lib (14 source files; PRIVATE_LINKS system+image+camera — all built). `panoramaEstimation` additionally needed `aliceVision_mesh` for the transitive geogram include path (same pattern as `sfmBootstrapping` in Phase 14.1).
  - **Bonus binary `aliceVision_sfmTransform`** (from `software/utils/`, not `pipeline/`) — the one remaining binary blocking `panoramaHdr` and `panoramaFisheyeHdr` coverage once the panorama sublib was in place. Links sfm+sfmData+sfmDataIO+track+geometry+mesh.
- **`panoramaHdr` + `panoramaFisheyeHdr` templates promoted to `EXPECTED_COVERED`** — all binaries they reference are now built; templates load cleanly + binaries pass `-h` smoke test. Full E2E gated on Phase 13 (C++ pyalicevision pybind11 bindings) for the same reason as `hdrFusion`. Coverage matrix: **7 / 25 templates** (was 5 / 25 after 14.2).
- New `test_template_loads_cleanly[panoramaHdr]` and `[panoramaFisheyeHdr]` always-on tests; both pass.

### Added (2026-05-24 Phase 14.9 — TinyRoMa CoreML port replaces Phase 14.7 matchMasking stub; ZERO honest stubs remaining)
- **New `src/roma/` module** — Mac-native CoreML wrapper around the user's `ai-models/tiny_roma_v1_480x640.mlpackage` (TinyRoMa: XFeat encoder + 4-block matcher heads, 2.84 M params, 5.5 MB FP16). Mirrors `src/sphere_detection/` + `src/moge/`:
  - `include/av/roma/CoreMLRomaMatcher.hpp` — pure-C++ public header. `RomaMatch` struct (coarse + fine flow + certainty + original image dims), opaque `CoreMLRomaMatcher` class with `match(imagePathA, imagePathB) -> RomaMatch`.
  - `src/CoreMLRomaMatcher.mm` — Objective-C++ implementation. Two-input pipeline: load → BGR-to-RGB → resize to 480×640 → float32 [0,1] → NCHW `MLMultiArray` per image → `predictionFromFeatures` with `{im_A, im_B}` → extract 4 outputs (coarse_flow, coarse_certainty, fine_flow, fine_certainty) via stride-aware reads.
  - `aliceVision_shim/main_matchMasking.cpp` — replaces the Phase 14.7 honest pass-through at `src/native_binaries/main_matchMasking.cpp` (deleted). Reads SfMData + image-pairs list, runs TinyRoMa per pair, writes per-pair `<viewIdA>_<viewIdB>_{coarse,fine}_{flow,certainty}.exr` to the output folders. Applies sigmoid to certainty (model emits unnormalized logits). Optional `--masksFolder <path>` with `<viewId>.png` per view zeros certainty where either view's mask is invalid (the "MatchMasking" semantics, now actually masking real match data instead of just forwarding empties).
- **CRITICAL load setting**: the wrapper uses `MLComputeUnitsCPUAndGPU` — NOT `MLComputeUnitsAll`. Per the user's benchmark (5-iter mean, subprocess-isolated), ANE is **4× SLOWER than CPU** on TinyRoMa because the decoder contains two `grid_sample` ops that each force a CPU↔ANE memory handoff. For a small 2.84 M-param model, the fixed per-handoff cost dominates the compute savings. `cpuAndGPU` gives ~12 ms / pair at 480×640 on M-series. This is the **fourth distinct ANE-outcome pattern** in the repo's models (BiRefNet hangs / YOLOv8 3× faster on ANE / MoGe 1.2× partial-ANE / TinyRoMa 4× slower) — documented at the top of `ai-models/README.md`.
- **`cameraTrackingRoma` template is now REAL E2E-capable** — verified end-to-end on a Monstree photo pair. Output: 4 EXRs per pair (~10–70 KB each), sigmoid'd certainty in [0, 1], flow values mostly within the normalized [-1, 1] range as documented. PyTorch ↔ CoreML diff is ~2.3% rel max on flow (driven by `argmax` quantization sensitivity in `pos_embed`; documented in production note).
- **`ai-models/README.md` rewritten** to cover all 4 CoreML models in one document: BiRefNet (×2), YOLOv8n, MoGe-2, TinyRoMa. Per-model load settings, input/output contracts, native-binary mapping, runtime discovery rules, and the "how to add a new model" recipe. The ANE outcome matrix is featured up top — "compiles to ANE" ≠ "runs faster on ANE", and TinyRoMa is the canary for `grid_sample` models that look ANE-friendly but aren't.
- **ZERO "honest stubs" remain in the build.** Phase 14.7's three Mac-port-native binaries are all now real:
  - `aliceVision_starListing` — real algorithmic (was real in 14.7).
  - `aliceVision_moGe` — real CoreML (Phase 14.8 retired stub).
  - `aliceVision_matchMasking` — real CoreML TinyRoMa (THIS PHASE retired stub).
  The 25/25 coverage matrix is no longer asterisked: every template's binary chain runs real inference / real algorithms.
- **Regression coverage**: `tests/python/test_roma_coreml.py` — 4 tests (3 always-on: binary built, model present, help advertises CoreML Roma and NOT the Phase 14.7 "pass-through" language; 1 gated `RUN_ROMA_COREML=1` running full inference on a real photo pair + asserting all 4 EXRs are present + sized > 1 KB + valid OpenEXR magic bytes + log line `pair(s) matched` is in the output). All pass.
- **Test suite: 76 → 79 passed** (+3 always-on Roma tests). No regressions.

### Added (2026-05-24 Phase 14.8 — CoreML MoGe port replaces Phase 14.7 stub; cameraTrackingDepth is now real)
- **New `src/moge/` module** — Mac-native CoreML wrapper around the user's `ai-models/moge2_504x672_t1728.mlpackage` (MoGe-2 with DINOv2 ViT-B/14 backbone). Mirrors `src/sphere_detection/` layout:
  - `include/av/moge/CoreMLMoGeRunner.hpp` — pure-C++ public header. `MoGeResult` struct (depth in meters + RGB normals + uint8 validity mask + metric_scale) and an opaque `CoreMLMoGeRunner` class.
  - `src/CoreMLMoGeRunner.mm` — Objective-C++ implementation. Reads image with OpenCV, resizes to 672×504, normalizes to [0,1], packs into a `[1, 3, 504, 672]` `MLMultiArray` NCHW, calls `predictionFromFeatures`, extracts `points` Z-component multiplied by `metric_scale` to produce per-pixel depth in meters, plus the `normal` and `mask` outputs.
  - `aliceVision_shim/main_moGe.cpp` — replaces the Phase 14.7 honest stub at `src/native_binaries/main_moGe.cpp` (deleted). Same CLI surface; auto-discovers the .mlpackage via `--modelPath` override, `$ALICEVISION_MOGE_MLPACKAGE` env var, `$ALICEVISION_ROOT/ai-models/`, or walking up from cwd.
- **`cameraTrackingDepth` template is now REAL E2E-capable** — the moGe binary produces realistic depth maps (verified: 0–19.8 m range, mean 3.4 m, 99% pixels valid) from a real photo. The output 504×672 EXRs are ~210 KB each (PIZ-compressed), vs the Phase 14.7 stub's ~1 KB constant-1.0m EXRs.
- **Uses `MLComputeUnits.all`** — the system schedules the network's many ops across ANE / GPU / CPU. User's measurement (2026-05-24): **~228 ms ANE vs ~384 ms CPU** per view on M-series; the model partially runs on ANE because not every op has an ANE kernel in CoreML 9.0. ANE-vs-GPU gain is only ~1.2×; ANE-vs-CPU gain is the meaningful one (1.7×).
- **Loads-time schema validation**: the wrapper constructor verifies the model exposes the expected input (`image`) and outputs (`points`, `normal`, `mask`, `metric_scale`). A mis-converted .mlpackage surfaces a clear `MoGe: model missing output '<name>'` error rather than decoding garbage floats later.
- **Regression coverage**: `tests/python/test_moge_coreml.py` — 4 tests (3 always-on: binary built, model present, help advertises CoreML not the stub language; 1 gated `RUN_MOGE_COREML=1` running full inference + EXR magic-byte check + file-size sanity > 10 KB so a regression-to-stub is caught). All pass.
- **CMake**: `add_subdirectory(src/moge)` next to `add_subdirectory(src/sphere_detection)`. The `aliceVision_moGe` binary block links `av_moge` (provides `av::moge::CoreMLMoGeRunner`).
- **Stub language is gone**: the binary no longer logs `MAC-PORT STUB — emitting constant-depth placeholders`; help text no longer advertises "stub" wording. Tests assert against the old stub strings to prevent regressions.

### Added (2026-05-24 Phase 14.7 — 3 Mac-port-native binaries; **25 / 25 templates covered**)
- **3 new binaries** at `src/native_binaries/` — written in this repo (no upstream provenance) for 2026.1.0 features that upstream defers to "external Python pipelines" (no C++ source in our snapshot). Total binary count: 57 → **60**:
  - `aliceVision_starListing` (2.2 MB) — REAL ALGORITHMIC. Reads input SfMData, sorts views by viewId, emits next-K star-topology image pairs to `imagePairsList` (one `<viewId1> <viewId2>` per line). Pass-through-copies the input SfMData to the output. Default `radiusKeyFrames=5`. Deterministic.
  - `aliceVision_matchMasking` (2.2 MB) — HONEST PASS-THROUGH. Copies inputs (`imagePairsList`, `warpFolder`, `certaintyFolder`, `inputSfMData`) verbatim to the corresponding output paths; missing input folders → treated as empty + output folders created. Logs `matchMasking: Mac-port pass-through (Roma pipeline virtualized)`.
  - `aliceVision_moGe` (2.2 MB) — HONEST STUB. For each view emits a 64×48 constant-1.0m depth EXR (and optional constant-up-vector normals EXR) named `<viewId>_depth.exr` / `<viewId>_normals.exr` to match what `DepthMapTracksInjecting` consumes. Logs `moGe: MAC-PORT STUB — emitting constant-depth placeholders; real MoGe inference requires CoreML model conversion`.
- **Coverage matrix: 23/25 → 25/25.** Promoted `cameraTrackingDepth` + `cameraTrackingRoma` to `EXPECTED_COVERED`. `EXPECTED_UNCOVERED` is now an empty set (with a docstring documenting the honesty caveats — moGe/matchMasking are stubs, real E2E for those templates needs MoGe + Roma CoreML model conversions in a future ML phase).
- **Test suite: 71 → 73 passed** (2 new auto-generated `test_template_loads_cleanly[<name>]` tests).
- **Why we wrote our own**: upstream's `ImageDescriber_Roma.hpp` comments these features as "extraction performed externally (e.g. via dedicated Python pipeline)". Two valid paths were available — ship our own native binaries or stop at 23/25 with documentation. Shipping minimal-but-correct binaries lets Meshroom's graph editor + descriptor loader wire against something, and gives users a working "load + smoke" experience even when the real ML pipelines aren't bundled on Mac. The honest-stub approach preserves user trust by loudly logging what's placeholder vs real.

### Added (2026-05-24 Phase 14.5 + 14.6 — 23 new binaries, Alembic enabled, 14 new templates covered)
- **23 new aliceVision_* binaries built**, bringing total from 34 → **57**:
  - Phase 14.5 (21 binaries via parallel agent in one sweep): applyCalibration, checkerboardDetection, colorCheckerCorrection, colorCheckerDetection, convertDistortion, convertSfMFormat, depthmapTracksInjecting, distortionCalibration, exportDistortion, exportImages, geometricFilterEstimating, imageProcessing (`_bin` target with OUTPUT_NAME = `aliceVision_imageProcessing`), intrinsicsTransforming, keyframeSelection, lidarDecimating, lidarMerging, lidarMeshing, nodalSfM, sfmColorizing, sfmMerge, tracksMerging.
  - Phase 14.6 (2 binaries unblocked by Alembic): exportAlembic (2.2 MB), exportAnimatedCamera (2.3 MB).
- **3 new aliceVision_* sublibs** via `add_subdirectory(...)`: `aliceVision_calibration`, `aliceVision_keyframe`, `aliceVision_imageProcessing`. Each PUBLIC_LINKS OpenCV — the agent attached `${OpenCV_INCLUDE_DIRS}` + `${OpenCV_LIBS}` post-add_subdirectory because the upstream sublib's OpenCV include path is gated behind `ALICEVISION_HAVE_OPENCV` (which we keep OFF for other reasons).
- **Alembic enabled** in the existing `aliceVision_sfmDataIO` sublib by setting `ALICEVISION_HAVE_ALEMBIC=1` BEFORE the `add_subdirectory(upstream/src/aliceVision/sfmDataIO)` call. The upstream sublib's own `if(ALICEVISION_HAVE_ALEMBIC)` blocks then fold AlembicExporter/Importer sources + link Homebrew's `Alembic::Alembic` IMPORTED target automatically — no source-list patching needed. `cmake/upstream-config.hpp.in` already had the `#define ALICEVISION_HAVE_ALEMBIC() @ALICEVISION_HAVE_ALEMBIC@` line.
- **One patched upstream header** at `${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/image/conversionOpenCV.hpp` — strips the `#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_OPENCV)` guard so `image::imageRGBAToCvMatBGR` / `cvMatBGRToImageRGBA` inline helpers are reachable from the colorChecker binaries (which need them but don't want to flip the global flag).
- **5 missing Meshroom descriptors handled** (parallel agent):
  - `SfMBootStrapping.py` + `SfMExpanding.py` — pre-existed on disk as `Sfm{Bootstrapping,Expanding}.py` but the .mg templates reference `SfM{BootStrapping,Expanding}`. macOS HFS+/APFS case-insensitivity hid this: `Read('SfM...py')` succeeds but `Path.glob('*.py')` returns the literal on-disk name, so the coverage script's map missed. Fixed via two-step `git mv` (Sfm → Sfm_tmp → SfM).
  - `MoGe.py`, `MatchMasking.py`, `StarListing.py` — new stubs created at `meshroom-mac/nodes/aliceVision/`. Route to `aliceVision_{moGe,matchMasking,starListing}` (not yet built — those are 2026 ML-model-port work).
  - `scripts/pipeline_coverage.py` — added `"ScenePreview"` to `PURE_PYTHON_NODES` so the discovery logic recognizes the Phase 3 pure-Python `desc.Node` subclass (no `commandLine = aliceVision_*` line to parse).
- **Coverage matrix: 9/25 → 23/25**. Newly covered templates promoted to `EXPECTED_COVERED` in `tests/python/test_pipeline_integration.py`:
  - Phase 14.5: colorCalibration, distortionCalibration, lidarMeshing, photogrammetry (modern SfM!), photogrammetryObjectTwoSides, rawImageConversion.
  - Phase 14.6: cameraTracking, cameraTrackingLegacy, cameraTrackingWithoutCalibration, cameraTrackingWithoutCalibrationLegacy, nodalCameraTracking, nodalCameraTrackingWithoutCalibration, photogrammetryAndCameraTracking, photogrammetryAndCameraTrackingLegacy.
- **Test suite: 57 → 71 passed** (auto-generated `test_template_loads_cleanly` for the 14 newly-covered templates). 25 skipped (gated E2E + RUN_APP_BUNDLE tests).
- **Still gated (2/25)**: `cameraTrackingDepth` (needs `aliceVision_moGe` — MoGe mono-depth model CoreML port), `cameraTrackingRoma` (needs `aliceVision_matchMasking` + `aliceVision_starListing` — Roma matcher family). Both are real ML+C++ port work in the same shape as the BiRefNet (Phase 1) and sphereDetection (Phase 14.4b) ports.
- **Parallelization win**: this entire phase shipped in a single user turn by dispatching two `general-purpose` Agents on non-overlapping file scopes (one for CMakeLists.txt + binaries, one for `meshroom-mac/nodes/`), then a third for Alembic. The case-insensitive-filesystem and Path-C-via-flag-before-add_subdirectory gotchas are documented in `memory/macos_case_insensitive_filesystem.md` and `memory/phase14_5_6_2026_05_24.md`.

### Added (2026-05-24 Phase 13 — native C++ pyalicevision SWIG bindings)
- **3 SWIG modules built** at `build/pyalicevision_native/`: `_hdr.so` + `hdr.py`, `_sfmData.so` + `sfmData.py`, `_sfmDataIO.so` + `sfmDataIO.py` (~6 MB total). Built from upstream's existing `.i` interface files in `upstream/src/aliceVision/{hdr,sfmData,sfmDataIO,camera,...}/`. **Note**: upstream uses SWIG, not pybind11 (a misnomer in this repo's earlier phase plans — the bindings are nonetheless functional and identical from the Python caller's POV).
- **New CMake option `AV_BUILD_PYALICEVISION`** (default ON). Wires upstream's `ALICEVISION_BUILD_SWIG_BINDING` flag through `find_package(SWIG)` + `find_package(Python3 ... Development.Module NumPy)`. The previously-stubbed `alicevision_swig_add_library()` shim now does real work: `swig_add_library(... TYPE MODULE LANGUAGE python)` + `-undefined dynamic_lookup` link flag for Python symbol resolution at load time on macOS.
- **CRITICAL macOS-specific gotcha**: SWIG's `.i` files use `#ifdef LINUXPLATFORM` to choose between `size_t = unsigned long` (UNIX) and `size_t = unsigned long long` (Windows). Upstream's own `Helpers.cmake` already conditionally defines `LINUXPLATFORM` on `UNIX`, and the macros are nominally Linux-named but mean "UNIX-class size_t" — we must pass `-DLINUXPLATFORM` to SWIG on macOS too, otherwise the generated `wrap.cxx` redefines `size_t` as `unsigned long long` and the typedef collision with darwin's `__darwin_size_t` (which is `unsigned long`) blows up compilation. Our shim mirrors upstream's pattern.
- **Native binding auto-discovery in `pyalicevision/__init__.py`**: at import time we discover `<repo>/build/pyalicevision_native/` (or honor `PYALICEVISION_NATIVE_DIR` env override), and **prepend that path to the package's `__path__`** so `from pyalicevision import hdr` resolves to the SWIG-generated `hdr.py` (which imports `_hdr.so`) instead of our pure-Python stub. When the native dir is absent (the package is relocated, AV_BUILD_PYALICEVISION=OFF), the stub `hdr.py` / `sfmData.py` / `sfmDataIO.py` next to `__init__.py` take over. Two-mode design preserves the load-only fallback that earlier phases shipped.
- **Real bracket detection unlocked**: `LdrToHdrSampling` / `LdrToHdrCalibration` / `LdrToHdrMerge` descriptors' `update()` hooks now run real C++ `aliceVision::hdr::estimateGroups()` against their input viewpoints instead of returning `[]`. A bracketed-exposure dataset will now produce a non-zero `nbBrackets` and proceed to merge.
- **Real `SfMData` / `sfmDataIO`** for `SfMFilter`, `SfMPoseFlattening`, and any other descriptor that bypasses the binary layer.
- **Regression coverage**: `tests/python/test_pyalicevision_native.py` (5 tests: 4 gated `skip_if_no_native`, 1 unconditional fallback-mode test that relocates the package to a temp dir + sets `PYALICEVISION_NATIVE_DIR` to a dead path and verifies the stub loads instead). All 5 pass. Full suite: **57 passed / 39 skipped** (up from 52/39).

### Added (2026-05-24 Phase 14.4b — CoreML port of aliceVision_sphereDetection)
- **New `src/sphere_detection/` module** — Mac-native replacement for upstream's ONNX-Runtime-based sphereDetection sublib. Two layers:
  - `av::sphere::CoreMLSphereDetector` (pure-C++ public API + Objective-C++ implementation backed by `MLModel` + `MLDictionaryFeatureProvider`). Runs `ai-models/yolov8n.mlpackage` (the user's Vision-style YOLOv8 export, NMS baked into the graph) on `MLComputeUnits.all` so the system schedules on ANE / GPU / CPU.
  - `aliceVision::sphereDetection::*` drop-in shim — same namespace + same public API as upstream's `sphereDetection.hpp`, except `Ort::Session&` is swapped for `CoreMLSphereDetector&`. `modelExplore()` (ONNX I/O dump) removed; everything else kept verbatim including `fillShapeTree()` + both `writeManualSphereJSON()` overloads.
- **New binary `aliceVision_sphereDetection`** (1.98 MB) — drop-in CLI replacement for upstream's. Same flag surface; auto-detect branch instantiates `CoreMLSphereDetector` instead of `Ort::Session`. Total binary count: 33 → **34**. **No ONNX Runtime dependency in the Mac port.**
- **`photometricStereo` + `multi-viewPhotometricStereo` templates promoted to `EXPECTED_COVERED`** — coverage matrix is now **9 / 25**. End-to-end CoreML inference smoke-tested against a real photo (model loaded, scored a detection, wrote upstream-format JSON).
- **CMake gotchas worked around**: (1) The OpenCV stitching umbrella header (`opencv2/opencv.hpp`) defines identifiers that conflict with macOS SDK Objective-C++ symbols (`NO`, etc.); the `.mm` file uses focused submodules (`core.hpp`, `imgcodecs.hpp`, `imgproc.hpp`) instead. (2) The binary's `target_include_directories(... PRIVATE)` list put `upstream/src` ahead of the shim, so the upstream header (with its ONNX include) won include resolution; fixed with a separate `BEFORE PRIVATE` include of the shim path that prepends it.
- **New regression suite** `tests/python/test_sphere_detection_coreml.py` — 4 tests (3 always-on: binary built, model file present, help advertises CoreML not ONNX; 1 gated E2E behind `RUN_SPHERE_DETECTION=1` that drives full inference on a real photo and validates the output JSON schema).

### Added (2026-05-24 Phase 14.4a — photometric stereo sublibs + 2 binaries; sphereDetection CoreML port deferred to 14.4b)
- **OpenCV pulled in via `find_package(OpenCV CONFIG REQUIRED)`** — Homebrew opencv@4. Required by the two new sublibs below.
- **2 new sublibs**: `aliceVision_photometricStereo` (`upstream/src/aliceVision/photometricStereo`, 3 sources) + `aliceVision_lightingEstimation` (`upstream/src/aliceVision/lightingEstimation`, 4 sources). Both PUBLIC_LINKS `${OpenCV_LIBS}` + already-built core libs.
- **3 new binaries** (total 30 → 33):
  - `aliceVision_lightingCalibration` — calibration from spheres.
  - `aliceVision_photometricStereo` — normal/albedo reconstruction. **CMake gotcha**: upstream uses the same name for the binary AND the static lib, which CMake forbids in a unified project. Worked around via target name `aliceVision_photometricStereo_bin` + `set_target_properties(... OUTPUT_NAME aliceVision_photometricStereo)` — the shipped filename matches what Meshroom's `PhotometricStereo` node descriptor expects.
  - **Bonus** `aliceVision_sfmTransfer` (from `software/utils/`) — unblocks `multi-viewPhotometricStereo` (was missing only this + sphereDetection).
- **Phase 14.4b queued**: a CoreML port of `aliceVision_sphereDetection` (replacing the upstream ONNX Runtime dep with the user's already-converted ai-models/yolov8n.mlpackage running on ANE). Will promote `photometricStereo` + `multi-viewPhotometricStereo` to `EXPECTED_COVERED`. Design + port surface documented in `memory/phase14_4a_2026_05_24.md`.

### Added (2026-05-23 Phase 14.1 + 14.2 — 9 new aliceVision_* binaries)
- **9 new pipeline binaries** built on Apple Silicon, bringing the total from 12 to **21**:
  - **6 modern-SfM + utility binaries (Phase 14.1)**: `aliceVision_sfmBootstrapping`, `aliceVision_sfmExpanding`, `aliceVision_relativePoseEstimating`, `aliceVision_sfmTriangulation`, `aliceVision_tracksBuilding`, `aliceVision_meshDecimate`. All use only sublibs we already built (sfm, sfmData, sfmDataIO, feature, track, dataio, mesh, mvsUtils) plus OpenMesh for `meshDecimate`. Zero new Homebrew dependencies.
  - **3 HDR binaries (Phase 14.2)**: `aliceVision_LdrToHdrSampling`, `aliceVision_LdrToHdrCalibration`, `aliceVision_LdrToHdrMerge`. Required adding `add_subdirectory(upstream/src/aliceVision/hdr)` for the `aliceVision_hdr` static lib. Unblocks the **`hdrFusion` template's coverage** — promoted from `EXPECTED_UNCOVERED` to `EXPECTED_COVERED` (full E2E still gated on real C++ pyalicevision bindings for bracket auto-detection; templates load + binaries launch cleanly).
- **CMake patches**:
  - `cmake/Metal.cmake` — tolerate missing Metal Toolchain (Xcode 26 removed `metallib` and ships the toolchain as opt-in `xcodebuild -downloadComponent MetalToolchain`). Without the toolchain installed, `av_compile_metal_library` declares an empty custom target and reuses the pre-built `default.metallib`. **Pure-C++ binary additions no longer require the Metal Toolchain.**
  - `upstream-patched/aliceVision/sfm/pipeline/relativePoses.hpp` — added `inline` to the two `tag_invoke<ReconstructedPair>` free functions. Upstream defines them in the header WITHOUT `inline` — every TU that includes it (transitively via `PairsScoring.hpp`) emits its own definition → duplicate-symbol link errors when `main_sfmBootstrapping.cpp` links against `libaliceVision_sfm.a` (which already contains the same defs via `PairsScoring.cpp`). Patch mirrors the existing `upstream-patched/` pattern (resection_kernel.cpp, l1.cpp).
  - `aliceVision_sfm` + `aliceVision_sfm_bundle` sublibs now have `${CMAKE_BINARY_DIR}/upstream-patched` prepended to their include paths via `target_include_directories(... BEFORE PRIVATE ...)` so the patched relativePoses.hpp wins resolution.
- **New `E2E_TEMPLATES` constant** in `tests/python/test_pipeline_integration.py` — decouples "binaries built / template loads" (`EXPECTED_COVERED`, used by the coverage matrix + load-only smoke tests) from "actually runs end-to-end on a real dataset" (`E2E_TEMPLATES`, used by the heavy E2E parametrize). hdrFusion lives in the former but not the latter.

### Fixed (2026-05-23 Phase 8 — bundle leak + regression coverage)
- **CRITICAL: `scripts/package_macos_app.sh` bundler missed 7+ dylib references per binary.** Root cause: the `while IFS= read -r dep; do ... done` loop variable `dep` was NOT auto-local in bash; recursive `_bundle_target` calls (one level deep for each transitive Homebrew dep) had their own `while read -r dep` that **clobbered the outer caller's `dep`**. After recursion returned, the outer `install_name_tool -change "$dep" ...` call ran with an empty/wrong `dep` argument and silently no-op'd (rc=0, no stderr). 7 of cameraInit's 21 deps leaked through in production. Fix: declared `local dep dep_name dest` at the top of `_bundle_target`. Verified: `aliceVision_cameraInit` now has **0 `/opt/homebrew/` references** in its load commands.
- **`tests/python/conftest.py` now invokes `meshroom.core.initNodes()`** at session start (with `MESHROOM_NODES_PATH` set first). Without this, every template-deserialize test silently created `CompatibilityNode` stubs everywhere — descriptor-load regressions (the exact ScenePreview bug from Phase 3) would have passed every "graph loaded" assertion. The Phase 3 fix went undetected by the test suite for a full session because of this.

### Added (2026-05-23 Phase 8 — automated coverage for the new viewer + bundle)
- **`tests/python/test_scenepreview_node.py`** (9 tests) — pins the Phase 3 ScenePreview descriptor: (a) module imports + `__version__ == "2.0"`, (b) registers in pluginManager, (c) input/output schema matches what templates wire, (d) loading 4 representative ScenePreview-using templates produces typed `Node` instances (NOT CompatibilityNodes), with **zero** `Expression '{ScenePreview_1.output}': 'output'` warnings — pins the original user-reported bug from `bugs.pdf` to a hard test, (e) `processChunk` on synthetic inputs produces the documented manifest + symlinks.
- **`tests/python/test_viewer3d_helpers.py`** (11 tests) — pins the Phase 4-6 viewer-3D Python helpers: `ScenePreviewLoader` manifest parsing (3 paths: real / missing / PLY-only-routes-to-empty-modelPath), `PointCloudGeometry` PLY parser (ASCII + binary_little_endian) + normal sentinel + missing-file error path, `FrustumGeometry` rebuild on param change + no-rebuild-on-same-value optimization.
- **`tests/python/test_app_bundle.py`** (11 tests, all under `RUN_APP_BUNDLE=1` gate) — pins Phase 7 + the bundler-bug fix: (a) bundle layout, (b) launcher executable + env-var refs, (c) **zero `/opt/homebrew/` leaks across binaries** (the exact regression we just hit; would have caught it pre-distribution), (d) zero leaks in bundled dylibs (recursive bundler completeness), (e) 4 representative binaries launch with empty DYLD path + no missing-dylib errors, (f) bundled venv python + PySide6 + coremltools all importable. **The bundler bug was found BY this test before reaching users — exactly the regression coverage requested.**
- Updated `memory/pipeline_coverage.md` to reflect the 4-of-25 covered state (Draft/Legacy/Object/ObjectTurntable). `photogrammetryObjectTwoSides` documented as "needs dual-capture fixture" (template loads + runs but mini3 isn't the right shape for it).

### Added (2026-05-23 Phase 7 — self-contained `.app` shipping)
- **`scripts/package_macos_app.sh`** now bundles all Homebrew dylibs into `Meshroom.app/Contents/Resources/lib/` and rewrites every `aliceVision_*` binary's `install_name` to `@executable_path/../lib/<name>` via `install_name_tool`. Self-contained mini-`dylibbundler` in bash (no `brew install dylibbundler` dep) walks each binary's `otool -L` output transitively + dedup via a visited-set. Smoke test on a fresh build: **91 dylibs (155 MB) bundled across 12 binaries**; `aliceVision_meshing` launches without `/opt/homebrew/lib` on the DYLD path. Plus an automatic ad-hoc resign step (`install_name_tool` invalidates Mach-O signatures → macOS Gatekeeper SIGKILLs at launch (rc=137) unless re-signed); 103 Mach-Os ad-hoc signed.
- **`scripts/codesign_macos_app.sh`** — Apple Developer ID codesigning pipeline. Signs in dependency order (innermost dylibs → binaries → frameworks → .app wrapper, per Apple's codesign requirements). Includes a synthesized entitlements plist with `allow-unsigned-executable-memory` + `allow-jit` + `disable-library-validation` for PySide6's embedded Python interpreter. Defaults to ad-hoc (`--sign -`); pass `--identity "Developer ID Application: ..."` for production. Verifies via `codesign --verify --deep --strict` + `spctl --assess`. Notarization commands are documented inline (`xcrun notarytool submit + stapler staple`).
- **`scripts/make_dmg.sh`** — produces a compressed UDZO DMG ready for distribution. Stages the .app + a `/Applications` symlink (drag-to-install convention) + a `README.txt` with launch instructions. First production run: **1.4 GB DMG from the 2.7 GB .app** (UDZO compression).

### Added (2026-05-23 Phase 6 — viewer polish)
- **Click-to-select camera frustum** — frustum Models are `pickable: true` + tagged with `cameraIndex`. A `TapHandler` on the View3D ray-casts via `view.pick(x, y)`, walks up the parent chain, and calls `CameraFrustumGroup.selectByIndex(idx)`. Selected frustum changes color (orange → cyan). New `cameraSelected(idx, pose)` signal flows up to `MetalScenePreview`, which flies the orbit camera to the picked viewpoint via the new `flyToPose(pose)` helper.
- **PLY normal support** — `PointCloudGeometry` interleaved layout grew 28 B → 40 B per vertex to carry per-vertex normals. Parser extracts `nx`/`ny`/`nz` when present; emits sentinel `(0, 0, 1)` when absent (the AliceVision dense-cloud case) so a future lighting-enabled material renders correctly. Added `NormalSemantic` alongside Position + Color. Verified on real Monstree `cloud_and_poses.ply` (7,433 points, 10-tuple verts).
- **Per-view BiRefNet mask overlay on frustums** — each frustum delegate wraps the wireframe Model + a textured Quad positioned at the camera's near plane, scaled to the intrinsics × near distance. Texture-mapped from `<masksPath>/<viewStem>_mask.png` (matches BiRefNet output naming). Blended at 0.55 opacity. Toggleable via `CameraFrustumGroup.showMasks`; defaults ON.

### Added (2026-05-23 Phase 5 — PLY point clouds + wireframe frustums + UX polish)
- **PLY point-cloud rendering** in `MetalScenePreview`. A new `PointCloudGeometry` (`meshroom-mac/meshroom/ui/components/pointCloudGeometry.py`) subclasses `QQuick3DGeometry`, parses PLY files (ASCII + binary_little_endian) with per-vertex RGB color, and uploads positions+colors interleaved into a single Qt RHI Metal shared-storage buffer. Verified end-to-end on real Monstree `cloud_and_poses.ply` (7,433 points) and `densePointCloud.ply` (7,810 points). Activates automatically when the ScenePreview manifest's model path resolves to a `.ply` (no GLTF/OBJ); routed via the empty-`modelPath` branch the Python `ScenePreviewLoader` was already returning.
- **True wireframe camera frustums** in `CameraFrustumGroup`. New `FrustumGeometry` (`meshroom-mac/meshroom/ui/components/frustumGeometry.py`) builds 12 line segments per frustum from `nearPlane × farPlane × fovYDegrees × aspectRatio` + a look-at stub from the origin to the near plane centre. One shared geometry, N draw calls per scene (transformed per-instance by the Model's worldTransform). Replaces the Phase 4 cube-marker placeholders. Pose conversion: `SfMData.poses[].transform.rotation` is a 3×3 world-to-camera matrix; we transpose to camera-to-world + convert to Euler XYZ degrees for QtQuick3D's `eulerRotation` property.
- **Real `fitCameraToBounds`** — reads `loadedNode.bounds` from QtQuick3D's RuntimeLoader for meshes, and `geometry.boundsMin/boundsMax` for the PLY path. Auto-scales the frustum near/far planes proportional to scene radius so per-view markers stay visible at any zoom.

### Fixed (2026-05-23 graph canvas pan UX)
- **Apple Magic Mouse swipe now pans the canvas.** Two latent bugs in the Phase 1 wheel handler made the Magic Mouse feel dead: (a) a `< 50` magnitude cap on `pixelDelta.y` (intended to filter pinch — but pinch routes through `PinchHandler`, not `onWheel`) rejected fast swipes; (b) horizontal-only swipes (`pixelDelta.x ≠ 0, angleDelta.y = 0`) fell into the zoom branch and zoomed OUT on every flick. Replaced with a cleaner decision tree in `GraphEditor.qml:onWheel`: any `pixelDelta` without `Ctrl` pans; `angleDelta` (or `Ctrl+pixelDelta.y`) zooms; horizontal `Ctrl+pixelDelta` is ignored. Works for traditional mouse wheel, trackpad two-finger swipe, AND Magic Mouse single-finger swipe — all three input devices land on the same code path.
- **LeftButton-drag now pans the canvas** (Figma/Miro/Excalidraw convention) so users can grab the canvas and move it to a different node without needing keyboard modifiers. Previous default required `MiddleButton` (rare on Mac trackpads) or `Alt+Left` (not discoverable). New mapping in `GraphEditor.qml:243-285`:
  - `LeftButton` + no modifier → **pan**
  - `LeftButton` + Shift → node-selection rectangle (additive)
  - `LeftButton` + Ctrl → node-selection rectangle (subtractive)
  - `LeftButton` + Alt → pan (kept for muscle memory)
  - `LeftButton` + Ctrl + Alt → edge-removal selection
  - `MiddleButton` → pan (legacy default kept)
- Cursor now shows `OpenHandCursor` over the empty canvas (telegraphing "drag me") and `ClosedHandCursor` while a pan is active. `CrossCursor` shown during selection-rectangle / edge-removal drags.

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
