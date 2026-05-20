# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIT LICENSE (overlay code) with third-party-notices section covering upstream MPL-2.0 / Boost-1.0 / Apache-2.0 components.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub issue / PR templates.
- `docs/dev/codebase-navigation.md` — one-page repo-layout guide.

### Changed
- `.gitignore` consolidated to single root file (removed redundancies, added explicit reasons per section).

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
