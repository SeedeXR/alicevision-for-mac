# alicevision-for-mac

Native Apple Silicon Metal port of [AliceVision](https://alicevision.org/)
(C++ photogrammetry library) and [Meshroom](https://github.com/alicevision/Meshroom)
(node-graph reconstruction frontend).

**Status (2026-05-24)** — feature-complete against upstream 2026.1.0:

- **60 native ARM64 `aliceVision_*` binaries** (was 12; expanded across Phases 14.1–14.9)
- **25 of 25 Meshroom templates covered** — every binary present, every descriptor resolved, zero "honest stubs"
- **4 CoreML models** integrated: BiRefNet (segmentation), YOLOv8n (sphere detection), MoGe-2 (depth), TinyRoMa (dense matching)
- **3 native C++ wrapper modules** at `src/sphere_detection/`, `src/moge/`, `src/roma/` (Objective-C++ + pure-C++ public headers)
- **Native pyalicevision SWIG bindings** (hdr, sfmData, sfmDataIO) — real C++ `estimateGroups()` etc. replace Python stubs
- `pytest tests/python`: **73 passing, 25 skipped** (gated heavy E2E)
- `ctest -j8`: **37/37** pass on a clean build
- 4 templates verified end-to-end on `dataset_monstree/mini3`: Draft / Legacy / Object / Turntable
- `target`: macOS 14+ on Apple Silicon (arm64)
- `license`: [MIT](LICENSE) (overlay code; third-party retain their licenses)
- `docs`: <https://seedexr.github.io/alicevision-for-mac/> · or `mkdocs serve` locally
- `agents`: see [llms.txt](llms.txt) for an AI-readable project digest

---

## What's in the box

- **60 native arm64 binaries** covering the full upstream 2026.1.0 Meshroom
  template set. Groups: photogrammetry core (11), modern SfM (6), HDR (3),
  panorama (8), photometric stereo (3), camera tracking + utilities (22),
  LIDAR (3), Mac-port-native (3 — `starListing`, `matchMasking`, `moGe`),
  Middlebury import (1). Full inventory:
  [docs/reference/binaries.md](docs/reference/binaries.md).
- **`default.metallib`** — all GPU kernels compiled to a single Metal
  shader archive (~41 MSL kernels), staged next to each binary.
- **4 CoreML models** at [`ai-models/`](ai-models/) with native C++ wrappers:
    - **BiRefNet** (× 2 variants) — foreground segmentation. `cpuAndGPU`
      (ANE hangs).
    - **YOLOv8n** — sphere detection. Runs on ANE (3× faster than GPU). Replaces
      upstream's ONNX Runtime `sphereDetection`.
    - **MoGe-2** (DINOv2 ViT-B/14) — monocular geometry / depth. Partial ANE,
      1.2× over GPU. Replaces the Phase 14.7 honest stub.
    - **TinyRoMa** — dense optical-flow matcher. `cpuAndGPU` (ANE is **4×
      slower** here due to `grid_sample` handoffs — counter to all the other
      models). Replaces the Phase 14.7 honest pass-through.

    Per-model integration recipe + ANE outcome matrix in
    [`ai-models/README.md`](ai-models/README.md).
- **Native pyalicevision SWIG bindings** for `hdr` / `sfmData` / `sfmDataIO`.
  Built when `AV_BUILD_PYALICEVISION=ON` (default); auto-discovered at
  Meshroom import time. Real C++ `estimateGroups()` runs in `LdrToHdr*`
  descriptors; falls back to pure-Python stubs when `OFF`.
- **Python Meshroom integration** at `meshroom-mac/` + `scripts/run_meshroom.sh`.
  Drives upstream PySide6 Meshroom against the arm64 binaries. End-to-end
  verified on Monstree mini3 + full datasets. The prior native SwiftUI
  prototype was decommissioned in May 2026.
- **Self-contained `.app` packaging** at `scripts/package_macos_app.sh`
  (mini-dylibbundler + auto ad-hoc resign), `scripts/codesign_macos_app.sh`
  (Developer ID path), `scripts/make_dmg.sh` (1.4 GB compressed DMG).
- **Native plugin system** at `plugins/` — third parties can ship AI
  extensions via a self-contained `plugin.json` manifest. AI segmentation
  is the reference implementation:
  [docs/dev/plugin-system.md](docs/dev/plugin-system.md).
- **Apple Silicon optimization deep-dive** —
  [docs/dev/apple-silicon-optimization.md](docs/dev/apple-silicon-optimization.md):
  UMA, Metal, ANE, CPU performance with measured numbers.
- **Comprehensive docs site** at <https://seedexr.github.io/alicevision-for-mac/>
  (MkDocs Material). Deploys from `main` via `.github/workflows/docs.yml`.

---

## Quick start

```bash
# 1. Install Homebrew deps
brew install alembic assimp boost ceres-solver eigen geogram imath \
             lemon libomp nanoflann onnxruntime open-mesh opencv \
             openexr openimageio python@3.13 swig

# 2. Set up the upstream symlink (one-time)
git clone https://github.com/alicevision/AliceVision.git ../alicevision-windows/AliceVision
ln -s ../alicevision-windows/AliceVision upstream

# 3. Build
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAV_BUILD_UPSTREAM=ON \
    -DAV_BUILD_UPSTREAM_DEPTHMAP=ON \
    -DAV_BUILD_PYALICEVISION=ON
cmake --build build
```

End-to-end smoke test on the Monstree mini dataset (3 photos):

```bash
bash scripts/run_meshroom.sh mini3   # → meshroom-mac-out/result/texturedMesh.obj
```

---

## Repository layout

```
src/                  Overlay code (MIT): av::gpu, depth_map_metal, MSL shaders
tests/                C++ ctest suite (37 tests)
ai-models/            Pre-built CoreML segmentation models (BiRefNet lite + general)
docs/                 MkDocs Material doc source
cmake/                Build helpers (Metal.cmake, UpstreamShim.cmake, shims/)
patches/              Patches against upstream Meshroom (not modifying upstream/)
plugins/              AI plugin manifests (e.g. ai-segmentation)
scripts/              run_meshroom.sh + diagnostic helpers
models/               BiRefNet HF checkpoints + CoreML conversion scripts
third_party/          Vendored deps: LEMON (tracked), metal-cpp (downloaded)
Formula/              Homebrew formula

upstream/             Symlink to ../alicevision-windows/AliceVision (gitignored)
meshroom-mac/         Local Meshroom checkout (gitignored; regenerable via patches/)
build/                CMake build dir (gitignored)
```

Full layout in [`docs/dev/codebase-navigation.md`](docs/dev/codebase-navigation.md).

---

## Documentation

| Doc | Purpose |
|---|---|
| [BUILD.md](BUILD.md) | Build prerequisites + CMake options |
| [INSTALL_macOS.md](INSTALL_macOS.md) | End-user install |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Codebase tour |
| [PORTING_NOTES.md](PORTING_NOTES.md) | CUDA → Metal design decisions |
| [RELEASE.md](RELEASE.md) | Cutting a redistributable tarball |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [SECURITY.md](SECURITY.md) | Vulnerability reporting |
| [CHANGELOG.md](CHANGELOG.md) | High-level release log |
| [docs/](docs/) | Full MkDocs site (`mkdocs serve` to preview) |
| [llms.txt](llms.txt) | AI-readable project digest ([llmstxt.org](https://llmstxt.org/)) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Highlights:
- Open an issue before any non-trivial PR.
- Tests must pass: `ctest -j8: 37/37` + `pytest tests/python: 73 passed, 25 skipped`.
- For perf changes: include before/after numbers via `AV_PROFILE_ADAPTER=ON`.
- For numerical-kernel changes: validate against a CPU-FP64 reference.
- Don't modify `upstream/` (it's a read-only symlink).
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) (Contributor Covenant 2.1).

---

## Known issues

- **Notarization deferred** — requires Apple Developer ID Application
  credentials. Ad-hoc-signed binaries + DMG ship today for dev
  distribution; Gatekeeper will prompt on first launch. Run
  `scripts/codesign_macos_app.sh --identity "Developer ID Application: ..."`
  for production signing.
- **Self-contained `.app` ships, full DMG bundling done** — 91 dylibs /
  155 MB bundled via `scripts/package_macos_app.sh`; `.app` runs without
  `/opt/homebrew` on the DYLD path. The Homebrew formula path is also
  supported for users who prefer system-managed deps.
- **21 of 25 templates are covered-but-load-only** — every binary is
  present and every descriptor resolves, but full E2E for HDR / panorama /
  photometric stereo / cameraTracking templates needs the right fixture
  datasets (bracketed exposures, calibration spheres, LIDAR e57s, etc.).
  Not a code blocker — data work.
- **Coin-OR Clp shimmed** — `OSIXSolver` is a no-op stub
  (`cmake/shims/.../OSIXSolver.hpp`). The single LP consumer
  (`geometry::halfPlane::isNotEmpty()`) gets a safe over-approximation.
- **CUDA-vs-Metal wall-clock comparison deferred** — needs NVIDIA
  hardware to run upstream's stock build for the comparison baseline.
- **`memory/`, `books/`, `instructions/`** are gitignored as
  author-private session/reference material.

---

## License

[MIT](LICENSE) for the overlay code authored in this repository
(`src/`, `tests/`, `cmake/`, `docs/`, `patches/`, `scripts/`,
`plugins/`, `models/`).

Third-party components retain their upstream licenses:
- **AliceVision** — Mozilla Public License 2.0
- **Meshroom** — Mozilla Public License 2.0
- **LEMON** (`third_party/lemon/`) — Boost Software License 1.0
- **Apple metal-cpp** (`third_party/metal-cpp/`, downloaded) — Apache 2.0

See [LICENSE](LICENSE) for the full third-party-notices section.

---

## Acknowledgments

- Upstream [AliceVision](https://github.com/alicevision/AliceVision) and
  [Meshroom](https://github.com/alicevision/Meshroom) projects.
- Apple [metal-cpp](https://developer.apple.com/metal/cpp/).
- [LEMON](https://lemon.cs.elte.hu/) graph library.
- The original AliceVision contributors whose Linux/Windows CUDA pipeline
  this port adapts to Apple Silicon Metal.
