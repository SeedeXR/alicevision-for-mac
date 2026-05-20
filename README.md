# alicevision-for-mac

Native Apple Silicon Metal port of [AliceVision](https://alicevision.org/)
(C++ photogrammetry library) and [Meshroom](https://github.com/alicevision/Meshroom)
(node-graph reconstruction frontend).

**Status** (plain text — CI wiring deferred):

- `ctest -j8`: **37/37** pass reliably
- `swift test`: **151/151** pass (native SwiftUI)
- `release`: `build/release/alicevision-for-mac-0.1.0-arm64.tar.gz` (15 MB) + `-dSYM.tar.gz` (161 MB)
- `version`: 0.1.0
- `target`: macOS 14+ on Apple Silicon (arm64)
- `license`: [MIT](LICENSE) (overlay code; third-party retain their licenses)
- `docs`: `mkdocs serve` → http://127.0.0.1:8000, or browse `docs/` directly
- `agents`: see [llms.txt](llms.txt) for an AI-readable project digest

---

## What's in the box

- **12 pipeline CLI binaries** covering the full photogrammetry workflow,
  raw photos → textured 3D mesh:
  `aliceVision_cameraInit` → `featureExtraction` → `imageMatching` →
  `featureMatching` → `incrementalSfM` → `prepareDenseScene` →
  `depthMapEstimation` → `depthMapFiltering` → `meshing` → `meshFiltering` →
  `texturing`, plus `importMiddlebury`.
- **`default.metallib`** — all GPU kernels compiled to a single Metal
  shader archive (~41 MSL kernels), staged next to each binary.
- **Native SwiftUI Meshroom** at `meshroom-native/` — alternative to
  Python/Qt Meshroom: graph viewer, drag-to-move nodes, parameter
  editing, drag-to-connect pins, type-checked connections, node-creation
  palette, scheduler with chunked DepthMap execution + per-chunk
  progress, Meshroom-compatible UID hashing for project interop.
- **Python Meshroom integration** at `meshroom-mac/` + `scripts/run_meshroom.sh`
  — runs upstream's PySide6 Meshroom against the Apple Silicon binaries
  (verified end-to-end on the Monstree dataset; produces textured 3D mesh).
- **Release tarballs** at `build/release/` — small binary + separate
  debug-symbol bundle (canonical Apple split).
- **Comprehensive docs site** at `docs/` (MkDocs Material, 19 pages, 16
  Mermaid diagrams, dark mode, search).

---

## Quick start

```bash
# 1. Install Homebrew deps
brew install alembic assimp boost ceres-solver eigen geogram imath \
             lemon libomp nanoflann onnxruntime open-mesh openexr \
             openimageio python@3.13

# 2. Set up the upstream symlink (one-time)
git clone https://github.com/alicevision/AliceVision.git ../alicevision-windows/AliceVision
ln -s ../alicevision-windows/AliceVision upstream

# 3. Build
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DAV_BUILD_UPSTREAM=ON \
    -DAV_BUILD_UPSTREAM_DEPTHMAP=ON
cmake --build build
```

End-to-end smoke test on the Monstree mini dataset (3 photos):

```bash
bash scripts/run_meshroom.sh mini3   # → meshroom-mac-out/result/texturedMesh.obj
```

Native SwiftUI app:

```bash
cd meshroom-native && swift run MeshroomNativeApp
```

---

## Repository layout

```
src/                  Overlay code (MIT): av::gpu, depth_map_metal, MSL shaders
tests/                C++ ctest suite (37 tests)
meshroom-native/      Native SwiftUI app (151 tests)
docs/                 MkDocs Material doc source
cmake/                Build helpers (Metal.cmake, UpstreamShim.cmake, shims/)
patches/              Patches against upstream Meshroom (not modifying upstream/)
scripts/              run_meshroom.sh + diagnostic helpers
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
- Tests must pass: `ctest -j8: 37/37` + `swift test: 151/151`.
- For perf changes: include before/after numbers via `AV_PROFILE_ADAPTER=ON`.
- For numerical-kernel changes: validate against a CPU-FP64 reference.
- Don't modify `upstream/` (it's a read-only symlink).
- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) (Contributor Covenant 2.1).

---

## Known issues

- **Notarization deferred** — requires Apple Developer ID Application
  credentials. Ad-hoc-signed tarball ships today for dev distribution;
  Gatekeeper will prompt on first launch.
- **Release tarball NOT fully vendored** — consumers need the Homebrew
  packages listed in `Formula/alicevision-for-mac.rb`.
- **LiDAR pipeline disabled** (`ALICEVISION_BUILD_LIDAR=OFF`); requires
  upstream's `E57Format` library which is not vendored.
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
(`src/`, `tests/`, `meshroom-native/`, `cmake/`, `docs/`, `patches/`,
`scripts/`).

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
