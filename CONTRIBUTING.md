# Contributing to alicevision-for-mac

Thanks for considering a contribution. This project ports
[AliceVision](https://alicevision.org) photogrammetry + Meshroom from CUDA
to native Apple Silicon Metal. The codebase is mature (S0-S54):
12 pipeline binaries, 32 upstream modules, upstream-compatible Python
Meshroom integration, AI segmentation via BiRefNet CoreML, comprehensive
docs.

This guide tells you **how to get involved**, **what kinds of changes are
welcome**, and **how to land a pull request**.

---

## Quick links

- **Codebase navigation**: `docs/dev/codebase-navigation.md`
- **Build instructions**: `BUILD.md` or `docs/user/install.md`
- **Architecture tour**: `ARCHITECTURE.md` or `docs/dev/architecture.md`
- **Code of conduct**: `CODE_OF_CONDUCT.md`
- **Security policy**: `SECURITY.md`
- **Issue templates**: `.github/ISSUE_TEMPLATE/`
- **PR template**: `.github/PULL_REQUEST_TEMPLATE.md`

---

## Before you start

1. **Open an issue first** for anything non-trivial. We'd rather discuss
   design + scope before you spend a week on a PR we'd reject.

2. **Read `memory/philosophy.md`** if you're touching kernel code. It
   distills the design rules: native Metal preferred over compatibility
   hacks; no shallow compile-only fixes; document architectural
   decisions.

3. **Run the existing test suite** before changing anything. You should
   be at `ctest -j8: 37/37 pass` + `pytest tests/python: 11 passed, 1 skipped`.
   If you can't get there on `main`, **file a build-issue first** — don't
   assume your patch is the cause.

---

## What kinds of changes are welcome

### High-value
- **More MSL kernel ports**: gaps in `src/shaders/depth_map/` vs upstream's
  CUDA surface. Bring your own numerical-validation test.
- **Bug fixes**: every fix needs a regression test that fails before + passes after.
- **Documentation improvements**: `docs/`, `README.md`, code comments.
  Especially welcome: new "I tried X and it didn't work, here's how to
  unblock" troubleshooting entries.
- **Meshroom integration improvements**: `meshroom-mac/` patches /
  `plugins/` AI nodes. Keep changes compatible with upstream's PySide6
  Meshroom.
- **Pipeline binary additions**: more `aliceVision_*` binaries from
  `upstream/src/software/pipeline/`. Follow the pattern in
  `docs/dev/adding-a-binary.md`.

### Medium-value
- **Perf optimizations** for hot Metal kernels. Required: before/after
  numbers via `AV_PROFILE_ADAPTER=ON` on Monstree mini3 view 0 + ctest
  37/37 still green.
- **Test infra**: better profiling tooling, GPU frame-capture integration,
  automated regression detection.
- **Packaging**: Homebrew formula refinements, Developer-ID signing
  (you'll need your own credentials), notarization automation.

### Low-value / out-of-scope
- **CUDA support** — this repo is Apple-only by design.
- **x86_64 macOS** — out of scope for milestone 1 (per CMakeLists.txt:28).
- **Reformatting / "style only" PRs** — the existing style is fine.
- **Adding heavy dependencies** without discussion.

---

## Setting up a dev environment

```bash
# 1. Clone (assume you fork first, then clone your fork)
git clone https://github.com/<your-user>/alicevision-for-mac.git
cd alicevision-for-mac

# 2. Set up the upstream symlink (one-time)
# This is the read-only AliceVision reference clone, NOT modified.
git clone https://github.com/alicevision/AliceVision.git ../alicevision-windows/AliceVision
ln -s ../alicevision-windows/AliceVision upstream

# 3. Install Homebrew deps (see BUILD.md for the full list)
brew install alembic assimp boost ceres-solver eigen geogram imath \
             lemon libomp nanoflann onnxruntime open-mesh openexr \
             openimageio python@3.13

# 4. Configure + build
cmake -S . -B build -DAV_BUILD_UPSTREAM=ON -DAV_BUILD_UPSTREAM_DEPTHMAP=ON
cmake --build build

# 5. Run tests
cd build && ctest -j8       # should be 37/37
cd .. && python -m pytest tests/python   # 11 passed, 1 skipped
```

See `BUILD.md` for details, troubleshooting, and the Homebrew dep list.

---

## Development workflow

1. **Branch from `main`** with a descriptive name:
   ```
   git checkout -b kernel/port-feature-extraction-akaze
   git checkout -b fix/depth-map-alpha-mask-threshold
   git checkout -b docs/clarify-meshroom-integration
   ```

2. **Make focused commits**. One logical change per commit. Write commit
   messages that explain WHY, not just WHAT.

3. **Run tests after every meaningful change**:
   ```bash
   cd build && ninja && ctest -j8
   cd .. && python -m pytest tests/python
   ```

4. **For kernel work**, also run a profile:
   ```bash
   cmake -S .. -B . -DAV_PROFILE_ADAPTER=ON
   ninja aliceVision_depthMapEstimation
   ALICEVISION_ROOT=$PWD/alicevision_root ./aliceVision_depthMapEstimation \
       -i monstree_work/sfm.sfm \
       --imagesFolder monstree_work/dense/ \
       -o /tmp/dm_check --rangeStart 0 --rangeSize 1 2> /tmp/profile.txt
   tail -20 /tmp/profile.txt    # per-forwarder timing table
   ```
   Include before/after numbers in your PR description.

5. **Update docs** for any user-facing behavior change. Pages under
   `docs/` follow MkDocs Material conventions; preview locally with:
   ```bash
   source docs-venv/bin/activate
   mkdocs serve     # → http://127.0.0.1:8000
   ```

6. **Open a PR** against `main` using the template at
   `.github/PULL_REQUEST_TEMPLATE.md`. Fill in every section honestly.

---

## Code style

- **C++**: C++20, modern idioms. Match the surrounding file's style; we
  don't enforce clang-format mechanically.
- **Metal Shading Language**: keep kernel files focused. One concept per
  `.metal` file (e.g., `volume_kernels.metal`, `comp_ncc.metal`).
- **Python**: 3.13, type-hint at module boundaries, follow `meshroom-mac`
  upstream style. Avoid heavy deps without discussion.
- **CMake**: `target_*` commands over directory-scoped ones. Quote paths.
- **Comments**: explain WHY for non-obvious code. Don't restate WHAT.

---

## Testing requirements

PRs must include:
- Tests for new functionality (matches existing test patterns under
  `tests/` for C++ or `tests/python/` for Python/Meshroom).
- For numerical-kernel work: validation against a CPU-FP64 reference with
  documented numerical budget (see existing kernel tests for the pattern).
- For Meshroom node work: at least a pytest harness covering the node
  parameter shape + a smoke run on a mini dataset.

PRs that lower test coverage or introduce flaky tests will be asked to
fix that before review.

---

## Reviewing

Maintainers will look at:
1. Does the test suite pass on a fresh build?
2. Does the change match the stated scope (no scope creep)?
3. Is the architectural decision documented (mental-note-worthy lessons
   go into `memory/mindmap.md` Validated learnings)?
4. Numerical / perf claims backed by reproducible commands?
5. Documentation updated where user-facing?

---

## Licensing

This project is **MIT-licensed** (`LICENSE`). By contributing, you agree
your contributions are licensed under MIT and may be redistributed as
part of this project.

**Important**: third-party components retain their upstream licenses
(AliceVision is MPL-2.0; LEMON is Boost-1.0; Apple metal-cpp is
Apache-2.0). Don't paste code from non-MIT-compatible sources without
discussing first.

---

## Getting help

- **Build issues**: open an issue with the `bug` label + full `cmake`/`ninja`
  output + `brew list --formula | sort` output.
- **Conceptual / design questions**: open a `discussion` (preferred) or
  `feature_request` issue.
- **Security issues**: see `SECURITY.md` — do NOT open a public issue.

Thanks for contributing.
