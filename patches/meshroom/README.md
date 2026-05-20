# Meshroom Darwin patches — Phase 11

Out-of-tree patches that add macOS / Apple Silicon support to the
upstream Meshroom Python repository without modifying the upstream
tree.

## Upstream target

- Repository: `alicevision/Meshroom`
- Branch checked out locally: `develop`
- Commit:   `0ab90c0b36df0c9773483a25aa95d91c5696f9d0`
- `git describe`: `v2025.1.5-114-g0ab90c0b`
- Meshroom `__version__` reported by `meshroom/__init__.py`: `2026.1.0+develop`

The local read-only clone lives at
`../alicevision-windows/Meshroom/` (sibling of `alicevision-for-mac/`).

## How to apply

The patches are in standard `git apply` unified-diff format with
`a/` and `b/` prefixes. To apply them to a fresh checkout:

```sh
cd ../alicevision-windows/Meshroom
git apply ../../alicevision-for-mac/patches/meshroom/*.patch
```

For our local read-only clone, apply against a working copy elsewhere
(e.g. a fork you maintain):

```sh
cd /path/to/your/meshroom-fork
git apply /path/to/alicevision-for-mac/patches/meshroom/*.patch
```

All four patches were validated with `git apply --check` against
the pinned upstream commit above. If a future upstream revision
changes the surrounding context, the hunks may need to be refreshed
— rerun the validation procedure documented in the project memory.

## What each patch does

1. **`01-init-darwin-libpath.patch`** — `meshroom/__init__.py`. Adds an
   `elif sys.platform == "darwin"` branch in `setupEnvironment()` that
   sets `DYLD_FALLBACK_LIBRARY_PATH` from `ALICEVISION_LIBPATH`,
   mirroring the existing `LD_LIBRARY_PATH` setup on Linux. Safety net
   only — our AliceVision build embeds `@rpath` install names so this
   normally never triggers.

2. **`02-stats-darwin-gpu.patch`** — `meshroom/core/stats.py`. Replaces
   the `nvidia-smi` GPU stats path on Darwin with a one-shot
   `system_profiler SPDisplaysDataType` probe that fills in `gpuName`
   and `gpuMemoryTotal`. Live per-process GPU memory / utilization is
   intentionally deferred (would need `pyobjc-framework-Metal`); the
   trade-off is documented inline in the patch.

3. **`03-cgroup-darwin-sysctl.patch`** — `meshroom/core/cgroup.py`.
   Short-circuits the Linux `/proc/<pid>/cgroup` + `/sys/fs/cgroup`
   probes when `sys.platform == "darwin"` and returns the true
   hardware values via `sysctl hw.memsize` and `sysctl hw.ncpu`.

4. **`04-startsh-readlink-portable.patch`** — `start.sh`. Replaces the
   GNU-only `readlink -f` invocation with a `python3 -c
   'os.path.realpath(...)'` one-liner. `python3` is already a hard
   prereq of the very next line, so this adds no new dependency and
   works identically on Linux and macOS.

## Upstream-PR readiness

| Patch | Status | Notes |
| --- | --- | --- |
| 01-init-darwin-libpath | **Ready for upstream PR** | Trivial 7-line additive branch; no behavioural change on Windows or Linux. |
| 03-cgroup-darwin-sysctl | **Ready for upstream PR** | Pure platform-gated short-circuit; preserves the original "-1 = unlimited" contract on failure. |
| 04-startsh-readlink-portable | **Ready for upstream PR** | Strict portability fix; the new form is functionally identical to `readlink -f` on Linux and works on macOS. |
| 02-stats-darwin-gpu | **Needs maintainer review** | Two design choices the upstream maintainer may want to revisit: (a) `system_profiler` is slow (~1-2 s) and called from `initOnFirstTime`; (b) `gpuMemoryTotal` is stored as a human-readable string ("8 GB") rather than the bare MiB integer the nvidia path uses. A future revision could switch to a `pyobjc-framework-Metal` shim for true live stats. |

## Validation log

- `git apply --check` against
  `alicevision-windows/Meshroom@0ab90c0b`: **all 4 patches OK**.
- `git status` of the upstream tree after generating these patches:
  **clean** (no upstream modifications).
- No C/C++ files touched; the Phase 10 `ctest` baseline of 33/33 is
  unaffected.
