# Meshroom integration

The reference Meshroom GUI is **PyQt5/QML + Python**. On macOS it doesn't
work out-of-the-box because (a) the `setupEnvironment()` host-OS branch
doesn't include Darwin, (b) `nvidia-smi` is Linux-only, (c) `/proc/cgroup`
doesn't exist on macOS, and (d) `readlink -f` is GNU-only.

This repo addresses those four issues with **out-of-tree patches** (the
upstream Meshroom and AliceVision trees are never modified on disk).

## What ships

```
patches/
├── meshroom/                          (Meshroom-app patches)
│   ├── 01-init-darwin-libpath.patch
│   ├── 02-stats-darwin-gpu.patch
│   ├── 03-cgroup-darwin-sysctl.patch
│   ├── 04-startsh-readlink-portable.patch
│   └── README.md
└── alicevision-meshroom/              (Meshroom node-descriptor patches)
    ├── 01-sfm-output-ply-sfm-not-abc.patch
    ├── 02-meshing-output-ply-not-abc.patch
    └── README.md

meshroom-mac/                          (working copy of Meshroom with patches applied)
meshroom-venv/                         (Python venv: PySide6, psutil, pyseq, ...)
scripts/run_meshroom.sh                (wrapper: sets env, exec's your command)
```

## Meshroom patches (4)

Upstream target: `alicevision/Meshroom` branch `develop` at commit
`0ab90c0b36df0c9773483a25aa95d91c5696f9d0` (`v2025.1.5-114-g0ab90c0b`).

| # | Patch | What it does | Upstream-PR ready? |
|---|---|---|---|
| 01 | `01-init-darwin-libpath.patch` | Adds a `darwin` branch in `setupEnvironment()` setting `DYLD_FALLBACK_LIBRARY_PATH` from `ALICEVISION_LIBPATH`, mirroring the Linux `LD_LIBRARY_PATH` setup. | Yes |
| 02 | `02-stats-darwin-gpu.patch` | Replaces `nvidia-smi` with a `system_profiler SPDisplaysDataType` probe for `gpuName` and `gpuMemoryTotal`. Live per-process GPU stats deferred. | Needs maintainer review |
| 03 | `03-cgroup-darwin-sysctl.patch` | Short-circuits the Linux `/proc/<pid>/cgroup` probes on Darwin and uses `sysctl hw.memsize` / `sysctl hw.ncpu` instead. | Yes |
| 04 | `04-startsh-readlink-portable.patch` | Replaces GNU-only `readlink -f` in `start.sh` with `python3 -c 'os.path.realpath(...)'`. | Yes |

All four pass `git apply --check` against the pinned upstream commit. Patches
1, 3, 4 are clean platform-gated additive changes; patch 2 has two design
choices the upstream maintainer may want to revisit (cost of
`system_profiler` on first launch, gpuMemoryTotal as `"8 GB"` string vs
integer MiB).

### Apply the Meshroom patches

```bash
cd /path/to/your/meshroom-fork
git apply /path/to/alicevision-for-mac/patches/meshroom/*.patch
```

For day-to-day use the repository ships a pre-applied working copy at
`meshroom-mac/` — `scripts/run_meshroom.sh` points `MESHROOM_NODES_PATH` and
`PYTHONPATH` at it.

## AliceVision node-descriptor patches (2)

Upstream target: `alicevision/AliceVision` (the node-descriptors at
`AliceVision/meshroom/aliceVision/*.py`).

| # | Patch | What it does | Upstream-PR ready? |
|---|---|---|---|
| 01 | `01-sfm-output-ply-sfm-not-abc.patch` | `StructureFromMotion.py`: default `output` → `sfm.sfm` (SfMData JSON), `interFileExtension` → `.ply`. Avoids the *"AliceVision is built without Alembic support"* fatal. | No — hard-coded fallback. Cleaner upstream would query AliceVision capabilities. |
| 02 | `02-meshing-output-ply-not-abc.patch` | `Meshing.py`: dense point cloud default → `densePointCloud.ply` (downstream Texturing reads PLY transparently). | No, same reason. |

Both patches are applied in the working copy at `meshroom-mac/nodes/`. They
are required because the macOS build sets `ALICEVISION_HAVE_ALEMBIC=0` (we
haven't built Alembic on Apple Silicon yet).

## Python venv

The Meshroom runtime is Python. The repo provides a pre-seeded venv at
`meshroom-venv/` containing PySide6, psutil, pyseq, and the other Meshroom
runtime deps.

=== "Use the bundled venv"

    ```bash
    source meshroom-venv/bin/activate
    pip list   # PySide6, psutil, pyseq, ...
    ```

=== "Recreate from scratch"

    ```bash
    python3 -m venv meshroom-venv
    source meshroom-venv/bin/activate
    pip install PySide6 psutil pyseq
    # plus any other deps Meshroom 2026.1.0 needs at runtime
    ```

## Running a job

The canonical end-user invocation is `meshroom_batch` through
`scripts/run_meshroom.sh`:

```bash
./scripts/run_meshroom.sh python meshroom-mac/bin/meshroom_batch \
    -i dataset_monstree/mini3 \
    -o /tmp/monstree-out \
    -p photogrammetryLegacy
```

The script sets:

```bash
ALICEVISION_ROOT=$ROOT/build/alicevision_root
ALICEVISION_BIN_PATH=$ROOT/build
ALICEVISION_SENSOR_DB=$ROOT/build/alicevision_root/share/aliceVision/cameraSensors.db
ALICEVISION_OCIO=$ROOT/build/alicevision_root/share/aliceVision/config.ocio
ALICEVISION_LIBPATH=/opt/homebrew/lib
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
MESHROOM_NODES_PATH=$ROOT/meshroom-mac/nodes
MESHROOM_PIPELINE_TEMPLATES_PATH=$ROOT/meshroom-mac/nodes
PYTHONPATH=$ROOT/meshroom-mac:$ROOT/src/python_shim
```

`src/python_shim` carries a tiny pure-Python `pyalicevision` shim used by
some Meshroom node descriptors for sizing decisions (the `DynamicViewsSize`
fix for `prepareDenseScene` lives here — see
[Developer → Segmentation pipeline](../dev/segmentation-pipeline.md) for
the related node-descriptor patterns).

## On the (retired) SwiftUI prototype

An earlier 0.1.0 release shipped a parallel native SwiftUI Meshroom
prototype at `meshroom-native/`. It was retired on 2026-05-23 to consolidate
work on the upstream-compatible PySide6 Meshroom (this page). Going forward
there is one Meshroom frontend on macOS.
