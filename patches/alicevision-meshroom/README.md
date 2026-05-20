# AliceVision Meshroom-node-descriptor Darwin patches — Phase 11

Out-of-tree patches against the `meshroom/aliceVision/*.py` node
descriptors that ship inside the upstream AliceVision repository.
These mirror the `patches/meshroom/` layout but target the *node-
descriptor* layer (which lives in the AliceVision repo, not the
Meshroom repo).

## Upstream target

The node descriptors live at:
`alicevision-windows/AliceVision/meshroom/aliceVision/*.py`
(commit pinned at the AliceVision snapshot we ported in Phase 1–10).

## How to apply

The patches are in standard `git apply` unified-diff format with
`a/meshroom/aliceVision/` and `b/meshroom/aliceVision/` prefixes,
i.e. anchored at the AliceVision repo root. To apply to a fresh
AliceVision checkout:

```sh
cd /path/to/your/alicevision-fork
git apply /path/to/alicevision-for-mac/patches/alicevision-meshroom/*.patch
```

For our local read-only `alicevision-windows/` clone, the working
copy used by Phase 11 lives at
`alicevision-for-mac/meshroom-mac/nodes/aliceVision/` — the patches
are already applied there. The `meshroom-mac` runner script points
`MESHROOM_NODES_PATH` at that local copy.

## What each patch does

1. **`01-sfm-output-ply-sfm-not-abc.patch`** —
   `meshroom/aliceVision/StructureFromMotion.py`. Changes the default
   `output` value from `sfm.abc` to `sfm.sfm` (SfMData JSON), and the
   `interFileExtension` default from `.abc` to `.ply`. The mac
   AliceVision build sets `ALICEVISION_HAVE_ALEMBIC=0` (Alembic is a
   heavy upstream dep we have not yet built on Apple Silicon), so the
   binary would otherwise abort with: *"Cannot save the ABC file [...],
   AliceVision is built without Alembic support."* SfMData JSON
   carries the same information the rest of the pipeline consumes; no
   information loss.

2. **`02-meshing-output-ply-not-abc.patch`** —
   `meshroom/aliceVision/Meshing.py`. Same rationale: change the
   dense point-cloud output from `densePointCloud.abc` to
   `densePointCloud.ply`. The downstream Texturing node reads PLY
   transparently.

## Upstream-PR readiness

Both patches are NOT ready for upstream as-is — they hard-code the
fallback. A cleaner upstream submission would be to gate the default
on a query of the AliceVision build capabilities (e.g., look for a
`aliceVision_haveAlembic` marker file or call `aliceVision_cameraInit
--features` and parse the report). That work belongs in a future
"Meshroom side: capability discovery" task.

For our purposes (running the Meshroom pipeline on Apple Silicon
today) the static change is correct: our build will never have
Alembic until Phase 13+.

## Validation log

Validated end-to-end on `dataset_monstree/mini3/`:
all 12 pipeline nodes complete; Texturing produces a textured mesh
(4790 vertices, 9449 faces, 256 KB EXR texture) matching the
structural shape of our manual `build/monstree_work/texturing/`
output.
