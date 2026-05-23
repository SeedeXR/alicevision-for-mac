# AI segmentation (`SegmentationBiRefNet`)

End-user guide for the AI-powered foreground/background segmentation
node shipped with this port. The architecture deep-dive is at
[`dev/segmentation-pipeline.md`](../dev/segmentation-pipeline.md); this
page covers what an operator needs to run the node.

## What is segmentation?

In photogrammetry, segmentation produces a per-view binary mask that
labels each pixel as **foreground** (the object you want to reconstruct)
or **background** (everything else — sky, walls, turntable, capture rig).
Downstream nodes (`DepthMap`, `Meshing`, `Texturing`) consume those masks
to restrict reconstruction to the foreground region, which improves
geometric accuracy on the subject and avoids wasting compute on the
background.

Typical uses on this Mac port:

- **Object scans** — turntable captures of a single subject; mask out the
  table, backdrop, and floor.
- **Outdoor objects** — statue, vehicle, building façade; mask out sky,
  people, vegetation that moves between frames.
- **Background-only reconstruction** — invert the masks to reconstruct
  only the static environment.

`SegmentationBiRefNet` uses [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)
loaded as a pre-converted CoreML `.mlpackage` and dispatched on the
Apple Silicon GPU via `MLComputeUnits.cpuAndGPU`.

!!! warning "The Apple Neural Engine is intentionally NOT used"
    BiRefNet's `ASPPDeformable` decoder requires deformable convolution
    v2, which CoreML lowers via `grid_sample`. The ANE compiler **cannot
    plan `grid_sample`** — passing `.all` or `.cpuAndNeuralEngine` hangs
    the load call in `com.apple.anef.p3` forever. The CPU + Metal GPU
    path is fast enough (~350 ms / 1024² frame for `lite`) that this
    isn't a battery/thermal limit for photogrammetry workloads. Full
    diagnosis: [`models/production_note.md`](https://github.com/placeholder/alicevision-for-mac/blob/main/models/production_note.md).

!!! note "What happened to the rembg/ONNX backend?"
    Earlier versions of this node ran inference through `rembg` +
    ONNX Runtime's CoreML Execution Provider. That path was removed
    2026-05-23: on Apple Silicon it ran 6–10 s per frame on CPU and
    ~233 s per frame via the CoreML EP `CPUAndGPU` mode (Metal
    command-buffer thrashing in ORT for swin-v1 graphs). The
    `.mlpackage` path is 10–35× faster and uses 5× less memory.

## Quick start

```bash
# 1. The CoreML mlpackage models are pre-shipped at ai-models/.
#    Confirm they're present:
ls ai-models/BiRefNet_lite.mlpackage  ai-models/BiRefNet.mlpackage

# 2. In Meshroom, add the SegmentationBiRefNet node to your graph.
#    The descriptor lives at:
#       plugins/ai-segmentation/nodes/aliceVision/SegmentationBiRefNet.py
#    (auto-discovered via MESHROOM_NODES_PATH from scripts/run_meshroom.sh)

# 3. Wire CameraInit.output → SegmentationBiRefNet.input
#    (same input contract as ImageMasking).
```

After the run, masks land in `{outputDir}/` next to the per-view
intermediates, one file per input view.

If the `.mlpackage` files are missing, follow
[`ai-models/README.md`](https://github.com/placeholder/alicevision-for-mac/blob/main/ai-models/README.md)
to re-convert them from the BiRefNet HuggingFace checkpoints (one
command per variant, ~70–125 s on M4).

## Model variants

Exposed as the node's `modelVariant` `ChoiceParam`. Both variants are
MIT-licensed (commercial use allowed) and produced from the
`ZhengPeng7/BiRefNet*` HuggingFace checkpoints.

| Variant | Backbone | `.mlpackage` size | Steady-state latency (M-series GPU) | When to pick it |
|---|---|---|---|---|
| `birefnet-lite` *(default)* | swin_v1_t | 90 MB | **~350 ms / frame** | Default. Fits 8 GB UMA comfortably. Fast enough for everyday photogrammetry. |
| `birefnet-general` | swin_v1_l | 447 MB | **~980 ms / frame** | Higher accuracy on hair, fine foliage, transparent edges. Use when you need the very best mask quality. |

Latencies from [`models/production_note.md`](https://github.com/placeholder/alicevision-for-mac/blob/main/models/production_note.md)
(3-iter mean after 2 warm-ups, single-threaded, no other GPU load,
fixed 1024² FP16 mlprogram).

### How to switch models

The `modelVariant` parameter takes either `birefnet-lite` or
`birefnet-general`.

**Python Meshroom (PySide6 UI):**

1. Select the `SegmentationBiRefNet` node in the graph editor.
2. In the Attribute Editor on the right, find **Model Variant** under
   the node's inputs.
3. Pick from the dropdown. The choice is part of the node UID, so
   swapping invalidates the cache — masks are regenerated on the next
   compute.

**Headless (`meshroom_compute` / direct CLI):**

```bash
source meshroom-venv/bin/activate
python -m meshroom.bin.node_run \
    --nodeType SegmentationBiRefNet \
    --input  /path/to/sfmData.sfm \
    --output /tmp/masks/ \
    --modelVariant birefnet-lite \
    --maskFormat png \
    --keepFilename true
```

The in-process session cache keeps each loaded `.mlpackage` in RAM for
the lifetime of the worker, so re-running with the same `modelVariant`
across chunks pays the ~3–5 s model-load cost only once. Switching
variants mid-process loads the second model without unloading the
first — so a two-variant comparison in one process needs
`90 + 447 = ~540 MB` resident in addition to the activation buffers.

## Workflow integration

`SegmentationBiRefNet` slots in between `CameraInit` and any downstream
node that consumes per-view masks. Two canonical pipelines:

### Object-only photogrammetry (turntable / isolated subject)

```text
CameraInit → SegmentationBiRefNet → FeatureExtraction → FeatureMatching
                                  ↘                   ↘
                                   DepthMap          Meshing → Texturing
```

Wire the **`output`** (masks folder) of `SegmentationBiRefNet` into:

- `FeatureExtraction.masksFolder`
- `DepthMap.masksFolder` (if available — masks suppress background depth)
- `Meshing.masksFolder` (if your scene has consistent foreground across views)

`FeatureExtraction.maskExtension` defaults to `png` and matches our
default `maskFormat`. Switch both to `exr` together if you need float
masks.

### Hybrid (foreground hint, background still reconstructed)

Connect masks only to `FeatureExtraction` and leave `DepthMap` unmasked.
This biases feature detection toward the subject without throwing away
background context — useful for free-roam outdoor captures where the
boundary is fuzzy.

## Apple Silicon optimization

This node is designed for the M-series UMA architecture. Key knobs and
constraints (see [`dev/apple-silicon-optimization.md`](../dev/apple-silicon-optimization.md)
for the deep dive):

- **Compute target is `MLComputeUnits.cpuAndGPU`, always.** Hard-coded in
  the session loader. There is no setting to change this, and there
  should not be — see the ANE warning above.
- **First prediction is slower.** CoreML JIT-compiles the Metal pipelines
  on first run; warm subsequent runs are what the latency table reports.
  The session loader runs one warm-up `predict()` on a zero tensor
  immediately after `MLModel.__init__`, so by the time you see
  `[SegmentationBiRefNet] Session ready` in the log, the pipelines are
  compiled.
- **Memory budget by variant** (peak RSS during prediction, from
  `models/production_note.md`):
  - `birefnet-lite`: ~1 GB → comfortable on 8 GB UMA.
  - `birefnet-general`: ~3 GB → comfortable on 16 GB+.
- **Unified memory (UMA).** CPU and GPU share one physical buffer — no
  host↔device copy. Model weights load once and CPU + Metal GPU see them.
- **Fixed 1024² input.** Both `.mlpackage` files are shape-locked to
  1024×1024. The node resizes source images to 1024² (bilinear) before
  predict and resizes the output mask back to source dimensions
  (bilinear). To support a different resolution, re-convert with
  `python models/convert/convert_to_coreml.py <variant> --resolution N`.

## Backend confirmation in the log

The node always emits, once per chunk:

```
[SegmentationBiRefNet] Host chip: Apple M…
[SegmentationBiRefNet] Compute target: CoreML (CPU + GPU dispatch, coremltools <version>)
[SegmentationBiRefNet] Loading BiRefNet_lite.mlpackage (cpuAndGPU)
[SegmentationBiRefNet] Session ready for variant=birefnet-lite
```

If the `Compute target` line says `UNAVAILABLE`, install
`coremltools` into the `meshroom-venv` (`pip install "coremltools>=8.0"`).

## Output format

Per-view masks are written to the node's output folder
(`{outputDir}/` — `{nodeCacheFolder}` in Meshroom's cache layout).
Naming follows the AliceVision convention used by `ImageMasking`:

| `maskFormat` | Filename | Notes |
|---|---|---|
| `png` *(default)* | `{imageStem}_mask.png` | 8-bit single channel, matches AliceVision `_mask.png` convention |
| `exr` | `{imageStem}.exr` | float32 in `[0, 1]`, useful when downstream nodes prefer EXR |

The mask is the sigmoid output of BiRefNet bilinearly resized back to
the source image dimensions.

## Troubleshooting

### `BiRefNet CoreML package missing at …`

The `.mlpackage` files aren't staged. Run the conversion from
[`ai-models/README.md`](https://github.com/placeholder/alicevision-for-mac/blob/main/ai-models/README.md)
(one command per variant). The packages must live at:

```
ai-models/BiRefNet_lite.mlpackage     (default variant)
ai-models/BiRefNet.mlpackage          (general variant)
```

You can also point the node at an alternative directory with
`export AV_AI_MODELS_DIR=/path/to/my/models` before launching Meshroom.

### Compute target reports `UNAVAILABLE`

`coremltools` is not importable inside `meshroom-venv`. Reinstall:

```bash
source meshroom-venv/bin/activate
pip install "coremltools>=8.0"
python -c "import coremltools as ct; print(ct.__version__)"
```

### Load call hangs forever

You are passing `MLComputeUnits.all` or `.cpuAndNeuralEngine` somewhere
in your stack — the BiRefNet model cannot be lowered to the ANE. The
node itself always uses `.cpuAndGPU`. If you've patched `session.py`,
revert the compute_units argument.

### Mask is all-black

The input image is single-channel grayscale. Convert it upstream (open
in Preview, re-export as RGB).
