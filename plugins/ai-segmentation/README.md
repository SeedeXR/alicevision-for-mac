# ai-segmentation

AI-powered foreground segmentation for AliceVision/Meshroom on Apple
Silicon. Ships the `SegmentationBiRefNet` node, which generates one
binary foreground mask per view using
[BiRefNet](https://github.com/ZhengPeng7/BiRefNet) on CoreML.

Inference runs entirely in-process via `coremltools.models.MLModel`
loaded at `MLComputeUnits.cpuAndGPU` — no CUDA, no PyTorch, no ONNX
Runtime, no `rembg`, no subprocess fan-out. The rembg + ONNX backend
that earlier versions used was removed 2026-05-23: on Apple Silicon it
was 10–35× slower than the pre-converted `.mlpackage` path on swin-v1
graphs.

Loading a model takes 3–5 s; per-image inference is sub-second on either
variant.

## What it does

For each view in an upstream SfMData JSON, the node:

1. Loads the source image.
2. Resizes to `1024×1024` and ImageNet-normalizes.
3. Runs the CoreML BiRefNet model on Metal GPU.
4. Resizes the sigmoid mask back to the original image dimensions.
5. Writes `<imageStem>_mask.png` (or `<viewId>.exr`) to the node cache.

The output folder is intended to be wired directly into
`FeatureExtraction.masksFolder` / `DepthMap.masksFolder` so the
photogrammetry pipeline can use the masks to suppress background
features.

## ⚠ The Apple Neural Engine is intentionally not used

BiRefNet's `ASPPDeformable` decoder requires deformable convolution v2,
which CoreML lowers via `grid_sample`. The Apple Neural Engine compiler
**cannot plan `grid_sample`** — loading with `MLComputeUnits.all` or
`.cpuAndNeuralEngine` hangs in `com.apple.anef.p3` forever. The node
always passes `.cpuAndGPU`. Full diagnosis + measurements at
[`models/production_note.md`](../../models/production_note.md).

## Models bundled

The two pre-converted CoreML `.mlpackage` files live at
[`../../ai-models/`](../../ai-models/):

| Variant | Path | Size | Backbone | When to use |
|--------|------|------|----------|-------------|
| `birefnet-lite` | `ai-models/BiRefNet_lite.mlpackage` | 90 MB | swin_v1_t | **Default.** ~350 ms/frame on M-series GPU. Fits 8 GB UMA comfortably. |
| `birefnet-general` | `ai-models/BiRefNet.mlpackage` | 447 MB | swin_v1_l | Higher accuracy on hair/fine edges. ~980 ms/frame on M-series GPU. |

Both packages are FP16 `mlprogram`, fixed `[1, 3, 1024, 1024]` input,
target macOS 14+. Conversion recipe + how-to: [`../../ai-models/README.md`](../../ai-models/README.md).

## Install

This plugin ships with the repository; nothing extra to install. To
confirm it is loaded:

```bash
python -m pytest tests/python/test_plugin_manifest.py -v
```

If you want to drop it into a fresh checkout:

1. Copy `plugins/ai-segmentation/` into your project.
2. Stage at least `ai-models/BiRefNet_lite.mlpackage` (the default
   variant). Add `ai-models/BiRefNet.mlpackage` if you want the higher
   accuracy variant.
3. Ensure `meshroom-venv/` has `coremltools`, `numpy`, `Pillow`,
   `imageio` installed.
4. Re-launch Meshroom — descriptor discovery is automatic via
   `MESHROOM_NODES_PATH` (set by `scripts/run_meshroom.sh`).

## How to use

### Meshroom (Python UI)

`SegmentationBiRefNet` appears in the node browser under **Utils**. Drag
it onto the canvas, connect its `input` pin to the upstream SfMData
output of `CameraInit`, and connect its `output` pin to the
`masksFolder` input of `FeatureExtraction` / `DepthMap`.

### Standalone CLI smoke test

```bash
source meshroom-venv/bin/activate
python -m meshroom.bin.node_run \
    --nodeType SegmentationBiRefNet \
    --input /tmp/some.sfm \
    --output /tmp/seg_out \
    --modelVariant birefnet-lite \
    --maskFormat png \
    --keepFilename true \
    --verboseLevel info
```

You should see
`[SegmentationBiRefNet] Compute target: CoreML (CPU + GPU dispatch, coremltools <ver>)`
in the log; if the input file does not exist the run fails with
`RuntimeError: SfMData file not found`.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `input` | file | SfMData JSON file. One mask is produced per `views[]` entry. |
| `modelVariant` | string | `birefnet-lite` (default) or `birefnet-general`. |
| `maskFormat` | string | `png` (default) or `exr`. |
| `keepFilename` | bool | If true, use the original image stem; otherwise the SfMData viewId. |
| `verboseLevel` | string | `fatal` / `error` / `warning` / `info` / `debug` / `trace`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | file (folder) | Folder containing one mask per view. |

## License + attribution

This plugin is MIT-licensed. The models it loads are governed by their
upstream licenses:

- BiRefNet: MIT, © Peng Zheng et al. — [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet).

Refer to the upstream repository for citation guidance if you use
the masks in a published reconstruction.
