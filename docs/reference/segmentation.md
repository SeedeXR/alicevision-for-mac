# Segmentation reference

API reference for the AI segmentation tooling. Narrative guide:
[`user/segmentation.md`](../user/segmentation.md). Architecture:
[`dev/segmentation-pipeline.md`](../dev/segmentation-pipeline.md).

## Default models on disk

| File | Backbone | Source |
|---|---|---|
| `ai-models/BiRefNet_lite.mlpackage` | swin_v1_t (90 MB) | `python models/convert/convert_to_coreml.py lite` — see [`ai-models/README.md`](https://github.com/placeholder/alicevision-for-mac/blob/main/ai-models/README.md). |
| `ai-models/BiRefNet.mlpackage` | swin_v1_l (447 MB) | `python models/convert/convert_to_coreml.py general`. |

Both `.mlpackage` files are FP16 `mlprogram`, fixed `[1, 3, 1024, 1024]`
input, target macOS 14+. The earlier rembg / ONNX Runtime fallback path
was removed 2026-05-23.

## `SegmentationBiRefNet` node parameters

The Meshroom node lives at
`plugins/ai-segmentation/nodes/aliceVision/SegmentationBiRefNet.py` and
follows the same `desc.Node` declaration style as upstream's
`ImageMasking.py`.

### Inputs

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `input` | `File` | — | SfMData JSON file. One mask is produced per `views[]` entry. Wire from `CameraInit.output`. |
| `modelVariant` | `ChoiceParam` | `birefnet-lite` | One of `birefnet-lite`, `birefnet-general`. See [user guide → Model variants](../user/segmentation.md#model-variants). |
| `maskFormat` | `ChoiceParam` | `png` | `png` (8-bit) or `exr` (float32). |
| `keepFilename` | `BoolParam` | `True` | If true, use the original image stem in the mask filename; otherwise use the SfMData viewId. |
| `verboseLevel` | `ChoiceParam` | `info` | `fatal`/`error`/`warning`/`info`/`debug`/`trace` — same `VERBOSE_LEVEL` enum as the AliceVision binaries. |

### Outputs

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `output` | `File` (output) | `{nodeCacheFolder}` | Per-view mask folder. Wire into `FeatureExtraction.masksFolder` / `DepthMap.masksFolder`. |
| `masks` | `File` (output, semantic=image) | derived | Per-view mask path pattern. |

### Output filename convention

For each input view named `IMG_1234.JPG`, the node emits:

| `maskFormat` | Output file | Pixel type |
|---|---|---|
| `png` *(default)* | `{output}/IMG_1234_mask.png` | 8-bit single channel |
| `exr` | `{output}/IMG_1234.exr` | float32 in `[0, 1]` |

This matches the `_mask.png` suffix convention used elsewhere in the
AliceVision pipeline.

## Environment variables

### `AV_AI_MODELS_DIR`

Override the directory the node looks in for the `.mlpackage` files.
Defaults to `<repo>/ai-models/`.

```bash
export AV_AI_MODELS_DIR=/Volumes/Models/birefnet-mlpackages
```

### `U2NET_HOME` *(legacy)*

Also honoured (lower precedence than `AV_AI_MODELS_DIR`) for operators
who set it during the previous rembg-era. New deployments should use
`AV_AI_MODELS_DIR`.

## Log line contract

At the start of `processChunk`, the node emits the following lines to
Meshroom's logger (not bare `print()`). If a run goes through without
these lines appearing in the log, the node was not entered — check
node discovery (`MESHROOM_NODES_PATH`) before anything else.

```
[SegmentationBiRefNet] ai-models dir: …/ai-models
[SegmentationBiRefNet] Host chip: Apple M…
[SegmentationBiRefNet] Compute target: CoreML (CPU + GPU dispatch, coremltools <version>)
[SegmentationBiRefNet] Loading BiRefNet_<variant>.mlpackage (cpuAndGPU)
[SegmentationBiRefNet] Session ready for variant=<variant>
```

If the third line says `UNAVAILABLE` instead of `CoreML (CPU + GPU dispatch …)`,
`coremltools` is not importable inside `meshroom-venv` — see
[user troubleshooting](../user/segmentation.md#compute-target-reports-unavailable).
The Apple Neural Engine is intentionally not used; see
[`models/production_note.md`](https://github.com/placeholder/alicevision-for-mac/blob/main/models/production_note.md)
for why.

## See also

- [Pipeline binaries](binaries.md) — the 12 native AliceVision CLI binaries; `SegmentationBiRefNet` is a Python node, not a binary, but is listed there for completeness.
- [`ai-models/README.md`](https://github.com/placeholder/alicevision-for-mac/blob/main/ai-models/README.md) — conversion recipe.
- [`models/production_note.md`](https://github.com/placeholder/alicevision-for-mac/blob/main/models/production_note.md) — production-decision document.
