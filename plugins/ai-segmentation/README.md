# ai-segmentation

AI-powered foreground segmentation for AliceVision/Meshroom on Apple
Silicon. Ships the `SegmentationBiRefNet` node, which generates one
binary foreground mask per view using
[BiRefNet](https://github.com/ZhengPeng7/BiRefNet) via
[rembg](https://github.com/danielgatis/rembg) and ONNX Runtime's
CoreML execution provider.

Inference runs on the unified GPU + Apple Neural Engine via CoreML —
no CUDA, no PyTorch, no Python subprocess fan-out. Loading a model
takes 5–10 s; per-image inference is 0.4–1.5 s depending on variant.

## What it does

For each view in an upstream SfMData JSON, the node:

1. Loads the source image.
2. Runs BiRefNet through rembg's CoreML-backed session.
3. Resizes the alpha channel to the original image dimensions.
4. Writes `<imageStem>_mask.png` (or `<viewId>.exr`) to the node cache.

The output folder is intended to be wired directly into
`FeatureExtraction.masksFolder` / `DepthMap.masksFolder` so the
photogrammetry pipeline can use the masks to suppress background
features.

## Models bundled

Three BiRefNet ONNX checkpoints are supported. Download with the
plugin's own model fetcher:

```bash
source meshroom-venv/bin/activate
python plugins/ai-segmentation/scripts/download_models.py --all
```

| Variant | Size | Backbone | When to use |
|--------|------|----------|-------------|
| `birefnet-general` | ~927 MB | swin_v1_large | Default — best general accuracy/speed ratio. |
| `birefnet-dis` | ~929 MB | swin_v1_large (DIS-trained) | Hair, fur, fine foliage, transparent edges. |
| `birefnet-lite` | ~213 MB | swin_v1_tiny | M1/M2 base (8 GB UMA) or large batch jobs. |

Models are staged under `<repo>/ai-models/` (overridable with the
`U2NET_HOME` env var; rembg honours the same variable).

## Install

This plugin ships with the repository; nothing extra to install. To
confirm it is loaded:

```bash
cd meshroom-native && swift test --filter PluginRegistryTests
```

If you want to drop it into a fresh checkout:

1. Copy `plugins/ai-segmentation/` into your project.
2. Ensure `meshroom-venv/` has `rembg`, `onnxruntime`, `Pillow`,
   `imageio`, `coremltools` installed.
3. Re-launch the app — discovery is automatic.

## How to use

### Meshroom (Python UI)

`SegmentationBiRefNet` appears in the node browser under **Utils**. Drag
it onto the canvas, connect its `input` pin to the upstream SfMData
output of `CameraInit`, and connect its `output` pin to the
`masksFolder` input of `FeatureExtraction` / `DepthMap`.

### Native Meshroom (SwiftUI)

The plugin is auto-discovered at startup. The node shows up in the
left-edge palette with the magic-wand icon, ready to drag onto the
canvas. The Swift `GraphExecutor` invokes
`meshroom-native/scripts/run_python_node.sh`, which activates the
project venv and dispatches to `meshroom.bin.node_run` with
`--nodeType SegmentationBiRefNet`.

To verify dispatch by hand:

```bash
bash meshroom-native/scripts/run_python_node.sh \
    --nodeType SegmentationBiRefNet \
    --input /tmp/some.sfm \
    --output /tmp/seg_out \
    --modelVariant birefnet-general \
    --outputResolution 1024 \
    --alphaMatting false \
    --maskFormat png \
    --keepFilename true \
    --verboseLevel info
```

You should see `[SegmentationBiRefNet] Compute target: CoreML (CPU+GPU+ANE)`
in the log; if the input file does not exist the run fails with
`RuntimeError: SfMData file not found`.

## Inputs

| Name | Type | Description |
|------|------|-------------|
| `input` | file | SfMData JSON file. One mask is produced per `views[]` entry. |
| `modelVariant` | string | `birefnet-general` / `birefnet-dis` / `birefnet-lite`. |
| `outputResolution` | string | Longest-edge resolution for inference (`512` / `1024` / `2048`). |
| `alphaMatting` | bool | Run rembg's alpha-matting post-pass for refined edges. Slower. |
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
- rembg: MIT, © Daniel Gatis — [danielgatis/rembg](https://github.com/danielgatis/rembg).

Refer to the upstream repositories for citation guidance if you use
the masks in a published reconstruction.
