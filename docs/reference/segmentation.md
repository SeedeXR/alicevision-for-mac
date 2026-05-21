# Segmentation reference

CLI + API reference for the AI segmentation tooling. Narrative guide:
[`user/segmentation.md`](../user/segmentation.md). Architecture:
[`dev/segmentation-pipeline.md`](../dev/segmentation-pipeline.md).

## `scripts/download_models.py`

Pre-flight ONNX downloader. Run once before the first Meshroom run that
includes a `SegmentationBiRefNet` node so weights are present locally.

```bash
python scripts/download_models.py [flags]
```

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--variant` | `{birefnet-general, birefnet-dis, birefnet-lite}` | `birefnet-general` | Which model to fetch. Names match the `modelVariant` `ChoiceParam` on the node. |
| `--all` | flag | off | Download every variant in the `MODELS` dict. Overrides `--variant`. |
| `--target` | path | `ai-models/` (relative to project root) | Destination directory. Created if missing. |
| `--force` | flag | off | Re-download even if the destination file already exists. Use to recover from a truncated/corrupt previous fetch. |

URLs are stored in the `MODELS` dict at the top of the script — the
authoritative table for HuggingFace paths. The script uses
`urllib.request.urlretrieve` with a percent-progress reporthook; there
is no resume support, so on a partial download use `--force` to
restart.

Source: `instructions/ai_instruction.md` §3c.

## `SegmentationBiRefNet` node parameters

The Meshroom node lives at
`meshroom-mac/nodes/aliceVision/SegmentationBiRefNet.py` and follows
the same `desc.Node` declaration style as
[`ImageMasking.py`](https://github.com/placeholder/alicevision-for-mac/blob/main/meshroom-mac/nodes/aliceVision/ImageMasking.py).
Mirrors `instructions/ai_instruction.md` §7.

### Inputs

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `inputImages` | `ListAttribute(File)` | — | Same input contract as adjacent AliceVision nodes; wire from `CameraInit.output` (or any node producing per-view image paths). |
| `modelVariant` | `ChoiceParam` | `birefnet-general` | One of `birefnet-general`, `birefnet-dis`, `birefnet-lite`. See [user guide → Model variants](../user/segmentation.md#model-variants). |
| `outputResolution` | `ChoiceParam` | `1024` | Internal inference resolution. One of `512`, `1024`, `2048`. Higher = sharper edges, more memory, slower. |
| `alphaMatting` | `BoolParam` | `False` | Enables rembg's edge-refinement post-pass. Cleaner soft edges (hair, fur) at the cost of additional CPU work. |
| `verboseLevel` | `ChoiceParam` | `info` | `fatal`/`error`/`warning`/`info`/`debug`/`trace` — same `VERBOSE_LEVEL` enum as the AliceVision binaries. |

### Outputs

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `outputMasks` | `File` (output) | `{nodeCacheFolder}/masks/` | Per-view mask folder. Path pattern matches the AliceVision mask-folder convention used by `ImageMasking.output`. |

### Output filename convention

For each input view named `IMG_1234.JPG`, the node emits:

| `maskFormat` | Output file | Pixel type |
|---|---|---|
| `png` *(default)* | `{outputMasks}/IMG_1234_mask.png` | 8-bit single channel |
| `exr` | `{outputMasks}/IMG_1234.exr` | float32 |

This matches the `_mask.png` suffix convention used elsewhere in the
AliceVision pipeline. Source: `instructions/ai_instruction.md` §1b,
§7.

## Environment variables

### `U2NET_HOME`

`rembg` resolves model files relative to its model-home directory.
Default: `~/.u2net/`. Override with:

```bash
export U2NET_HOME=/Users/alexmkwizu/Documents/SoftwareProjects/alicevision-mac/alicevision-for-mac/ai-models
```

Set this to point `rembg` at the repo-local `ai-models/` cache instead
of the global `~/.u2net/` directory. Useful when you want models
versioned alongside the project (and gitignored under `ai-models/`).
`scripts/run_meshroom.sh` exports this for you when the
`SegmentationBiRefNet` node is present in the graph.

## Log line contract

At the start of `processChunk`, the node emits the following lines to
Meshroom's logger (not bare `print()`). If a run goes through without
these lines appearing in the log, the node was not entered — check
node discovery (`MESHROOM_NODES_PATH`) before anything else.

```
[SegmentationBiRefNet] Host chip: Apple M…
[SegmentationBiRefNet] Compute target: CoreML (CPU+GPU+ANE)
```

If the second line says anything other than `CoreML`, the
`onnxruntime` wheel lacks the CoreML EP — see
[user troubleshooting](../user/segmentation.md#log-doesnt-show-compute-target-coreml).
Source: `instructions/ai_instruction.md` §8.

## See also

- [Pipeline binaries](binaries.md) — the 12 native AliceVision CLI binaries; `SegmentationBiRefNet` is a Python node, not a binary, but is listed there for completeness.
- `instructions/ai_instruction.md` (repo root) — implementation specification.
