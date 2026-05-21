# AI segmentation (`SegmentationBiRefNet`)

End-user guide for the AI-powered foreground/background segmentation node
shipped with this port. The implementation specification lives in
`instructions/ai_instruction.md`; this page covers what an operator needs
to run the node.

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

`SegmentationBiRefNet` uses the [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)
ONNX model dispatched through ONNX Runtime's CoreML Execution Provider,
so inference lands on the Apple Neural Engine + GPU + CPU on Apple
Silicon (M1–M4).

## Quick start

```bash
# 1. One-time: pre-download the ONNX weights into ai-models/
python scripts/download_models.py

# 2. In Meshroom (Python or Native UI), add the SegmentationBiRefNet node
#    to your graph. The Python node ships at:
#       meshroom-mac/nodes/aliceVision/SegmentationBiRefNet.py

# 3. Wire CameraInit.output → SegmentationBiRefNet.inputImages
#    (same input contract as ImageMasking).
```

After the run, masks land in `{outputDir}/` next to the per-view
intermediates, one file per input view.

!!! tip "Why pre-download?"
    `rembg` will auto-download the model on first use, but that happens
    mid-pipeline and stalls the run with no progress bar in the Meshroom
    log. Always run `scripts/download_models.py` once before the first
    pipeline execution. Source: `instructions/ai_instruction.md` §3c.

## Model variants

Exposed as the node's `modelVariant` `ChoiceParam`. All variants live
under the `ZhengPeng7` HuggingFace namespace; license is MIT (commercial
use allowed).

| Variant | Backbone | Approx. size | Measured CPU EP latency (M4) | When to pick it |
|---|---|---|---|---|
| `birefnet-general` *(default)* | swin_v1_large | ~927 MB | **~10.1 s / image** | Default for everyday photogrammetry: rigid subjects, mixed indoor/outdoor scenes, turntable captures. Best general-purpose accuracy/speed ratio. |
| `birefnet-dis` | swin_v1_large + DIS-trained | ~929 MB | **~9.5 s / image** | High-detail / dichotomous image segmentation: hair, fur, fine foliage, mesh-like or transparent edges. Marginal slowdown vs `general` for noticeably better complex edges. |
| `birefnet-lite` | swin_v1_tiny | ~213 MB | **~6.7 s / image** | Fast / memory-constrained: M1, M2 base (8 GB UMA), or large batches where wall-clock matters more than the last few percent of mask precision. ~4× smaller, **~33% faster on CPU EP**. |

Latencies from `memory/perf_segmentation_s52.md` (S52 light profile, M4 / 24 GB / macOS 26.5 / onnxruntime 1.26, CPU Execution Provider warm steady-state, `OMP_NUM_THREADS=4`). The CPU EP path is the empirically-fastest current option — see [Hardware acceleration](#hardware-acceleration) below.

Source: rembg release v0.0.0 binaries + `ZhengPeng7/BiRefNet*` HuggingFace.
License: MIT (commercial use allowed). License source: `instructions/ai_instruction.md` §3a–3b.

### How to switch models

The `modelVariant` parameter takes one of: `birefnet-general`, `birefnet-dis`, `birefnet-lite`.

**Python Meshroom (PySide6 UI):**

1. Select the `SegmentationBiRefNet` node in the graph editor.
2. In the Attribute Editor on the right, find **Model Variant** under the node's inputs.
3. Pick from the dropdown. The choice is part of the node UID, so swapping invalidates the cache — masks are regenerated on the next compute.

**Native SwiftUI Meshroom:**

1. Click the node on the canvas. The inspector pane on the right shows its parameters.
2. The **modelVariant** dropdown is at the top of the inputs list. The default is `birefnet-general`.
3. Connections are preserved across model swaps — only the per-node cache invalidates.

**Headless (`meshroom_compute` or the Native runner):**

```bash
# Native Swift wrapper — sets U2NET_HOME + runs in-process
bash meshroom-native/scripts/run_python_node.sh \
    --nodeType SegmentationBiRefNet \
    --input  /path/to/sfmData.sfm \
    --output /tmp/masks/ \
    --modelVariant birefnet-dis \
    --maskFormat png \
    --alphaMatting false \
    --outputResolution 1024
```

The session loader keeps a per-model cache, so running the same node again with the same `modelVariant` reuses the already-loaded ONNX session — no reload cost. Switching variants mid-pipeline allocates a fresh session but does NOT release the prior one (so a 3-variant comparison in one process needs `213 + 927 + 929 = ~2.1 GB` resident memory; well within M-series UMA).

## Workflow integration

`SegmentationBiRefNet` slots in between CameraInit and any downstream node
that consumes per-view masks. Two canonical pipelines:

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

`FeatureExtraction.maskExtension` defaults to `png` and matches our default `maskFormat`. Switch both to `exr` together if you need float masks.

### Hybrid (foreground hint, background still reconstructed)

Connect masks only to `FeatureExtraction` and leave `DepthMap` unmasked. This biases feature detection toward the subject without throwing away background context — useful for free-roam outdoor captures where the boundary is fuzzy.

### Enabling in the Native SwiftUI app

In the node palette (bottom-left), drag `SegmentationBiRefNet` onto the canvas. The wand-and-stars icon (`wand.and.stars.inverse`) distinguishes it from the C++ AliceVision nodes. The node spawns one OS process per compute call via `meshroom-native/scripts/run_python_node.sh`, which activates `meshroom-venv/`, sets `U2NET_HOME=<repo>/ai-models/`, and dispatches to `python -m meshroom.bin.node_run` — no per-node Python interpreter lifetime concerns.

## Apple Silicon optimization

This node is designed for the M-series UMA architecture. Key knobs and constraints (see `docs/dev/apple-silicon-optimization.md` for the deep dive):

- **Default to CPU EP today.** Set `ONNX_FORCE_CPU=1` in your shell before launching Meshroom. Measured 6.7 s/image (lite) or ~10 s/image (general/dis) on M4. This bypasses both the 25-min ANE compile AND the broken `CPUAndGPU` path. `ONNX_FORCE_CPU` is honored by `plugins/ai-segmentation/python/segmentation/session.py`.
- **`OMP_NUM_THREADS=4`** is the sweet spot on M4 (4 P-cores). Going higher spills to E-cores and increases per-image latency. The pipeline does NOT set this automatically — `scripts/run_meshroom.sh` honors whatever you export.
- **Unified memory (UMA).** CPU, GPU, and ANE share one physical buffer — no host↔device copy. Model weights load once and all backends see them. Memory budgets are wall-shared with kernels and dylibs.
- **First-run ANE compile (optional).** If you want the CoreML EP + `MLComputeUnits=ALL` path, run `python scripts/download_models.py --warmup` once per machine. `ANECompilerService` will lower the swin transformer to ANE bytecode in 15-25 minutes, single-threaded, ~17 GB peak RSS. Cached at `~/Library/Caches/com.apple.e5rt.e5bundlecache/` thereafter. Warm-cache inference latency on this port is currently **unmeasured**.
- **`outputResolution`** is handled by CoreML's / ORT's lowering passes — no user-tunable threadgroup. Best practice: 1024 is sufficient for almost every photogrammetry subject; 2048 quadruples ANE/CPU work for marginal mask quality on natural-image subjects.
- **Memory budget by model.** From S52 light profile (peak RSS during inference):
  - `birefnet-lite`: 6.5 GB → 8 GB tight, 16 GB OK.
  - `birefnet-general`: 8.6 GB → 16 GB OK.
  - `birefnet-dis`: 8.9 GB → 16 GB OK.
  - ANE compile peak: 17 GB → only safe on 24+ GB Macs without swap.
- **Model-swap cost.** The Python session cache (`segmentation.session._SESSION_CACHE`) keeps the loaded ONNX in RAM until process exit. Swapping variants mid-process loads the new ONNX (~2 s) but doesn't unload the prior — so a 3-variant comparison needs `213 + 927 + 929 = ~2.1 GB` resident.

## Hardware acceleration

On Apple Silicon (macOS 26.5 + onnxruntime 1.26 stack as of S53), there
are three viable inference paths through the BiRefNet ONNX graph. They
are NOT equivalent, despite both involving CoreML/Metal/ANE in some
form. Empirically measured numbers (`memory/perf_segmentation_s52.md`):

| Path | Wall-clock (lite, M4) | When to use |
|---|---|---|
| **CPU EP** (`ONNX_FORCE_CPU=1`) | **6.7 s / image** | Default. No setup cost. Recommended today. |
| **CoreML EP + ANE warm cache** (`MLComputeUnits=ALL`) | TBD (3-10× CPU per public BiRefNet benchmarks, unmeasured on this port) | Best for production once warmed. Costs 15-25 min one-time ANE compile. |
| **CoreML EP + `CPUAndGPU`** | **~233 s / image (35× SLOWER)** | Never use this on the current stack. |

### Why CPU EP wins today

Counterintuitive on Apple Silicon, but measured: on swin-transformer
BiRefNet via ORT 1.26's CoreML EP with `MLComputeUnits=CPUAndGPU`, the
Metal command-buffer overhead per ONNX subgraph dominates wall-clock.
ORT partitions the swin attention layers into many small subgraphs,
each issuing its own Metal dispatch — the dispatch teardown cost
exceeds the actual compute. See `docs/dev/apple-silicon-optimization.md`
for the full analysis.

### Switching to CPU EP

Set the environment variable before launching Meshroom:

```bash
export ONNX_FORCE_CPU=1
bash scripts/run_meshroom.sh python bin/meshroom_batch -p ...
```

The node will log:

```
[SegmentationBiRefNet] ONNX_FORCE_CPU=1 — pinning to CPUExecutionProvider (skipping CoreML EP).
[SegmentationBiRefNet] Loading model 'birefnet-general-lite' with providers=['CPUExecutionProvider']
```

### When you want the ANE path

Pre-warm once per machine (15-25 min, single-threaded, ~17 GB peak RSS
during compile):

```bash
python scripts/download_models.py --warmup
```

After that, the ANE bytecode bundle is cached at
`~/Library/Caches/com.apple.e5rt.e5bundlecache/` and subsequent
inferences are mmap'd directly. Then run WITHOUT `ONNX_FORCE_CPU` to
take the CoreML EP + `MLComputeUnits=ALL` path.

### Backend confirmation in the log

The node always emits:

```
[SegmentationBiRefNet] Host chip: Apple M…
[SegmentationBiRefNet] Compute target: CoreML (CPU+GPU+ANE)
```

The `Compute target` line reports what `onnxruntime.get_available_providers()`
returns, NOT which provider is actually used at inference. If
`ONNX_FORCE_CPU=1` is set, look for the `pinning to CPUExecutionProvider`
line emitted by the session loader.

## Output format

Per-view masks are written to the node's output folder
(`{outputDir}/` — `{nodeCacheFolder}` in Meshroom's cache layout).
Naming follows the AliceVision convention used by `ImageMasking`:

| `maskFormat` | Filename | Notes |
|---|---|---|
| `png` *(default)* | `{imageStem}_mask.png` | 8-bit single channel, matches AliceVision `_mask.png` convention |
| `exr` | `{imageStem}.exr` | float32, useful when downstream nodes prefer EXR |

The mask is the alpha channel produced by `rembg.remove()` on the input
image. Source: `instructions/ai_instruction.md` §7.

## Troubleshooting

### Log doesn't show `Compute target: CoreML`

The installed `onnxruntime` wheel lacks the CoreML Execution Provider.
Reinstall the standard macOS wheel (not the `onnxruntime-gpu` package,
which pulls CUDA):

```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime>=1.18.0"
python -c "import onnxruntime as ort; \
    assert 'CoreMLExecutionProvider' in ort.get_available_providers(), \
    'CoreML EP missing'; print('OK')"
```

The standard `onnxruntime` pip wheel for macOS has included the CoreML
EP since v1.16. Source: `instructions/ai_instruction.md` §3d–3e.

### Model download fails

`scripts/download_models.py` streams the ONNX files directly from
HuggingFace via `urllib`. If the download stalls:

1. Check disk space at the target directory (defaults to `ai-models/`).
2. Clear any partial pip caches under `~/Library/Caches/pip`.
3. Retry with `--force` to re-fetch a corrupted partial download.
4. As a last resort, download the URL manually — the URLs are listed
   directly in the `MODELS` dict of `scripts/download_models.py`, and
   in the [Model variants](#model-variants) table above.

### Mask is all-black

`rembg` requires at least 3 channels (RGB). If the input image is
single-channel grayscale, convert it upstream — or open it in Preview,
re-export as RGB, and re-run. Source: `instructions/ai_instruction.md`
§9 (model-input contract).

### Node runs but Activity Monitor shows no GPU activity

Open **Activity Monitor → Window → GPU History** during the node's
run. If the GPU line is flat, CoreML is dispatching to CPU/ANE only —
that's not necessarily wrong (ANE work doesn't show on the GPU graph),
but combined with the log line missing `(CPU+GPU+ANE)`, it indicates
a CoreML conversion that fell back to CPU. Re-check
`coremltools >= 7.2` is installed and re-delete the cached
`.mlpackage` to force reconversion.

## See also

- [`docs/dev/segmentation-pipeline.md`](../dev/segmentation-pipeline.md) — architecture, helper modules, adding new variants.
- [`docs/reference/segmentation.md`](../reference/segmentation.md) — full flag reference for `scripts/download_models.py` and the node parameters.
- `instructions/ai_instruction.md` (repo root) — implementation specification.
