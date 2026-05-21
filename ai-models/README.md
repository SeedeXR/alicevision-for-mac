# `ai-models/` — segmentation model cache

This directory holds the ONNX weights consumed by the AI segmentation
plugin (`plugins/ai-segmentation/`, node `SegmentationBiRefNet`). The
folder itself is tracked (so the repo lays out cleanly); the `*.onnx`
files inside are NOT — they're large (~213 MB to ~929 MB each) and
deterministically fetchable from upstream releases.

> **Pre-flight:** run `python scripts/download_models.py` **once**
> before your first segmentation node execution. `rembg` will lazily
> download if a model is missing, but mid-pipeline downloads stall the
> Meshroom log without progress reporting. See [Quick
> start](#quick-start).

---

## What lives here

| File | Variant | Backbone | Size | MD5 | Source |
|---|---|---|---|---|---|
| `birefnet-general.onnx` | `birefnet-general` | swin\_v1\_large | 927 MB | `7a35a0141cbbc80de11d9c9a28f52697` | `danielgatis/rembg` v0.0.0 release |
| `birefnet-dis.onnx` | `birefnet-dis` | swin\_v1\_large + DIS-trained | 929 MB | `2d4d44102b446f33a4ebb2e56c051f2b` | `danielgatis/rembg` v0.0.0 release |
| `birefnet-general-lite.onnx` | `birefnet-lite` *(alias)* | swin\_v1\_tiny | 213 MB | `4fab47adc4ff364be1713e97b7e66334` | `danielgatis/rembg` v0.0.0 release |

All three are MIT-licensed (commercial use allowed). Upstream:
<https://huggingface.co/ZhengPeng7/BiRefNet> (full) /
<https://huggingface.co/ZhengPeng7/BiRefNet-lite> (tiny).

Verify your local copies match upstream:

```bash
cd ai-models/
md5 -q birefnet-general.onnx        # 7a35a0141cbbc80de11d9c9a28f52697
md5 -q birefnet-dis.onnx            # 2d4d44102b446f33a4ebb2e56c051f2b
md5 -q birefnet-general-lite.onnx   # 4fab47adc4ff364be1713e97b7e66334
```

Mismatch → re-fetch with `python scripts/download_models.py --variant <name> --force`.

---

## What each model does

### `birefnet-general` (default, ~927 MB, swin\_v1\_large)

**The everyday photogrammetry model.** Trained on diverse foreground/
background pairs; produces a single binary alpha mask per view.

- **Best for:** rigid subjects on a known background, turntable
  captures, mixed indoor/outdoor scenes, building façades, statues,
  vehicles.
- **Edge quality:** clean on hard-edged subjects (objects, machinery).
  Acceptable on soft edges (cloth, paper) but not optimal.
- **Mask characteristics:** confident foreground, decisive
  background — minimal "uncertain" middle-gray pixels.

This is the variant most Meshroom users should start with.

### `birefnet-dis` (~929 MB, swin\_v1\_large + DIS-trained)

**The high-detail / complex-edge model.** Trained on the DIS5K
"dichotomous image segmentation" dataset, which emphasizes very fine
boundaries.

- **Best for:** hair, fur, foliage (leaves, grass blades), wire mesh,
  jewellery, transparent/translucent edges, plush toys.
- **Edge quality:** noticeably better on the cases above; comparable
  to `general` on hard-edged subjects.
- **Trade-off:** ~2-5% slower per inference than `general` (the cost
  is the additional refinement passes in the trained head).
- **Mask characteristics:** more middle-gray pixels at boundaries
  (deliberate — this is what makes it good at hair). If your
  downstream consumer is `FeatureExtraction`, the binary thresholding
  is unaffected. If you're using the mask as alpha for compositing,
  you get a smoother roll-off.

Switch to this when `general`'s edges look "chunky" on hair/foliage.

### `birefnet-lite` (~213 MB, swin\_v1\_tiny — alias for `birefnet-general-lite`)

**The fast, memory-light model.** Same training data as `general`,
distilled into a swin\_v1\_tiny backbone — 4× smaller, **~33%
faster** on CPU EP.

- **Best for:** M1/M2 base (8 GB UMA), large batch jobs, CI/test
  loops, anything where wall-clock matters more than the last few
  percent of mask precision.
- **Edge quality:** clearly worse than the large models — small
  details (twigs, hair strands) get smoothed out. Center-of-mass
  segmentation is still reliable.
- **Mask characteristics:** softer edges, occasional missed thin
  features. Good enough for the "mask out the obvious background"
  use case.

The plugin internally maps `birefnet-lite` → `birefnet-general-lite`
(the rembg-canonical name); the on-disk filename is
`birefnet-general-lite.onnx`.

---

## Quick start

### Download all three (~2 GB total)

```bash
source meshroom-venv/bin/activate
python scripts/download_models.py --all
```

### Download a single variant

```bash
python scripts/download_models.py --variant birefnet-lite
python scripts/download_models.py --variant birefnet-general
python scripts/download_models.py --variant birefnet-dis
```

### Pre-warm the ANE bundle cache (production)

The first time CoreML+ANE loads a model, `ANECompilerService` lowers
the swin-transformer graph to ANE bytecode — **15-25 minutes**,
single-threaded, ~17 GB peak RSS. Once done it's cached at
`~/Library/Caches/com.apple.e5rt.e5bundlecache/` and reused forever
on this machine.

For production deployments, pre-bake before the first photogrammetry
job:

```bash
python scripts/download_models.py --variant birefnet-general --warmup
```

The `--warmup` flag runs one dummy inference through the CoreML EP
with `ComputeUnits.ALL`, paying the compile cost up front. Skip on
dev machines if you don't intend to use the ANE path (see
[Resource consumption](#resource-consumption) below).

### Where the files end up

The downloader defaults to this directory (`<repo>/ai-models/`) and
exports `U2NET_HOME` so `rembg` finds them. To override:

```bash
export U2NET_HOME=/some/other/cache
python scripts/download_models.py --target /some/other/cache
```

---

## How they're used in the pipeline

The plugin's Meshroom node descriptor lives at
`plugins/ai-segmentation/nodes/aliceVision/SegmentationBiRefNet.py`.
Both the Python Meshroom UI and the native SwiftUI app route through
the same code path:

```
SfMData input view list
   │
   ▼
get_session(modelVariant)  ──── ai-models/<variant>.onnx
   │                              ▲
   ▼                              │
rembg.remove(image, session)  ───┘ (CoreML EP or CPU EP)
   │
   ▼
alpha channel → PNG / EXR mask, named <imageStem>_mask.png
   │
   ▼
downstream nodes (FeatureExtraction, DepthMap, Meshing) consume
masksFolder = <SegmentationBiRefNet.output>
```

Switch variants at runtime via the node's `modelVariant` parameter
(dropdown in both UIs, or `--modelVariant <name>` on the native
runner CLI). The session loader keeps a per-model cache, so swapping
during a graph re-compute reuses the previously-loaded ONNX session
when re-entering the same variant — no reload cost.

---

## Capacity & resource consumption

Measured on Apple M4 (24 GB UMA, macOS 26.5, Python 3.13.3,
onnxruntime 1.26, rembg 2.0.75). Numbers per single inference of one
Monstree-size image (3024×4032). See `memory/perf_segmentation_s52.md`
for the full table.

### Wall-clock per inference

| Variant | CPU EP (4 threads) | CoreML CPUAndGPU | CoreML ALL (ANE, warm) |
|---|---|---|---|
| `birefnet-lite` | **6.7 s** | 50-234 s ⚠️ | TBD (15-25 min cold) |
| `birefnet-general` | **10.1 s** | >5 min ⚠️ | TBD (15-25 min cold) |
| `birefnet-dis` | **9.5 s** | >5 min ⚠️ | TBD (15-25 min cold) |

**Recommended path today is CPU EP.** Set `ONNX_FORCE_CPU=1` in your
shell or `.env`. The CoreML `CPUAndGPU` path is empirically broken
for swin transformers on the current onnxruntime macOS build (ORT's
CoreML EP fences subgraph boundaries off the ANE-eligible fused path
→ CPU↔GPU bouncing dominates). The CoreML `ALL` (with ANE) path is
the right destination but pays a 15-25 min first-time compile that
should be `--warmup`-baked separately.

### Memory (peak RSS during inference)

| Variant | Peak RSS | 8 GB UMA | 16 GB UMA | 24+ GB UMA |
|---|---|---|---|---|
| `birefnet-lite` | ~6.5 GB | tight (will swap) | comfortable | plenty |
| `birefnet-general` | ~8.6 GB | overflows | comfortable | plenty |
| `birefnet-dis` | ~8.9 GB | overflows | comfortable | plenty |

During ANE compile (one-time): peak ~17 GB regardless of variant —
the ANE compiler materializes multiple intermediate graph
representations simultaneously. On 8 GB Macs, `ANECompilerService`
swaps heavily but still completes (just slower).

### Disk

- This directory after `--all`: **~2.07 GB** (3 ONNX files + README).
- `~/Library/Caches/com.apple.e5rt.e5bundlecache/`: up to ~3× the
  ONNX size per variant, written on first CoreML+ANE load. With all
  three pre-warmed: expect ~6 GB cached.

### Network

- Initial download is one-shot: 213 + 927 + 929 = ~2 GB total.
- Downloads come from GitHub release artifacts (primary) with a
  HuggingFace fallback. No telemetry; no per-run network at
  inference time.

---

## Tweaking

The node exposes these knobs (Meshroom Attribute Editor or
`--<param> <value>` on the native CLI). Defaults in **bold**.

| Parameter | Values | Effect |
|---|---|---|
| `modelVariant` | `birefnet-general` *(bold)*, `birefnet-dis`, `birefnet-lite` | Swap which ONNX is loaded. Per-variant session cache + UID-bound, so swaps invalidate the node cache and regenerate masks. |
| `outputResolution` | `512`, **`1024`**, `2048` | Longest-edge resolution fed into the network. BiRefNet's preprocessor downscales any input to this. **1024 is the sweet spot.** 2048 quadruples ANE/GPU work for marginal mask-quality gain on natural images. 512 gives roughly 4× faster inference at the cost of softer edges. |
| `alphaMatting` | **`false`**, `true` | Run rembg's alpha-matting post-pass for refined edges (used by `general` and `dis` to produce smoother boundaries on hair/fur). Adds ~30-50% per-image latency. Off by default. |
| `maskFormat` | **`png`**, `exr` | PNG is 8-bit and matches AliceVision's `FeatureExtraction.maskExtension=png` default. EXR is 32-bit float in [0,1], useful if you want soft-mask probabilities downstream. Format choice does not affect inference cost. |
| `keepFilename` | **`true`**, `false` | `true` → `IMG_1024_mask.png` (matches `ImageMasking` convention); `false` → `<viewId>.png` (matches `aliceVision_imageSegmentation` upstream). Pick the convention your downstream node expects. |
| `verboseLevel` | `fatal`, `error`, `warning`, **`info`**, `debug`, `trace` | Logging level. `info` emits the acceptance marker `[SegmentationBiRefNet] Compute target: ...`. `debug` adds per-image timing. |

### Environment variable knobs

| Variable | Default | Effect |
|---|---|---|
| `U2NET_HOME` | `<repo>/ai-models` | Where rembg looks for ONNX files. The plugin sets this automatically; override if you want to share weights across projects. |
| `ONNX_FORCE_CPU` | unset | Set to `1` to pin onnxruntime to `CPUExecutionProvider` and skip the CoreML EP entirely. **Currently the recommended default** on this M4 / macOS 26.5 / ort 1.26 stack until the CoreML+ANE path is properly warmed. |
| `OMP_NUM_THREADS` | unset (ORT defaults) | onnxruntime's intra-op parallelism cap. On M4 (4 P-cores + 6 E-cores), `OMP_NUM_THREADS=4` is empirically the sweet spot — higher spills work onto E-cores and **increases** per-image latency. |

### Picking the right variant for your case

```
Are you on 8 GB UMA (M1/M2 base)?           → birefnet-lite
Are subject edges soft (hair, fur, foliage)? → birefnet-dis
Otherwise                                    → birefnet-general
```

### When to enable `alphaMatting`

- **Yes:** subject has hair, fur, fine plant matter, or wire-mesh
  edges AND you intend to use the mask as a continuous-tone alpha
  channel (compositing, not photogrammetry).
- **No:** photogrammetry use case. `FeatureExtraction` thresholds
  the mask anyway, so the post-pass adds latency without changing
  the downstream behavior.

---

## Adding another model

To make a new ONNX variant selectable from the dropdown:

1. Add an entry to `MODELS` in
   `plugins/ai-segmentation/scripts/download_models.py` (primary
   URL, fallback URL, on-disk basename).
2. Add the variant id to `modelVariant.values` in
   `plugins/ai-segmentation/nodes/aliceVision/SegmentationBiRefNet.py`.
3. If rembg already supports the new model name (check
   `from rembg.sessions import sessions_class`), no further code
   change is needed. Otherwise add an alias in
   `plugins/ai-segmentation/python/segmentation/session.py`
   `_ALIASES` mapping.
4. Run `python scripts/download_models.py --variant <new-id>` to
   fetch.
5. Run `pytest tests/python` to verify the plugin manifest still
   parses.

The plugin contract is documented in detail at
`docs/dev/plugin-system.md`; a third-party plugin can register its
own AI models without modifying this directory at all (use a separate
`<plugin>/models/` cache and a plugin-scoped `U2NET_HOME`).

---

## See also

- `docs/user/segmentation.md` — end-user node guide
- `docs/dev/segmentation-pipeline.md` — architecture + helper modules
- `docs/dev/apple-silicon-optimization.md` — UMA, Metal, ANE, CPU
  trade-offs with measured numbers
- `docs/dev/plugin-system.md` — how to build your own plugin
- `memory/perf_segmentation_s52.md` — raw perf data
- `instructions/ai_instruction.md` — original implementation spec
