# `ai-models/` — CoreML packages for alicevision-for-mac

This directory holds the pre-converted CoreML `.mlpackage` files that the
Mac port's native ML binaries and segmentation plugin load at runtime.
The packages were produced from upstream PyTorch / ONNX checkpoints by
the conversion pipelines documented in [`../models/`](../models/).

The entire AliceVision-for-Mac stack is **CoreML-only** as of 2026-05-23.
The rembg / ONNX Runtime backend was removed; ONNX Runtime is not a
runtime dep of any binary in this build.

---

## Models in this directory

| File | Size | Backbone | Native binary | Compute units | Per-call latency |
|---|---|---|---|---|---|
| `BiRefNet_lite.mlpackage` | 90 MB | `swin_v1_t` | (Python plugin) | `cpuAndGPU` | ~350 ms / 1024² |
| `BiRefNet.mlpackage` | 447 MB | `swin_v1_l` | (Python plugin) | `cpuAndGPU` | ~980 ms / 1024² |
| `yolov8n.mlpackage` | 13 MB | YOLOv8n | `aliceVision_sphereDetection` | `all` (ANE) | ~30 ms / 640² |
| `moge2_504x672_t1728.mlpackage` | 187 MB | DINOv2 ViT-B/14 | `aliceVision_moGe` | `all` (partial ANE) | ~228 ms / 504×672 |
| `tiny_roma_v1_480x640.mlpackage` | 5.5 MB | XFeat + matcher | `aliceVision_matchMasking` | **`cpuAndGPU` (NOT `.all`)** | ~12 ms / 480×640 pair |

**Compute-units pattern across the four model families** — each is a
distinct ANE outcome and the right load setting differs per model:

| Model | What ANE does | Recommendation |
|---|---|---|
| BiRefNet | Compile *hangs* in `ANECompilerService` (graph not lowerable). | `cpuAndGPU` (mandatory). |
| YOLOv8n | Entire graph lowers to ANE; **3× faster than GPU**. | `all` (= ANE). |
| MoGe-2 | Partial — some ops fall back to CPU. **1.2×** over GPU. | `all` (partial ANE). |
| TinyRoMa | Lowers, but two `grid_sample` ops force CPU↔ANE handoffs. **4× SLOWER** than CPU. | `cpuAndGPU` (NOT `.all`). |

**Remember TinyRoMa next time you see a model with `grid_sample` in its
decoder.** "Compiles to ANE" is not the same as "runs faster on ANE."

---

## Per-model usage details

### 1. BiRefNet — foreground / object segmentation

**Used by** Meshroom's `SegmentationBiRefNet` Python descriptor at
`meshroom-mac/nodes/aliceVision/SegmentationBiRefNet.py` (and the
plugin at `plugins/ai-segmentation/`).

**Input contract**: `mlprogram`, FP16, fixed `[1, 3, 1024, 1024]`,
ImageNet-normalized RGB (`mean=[0.485, 0.456, 0.406]`,
`std=[0.229, 0.224, 0.225]`).

**Output**: `[1, 1, 1024, 1024]` sigmoid mask in `[0, 1]`.

**Don't pass `.all`**. The graph is not lowerable for ANE — Apple's
ANECompilerService hangs trying. Use `cpuAndGPU` explicitly. See
[`../models/production_note.md`](../models/production_note.md) for the
incident report.

Two variants ship: the lite (`swin_v1_t` backbone, 90 MB) is the
default; the full (`swin_v1_l`, 447 MB) trades 3× the runtime for
sharper hair/edge masks. The plugin picks the lite by default; set
`SEGMENTATION_BIREFNET_MODEL=full` to override.

### 2. YOLOv8n — sphere detection for photometric stereo

**Used by** the `aliceVision_sphereDetection` C++ binary (Phase 14.4b
CoreML port). Source: `src/sphere_detection/`. Replaced upstream's
ONNX-Runtime-based binary.

**Input contract**: `image`-type CoreML input, 640×640 BGRA →
auto-converted to RGB by CoreML at predict time. Two double scalar
inputs `iouThreshold`, `confidenceThreshold` for the baked-in NMS.

**Output**: Vision-style `coordinates` (Nx4 normalized cx,cy,w,h) +
`confidence` (NxC class probs). NMS is baked in.

**Use `.all`**: entire graph runs on ANE with no handoffs. ~3× faster
than GPU. See [`../memory/yolo_coreml_sphere_detection.md`](../memory/yolo_coreml_sphere_detection.md).

Integration: the C++ wrapper is `src/sphere_detection/src/CoreMLSphereDetector.mm`
(public header at `src/sphere_detection/include/av/sphere/CoreMLSphereDetector.hpp`).
Loads `.mlpackage` on first call, compiles to a temp `.mlmodelc`, caches.

### 3. MoGe-2 — monocular geometry (depth + normals)

**Used by** the `aliceVision_moGe` C++ binary (Phase 14.8). Source:
`src/moge/`. Powers the `cameraTrackingDepth.mg` template's depth-prior
arm.

**Input contract**: `MultiArray`, fixed `[1, 3, 504, 672]` float32 RGB
in `[0, 1]`. The conversion baked in plain resize (no letterbox); the
wrapper does `cv::resize` to match.

**Outputs**: `points [1, 504, 672, 3]` (XYZ per pixel; Z is forward),
`normal [1, 504, 672, 3]` (surface normal), `mask [1, 504, 672]`
(validity), `metric_scale [1]` (relative → meters multiplier).

**Use `.all`**: model partially runs on ANE (~228 ms vs ~384 ms CPU on
M-series). ANE-vs-GPU gain is only ~1.2× because not every op has an
ANE kernel in CoreML 9.0; the system schedules what it can.

The binary writes `<viewId>_depth.exr` + (optional) `<viewId>_normals.exr`
to the output directory at the model's native 504×672 resolution.
Masked-invalid pixels are 0.0 (matches `DepthMapTracksInjecting`'s
"0 = no data" convention).

### 4. TinyRoMa — dense optical-flow matcher

**Used by** the `aliceVision_matchMasking` C++ binary (Phase 14.9).
Source: `src/roma/`. Powers the `cameraTrackingRoma.mg` template's
dense-matching arm. Replaced the Phase 14.7 honest pass-through.

**Input contract**: TWO `MultiArray` inputs `im_A` and `im_B`, each
`[1, 3, 480, 640]` float32 RGB in `[0, 1]`. Width/height **must be
multiples of 32** (XFeat strides down by 32). Re-run `convert/convert_roma.py`
to get a different fixed shape.

**Outputs**:
- `coarse_flow [1, 2, 60, 80]` — normalized `[-1, 1]` A→B flow at stride 8.
- `coarse_certainty [1, 1, 60, 80]` — match certainty *logits* (no sigmoid).
- `fine_flow [1, 2, 120, 160]` — same at stride 4.
- `fine_certainty [1, 1, 120, 160]` — fine logits.

To convert flow to pixel coords: `x_B = (flow_x + 1) * W / 2`.
To convert certainty to a probability: `sigmoid(logit) = 1 / (1 + exp(-x))`.
Our wrapper does the sigmoid before writing EXRs.

**CRITICAL: DO NOT use `MLComputeUnits.all`.** TinyRoMa has two
`grid_sample` ops in the decoder, each forcing a CPU↔ANE memory
handoff. On a tiny model like this, handoff overhead dominates and the
result is **4× SLOWER than CPU**. `cpuAndGPU` is the production target
(~12 ms / pair at 480×640 on M-series GPU).

The matchMasking binary writes per-pair EXRs:
- `outputWarpFolder/<viewIdA>_<viewIdB>_{coarse,fine}_flow.exr`
  (3-channel R=flow_x, G=flow_y, B=0)
- `outputCertaintyFolder/<viewIdA>_<viewIdB>_{coarse,fine}_certainty.exr`
  (1-channel sigmoid'd in [0, 1])

If `--masksFolder` is provided (with `<viewId>.png` per view), certainty
is zeroed at pixels where either view's mask is invalid.

---

## How models are discovered at runtime

Each native binary searches for its `.mlpackage` in this order:

1. CLI `--modelPath` override.
2. `$ALICEVISION_<NAME>_MLPACKAGE` env var. Specifically:
   - `ALICEVISION_MOGE_MLPACKAGE` → MoGe.
   - `ALICEVISION_ROMA_MLPACKAGE` → Roma.
   - sphereDetection requires `--modelPath` (descriptor passes it).
3. `$ALICEVISION_ROOT/ai-models/<file>.mlpackage`.
4. Walk-up from `cwd` for `ai-models/<file>.mlpackage` (handy when
   running from the repo).

For the `.app` bundle (Phase 7 packaging), `ai-models/` is co-located
with the binaries via the bundler script, so the walk-up branch picks
it up at runtime.

For Meshroom — set `ALICEVISION_ROOT` to the repo before launching.
The provided `scripts/run_meshroom.sh` already does this.

---

## How to add a new model

The pattern across the four existing wrappers (`src/sphere_detection/`,
`src/moge/`, `src/roma/`) is uniform:

1. **Convert** your PyTorch/ONNX model to a fixed-shape `.mlpackage`
   using `coremltools`. Place it under `ai-models/`. Include FP16
   weights unless you need FP32 — the size and runtime wins are large
   and the accuracy hit on M-series GPU is usually < 0.1% relative.
2. **Inspect the spec** to confirm input/output schemas:
   ```python
   from coremltools.proto import Model_pb2
   spec = Model_pb2.Model()
   spec.ParseFromString(open('your_model.mlpackage/Data/com.apple.CoreML/model.mlmodel', 'rb').read())
   for inp in spec.description.input: ...
   for out in spec.description.output: ...
   ```
3. **Benchmark each compute_units** (cpuOnly / cpuAndGPU /
   cpuAndNeuralEngine / all) in a subprocess-isolated harness. CoreML's
   first call is slow due to compilation; warm up 2× before the
   measured run. Pick the fastest. Document the result in this README's
   matrix.
4. **Write the wrapper** at `src/<name>/` mirroring the existing layout:
   - `include/av/<name>/CoreML<Name>Runner.hpp` — pure-C++ public header.
   - `src/CoreML<Name>Runner.mm` — Objective-C++ implementation that
     loads the `.mlpackage`, validates the I/O schema at construct time,
     and exposes one `predict(...)` method per call site.
   - `CMakeLists.txt` — `add_library(... STATIC)` linking
     `CoreML`/`Foundation` frameworks and OpenCV.
5. **Integrate** by adding the wrapper to the top-level CMakeLists and
   pointing the consumer binary at it.
6. **Test** at `tests/python/test_<name>_coreml.py`: 3 always-on
   (binary exists, model present, help advertises CoreML), 1 gated E2E
   behind `RUN_<NAME>_COREML=1` that drives full inference + checks
   output sanity.

The two ObjC++ gotchas across all wrappers:
- Don't `#include <opencv2/opencv.hpp>` in a `.mm` file — pulls in
  `opencv2/stitching/*` which clashes with macOS SDK ObjC symbols.
  Use focused submodule headers (`core.hpp`, `imgcodecs.hpp`, `imgproc.hpp`).
- `MLMultiArray.strides` is `NSArray<NSNumber*>`, NOT `NSData`.
  Access via `[arr.strides[i] integerValue]`.

---

## Provenance

| Model | Upstream | Conversion script | Conversion date |
|---|---|---|---|
| BiRefNet (×2) | [HuggingFace ZhengPeng7](https://huggingface.co/ZhengPeng7) | [`../models/convert_birefnet.py`](../models/convert_birefnet.py) | 2026-05-23 |
| YOLOv8n | Ultralytics | (Ultralytics export with NMS) | 2026-05-24 |
| MoGe-2 | [Microsoft MoGe](https://github.com/microsoft/MoGe) (torch==2.5.1) | (user-converted via coremltools 9.0, TorchScript dialect) | 2026-05-24 |
| TinyRoMa | [Parskatt/RoMa](https://github.com/Parskatt/RoMa) + [verlab/accelerated_features](https://github.com/verlab/accelerated_features) (XFeat) | `convert/convert_roma.py` (4 static-shape patches; see ai-models/README "What had to be patched") | 2026-05-24 |

Don't commit the `.mlpackage` directories to git unless they're small;
the BiRefNet full and MoGe-2 are large enough to balloon the repo. Use
Git LFS or a release attachment if you need to ship them with a tag.
