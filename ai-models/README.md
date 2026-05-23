# `ai-models/` — BiRefNet CoreML packages for AliceVision-for-Mac

This directory holds the pre-converted CoreML model packages that the
`SegmentationBiRefNet` Meshroom node loads at runtime. **Two production
models ship here**, both produced from the official HuggingFace
checkpoints by the conversion pipeline at [`../models/`](../models/).

The plugin only loads CoreML `.mlpackage` files; the rembg / ONNX
Runtime backend was removed 2026-05-23.

| File | Size | Backbone | HF source | Default use |
|---|---|---|---|---|
| `BiRefNet_lite.mlpackage` | 90 MB | `swin_v1_t` | [`ZhengPeng7/BiRefNet_lite`](https://huggingface.co/ZhengPeng7/BiRefNet_lite) | **Default.** ~350 ms / 1024² frame on Apple Silicon GPU. M1/M2 8 GB UMA safe. |
| `BiRefNet.mlpackage` | 447 MB | `swin_v1_l` | [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) | Higher accuracy (hair, fine edges). ~980 ms / 1024² frame. |

Both packages:

- Are `mlprogram` format, **FP16** weights, fixed `[1, 3, 1024, 1024]`
  input.
- Target **macOS 14+**.
- Expect ImageNet-normalized RGB input (`mean=[0.485, 0.456, 0.406]`,
  `std=[0.229, 0.224, 0.225]`).
- Emit a `[1, 1, 1024, 1024]` sigmoid mask in `[0, 1]`.

## ⚠ Loading: `MLComputeUnits.cpuAndGPU` only

**Do not load these models with `.all` or `.cpuAndNeuralEngine`.** They
hang in `ANECompilerService` at compile time and never return. The
production path is:

```python
import coremltools as ct
mlmodel = ct.models.MLModel(
    "ai-models/BiRefNet_lite.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU,   # <-- mandatory
)
```

Reason: BiRefNet's `ASPPDeformable` decoder uses deformable convolution
v2. CoreML lowers it via `grid_sample`, which the Apple Neural Engine
compiler cannot plan. Full diagnosis + measurements in
[`../models/production_note.md`](../models/production_note.md).

---

## How to reproduce the conversion (anyone can do it)

The exact recipe used to produce the `.mlpackage` files in this folder.
You only need to re-run this if you want to:

- Pull newer BiRefNet weights from HuggingFace.
- Change the fixed input resolution (default `1024`).
- Switch to FP32 precision (default FP16).

### 0. Prerequisites

- macOS 14+ on Apple Silicon (M1/M2/M3/M4).
- Xcode Command Line Tools + Python 3.11 or 3.13.
- ~12 GB free disk space (HF checkpoints are ~620 MB combined, peak
  conversion RSS hits ~5 GB on the `general` model).
- ~30–45 minutes wall-clock for both variants on an M4.

### 1. Get the BiRefNet checkpoints

The conversion script reads the HF checkpoints from `../models/lite/`
and `../models/general/`. Each directory needs:

```
models/<variant>/
├── model.safetensors          # the weights
├── birefnet.py                # model code (mirrored from HF)
├── BiRefNet_config.py         # config class (mirrored from HF)
├── config.json
├── handler.py                 # HF inference handler (unused at convert time)
└── requirements.txt           # HF-side runtime deps
```

The simplest way to get them is `git lfs clone` from HuggingFace:

```bash
cd models
git lfs clone https://huggingface.co/ZhengPeng7/BiRefNet_lite lite
git lfs clone https://huggingface.co/ZhengPeng7/BiRefNet      general
```

(You can also use the `huggingface-cli download` flow if you have it
configured.)

### 2. Set up the Python environment

The repo already ships `models/.venv` with the right deps; activate
it:

```bash
source models/.venv/bin/activate
```

If you're starting clean, the conversion needs:

```bash
python -m venv models/.venv
source models/.venv/bin/activate
pip install --upgrade pip
pip install \
    'torch>=2.4'  'torchvision>=0.19' \
    'coremltools>=8.0' \
    'safetensors>=0.4' \
    'transformers>=4.40' \
    'numpy<2.0' \
    'Pillow' 'kornia' 'einops'
```

### 3. Run the converter (one command per variant)

```bash
python models/convert/convert_to_coreml.py lite
python models/convert/convert_to_coreml.py general
```

This produces `models/lite/BiRefNet_lite.mlpackage` and
`models/general/BiRefNet.mlpackage`. Each run takes ~70 s on the lite
model and ~125 s on the general model (M4, FP16 path).

The conversion script:

1. Patches `DeformableConv2d.forward` to use `grid_sample` instead of
   torchvision's `deform_conv2d` op (CoreML has no deformable-conv op).
   Numerically identical to within ~1e-5 abs error on FP16.
2. Loads the safetensors weights into a fresh `BiRefNet` PyTorch model.
3. Wraps the model so it returns only the final sigmoid mask.
4. Traces at fixed `1024×1024`.
5. Converts via `coremltools.convert(..., convert_to='mlprogram',
   compute_precision=FLOAT16, compute_units=CPU_AND_GPU,
   minimum_deployment_target=macOS14)`.

> **Critical**: do NOT omit `compute_units=ct.ComputeUnit.CPU_AND_GPU`
> from `ct.convert(...)`. Without it, coremltools tries to compile an
> ANE plan at save time, hangs the lite variant for ~80 minutes, and
> kills the general variant outright. The flag exists in
> `convert_to_coreml.py:149`; do not remove it.

### 4. Validate the conversion

The validator runs PyTorch and CoreML on the same random input and
reports the diff:

```bash
python models/convert/validate_coreml.py lite      --compute-units CPU_AND_GPU
python models/convert/validate_coreml.py general   --compute-units CPU_AND_GPU
```

Expected: `Mask diff: max ≈ 3e-5, mean ≈ 5e-6`, `IoU@0.5 = 0.999+`. The
binary mask is byte-identical to PyTorch within sigmoid noise.

### 5. Benchmark + render a demo mask

```bash
python models/convert/bench_and_demo.py lite      sample.jpg
python models/convert/bench_and_demo.py general   sample.jpg
```

For each compute unit it prints `load=…s predict=…ms` and saves a PNG
of the predicted mask next to the model package.

Reference numbers on M-series (from `models/production_note.md`):

|        | `cpuOnly` | `cpuAndGPU` |
|--------|----------:|------------:|
| lite    | ~750 ms | **~350 ms** |
| general | ~2150 ms | **~980 ms** |

### 6. Stage them in this directory

```bash
cp -R models/lite/BiRefNet_lite.mlpackage ai-models/
cp -R models/general/BiRefNet.mlpackage   ai-models/
```

`ai-models/*.mlpackage` is git-ignored by default (see [`.gitignore`](../.gitignore)
"AI segmentation models" section) — the 537 MB combined size exceeds
GitHub's 100 MB per-file hard limit without git-lfs. Each operator
re-runs the conversion above (or downloads the packages from an
internal distribution channel) for their working tree. The conversion
is deterministic on a given torch/coremltools version, so two operators
on the same toolchain produce byte-identical packages.

---

## File-by-file convert-pipeline tour

| Path | Role |
|---|---|
| `../models/lite/birefnet.py`, `../models/general/birefnet.py` | Mirrored BiRefNet model code from HF. The conversion script imports these dynamically; the per-variant copy is what makes the variants self-contained. |
| `../models/lite/BiRefNet_config.py`, `../models/general/BiRefNet_config.py` | The matching `BiRefNetConfig` class. |
| `../models/convert/deformable_patch.py` | `grid_sample`-based reimplementation of `torchvision.ops.deform_conv2d`. Two modes: `grid_sample` (numerically faithful, used) and `plain` (drops the deformable behavior — kept around as a debugging ablation). |
| `../models/convert/convert_to_coreml.py` | The end-to-end converter. Reads HF checkpoint → patches deformable conv → traces → `ct.convert` → saves `.mlpackage`. |
| `../models/convert/validate_coreml.py` | Compares CoreML output against the PyTorch reference. |
| `../models/convert/bench_and_demo.py` | Per-compute-unit benchmark + demo mask render. |
| `../models/convert/compare_modes.py` | `grid_sample` vs `plain` deformable-conv ablation (mask quality study). |
| `../models/convert/test_deformable_patch.py` | Unit test asserting the `grid_sample` patch matches `torchvision.ops.deform_conv2d` numerically. |
| `../models/production_note.md` | Apple-Silicon production notes — perf table, compute-unit policy, ANE root cause. **Read this before changing anything in the conversion pipeline.** |

---

## Why is the Apple Neural Engine off the table?

Quick version (full version in
[`../models/production_note.md`](../models/production_note.md)):

1. BiRefNet's decoder uses `ASPPDeformable`. The trained weights have
   non-zero `offset_conv` / `modulator_conv` parameters — dropping the
   deformable behavior produces visibly broken masks (holes inside
   objects, edge halos) even though binary-mask IoU stays ~0.95.
2. CoreML has no deformable-conv op. We lower it via per-kernel-position
   `grid_sample` calls (~240 of them per `ASPPDeformable` block).
3. The Apple Neural Engine compiler cannot lower `grid_sample` at all.
   `ANECCompile` produces `Error in building plan` and the model load
   call hangs inside `com.apple.anef.p3`.
4. Reaching the ANE would require **retraining** the decoder with plain
   `ASPP` (no deformable conv). Out of scope here.

The GPU path is fast enough (~350 ms / frame on `lite`) that this
isn't a battery/thermal limit for our segmentation workload.

---

## Validation status of the bundled models

The packages currently in this directory were produced by:

| | |
|---|---|
| coremltools | `8.x` |
| PyTorch | `2.4+` |
| Source | HF checkpoints `ZhengPeng7/BiRefNet` and `ZhengPeng7/BiRefNet_lite` at convert time |
| Convert command | `python models/convert/convert_to_coreml.py {lite,general}` (no flag overrides) |
| Validation | `validate_coreml.py {lite,general} --compute-units CPU_AND_GPU` → IoU@0.5 = 1.000, max abs diff ~3e-5 |
| Benchmark | `bench_and_demo.py …` on M-series → CPU+GPU column matches the table above ±5 % |

Re-run `validate_coreml.py` if you touch anything in `models/convert/`
or replace the weights in `models/{lite,general}/`.
