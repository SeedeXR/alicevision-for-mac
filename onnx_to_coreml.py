#!/usr/bin/env python3
"""
convert_to_coreml.py

Converts ~/ai-models/birefnet-general-lite-static-512.onnx to CoreML
via onnxruntime's CoreML Execution Provider (no coremltools needed).

onnxruntime handles ONNX → CoreML internally.
A warmup inference run triggers the one-time ANE compilation.
The compiled model is cached by macOS automatically.

Run: python3 convert_to_coreml.py
"""

import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "ai-models", "birefnet-general-lite-static-512.onnx")
INPUT_SIZE = 512


# ── banner ────────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("BiRefNet-lite  →  CoreML via onnxruntime CoreML EP")
print("=" * 55)
print()


# ── check model file ──────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at:")
    print(f"  {MODEL_PATH}")
    print()
    print("Make sure this file exists in ai-models/ under the repo root.")
    sys.exit(1)

size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"Model  : {MODEL_PATH}")
print(f"Size   : {size_mb:.1f} MB")
print()

if size_mb < 10:
    print(f"ERROR: File too small ({size_mb:.1f} MB) — likely corrupted.")
    print("Re-download the model and re-run prepare_model.py first.")
    sys.exit(1)


# ── check dependencies ───────────────────────────────────────────────────────
print("Checking dependencies...")
try:
    import onnxruntime as ort
    print(f"  onnxruntime : {ort.__version__}  OK")
except ImportError:
    print("  ERROR: onnxruntime not installed.")
    print("  Fix : pip3 install onnxruntime --break-system-packages")
    sys.exit(1)

try:
    import onnx
    print(f"  onnx        : {onnx.__version__}  OK")
except ImportError:
    print("  ERROR: onnx not installed.")
    print("  Fix : pip3 install onnx --break-system-packages")
    sys.exit(1)

try:
    import onnxsim
    print(f"  onnxsim     : available  OK")
    HAS_ONNXSIM = True
except ImportError:
    print("  onnxsim     : not installed (will try without simplification)")
    print("  Tip : pip3 install onnxsim --break-system-packages")
    HAS_ONNXSIM = False

print()

# Confirm CoreML EP is available
available = ort.get_available_providers()
print(f"  Providers   : {available}")

if "CoreMLExecutionProvider" not in available:
    print()
    print("  ERROR: CoreML Execution Provider not available.")
    print("  The standard onnxruntime wheel includes it on macOS.")
    print("  Fix : pip3 install --upgrade onnxruntime --break-system-packages")
    sys.exit(1)

print(f"  CoreML EP   : available  OK")
print()


# ── simplify model (fixes Split/shape-inference issues with CoreML EP) ────────
if HAS_ONNXSIM:
    print("Simplifying ONNX model with onnxsim...")
    try:
        raw_model = onnx.load(MODEL_PATH)
        simplified, ok = onnxsim.simplify(raw_model)
        if ok:
            print("  Simplification OK — using simplified model")
            import tempfile, atexit
            _tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            onnx.save(simplified, _tmp.name)
            _tmp.close()
            atexit.register(os.unlink, _tmp.name)
            SESSION_MODEL = _tmp.name
        else:
            print("  Simplification returned check=False — using original model")
            SESSION_MODEL = MODEL_PATH
    except Exception as e:
        print(f"  WARNING: onnxsim failed ({e}) — using original model")
        SESSION_MODEL = MODEL_PATH
    print()
else:
    SESSION_MODEL = MODEL_PATH


# ── create CoreML session ─────────────────────────────────────────────────────
# onnxruntime internally converts ONNX → CoreML MLProgram at session creation.
# CPUAndNeuralEngine targets the ANE and avoids Metal GPU command-buffer thrashing.

print("Creating CoreML session...")
print("onnxruntime is converting ONNX → CoreML internally.")
print("This may take a few minutes on first run.")
print()

coreml_options = {
    "ModelFormat":             "MLProgram",       # ANE-eligible format
    "MLComputeUnits":          "CPUAndNeuralEngine", # target ANE, skip Metal GPU
    "RequireStaticInputShapes": "1",              # enables op fusion at compile time
}

providers = [
    ("CoreMLExecutionProvider", coreml_options),
    "CPUExecutionProvider",                       # fallback if CoreML fails
]

t0 = time.time()

try:
    session = ort.InferenceSession(SESSION_MODEL, providers=providers)
except Exception as e:
    print(f"WARNING: CoreML EP failed ({e})")
    print("Retrying with CPU EP only...")
    print()
    try:
        session = ort.InferenceSession(SESSION_MODEL, providers=["CPUExecutionProvider"])
    except Exception as e2:
        print(f"ERROR: Session creation failed: {e2}")
        sys.exit(1)

session_time = time.time() - t0
active = session.get_providers()[0]

print(f"Session created in {session_time:.1f}s")
print(f"Active provider : {active}")
print()

if "CoreML" not in active:
    print("WARNING: CoreML EP did not load — running on CPU fallback.")
    print("Inference will still work but ANE will not be used.")
    print()


# ── warm-up run to trigger ANE compilation ────────────────────────────────────
# The ANE model compilation happens on the first inference call.
# This is a one-time cost — macOS caches the compiled model automatically.
# Subsequent runs skip this step entirely.

input_name = session.get_inputs()[0].name
print(f"Input name : {input_name}")
print(f"Input shape: 1 x 3 x {INPUT_SIZE} x {INPUT_SIZE}")
print()
print("Running warmup inference to trigger ANE compilation...")
print("NOTE: First run can take 5–25 minutes for ANE compile.")
print("      Do not interrupt. This only happens once.")
print("      macOS caches the compiled model automatically.")
print()

dummy = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)

t1 = time.time()
try:
    outputs = session.run(None, {input_name: dummy})
except Exception as e:
    print(f"ERROR during inference: {e}")
    print()
    print("Common causes:")
    print("  - Input name mismatch — check model with Netron (https://netron.app)")
    print("  - Shape mismatch — confirm model was saved with static 512x512 input")
    print("  - Memory pressure — close other apps and retry")
    sys.exit(1)

warmup_time = time.time() - t1

print(f"Warmup inference done in {warmup_time:.1f}s")
print()


# ── second timed run (should be fast — ANE now compiled and cached) ───────────
print("Running timed inference (should be fast now)...")

t2 = time.time()
outputs = session.run(None, {input_name: dummy})
timed = time.time() - t2

print(f"Timed inference : {timed:.2f}s")
print()


# ── output shape ─────────────────────────────────────────────────────────────
print(f"Output tensors  : {len(outputs)}")
for i, o in enumerate(outputs):
    print(f"  [{i}] shape: {o.shape}  dtype: {o.dtype}")
print()


# ── final report ──────────────────────────────────────────────────────────────
print("=" * 55)
print("RESULTS")
print("=" * 55)
print(f"Active provider   : {active}")
print(f"Session load time : {session_time:.1f}s")
print(f"Warmup time       : {warmup_time:.1f}s  (one-time ANE compile)")
print(f"Timed inference   : {timed:.2f}s  (steady-state speed)")
print()

if "CoreML" in active:
    print("PASS — CoreML EP is active")
    if timed < 5.0:
        print("PASS — Inference fast — ANE fusion likely working")
    elif timed < 10.0:
        print("OK   — Inference moderate — ANE may be partially fused")
    else:
        print("WARN — Inference slow — ANE may not have fused this model")
        print("       CPU EP (no CoreML) may be faster for this architecture")
else:
    print("WARN — Running on CPU fallback")
    print("       CoreML EP failed to load for this model")

print()
print("To verify ANE dispatch, run in a second terminal while inferring:")
print("  sudo powermetrics --samplers ane -i 500 | grep 'ANE Power'")
print()
print("Next step:")
print("  python3 test_inference.py")