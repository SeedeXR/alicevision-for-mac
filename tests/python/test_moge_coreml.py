"""Regression tests for the CoreML MoGe binary (Phase 14.8).

The Phase 14.7 honest stub at src/native_binaries/main_moGe.cpp was
replaced by a real CoreML-backed implementation at
src/moge/aliceVision_shim/main_moGe.cpp, wrapping the user's
ai-models/moge2_504x672_t1728.mlpackage (DINOv2 ViT-B/14).

Three lightweight always-on tests + one E2E test gated behind
RUN_MOGE_COREML=1 (loads + runs the model, ~228 ms on ANE).
"""
from __future__ import annotations

import json
import os
import subprocess
import struct
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BINARY = REPO_ROOT / "build" / "aliceVision_moGe"
MLPACKAGE = REPO_ROOT / "ai-models" / "moge2_504x672_t1728.mlpackage"

RUN_E2E = os.environ.get("RUN_MOGE_COREML") == "1"


def test_binary_built():
    assert BINARY.exists(), f"binary missing: {BINARY}"
    assert os.access(BINARY, os.X_OK)


def test_mlpackage_present():
    assert MLPACKAGE.is_dir(), f"mlpackage missing: {MLPACKAGE}"
    assert (MLPACKAGE / "Manifest.json").exists()
    assert (MLPACKAGE / "Data" / "com.apple.CoreML" / "model.mlmodel").exists()


def test_binary_help_mentions_coreml():
    """Help output advertises the CoreML model path, not the stub language."""
    proc = subprocess.run(
        [str(BINARY), "-h"], capture_output=True, text=True, timeout=30
    )
    out = (proc.stdout + proc.stderr).lower()
    # The new binary mentions modelPath / .mlpackage. The old stub said
    # "constant-depth placeholders" — that language must NOT be present.
    assert "modelpath" in out or ".mlpackage" in out, (
        f"binary -h doesn't mention modelPath / .mlpackage:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "constant-depth placeholders" not in out, (
        "binary -h still advertises the Phase 14.7 stub language"
    )


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set RUN_MOGE_COREML=1 to run the full inference test",
)
def test_coreml_inference_emits_real_depth(tmp_path):
    """Drive full MoGe inference on a real photo + sanity-check the EXR."""
    candidates = list((REPO_ROOT / "dataset_monstree").rglob("*.JPG"))
    candidates += list((REPO_ROOT / "dataset_monstree").rglob("*.jpg"))
    if not candidates:
        pytest.skip("no dataset_monstree photos for inference smoke test")
    img = candidates[0]

    sfm = tmp_path / "input.sfm"
    sfm.write_text(json.dumps({
        "version": ["1", "0", "0"],
        "views": [{
            "viewId": "100",
            "poseId": "100",
            "frameId": "0",
            "intrinsicId": "200",
            "width": "1280",
            "height": "720",
            "path": str(img),
            "metadata": {},
        }],
        "intrinsics": [{
            "intrinsicId": "200",
            "width": "1280",
            "height": "720",
            "sensorWidth": "6.16",
            "sensorHeight": "4.62",
            "serialNumber": "smoke",
            "type": "pinhole",
            "pxInitialFocalLength": "-1",
            "pxFocalLength": "-1",
            "principalPoint": ["640", "360"],
        }],
    }))
    out_dir = tmp_path / "depth"
    out_dir.mkdir()

    proc = subprocess.run(
        [
            str(BINARY),
            "--input", str(sfm),
            "--output", str(out_dir),
            "--modelPath", str(MLPACKAGE),
            "--outputNormals", "1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"binary exited {proc.returncode}\nstdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    depth_exr = out_dir / "100_depth.exr"
    normals_exr = out_dir / "100_normals.exr"
    assert depth_exr.exists(), "depth EXR not written"
    assert normals_exr.exists(), "normals EXR not written"
    # File-size sanity: a 504x672 float32 EXR uncompressed is ~1.35 MB;
    # PIZ-compressed real depth maps come out at 100-300 KB. The Phase
    # 14.7 stub emitted 64x48 constant-1.0m maps that compressed to a
    # tiny ~1 KB. Anything below 10 KB is suspicious.
    assert depth_exr.stat().st_size > 10_000, (
        f"depth EXR is suspiciously small ({depth_exr.stat().st_size} B) — "
        "did we accidentally revert to the stub?"
    )
    # Parse the EXR header to confirm 672x504 resolution. We don't need
    # the OpenEXR Python lib — read the file's data-window directly out
    # of the OpenEXR header chunk. Format ref: openexr.com/en/latest.
    # Quick proxy: search the header bytes for the dataWindow ints, but
    # we'll just check the file starts with the OpenEXR magic.
    with open(depth_exr, "rb") as f:
        magic = f.read(4)
    assert magic == b"\x76\x2f\x31\x01", (
        f"depth EXR magic mismatch: {magic.hex()}"
    )
