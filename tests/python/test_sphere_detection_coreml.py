"""End-to-end regression tests for the CoreML sphereDetection binary.

The binary replaces upstream's ONNX-Runtime-based aliceVision_sphereDetection
with av::sphere::CoreMLSphereDetector wrapping ai-models/yolov8n.mlpackage.

Heavy E2E test (requires the .mlpackage to load on first invocation, which
compiles it to a temp .mlmodelc) is gated behind RUN_SPHERE_DETECTION=1 to
avoid slowing the default test loop. Lightweight static asserts (binary
exists, model file present) always run.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BINARY = REPO_ROOT / "build" / "aliceVision_sphereDetection"
MLPACKAGE = REPO_ROOT / "ai-models" / "yolov8n.mlpackage"

RUN_E2E = os.environ.get("RUN_SPHERE_DETECTION") == "1"


def test_binary_built():
    """aliceVision_sphereDetection is built and executable."""
    assert BINARY.exists(), f"binary missing: {BINARY} (run cmake --build .)"
    assert os.access(BINARY, os.X_OK), f"binary not executable: {BINARY}"


def test_mlpackage_present():
    """The CoreML YOLOv8 model is in place at the expected path."""
    assert MLPACKAGE.is_dir(), f"mlpackage missing: {MLPACKAGE}"
    assert (MLPACKAGE / "Manifest.json").exists()
    assert (MLPACKAGE / "Data" / "com.apple.CoreML" / "model.mlmodel").exists()


def test_binary_help_mentions_coreml():
    """`-h` output advertises the CoreML model path, not ONNX."""
    proc = subprocess.run(
        [str(BINARY), "-h"], capture_output=True, text=True, timeout=30
    )
    out = (proc.stdout + proc.stderr).lower()
    # The patched help text says "CoreML .mlpackage ..." for --modelPath.
    assert "coreml" in out or ".mlpackage" in out, (
        f"binary -h output doesn't mention CoreML/.mlpackage:\n{proc.stdout}"
    )
    # And does NOT advertise ONNX.
    assert "onnx" not in out, f"-h output unexpectedly mentions ONNX:\n{proc.stdout}"


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set RUN_SPHERE_DETECTION=1 to run the full CoreML inference test",
)
def test_coreml_inference_produces_upstream_format_json(tmp_path):
    """Drive the full inference path on a real photo and verify output."""
    # Use any photo from dataset_monstree (any image works for smoke).
    candidates = list((REPO_ROOT / "dataset_monstree").rglob("*.JPG"))
    candidates += list((REPO_ROOT / "dataset_monstree").rglob("*.jpg"))
    if not candidates:
        pytest.skip("no dataset_monstree photos available for smoke test")
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
    out_json = tmp_path / "out.json"

    proc = subprocess.run(
        [
            str(BINARY),
            "--input", str(sfm),
            "--modelPath", str(MLPACKAGE),
            "--autoDetect", "1",
            "--output", str(out_json),
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
    assert out_json.exists(), "output JSON not written"
    payload = json.loads(out_json.read_text())
    # Upstream format: shapes[0].observations is keyed by view ID.
    assert "shapes" in payload
    assert len(payload["shapes"]) == 1
    shape = payload["shapes"][0]
    assert shape["type"] == "Circle"
    assert "observations" in shape
    # The observations dict may be empty (low confidence + minScore filter)
    # or contain entries — either is valid; we just check schema.
    for view_id, obs in shape["observations"].items():
        assert "center" in obs
        assert "x" in obs["center"] and "y" in obs["center"]
        assert "radius" in obs
        assert "score" in obs
