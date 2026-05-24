"""Regression tests for the TinyRoMa-backed matchMasking binary (Phase 14.9).

The Phase 14.7 honest pass-through at src/native_binaries/main_matchMasking.cpp
was replaced by a real CoreML-backed implementation at
src/roma/aliceVision_shim/main_matchMasking.cpp, wrapping the user's
ai-models/tiny_roma_v1_480x640.mlpackage (TinyRoMa, 5.5 MB FP16).

Three lightweight always-on tests + one E2E test gated behind
RUN_ROMA_COREML=1 (loads + runs the model on one image pair, ~3 s total
on first call due to CoreML compile; subsequent calls ~12 ms each).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BINARY = REPO_ROOT / "build" / "aliceVision_matchMasking"
MLPACKAGE = REPO_ROOT / "ai-models" / "tiny_roma_v1_480x640.mlpackage"

RUN_E2E = os.environ.get("RUN_ROMA_COREML") == "1"


def test_binary_built():
    assert BINARY.exists(), f"binary missing: {BINARY}"
    assert os.access(BINARY, os.X_OK)


def test_mlpackage_present():
    assert MLPACKAGE.is_dir(), f"mlpackage missing: {MLPACKAGE}"
    assert (MLPACKAGE / "Manifest.json").exists()
    assert (MLPACKAGE / "Data" / "com.apple.CoreML" / "model.mlmodel").exists()


def test_binary_help_mentions_coreml_roma():
    """Help output advertises the CoreML Roma model — not the Phase 14.7 stub."""
    proc = subprocess.run(
        [str(BINARY), "-h"], capture_output=True, text=True, timeout=30
    )
    out = (proc.stdout + proc.stderr).lower()
    # New binary mentions modelPath / Roma. Stub said "pass-through".
    assert "modelpath" in out or "roma" in out, (
        f"binary -h doesn't mention modelPath / Roma:\n{proc.stdout}"
    )
    assert "pass-through forwarding" not in out, (
        "binary -h still advertises the Phase 14.7 pass-through stub"
    )


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set RUN_ROMA_COREML=1 to run the full Roma inference test",
)
def test_coreml_inference_emits_flow_and_certainty(tmp_path):
    """Drive full Roma matching on a real photo pair + sanity-check the EXRs."""
    candidates = list((REPO_ROOT / "dataset_monstree").rglob("*.JPG"))
    candidates += list((REPO_ROOT / "dataset_monstree").rglob("*.jpg"))
    if len(candidates) < 2:
        pytest.skip("need ≥ 2 dataset_monstree photos for the pair test")
    img1, img2 = candidates[0], candidates[1]

    sfm = tmp_path / "input.sfm"
    sfm.write_text(json.dumps({
        "version": ["1", "0", "0"],
        "views": [
            {
                "viewId": "100", "poseId": "100", "frameId": "0",
                "intrinsicId": "200", "width": "1280", "height": "720",
                "path": str(img1), "metadata": {},
            },
            {
                "viewId": "101", "poseId": "101", "frameId": "1",
                "intrinsicId": "200", "width": "1280", "height": "720",
                "path": str(img2), "metadata": {},
            },
        ],
        "intrinsics": [{
            "intrinsicId": "200", "width": "1280", "height": "720",
            "sensorWidth": "6.16", "sensorHeight": "4.62",
            "serialNumber": "smoke", "type": "pinhole",
            "pxInitialFocalLength": "-1", "pxFocalLength": "-1",
            "principalPoint": ["640", "360"],
        }],
    }))
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("100 101\n")
    warp_dir = tmp_path / "warp"; warp_dir.mkdir()
    cert_dir = tmp_path / "cert"; cert_dir.mkdir()

    proc = subprocess.run(
        [
            str(BINARY),
            "--input", str(sfm),
            "--output", str(tmp_path / "out.sfm"),
            "--imagePairsList", str(pairs),
            "--outputCertaintyFolder", str(cert_dir),
            "--outputWarpFolder", str(warp_dir),
            "--modelPath", str(MLPACKAGE),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"binary exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # Expected output: 4 EXR files per pair.
    expected = [
        cert_dir / "100_101_coarse_certainty.exr",
        cert_dir / "100_101_fine_certainty.exr",
        warp_dir / "100_101_coarse_flow.exr",
        warp_dir / "100_101_fine_flow.exr",
    ]
    for f in expected:
        assert f.exists(), f"missing output EXR: {f}\nstdout:\n{proc.stdout}"

    # Sanity: file sizes must be > 1 KB so a regression-to-stub is caught.
    # The Phase 14.7 pass-through never produced these files at all, so
    # presence already proves it's not the stub; the size check is a
    # second-order guard against any "write empty placeholder" regression.
    for f in expected:
        assert f.stat().st_size > 1_000, (
            f"output EXR is suspiciously small: {f} ({f.stat().st_size} B)"
        )

    # Magic-byte check for OpenEXR.
    for f in expected:
        with open(f, "rb") as fh:
            magic = fh.read(4)
        assert magic == b"\x76\x2f\x31\x01", f"{f}: bad EXR magic {magic.hex()}"

    # Log assertion: the binary logs to stderr (AliceVision_LOG_INFO
    # goes to stderr through Boost.Log). The "pair(s) matched" line
    # must be present — a Phase 14.7 stub regression would say
    # "pass-through" instead.
    combined = proc.stdout + proc.stderr
    assert "pair(s) matched" in combined, (
        f"binary output missing 'pair(s) matched' line:\n{combined}"
    )
    assert "pass-through forwarding" not in combined, (
        "binary output suggests regression to Phase 14.7 pass-through"
    )
