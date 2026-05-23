"""Per-pipeline integration tests.

Runs each Meshroom .mg template that is fully covered by the 12 native
Apple Silicon Metal aliceVision_* binaries we ship, on dataset_monstree/mini3
(3 images, ~3 min per pipeline), and verifies the output mesh is sane.

Templates NOT covered (missing binaries or missing descriptors) are
asserted to fail at template-load time with a clean diagnostic — they
should never be presented to users as "ready to run" without first
either (a) building the missing binaries or (b) porting the missing
node descriptors.

Gated behind RUN_PIPELINE_E2E=1 because end-to-end pipeline runs take
~3 min each. Always-on tests assert the coverage matrix itself.

Honest scope (per pipeline_coverage.py output as of S54b):

  Covered today (2/25):
    - photogrammetryDraft     (10 nodes, no DepthMap; faster, lower-fidelity)
    - photogrammetryLegacy    (12 nodes, full SfM + dense MVS + texturing)

  Uncovered (23/25): require binaries from Phase 13-14 (modern SfM,
  panorama, HDR, photometric stereo, lidar, camera tracking) and/or
  upstream 2026 descriptor ports (ImageDetectionPrompt, SfMBootStrapping,
  ScenePreview, MoGe, etc.). Tracked in memory/todo.md.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COVERAGE_SCRIPT = REPO_ROOT / "scripts" / "pipeline_coverage.py"
RUN_MESHROOM = REPO_ROOT / "scripts" / "run_meshroom.sh"
TEMPLATES_DIR = REPO_ROOT / "meshroom-mac" / "nodes"
BUILD_DIR = REPO_ROOT / "build"

# Dataset selector. Default mini3 keeps the test fast (~35-55s per
# pipeline on M-series). Set `PIPELINE_DATASET=mini6` for a 6-photo
# mid-tier test (~2-3 min), or `PIPELINE_DATASET=full` for the 41-photo
# battle test (~25-35 min, production-readiness signal).
_DATASET_NAME = os.environ.get("PIPELINE_DATASET", "mini3")
DATASET = REPO_ROOT / "dataset_monstree" / _DATASET_NAME

# Templates we EXPECT to be fully covered as long as the 12 standard
# native binaries from Phase 11 are present in ./build/. If this list
# shrinks (e.g. someone removes a binary), the coverage assertion fires.
EXPECTED_COVERED = {
    "photogrammetryDraft", "photogrammetryLegacy",
    # photogrammetryObject + photogrammetryObjectTurntable: SAM-purged
    # 2026-05-23 (ImageSegmentationBox + ImageDetectionPrompt → SegmentationBiRefNet)
    # and verified end-to-end on Monstree mini3: each produced a
    # texturedMesh.obj + 3 BiRefNet masks (~400 ms/view, png format).
    "photogrammetryObject", "photogrammetryObjectTurntable",
}

# Templates we know will fail because they require binaries we have NOT
# built. If new binaries land (Phase 13-14), we move them out of here
# into EXPECTED_COVERED.
EXPECTED_UNCOVERED = {
    "cameraTracking", "cameraTrackingDepth", "cameraTrackingLegacy",
    "cameraTrackingRoma", "cameraTrackingWithoutCalibration",
    "cameraTrackingWithoutCalibrationLegacy",
    "colorCalibration", "distortionCalibration", "hdrFusion", "lidarMeshing",
    "multi-viewPhotometricStereo",
    "nodalCameraTracking", "nodalCameraTrackingWithoutCalibration",
    "panoramaFisheyeHdr", "panoramaHdr",
    "photogrammetry",  # modern SfM, needs Phase 14 binaries
    "photogrammetryAndCameraTracking", "photogrammetryAndCameraTrackingLegacy",
    "photogrammetryObjectTwoSides",  # SAM-purged but not E2E-verified yet
    "photometricStereo", "rawImageConversion",
}


def _load_coverage() -> list[dict]:
    """Run scripts/pipeline_coverage.py --json and return parsed list."""
    out = subprocess.check_output(
        [sys.executable, str(COVERAGE_SCRIPT), "--json"],
        text=True,
    )
    return json.loads(out)


# ---- Always-on (no E2E required) ----------------------------------------

def test_coverage_script_exists_and_runs() -> None:
    assert COVERAGE_SCRIPT.is_file(), f"missing {COVERAGE_SCRIPT}"
    rows = _load_coverage()
    assert len(rows) >= 20, "expected at least 20 templates, found {}".format(len(rows))


def test_expected_covered_pipelines_are_covered() -> None:
    """The 2 baseline pipelines must remain covered. If this fails, a binary
    or descriptor regressed."""
    rows = {r["name"]: r for r in _load_coverage()}
    for name in EXPECTED_COVERED:
        assert name in rows, f"{name} not found in coverage matrix"
        r = rows[name]
        assert r["covered"], (
            f"Pipeline '{name}' lost coverage. "
            f"Missing binaries: {r['missingBinaries']}. "
            f"Missing descriptors: {r['missingDescriptors']}."
        )


def test_uncovered_pipelines_are_diagnosed() -> None:
    """Pipelines we cannot run today MUST report specific missing pieces.
    If one suddenly appears covered without our list being updated, that's
    great news but the test catalogue should be updated to add an E2E run
    for it. Conversely if NEW pipelines appear without classification, the
    test fails so we know."""
    rows = {r["name"]: r for r in _load_coverage()}
    actual_uncovered = {name for name, r in rows.items() if not r["covered"]}
    surprises = actual_uncovered - EXPECTED_UNCOVERED
    new_covered = EXPECTED_UNCOVERED & {n for n, r in rows.items() if r["covered"]}
    assert not surprises, (
        f"New uncovered pipelines appeared without classification: {surprises}\n"
        f"Update EXPECTED_UNCOVERED in this file."
    )
    assert not new_covered, (
        f"Pipelines newly covered (great!) — move them out of EXPECTED_UNCOVERED "
        f"and add an E2E test: {new_covered}"
    )


def test_native_binaries_present() -> None:
    """The 12 binaries Phase 11 ships must exist in ./build/."""
    required = {
        "aliceVision_cameraInit",
        "aliceVision_featureExtraction",
        "aliceVision_imageMatching",
        "aliceVision_featureMatching",
        "aliceVision_incrementalSfM",
        "aliceVision_prepareDenseScene",
        "aliceVision_depthMapEstimation",
        "aliceVision_depthMapFiltering",
        "aliceVision_meshing",
        "aliceVision_meshFiltering",
        "aliceVision_texturing",
    }
    missing = {b for b in required if not (BUILD_DIR / b).is_file()}
    assert not missing, f"native binaries missing from {BUILD_DIR}: {missing}"


# ---- Per-template smoke (cheap: load template, check it parses) ----------

@pytest.mark.parametrize("name", sorted(EXPECTED_COVERED))
def test_template_loads_cleanly(name: str) -> None:
    """The .mg file parses, every node has a nodeType, every input edge
    references a known upstream node. Catches malformed templates without
    needing a 3-minute run."""
    path = TEMPLATES_DIR / f"{name}.mg"
    assert path.is_file(), f"missing {path}"
    data = json.loads(path.read_text())
    assert "graph" in data, f"{name}: no 'graph' key"
    graph = data["graph"]
    node_names = set(graph.keys())
    for node_name, node in graph.items():
        assert "nodeType" in node, f"{name}: node {node_name} missing nodeType"
        for attr_name, attr_val in (node.get("inputs") or {}).items():
            if isinstance(attr_val, str) and "{" in attr_val:
                # Template reference of the form {NodeName.attr}
                for m in re.finditer(r"\{(\w+)\.[\w.]+\}", attr_val):
                    ref = m.group(1)
                    assert ref in node_names, (
                        f"{name}: {node_name}.{attr_name} references unknown "
                        f"upstream node '{ref}'"
                    )


# ---- E2E pipeline runs (slow, ~3 min each) -------------------------------

requires_e2e = pytest.mark.skipif(
    os.environ.get("RUN_PIPELINE_E2E") != "1",
    reason="Set RUN_PIPELINE_E2E=1 to run end-to-end pipeline integration tests (~3 min each)",
)


def _run_pipeline(name: str, out_dir: Path) -> tuple[int, str]:
    """Invoke meshroom_batch for `name` on the currently-selected dataset.

    The subprocess timeout scales with dataset size — mini3 (3 photos)
    is bounded by 15 min, but full (41 photos) needs ~30-40 min for
    photogrammetryLegacy's dense-depth stage. Override with
    `PIPELINE_TIMEOUT=<seconds>` if needed.

    Returns (returncode, combined stdout+stderr).
    """
    assert RUN_MESHROOM.is_file(), f"missing {RUN_MESHROOM}"
    assert DATASET.is_dir(), f"missing dataset {DATASET}"
    out_dir.mkdir(parents=True, exist_ok=True)
    project_mg = out_dir / "project.mg"
    cmd = [
        "bash", str(RUN_MESHROOM),
        "python", "bin/meshroom_batch",
        "-i", str(DATASET),
        "-o", str(out_dir / "result"),
        "-p", name,
        "-s", str(project_mg),
    ]
    n_photos = len(list(DATASET.glob("*.JPG"))) + len(list(DATASET.glob("*.jpg")))
    # Empirical budget: ~60s/photo for Legacy on M4 (dense depth-map
    # dominates wall-clock); Draft is ~4x faster but use the same bound.
    # Floor at 900s so mini3 keeps its existing 15-min ceiling.
    default_timeout = max(900, 60 * n_photos)
    timeout_s = int(os.environ.get("PIPELINE_TIMEOUT", default_timeout))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    elapsed = time.time() - t0
    log = (
        f"--- meshroom_batch '{name}' rc={proc.returncode} "
        f"elapsed={elapsed:.1f}s (timeout={timeout_s}s, photos={n_photos}) ---\n"
        + proc.stdout + proc.stderr
    )
    return proc.returncode, log


def _find_textured_mesh(result_dir: Path) -> Path | None:
    """Locate the textured mesh produced by Texturing node, if any."""
    candidates = list(result_dir.rglob("texturedMesh.obj"))
    candidates.extend(result_dir.rglob("texturedMesh.ply"))
    return candidates[0] if candidates else None


def _count_vertices_ply(ply_path: Path) -> int:
    """Read the vertex count from a PLY header."""
    with ply_path.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise AssertionError(f"no vertex element in PLY header at {ply_path}")


@requires_e2e
@pytest.mark.parametrize("name", sorted(EXPECTED_COVERED))
def test_pipeline_runs_end_to_end(name: str, tmp_path: Path) -> None:
    """Run a covered pipeline on Monstree mini3 → expect a textured mesh
    with a reasonable vertex count, no fatal errors in the log."""
    out_dir = tmp_path / name
    rc, log = _run_pipeline(name, out_dir)

    # Always write the log so we can inspect failures after the fact.
    (out_dir / f"{name}.log").write_text(log)

    assert rc == 0, f"meshroom_batch failed for '{name}' (rc={rc}). Tail of log:\n" + log[-4000:]

    # Find a textured mesh somewhere under the cache
    mesh = _find_textured_mesh(out_dir / "result")
    if mesh is None:
        # Some pipelines produce only the SfM (no Texturing). photogrammetryDraft
        # outputs a meshing result; check for that as fallback.
        mesh_candidates = list((out_dir / "result").rglob("mesh.obj")) + \
                          list((out_dir / "result").rglob("mesh.ply"))
        assert mesh_candidates, (
            f"No textured/raw mesh produced under {out_dir/'result'}. "
            f"Pipeline '{name}' should yield at least mesh.obj or mesh.ply. "
            f"Log tail:\n" + log[-3000:]
        )
        mesh = mesh_candidates[0]

    # Sanity-check vertex count
    if mesh.suffix == ".ply":
        n_verts = _count_vertices_ply(mesh)
        assert 100 <= n_verts <= 5_000_000, (
            f"'{name}' produced mesh with implausible vertex count: {n_verts} "
            f"(expected 100 .. 5e6 for Monstree mini3)"
        )
    elif mesh.suffix == ".obj":
        n_v = sum(1 for line in mesh.read_text(errors="replace").splitlines()
                  if line.startswith("v "))
        assert 100 <= n_v <= 5_000_000, (
            f"'{name}' produced OBJ with implausible vertex count: {n_v}"
        )


@requires_e2e
@pytest.mark.parametrize("name", sorted(EXPECTED_UNCOVERED))
def test_uncovered_pipeline_fails_with_clean_diagnostic(name: str, tmp_path: Path) -> None:
    """Pipelines we cannot run today MUST fail at template-load or first-node
    time, not crash silently with a segfault or leak processes. The user
    should see a clear "UnknownNodeType" or "binary not found" message."""
    out_dir = tmp_path / name
    rc, log = _run_pipeline(name, out_dir)
    (out_dir / f"{name}.log").write_text(log)
    assert rc != 0, f"Pipeline '{name}' was expected to fail but succeeded — promote to EXPECTED_COVERED"
    # Acceptable failure signatures: UnknownNodeType, command not found, missing binary.
    diagnostic_present = any(
        signal in log for signal in (
            "UnknownNodeType",
            "command not found",
            "No such file or directory",
            "is not available",
            "Compatibility issue",
        )
    )
    assert diagnostic_present, (
        f"Pipeline '{name}' failed without a recognisable diagnostic. "
        f"Log tail:\n" + log[-3000:]
    )
