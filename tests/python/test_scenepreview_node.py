"""
Tests for the Mac-native ScenePreview Meshroom node descriptor
(`meshroom-mac/nodes/aliceVision/ScenePreview.py`).

Phase 3 added the descriptor to silence the recurring
`WARNING:root:Expression '{ScenePreview_1.output}': 'output'` warning
across 10 templates that reference it. Without these tests, a future
edit that breaks descriptor registration would only surface as the
warning resurfacing — and only on a `.mg` load. These pin:

  1. The descriptor module imports cleanly + declares __version__ = "2.0".
  2. The Meshroom plugin manager registers it under the name `ScenePreview`.
  3. The expected input/output schema (cameras / model / undistortedImages
     / masks / verboseLevel + a pointCloudParams group; output File).
  4. Loading a real ScenePreview-using template (cameraTracking.mg)
     produces a typed Node, NOT a CompatibilityNode, AND no
     `Expression '{ScenePreview_1.output}': 'output'` warning fires.
  5. A direct call to processChunk on a synthetic input folder
     produces the documented output (scene_preview.json + symlinks).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# --------------------------------------------------------------------------- #
# (1) descriptor module + __version__
# --------------------------------------------------------------------------- #

def test_scenepreview_module_imports():
    import importlib.util
    spec = importlib.util.find_spec("aliceVision.ScenePreview")
    if spec is None:
        # Try via the MESHROOM_NODES_PATH-managed sys.path; conftest
        # should have added it.
        repo_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(repo_root / "meshroom-mac" / "nodes"))
        spec = importlib.util.find_spec("aliceVision.ScenePreview")
    assert spec is not None, "ScenePreview module not findable on sys.path"
    mod = importlib.import_module("aliceVision.ScenePreview")
    assert mod.__version__ == "2.0", (
        f"ScenePreview.__version__ must be '2.0' (templates declare "
        f"'\"ScenePreview\": \"2.0\"' in nodesVersions); got "
        f"{mod.__version__!r}"
    )
    assert hasattr(mod, "ScenePreview"), "missing ScenePreview class"


# --------------------------------------------------------------------------- #
# (2) plugin manager registration
# --------------------------------------------------------------------------- #

def test_scenepreview_registered_in_plugin_manager():
    import meshroom.core
    # conftest invokes initNodes(); confirm the descriptor landed.
    assert meshroom.core.pluginManager.isRegistered("ScenePreview"), (
        "ScenePreview not in pluginManager._nodePlugins. "
        "Most likely cause: MESHROOM_NODES_PATH doesn't include "
        "meshroom-mac/nodes, OR the descriptor failed validateNodeDesc."
    )
    plugin = meshroom.core.pluginManager.getRegisteredNodePlugin("ScenePreview")
    assert plugin is not None
    assert plugin.nodeDescriptor.__name__ == "ScenePreview"


# --------------------------------------------------------------------------- #
# (3) input/output schema matches what templates expect
# --------------------------------------------------------------------------- #

def test_scenepreview_input_output_schema():
    import importlib
    if "aliceVision.ScenePreview" not in sys.modules:
        repo_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(repo_root / "meshroom-mac" / "nodes"))
    mod = importlib.import_module("aliceVision.ScenePreview")
    SP = mod.ScenePreview

    input_names = {a.name for a in SP.inputs}
    # Every template references these four file inputs.
    required = {"cameras", "model", "undistortedImages", "masks"}
    missing = required - input_names
    assert not missing, f"ScenePreview missing required inputs: {missing}"

    # `pointCloudParams` group is referenced by nodalCameraTracking templates;
    # without it, those templates fail with DescriptionConflict at load time.
    assert "pointCloudParams" in input_names, (
        "Missing pointCloudParams group — nodalCameraTracking* templates "
        "fail DescriptionConflict without it"
    )

    output_names = {a.name for a in SP.outputs}
    assert "output" in output_names, "missing required `output` attribute"


# --------------------------------------------------------------------------- #
# (4) template load: typed Node, zero warnings
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("template", [
    "cameraTracking",
    "cameraTrackingDepth",
    "nodalCameraTracking",
    "photogrammetryAndCameraTracking",
])
def test_template_loads_with_typed_scenepreview(template, repo_root, caplog):
    from meshroom.core.graph import Graph
    caplog.set_level(logging.WARNING)
    tmpl = repo_root / "meshroom-mac" / "nodes" / f"{template}.mg"
    assert tmpl.is_file(), f"missing template fixture {tmpl}"

    g = Graph("")
    g.initFromTemplate(str(tmpl))
    sp = g.nodes.get("ScenePreview_1")
    assert sp is not None, f"ScenePreview_1 not in {template}"
    assert not sp.isCompatibilityNode, (
        f"ScenePreview_1 deserialized as CompatibilityNode in {template} — "
        f"descriptor registration broke. Issue: {getattr(sp, 'issue', None)}"
    )
    assert sp.hasAttribute("output"), (
        f"ScenePreview_1.output not present as a typed attribute in {template}"
    )
    # The specific warning the user kept seeing: must not appear.
    expression_warnings = [
        r.message for r in caplog.records
        if "ScenePreview_1.output" in r.message and "Expression" in r.message
    ]
    assert not expression_warnings, (
        f"Recurring user-reported warning leaked back in for {template}: "
        f"{expression_warnings}"
    )


# --------------------------------------------------------------------------- #
# (5) processChunk smoke — produces scene_preview.json + symlinks
# --------------------------------------------------------------------------- #

def _make_synthetic_inputs(root: Path) -> dict[str, Path]:
    """Build small input folders the descriptor can aggregate."""
    cameras = root / "cameras"
    model = root / "model"
    images = root / "images"
    masks = root / "masks"
    for d in (cameras, model, images, masks):
        d.mkdir(parents=True)
    (cameras / "cameras.sfm").write_text('{"version":[2,0,0]}', encoding="utf-8")
    (model / "mesh.obj").write_text("v 0 0 0\nv 1 0 0\n", encoding="utf-8")
    (images / "IMG_001.JPG").write_bytes(b"\xff\xd8\xff\xe0")
    (images / "IMG_002.JPG").write_bytes(b"\xff\xd8\xff\xe0")
    (masks / "IMG_001_mask.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (masks / "IMG_002_mask.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return {"cameras": cameras, "model": model, "images": images, "masks": masks}


def test_scenepreview_processchunk_writes_manifest_and_symlinks(tmp_path):
    """processChunk should produce scene_preview.json + 4 symlinks."""
    inputs = _make_synthetic_inputs(tmp_path / "inputs")
    out_dir = tmp_path / "output"

    # Build a minimal chunk mock. The descriptor only reads:
    #   chunk.node.<attr>.value, chunk.node.output.value,
    #   chunk.logger, chunk.logManager.{start,end}, chunk.node.verboseLevel.value
    chunk = MagicMock()
    chunk.node.cameras.value = str(inputs["cameras"])
    chunk.node.model.value = str(inputs["model"])
    chunk.node.undistortedImages.value = str(inputs["images"])
    chunk.node.masks.value = str(inputs["masks"])
    chunk.node.output.value = str(out_dir)
    chunk.node.verboseLevel.value = "info"
    chunk.logger = logging.getLogger("scenepreview-test")

    import importlib
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root / "meshroom-mac" / "nodes") not in sys.path:
        sys.path.insert(0, str(repo_root / "meshroom-mac" / "nodes"))
    mod = importlib.import_module("aliceVision.ScenePreview")
    desc = mod.ScenePreview()
    desc.processChunk(chunk)

    manifest = out_dir / "scene_preview.json"
    assert manifest.is_file(), "processChunk should write scene_preview.json"
    data = json.loads(manifest.read_text())
    assert data["schema"] == "scene_preview/1.0"
    assert data["node"] == "ScenePreview"
    # Each input key should appear with linked=True + a fileCount.
    for key in ("cameras", "model", "undistortedImages", "masks"):
        assert key in data["inputs"], f"missing {key} in manifest"
        assert data["inputs"][key]["linked"] is True, (
            f"{key} should be symlinked; got linked={data['inputs'][key]['linked']}"
        )

    # The 4 symlinks (cameras, model, images, masks).
    for link_name in ("cameras", "model", "images", "masks"):
        link = out_dir / link_name
        assert link.is_symlink() or link.exists(), f"missing symlink {link}"

    # File counts reflect what we created.
    assert data["inputs"]["undistortedImages"]["fileCount"] == 2
    assert data["inputs"]["masks"]["fileCount"] == 2


def test_scenepreview_processchunk_handles_missing_inputs_gracefully(tmp_path):
    """If an input path is empty, processChunk should NOT crash; it
    should write the manifest with linked=False for that input."""
    inputs = _make_synthetic_inputs(tmp_path / "inputs")
    out_dir = tmp_path / "output"

    chunk = MagicMock()
    chunk.node.cameras.value = str(inputs["cameras"])
    chunk.node.model.value = ""   # ← intentionally empty
    chunk.node.undistortedImages.value = str(inputs["images"])
    chunk.node.masks.value = ""   # ← intentionally empty
    chunk.node.output.value = str(out_dir)
    chunk.node.verboseLevel.value = "info"
    chunk.logger = logging.getLogger("scenepreview-test")

    import importlib
    if "aliceVision.ScenePreview" not in sys.modules:
        repo_root = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(repo_root / "meshroom-mac" / "nodes"))
    mod = importlib.import_module("aliceVision.ScenePreview")
    mod.ScenePreview().processChunk(chunk)

    data = json.loads((out_dir / "scene_preview.json").read_text())
    assert data["inputs"]["model"]["linked"] is False
    assert data["inputs"]["masks"]["linked"] is False
    # Present inputs still link.
    assert data["inputs"]["cameras"]["linked"] is True
    assert data["inputs"]["undistortedImages"]["linked"] is True
