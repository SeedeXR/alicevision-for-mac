"""
Plugin-manifest integration tests for the ai-segmentation plugin (S53).

These tests load `plugin.json` directly via `json.load` (stdlib only — no
new dependencies per the S53 hard constraints), then cross-check it
against the on-disk plugin tree:

  1. The JSON parses and carries the expected top-level fields.
  2. Every node listed in `nodes[].name` has a matching
     `nodes/aliceVision/<Name>.py` file.
  3. Paths referenced relative to the manifest (wrapper_script,
     python.package_path) all resolve to existing files / directories.
  4. The `model_variants` table is non-empty and every entry carries the
     three required keys.

If a third-party plugin author copies this layout to ship a new plugin,
they get the same tests "for free" by running pytest against their tree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# The manifest sits two levels above this test file:
# tests/test_plugin_manifest.py -> ai-segmentation/ -> plugin.json
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PLUGIN_ROOT / "plugin.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    """Load and parse plugin.json once per module."""
    assert MANIFEST_PATH.is_file(), f"missing manifest at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_parses_with_required_fields(manifest):
    """`plugin.json` must declare the core descriptor fields."""
    # `wrapper_script` was retired 2026-05-23 when the Swift loader that
    # consumed it was decommissioned (see CHANGELOG: meshroom-native).
    # The Python Meshroom discovers descriptors via MESHROOM_NODES_PATH
    # directly, so the manifest no longer needs to spell out a wrapper.
    required = {"name", "version", "description", "nodes"}
    missing = required - manifest.keys()
    assert not missing, f"plugin.json missing required keys: {missing}"
    assert manifest["name"] == "ai-segmentation"
    assert isinstance(manifest["nodes"], list) and manifest["nodes"], (
        "nodes[] must be a non-empty array"
    )


def test_manifest_node_names_match_node_files(manifest):
    """Every `nodes[].name` must have a matching .py descriptor on disk."""
    nodes_dir = PLUGIN_ROOT / "nodes" / "aliceVision"
    assert nodes_dir.is_dir(), f"missing nodes dir at {nodes_dir}"
    for node in manifest["nodes"]:
        name = node["name"]
        node_file = nodes_dir / f"{name}.py"
        # `node_file` may resolve through a back-compat symlink at the
        # legacy meshroom-mac path; either way the file must exist.
        assert node_file.exists(), (
            f"manifest declares node '{name}' but {node_file} is missing"
        )
        # And `Path.resolve()` must land inside the plugin tree —
        # otherwise the symlink is dangling.
        real = node_file.resolve()
        assert PLUGIN_ROOT in real.parents, (
            f"node file {node_file} resolves outside plugin tree to {real}"
        )


def test_manifest_referenced_paths_exist(manifest):
    """python.package_path and models_dir must all resolve."""
    # The python package_path field points at the dir holding the
    # `segmentation/` Python package this plugin ships.
    py_cfg = manifest.get("python", {})
    pkg = (PLUGIN_ROOT / py_cfg.get("package_path", "python")).resolve()
    assert pkg.is_dir(), f"python.package_path does not exist: {pkg}"
    assert (pkg / "segmentation" / "__init__.py").is_file(), (
        "segmentation package init missing"
    )
    # `models_dir` is optional but if present must resolve to a dir.
    if "models_dir" in manifest:
        models = (PLUGIN_ROOT / manifest["models_dir"]).resolve()
        assert models.is_dir(), f"models_dir does not exist: {models}"


def test_model_variants_table(manifest):
    """`model_variants` must list every BiRefNet CoreML variant we ship."""
    # The rembg/ONNX backend was removed 2026-05-23; only the two
    # CoreML mlpackage variants ship now.
    variants = manifest.get("model_variants", [])
    assert variants, "model_variants must be a non-empty array"
    ids = {v["id"] for v in variants}
    assert ids == {"birefnet-lite", "birefnet-general"}, (
        f"unexpected model_variants ids: {ids}"
    )
    for v in variants:
        for key in ("id", "size_mb", "backbone", "package"):
            assert key in v, f"variant {v.get('id')} missing key {key!r}"
        assert isinstance(v["size_mb"], (int, float)) and v["size_mb"] > 0
        assert v["package"].endswith(".mlpackage"), (
            f"variant {v['id']} package must be a .mlpackage, got {v['package']}"
        )
