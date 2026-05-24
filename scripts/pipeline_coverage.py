#!/usr/bin/env python3
"""Compute the binary-coverage matrix for every Meshroom pipeline template.

A template is "covered" when every aliceVision_* binary it shells out to
is present in ./build/, AND every node-type referenced in the template
either has a descriptor on disk or is in the small set of pure-Python
helpers that don't need a binary.

Output (TSV on stdout):
    name<TAB>covered<TAB>nodeTypes<TAB>missingBinaries<TAB>missingDescriptors

Used by:
    tests/python/test_pipeline_coverage.py — assertions on which
        pipelines must be covered, regression guard for regression.
    scripts/run_all_covered_pipelines.sh — wall-clock smoke runner.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
NODES_DIR = REPO_ROOT / "meshroom-mac" / "nodes" / "aliceVision"
TEMPLATES_DIR = REPO_ROOT / "meshroom-mac" / "nodes"
BUILD_DIR = REPO_ROOT / "build"

# Nodes that don't shell out to an aliceVision_* binary. They're either
# pure-Python (`desc.Node` subclasses) or Meshroom-framework built-ins.
PURE_PYTHON_NODES = frozenset({
    "ExtractMetadata",
    "SfMPoseFlattening",
    "SfMFilter",
    "SfMRigApplying",
    "SfMChecking",
    "SketchfabUpload",
    "SegmentationBiRefNet",  # Our S52 segmentation plugin
    "CopyFiles",  # Meshroom framework node
    "ImportE57",  # LiDAR ingestion, pure-Python
    "Backdrop",  # Visual group, no compute
    # Phase 3 Mac-native aggregator (no commandLine = aliceVision_*):
    "ScenePreview",
    # Roma legacy nodes route to pyalicevision and are virtual:
    "RomaMatcher",
    "RomaSampler",
    "RomaReducer",
})


def discover_native_binaries() -> set[str]:
    """Return the basenames of all aliceVision_* executables in ./build/."""
    if not BUILD_DIR.is_dir():
        return set()
    return {p.name for p in BUILD_DIR.glob("aliceVision_*") if p.is_file()}


def node_to_binary_map() -> dict[str, str]:
    """Map nodeType -> 'aliceVision_<sub>' by parsing the commandLine= line in each .py descriptor."""
    mapping: dict[str, str] = {}
    for py in NODES_DIR.glob("*.py"):
        if py.name == "__init__.py":
            continue
        text = py.read_text(errors="ignore")
        m = re.search(r"commandLine\s*=\s*[\"']aliceVision_(\w+)", text)
        if m:
            mapping[py.stem] = f"aliceVision_{m.group(1)}"
    return mapping


def template_node_types(template_path: Path) -> list[str]:
    """Sorted list of nodeTypes referenced in a .mg JSON template."""
    data = json.loads(template_path.read_text())
    return sorted({v.get("nodeType", "?") for v in data["graph"].values()})


def classify_template(
    template_path: Path,
    have_binaries: set[str],
    node_binary: dict[str, str],
) -> dict:
    """Return {name, covered, nodeTypes, missingBinaries, missingDescriptors}."""
    types = template_node_types(template_path)
    missing_bins: list[str] = []
    missing_descs: list[str] = []
    for t in types:
        if t in PURE_PYTHON_NODES:
            continue
        bin_name = node_binary.get(t)
        if bin_name is None:
            missing_descs.append(t)
        elif bin_name not in have_binaries:
            missing_bins.append(bin_name)
    return {
        "name": template_path.stem,
        "covered": not missing_bins and not missing_descs,
        "nodeTypes": types,
        "missingBinaries": sorted(set(missing_bins)),
        "missingDescriptors": sorted(set(missing_descs)),
    }


def coverage_matrix() -> list[dict]:
    """Classify every .mg template in TEMPLATES_DIR. Returns list of dicts."""
    have_bins = discover_native_binaries()
    node_bin = node_to_binary_map()
    out = []
    for tpl in sorted(TEMPLATES_DIR.glob("*.mg")):
        out.append(classify_template(tpl, have_bins, node_bin))
    return out


def main(argv: list[str]) -> int:
    matrix = coverage_matrix()
    if "--json" in argv:
        json.dump(matrix, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    # TSV: name, covered, nNodeTypes, missingBins, missingDescs
    sys.stdout.write("name\tcovered\tnNodeTypes\tmissingBinaries\tmissingDescriptors\n")
    for row in matrix:
        sys.stdout.write(
            "\t".join((
                row["name"],
                "yes" if row["covered"] else "no",
                str(len(row["nodeTypes"])),
                ",".join(row["missingBinaries"]) or "-",
                ",".join(row["missingDescriptors"]) or "-",
            )) + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
