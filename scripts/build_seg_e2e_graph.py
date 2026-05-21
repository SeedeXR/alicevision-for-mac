#!/usr/bin/env python3
"""
scripts/build_seg_e2e_graph.py

Build a Monstree mini3 photogrammetry graph with SegmentationBiRefNet
wired between CameraInit and FeatureExtraction. Uses the existing
baseline `meshroom-mac-out/project.mg` (which is known to produce a
textured mesh) as a starting point and surgically inserts the
segmentation node + mask wiring. Output: `<out>/project.mg`.

Used by the S53 E2E test. Requires the env that `scripts/run_meshroom.sh`
sets up (MESHROOM_NODES_PATH, PYTHONPATH, ALICEVISION_BIN_PATH, etc).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        default="meshroom-mac-out/project.mg",
        help="Working .mg to start from (must already produce a mesh).",
    )
    ap.add_argument(
        "--output",
        default="meshroom-mac-out/seg-e2e/project.mg",
        help="Path to the .mg file to write (relative to cwd or absolute).",
    )
    ap.add_argument(
        "--model-variant",
        default="birefnet-lite",
        choices=["birefnet-lite", "birefnet-general", "birefnet-dis"],
    )
    ap.add_argument(
        "--mask-format",
        default="png",
        choices=["png", "exr"],
    )
    args = ap.parse_args()

    # Late import so MESHROOM_NODES_PATH from the wrapper script is picked up.
    import meshroom
    import meshroom.core
    import meshroom.core.graph
    meshroom.setupEnvironment()
    meshroom.core.initNodes()
    meshroom.core.initPipelines()
    from meshroom.core import pluginManager

    av = pluginManager.getPlugin("aliceVision")
    if not av or "SegmentationBiRefNet" not in av.nodes:
        sys.stderr.write(
            "SegmentationBiRefNet not registered. Ensure MESHROOM_NODES_PATH "
            "points at meshroom-mac/nodes/ (back-compat symlink to the plugin).\n"
        )
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    baseline = (repo_root / args.baseline) if not Path(args.baseline).is_absolute() else Path(args.baseline)
    output_mg = (repo_root / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    output_mg.parent.mkdir(parents=True, exist_ok=True)

    if not baseline.exists():
        sys.stderr.write(f"Baseline graph not found: {baseline}\n")
        return 2

    g = meshroom.core.graph.loadGraph(str(baseline))

    # Add the SegmentationBiRefNet node.
    seg = g.addNewNode("SegmentationBiRefNet")
    seg.modelVariant.value = args.model_variant
    seg.maskFormat.value = args.mask_format
    seg.keepFilename.value = True
    seg.outputResolution.value = "1024"

    cam = g.findNode("CameraInit_1")
    fe = g.findNode("FeatureExtraction_1")
    pds = g.findNode("PrepareDenseScene_1")
    tex = g.findNode("Texturing_1")

    # Seg.input <- CameraInit.output (SfMData)
    g.addEdge(cam.output, seg.input)

    # Wire masks downstream.
    # FE.masksFolder accepts a single folder; set maskExtension to png/exr.
    g.addEdge(seg.output, fe.masksFolder)
    fe.maskExtension.value = args.mask_format

    # PrepareDenseScene.masksFolders is a ListAttribute — append our folder.
    pds.masksFolders.append("")
    g.addEdge(seg.output, pds.masksFolders.at(len(pds.masksFolders) - 1))
    pds.maskExtension.value = args.mask_format

    # Texturing.masksFolders is also a ListAttribute (use it to skip
    # texturing background patches).
    if hasattr(tex, "masksFolders"):
        tex.masksFolders.append("")
        g.addEdge(seg.output, tex.masksFolders.at(len(tex.masksFolders) - 1))

    g.update()
    g.save(str(output_mg), setupProjectFile=True)
    print(f"Wrote graph: {output_mg}")
    for n in g.nodes:
        try:
            sz = n.size
        except Exception:
            sz = "?"
        print(f"  node {n.name}: size={sz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
