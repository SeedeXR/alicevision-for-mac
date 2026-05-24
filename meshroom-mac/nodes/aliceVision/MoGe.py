__version__ = "2.0"

#
# MoGe — Monocular Geometry / mono-depth estimation node.
#
# Background: The 2026.1.0 `cameraTrackingDepth.mg` template introduces
# a depth-prior arm to the camera-tracking pipeline by running a
# monocular geometry model (MoGe) over each input frame to produce a
# per-view depth/normals prior. The resulting `output` directory is then
# consumed by `DepthMapTracksInjecting` as a track depth-source for the
# `SfMBootStrapping` / `SfMExpanding` arm of the pipeline.
#
# Binary: `aliceVision_moGe` (NOT yet built on the Mac port — this
# descriptor is the loadable shim that will route to the binary once
# Phase 14.x port is finished; until then coverage reports it as
# `missingBinaries` rather than `missingDescriptors`).
#
# Used by templates: cameraTrackingDepth (only).
#
# Output attributes referenced by downstream nodes:
#   - output : folder containing per-view depth maps (and, optionally,
#     normal maps when `outputNormals=True`).
#

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class MoGe(desc.AVCommandLineNode):
    """
Run the MoGe (Monocular Geometry) prior over a set of input images to
produce a per-view depth (and optionally normals) folder. The output is
typically consumed by `DepthMapTracksInjecting` as a depth source for
the depth-aware SfM arm.
"""

    commandLine = "aliceVision_moGe {allParams}"
    size = desc.DynamicNodeSize("inputImages")

    category = "Dense Reconstruction"
    cpu = desc.Level.INTENSIVE
    gpu = desc.Level.INTENSIVE
    ram = desc.Level.INTENSIVE

    inputs = [
        desc.File(
            name="inputImages",
            label="Input Images",
            description="SfMData JSON (or folder of images) over which to run the "
                        "monocular geometry / depth model.",
            value="",
        ),
        desc.ChoiceParam(
            name="foVEstimationMode",
            label="FoV Estimation Mode",
            description="How to infer the field-of-view for the mono-depth model.\n"
                        " * Metadata : use the EXIF metadata of each frame.\n"
                        " * Estimate : estimate FoV from the image content.\n"
                        " * Fixed    : assume a fixed default FoV.",
            value="Metadata",
            values=["Metadata", "Estimate", "Fixed"],
            exclusive=True,
        ),
        desc.FloatParam(
            name="fixedFoV",
            label="Fixed FoV",
            description="FoV (in degrees) used when foVEstimationMode == 'Fixed'.",
            value=60.0,
            range=(10.0, 170.0, 0.1),
            enabled=lambda node: node.foVEstimationMode.value == "Fixed",
        ),
        desc.BoolParam(
            name="outputNormals",
            label="Output Normals",
            description="If true, also emit per-view normal maps alongside the depth maps.",
            value=False,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Depth Folder",
            description="Folder containing per-view depth maps (and optional normals).",
            value="{nodeCacheFolder}",
        ),
    ]
