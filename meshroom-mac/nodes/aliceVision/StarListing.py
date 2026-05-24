__version__ = "1.0"

#
# StarListing — build a "star" image-pair list around each keyframe.
#
# Background: For the Roma camera-tracking arm the dense matcher
# (RomaMatcher) needs an explicit list of image pairs to match. A
# "star" listing picks, for every keyframe, the N nearest non-keyframe
# views around it (within `radiusKeyFrames` frames before/after), so
# the matcher concentrates its budget around keyframes rather than
# enumerating every (i, j) pair.
#
# StarListing's outputs (`inputSfMData`, `imagePairsList`) are then
# consumed by RomaMatcher_1 in cameraTrackingRoma.mg.
#
# Binary: `aliceVision_starListing` (NOT yet built — this is a
# loadable shim).
#

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class StarListing(desc.AVCommandLineNode):
    """
Build a "star" image-pair list around each keyframe. For every keyframe
selected by `KeyframeSelection`, list its N nearest non-keyframe views
(within `radiusKeyFrames` frames before and after) as matching
candidates. Used by the Roma camera-tracking arm to constrain the
dense matcher's budget.
"""

    commandLine = "aliceVision_starListing {allParams}"
    size = desc.DynamicNodeSize("inputSfMData")

    category = "Sparse Reconstruction"
    cpu = desc.Level.NORMAL
    ram = desc.Level.NORMAL

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Full SfMData (all views, including non-keyframes).",
            value="",
        ),
        desc.File(
            name="keySfMData",
            label="Keyframe SfMData",
            description="SfMData containing the keyframes only "
                        "(typically `{KeyframeSelection.outputSfMDataKeyframes}`).",
            value="",
        ),
        desc.IntParam(
            name="radiusKeyFrames",
            label="Radius Around Keyframes",
            description="For each keyframe, include the N nearest non-keyframe views "
                        "before and after as match candidates.",
            value=5,
            range=(1, 50, 1),
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
        # Pass-through SfMData so downstream consumers can keep using
        # `{StarListing_1.inputSfMData}` to refer to the full SfMData.
        desc.File(
            name="inputSfMData",
            label="SfMData (passthrough)",
            description="SfMData passed through (matches the input).",
            value="{nodeCacheFolder}/sfm.sfm",
        ),
        desc.File(
            name="imagePairsList",
            label="Image Pairs",
            description="Output image-pairs list file.",
            value="{nodeCacheFolder}/pairs.txt",
        ),
    ]
