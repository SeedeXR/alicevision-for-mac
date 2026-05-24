__version__ = "1.0"

#
# MatchMasking — apply foreground/object masks to per-pair dense match
# data (warp + certainty volumes produced by RomaMatcher) so that
# match observations that fall outside the segmentation mask are
# pruned before downstream geometric filtering.
#
# Background: cameraTrackingRoma.mg routes
#   KeyframeSelection -> StarListing -> RomaMatcher -> MatchMasking
#                                              \--> SegmentationBiRefNet -+
# and MatchMasking forwards (inputSfMData, imagePairsList, warpFolder)
# plus a filtered certaintyFolder downstream.
#
# Binary: `aliceVision_matchMasking` (NOT yet built — this is a
# loadable shim).
#

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class MatchMasking(desc.AVCommandLineNode):
    """
Apply per-view foreground masks to the dense matching data produced by
RomaMatcher (or any Roma-flavored dense matcher). For every image pair,
match certainty volume entries that fall outside the masked region are
zeroed out; warp data, image-pair list and SfMData are forwarded
unchanged for downstream consumers.
"""

    commandLine = "aliceVision_matchMasking {allParams}"
    size = desc.DynamicNodeSize("inputSfMData")

    category = "Sparse Reconstruction"
    cpu = desc.Level.NORMAL
    ram = desc.Level.NORMAL

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="SfMData file forwarded from the dense-matcher node.",
            value="",
        ),
        desc.File(
            name="imagePairsList",
            label="Image Pairs",
            description="Image-pairs list forwarded from the dense-matcher node.",
            value="",
        ),
        desc.File(
            name="warpFolder",
            label="Warp Folder",
            description="Folder of per-pair warp data produced by the dense matcher.",
            value="",
        ),
        desc.File(
            name="certaintyFolder",
            label="Certainty Folder",
            description="Folder of per-pair certainty volumes produced by the dense matcher.",
            value="",
        ),
        desc.File(
            name="masksFolder",
            label="Masks Folder",
            description="Folder of per-view binary foreground masks "
                        "(typically wired from `SegmentationBiRefNet.output`).",
            value="",
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
        # Pass-through outputs so downstream graph wiring keeps working.
        desc.File(
            name="inputSfMData",
            label="SfMData (passthrough)",
            description="SfMData passed through from the matcher.",
            value="{nodeCacheFolder}/sfm.sfm",
        ),
        desc.File(
            name="imagePairsList",
            label="Image Pairs (passthrough)",
            description="Image-pairs list passed through from the matcher.",
            value="{nodeCacheFolder}/pairs.txt",
        ),
        desc.File(
            name="warpFolder",
            label="Warp Folder (passthrough)",
            description="Warp folder passed through from the matcher.",
            value="{nodeCacheFolder}/warp",
        ),
        desc.File(
            name="outputCertaintyFolder",
            label="Filtered Certainty Folder",
            description="Filtered certainty folder (entries outside masks are zeroed).",
            value="{nodeCacheFolder}/certainty",
        ),
    ]
