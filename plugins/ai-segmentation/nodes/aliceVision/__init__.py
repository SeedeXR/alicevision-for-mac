# ai-segmentation plugin: aliceVision node namespace.
#
# This package mirrors the upstream `meshroom-mac/nodes/aliceVision/` layout so
# `meshroom.bin.node_run` can `import aliceVision.<NodeType>` without caring
# whether the node ships with Meshroom or with a third-party plugin. The
# Swift `PluginRegistry` adds this directory's parent (the plugin's `nodes/`
# dir) to `MESHROOM_NODES_PATH`, and the wrapper script puts that on
# `PYTHONPATH`, which together make `aliceVision.SegmentationBiRefNet`
# resolve to the file living at
# `plugins/ai-segmentation/nodes/aliceVision/SegmentationBiRefNet.py`.
