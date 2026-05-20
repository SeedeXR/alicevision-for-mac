import Foundation
import ProjectModel

/// Static mapping of Meshroom `nodeType` -> the `aliceVision_*` binary that
/// implements the node and a small declarative argv template.
///
/// We deliberately avoid trying to *evaluate* the upstream Python
/// `commandLineGroup` strings at runtime — that machinery is full of
/// node-instance lambdas (`enabled=lambda node: …`) that we cannot port to
/// Swift without re-implementing Python.  Instead, M5 ships a curated list of
/// the standard photogrammetry-pipeline flags for each of the 12 native
/// binaries we have on macOS.  Exotic / advanced flags become per-attribute
/// `--<name> <value>` pairs when present and emit nothing when absent.
///
/// Source of truth for binary names: the `commandLine = "aliceVision_*"` line
/// in `meshroom-mac/nodes/aliceVision/<NodeName>.py`.  Source of truth for
/// flag spelling: AliceVision CLI itself — every binary documents its flags
/// via `--help`, and Meshroom's `desc.File(name="foo", …)` map 1:1 to
/// `--foo` arguments.
public enum NodeBinary {

    /// Per-node chunked-execution spec.  Mirrors Meshroom upstream's
    /// `desc.Parallelization(blockSize=N)` + `commandLineRange` pattern.
    ///
    /// When set, the executor reads the node's input SfMData to determine the
    /// number of views, divides them into `ceil(views/blockSize)` chunks, and
    /// spawns one `Process` per chunk with `rangeArg` appended.  The two
    /// placeholders `{start}` and `{size}` are substituted with the chunk's
    /// `rangeStart` and `rangeSize` respectively before the string is
    /// whitespace-split into argv tokens.
    ///
    /// Chunks run sequentially by default (single GPU on Apple Silicon —
    /// parallel chunks would thrash the device).  A future setting may opt-in
    /// to parallel execution on larger UMA targets (Mac Studio with M2 Ultra).
    public struct Parallelization: Hashable, Sendable {
        public let blockSize: Int
        public let rangeArg: String

        public init(blockSize: Int, rangeArg: String) {
            self.blockSize = blockSize
            self.rangeArg = rangeArg
        }
    }

    /// Description of what to run for a given Meshroom node type.
    public struct Spec {
        /// File name of the native binary, e.g. `"aliceVision_cameraInit"`.
        public let executable: String
        /// Names of input attributes that should be emitted as
        /// `--<name> <value>` when present on the node.  Order matters only
        /// for reproducibility in tests; AliceVision binaries accept any order.
        public let inputFlags: [String]
        /// Names of output attributes the binary writes.  We pass these as
        /// `--<name> <resolvedPath>` so AliceVision knows where to drop
        /// artefacts inside our `MeshroomCache/` layout.
        public let outputFlags: [String]
        /// Extra constant flags appended verbatim, e.g.
        /// `["--allowSingleView", "1"]` for CameraInit.
        public let constantFlags: [String]
        /// Chunked-execution spec.  `nil` means single-invocation (default).
        public let parallelization: Parallelization?
        /// Canonical artefact filename written inside the node cache
        /// directory by a successful run.  Used by M6's warm-cache
        /// detection: when this file already exists under the
        /// node's resolved cache dir, the executor skips the Process
        /// spawn entirely.  Sourced by inspecting an existing Python
        /// Meshroom cache at
        /// `meshroom-mac-out/MeshroomCache/<Node>/<UID>/`.  Set to `nil`
        /// for nodes whose output is a directory of unpredictably-named
        /// files (e.g. FeatureExtraction emits per-view `.feat`/`.desc`
        /// files keyed by view ID — too fragile to gate on a single
        /// filename).  Such nodes always re-run; correctness wins over
        /// warm-cache aggressiveness.
        public let expectedOutputFile: String?

        /// M7: list of distinct output attribute names the node exposes.
        /// One pin is drawn per entry on the right edge of the node.  This is
        /// the *visible* output set — extracted from the upstream
        /// `meshroom-mac/nodes/aliceVision/<Node>.py` `outputs = [...]`
        /// block, filtering out `commandLineGroup=""` entries that are
        /// metadata-only descriptors (e.g. DepthMap's `depth`/`sim` per-view
        /// sequences exist only so the 2D viewer can recognise the per-view
        /// outputs, but they don't participate in graph connections).
        /// Defaults to `["output"]` because that's the canonical primary
        /// output of every aliceVision node.
        public let outputs: [String]

        /// M8: declared types for input attributes, keyed by attr name.
        /// Type tokens are deliberately small strings (not an enum) so a
        /// future Spec-loader could read them from JSON without a discriminator
        /// schema migration.  Recognised values:
        ///   "file" (desc.File), "int" (IntParam), "double" (FloatParam),
        ///   "string" (StringParam/ChoiceParam), "bool" (BoolParam),
        ///   "fileArray" (ListAttribute of desc.File), "intArray", "stringArray".
        /// Any attribute *not* present in this map is treated as "unknown" and
        /// the type checker defaults to ALLOW (M8 spec rule #5).
        public let inputTypes: [String: String]

        /// M8: declared types for output attributes, keyed by attr name.
        /// Outputs are almost always "file" (aliceVision binaries emit files
        /// and folders); we leave the map sparse for the same "unknown = ALLOW"
        /// reason as inputs.
        public let outputTypes: [String: String]

        public init(
            executable: String,
            inputFlags: [String],
            outputFlags: [String],
            constantFlags: [String],
            parallelization: Parallelization? = nil,
            expectedOutputFile: String? = nil,
            outputs: [String] = ["output"],
            inputTypes: [String: String] = [:],
            outputTypes: [String: String] = [:]
        ) {
            self.executable = executable
            self.inputFlags = inputFlags
            self.outputFlags = outputFlags
            self.constantFlags = constantFlags
            self.parallelization = parallelization
            self.expectedOutputFile = expectedOutputFile
            self.outputs = outputs
            self.inputTypes = inputTypes
            self.outputTypes = outputTypes
        }
    }

    /// The 12-binary mapping the macOS port ships with.  See
    /// `<repo>/build/aliceVision_*` for the on-disk set.
    public static let specs: [String: Spec] = [
        "CameraInit": Spec(
            executable: "aliceVision_cameraInit",
            inputFlags: [
                "viewpoints", "intrinsics", "sensorDatabase", "defaultFieldOfView",
                "groupCameraFallback", "allowedCameraModels", "rawColorInterpretation",
                "viewIdRegex", "verboseLevel",
            ],
            outputFlags: ["output"],
            // The upstream `commandLine` hardcodes --allowSingleView 1, so we
            // preserve that behaviour for byte-equivalent runs.
            constantFlags: ["--allowSingleView", "1"],
            expectedOutputFile: "cameraInit.sfm",
            // CameraInit upstream Python def exposes exactly one output:
            // `desc.File(name="output", value="{nodeCacheFolder}/cameraInit.sfm")`.
            outputs: ["output"],
            inputTypes: [
                "sensorDatabase": "file",
                "defaultFieldOfView": "double",
                "groupCameraFallback": "string",
                "allowedCameraModels": "string",
                "rawColorInterpretation": "string",
                "viewIdRegex": "string",
                "verboseLevel": "string",
            ],
            outputTypes: ["output": "file"]
        ),
        "FeatureExtraction": Spec(
            executable: "aliceVision_featureExtraction",
            inputFlags: [
                "input", "describerTypes", "describerPreset", "describerQuality",
                "contrastFiltering", "relativePeakThreshold", "gridFiltering",
                "workingColorSpace", "forceCpuExtraction", "maxThreads", "verboseLevel",
            ],
            outputFlags: ["output"],
            constantFlags: [],
            // No expectedOutputFile: emits per-view .feat/.desc files keyed
            // by view ID — too fragile to gate cache hit on a single name.
            outputs: ["output"],
            inputTypes: [
                "input": "file",
                "masksFolder": "file",
                "maskExtension": "string",
                "maskInvert": "bool",
                "describerTypes": "string",
                "describerPreset": "string",
                "maxNbFeatures": "int",
                "describerQuality": "string",
                "contrastFiltering": "string",
                "relativePeakThreshold": "double",
                "gridFiltering": "bool",
                "workingColorSpace": "string",
                "forceCpuExtraction": "bool",
                "maxThreads": "int",
                "verboseLevel": "string",
            ],
            outputTypes: ["output": "file"]
        ),
        "ImageMatching": Spec(
            executable: "aliceVision_imageMatching",
            inputFlags: [
                "input", "featuresFolders", "method", "tree", "minNbImages",
                "maxDescriptors", "nbMatches", "verboseLevel",
            ],
            outputFlags: ["output"],
            constantFlags: [],
            expectedOutputFile: "imageMatches.txt",
            outputs: ["output"],
            inputTypes: [
                "input": "file",
                "featuresFolders": "fileArray",
                "method": "string",
                "tree": "file",
                "minNbImages": "int",
                "maxDescriptors": "int",
                "nbMatches": "int",
                "verboseLevel": "string",
            ],
            outputTypes: ["output": "file"]
        ),
        "FeatureMatching": Spec(
            executable: "aliceVision_featureMatching",
            inputFlags: [
                "input", "featuresFolders", "imagePairsList", "describerTypes",
                "photometricMatchingMethod", "geometricEstimator", "geometricFilterType",
                "distanceRatio", "maxIteration", "verboseLevel",
            ],
            outputFlags: ["output"],
            constantFlags: [],
            // No expectedOutputFile: chunked, output is `<chunk>.matches.txt`.
            outputs: ["output"],
            inputTypes: [
                "input": "file",
                "featuresFolders": "fileArray",
                "imagePairsList": "file",
                "describerTypes": "string",
                "photometricMatchingMethod": "string",
                "geometricEstimator": "string",
                "geometricFilterType": "string",
                "distanceRatio": "double",
                "maxIteration": "int",
                "verboseLevel": "string",
            ],
            outputTypes: ["output": "file"]
        ),
        "StructureFromMotion": Spec(
            executable: "aliceVision_incrementalSfM",
            inputFlags: [
                "input", "featuresFolders", "matchesFolders", "describerTypes",
                "localizerEstimator", "observationConstraint", "verboseLevel",
            ],
            outputFlags: ["output", "outputViewsAndPoses", "extraInfoFolder"],
            constantFlags: [],
            // S42 patch routes SfM output via `sfm.sfm` (not the upstream
            // `.abc` ABC format which we don't have).  Confirmed present in
            // the existing Python-Meshroom cache.
            expectedOutputFile: "sfm.sfm",
            // SfM upstream exposes three distinct outputs.  See
            // `meshroom-mac/nodes/aliceVision/StructureFromMotion.py:374`.
            outputs: ["extraInfoFolder", "output", "outputViewsAndPoses"],
            inputTypes: [
                "input": "file",
                "featuresFolders": "fileArray",
                "matchesFolders": "fileArray",
                "describerTypes": "string",
                "localizerEstimator": "string",
                "observationConstraint": "string",
                "verboseLevel": "string",
            ],
            outputTypes: [
                "output": "file",
                "outputViewsAndPoses": "file",
                "extraInfoFolder": "file",
            ]
        ),
        "PrepareDenseScene": Spec(
            executable: "aliceVision_prepareDenseScene",
            inputFlags: [
                "input", "imagesFolders", "masksFolders", "outputFileType",
                "saveMetadata", "saveMatricesTxtFiles", "evCorrection", "verboseLevel",
            ],
            outputFlags: ["output"],
            constantFlags: [],
            // No expectedOutputFile: emits per-view `<viewId>.exr`.
            // PrepareDenseScene declares an `undistorted` semantic-only output
            // alongside `output`, but `commandLineGroup=""` excludes it from
            // graph connection use — we expose just `output`.
            outputs: ["output"],
            inputTypes: [
                "input": "file",
                "imagesFolders": "fileArray",
                "masksFolders": "fileArray",
                "outputFileType": "string",
                "saveMetadata": "bool",
                "saveMatricesTxtFiles": "bool",
                "evCorrection": "bool",
                "verboseLevel": "string",
            ],
            outputTypes: ["output": "file"]
        ),
        "DepthMap": Spec(
            executable: "aliceVision_depthMapEstimation",
            inputFlags: [
                "input", "imagesFolder", "downscale", "minViewAngle", "maxViewAngle",
                "verboseLevel",
            ],
            outputFlags: ["output"],
            constantFlags: [],
            // Upstream Meshroom: `parallelization = desc.Parallelization(blockSize=3)`
            // + `commandLineRange = "--rangeStart {rangeStart} --rangeSize {rangeSize}"`.
            // We rephrase `{rangeStart}`/`{rangeSize}` as `{start}`/`{size}` so the
            // template is decoupled from Meshroom's Python attribute names.
            parallelization: Parallelization(
                blockSize: 3,
                rangeArg: "--rangeStart {start} --rangeSize {size}"
            ),
            // DepthMap also declares `depth`/`sim`/`tilePattern` outputs but
            // those carry `commandLineGroup=""` and are descriptive-only
            // (read by the 2D viewer); we expose only `output`.
            outputs: ["output"],
            inputTypes: [
                "input": "file",
                "imagesFolder": "file",
                "downscale": "int",
                "minViewAngle": "double",
                "maxViewAngle": "double",
                "verboseLevel": "string",
            ],
            outputTypes: ["output": "file"]
        ),
        "DepthMapFilter": Spec(
            executable: "aliceVision_depthMapFiltering",
            inputFlags: [
                "input", "depthMapsFolder", "minViewAngle", "maxViewAngle",
                "minNumOfConsistentCams", "minNumOfConsistentCamsWithLowSimilarity",
                "verboseLevel",
            ],
            outputFlags: ["output"],
            constantFlags: [],
            // Upstream: blockSize=10 for the filtering stage (cheaper per view).
            parallelization: Parallelization(
                blockSize: 10,
                rangeArg: "--rangeStart {start} --rangeSize {size}"
            ),
            // Same reasoning as DepthMap: `depth`/`sim` outputs are
            // descriptive-only.
            outputs: ["output"],
            inputTypes: [
                "input": "file",
                "depthMapsFolder": "file",
                "minViewAngle": "double",
                "maxViewAngle": "double",
                "minNumOfConsistentCams": "int",
                "minNumOfConsistentCamsWithLowSimilarity": "int",
                "verboseLevel": "string",
            ],
            outputTypes: ["output": "file"]
        ),
        "Meshing": Spec(
            executable: "aliceVision_meshing",
            inputFlags: [
                "input", "depthMapsFolder", "estimateSpaceFromSfM",
                "estimateSpaceMinObservations", "estimateSpaceMinObservationAngle",
                "maxInputPoints", "maxPoints", "maxPointsPerVoxel", "minStep",
                "partitioning", "repartition", "angleFactor",
                "saveRawDensePointCloud", "voteFilteringForWeaklySupportedSurfaces",
                "outputMeshFileType", "verboseLevel",
            ],
            outputFlags: ["output", "outputMesh"],
            constantFlags: [],
            expectedOutputFile: "densePointCloud.ply",
            // Meshing upstream exposes `outputMesh` (the .obj/.fbx mesh) and
            // `output` (the dense point cloud SfMData/PLY).
            outputs: ["output", "outputMesh"],
            inputTypes: [
                "input": "file",
                "depthMapsFolder": "file",
                "estimateSpaceFromSfM": "bool",
                "estimateSpaceMinObservations": "int",
                "estimateSpaceMinObservationAngle": "double",
                "maxInputPoints": "int",
                "maxPoints": "int",
                "maxPointsPerVoxel": "int",
                "minStep": "int",
                "partitioning": "string",
                "repartition": "string",
                "angleFactor": "double",
                "saveRawDensePointCloud": "bool",
                "voteFilteringForWeaklySupportedSurfaces": "bool",
                "outputMeshFileType": "string",
                "verboseLevel": "string",
            ],
            outputTypes: [
                "output": "file",
                "outputMesh": "file",
            ]
        ),
        "MeshFiltering": Spec(
            executable: "aliceVision_meshFiltering",
            inputFlags: [
                "inputMesh", "outputMeshFileType", "keepLargestMeshOnly",
                "smoothingSubset", "smoothingBoundariesNeighbours", "smoothingIterations",
                "smoothingLambda", "filteringSubset", "filteringIterations",
                "filterLargeTrianglesFactor", "filterTrianglesRatio", "verboseLevel",
            ],
            outputFlags: ["outputMesh"],
            constantFlags: [],
            expectedOutputFile: "mesh.obj",
            // MeshFiltering upstream exposes a single output named `outputMesh`.
            outputs: ["outputMesh"],
            inputTypes: [
                "inputMesh": "file",
                "outputMeshFileType": "string",
                "keepLargestMeshOnly": "bool",
                "smoothingSubset": "string",
                "smoothingBoundariesNeighbours": "int",
                "smoothingIterations": "int",
                "smoothingLambda": "double",
                "filteringSubset": "string",
                "filteringIterations": "int",
                "filterLargeTrianglesFactor": "double",
                "filterTrianglesRatio": "double",
                "verboseLevel": "string",
            ],
            outputTypes: ["outputMesh": "file"]
        ),
        "Texturing": Spec(
            executable: "aliceVision_texturing",
            inputFlags: [
                "input", "inputMesh", "inputRefMesh", "imagesFolder",
                "textureSide", "downscale", "outputMeshFileType",
                "colorMappingFileType", "unwrapMethod", "useUDIM", "fillHoles",
                "padding", "verboseLevel",
            ],
            outputFlags: ["output", "outputMesh", "outputMaterial", "outputTextures"],
            constantFlags: [],
            // No expectedOutputFile: emits multiple texture artefacts and
            // `texturedMesh.<ext>` whose extension depends on the
            // outputMeshFileType param.  Skip warm-cache detection.
            // Texturing upstream exposes 4 distinct outputs.
            outputs: ["output", "outputMaterial", "outputMesh", "outputTextures"],
            inputTypes: [
                "input": "file",
                "inputMesh": "file",
                "inputRefMesh": "file",
                "imagesFolder": "file",
                "textureSide": "int",
                "downscale": "int",
                "outputMeshFileType": "string",
                "colorMappingFileType": "string",
                "unwrapMethod": "string",
                "useUDIM": "bool",
                "fillHoles": "bool",
                "padding": "int",
                "verboseLevel": "string",
            ],
            outputTypes: [
                "output": "file",
                "outputMesh": "file",
                "outputMaterial": "file",
                "outputTextures": "file",
            ]
        ),
        // ImportMiddlebury isn't a native pipeline node but we ship the binary
        // — wiring it here too for completeness.
        "ImportMiddlebury": Spec(
            executable: "aliceVision_importMiddlebury",
            inputFlags: ["sceneName", "imageFolder"],
            outputFlags: ["output"],
            constantFlags: [],
            outputs: ["output"],
            inputTypes: [
                "sceneName": "string",
                "imageFolder": "file",
            ],
            outputTypes: ["output": "file"]
        ),
    ]

    /// Lookup helper.  Returns `nil` for node types the M5 runner does not
    /// support yet (e.g., `CopyFiles`, `Publish`) — the caller treats those as
    /// skipped rather than failing the pipeline outright.
    public static func spec(for nodeType: String) -> Spec? {
        specs[nodeType]
    }
}
