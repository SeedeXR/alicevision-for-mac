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
    ///
    /// `coreSpecs` is the set of native AliceVision binaries hardcoded into
    /// the executable.  Plugin-supplied specs (see `loadPluginSpecs()`) are
    /// merged on top via the `specs` lazy-static below.  Tests and the
    /// palette read `specs`, which is the union — they never see
    /// `coreSpecs` directly.
    internal static let coreSpecs: [String: Spec] = [
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
        // Note: SegmentationBiRefNet (Python-only AI segmentation) was
        // previously listed here as a hardcoded entry.  After the S53
        // refactor it is supplied by the `ai-segmentation` plugin and
        // merged in at runtime via `loadPluginSpecs()`.  This keeps the
        // core Swift binary scope-limited to the 12 native AliceVision
        // binaries and lets third-party plugins ship their own Python or
        // wrapped-binary nodes without patching this file.
    ]

    /// Union of `coreSpecs` and every plugin-declared spec discovered at
    /// startup.  Computed exactly once and cached for the lifetime of the
    /// process.  Tests, the palette, and the executor all read this — they
    /// never see `coreSpecs` directly so plugin nodes are first-class.
    ///
    /// Swift's `static let` initialiser is run lazily and is thread-safe,
    /// so concurrent first reads from the palette/executor are safe.
    public static let specs: [String: Spec] = {
        var merged = coreSpecs
        for (name, spec) in loadPluginSpecs() {
            if merged[name] != nil {
                // Plugin specs deliberately override coreSpecs when names
                // collide — the plugin author "owns" the node namespace
                // once they declare it.  No warning yet; revisit when we
                // gain a logging facility.
            }
            merged[name] = spec
        }
        return merged
    }()

    /// Lookup helper.  Returns `nil` for node types the M5 runner does not
    /// support yet (e.g., `CopyFiles`, `Publish`) — the caller treats those as
    /// skipped rather than failing the pipeline outright.
    public static func spec(for nodeType: String) -> Spec? {
        specs[nodeType]
    }

    // MARK: - Plugin discovery (S53)

    /// Plugin-discovery path resolution rules, applied in order:
    ///
    /// 1. **Bundled plugins.** `Bundle.main.bundlePath/Contents/Resources/plugins/`
    ///    (when the app is packaged as `.app`) or `<bundlePath>/plugins/`.
    ///    Lets shipped binaries find plugins that were copied in at build time.
    /// 2. **Dev-mode fallback.** `<repo-root>/plugins/`, computed from `#filePath`
    ///    by walking up to the directory containing `Package.swift`.  Lets
    ///    `swift run` / `swift test` find plugins without an Xcode build.
    /// 3. **Environment override.** `MESHROOM_PLUGINS_DIR` if set — supports
    ///    out-of-tree integration tests and packaged distributions whose
    ///    plugins live elsewhere on disk.
    ///
    /// The first existing directory wins.  A missing plugins/ directory is
    /// not an error: the executor simply ships with just `coreSpecs`.
    static func pluginSearchPaths() -> [URL] {
        var paths: [URL] = []

        // 1. App bundle (.app/Contents/Resources/plugins/ or .app/plugins/)
        let bundleURL = URL(fileURLWithPath: Bundle.main.bundlePath)
        paths.append(bundleURL.appendingPathComponent("Contents/Resources/plugins"))
        paths.append(bundleURL.appendingPathComponent("plugins"))

        // 2. Dev-mode: walk up from this source file to repo-root/plugins/.
        var probe = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        while probe.path != "/" {
            let pkg = probe.appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: pkg.path) {
                // Package.swift lives at meshroom-native/Package.swift;
                // plugins/ is a sibling of meshroom-native/.
                paths.append(probe.deletingLastPathComponent()
                    .appendingPathComponent("plugins"))
                break
            }
            probe.deleteLastPathComponent()
        }

        // 3. Environment override.
        if let env = ProcessInfo.processInfo.environment["MESHROOM_PLUGINS_DIR"],
           !env.isEmpty {
            paths.append(URL(fileURLWithPath: env))
        }

        return paths
    }

    /// Glob `<pluginsDir>/*/plugin.json` across every candidate plugins
    /// directory, decode the manifest, and convert each declared node into
    /// a `Spec`.  Returns `[nodeName: Spec]` — the same shape as
    /// `coreSpecs` so the merge in `specs` is a single dict-overlay.
    public static func loadPluginSpecs() -> [String: Spec] {
        var registry: [String: Spec] = [:]
        PluginRegistry.shared.reset()

        for pluginsDir in pluginSearchPaths() {
            guard FileManager.default.fileExists(atPath: pluginsDir.path) else {
                continue
            }
            let contents: [URL]
            do {
                contents = try FileManager.default.contentsOfDirectory(
                    at: pluginsDir,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles]
                )
            } catch {
                continue
            }
            for pluginDir in contents {
                let manifestURL = pluginDir.appendingPathComponent("plugin.json")
                guard FileManager.default.fileExists(atPath: manifestURL.path) else {
                    continue
                }
                guard let manifest = try? PluginManifest.load(from: manifestURL) else {
                    // Swallow malformed manifests rather than crashing the
                    // host app — a broken third-party plugin must not take
                    // down the whole pipeline.  A future logging facility
                    // can surface these to the user.
                    continue
                }
                PluginRegistry.shared.register(manifest)
                let wrapperScript = manifest.resolvedWrapperScript(
                    relativeTo: pluginDir
                )
                for node in manifest.nodes {
                    let spec = Spec(
                        executable: wrapperScript,
                        inputFlags: Array(node.inputs.keys),
                        outputFlags: Array(node.outputs.keys),
                        constantFlags: node.constantFlags,
                        parallelization: node.parallelization,
                        expectedOutputFile: nil,
                        outputs: Array(node.outputs.keys),
                        inputTypes: node.inputs,
                        outputTypes: node.outputs
                    )
                    registry[node.name] = spec
                }
            }
            // First plugins/ dir to exist wins.  Stop after we found one.
            return registry
        }
        return registry
    }
}

// MARK: - Plugin manifest decoding

/// In-memory representation of a `plugin.json` manifest.  Mirrors the JSON
/// schema documented in `docs/dev/plugin-system.md`.
public struct PluginManifest: Decodable, Sendable, Hashable {
    public let name: String
    public let version: String
    public let description: String
    public let license: String?
    public let computeBackends: [String]?
    public let nodes: [PluginNodeDescriptor]
    public let wrapperScript: String

    enum CodingKeys: String, CodingKey {
        case name
        case version
        case description
        case license
        case computeBackends = "compute_backends"
        case nodes
        case wrapperScript = "wrapper_script"
    }

    public static func load(from url: URL) throws -> PluginManifest {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PluginManifest.self, from: data)
    }

    /// Compute the executable string to feed `GraphExecutor.runOneProcess`.
    /// The wrapper_script field in the manifest is relative to the plugin
    /// directory; we resolve it to a path the Swift executor's `binDir`
    /// joining logic produces a correct URL for.  The executor treats
    /// strings containing `/` as relative to `binDir`, so we keep the
    /// `../scripts/run_python_node.sh` shape that the old hardcoded entry
    /// used.  For the canonical ai-segmentation plugin this resolves to
    /// `../scripts/run_python_node.sh` (relative to `<repo>/build/`).
    public func resolvedWrapperScript(relativeTo pluginDir: URL) -> String {
        // The manifest stores the path relative to the plugin dir, e.g.
        // `../../meshroom-native/scripts/run_python_node.sh`.  The Swift
        // GraphExecutor's `binDir` is `<repo>/build/`, so to reach the
        // wrapper we need `../meshroom-native/scripts/run_python_node.sh`
        // (one level up from `build/`).  We translate by:
        //   absolute(plugin/wrapper) -> path relative to <repo>/build/
        let absoluteWrapper = pluginDir
            .appendingPathComponent(wrapperScript)
            .standardizedFileURL
        // Find the repo root from the plugin dir: pluginDir is at
        // `<repo>/plugins/<name>`, so repo root is two levels up.
        let repoRoot = pluginDir.deletingLastPathComponent().deletingLastPathComponent()
        let binDir = repoRoot.appendingPathComponent("build")
        // Build a relative path from `binDir` to `absoluteWrapper`.
        let wrapperComponents = absoluteWrapper.standardizedFileURL.pathComponents
        let binComponents = binDir.standardizedFileURL.pathComponents
        var common = 0
        while common < wrapperComponents.count,
              common < binComponents.count,
              wrapperComponents[common] == binComponents[common] {
            common += 1
        }
        let upHops = binComponents.count - common
        let downComponents = wrapperComponents[common...]
        var rel = Array(repeating: "..", count: upHops)
        rel.append(contentsOf: downComponents)
        return rel.joined(separator: "/")
    }
}

/// One node entry inside a plugin manifest.
public struct PluginNodeDescriptor: Decodable, Sendable, Hashable {
    public let name: String
    public let icon: String
    public let category: String
    public let inputs: [String: String]
    public let outputs: [String: String]
    public let constantFlags: [String]
    public let parallelization: NodeBinary.Parallelization?

    enum CodingKeys: String, CodingKey {
        case name
        case icon
        case category
        case inputs
        case outputs
        case constantFlags = "constant_flags"
        case parallelization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try c.decode(String.self, forKey: .name)
        self.icon = try c.decode(String.self, forKey: .icon)
        self.category = try c.decode(String.self, forKey: .category)
        self.inputs = try c.decode([String: String].self, forKey: .inputs)
        self.outputs = try c.decode([String: String].self, forKey: .outputs)
        self.constantFlags = try c.decodeIfPresent([String].self,
                                                   forKey: .constantFlags) ?? []
        self.parallelization = nil
    }
}

// MARK: - Plugin registry (diagnostics)

/// Process-wide registry of loaded plugins, populated by `loadPluginSpecs()`.
/// Lets diagnostic UI / `--plugins` CLI introspection list what is enabled.
public final class PluginRegistry: @unchecked Sendable {
    public static let shared = PluginRegistry()

    private let lock = NSLock()
    private var manifests: [PluginManifest] = []

    private init() {}

    /// Add a manifest to the registry.  Thread-safe.
    public func register(_ manifest: PluginManifest) {
        lock.lock()
        defer { lock.unlock() }
        if !manifests.contains(where: { $0.name == manifest.name }) {
            manifests.append(manifest)
        }
    }

    /// Drop every previously-registered manifest.  Used by
    /// `loadPluginSpecs()` to repopulate on each call, and by tests that
    /// want a clean slate.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        manifests.removeAll()
    }

    /// All currently-registered manifests.
    public var all: [PluginManifest] {
        lock.lock()
        defer { lock.unlock() }
        return manifests
    }

    /// Look up a manifest by plugin name.
    public func manifest(named name: String) -> PluginManifest? {
        lock.lock()
        defer { lock.unlock() }
        return manifests.first(where: { $0.name == name })
    }
}
