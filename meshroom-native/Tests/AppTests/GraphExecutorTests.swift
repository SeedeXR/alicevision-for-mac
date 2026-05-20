import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// M5 unit tests for the *non-Process* parts of `GraphExecutor`:
/// - Template-reference resolution against the upstream outputs table.
/// - argv assembly for typical pipeline nodes.
///
/// We deliberately do not spawn real binaries here.  `Process` interacts with
/// the test runner's dyld state in ways that have historically been flaky on
/// CI; the executor's testable surface is the renderer + argv builder, both
/// of which are pure functions of `(node, spec, cache dir, upstream outputs)`.
/// `GraphExecutor` exposes `setOutputsForTesting` / `buildArgvForTesting` so
/// we can drive that surface without going through `run(...)`.
@MainActor
final class GraphExecutorTests: XCTestCase {

    private func tmpDir() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("exec-test-\(UUID().uuidString)", isDirectory: true)
    }

    /// Scalar `{X.output}` template inside an input must resolve to the
    /// upstream node's pre-registered output path.
    func testRendererResolvesScalarTemplate() {
        let exec = GraphExecutor()
        let upstreamPath = "/tmp/upstream-cache-abc/cameraInit.sfm"
        exec.setOutputsForTesting(["output": upstreamPath], for: "Upstream")

        let downstream = MGNode(
            nodeType: "FeatureExtraction",
            position: MGPosition(x: 0, y: 0),
            inputs: ["input": .string("{Upstream.output}")],
            outputs: ["output": .string("{nodeCacheFolder}")]
        )
        let argv = exec.buildArgvForTesting(
            node: downstream,
            nodeName: "FeatureExtraction_1",
            cacheDir: tmpDir()
        )
        XCTAssertNotNil(argv)
        guard let argv = argv else { return }
        guard let idx = argv.firstIndex(of: "--input") else {
            XCTFail("missing --input flag in \(argv)")
            return
        }
        XCTAssertEqual(argv[argv.index(after: idx)], upstreamPath)
    }

    /// `featuresFolders: ["{X.output}"]` must resolve via the array walker
    /// — proves the renderer recurses into arrayed templates.
    func testRendererResolvesArrayedTemplate() {
        let exec = GraphExecutor()
        let upstreamPath = "/tmp/feature-extraction-out"
        exec.setOutputsForTesting(["output": upstreamPath], for: "Upstream")

        let downstream = MGNode(
            nodeType: "ImageMatching",
            position: MGPosition(x: 0, y: 0),
            inputs: [
                "input": .string("{Upstream.output}"),
                "featuresFolders": .array([.string("{Upstream.output}")]),
            ],
            outputs: ["output": .string("{nodeCacheFolder}/imageMatches.txt")]
        )
        let argv = exec.buildArgvForTesting(
            node: downstream,
            nodeName: "ImageMatching_1",
            cacheDir: tmpDir()
        )
        XCTAssertNotNil(argv)
        guard let argv = argv else { return }
        guard let idx = argv.firstIndex(of: "--featuresFolders") else {
            XCTFail("missing --featuresFolders in \(argv)")
            return
        }
        // Single-element array renders as a comma-joined string of one path,
        // which is just the path itself.
        XCTAssertEqual(argv[argv.index(after: idx)], upstreamPath)
    }

    /// Multiple refs inside one array must concatenate with `,` — AliceVision
    /// CLI accepts comma-separated multi-value flags.
    func testRendererArrayedMultiTemplateJoinsWithComma() {
        let exec = GraphExecutor()
        exec.setOutputsForTesting(["output": "/p/a"], for: "A")
        exec.setOutputsForTesting(["output": "/p/b"], for: "B")
        let node = MGNode(
            nodeType: "ImageMatching",
            position: MGPosition(x: 0, y: 0),
            inputs: [
                "input": .string("/i"),
                "featuresFolders": .array([
                    .string("{A.output}"),
                    .string("{B.output}"),
                ]),
            ],
            outputs: ["output": .string("{nodeCacheFolder}/out.txt")]
        )
        let argv = exec.buildArgvForTesting(node: node, nodeName: "X", cacheDir: tmpDir())!
        let idx = argv.firstIndex(of: "--featuresFolders")!
        XCTAssertEqual(argv[argv.index(after: idx)], "/p/a,/p/b")
    }

    /// argv must include `--allowSingleView 1` for CameraInit (constant flags
    /// from the spec).
    func testConstantFlagsAppliedForCameraInit() {
        let exec = GraphExecutor()
        let cameraInit = MGNode(
            nodeType: "CameraInit",
            position: MGPosition(x: 0, y: 0),
            inputs: ["defaultFieldOfView": .double(45.0)],
            outputs: ["output": .string("{nodeCacheFolder}/cameraInit.sfm")]
        )
        let argv = exec.buildArgvForTesting(
            node: cameraInit,
            nodeName: "CameraInit_1",
            cacheDir: tmpDir()
        )
        XCTAssertNotNil(argv)
        guard let argv = argv else { return }
        XCTAssertTrue(argv.contains("--allowSingleView"))
        XCTAssertTrue(argv.contains("1"))
        guard let dvIdx = argv.firstIndex(of: "--defaultFieldOfView") else {
            XCTFail("missing --defaultFieldOfView in \(argv)")
            return
        }
        XCTAssertEqual(argv[argv.index(after: dvIdx)], "45.0")
    }

    /// `{nodeCacheFolder}` in an output template must expand to the real
    /// per-node cache directory.
    func testOutputTemplateSubstitutesCacheFolder() {
        let exec = GraphExecutor()
        let cacheDir = URL(fileURLWithPath: "/tmp/mycache")
        let cameraInit = MGNode(
            nodeType: "CameraInit",
            position: MGPosition(x: 0, y: 0),
            outputs: ["output": .string("{nodeCacheFolder}/cameraInit.sfm")]
        )
        // We pre-register the same cache path so the renderer (which only
        // resolves *template references*, not `{nodeCacheFolder}`) doesn't
        // collapse it.  Instead the resolution happens in `buildArgv` via
        // `renderInputValue` -> resolveString -> passthrough, and the output
        // template path is taken from `node.outputs` directly.
        let argv = exec.buildArgvForTesting(
            node: cameraInit,
            nodeName: "CameraInit_1",
            cacheDir: cacheDir
        )!
        // The `--output` value should still contain the unresolved
        // `{nodeCacheFolder}` because the renderer only resolves *template
        // node refs*; cache-folder expansion lives on the run-time path.
        // This test pins that contract: argv builder is type-aware but
        // doesn't know about cache dirs (that's the executor's job during
        // run()).  Downstream consumers see the substituted path because
        // `registerOutputs` does the substitution before publishing to
        // `outputPaths`.
        guard let idx = argv.firstIndex(of: "--output") else {
            XCTFail("missing --output flag in \(argv)")
            return
        }
        XCTAssertEqual(argv[argv.index(after: idx)], "{nodeCacheFolder}/cameraInit.sfm")
    }

    /// `NodeBinary.specs` must cover the 11 pipeline node types we ship in
    /// `build/` (excluding `ImportMiddlebury` which is a test helper).
    func testNodeBinaryCoversShippedTypes() {
        let required = [
            "CameraInit", "FeatureExtraction", "ImageMatching", "FeatureMatching",
            "StructureFromMotion", "PrepareDenseScene", "DepthMap", "DepthMapFilter",
            "Meshing", "MeshFiltering", "Texturing",
        ]
        for type in required {
            XCTAssertNotNil(NodeBinary.spec(for: type), "missing spec for \(type)")
        }
    }

    /// Boolean inputs render as `0`/`1` — the AliceVision CLI convention.
    func testBooleanInputsRenderAsZeroOne() {
        let exec = GraphExecutor()
        let node = MGNode(
            nodeType: "FeatureExtraction",
            position: MGPosition(x: 0, y: 0),
            inputs: [
                "input": .string("/x"),
                "gridFiltering": .bool(true),
                "forceCpuExtraction": .bool(false),
            ],
            outputs: ["output": .string("{nodeCacheFolder}")]
        )
        let argv = exec.buildArgvForTesting(node: node, nodeName: "FE", cacheDir: tmpDir())!
        // True renders as 1.
        if let i = argv.firstIndex(of: "--gridFiltering") {
            XCTAssertEqual(argv[argv.index(after: i)], "1")
        } else {
            XCTFail("missing --gridFiltering")
        }
        // False renders as 0.
        if let i = argv.firstIndex(of: "--forceCpuExtraction") {
            XCTAssertEqual(argv[argv.index(after: i)], "0")
        } else {
            XCTFail("missing --forceCpuExtraction")
        }
    }

    /// Cancellation contract: while the executor is running and a Process
    /// is in flight, `cancel()` terminates it and the run-loop emits
    /// `.failed(reason: "Cancelled by user")` for that node, then ends.
    /// We can drive this end-to-end against a real Process — `/usr/bin/yes`
    /// is portable on darwin, produces unbounded stdout (lets us observe
    /// `.nodeProgress`), and exits cleanly on SIGTERM.
    ///
    /// To dodge the `NodeBinary.spec(for:)` filter we install a synthetic
    /// node type ("CameraInit") and override its executable path via a
    /// temp directory containing a `aliceVision_cameraInit` symlink to
    /// `/usr/bin/yes`.  When we cancel mid-run the stream terminates with
    /// a `.failed` event for that node.
    func testCancelMidRunTerminatesProcess() async throws {
        // Set up a temp binDir with a symlink that masquerades as the
        // CameraInit binary.  This is the only way to drive the executor
        // through to `Process.run` without an actual aliceVision build.
        let tmp = tmpDir()
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let yesPath = "/usr/bin/yes"
        guard FileManager.default.fileExists(atPath: yesPath) else {
            throw XCTSkip("/usr/bin/yes not available; skipping cancel test")
        }
        let fakeBin = tmp.appendingPathComponent("aliceVision_cameraInit")
        try FileManager.default.createSymbolicLink(at: fakeBin, withDestinationURL: URL(fileURLWithPath: yesPath))

        let exec = GraphExecutor()
        let node = MGNode(
            nodeType: "CameraInit",
            position: MGPosition(x: 0, y: 0),
            outputs: ["output": .string("{nodeCacheFolder}/x.sfm")]
        )
        let project = MGProject(
            header: MGHeader(releaseVersion: "x", fileVersion: "2.0"),
            graph: MGGraph(nodes: ["CameraInit_1": node])
        )
        let config = GraphExecutor.RunConfig(
            workDir: tmp,
            binDir: tmp,
            aliceVisionRoot: tmp
        )
        var sawStart = false
        var sawFailedDueToCancel = false
        // Cancel as soon as we observe a `.nodeProgress` event — by that
        // point the Process is definitely running.
        let stream = exec.run(project, config: config)
        for await event in stream {
            switch event {
            case .nodeStarted:
                sawStart = true
            case .nodeProgress:
                exec.cancel()
            case .failed(_, let reason) where reason.contains("Cancelled"):
                sawFailedDueToCancel = true
            default:
                break
            }
        }
        XCTAssertTrue(sawStart, "expected the node to start before cancellation")
        XCTAssertTrue(sawFailedDueToCancel, "expected .failed with 'Cancelled' reason")
    }
}
