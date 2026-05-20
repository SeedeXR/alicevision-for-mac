import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// Tests for the DepthMap-style chunked execution support added on top of
/// `GraphExecutor`.  We cover:
///
/// - The pure chunk planner (`GraphExecutor.planChunks`) for the boundary
///   cases called out in the M6 spec (13 views @ blockSize 3, 1 view, 0 views,
///   uneven last chunk).
/// - The SfMData view counter against a hand-rolled JSON fixture mirroring
///   both `cameraInit.sfm` (post-CameraInit) and downstream `sfm.sfm` shapes.
/// - Range-arg substitution (`{start}` / `{size}` -> argv tokens).
/// - The `NodeBinary` spec table: DepthMap + DepthMapFilter expose
///   `parallelization`, every other node does not.
/// - Backward compatibility: argv assembly for a non-parallelized node is
///   unchanged (no `--rangeStart` / `--rangeSize`).
///
/// We deliberately do not spawn real Processes here; the chunk-aware run loop
/// is already exercised by `GraphExecutorTests.testCancelMidRunTerminatesProcess`
/// indirectly (a cancellation mid-chunk would surface as a `.failed` event the
/// same way).  Tests that need a Process round-trip would have to ship a
/// throwaway depth-map binary, which costs more than it gains.
@MainActor
final class ChunkedExecutionTests: XCTestCase {

    private func tmpDir() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("chunk-test-\(UUID().uuidString)", isDirectory: true)
    }

    // MARK: - planChunks: pure planning

    /// 13 views / blockSize 3 -> 5 chunks at offsets (0,3) (3,3) (6,3) (9,3) (12,1).
    /// This is the canonical mission spec example.
    func testPlanChunksMission13Views() {
        let chunks = GraphExecutor.planChunks(viewCount: 13, blockSize: 3)
        XCTAssertEqual(chunks.count, 5)
        XCTAssertEqual(chunks.map(\.start), [0, 3, 6, 9, 12])
        XCTAssertEqual(chunks.map(\.size), [3, 3, 3, 3, 1])
        for (i, c) in chunks.enumerated() {
            XCTAssertEqual(c.index, i)
            XCTAssertEqual(c.total, 5)
        }
    }

    /// Single view -> single chunk of size 1.  Real-world test cases include
    /// Monstree mini1 datasets.
    func testPlanChunksSingleView() {
        let chunks = GraphExecutor.planChunks(viewCount: 1, blockSize: 3)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].start, 0)
        XCTAssertEqual(chunks[0].size, 1)
        XCTAssertEqual(chunks[0].total, 1)
    }

    /// Zero views -> empty plan.  The executor turns this into a `.nodeSkipped`
    /// event rather than spawning an empty Process invocation.
    func testPlanChunksZeroViewsYieldsEmpty() {
        let chunks = GraphExecutor.planChunks(viewCount: 0, blockSize: 3)
        XCTAssertTrue(chunks.isEmpty)
    }

    /// blockSize > viewCount -> one chunk that processes all views.
    /// This covers DepthMapFilter's blockSize=10 with smaller datasets.
    func testPlanChunksBlockLargerThanInput() {
        let chunks = GraphExecutor.planChunks(viewCount: 4, blockSize: 10)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].start, 0)
        XCTAssertEqual(chunks[0].size, 4)
    }

    /// Exact multiples: 9 / 3 -> 3 equal chunks, no remainder.
    func testPlanChunksExactMultiple() {
        let chunks = GraphExecutor.planChunks(viewCount: 9, blockSize: 3)
        XCTAssertEqual(chunks.count, 3)
        XCTAssertEqual(chunks.map(\.size), [3, 3, 3])
    }

    /// Degenerate blockSize 0 should not loop forever — return empty plan.
    /// Defensive coverage for malformed specs.
    func testPlanChunksZeroBlockSize() {
        let chunks = GraphExecutor.planChunks(viewCount: 10, blockSize: 0)
        XCTAssertTrue(chunks.isEmpty)
    }

    // MARK: - SfMData view counting

    /// `cameraInit.sfm`-shaped JSON: top-level `views` array of N objects.
    func testCountViewsInSfMData() throws {
        let tmp = tmpDir()
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let path = tmp.appendingPathComponent("cameraInit.sfm").path
        let sfm: [String: Any] = [
            "version": ["1", "2", "8"],
            "views": (0..<7).map { i in
                ["viewId": "\(100 + i)", "intrinsicId": "1", "path": "/tmp/img-\(i).jpg"]
            },
            "intrinsics": [],
            "poses": [],
        ]
        let data = try JSONSerialization.data(withJSONObject: sfm)
        try data.write(to: URL(fileURLWithPath: path))
        XCTAssertEqual(GraphExecutor.countViewsInSfMData(at: path), 7)
    }

    /// Downstream `sfm.sfm` shape (post-SfM): same `views` array key.
    /// SfM strips un-reconstructed views — the count drops but field name
    /// stays.  Our counter must work uniformly for both pre- and post-SfM
    /// inputs.
    func testCountViewsInSfMDataDownstreamShape() throws {
        let tmp = tmpDir()
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let path = tmp.appendingPathComponent("sfm.sfm").path
        let sfm: [String: Any] = [
            "version": ["1", "2", "8"],
            "views": (0..<3).map { i in ["viewId": "\(i)"] },
            "intrinsics": [["intrinsicId": "1", "type": "pinhole"]],
            "poses": [["poseId": "0", "pose": ["transform": ["rotation": []]]]],
            "structure": [],
        ]
        let data = try JSONSerialization.data(withJSONObject: sfm)
        try data.write(to: URL(fileURLWithPath: path))
        XCTAssertEqual(GraphExecutor.countViewsInSfMData(at: path), 3)
    }

    /// Missing file -> 0 views (caller turns into `.nodeSkipped`, not crash).
    func testCountViewsMissingFileReturnsZero() {
        XCTAssertEqual(
            GraphExecutor.countViewsInSfMData(at: "/tmp/this-path-does-not-exist.sfm"),
            0
        )
    }

    /// Malformed JSON (e.g. an Alembic `.abc` binary or truncated file) -> 0.
    func testCountViewsMalformedReturnsZero() throws {
        let tmp = tmpDir()
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let path = tmp.appendingPathComponent("garbage.bin").path
        try Data([0x00, 0xff, 0x42, 0x10]).write(to: URL(fileURLWithPath: path))
        XCTAssertEqual(GraphExecutor.countViewsInSfMData(at: path), 0)
    }

    // MARK: - Range-arg rendering

    /// Default DepthMap template renders to two flag/value pairs in argv.
    func testRenderRangeArgsDefaultTemplate() {
        let chunk = GraphExecutor.Chunk(index: 1, total: 5, start: 3, size: 3)
        let args = GraphExecutor.renderRangeArgs(
            template: "--rangeStart {start} --rangeSize {size}",
            chunk: chunk
        )
        XCTAssertEqual(args, ["--rangeStart", "3", "--rangeSize", "3"])
    }

    /// Last (partial) chunk's size is honoured in the argv tokens.
    func testRenderRangeArgsPartialChunk() {
        let chunk = GraphExecutor.Chunk(index: 4, total: 5, start: 12, size: 1)
        let args = GraphExecutor.renderRangeArgs(
            template: "--rangeStart {start} --rangeSize {size}",
            chunk: chunk
        )
        XCTAssertEqual(args, ["--rangeStart", "12", "--rangeSize", "1"])
    }

    // MARK: - NodeBinary spec coverage

    /// Mission requirement: DepthMap declares parallelization with blockSize 3.
    func testDepthMapHasParallelizationSpec() throws {
        let spec = try XCTUnwrap(NodeBinary.spec(for: "DepthMap"))
        let par = try XCTUnwrap(spec.parallelization)
        XCTAssertEqual(par.blockSize, 3)
        XCTAssertTrue(par.rangeArg.contains("--rangeStart"))
        XCTAssertTrue(par.rangeArg.contains("{start}"))
        XCTAssertTrue(par.rangeArg.contains("{size}"))
    }

    /// DepthMapFilter declares parallelization with blockSize 10 (per upstream).
    func testDepthMapFilterHasParallelizationSpec() {
        let spec = NodeBinary.spec(for: "DepthMapFilter")
        XCTAssertNotNil(spec)
        XCTAssertEqual(spec?.parallelization?.blockSize, 10)
    }

    /// Non-parallelized nodes leave `parallelization` nil so the executor
    /// falls back to single-invocation behaviour (regression guard).
    func testOtherNodesHaveNoParallelization() {
        for type in [
            "CameraInit", "FeatureExtraction", "ImageMatching", "FeatureMatching",
            "StructureFromMotion", "PrepareDenseScene", "Meshing", "MeshFiltering",
            "Texturing",
        ] {
            XCTAssertNil(
                NodeBinary.spec(for: type)?.parallelization,
                "\(type) should not declare parallelization in M6"
            )
        }
    }

    // MARK: - Backward compatibility

    /// argv for a non-parallelized node must not contain range flags — the
    /// `buildArgv` path is purely about flags from the spec; chunking happens
    /// at the run-loop level.
    func testArgvForNonParallelizedNodeOmitsRangeFlags() {
        let exec = GraphExecutor()
        let node = MGNode(
            nodeType: "FeatureExtraction",
            position: MGPosition(x: 0, y: 0),
            inputs: ["input": .string("/tmp/x.sfm")],
            outputs: ["output": .string("{nodeCacheFolder}")]
        )
        let argv = exec.buildArgvForTesting(
            node: node,
            nodeName: "FE_1",
            cacheDir: tmpDir()
        )!
        XCTAssertFalse(argv.contains("--rangeStart"))
        XCTAssertFalse(argv.contains("--rangeSize"))
    }

    /// argv for DepthMap also omits range flags — the executor appends them
    /// per-chunk, not via `buildArgv`.  Pins the contract that the chunk
    /// expansion happens at the run loop layer.
    func testBuildArgvForDepthMapStillOmitsRangeFlags() {
        let exec = GraphExecutor()
        let node = MGNode(
            nodeType: "DepthMap",
            position: MGPosition(x: 0, y: 0),
            inputs: ["input": .string("/tmp/x.sfm"), "downscale": .int(2)],
            outputs: ["output": .string("{nodeCacheFolder}")]
        )
        let argv = exec.buildArgvForTesting(
            node: node,
            nodeName: "DepthMap_1",
            cacheDir: tmpDir()
        )!
        XCTAssertTrue(argv.contains("--input"))
        XCTAssertTrue(argv.contains("--downscale"))
        XCTAssertFalse(argv.contains("--rangeStart"))
        XCTAssertFalse(argv.contains("--rangeSize"))
    }

    // MARK: - ProjectViewModel chunk key parsing

    /// The chunk-key parser must round-trip the names the executor emits.
    func testParentNameForChunkKey() {
        XCTAssertEqual(
            ProjectViewModel.parentName(forChunkKey: "DepthMap_1[chunk 1/5]"),
            "DepthMap_1"
        )
        XCTAssertEqual(
            ProjectViewModel.parentName(forChunkKey: "DepthMapFilter_3[chunk 12/12]"),
            "DepthMapFilter_3"
        )
    }

    /// Plain node names without `[chunk ...]` markers must return nil so the
    /// UI doesn't accidentally hide them under a non-existent parent.
    func testParentNameForPlainKeyReturnsNil() {
        XCTAssertNil(ProjectViewModel.parentName(forChunkKey: "CameraInit_1"))
        XCTAssertNil(ProjectViewModel.parentName(forChunkKey: "FeatureExtraction"))
        // Defensive: a name containing `[` but not a chunk suffix shouldn't
        // misclassify.  Meshroom doesn't generate such names today, but the
        // parser is forgiving so a future identifier scheme doesn't surprise.
        XCTAssertNil(ProjectViewModel.parentName(forChunkKey: "Weird[Name"))
    }
}
