import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// M6 unit tests for `NodeUIDHasher`.
///
/// All expected hex values come from a Python reference run:
/// ```
/// hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
/// ```
/// — see comments above each test for the exact one-liner.  If the canonical
/// JSON writer drifts from Python's `json.dumps` for any reason (whitespace,
/// escape rules, float formatting), these tests will detect it before it
/// silently corrupts the cache layout.
final class NodeUIDHasherTests: XCTestCase {

    /// Python: `hashlib.sha1(json.dumps({}, sort_keys=True, default=str).encode()).hexdigest()`
    func testEmptyInputsMatchesPythonSha1() {
        let uid = NodeUIDHasher.uid(for: [:])
        XCTAssertEqual(uid, "bf21a9e8fbc5a3846fb05b4fa0859e0917b2202f")
    }

    /// Python: `... {"a": 1, "b": 2} ...`  Verifies the writer emits the
    /// default `, ` and `: ` separators (compact-with-spaces, Python's default).
    func testSortedIntKeysMatchesPythonSha1() {
        let inputs: [String: MGJSONValue] = [
            "a": .int(1),
            "b": .int(2),
        ]
        let uid = NodeUIDHasher.uid(for: inputs)
        XCTAssertEqual(uid, "1744f53e00fc23bd3e515b298e42936485061dba")
    }

    /// Python: `... {"v": "info", "n": 42, "f": 0.5, "b": True, "nil": None, "arr": [1,2,3]} ...`
    /// Covers strings, ints, floats, bools, null, and arrays in one shot.
    func testMixedTypesMatchPython() {
        let inputs: [String: MGJSONValue] = [
            "v": .string("info"),
            "n": .int(42),
            "f": .double(0.5),
            "b": .bool(true),
            "nil": .null,
            "arr": .array([.int(1), .int(2), .int(3)]),
        ]
        let uid = NodeUIDHasher.uid(for: inputs)
        XCTAssertEqual(uid, "49f0d009086dce4f31cac187632e2ca6e52569ce")
    }

    /// Python: `... {"input": "/img", "advanced": {"foo": 1, "bar": [True, False]}} ...`
    /// Exercises nested objects + nested arrays inside an object.
    func testNestedObjectMatchesPython() {
        let inputs: [String: MGJSONValue] = [
            "input": .string("/img"),
            "advanced": .object([
                "foo": .int(1),
                "bar": .array([.bool(true), .bool(false)]),
            ]),
        ]
        let uid = NodeUIDHasher.uid(for: inputs)
        XCTAssertEqual(uid, "895830d3317e6002b29d570fa40c6da082e6a5b6")
    }

    /// Python: `... {"path": "/img/é.jpg", "label": "中"} ...`
    /// `ensure_ascii=True` default: é -> `é`, 中 -> `中`, lowercase hex.
    func testUnicodeEscapesLikePython() {
        let inputs: [String: MGJSONValue] = [
            "path": .string("/img/é.jpg"),
            "label": .string("中"),
        ]
        let uid = NodeUIDHasher.uid(for: inputs)
        XCTAssertEqual(uid, "24d712bda67fd67df7c342a35ee5aa496c9070c0")
    }

    /// Determinism: identical input dicts must produce identical UIDs across
    /// calls (no hidden hash-randomization, no time component).
    func testDeterministic() {
        let inputs: [String: MGJSONValue] = [
            "viewpoints": .string("/img"),
            "defaultFieldOfView": .double(45.0),
            "verboseLevel": .string("info"),
        ]
        let a = NodeUIDHasher.uid(for: inputs)
        let b = NodeUIDHasher.uid(for: inputs)
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.count, 40, "SHA-1 hex is 40 chars; got '\(a)'")
    }

    /// Sensitivity: changing a single input value must change the UID.
    /// This is the core property that makes the cache key sound.
    func testSensitivityToInputChange() {
        let a: [String: MGJSONValue] = ["defaultFieldOfView": .double(45.0)]
        let b: [String: MGJSONValue] = ["defaultFieldOfView": .double(60.0)]
        XCTAssertNotEqual(NodeUIDHasher.uid(for: a), NodeUIDHasher.uid(for: b))
    }

    /// Sort-key determinism: a dict literal whose key insertion order differs
    /// from sorted order must still hash the same as the canonically-sorted
    /// version.  This pins down the `sort_keys=True` behaviour.
    func testKeyOrderIndependent() {
        // Same dict, different conceptual insertion orders.  Swift dicts are
        // already unordered, but we construct two literals to make the
        // contract explicit.
        let a: [String: MGJSONValue] = [
            "b": .int(2),
            "a": .int(1),
        ]
        let b: [String: MGJSONValue] = [
            "a": .int(1),
            "b": .int(2),
        ]
        XCTAssertEqual(NodeUIDHasher.uid(for: a), NodeUIDHasher.uid(for: b))
        XCTAssertEqual(
            NodeUIDHasher.uid(for: a),
            "1744f53e00fc23bd3e515b298e42936485061dba"
        )
    }

    /// Float `1.0` must serialize as `"1.0"` (with the trailing decimal),
    /// not `"1"` — Python's `json.dumps(1.0) == "1.0"`.  This is the gotcha
    /// the M6 brief explicitly calls out.
    func testFloatPreservesTrailingZero() {
        let json = NodeUIDHasher.canonicalJSON(.object(["x": .double(1.0)]))
        XCTAssertEqual(json, #"{"x": 1.0}"#)
    }

    /// Int `1` must serialize as `"1"`, not `"1.0"` — keeps integer flags
    /// distinct from float flags.
    func testIntHasNoTrailingZero() {
        let json = NodeUIDHasher.canonicalJSON(.object(["x": .int(1)]))
        XCTAssertEqual(json, #"{"x": 1}"#)
    }

    /// Slash escaping: Python's `json.dumps` does NOT escape `/` by default.
    /// If we ever switch to e.g. `JSONEncoder` with the default
    /// `OutputFormatting`, this test will catch the regression.
    func testSlashNotEscaped() {
        let json = NodeUIDHasher.canonicalJSON(.object(["p": .string("/a/b")]))
        XCTAssertEqual(json, #"{"p": "/a/b"}"#)
    }

    /// Backslash + quote escaping.  Python: `"\\\"and\\\\back"` for input
    /// `"and\backslash` etc.  We pin a representative case.
    func testBackslashAndQuoteEscaping() {
        let json = NodeUIDHasher.canonicalJSON(.string("a\"b\\c"))
        XCTAssertEqual(json, #""a\"b\\c""#)
    }

    /// Control char `\t` becomes `\t`, `\x1f` becomes `` (lowercase).
    func testControlCharsEscapeLikePython() {
        let inputString = "a\tb\u{1f}c"
        let json = NodeUIDHasher.canonicalJSON(.string(inputString))
        // Build expected via concatenation to keep the source file free of
        // unprintable bytes.  The JSON representation itself is ASCII-safe.
        let expected = "\"a\\tb\\u001fc\""
        XCTAssertEqual(json, expected)
    }
}

/// M6 tests for `NodeUIDHasher.uids(for:topoOrder:)` — the graph-level pass
/// that resolves template references against upstream UIDs and produces a
/// Merkle-style dependency chain.
final class NodeUIDHasherGraphTests: XCTestCase {

    /// Same graph topology, same inputs -> same UIDs across two passes.
    func testGraphUIDsDeterministic() throws {
        let project = try makeTwoNodeProject()
        let order = try GraphScheduler.topologicalOrder(project.graph)
        let a = NodeUIDHasher.uids(for: project.graph, topoOrder: order)
        let b = NodeUIDHasher.uids(for: project.graph, topoOrder: order)
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.count, 2)
    }

    /// Changing an upstream input must also change the downstream UID —
    /// proves the Merkle property: downstream UIDs depend on the upstream's
    /// resolved output path which itself contains the upstream UID.
    func testUpstreamChangeInvalidatesDownstream() throws {
        var project = try makeTwoNodeProject()
        let order = try GraphScheduler.topologicalOrder(project.graph)
        let original = NodeUIDHasher.uids(for: project.graph, topoOrder: order)

        // Mutate one upstream input.
        var camera = project.graph["CameraInit_1"]!
        camera.inputs?["defaultFieldOfView"] = .double(60.0)  // was 45.0
        project.graph["CameraInit_1"] = camera

        let updated = NodeUIDHasher.uids(for: project.graph, topoOrder: order)
        XCTAssertNotEqual(original["CameraInit_1"], updated["CameraInit_1"])
        XCTAssertNotEqual(
            original["FeatureExtraction_1"], updated["FeatureExtraction_1"],
            "downstream UID must change when upstream input changes"
        )
    }

    /// Position changes must NOT invalidate the UID — position lives in
    /// `internalInputs` (or `position` directly), neither of which are part
    /// of the hash.  Crucial for the M6 promise that "dragging a node on
    /// the canvas doesn't blow away the cache".
    func testPositionChangeDoesNotInvalidate() throws {
        var project = try makeTwoNodeProject()
        let order = try GraphScheduler.topologicalOrder(project.graph)
        let before = NodeUIDHasher.uids(for: project.graph, topoOrder: order)

        // Drag CameraInit_1 to a new position.
        var camera = project.graph["CameraInit_1"]!
        camera.position = MGPosition(x: 9999, y: -1234)
        // Also mutate an internalInputs label (color, etc.) — those must be
        // ignored too.
        camera.internalInputs = ["label": .string("Camera (renamed)")]
        project.graph["CameraInit_1"] = camera

        let after = NodeUIDHasher.uids(for: project.graph, topoOrder: order)
        XCTAssertEqual(before, after, "position/internalInputs must not affect UID")
    }

    /// Helper: build a two-node project with a CameraInit -> FeatureExtraction
    /// edge using a template ref.  Exposed via `MGProject(jsonObject:)` to
    /// keep test fixtures terse.
    private func makeTwoNodeProject() throws -> MGProject {
        let cameraInit = MGNode(
            nodeType: "CameraInit",
            position: MGPosition(x: 0, y: 0),
            inputs: [
                "viewpoints": .string("/img"),
                "defaultFieldOfView": .double(45.0),
            ],
            outputs: ["output": .string("{nodeCacheFolder}/cameraInit.sfm")]
        )
        let featureExtraction = MGNode(
            nodeType: "FeatureExtraction",
            position: MGPosition(x: 200, y: 0),
            inputs: ["input": .string("{CameraInit_1.output}")],
            outputs: ["output": .string("{nodeCacheFolder}")]
        )
        let header = MGHeader(releaseVersion: "2025.1", fileVersion: "2.0")
        let graph = MGGraph(nodes: [
            "CameraInit_1": cameraInit,
            "FeatureExtraction_1": featureExtraction,
        ])
        return MGProject(header: header, graph: graph)
    }
}
