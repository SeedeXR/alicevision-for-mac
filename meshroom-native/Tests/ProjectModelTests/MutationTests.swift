import XCTest
@testable import ProjectModel

/// M3 model-layer tests for the mutation API introduced on `MGProject`.
///
/// These live in `ProjectModelTests` (not `AppTests`) because they exercise
/// nothing SwiftUI: just struct mutation and the on-disk round-trip.  Keeping
/// them at the model layer means future ports (a CLI tool, a non-SwiftUI
/// front-end) can rely on the same guarantees.
final class MutationTests: XCTestCase {

    // MARK: - Fixture helpers

    /// Minimal in-memory project with one node that has an integer position.
    /// Building from scratch (rather than loading a fixture) makes the
    /// mutation tests deterministic and independent of the bundled `.mg`
    /// files' contents.
    private func makeProject() -> MGProject {
        let node = MGNode(
            nodeType: "FeatureExtraction",
            position: MGPosition(x: 100, y: 50, integerStorage: true),
            inputs: ["scale": .int(2)]
        )
        return MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: ["FeatureExtraction_1": node])
        )
    }

    // MARK: - updateNodePosition

    func testUpdateNodePositionMutatesTheRightNode() {
        var project = makeProject()
        let didUpdate = project.updateNodePosition(
            name: "FeatureExtraction_1",
            position: MGPosition(x: 300, y: 200)
        )
        XCTAssertTrue(didUpdate)
        let pos = project.graph["FeatureExtraction_1"]?.position
        XCTAssertEqual(pos?.x, 300)
        XCTAssertEqual(pos?.y, 200)
    }

    func testUpdateNodePositionPreservesIntegerStorage() {
        // The on-disk format flag (int vs float storage) must survive a
        // mutation — Meshroom's writer keys on this flag and we want to
        // re-emit the file in the same form the user opened it in.
        var project = makeProject()
        project.updateNodePosition(
            name: "FeatureExtraction_1",
            position: MGPosition(x: 1, y: 2, integerStorage: false) // wrong flag
        )
        // Should have been coerced back to the original integer-storage flag.
        XCTAssertEqual(project.graph["FeatureExtraction_1"]?.position?.integerStorage, true)
    }

    func testUpdateNodePositionPreservesFloatStorage() {
        // Inverse case: a node that originally stored floats keeps floats.
        let floatNode = MGNode(
            nodeType: "X",
            position: MGPosition(x: 1.5, y: 2.5, integerStorage: false)
        )
        var project = MGProject(
            header: MGHeader(releaseVersion: "t", fileVersion: "2.0"),
            graph: MGGraph(nodes: ["X_1": floatNode])
        )
        project.updateNodePosition(
            name: "X_1",
            position: MGPosition(x: 9, y: 9, integerStorage: true) // wrong flag
        )
        XCTAssertEqual(project.graph["X_1"]?.position?.integerStorage, false)
    }

    func testUpdateNodePositionUnknownNodeIsNoOp() {
        var project = makeProject()
        let didUpdate = project.updateNodePosition(
            name: "DoesNotExist_42",
            position: MGPosition(x: 0, y: 0)
        )
        XCTAssertFalse(didUpdate)
        // Existing node untouched.
        XCTAssertEqual(project.graph["FeatureExtraction_1"]?.position?.x, 100)
    }

    // MARK: - Round-trip after edit

    /// The critical invariant: after editing a position, encode + decode
    /// must reproduce the edited value byte-for-byte.  This guards against a
    /// later refactor that accidentally drops mutations on the writer side.
    func testRoundTripPreservesEditedPosition() throws {
        var project = makeProject()
        project.updateNodePosition(
            name: "FeatureExtraction_1",
            position: MGPosition(x: 777, y: 888)
        )
        let data = try project.encode()
        let reDecoded = try MGProject.decode(from: data)
        XCTAssertEqual(reDecoded.graph["FeatureExtraction_1"]?.position?.x, 777)
        XCTAssertEqual(reDecoded.graph["FeatureExtraction_1"]?.position?.y, 888)
        // Whole-project equality also holds: other fields are untouched.
        XCTAssertEqual(project, reDecoded)
    }

    // MARK: - Atomic save

    /// Atomic save must:
    /// 1. Produce a file at the target URL with the encoded bytes.
    /// 2. Leave NO `.tmp` sibling behind after success.
    /// 3. Be safe to call repeatedly (replacing the existing file).
    func testWriteAtomicallyProducesValidFile() throws {
        var project = makeProject()
        project.updateNodePosition(
            name: "FeatureExtraction_1",
            position: MGPosition(x: 42, y: 24)
        )
        let url = uniqueTempURL(ext: "mg")
        defer { try? FileManager.default.removeItem(at: url) }

        try project.writeAtomically(to: url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        // No stale .tmp sibling.
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.appendingPathExtension("tmp").path))

        // Re-decode and confirm content fidelity.
        let reloaded = try MGProject.load(from: url)
        XCTAssertEqual(reloaded.graph["FeatureExtraction_1"]?.position?.x, 42)
        XCTAssertEqual(reloaded.graph["FeatureExtraction_1"]?.position?.y, 24)
    }

    func testWriteAtomicallyOverwritesExisting() throws {
        var project = makeProject()
        let url = uniqueTempURL(ext: "mg")
        defer { try? FileManager.default.removeItem(at: url) }

        try project.writeAtomically(to: url)
        // Second write with a different position must replace the first.
        project.updateNodePosition(
            name: "FeatureExtraction_1",
            position: MGPosition(x: 1, y: 1)
        )
        try project.writeAtomically(to: url)

        let reloaded = try MGProject.load(from: url)
        XCTAssertEqual(reloaded.graph["FeatureExtraction_1"]?.position?.x, 1)
        XCTAssertEqual(reloaded.graph["FeatureExtraction_1"]?.position?.y, 1)
    }

    // MARK: - updateAttribute (M4)

    /// Project fixture for the attribute-mutation tests.  We compose a richer
    /// `inputs` dict here — nested dict and nested array — so we can exercise
    /// all three path shapes (`"foo"`, `"foo.bar"`, `"foo.0.bar"`) against a
    /// single project value.
    private func makeAttrProject() -> MGProject {
        let nestedDict: MGJSONValue = .object([
            "bar": .object(["baz": .int(7)]),
        ])
        let nestedArray: MGJSONValue = .array([
            .object(["bar": .double(1.5)]),
            .object(["bar": .double(2.5)]),
        ])
        let node = MGNode(
            nodeType: "FeatureExtraction",
            position: MGPosition(x: 0, y: 0, integerStorage: true),
            inputs: [
                "focalLength": .double(45.0),
                "nestedDict": nestedDict,
                "nestedArr": nestedArray,
                "label": .string("hello"),
                "enabled": .bool(true),
            ]
        )
        return MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: ["FeatureExtraction_1": node])
        )
    }

    func testUpdateAttributeSimplePath() {
        var project = makeAttrProject()
        let ok = project.updateAttribute(
            node: "FeatureExtraction_1",
            attrPath: "focalLength",
            newValue: .double(60.5)
        )
        XCTAssertTrue(ok)
        XCTAssertEqual(
            project.graph["FeatureExtraction_1"]?.inputs?["focalLength"],
            .double(60.5)
        )
    }

    func testUpdateAttributeNestedDictPath() {
        var project = makeAttrProject()
        let ok = project.updateAttribute(
            node: "FeatureExtraction_1",
            attrPath: "nestedDict.bar.baz",
            newValue: .int(99)
        )
        XCTAssertTrue(ok)
        // Surrounding structure intact: bar still an object, just baz changed.
        guard case let .object(top) = project.graph["FeatureExtraction_1"]?.inputs?["nestedDict"],
              case let .object(bar) = top["bar"]
        else {
            return XCTFail("nestedDict.bar shape was destroyed by the mutation")
        }
        XCTAssertEqual(bar["baz"], .int(99))
    }

    func testUpdateAttributeNestedArrayPath() {
        var project = makeAttrProject()
        let ok = project.updateAttribute(
            node: "FeatureExtraction_1",
            attrPath: "nestedArr.0.bar",
            newValue: .double(9.5)
        )
        XCTAssertTrue(ok)
        guard case let .array(arr) = project.graph["FeatureExtraction_1"]?.inputs?["nestedArr"],
              arr.count == 2,
              case let .object(first) = arr[0],
              case let .object(second) = arr[1]
        else {
            return XCTFail("nestedArr shape was destroyed by the mutation")
        }
        XCTAssertEqual(first["bar"], .double(9.5))
        // Sibling element untouched — the mutation must not splash across
        // array indices.
        XCTAssertEqual(second["bar"], .double(2.5))
    }

    func testUpdateAttributeUnknownNodeIsNoOp() {
        var project = makeAttrProject()
        let ok = project.updateAttribute(
            node: "Ghost_99",
            attrPath: "focalLength",
            newValue: .double(0)
        )
        XCTAssertFalse(ok)
        // Existing project untouched.
        XCTAssertEqual(
            project.graph["FeatureExtraction_1"]?.inputs?["focalLength"],
            .double(45.0)
        )
    }

    func testUpdateAttributeInvalidPathIsNoOp() {
        var project = makeAttrProject()
        // Indexing past a scalar leaf must fail rather than silently corrupt.
        let ok = project.updateAttribute(
            node: "FeatureExtraction_1",
            attrPath: "focalLength.nope",
            newValue: .int(0)
        )
        XCTAssertFalse(ok)
        // focalLength must remain a double, not have been replaced by an
        // object that contains "nope".
        XCTAssertEqual(
            project.graph["FeatureExtraction_1"]?.inputs?["focalLength"],
            .double(45.0)
        )
    }

    func testUpdateAttributeRoundTripPreservesChange() throws {
        var project = makeAttrProject()
        project.updateAttribute(
            node: "FeatureExtraction_1",
            attrPath: "focalLength",
            newValue: .double(60.5)
        )
        project.updateAttribute(
            node: "FeatureExtraction_1",
            attrPath: "nestedDict.bar.baz",
            newValue: .int(99)
        )
        project.updateAttribute(
            node: "FeatureExtraction_1",
            attrPath: "nestedArr.1.bar",
            newValue: .double(7.25)
        )
        let data = try project.encode()
        let reDecoded = try MGProject.decode(from: data)
        let inputs = reDecoded.graph["FeatureExtraction_1"]?.inputs
        XCTAssertEqual(inputs?["focalLength"], .double(60.5))
        if case let .object(top) = inputs?["nestedDict"],
           case let .object(bar) = top["bar"] {
            XCTAssertEqual(bar["baz"], .int(99))
        } else {
            XCTFail("nested dict structure lost across round-trip")
        }
        if case let .array(arr) = inputs?["nestedArr"],
           case let .object(second) = arr[1] {
            XCTAssertEqual(second["bar"], .double(7.25))
        } else {
            XCTFail("nested array structure lost across round-trip")
        }
        // Full-project equality holds — nothing else changed.
        XCTAssertEqual(project, reDecoded)
    }

    func testUpdateAttributePreservesUntouchedFields() {
        var project = makeAttrProject()
        // Snapshot the unrelated inputs we DO NOT touch in this mutation.
        let originalLabel = project.graph["FeatureExtraction_1"]?.inputs?["label"]
        let originalEnabled = project.graph["FeatureExtraction_1"]?.inputs?["enabled"]
        let originalPosition = project.graph["FeatureExtraction_1"]?.position
        let originalNodeType = project.graph["FeatureExtraction_1"]?.nodeType

        project.updateAttribute(
            node: "FeatureExtraction_1",
            attrPath: "focalLength",
            newValue: .double(60.5)
        )

        XCTAssertEqual(project.graph["FeatureExtraction_1"]?.inputs?["label"], originalLabel)
        XCTAssertEqual(project.graph["FeatureExtraction_1"]?.inputs?["enabled"], originalEnabled)
        XCTAssertEqual(project.graph["FeatureExtraction_1"]?.position, originalPosition)
        XCTAssertEqual(project.graph["FeatureExtraction_1"]?.nodeType, originalNodeType)
    }

    // MARK: - Utilities

    /// Produce a uniquely-named URL in the per-test temporary directory.
    /// Using `URL.temporaryDirectory` rather than `NSTemporaryDirectory()`
    /// gives us a URL on the same volume as the eventual rename target
    /// (which matters because cross-volume `rename(2)` is not atomic — but
    /// `replaceItemAt` handles that case for us regardless).
    private func uniqueTempURL(ext: String) -> URL {
        let dir = FileManager.default.temporaryDirectory
        let name = "meshroom-mutation-test-\(UUID().uuidString).\(ext)"
        return dir.appendingPathComponent(name)
    }
}
