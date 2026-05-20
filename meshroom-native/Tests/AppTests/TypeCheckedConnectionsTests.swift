import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// M8 tests for type-checked connections.
///
/// We exercise the matrix directly (`PinLayout.areCompatible`), the
/// view-model's lookup (`isConnectionTypeCompatible`), and the commit-reject
/// path (`commitConnection` returns `false` + sets `shakeTarget`).  The
/// rubber-band's red-tint is a render-only concern; covered by manual smoke
/// test rather than XCTest.
@MainActor
final class TypeCheckedConnectionsTests: XCTestCase {

    // MARK: - Compatibility matrix

    func testFileToFileIsCompatible() {
        XCTAssertTrue(PinLayout.areCompatible(sourceType: "file", targetType: "file"))
    }

    func testFileToFileArrayIsCompatible() {
        XCTAssertTrue(PinLayout.areCompatible(sourceType: "file", targetType: "fileArray"))
    }

    func testIntToIntAndArrayAreCompatible() {
        XCTAssertTrue(PinLayout.areCompatible(sourceType: "int", targetType: "int"))
        XCTAssertTrue(PinLayout.areCompatible(sourceType: "int", targetType: "intArray"))
    }

    func testStringToStringIsCompatible() {
        XCTAssertTrue(PinLayout.areCompatible(sourceType: "string", targetType: "string"))
    }

    func testBoolToBoolIsCompatible() {
        XCTAssertTrue(PinLayout.areCompatible(sourceType: "bool", targetType: "bool"))
    }

    func testCrossTypeIsRejected() {
        XCTAssertFalse(PinLayout.areCompatible(sourceType: "file", targetType: "int"))
        XCTAssertFalse(PinLayout.areCompatible(sourceType: "int", targetType: "file"))
        XCTAssertFalse(PinLayout.areCompatible(sourceType: "string", targetType: "bool"))
        XCTAssertFalse(PinLayout.areCompatible(sourceType: "bool", targetType: "int"))
        // Array → scalar shrinks information; reject.
        XCTAssertFalse(PinLayout.areCompatible(sourceType: "fileArray", targetType: "file"))
    }

    /// Spec rule #5: unknown types must default to ALLOW so we never block
    /// the user on incomplete metadata.
    func testUnknownTypesDefaultToAllow() {
        XCTAssertTrue(PinLayout.areCompatible(sourceType: nil, targetType: "file"))
        XCTAssertTrue(PinLayout.areCompatible(sourceType: "file", targetType: nil))
        XCTAssertTrue(PinLayout.areCompatible(sourceType: nil, targetType: nil))
        XCTAssertTrue(PinLayout.areCompatible(sourceType: "", targetType: "file"))
        XCTAssertTrue(PinLayout.areCompatible(sourceType: "file", targetType: ""))
    }

    // MARK: - Spec-resolved compatibility

    /// CameraInit.output (file) → FeatureExtraction.input (file): compatible.
    func testRealPipelineConnectionFileToFileCompatible() {
        let project = makeTwoNodeProject(
            sourceType: "CameraInit",
            targetType: "FeatureExtraction"
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        XCTAssertTrue(vm.isConnectionTypeCompatible(
            from: "CameraInit_1", fromAttr: "output",
            to: "FeatureExtraction_1", toAttrPath: "input"
        ))
    }

    /// FeatureExtraction.output (file) → StructureFromMotion.featuresFolders
    /// (fileArray): file→fileArray is allowed.
    func testRealPipelineFileToFileArrayCompatible() {
        var project = makeTwoNodeProject(
            sourceType: "FeatureExtraction",
            targetType: "StructureFromMotion"
        )
        // SfM's relevant input is `featuresFolders` (a list-of-file).  Seed
        // it so the input attr exists on the node.
        var sfm = project.graph["StructureFromMotion_1"]!
        sfm.inputs?["featuresFolders"] = .array([])
        project.graph["StructureFromMotion_1"] = sfm

        let vm = ProjectViewModel(project: project, projectURL: nil)
        XCTAssertTrue(vm.isConnectionTypeCompatible(
            from: "FeatureExtraction_1", fromAttr: "output",
            to: "StructureFromMotion_1", toAttrPath: "featuresFolders"
        ))
    }

    /// FeatureExtraction.output (file) → FeatureExtraction.maxThreads (int):
    /// incompatible — file → int is rejected.
    func testFileToIntIsRejected() {
        let project = makeTwoNodeProject(
            sourceType: "FeatureExtraction",
            targetType: "FeatureExtraction"
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        XCTAssertFalse(vm.isConnectionTypeCompatible(
            from: "FeatureExtraction_1", fromAttr: "output",
            to: "FeatureExtraction_2", toAttrPath: "maxThreads"
        ))
    }

    /// Commit must short-circuit before mutating state on an incompatible
    /// drop AND populate `shakeTarget` so the canvas can run its animation.
    func testIncompatibleCommitRejectsAndSetsShakeTarget() {
        let project = makeTwoNodeProject(
            sourceType: "FeatureExtraction",
            targetType: "FeatureExtraction"
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        XCTAssertNil(vm.shakeTarget)
        let originalMaxThreads = vm.project.graph["FeatureExtraction_2"]?.inputs?["maxThreads"]
        let ok = vm.commitConnection(
            from: "FeatureExtraction_1", fromAttr: "output",
            to: "FeatureExtraction_2", toAttrPath: "maxThreads",
            undoManager: nil
        )
        XCTAssertFalse(ok)
        XCTAssertFalse(vm.isDirty, "incompatible drop must not dirty the project")
        XCTAssertEqual(
            vm.project.graph["FeatureExtraction_2"]?.inputs?["maxThreads"],
            originalMaxThreads,
            "incompatible drop must not mutate the target attribute"
        )
        XCTAssertNotNil(vm.shakeTarget)
        XCTAssertEqual(vm.shakeTarget?.nodeName, "FeatureExtraction_2")
        XCTAssertEqual(vm.shakeTarget?.attrPath, "maxThreads")
    }

    /// Compatible drops must still commit (the M8 type-check is a *filter*,
    /// not a blanket disable).
    func testCompatibleCommitStillCommits() {
        let project = makeTwoNodeProject(
            sourceType: "CameraInit",
            targetType: "FeatureExtraction"
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let ok = vm.commitConnection(
            from: "CameraInit_1", fromAttr: "output",
            to: "FeatureExtraction_1", toAttrPath: "input",
            undoManager: nil
        )
        XCTAssertTrue(ok)
        XCTAssertTrue(vm.isDirty)
        XCTAssertNil(vm.shakeTarget, "compatible drop should not set shakeTarget")
    }

    /// When the target node has no Spec (or attr isn't in the Spec map) the
    /// connection must still be allowed — rule #5.
    func testUnknownTargetTypeIsAllowed() {
        let source = MGNode(nodeType: "CameraInit", position: MGPosition(x: 0, y: 0))
        // Target node type has no Spec — so its input type is unknown.
        let target = MGNode(
            nodeType: "TotallyUnregistered",
            position: MGPosition(x: 400, y: 0),
            inputs: ["whatever": .null]
        )
        let project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: [
                "CameraInit_1": source,
                "Strange_1": target,
            ])
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        XCTAssertTrue(vm.isConnectionTypeCompatible(
            from: "CameraInit_1", fromAttr: "output",
            to: "Strange_1", toAttrPath: "whatever"
        ))
        let ok = vm.commitConnection(
            from: "CameraInit_1", fromAttr: "output",
            to: "Strange_1", toAttrPath: "whatever",
            undoManager: nil
        )
        XCTAssertTrue(ok)
    }

    // MARK: - Fixture helper

    /// Two-node project with matching positions, distinct names by suffix.
    /// `_1` and `_2` so a same-type pair can talk to itself without
    /// colliding on the graph key.
    private func makeTwoNodeProject(sourceType: String, targetType: String) -> MGProject {
        let source = MGNode(
            nodeType: sourceType,
            position: MGPosition(x: 0, y: 0),
            inputs: ["input": .null]
        )
        let target = MGNode(
            nodeType: targetType,
            position: MGPosition(x: 400, y: 0),
            inputs: [
                "input": .null,
                "maxThreads": .int(0),
                "featuresFolders": .array([]),
            ]
        )
        // Use `_1` and `_2` so same-type pairs are still uniquely keyed.
        let sourceName = sourceType + "_1"
        let targetName = (sourceType == targetType ? targetType + "_2" : targetType + "_1")
        return MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: [
                sourceName: source,
                targetName: target,
            ])
        )
    }
}
