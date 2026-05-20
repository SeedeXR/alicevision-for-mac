import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// M6 tests for the graphical connection-editing UI.
///
/// Like the rest of the App test suite we exercise the *data* contract — pin
/// geometry, hit-testing, connection commit, undo, edge selection/delete —
/// rather than driving the SwiftUI gesture pipeline.  That keeps the tests
/// fast and free of AppKit responder-chain dependencies.  The visual
/// behaviour (rubber-band line, hover glow) is best validated by manual
/// smoke testing on a real window.
@MainActor
final class ConnectionEditingTests: XCTestCase {

    // MARK: - Fixture helpers

    /// Two-node project with one possible edge slot: a `CameraInit_1` whose
    /// `output` could connect into `FeatureExtraction_1.input`.  No edges in
    /// the initial state — each test commits the connection itself.
    private func makeTwoNodeProject(
        sourcePos: (Double, Double) = (0, 0),
        targetPos: (Double, Double) = (400, 0)
    ) -> MGProject {
        let source = MGNode(
            nodeType: "CameraInit",
            position: MGPosition(x: sourcePos.0, y: sourcePos.1),
            inputs: ["focal": .double(35.0)],
            outputs: ["output": .string("{nodeCacheFolder}/cameras.sfm")]
        )
        let target = MGNode(
            nodeType: "FeatureExtraction",
            position: MGPosition(x: targetPos.0, y: targetPos.1),
            // Two input slots so the per-slot vertical pin layout has more
            // than one row to exercise — alphabetical order puts "input"
            // first, "scale" second.
            inputs: [
                "input": .null,
                "scale": .int(2),
            ]
        )
        return MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: [
                "CameraInit_1": source,
                "FeatureExtraction_1": target,
            ])
        )
    }

    // MARK: - Pin geometry

    func testInputPinLocalYAdvancesPerSlot() {
        let y0 = PinLayout.inputPinLocalY(slot: 0)
        let y1 = PinLayout.inputPinLocalY(slot: 1)
        let y2 = PinLayout.inputPinLocalY(slot: 2)
        XCTAssertEqual(y1 - y0, NodeMetrics.pinSpacing, accuracy: 0.0001)
        XCTAssertEqual(y2 - y1, NodeMetrics.pinSpacing, accuracy: 0.0001)
    }

    func testOutputPinModelCenterOnRightEdge() {
        let pos = MGPosition(x: 100, y: 50)
        let centre = PinLayout.outputPinModelCenter(nodePosition: pos)
        XCTAssertEqual(centre.x, 100 + NodeMetrics.width, accuracy: 0.0001)
        XCTAssertEqual(centre.y, 50 + NodeMetrics.height / 2, accuracy: 0.0001)
    }

    func testInputPinAttrNamesAreSortedAlphabetically() {
        let node = MGNode(
            nodeType: "X",
            inputs: ["zeta": .int(1), "alpha": .int(2), "mid": .int(3)]
        )
        let names = PinLayout.inputAttrNames(of: node)
        XCTAssertEqual(names, ["alpha", "mid", "zeta"])
    }

    // MARK: - Hit-testing

    func testFindInputPinExactCentreHits() {
        let project = makeTwoNodeProject(sourcePos: (0, 0), targetPos: (400, 100))
        let vm = ProjectViewModel(project: project, projectURL: nil)
        // The first input slot of FeatureExtraction_1 is "input"; its centre
        // sits at (target.x, target.y + inputPinLocalY(slot: 0)).
        let target = project.graph["FeatureExtraction_1"]!
        let centre = PinLayout.inputPinModelCenter(
            node: target,
            nodePosition: target.position!,
            slot: 0
        )
        let hit = vm.findInputPin(at: centre)
        XCTAssertNotNil(hit)
        XCTAssertEqual(hit?.nodeName, "FeatureExtraction_1")
        XCTAssertEqual(hit?.attrPath, "input")
        XCTAssertTrue(hit?.isInput ?? false)
    }

    func testFindInputPinNearbyWithinToleranceHits() {
        let project = makeTwoNodeProject(targetPos: (400, 100))
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let target = project.graph["FeatureExtraction_1"]!
        let centre = PinLayout.inputPinModelCenter(
            node: target,
            nodePosition: target.position!,
            slot: 0
        )
        // Within tolerance — should still hit.
        let nearby = CGPoint(x: centre.x + 3, y: centre.y - 4)
        XCTAssertNotNil(vm.findInputPin(at: nearby))
    }

    func testFindInputPinFarMisses() {
        let project = makeTwoNodeProject()
        let vm = ProjectViewModel(project: project, projectURL: nil)
        // Way outside any pin.
        XCTAssertNil(vm.findInputPin(at: CGPoint(x: -1000, y: -1000)))
    }

    // MARK: - Connection commit

    func testCommitConnectionSetsTemplateString() {
        let project = makeTwoNodeProject()
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let ok = vm.commitConnection(
            from: "CameraInit_1",
            fromAttr: "output",
            to: "FeatureExtraction_1",
            toAttrPath: "input",
            undoManager: nil
        )
        XCTAssertTrue(ok)
        XCTAssertTrue(vm.isDirty)
        let v = vm.project.graph["FeatureExtraction_1"]?.inputs?["input"]
        XCTAssertEqual(v, .string("{CameraInit_1.output}"))
        // Parse-back round-trip — the canvas's edge extractor would see this
        // as an actual edge.
        if case let .string(s) = v {
            let ref = MGTemplateReference.parse(s)
            XCTAssertEqual(ref?.nodeName, "CameraInit_1")
            XCTAssertEqual(ref?.attributePath, ["output"])
        } else {
            XCTFail("expected string value")
        }
    }

    func testCommitConnectionRejectsSelfLoop() {
        let project = makeTwoNodeProject()
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let ok = vm.commitConnection(
            from: "CameraInit_1",
            fromAttr: "output",
            to: "CameraInit_1",
            toAttrPath: "focal",
            undoManager: nil
        )
        XCTAssertFalse(ok)
        XCTAssertFalse(vm.isDirty, "self-loop must not dirty the project")
    }

    func testCommitConnectionNoOpDoesNotDirty() {
        // Pre-wire the connection then attempt to commit the same one.
        var project = makeTwoNodeProject()
        var target = project.graph["FeatureExtraction_1"]!
        target.inputs?["input"] = .string("{CameraInit_1.output}")
        project.graph["FeatureExtraction_1"] = target
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let ok = vm.commitConnection(
            from: "CameraInit_1",
            fromAttr: "output",
            to: "FeatureExtraction_1",
            toAttrPath: "input",
            undoManager: nil
        )
        XCTAssertFalse(ok)
        XCTAssertFalse(vm.isDirty)
    }

    // MARK: - Round-trip through disk

    /// Commit a connection, encode to disk, reload, and check that the
    /// template string survives the JSON round-trip.  Catches encoding bugs
    /// that would silently strip the `{Node.attr}` syntax.
    func testCommitConnectionSurvivesRoundTrip() throws {
        let project = makeTwoNodeProject()
        let vm = ProjectViewModel(project: project, projectURL: nil)
        vm.commitConnection(
            from: "CameraInit_1",
            fromAttr: "output",
            to: "FeatureExtraction_1",
            toAttrPath: "input",
            undoManager: nil
        )
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("conn-rt-\(UUID().uuidString).mg")
        defer { try? FileManager.default.removeItem(at: url) }
        try vm.project.writeAtomically(to: url)
        let reloaded = try MGProject.load(from: url)
        XCTAssertEqual(
            reloaded.graph["FeatureExtraction_1"]?.inputs?["input"],
            .string("{CameraInit_1.output}")
        )
        // GraphLayout should now see the edge.
        let edges = GraphLayout.edges(in: reloaded)
        XCTAssertTrue(edges.contains { e in
            e.sourceNode == "CameraInit_1" &&
            e.targetNode == "FeatureExtraction_1" &&
            e.targetAttribute == "input"
        })
    }

    // MARK: - Undo

    /// Replicates the M4 undo contract for the connection path: after a
    /// commit, calling `applyInputValue` with the captured old value restores
    /// it.  This is the same code path Cmd-Z drives in the live app.
    func testUndoRestoresPriorValueAfterConnect() {
        let project = makeTwoNodeProject()
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let original = vm.project.graph["FeatureExtraction_1"]?.inputs?["input"]
        XCTAssertEqual(original, .null)
        vm.commitConnection(
            from: "CameraInit_1",
            fromAttr: "output",
            to: "FeatureExtraction_1",
            toAttrPath: "input",
            undoManager: nil
        )
        // Simulate undo: re-apply the captured prior value.
        vm.applyInputValue(
            node: "FeatureExtraction_1",
            attrPath: "input",
            value: original ?? .null,
            undoManager: nil
        )
        XCTAssertEqual(
            vm.project.graph["FeatureExtraction_1"]?.inputs?["input"],
            .null
        )
    }

    // MARK: - Edge deletion

    func testDeleteEdgeClearsTemplate() {
        var project = makeTwoNodeProject()
        var target = project.graph["FeatureExtraction_1"]!
        target.inputs?["input"] = .string("{CameraInit_1.output}")
        project.graph["FeatureExtraction_1"] = target
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let edge = GraphEdge(
            sourceNode: "CameraInit_1",
            sourceAttribute: "output",
            targetNode: "FeatureExtraction_1",
            targetAttribute: "input"
        )
        vm.selectedEdge = edge
        vm.deleteEdge(edge, undoManager: nil)
        // After delete the input should be `.null`; the dirty flag should fire.
        XCTAssertEqual(
            vm.project.graph["FeatureExtraction_1"]?.inputs?["input"],
            .null
        )
        XCTAssertTrue(vm.isDirty)
        XCTAssertNil(vm.selectedEdge)
    }

    func testDeleteEdgeDoesNotTouchLiteralInput() {
        // If the user replaced the template with a literal *between* selecting
        // and deleting, deleteEdge must leave the literal intact.
        var project = makeTwoNodeProject()
        var target = project.graph["FeatureExtraction_1"]!
        target.inputs?["input"] = .string("plain-string")
        project.graph["FeatureExtraction_1"] = target
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let edge = GraphEdge(
            sourceNode: "CameraInit_1",
            sourceAttribute: "output",
            targetNode: "FeatureExtraction_1",
            targetAttribute: "input"
        )
        vm.deleteEdge(edge, undoManager: nil)
        XCTAssertEqual(
            vm.project.graph["FeatureExtraction_1"]?.inputs?["input"],
            .string("plain-string")
        )
        XCTAssertFalse(vm.isDirty)
    }

    func testUndoRestoresDeletedEdge() {
        var project = makeTwoNodeProject()
        var target = project.graph["FeatureExtraction_1"]!
        target.inputs?["input"] = .string("{CameraInit_1.output}")
        project.graph["FeatureExtraction_1"] = target
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let priorRef = MGJSONValue.string("{CameraInit_1.output}")
        let edge = GraphEdge(
            sourceNode: "CameraInit_1",
            sourceAttribute: "output",
            targetNode: "FeatureExtraction_1",
            targetAttribute: "input"
        )
        vm.deleteEdge(edge, undoManager: nil)
        vm.applyInputValue(
            node: "FeatureExtraction_1",
            attrPath: "input",
            value: priorRef,
            undoManager: nil
        )
        XCTAssertEqual(
            vm.project.graph["FeatureExtraction_1"]?.inputs?["input"],
            priorRef
        )
    }

    // MARK: - ConnectionDrag state surface

    /// The canvas's drag handler writes a `ConnectionDrag` into the
    /// view-model on every drag tick.  We can't simulate a SwiftUI gesture
    /// from XCTest, but we can confirm the state shape supports both ends
    /// (origin + cursor + hover target).
    func testConnectionDragStateRoundTrips() {
        let drag = ConnectionDrag(
            fromNode: "CameraInit_1",
            fromAttr: "output",
            origin: CGPoint(x: 100, y: 50),
            cursor: CGPoint(x: 200, y: 100),
            hoverTarget: PinHandle(
                nodeName: "FeatureExtraction_1",
                attrPath: "input",
                isInput: true
            )
        )
        XCTAssertEqual(drag.fromNode, "CameraInit_1")
        XCTAssertEqual(drag.hoverTarget?.nodeName, "FeatureExtraction_1")
        XCTAssertEqual(drag.hoverTarget?.attrPath, "input")
    }
}
