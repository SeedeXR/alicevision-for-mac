import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// M7 tests for multi-output-pin support.
///
/// We verify the *data contract* — `NodeBinary.Spec.outputs`, the per-output
/// geometry helper, and that connection commits route through the chosen
/// output attribute name — rather than driving SwiftUI gestures.  The visual
/// rendering of stacked pins is exercised manually on a live window.
@MainActor
final class MultiOutputPinsTests: XCTestCase {

    // MARK: - Spec coverage

    /// Every node listed in `NodeBinary.specs` must have at least one
    /// declared output.  Defaulting to `["output"]` covers single-output
    /// nodes; multi-output specs (SfM, Meshing, Texturing) must override.
    func testEverySpecHasAtLeastOneOutput() {
        for (type, spec) in NodeBinary.specs {
            XCTAssertFalse(
                spec.outputs.isEmpty,
                "Spec for \(type) declares no outputs — at least `output` is required"
            )
        }
    }

    /// Multi-output nodes from the M7 spec must declare the exact attr
    /// names the upstream Python defs use.  This locks down our extraction
    /// against drift in `meshroom-mac/nodes/aliceVision/*.py`.
    func testMultiOutputSpecsMatchUpstreamPythonDefs() {
        // StructureFromMotion.py:374 — `output`, `outputViewsAndPoses`,
        // `extraInfoFolder`.
        XCTAssertEqual(
            Set(NodeBinary.spec(for: "StructureFromMotion")!.outputs),
            ["output", "outputViewsAndPoses", "extraInfoFolder"]
        )
        // Meshing.py:470 — `outputMesh`, `output`.
        XCTAssertEqual(
            Set(NodeBinary.spec(for: "Meshing")!.outputs),
            ["output", "outputMesh"]
        )
        // Texturing.py:334 — output, outputMesh, outputMaterial, outputTextures.
        XCTAssertEqual(
            Set(NodeBinary.spec(for: "Texturing")!.outputs),
            ["output", "outputMesh", "outputMaterial", "outputTextures"]
        )
    }

    /// Single-output nodes must keep emitting just `["output"]` so M6
    /// fixtures and edges continue to work unchanged.
    func testSingleOutputSpecsRemainSingleOutput() {
        for type in ["CameraInit", "FeatureExtraction", "ImageMatching",
                     "FeatureMatching", "PrepareDenseScene",
                     "DepthMap", "DepthMapFilter", "ImportMiddlebury"] {
            XCTAssertEqual(
                NodeBinary.spec(for: type)?.outputs,
                ["output"],
                "\(type) should be single-output (`output`)"
            )
        }
        XCTAssertEqual(
            NodeBinary.spec(for: "MeshFiltering")?.outputs,
            ["outputMesh"],
            "MeshFiltering's single output is `outputMesh`, not `output`"
        )
    }

    // MARK: - PinLayout multi-output geometry

    func testOutputAttrNamesAreSortedAlphabetically() {
        let node = MGNode(nodeType: "StructureFromMotion")
        let names = PinLayout.outputAttrNames(of: node)
        // `extraInfoFolder` sorts before `output` which sorts before
        // `outputViewsAndPoses`.
        XCTAssertEqual(names, ["extraInfoFolder", "output", "outputViewsAndPoses"])
    }

    /// Unknown node types fall back to a single `output` pin so freshly
    /// created custom-type nodes still get a working drag handle.
    func testOutputAttrNamesUnknownTypeFallsBackToOutput() {
        let node = MGNode(nodeType: "SomeFutureTypeNoSpecFor")
        XCTAssertEqual(PinLayout.outputAttrNames(of: node), ["output"])
    }

    /// The slot-N centre of an output pin sits on the node's right edge at
    /// the canonical per-slot vertical offset (matching the input pin
    /// formula, just mirrored to the right).
    func testOutputPinModelCenterSlotN() {
        let node = MGNode(nodeType: "StructureFromMotion")
        let pos = MGPosition(x: 100, y: 50)
        for slot in 0..<3 {
            let c = PinLayout.outputPinModelCenter(node: node, nodePosition: pos, slot: slot)
            XCTAssertEqual(c.x, 100 + NodeMetrics.width, accuracy: 0.0001)
            XCTAssertEqual(
                c.y,
                50 + PinLayout.outputPinLocalY(slot: slot),
                accuracy: 0.0001
            )
        }
    }

    /// Looking up by attribute name resolves to the same centre as the
    /// slot-N variant.  Catches drift between the index-based and
    /// name-based overloads.
    func testOutputPinModelCenterByAttrMatchesSlotLookup() {
        let node = MGNode(nodeType: "Meshing")
        let pos = MGPosition(x: 0, y: 0)
        let names = PinLayout.outputAttrNames(of: node)
        for (slot, attr) in names.enumerated() {
            let bySlot = PinLayout.outputPinModelCenter(node: node, nodePosition: pos, slot: slot)
            let byAttr = PinLayout.outputPinModelCenter(node: node, nodePosition: pos, attr: attr)
            XCTAssertEqual(byAttr, bySlot)
        }
    }

    func testOutputPinModelCenterByAttrUnknownReturnsNil() {
        let node = MGNode(nodeType: "Meshing")
        let pos = MGPosition(x: 0, y: 0)
        XCTAssertNil(PinLayout.outputPinModelCenter(node: node, nodePosition: pos, attr: "ghost"))
    }

    // MARK: - Connection commit per output

    /// Dragging from the SfM `outputViewsAndPoses` pin must commit a
    /// template string using *that* attribute, not the default `output`.
    func testCommitConnectionUsesChosenSourceAttr() {
        let sfm = MGNode(
            nodeType: "StructureFromMotion",
            position: MGPosition(x: 0, y: 0)
        )
        let downstream = MGNode(
            nodeType: "PrepareDenseScene",
            position: MGPosition(x: 400, y: 0),
            inputs: ["input": .null]
        )
        let project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: [
                "StructureFromMotion_1": sfm,
                "PrepareDenseScene_1": downstream,
            ])
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let ok = vm.commitConnection(
            from: "StructureFromMotion_1",
            fromAttr: "outputViewsAndPoses",
            to: "PrepareDenseScene_1",
            toAttrPath: "input",
            undoManager: nil
        )
        XCTAssertTrue(ok)
        XCTAssertEqual(
            vm.project.graph["PrepareDenseScene_1"]?.inputs?["input"],
            .string("{StructureFromMotion_1.outputViewsAndPoses}")
        )
        // GraphLayout's reader recovers the attr name so edges round-trip.
        let edges = GraphLayout.edges(in: vm.project)
        XCTAssertTrue(edges.contains { e in
            e.sourceNode == "StructureFromMotion_1" &&
            e.sourceAttribute == "outputViewsAndPoses" &&
            e.targetNode == "PrepareDenseScene_1" &&
            e.targetAttribute == "input"
        })
    }
}
