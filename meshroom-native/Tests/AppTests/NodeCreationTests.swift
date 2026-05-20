import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// M9 tests for node creation from the palette + undoable graph mutation.
///
/// Two areas of coverage:
/// 1. `MGProject.addNode/nextAvailableNodeName/removeNode` — the model-layer
///    contract.  Pure value-type tests.
/// 2. `ProjectViewModel.addNode(nodeType:at:)` — the view-model glue:
///    dirty flag, undo registration, name uniqueness across rapid adds.
@MainActor
final class NodeCreationTests: XCTestCase {

    // MARK: - MGProject.addNode

    func testAddNodeInsertsWithProvidedProperties() {
        var project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph()
        )
        let ok = project.addNode(
            name: "CameraInit_1",
            nodeType: "CameraInit",
            position: MGPosition(x: 10, y: 20)
        )
        XCTAssertTrue(ok)
        let node = project.graph["CameraInit_1"]
        XCTAssertNotNil(node)
        XCTAssertEqual(node?.nodeType, "CameraInit")
        XCTAssertEqual(node?.position?.x, 10)
        XCTAssertEqual(node?.position?.y, 20)
        XCTAssertNotNil(node?.uid, "M9 nodes get a freshly generated UID")
        XCTAssertEqual(node?.inputs ?? [:], [:], "M9 nodes start with empty inputs")
    }

    func testAddNodeRejectsDuplicateName() {
        var project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: [
                "CameraInit_1": MGNode(nodeType: "CameraInit")
            ])
        )
        let ok = project.addNode(
            name: "CameraInit_1",
            nodeType: "CameraInit",
            position: MGPosition(x: 0, y: 0)
        )
        XCTAssertFalse(ok, "must not overwrite an existing node")
        XCTAssertEqual(project.graph.count, 1)
    }

    func testNextAvailableNodeNameStartsAt1() {
        let project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph()
        )
        XCTAssertEqual(project.nextAvailableNodeName(forType: "CameraInit"), "CameraInit_1")
    }

    func testNextAvailableNodeNameSkipsTaken() {
        let project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: [
                "CameraInit_1": MGNode(nodeType: "CameraInit"),
                "CameraInit_3": MGNode(nodeType: "CameraInit"),
            ])
        )
        // We use "smallest unused suffix above the max", not "smallest hole",
        // so the third CameraInit becomes _4, not _2.  Matches the spec:
        // "smallest unused suffix" interpreted as monotonic.
        XCTAssertEqual(project.nextAvailableNodeName(forType: "CameraInit"), "CameraInit_4")
    }

    func testNextAvailableNodeNameIgnoresUnrelatedTypes() {
        let project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: [
                "FeatureExtraction_7": MGNode(nodeType: "FeatureExtraction"),
            ])
        )
        XCTAssertEqual(project.nextAvailableNodeName(forType: "CameraInit"), "CameraInit_1")
    }

    func testRemoveNodeReturnsFalseForMissing() {
        var project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph()
        )
        XCTAssertFalse(project.removeNode(name: "Ghost"))
    }

    // MARK: - ProjectViewModel.addNode

    func testViewModelAddNodeDirtiesAndCreates() {
        let project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph()
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        XCTAssertFalse(vm.isDirty)
        let name = vm.addNode(
            nodeType: "CameraInit",
            at: MGPosition(x: 50, y: 60),
            undoManager: nil
        )
        XCTAssertEqual(name, "CameraInit_1")
        XCTAssertTrue(vm.isDirty)
        XCTAssertEqual(vm.project.graph["CameraInit_1"]?.position?.x, 50)
        XCTAssertEqual(vm.project.graph["CameraInit_1"]?.position?.y, 60)
    }

    /// Spec requirement: three sequential CameraInit adds produce
    /// CameraInit_1, _2, _3 with no collisions.
    func testThreeSequentialCameraInitAddsGetDistinctNames() {
        let project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph()
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let n1 = vm.addNode(nodeType: "CameraInit", at: MGPosition(x: 0, y: 0), undoManager: nil)
        let n2 = vm.addNode(nodeType: "CameraInit", at: MGPosition(x: 100, y: 100), undoManager: nil)
        let n3 = vm.addNode(nodeType: "CameraInit", at: MGPosition(x: 200, y: 200), undoManager: nil)
        XCTAssertEqual(n1, "CameraInit_1")
        XCTAssertEqual(n2, "CameraInit_2")
        XCTAssertEqual(n3, "CameraInit_3")
        XCTAssertEqual(vm.project.graph.count, 3)
    }

    /// Undo of an add removes the node.  We can't drive a real UndoManager
    /// menu action from XCTest, but we *can* call the apply-remove path
    /// directly to verify the round-trip.
    func testApplyRemoveNodeRemovesIt() {
        let project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph()
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        _ = vm.addNode(nodeType: "CameraInit", at: MGPosition(x: 0, y: 0), undoManager: nil)
        XCTAssertNotNil(vm.project.graph["CameraInit_1"])
        vm.applyRemoveNode(name: "CameraInit_1", undoManager: nil)
        XCTAssertNil(vm.project.graph["CameraInit_1"])
    }

    func testAddNodeRegistersUndoOnUndoManager() {
        let project = MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph()
        )
        let vm = ProjectViewModel(project: project, projectURL: nil)
        let undo = UndoManager()
        undo.groupsByEvent = false
        undo.beginUndoGrouping()
        _ = vm.addNode(nodeType: "CameraInit", at: MGPosition(x: 0, y: 0), undoManager: undo)
        undo.endUndoGrouping()
        XCTAssertTrue(undo.canUndo)
    }

    // MARK: - Palette payload encoding

    func testPalettePayloadRoundTrip() {
        XCTAssertEqual(PalettePayload.decode("avNodeType:CameraInit"), "CameraInit")
        XCTAssertEqual(PalettePayload.decode("avNodeType:StructureFromMotion"), "StructureFromMotion")
    }

    func testPalettePayloadRejectsForeignStrings() {
        XCTAssertNil(PalettePayload.decode("file:///tmp/foo.jpg"))
        XCTAssertNil(PalettePayload.decode("avNodeType:"))
        XCTAssertNil(PalettePayload.decode(""))
        XCTAssertNil(PalettePayload.decode("avNode:CameraInit"))
    }

    /// Every NodeBinary spec entry must appear in the palette list so
    /// users can drag-create every supported node type.
    func testPaletteCoversEveryRegisteredSpec() {
        let palette = Set(NodePalettePanel.allNodeTypes)
        let specs = Set(NodeBinary.specs.keys)
        XCTAssertEqual(palette, specs, "Palette and NodeBinary.specs must agree")
    }

    /// Icons must resolve to non-empty SF Symbol names for every palette
    /// type — catches typos that would render a blank row.
    func testEveryPaletteRowHasAnIcon() {
        for type in NodePalettePanel.allNodeTypes {
            let icon = NodePalettePanel.iconName(forType: type)
            XCTAssertFalse(icon.isEmpty, "missing icon for \(type)")
        }
    }
}
