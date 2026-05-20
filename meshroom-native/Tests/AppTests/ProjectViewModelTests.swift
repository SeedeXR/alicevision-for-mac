import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// M3 tests for the view-model that hosts editing state.
///
/// We deliberately *don't* try to drive SwiftUI's UndoManager from XCTest —
/// that requires a real responder chain and a window, both of which are
/// fragile in package-level tests.  Instead we verify the *contract* the
/// view-model offers: dirty flag transitions, save round-trip, and direct
/// invocation of `applyPosition` (which is what the undo block ultimately
/// calls).
@MainActor
final class ProjectViewModelTests: XCTestCase {

    private func makeProject(positionX: Double = 100, positionY: Double = 50) -> MGProject {
        let node = MGNode(
            nodeType: "FeatureExtraction",
            position: MGPosition(x: positionX, y: positionY, integerStorage: true),
            inputs: ["scale": .int(2)]
        )
        return MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: ["FeatureExtraction_1": node])
        )
    }

    private func uniqueTempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("vm-test-\(UUID().uuidString).mg")
    }

    // MARK: - Dirty flag

    func testInitialStateIsClean() {
        let vm = ProjectViewModel(project: makeProject(), projectURL: nil)
        XCTAssertFalse(vm.isDirty)
    }

    /// First mutation must flip the dirty flag.  This is the contract the
    /// canvas relies on to enable the Save button after the first drag.
    func testDirtyFlipsOnFirstCommit() {
        let vm = ProjectViewModel(project: makeProject(), projectURL: nil)
        vm.commitNodePosition(
            name: "FeatureExtraction_1",
            to: MGPosition(x: 200, y: 100),
            undoManager: nil
        )
        XCTAssertTrue(vm.isDirty)
        XCTAssertEqual(vm.project.graph["FeatureExtraction_1"]?.position?.x, 200)
    }

    /// Committing the same coordinates the node already has is a no-op:
    /// must NOT mark the project dirty.  Preserves the "open and don't edit
    /// → stays clean" invariant when the user happens to mouse-down without
    /// actually moving the node.
    func testNoOpCommitDoesNotDirty() {
        let vm = ProjectViewModel(project: makeProject(positionX: 7, positionY: 9), projectURL: nil)
        vm.commitNodePosition(
            name: "FeatureExtraction_1",
            to: MGPosition(x: 7, y: 9),
            undoManager: nil
        )
        XCTAssertFalse(vm.isDirty)
    }

    func testCommitOnUnknownNodeIsNoOp() {
        let vm = ProjectViewModel(project: makeProject(), projectURL: nil)
        vm.commitNodePosition(
            name: "Ghost_99",
            to: MGPosition(x: 1, y: 1),
            undoManager: nil
        )
        XCTAssertFalse(vm.isDirty)
    }

    // MARK: - Save round-trip

    /// Save must (1) write the encoded project to disk, (2) clear the dirty
    /// flag, and (3) leave no surface error.  This is the end-to-end Cmd-S
    /// path minus the SwiftUI shortcut binding.
    func testSaveWritesAndClearsDirty() throws {
        let url = uniqueTempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        // Pre-seed an initial file so the URL is valid for reload.
        try makeProject().writeAtomically(to: url)

        let vm = ProjectViewModel(project: makeProject(), projectURL: url)
        vm.commitNodePosition(
            name: "FeatureExtraction_1",
            to: MGPosition(x: 555, y: 666),
            undoManager: nil
        )
        XCTAssertTrue(vm.isDirty)
        vm.saveProject()
        XCTAssertFalse(vm.isDirty)
        XCTAssertNil(vm.saveError, "save should have succeeded")

        // The on-disk file now reflects the edit.
        let reloaded = try MGProject.load(from: url)
        XCTAssertEqual(reloaded.graph["FeatureExtraction_1"]?.position?.x, 555)
        XCTAssertEqual(reloaded.graph["FeatureExtraction_1"]?.position?.y, 666)
    }

    func testSaveWithoutURLSurfacesError() {
        let vm = ProjectViewModel(project: makeProject(), projectURL: nil)
        vm.saveProject()
        XCTAssertNotNil(vm.saveError, "saving with no URL should populate saveError")
    }

    // MARK: - applyPosition (undo entry point)

    /// `applyPosition` is what the undo closure calls.  Driving it directly
    /// verifies the same code path Cmd-Z would hit, without needing a real
    /// UndoManager (which is hard to test outside an AppKit responder chain).
    func testApplyPositionRestoresPriorCoordinates() {
        let vm = ProjectViewModel(project: makeProject(positionX: 10, positionY: 20), projectURL: nil)
        // Initial commit.
        vm.commitNodePosition(
            name: "FeatureExtraction_1",
            to: MGPosition(x: 500, y: 500),
            undoManager: nil
        )
        // Simulate "undo": apply the previous position back.
        vm.applyPosition(
            MGPosition(x: 10, y: 20),
            for: "FeatureExtraction_1",
            undoManager: nil
        )
        XCTAssertEqual(vm.project.graph["FeatureExtraction_1"]?.position?.x, 10)
        XCTAssertEqual(vm.project.graph["FeatureExtraction_1"]?.position?.y, 20)
    }

    // MARK: - commitInputValue (M4)

    /// Editing any input attribute must flip the dirty flag — the contract
    /// the Save button binds to, identical to the position-commit path.
    func testCommitInputValueMarksDirty() {
        let vm = ProjectViewModel(project: makeProject(), projectURL: nil)
        vm.commitInputValue(
            node: "FeatureExtraction_1",
            attrPath: "scale",
            oldValue: .int(2),
            newValue: .int(4),
            undoManager: nil
        )
        XCTAssertTrue(vm.isDirty)
        XCTAssertEqual(
            vm.project.graph["FeatureExtraction_1"]?.inputs?["scale"],
            .int(4)
        )
    }

    /// Committing the value the model already holds must NOT dirty the
    /// project nor consume an undo slot.  Matches the position-commit "open
    /// and don't edit → stays clean" invariant.
    func testCommitInputValueNoOpDoesNotDirty() {
        let vm = ProjectViewModel(project: makeProject(), projectURL: nil)
        vm.commitInputValue(
            node: "FeatureExtraction_1",
            attrPath: "scale",
            oldValue: .int(2),
            newValue: .int(2),
            undoManager: nil
        )
        XCTAssertFalse(vm.isDirty)
    }

    /// `commitInputValue` must register an undo action with the supplied
    /// `UndoManager`.  We assert by inspecting `canUndo` rather than
    /// driving the undo — that part is exercised by the next test.
    ///
    /// `UndoManager` requires an open group before `registerUndo` will accept
    /// new actions; in a live app the run-loop opens one per event automatically
    /// (`groupsByEvent == true`).  In a unit-test we open one manually because
    /// there is no event-loop tick to flush groups for us.
    func testCommitInputValueRegistersUndo() {
        let vm = ProjectViewModel(project: makeProject(), projectURL: nil)
        let undo = UndoManager()
        undo.groupsByEvent = false
        undo.beginUndoGrouping()
        vm.commitInputValue(
            node: "FeatureExtraction_1",
            attrPath: "scale",
            oldValue: .int(2),
            newValue: .int(4),
            undoManager: undo
        )
        undo.endUndoGrouping()
        XCTAssertTrue(undo.canUndo)
    }

    /// End-to-end undo: register, then drive `applyInputValue` directly (the
    /// same code path the undo block executes).  This avoids needing a real
    /// responder chain to dispatch Cmd-Z, which is brittle in package-level
    /// tests.
    func testUndoRestoresOldInputValue() {
        let vm = ProjectViewModel(project: makeProject(), projectURL: nil)
        vm.commitInputValue(
            node: "FeatureExtraction_1",
            attrPath: "scale",
            oldValue: .int(2),
            newValue: .int(4),
            undoManager: nil
        )
        // Simulate "undo": apply the previous value back.
        vm.applyInputValue(
            node: "FeatureExtraction_1",
            attrPath: "scale",
            value: .int(2),
            undoManager: nil
        )
        XCTAssertEqual(
            vm.project.graph["FeatureExtraction_1"]?.inputs?["scale"],
            .int(2)
        )
    }
}
