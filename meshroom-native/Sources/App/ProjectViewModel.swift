import Foundation
import SwiftUI
import ProjectModel

/// Reference-type view-model that owns the mutable `MGProject` shown by the
/// project view.  We use a class (rather than a `@State` value) for three
/// reasons:
///
/// 1. `UndoManager.registerUndo(withTarget:)` retains its target *weakly* and
///    requires an `AnyObject`.  Value types can't sit on that API.
/// 2. The undo closures must mutate "the project the UI is rendering"; if the
///    view-model were a struct, each closure would capture a stale snapshot.
/// 3. SwiftUI's `@ObservedObject` / `@StateObject` integration gives free
///    change notifications when `@Published` properties move — the canvas
///    redraws as soon as a node's position is committed.
@MainActor
final class ProjectViewModel: ObservableObject {
    @Published var project: MGProject
    @Published var projectURL: URL?
    /// `true` once any committed mutation has happened since the last save.
    ///
    /// Importantly we do NOT set this flag during the *preview* of a drag
    /// (the in-flight offset lives in the canvas's `@State`), only on the
    /// drag-end commit.  That preserves M2's invariant: simply opening a `.mg`
    /// and not editing it leaves `isDirty == false`.
    @Published var isDirty: Bool = false
    /// Surface save errors to the UI rather than swallowing them.  Cleared
    /// when the user dismisses the alert.
    @Published var saveError: String?

    // MARK: - M5 execution state
    //
    // The pipeline runner streams events into these fields so the UI can
    // render per-node status without each row having to subscribe to its own
    // Combine pipeline.  Reset to empty on every fresh "Run" invocation.

    /// Status of every node in the most recent (or in-flight) run.  Keyed by
    /// node name.  Absence means "not part of this run" — useful when the
    /// scheduler skips an unsupported node so its row stays neutral.
    ///
    /// For chunked (parallelized) nodes the parent node keeps its own entry
    /// (e.g. `"DepthMap_1": .running`) and each chunk lives under a synthetic
    /// key `"<NodeName>[chunk i/N]"` so the UI can render both rows.
    @Published var nodeStatuses: [String: NodeRunStatus] = [:]
    /// Tail of stdout/stderr for each node, last N lines.  Bounded so a chatty
    /// process can't grow this dictionary unboundedly during a long run.
    @Published var nodeLogTails: [String: [String]] = [:]
    /// Parent-node -> ordered list of chunk child keys.  Populated when
    /// `.nodeStarted` arrives with a `"<Parent>[chunk i/N]"`-shaped name.
    /// Drives the `DisclosureGroup` rows in `PipelinePanel`.
    @Published var nodeChunks: [String: [String]] = [:]
    /// `true` while a pipeline is currently running.  Drives the Run button's
    /// "Stop" alternate state and the toolbar busy indicator.
    @Published var isRunning: Bool = false
    /// Set when a node fails so the canvas can highlight it red.  Cleared on
    /// the next successful run.
    @Published var failedNode: String?
    /// Stderr/log tail of the failed node, surfaced via a sheet so the user
    /// gets actionable diagnostics on the first failure.
    @Published var failedNodeLog: String?

    /// Soft cap on stored log lines per node.  Hold the most recent N so the
    /// inspector's "last line" indicator can still update mid-run without
    /// pegging memory.
    private let maxLogLinesPerNode = 200

    /// The executor that actually drives the binaries.  Owned by the view-model
    /// so a Stop button can call `executor.cancel()` from outside the
    /// streaming `Task`.
    private let executor = GraphExecutor()
    /// Handle on the currently-running task so we can confirm
    /// "no second concurrent run" and let the UI know when it finishes.
    private var runTask: Task<Void, Never>?

    // MARK: - M6 connection-editing state
    //
    // The canvas drives a transient drag-to-connect interaction.  We park the
    // in-flight drag origin + cursor here (instead of inside the canvas's
    // private @State) so the rubber-band line, the source-pin highlight, and
    // the drop-target hover indicator can all observe the same state without
    // a tangle of bindings.

    /// In-flight drag-to-connect.  `nil` when no pin is being dragged.
    @Published var connectionDrag: ConnectionDrag?

    /// Currently selected edge (if any).  Set by clicking on an edge curve;
    /// cleared on background click or when the connection is removed.  The
    /// canvas hosts the Delete-key handler that consumes this.
    @Published var selectedEdge: GraphEdge?

    /// M8: target pin that just rejected a type-incompatible drop.  The
    /// canvas observes this and runs a brief shake animation on the matching
    /// input pin so the user sees *why* the drop didn't commit.  Cleared
    /// after a short delay (the animation block resets it).
    @Published var shakeTarget: PinHandle?

    init(project: MGProject, projectURL: URL?) {
        self.project = project
        self.projectURL = projectURL
    }

    // MARK: - Position editing

    /// Commit a new node position, registering an undo action so Cmd-Z walks
    /// the user back to the previous coordinates.
    ///
    /// Called on drag-end (not during the drag) so the undo stack contains one
    /// entry per gesture, not one per gesture *tick* — which would otherwise
    /// flood the stack with hundreds of micro-moves.
    func commitNodePosition(
        name: String,
        to newPosition: MGPosition,
        undoManager: UndoManager?
    ) {
        let oldPosition = project.graph[name]?.position
        // Guard against committing a no-op move; without this, undoing a
        // selection-tap-drag of zero pixels would still consume an undo slot.
        if let oldPosition,
           oldPosition.x == newPosition.x,
           oldPosition.y == newPosition.y {
            return
        }
        guard project.updateNodePosition(name: name, position: newPosition) else {
            return
        }
        isDirty = true
        registerUndo(name: name, oldPosition: oldPosition, undoManager: undoManager)
    }

    /// Internal entry-point used by undo to apply a position without setting
    /// the dirty flag's *meaning* differently — every successful apply (undo
    /// or redo) marks the project dirty until the user saves again.
    func applyPosition(_ position: MGPosition?, for name: String, undoManager: UndoManager?) {
        let oldPosition = project.graph[name]?.position
        if let position {
            guard project.updateNodePosition(name: name, position: position) else { return }
        } else if var node = project.graph[name] {
            // Restoring to "no position".  Unusual (Meshroom rarely emits a
            // node without one) but possible for synthetic graphs.
            node.position = nil
            project.graph[name] = node
        }
        isDirty = true
        registerUndo(name: name, oldPosition: oldPosition, undoManager: undoManager)
    }

    private func registerUndo(
        name: String,
        oldPosition: MGPosition?,
        undoManager: UndoManager?
    ) {
        guard let undoManager else { return }
        undoManager.registerUndo(withTarget: self) { target in
            // This closure runs on the main actor because the view-model is
            // `@MainActor`-isolated and UndoManager invokes its undo blocks
            // synchronously from the menu action — which is itself on main.
            MainActor.assumeIsolated {
                target.applyPosition(oldPosition, for: name, undoManager: undoManager)
            }
        }
        undoManager.setActionName("Move Node")
    }

    // MARK: - Input attribute editing (M4)

    /// Commit a new value for a node input attribute, addressed by a
    /// dot-separated path.  Mirrors `commitNodePosition` — registers undo,
    /// flips the dirty flag, dispatches the mutation through `MGProject`.
    ///
    /// `oldValue` is supplied by the caller (the inspector already has the
    /// previous value in hand) rather than re-reading from the model.  That
    /// keeps the undo block self-contained even if the underlying model gets
    /// further mutated before the user hits Cmd-Z.
    func commitInputValue(
        node nodeName: String,
        attrPath: String,
        oldValue: MGJSONValue,
        newValue: MGJSONValue,
        undoManager: UndoManager?
    ) {
        // No-op guard: committing the same value the model already holds must
        // not dirty the project nor consume an undo slot.  Matches the
        // position-commit "open-and-don't-edit-stays-clean" invariant.
        if oldValue == newValue { return }
        guard project.updateAttribute(node: nodeName, attrPath: attrPath, newValue: newValue) else {
            return
        }
        isDirty = true
        registerInputValueUndo(
            node: nodeName,
            attrPath: attrPath,
            oldValue: oldValue,
            undoManager: undoManager
        )
    }

    /// Direct-apply variant invoked by the undo block.  Marks dirty (every
    /// apply, undo or redo, is a state change worth saving) and registers a
    /// counter-undo so Cmd-Z and Cmd-Shift-Z walk in both directions.
    func applyInputValue(
        node nodeName: String,
        attrPath: String,
        value: MGJSONValue,
        undoManager: UndoManager?
    ) {
        let current = currentInputValue(node: nodeName, attrPath: attrPath) ?? .null
        guard project.updateAttribute(node: nodeName, attrPath: attrPath, newValue: value) else {
            return
        }
        isDirty = true
        registerInputValueUndo(
            node: nodeName,
            attrPath: attrPath,
            oldValue: current,
            undoManager: undoManager
        )
    }

    private func registerInputValueUndo(
        node nodeName: String,
        attrPath: String,
        oldValue: MGJSONValue,
        undoManager: UndoManager?
    ) {
        guard let undoManager else { return }
        undoManager.registerUndo(withTarget: self) { target in
            MainActor.assumeIsolated {
                target.applyInputValue(
                    node: nodeName,
                    attrPath: attrPath,
                    value: oldValue,
                    undoManager: undoManager
                )
            }
        }
        undoManager.setActionName("Edit Attribute")
    }

    // MARK: - M6 connection editing
    //
    // Connections are stored on-disk as template-reference strings inside an
    // `inputs` value: `"{Source.output}"`.  Creating an edge therefore reduces
    // to "set the target input to a template-string MGJSONValue" — which routes
    // through the same `commitInputValue` path M4 already exercises for
    // undo/redo + dirty-tracking.  The UI code never has to fabricate undo
    // glue for connections, which keeps the surface area small.

    /// Hit-test the cursor location (in model coordinates) against every input
    /// pin in the graph.  Returns the (nodeName, attrPath) of the nearest pin
    /// within `NodeMetrics.pinHitToleranceModel`, or nil for a miss.
    ///
    /// We walk every node and every input attribute rather than maintaining a
    /// spatial index — graphs are small (typical photogrammetry pipeline is
    /// <20 nodes) so a linear sweep is cheaper than the bookkeeping a quadtree
    /// would add.  Ties broken by the alphabetical iteration order, which
    /// matches the visual stacking order.
    func findInputPin(at point: CGPoint) -> PinHandle? {
        let tol = NodeMetrics.pinHitToleranceModel
        var best: (PinHandle, CGFloat)? = nil
        for name in project.graph.sortedNodeNames {
            guard let node = project.graph[name], let pos = node.position else { continue }
            let attrs = PinLayout.inputAttrNames(of: node)
            for (slot, attr) in attrs.enumerated() {
                let centre = PinLayout.inputPinModelCenter(
                    node: node, nodePosition: pos, slot: slot
                )
                let dx = centre.x - point.x
                let dy = centre.y - point.y
                let dist = (dx * dx + dy * dy).squareRoot()
                if dist <= tol, best == nil || dist < best!.1 {
                    best = (
                        PinHandle(nodeName: name, attrPath: attr, isInput: true),
                        dist
                    )
                }
            }
        }
        return best?.0
    }

    /// Commit a graph connection: set `to.toAttrPath` to a template-reference
    /// string `"{from.fromAttr}"`.  Routes through `commitInputValue` so the
    /// existing M4 undo/dirty pipeline does all the heavy lifting.
    ///
    /// Self-loops are silently rejected: M6 explicitly forbids output → input
    /// on the same node since it would never be executable.
    ///
    /// M8 adds *type-checked* rejection: if the source attribute's declared
    /// type is incompatible with the target's, the call returns `false` and
    /// flags `shakeTarget` for the canvas to animate.  When either side's
    /// type is unknown (no Spec, attribute not in `inputTypes`/`outputTypes`)
    /// the check defaults to "allow" — we never block the user on incomplete
    /// type information.
    @discardableResult
    func commitConnection(
        from sourceNode: String,
        fromAttr sourceAttr: String,
        to targetNode: String,
        toAttrPath targetAttrPath: String,
        undoManager: UndoManager?
    ) -> Bool {
        guard sourceNode != targetNode else { return false }
        // M8: short-circuit before any state mutation when types disagree.
        if !isConnectionTypeCompatible(
            from: sourceNode, fromAttr: sourceAttr,
            to: targetNode, toAttrPath: targetAttrPath
        ) {
            shakeTarget = PinHandle(
                nodeName: targetNode,
                attrPath: targetAttrPath,
                isInput: true
            )
            return false
        }
        // Capture the previous value verbatim so undo restores it bit-for-bit.
        // Defaulting to `.null` (rather than just bailing out) lets the user
        // *create* a connection on an input that wasn't present in the on-disk
        // file — Meshroom synthesises defaults at load time, so an unset input
        // is normal.
        let old = currentInputValue(node: targetNode, attrPath: targetAttrPath) ?? .null
        let template = "{\(sourceNode).\(sourceAttr)}"
        let new = MGJSONValue.string(template)
        if old == new { return false }
        commitInputValue(
            node: targetNode,
            attrPath: targetAttrPath,
            oldValue: old,
            newValue: new,
            undoManager: undoManager
        )
        return true
    }

    /// M8 helper: resolve declared types on both ends and ask `PinLayout`
    /// whether the pair compose.  Public-by-default so the canvas can call
    /// it during the drag preview (to red-tint the rubber-band) without
    /// having to duplicate the lookup chain.
    func isConnectionTypeCompatible(
        from sourceNode: String,
        fromAttr sourceAttr: String,
        to targetNode: String,
        toAttrPath targetAttrPath: String
    ) -> Bool {
        guard let src = project.graph[sourceNode],
              let dst = project.graph[targetNode]
        else {
            // Either endpoint missing — let the call proceed and fail at
            // commit time rather than masking the underlying error here.
            return true
        }
        // For nested attribute paths we only know the type of the top-level
        // attr; deeper paths default to "unknown = allow".
        let targetTopAttr = targetAttrPath.split(separator: ".").first.map(String.init)
            ?? targetAttrPath
        let srcType = PinLayout.outputType(of: src, attr: sourceAttr)
        let dstType = PinLayout.inputType(of: dst, attr: targetTopAttr)
        return PinLayout.areCompatible(sourceType: srcType, targetType: dstType)
    }

    /// Delete an edge.  Spec: clear the template ref — replace it with `.null`.
    /// The previous template-string is captured so Cmd-Z restores the edge.
    func deleteEdge(_ edge: GraphEdge, undoManager: UndoManager?) {
        let old = currentInputValue(node: edge.targetNode, attrPath: edge.targetAttribute)
            ?? .null
        // Only nuke if the current value still looks like the edge we're
        // deleting; if the user just retyped the input field, we don't want to
        // squash their new literal.
        if case let .string(s) = old, MGTemplateReference.parse(s) != nil {
            commitInputValue(
                node: edge.targetNode,
                attrPath: edge.targetAttribute,
                oldValue: old,
                newValue: .null,
                undoManager: undoManager
            )
        } else {
            // The attribute already isn't a ref (someone replaced it via the
            // inspector mid-selection).  Drop the selection and bail out.
        }
        if selectedEdge == edge { selectedEdge = nil }
    }

    // Make `currentInputValue` available to the connection helpers in this
    // file — keeps the M4 read path canonical.
    internal func currentInputValue(node nodeName: String, attrPath: String) -> MGJSONValue? {
        guard let inputs = project.graph[nodeName]?.inputs else { return nil }
        let components = attrPath.split(separator: ".").map(String.init)
        guard let first = components.first, let top = inputs[first] else { return nil }
        var value = top
        for comp in components.dropFirst() {
            switch value {
            case .object(let dict):
                guard let v = dict[comp] else { return nil }
                value = v
            case .array(let arr):
                guard let idx = Int(comp), idx >= 0, idx < arr.count else { return nil }
                value = arr[idx]
            default:
                return nil
            }
        }
        return value
    }

    // MARK: - M9 node creation
    //
    // The palette panel drags new nodes onto the canvas.  We route the
    // creation through this method so undo registration + dirty flag flip
    // happen exactly once per add, identical to the position/input-value
    // path.  Returns the chosen name on success so the canvas can select
    // the new node immediately after creation.

    /// Add a new node of `nodeType` at `position`.  Generates a unique name
    /// of the form `<NodeType>_<n>` using the lowest unused suffix in the
    /// current graph (so a third CameraInit becomes `CameraInit_3`).  Undo
    /// removes the node again.
    @discardableResult
    func addNode(
        nodeType: String,
        at position: MGPosition,
        undoManager: UndoManager?
    ) -> String? {
        let name = project.nextAvailableNodeName(forType: nodeType)
        guard project.addNode(name: name, nodeType: nodeType, position: position) else {
            return nil
        }
        isDirty = true
        registerAddNodeUndo(name: name, undoManager: undoManager)
        return name
    }

    /// Apply an add-node mutation directly (the redo path).  Mirrors
    /// `applyPosition` — used by the undo block so Cmd-Shift-Z re-adds a
    /// just-removed node.
    func applyAddNode(
        name: String,
        nodeType: String,
        position: MGPosition,
        uid: String?,
        undoManager: UndoManager?
    ) {
        guard project.addNode(
            name: name, nodeType: nodeType, position: position, uid: uid
        ) else { return }
        isDirty = true
        registerAddNodeUndo(name: name, undoManager: undoManager)
    }

    /// Apply a remove-node mutation directly (the undo path).
    func applyRemoveNode(name: String, undoManager: UndoManager?) {
        guard let node = project.graph[name] else { return }
        // Capture state for the redo before we tear it down.
        let nodeType = node.nodeType
        let position = node.position ?? MGPosition(x: 0, y: 0)
        let uid = node.uid
        guard project.removeNode(name: name) else { return }
        isDirty = true
        guard let undoManager else { return }
        undoManager.registerUndo(withTarget: self) { target in
            MainActor.assumeIsolated {
                target.applyAddNode(
                    name: name,
                    nodeType: nodeType,
                    position: position,
                    uid: uid,
                    undoManager: undoManager
                )
            }
        }
        undoManager.setActionName("Add Node")
    }

    private func registerAddNodeUndo(name: String, undoManager: UndoManager?) {
        guard let undoManager else { return }
        undoManager.registerUndo(withTarget: self) { target in
            MainActor.assumeIsolated {
                target.applyRemoveNode(name: name, undoManager: undoManager)
            }
        }
        undoManager.setActionName("Add Node")
    }

    // MARK: - M5 pipeline execution

    /// Kick off a pipeline run.  Stream events from the executor onto the
    /// `@Published` state so the UI auto-updates.  No-op if a run is already
    /// in flight (the Run button collapses into Stop in that case).
    func runPipeline(config: GraphExecutor.RunConfig) {
        guard !isRunning else { return }
        isRunning = true
        failedNode = nil
        failedNodeLog = nil
        nodeStatuses = [:]
        nodeLogTails = [:]
        nodeChunks = [:]
        // Seed every supported node to `.queued` so the pipeline panel shows
        // the full list up-front rather than nodes blinking in mid-run.
        for name in project.graph.nodes.keys {
            nodeStatuses[name] = .queued
        }
        let stream = executor.run(project, config: config)
        runTask = Task { @MainActor [weak self] in
            for await event in stream {
                self?.handleExecutorEvent(event)
            }
            self?.isRunning = false
        }
    }

    /// Send SIGTERM to the in-flight process and let the run loop unwind.
    func cancelPipeline() {
        executor.cancel()
        // Don't null out runTask — it'll finish on its own when the stream
        // closes.  Setting isRunning here would race with the for-await.
    }

    /// Push one executor event into the `@Published` state.  Pulled out of
    /// the for-await loop so the dispatch table is easy to read and grow.
    private func handleExecutorEvent(_ event: GraphExecutor.Event) {
        switch event {
        case .nodeStarted(let name, _, _):
            nodeStatuses[name] = .running
            // If this is a chunk row (`ParentNode[chunk i/N]`), record it
            // under its parent so the pipeline panel can expand it.
            if let parent = Self.parentName(forChunkKey: name) {
                var chunks = nodeChunks[parent] ?? []
                if !chunks.contains(name) { chunks.append(name) }
                nodeChunks[parent] = chunks
            }
        case .nodeProgress(let name, let line):
            var tail = nodeLogTails[name] ?? []
            tail.append(line)
            if tail.count > maxLogLinesPerNode {
                tail.removeFirst(tail.count - maxLogLinesPerNode)
            }
            nodeLogTails[name] = tail
        case .nodeSkipped(let name, let reason):
            nodeStatuses[name] = .skipped
            var tail = nodeLogTails[name] ?? []
            tail.append("[skipped] \(reason)")
            nodeLogTails[name] = tail
        case .nodeFinished(let name, _):
            nodeStatuses[name] = .succeeded
        case .failed(let name, let reason):
            nodeStatuses[name] = .failed
            // Don't promote a per-chunk failure to the top-level "failed
            // node" sheet — the executor follows it with a parent-level
            // `.failed` carrying the full context.  Surfacing the chunk
            // would have the sheet show the chunk key (`[chunk i/N]`),
            // which doesn't match the canvas node identifier.
            if Self.parentName(forChunkKey: name) == nil {
                failedNode = name
                let tail = nodeLogTails[name] ?? []
                failedNodeLog = (tail + ["", "FAILED: \(reason)"]).joined(separator: "\n")
            }
        case .allDone:
            break  // isRunning is flipped when the for-await loop exits.
        }
    }

    /// Parse a chunk event key (`"<Parent>[chunk i/N]"`) into the parent node
    /// name.  Returns `nil` for plain node names so callers can branch on
    /// "is this a chunk row?".
    static func parentName(forChunkKey key: String) -> String? {
        guard let bracket = key.range(of: "[chunk ") else { return nil }
        // Must end with a `]` and look like `[chunk <num>/<num>]` — we don't
        // need to fully validate, just guard against arbitrary `[` in node
        // names (Meshroom doesn't generate those but the parser should be
        // forgiving rather than misclassify a future identifier).
        guard key.hasSuffix("]") else { return nil }
        return String(key[..<bracket.lowerBound])
    }

    // MARK: - Save

    /// Encode and write the project to its `projectURL`.  No-op if there is
    /// no URL (the project came from an in-memory factory) or nothing has
    /// changed since the last save.
    func saveProject() {
        guard let url = projectURL else {
            saveError = "No file URL associated with this project."
            return
        }
        do {
            try project.writeAtomically(to: url)
            isDirty = false
        } catch {
            saveError = "Failed to save \(url.lastPathComponent): \(error.localizedDescription)"
        }
    }
}

/// Transient state for the M6/M7 drag-to-connect interaction.
///
/// Captured in *model* coordinates (the same space `MGNode.position` and
/// `PinLayout` work in) so the canvas can render the rubber-band line under
/// the same `.scaleEffect` + `.offset` transform that positions everything
/// else.  The canvas converts the screen-space cursor into model space before
/// updating `cursor`.
struct ConnectionDrag: Equatable, Sendable {
    /// Source node (`{fromNode.fromAttr}` once committed).
    var fromNode: String
    /// Source attribute name.  M7 added multi-output pins, so this is now
    /// one of the node's declared outputs (`"output"`, `"outputMesh"`, …)
    /// rather than the hard-coded `"output"` literal.
    var fromAttr: String
    /// Model-coordinate origin of the rubber-band — the source pin centre.
    var origin: CGPoint
    /// Current cursor location in model coordinates.
    var cursor: CGPoint
    /// Resolved drop target (if the cursor is over an input pin).  Drives the
    /// green hover glow at the destination end.
    var hoverTarget: PinHandle?
    /// M8: `true` when the cursor sits over an input pin whose declared type
    /// is incompatible with the source attribute's type.  Drives the red
    /// rubber-band line + red source-pin glow + shake animation on the target.
    /// `false` when the hover target is missing (no drop yet) or compatible.
    var invalid: Bool = false
}

/// Per-node visual status used by the pipeline panel + canvas highlight.
///
/// Kept as a plain enum (not `Identifiable` or `RawRepresentable`) so the
/// dictionary in `ProjectViewModel.nodeStatuses` doesn't have to box every
/// value.  The UI maps each case to a colour + SF Symbol in the row builder.
public enum NodeRunStatus: Hashable, Sendable {
    case queued
    case running
    case succeeded
    case failed
    case skipped
}
