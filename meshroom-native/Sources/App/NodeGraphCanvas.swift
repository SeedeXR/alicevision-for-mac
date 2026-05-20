import SwiftUI
import ProjectModel
import UniformTypeIdentifiers

/// Pan/zoom-able SwiftUI canvas that renders the node graph.
///
/// Implementation choices:
/// - Edges are drawn with the lower-level `Canvas` API.  This is the cheapest
///   way to draw N curves without paying a `View` materialisation cost per
///   edge, which matters for graphs with 30+ nodes.
/// - Nodes are individual `NodeView`s laid out via `.offset(_:)`.  Keeping
///   them as real Views means hit-testing (selection) "just works" through
///   the normal SwiftUI gesture pipeline.
/// - Pan + zoom are *pure* SwiftUI gestures, applied as a manual
///   `CGAffineTransform`-equivalent (`.scaleEffect` + `.offset`).  No
///   `NSScrollView` wrapping — the spec explicitly requires this.
/// - In M3, each node also has its own `DragGesture` that *consumes* drags
///   originating on the node itself.  SwiftUI's gesture-resolution rule is
///   "innermost wins" — attaching the per-node drag to the `NodeView` means
///   the canvas's pan gesture only fires when the user starts the drag on
///   empty background, which is the intuitive behaviour.
struct NodeGraphCanvas: View {
    @ObservedObject var viewModel: ProjectViewModel
    @Binding var selectedNodeName: String?
    @Environment(\.undoManager) private var undoManager

    /// Edges precomputed once per *project value*, refreshed on graph mutation
    /// via `onChange`.  Recomputing on every render would burn time on every
    /// gesture tick; recomputing only when the project actually changes keeps
    /// edge geometry in sync with node moves.
    @State private var edges: [GraphEdge] = []
    @State private var modelBounds: CGRect = CGRect(x: 0, y: 0, width: 600, height: 400)

    @State private var zoom: CGFloat = 1.0
    @State private var pivot: CGSize = .zero // committed pan
    @State private var dragOffset: CGSize = .zero // in-flight pan delta
    @State private var gestureZoom: CGFloat = 1.0 // in-flight magnify delta
    @State private var didFitOnce = false

    /// In-flight per-node drag offset (preview only — not yet committed to the
    /// model).  `nil` when no node is being dragged.  Holding this in a
    /// separate piece of state lets us avoid mutating `MGProject` on every
    /// gesture tick (which would also bloat the undo stack).
    @State private var nodeDrag: (name: String, offset: CGSize)?

    var body: some View {
        GeometryReader { geo in
            let totalOffset = CGSize(
                width: pivot.width + dragOffset.width,
                height: pivot.height + dragOffset.height
            )
            let scale = zoom * gestureZoom

            ZStack {
                // Background that swallows clicks to deselect.
                Color(NSColor.windowBackgroundColor)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        selectedNodeName = nil
                        viewModel.selectedEdge = nil
                    }

                // Edges layer — also used for hit-testing edge clicks (M6).
                // We give it a tap gesture, but keep nodes above it in z-order
                // so a click on a node still routes to the node.
                Canvas { ctx, _ in
                    drawEdges(in: ctx)
                    drawRubberBand(in: ctx)
                }
                .scaleEffect(scale, anchor: .topLeading)
                .offset(totalOffset)
                // Intercept clicks in the edge layer to pick edges.  Returns
                // a model-coord cursor location from the gesture's tap.
                .gesture(
                    SpatialTapGesture(coordinateSpace: .named("nodeGraph"))
                        .onEnded { value in
                            if let hit = hitTestEdge(at: value.location) {
                                viewModel.selectedEdge = hit
                                selectedNodeName = nil
                            } else {
                                viewModel.selectedEdge = nil
                            }
                        }
                )

                // Nodes layer
                ZStack(alignment: .topLeading) {
                    ForEach(viewModel.project.graph.sortedNodeNames, id: \.self) { name in
                        if let node = viewModel.project.graph[name] {
                            nodeWithGesture(name: name, node: node, scale: scale)
                        }
                    }
                }
                .scaleEffect(scale, anchor: .topLeading)
                .offset(totalOffset)

                // Floating zoom HUD (top trailing)
                VStack {
                    HStack {
                        Spacer()
                        zoomHUD
                            .padding(8)
                    }
                    Spacer()
                }
            }
            .contentShape(Rectangle())
            // Named coordinate space so pin drag gestures (which live deep
            // inside NodeView) report locations in this canvas's pre-zoom
            // model space.  Without `.coordinateSpace`, drag events arrive in
            // the pin's local space and we'd have to multiply through every
            // transform to recover model coords.  The `scaleEffect` + `offset`
            // applied to the inner layers means a `.named` space *here* still
            // reflects the model coordinates the layers were positioned in.
            .coordinateSpace(name: "nodeGraph")
            .gesture(panGesture)
            .gesture(magnifyGesture)
            // Delete key removes the currently selected edge.  Attached as a
            // focus-able invisible button so SwiftUI routes the keypress to
            // it; `.keyboardShortcut(.delete)` and the related `forwardDelete`
            // both fire.
            .background(
                Button("Delete edge") {
                    if let e = viewModel.selectedEdge {
                        viewModel.deleteEdge(e, undoManager: undoManager)
                    }
                }
                .keyboardShortcut(.delete, modifiers: [])
                .disabled(viewModel.selectedEdge == nil)
                .hidden()
            )
            .onAppear {
                recomputeGeometry()
                if !didFitOnce {
                    fitToView(in: geo.size)
                    didFitOnce = true
                }
            }
            // Edge geometry must follow node moves.  We can't observe the
            // graph dictionary directly, but the project value compares by
            // content so `onChange` fires whenever any node mutates.
            .onChange(of: viewModel.project) { _, _ in
                recomputeGeometry()
            }
            // M9 drop target.  Accepts `public.plain-text` payloads
            // formatted as `"avNodeType:<Type>"` and turns them into a new
            // node positioned at the drop point in model coordinates.  We
            // use `onDrop(of:isTargeted:perform:)` rather than the newer
            // `.dropDestination` modifier because the older API gives us
            // the unconverted *view-local* drop point, which is exactly
            // what `toModelSpace` already knows how to handle.
            .onDrop(of: [UTType.plainText], isTargeted: nil) { providers, location in
                handlePaletteDrop(providers: providers, viewLocation: location, scale: zoom * gestureZoom)
            }
        }
    }

    /// Process a palette-originated drop.  Decodes the payload off the main
    /// thread (NSItemProvider hands the data over asynchronously), then
    /// commits the new node from the main actor.  Returns `true` to tell
    /// SwiftUI the drop was *accepted* — the actual commit may still bail
    /// if the payload turns out not to be a palette item.
    private func handlePaletteDrop(
        providers: [NSItemProvider],
        viewLocation: CGPoint,
        scale: CGFloat
    ) -> Bool {
        guard let provider = providers.first else { return false }
        // Convert the drop point now (before the async hop) so we capture
        // the gesture's snapshot of pan/zoom.  We snap the model-space
        // point to integer pixels for byte-stable re-encode (`.mg`
        // positions are integer-storage by Meshroom convention).
        let model = toModelSpace(viewLocation, scale: scale)
        let snapped = MGPosition(x: model.x.rounded(), y: model.y.rounded())
        provider.loadObject(ofClass: NSString.self) { item, _ in
            guard let raw = (item as? String),
                  let nodeType = PalettePayload.decode(raw)
            else { return }
            Task { @MainActor in
                viewModel.addNode(
                    nodeType: nodeType,
                    at: snapped,
                    undoManager: undoManager
                )
            }
        }
        return true
    }

    // MARK: - Coordinate translation
    //
    // `pinGestureLocation` arrives in the named "nodeGraph" coord space — but
    // that space *includes* the canvas's own `.scaleEffect` and `.offset`.  We
    // therefore need to undo those before hit-testing against model-space pin
    // centres.  Pulled into a helper because the same conversion is used by
    // both the drag-changed (to update the rubber band) and drag-ended (to
    // pick a drop target) handlers.
    private func toModelSpace(_ p: CGPoint, scale: CGFloat) -> CGPoint {
        let totalOffset = CGSize(
            width: pivot.width + dragOffset.width,
            height: pivot.height + dragOffset.height
        )
        let s = max(scale, 0.0001)
        return CGPoint(
            x: (p.x - totalOffset.width) / s,
            y: (p.y - totalOffset.height) / s
        )
    }

    // MARK: - Per-node rendering + drag gesture

    /// Build the offset+gesture-attached `NodeView` for one node.  Extracted
    /// because the inline ternary for the in-flight drag offset became
    /// unreadable inside the `ForEach`.
    @ViewBuilder
    private func nodeWithGesture(name: String, node: MGNode, scale: CGFloat) -> some View {
        let basePos = node.position ?? MGPosition(x: 0, y: 0)
        // While dragging, *display* the preview offset; commit happens on end.
        let previewDX: CGFloat = (nodeDrag?.name == name) ? nodeDrag!.offset.width : 0
        let previewDY: CGFloat = (nodeDrag?.name == name) ? nodeDrag!.offset.height : 0

        let drag = viewModel.connectionDrag
        let outputDragAttr: String? = (drag?.fromNode == name) ? drag?.fromAttr : nil
        let hoveredAttr: String? = {
            guard let h = drag?.hoverTarget, h.nodeName == name else { return nil }
            return h.attrPath
        }()

        NodeView(
            name: name,
            node: node,
            isSelected: selectedNodeName == name,
            runStatus: viewModel.nodeStatuses[name],
            onOutputDragChanged: { outputAttr, screenPoint in
                handleOutputDragChanged(
                    sourceNode: name,
                    sourceAttr: outputAttr,
                    sourcePos: basePos,
                    sourceNodeValue: node,
                    screenPoint: screenPoint,
                    scale: scale
                )
            },
            onOutputDragEnded: { outputAttr, screenPoint in
                handleOutputDragEnded(
                    sourceNode: name,
                    sourceAttr: outputAttr,
                    screenPoint: screenPoint,
                    scale: scale
                )
            },
            hoveredInputAttr: hoveredAttr,
            outputDragSourceAttr: outputDragAttr,
            dragInvalid: drag?.invalid ?? false
        )
        .frame(width: NodeMetrics.width, height: NodeMetrics.height)
        .offset(
            x: basePos.x + previewDX,
            y: basePos.y + previewDY
        )
        .onTapGesture {
            selectedNodeName = name
            viewModel.selectedEdge = nil
        }
        // Per-node drag.  `minimumDistance: 2` keeps a plain click from being
        // mis-interpreted as a zero-length drag, which otherwise would race
        // with `onTapGesture` and sometimes swallow the selection tap.
        .gesture(
            DragGesture(minimumDistance: 2)
                .onChanged { value in
                    // Divide by scale so the on-screen drag distance maps
                    // 1:1 to model coordinates regardless of zoom level.
                    let s = max(scale, 0.0001)
                    nodeDrag = (name, CGSize(
                        width: value.translation.width / s,
                        height: value.translation.height / s
                    ))
                    // Selecting on drag-start matches every other node editor
                    // (Final Cut, Origami, Houdini) — moving a thing implies
                    // wanting it selected.
                    if selectedNodeName != name {
                        selectedNodeName = name
                    }
                }
                .onEnded { value in
                    let s = max(scale, 0.0001)
                    let dx = value.translation.width / s
                    let dy = value.translation.height / s
                    // Snap to integer pixels.  Meshroom stores positions as
                    // integers and the canvas grid is implicit-integer too,
                    // so rounding here keeps round-tripped files byte-stable.
                    let newPos = MGPosition(
                        x: (basePos.x + dx).rounded(),
                        y: (basePos.y + dy).rounded()
                    )
                    nodeDrag = nil
                    viewModel.commitNodePosition(
                        name: name,
                        to: newPos,
                        undoManager: undoManager
                    )
                }
        )
    }

    // MARK: - M6 connection-drag handlers
    //
    // Both handlers convert the screen-space point delivered by the pin's
    // DragGesture into model space (undo the canvas's scale+pan), then update
    // the view-model's `connectionDrag`.  The drag-ended path additionally
    // hit-tests against every input pin and, on a hit, commits the new edge.

    private func handleOutputDragChanged(
        sourceNode: String,
        sourceAttr: String,
        sourcePos: MGPosition,
        sourceNodeValue: MGNode,
        screenPoint: CGPoint,
        scale: CGFloat
    ) {
        let model = toModelSpace(screenPoint, scale: scale)
        // M7: the rubber-band's origin must follow the *specific* output pin
        // the user grabbed — otherwise a multi-output node (e.g. SfM) draws
        // its in-flight line from the wrong y coordinate.
        let origin = PinLayout.outputPinModelCenter(
            node: sourceNodeValue,
            nodePosition: sourcePos,
            attr: sourceAttr
        ) ?? PinLayout.outputPinModelCenter(nodePosition: sourcePos)
        let hover = viewModel.findInputPin(at: model)
        // Forbid self-loops at *preview* time too — green-glowing a node
        // dropping into itself would imply something illegal is about to
        // happen.  The drop handler also rejects, but blocking here keeps the
        // UI cue honest.
        let safeHover = (hover?.nodeName == sourceNode) ? nil : hover
        // M8: when we have a hover target, pre-check type compatibility so the
        // rubber-band can red-tint *before* the user drops.  No target = no
        // verdict (drag is floating in empty space), so leave `invalid = false`.
        let invalid: Bool = {
            guard let h = safeHover else { return false }
            return !viewModel.isConnectionTypeCompatible(
                from: sourceNode, fromAttr: sourceAttr,
                to: h.nodeName, toAttrPath: h.attrPath
            )
        }()
        viewModel.connectionDrag = ConnectionDrag(
            fromNode: sourceNode,
            fromAttr: sourceAttr,
            origin: origin,
            cursor: model,
            hoverTarget: safeHover,
            invalid: invalid
        )
    }

    private func handleOutputDragEnded(
        sourceNode: String,
        sourceAttr: String,
        screenPoint: CGPoint,
        scale: CGFloat
    ) {
        let model = toModelSpace(screenPoint, scale: scale)
        if let hit = viewModel.findInputPin(at: model), hit.nodeName != sourceNode {
            // commitConnection internally rejects type-incompatible drops via
            // `shakeTarget`; we just call it and let the model decide.
            viewModel.commitConnection(
                from: sourceNode,
                fromAttr: sourceAttr,
                to: hit.nodeName,
                toAttrPath: hit.attrPath,
                undoManager: undoManager
            )
        }
        viewModel.connectionDrag = nil
    }

    // MARK: - Rubber-band line (M6)

    /// Draw the dashed in-flight connection line from the source pin to the
    /// cursor.  Renders inside the Canvas's edge layer so it's correctly
    /// scaled+panned with the rest of the graph.
    private func drawRubberBand(in ctx: GraphicsContext) {
        guard let drag = viewModel.connectionDrag else { return }
        let sx = drag.origin.x
        let sy = drag.origin.y
        let tx = drag.cursor.x
        let ty = drag.cursor.y
        let dx = max(40, abs(tx - sx) * 0.5)
        var path = Path()
        path.move(to: CGPoint(x: sx, y: sy))
        path.addCurve(
            to: CGPoint(x: tx, y: ty),
            control1: CGPoint(x: sx + dx, y: sy),
            control2: CGPoint(x: tx - dx, y: ty)
        )
        // Snap dashed style — clearly distinct from committed edges (solid).
        // Green when over a valid drop target, red on a type-incompatible
        // target (M8), and neutral while floating in empty space.
        let colour: Color
        if drag.invalid {
            colour = .red
        } else if drag.hoverTarget != nil {
            colour = .green
        } else {
            colour = .accentColor
        }
        ctx.stroke(
            path,
            with: .color(colour),
            style: StrokeStyle(lineWidth: 2, lineCap: .round, dash: [6, 4])
        )
    }

    // MARK: - Edge hit-test (M6)

    /// Sample N points along each committed edge's cubic curve and pick the
    /// nearest match within tolerance.  The 24-sample resolution is enough to
    /// keep a few-pixel hit gap from opening up on the curve's tightest
    /// sections at default zoom.  Cost is bounded by `edges.count * samples`,
    /// which for a 30-node graph is ~720 distance checks — negligible.
    private func hitTestEdge(at point: CGPoint) -> GraphEdge? {
        let samples = 24
        let tol: CGFloat = 8
        var best: (GraphEdge, CGFloat)? = nil
        for edge in edges {
            guard let sourceNode = viewModel.project.graph[edge.sourceNode],
                  let source = sourceNode.position,
                  let targetNode = viewModel.project.graph[edge.targetNode],
                  let target = targetNode.position
            else { continue }
            // Mirror `drawEdges` so hit-test geometry matches the rendered
            // curve — otherwise a click on the visible edge could miss
            // because the sampled bezier was computed at a different y.
            let sourceOutputs = PinLayout.outputAttrNames(of: sourceNode)
            let sourceSlot = sourceOutputs.firstIndex(of: edge.sourceAttribute) ?? -1
            let targetInputs = PinLayout.inputAttrNames(of: targetNode)
            let targetSlot = targetInputs.firstIndex(of: edge.targetAttribute) ?? -1
            let sx = source.x + NodeMetrics.width
            let sy: CGFloat = sourceSlot >= 0
                ? source.y + PinLayout.outputPinLocalY(slot: sourceSlot)
                : source.y + NodeMetrics.height / 2
            let tx = target.x
            let ty: CGFloat = targetSlot >= 0
                ? target.y + PinLayout.inputPinLocalY(slot: targetSlot)
                : target.y + NodeMetrics.height / 2
            let cdx = max(40, abs(tx - sx) * 0.5)
            let c1 = CGPoint(x: sx + cdx, y: sy)
            let c2 = CGPoint(x: tx - cdx, y: ty)
            for i in 0...samples {
                let t = CGFloat(i) / CGFloat(samples)
                let pt = cubicBezier(
                    t: t,
                    p0: CGPoint(x: sx, y: sy),
                    p1: c1,
                    p2: c2,
                    p3: CGPoint(x: tx, y: ty)
                )
                let dx = pt.x - point.x
                let dy = pt.y - point.y
                let d = (dx * dx + dy * dy).squareRoot()
                if d <= tol, best == nil || d < best!.1 {
                    best = (edge, d)
                }
            }
        }
        return best?.0
    }

    /// Cubic Bézier evaluation at parameter `t` ∈ [0,1].  Inlined helper so
    /// the hit-tester doesn't pull in CoreGraphics' path-flattening API,
    /// which is heavier than we need for a 24-sample sweep.
    private func cubicBezier(
        t: CGFloat, p0: CGPoint, p1: CGPoint, p2: CGPoint, p3: CGPoint
    ) -> CGPoint {
        let u = 1 - t
        let x = u*u*u*p0.x + 3*u*u*t*p1.x + 3*u*t*t*p2.x + t*t*t*p3.x
        let y = u*u*u*p0.y + 3*u*u*t*p1.y + 3*u*t*t*p2.y + t*t*t*p3.y
        return CGPoint(x: x, y: y)
    }

    // MARK: - Edge rendering

    /// Render every edge as a horizontal cubic bezier from source-node right
    /// edge to target-node left edge.  Cubic curves give the familiar
    /// node-editor look and avoid the visual noise of straight diagonal lines
    /// when nodes overlap vertically.
    private func drawEdges(in ctx: GraphicsContext) {
        for edge in edges {
            guard let sourceNode = viewModel.project.graph[edge.sourceNode],
                  let source = sourceNode.position,
                  let targetNode = viewModel.project.graph[edge.targetNode],
                  let target = targetNode.position
            else { continue }

            // If either endpoint is being dragged, draw the edge to its
            // preview position so the curve tracks the node in real time.
            let sourceDX = (nodeDrag?.name == edge.sourceNode) ? (nodeDrag?.offset.width ?? 0) : 0
            let sourceDY = (nodeDrag?.name == edge.sourceNode) ? (nodeDrag?.offset.height ?? 0) : 0
            let targetDX = (nodeDrag?.name == edge.targetNode) ? (nodeDrag?.offset.width ?? 0) : 0
            let targetDY = (nodeDrag?.name == edge.targetNode) ? (nodeDrag?.offset.height ?? 0) : 0

            // M7: anchor the source end to the specific output pin (multi-
            // output nodes stack their pins down the right edge).  Falls
            // back to the legacy centre-of-right-edge when the attr isn't in
            // the Spec.outputs (e.g. a stale edge to a removed attribute).
            let sourceOutputs = PinLayout.outputAttrNames(of: sourceNode)
            let sourceSlot = sourceOutputs.firstIndex(of: edge.sourceAttribute) ?? -1
            let sx = source.x + sourceDX + NodeMetrics.width
            let sy: CGFloat
            if sourceSlot >= 0 {
                sy = source.y + sourceDY + PinLayout.outputPinLocalY(slot: sourceSlot)
            } else {
                sy = source.y + sourceDY + NodeMetrics.height / 2
            }
            // Likewise anchor the target end to the specific input pin slot.
            let targetInputs = PinLayout.inputAttrNames(of: targetNode)
            let targetSlot = targetInputs.firstIndex(of: edge.targetAttribute) ?? -1
            let tx = target.x + targetDX
            let ty: CGFloat
            if targetSlot >= 0 {
                ty = target.y + targetDY + PinLayout.inputPinLocalY(slot: targetSlot)
            } else {
                ty = target.y + targetDY + NodeMetrics.height / 2
            }

            // Control point horizontal offset — half the gap, clamped to a
            // sensible minimum so curves stay smooth even when nodes are
            // close together horizontally.
            let dx = max(40, abs(tx - sx) * 0.5)

            var path = Path()
            path.move(to: CGPoint(x: sx, y: sy))
            path.addCurve(
                to: CGPoint(x: tx, y: ty),
                control1: CGPoint(x: sx + dx, y: sy),
                control2: CGPoint(x: tx - dx, y: ty)
            )
            let isSelected = viewModel.selectedEdge == edge
            ctx.stroke(
                path,
                with: .color(isSelected ? Color.accentColor : .secondary),
                lineWidth: isSelected ? 3.0 : 1.5
            )
        }
    }

    private func recomputeGeometry() {
        self.edges = GraphLayout.edges(in: viewModel.project)
        self.modelBounds = GraphLayout.bounds(of: viewModel.project)
            ?? CGRect(x: 0, y: 0, width: 600, height: 400)
    }

    // MARK: - Gestures

    private var panGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                dragOffset = value.translation
            }
            .onEnded { value in
                pivot.width += value.translation.width
                pivot.height += value.translation.height
                dragOffset = .zero
            }
    }

    private var magnifyGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                gestureZoom = value
            }
            .onEnded { value in
                zoom = max(0.1, min(4.0, zoom * value))
                gestureZoom = 1.0
            }
    }

    // MARK: - Fit-to-view

    /// Pick an initial zoom + pan that centres the graph bounds in the view.
    /// Called once on first appearance.  We use a 10% margin so nodes don't
    /// touch the window edges.
    private func fitToView(in viewSize: CGSize) {
        guard viewSize.width > 0, viewSize.height > 0 else { return }
        let margin: CGFloat = 0.9
        let sx = (viewSize.width * margin) / max(modelBounds.width, 1)
        let sy = (viewSize.height * margin) / max(modelBounds.height, 1)
        let s = max(0.1, min(2.0, min(sx, sy)))
        zoom = s
        // Centre: place the bounds midpoint at the view midpoint.
        let centreX = (modelBounds.minX + modelBounds.width / 2) * s
        let centreY = (modelBounds.minY + modelBounds.height / 2) * s
        pivot = CGSize(
            width: viewSize.width / 2 - centreX,
            height: viewSize.height / 2 - centreY
        )
    }

    // MARK: - Zoom HUD

    private var zoomHUD: some View {
        HStack(spacing: 8) {
            Button {
                zoom = max(0.1, zoom / 1.25)
            } label: {
                Image(systemName: "minus.magnifyingglass")
            }
            Text("\(Int(zoom * 100))%")
                .font(.system(.caption, design: .monospaced))
                .frame(minWidth: 44)
            Button {
                zoom = min(4.0, zoom * 1.25)
            } label: {
                Image(systemName: "plus.magnifyingglass")
            }
            Button("Fit") {
                // Reset and re-fit on the next layout pass.
                pivot = .zero
                zoom = 1.0
                didFitOnce = false
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
