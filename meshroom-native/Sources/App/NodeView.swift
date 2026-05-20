import SwiftUI
import ProjectModel

/// Single node card.  Renders the node's display name (Meshroom's internal
/// `label` if present, else the graph key) on top and the node type underneath.
///
/// M6 adds pin overlays: small circular dots on the left edge (one per input
/// attribute, alphabetically) and a single output pin on the right edge.  The
/// pins are children of the node's frame, so they translate, rotate, and zoom
/// in lock-step with the body — no per-pin coordinate math in the canvas.
struct NodeView: View {
    let name: String
    let node: MGNode
    let isSelected: Bool
    /// M5: the most recent execution status, used to tint the border so the
    /// user sees failed/running nodes at a glance.  `nil` for nodes that have
    /// never been part of a run.
    var runStatus: NodeRunStatus? = nil
    /// M6/M7: closures supplied by the canvas to start/update/end a drag from
    /// one of the node's output pins.  The first argument identifies which
    /// output attribute was grabbed (M7 multi-output nodes — e.g.
    /// StructureFromMotion's `output` / `outputViewsAndPoses` / `extraInfoFolder`
    /// each get their own pin).  Pulled out as closures (rather than a direct
    /// reference to `ProjectViewModel`) so this view stays trivially
    /// previewable without a live view-model.  Both are `nil` for tests that
    /// don't exercise connection editing.
    var onOutputDragChanged: ((_ outputAttr: String, _ point: CGPoint) -> Void)? = nil
    var onOutputDragEnded: ((_ outputAttr: String, _ point: CGPoint) -> Void)? = nil
    /// M6: highlight state forwarded from the view-model.  `hoveredInputAttr`
    /// being non-nil tints the matching input pin green; `outputDragSourceAttr`
    /// (M7) names the output pin currently being dragged from so multi-output
    /// nodes only glow the one the user grabbed.
    var hoveredInputAttr: String? = nil
    var outputDragSourceAttr: String? = nil
    /// M8: when an in-flight drag is over an incompatible drop target the
    /// canvas marks `dragInvalid = true` so the source-pin glow turns red
    /// instead of green and the rubber-band line draws in red.  A `nil`
    /// `outputDragSourceAttr` makes this field irrelevant.
    var dragInvalid: Bool = false

    var body: some View {
        ZStack(alignment: .topLeading) {
            VStack(alignment: .leading, spacing: 2) {
                Text(displayName)
                    .font(.system(.callout, design: .default).weight(.semibold))
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text(node.nodeType)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .padding(8)
            .frame(width: NodeMetrics.width, height: NodeMetrics.height, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: NodeMetrics.cornerRadius)
                    .fill(nodeFill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: NodeMetrics.cornerRadius)
                    .strokeBorder(borderColor, lineWidth: borderWidth)
            )
            .contentShape(RoundedRectangle(cornerRadius: NodeMetrics.cornerRadius))
            .help("\(name) (\(node.nodeType))")

            pinOverlay
        }
        // Frame the ZStack to the node size so pin gestures (sticking out on
        // the right edge by half a pin) still resolve hit-tests cleanly.
        .frame(width: NodeMetrics.width, height: NodeMetrics.height, alignment: .topLeading)
    }

    // MARK: - Pin overlay (M6)

    /// Pins on the left edge (inputs, one per input attr) and the right edge
    /// (single output pin).  Positioned via `.offset(...)` from the node's
    /// top-left so they line up with `PinLayout.inputPinModelCenter` (which is
    /// what the hit-tester uses) — keep the two formulas in sync.  Each pin's
    /// SwiftUI frame is `pinHitDiameter` square (a transparent hit target);
    /// the visible circle is centred inside via ZStack, so we offset by half
    /// the hit-diameter to put the centre at the canonical pin location.
    @ViewBuilder
    private var pinOverlay: some View {
        let inputs = PinLayout.inputAttrNames(of: node)
        let outputs = PinLayout.outputAttrNames(of: node)
        let half = NodeMetrics.pinHitDiameter / 2
        ForEach(Array(inputs.enumerated()), id: \.offset) { slot, attr in
            let localY = PinLayout.inputPinLocalY(slot: slot)
            inputPin(attr: attr)
                .offset(
                    x: -half,
                    y: localY - half
                )
        }
        // M7: one output pin per declared output attribute, alphabetically
        // ordered down the right edge.  Single-output nodes still render
        // exactly one pin (the legacy layout); multi-output nodes (SfM,
        // Meshing, Texturing) get a stack matching their Spec.outputs.
        ForEach(Array(outputs.enumerated()), id: \.offset) { slot, attr in
            let localY = PinLayout.outputPinLocalY(slot: slot)
            outputPin(attr: attr)
                .offset(
                    x: NodeMetrics.width - half,
                    y: localY - half
                )
        }
    }

    /// Single input pin.  We do *not* attach a drag gesture here — the spec
    /// says "drag from output to input"; the drop target is hit-tested from
    /// the canvas's drag-changed handler rather than being a per-pin gesture.
    /// Keeping inputs gesture-less also frees them from racing with the
    /// per-node body drag for the same hit area.
    private func inputPin(attr: String) -> some View {
        ZStack {
            Color.clear
                .frame(
                    width: NodeMetrics.pinHitDiameter,
                    height: NodeMetrics.pinHitDiameter
                )
            Circle()
                .fill(pinFillColor(hovered: hoveredInputAttr == attr))
                .overlay(
                    Circle()
                        .strokeBorder(Color.primary.opacity(0.35), lineWidth: 0.5)
                )
                .frame(width: NodeMetrics.pinDiameter, height: NodeMetrics.pinDiameter)
        }
        // No gesture and no contentShape: input pins are hit-tested from the
        // canvas during drag-end via `findInputPin(at:)`, not via SwiftUI's
        // local gesture machinery.  `.allowsHitTesting(false)` keeps the
        // transparent frame from swallowing taps that should reach the node
        // body underneath.
        .allowsHitTesting(false)
        .help("input: \(attr)")
        .accessibilityLabel("Input pin \(attr) on \(name)")
    }

    /// Single output pin for `attr`.  Hosts the drag-to-connect gesture; the
    /// `minimumDistance: 0` value lets a tap-and-drag start immediately
    /// (without it, SwiftUI swallows the first few pixels and the rubber-band
    /// only appears after the cursor has moved 10pt).  The gesture is
    /// attached here rather than to the node body so SwiftUI's
    /// innermost-wins rule routes pin drags away from the move-node drag.
    ///
    /// The pin's visible circle is `pinDiameter` wide; the surrounding ZStack
    /// (sized `pinHitDiameter`) supplies a larger transparent hit target so
    /// the user doesn't have to land pixel-precise on a 12-pt dot.
    private func outputPin(attr: String) -> some View {
        ZStack {
            Color.clear
                .frame(
                    width: NodeMetrics.pinHitDiameter,
                    height: NodeMetrics.pinHitDiameter
                )
            Circle()
                .fill(outputPinFill(attr: attr))
                .overlay(
                    Circle()
                        .strokeBorder(Color.primary.opacity(0.35), lineWidth: 0.5)
                )
                .frame(width: NodeMetrics.pinDiameter, height: NodeMetrics.pinDiameter)
        }
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0, coordinateSpace: .named("nodeGraph"))
                .onChanged { value in
                    onOutputDragChanged?(attr, value.location)
                }
                .onEnded { value in
                    onOutputDragEnded?(attr, value.location)
                }
        )
        .help("output: \(attr)")
        .accessibilityLabel("Output pin \(attr) on \(name)")
    }

    /// Resolve the fill colour of a pin.  Hover/active pins glow green so the
    /// user sees a valid drop target.  Neutral pins use a muted accent.
    private func pinFillColor(hovered: Bool) -> Color {
        if hovered { return Color.green }
        return Color.accentColor.opacity(0.75)
    }

    /// Output-pin fill colour.  Active drag source glows green for valid
    /// drop targets and red when M8 reports the in-flight drag would be
    /// rejected.  Idle pins use the muted accent.
    private func outputPinFill(attr: String) -> Color {
        if outputDragSourceAttr == attr {
            return dragInvalid ? Color.red : Color.green
        }
        return Color.accentColor.opacity(0.75)
    }

    /// Border colour priority: execution status > selection > default.  This
    /// keeps a failed (red) node visually distinct even when it's the user's
    /// current selection.
    private var borderColor: Color {
        switch runStatus {
        case .failed: return .red
        case .running: return .blue
        case .succeeded: return .green
        case .skipped: return .orange
        case .queued, .none:
            return isSelected ? Color.accentColor : Color.gray.opacity(0.6)
        }
    }

    private var borderWidth: CGFloat {
        if runStatus == .failed || runStatus == .running { return 2.5 }
        return isSelected ? 2 : 1
    }

    /// Prefer the user-visible label Meshroom stores under
    /// `internalInputs.label` if present; otherwise fall back to the unique
    /// graph key.  Mirrors Meshroom UI behaviour.
    private var displayName: String {
        if case let .string(s)? = node.internalInputs?["label"], !s.isEmpty {
            return s
        }
        return name
    }

    /// Background fill.  Uses `internalInputs.color` if Meshroom stored one
    /// (a `#RRGGBB` hex string), else a neutral tint.  The Meshroom UI uses
    /// this to colour-code node families in the user's taste — we honour it.
    private var nodeFill: Color {
        if case let .string(hex)? = node.internalInputs?["color"],
           let c = Self.parseHexColor(hex) {
            return c.opacity(0.18)
        }
        return Color(NSColor.controlBackgroundColor)
    }

    /// Parse a `#RRGGBB` (with optional alpha `#RRGGBBAA`) hex string.  Returns
    /// nil for malformed input rather than throwing — colour parsing is
    /// best-effort and a broken value should just fall back to the default.
    static func parseHexColor(_ s: String) -> Color? {
        var str = s
        if str.hasPrefix("#") { str.removeFirst() }
        guard str.count == 6 || str.count == 8 else { return nil }
        var value: UInt64 = 0
        guard Scanner(string: str).scanHexInt64(&value) else { return nil }
        let r, g, b, a: Double
        if str.count == 6 {
            r = Double((value >> 16) & 0xFF) / 255
            g = Double((value >> 8) & 0xFF) / 255
            b = Double(value & 0xFF) / 255
            a = 1.0
        } else {
            r = Double((value >> 24) & 0xFF) / 255
            g = Double((value >> 16) & 0xFF) / 255
            b = Double((value >> 8) & 0xFF) / 255
            a = Double(value & 0xFF) / 255
        }
        return Color(red: r, green: g, blue: b, opacity: a)
    }
}
