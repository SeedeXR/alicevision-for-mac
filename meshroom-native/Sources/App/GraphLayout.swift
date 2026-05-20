import Foundation
import ProjectModel

/// A directed edge in the node graph: `source.attr -> target.attr`.
///
/// Meshroom stores edges implicitly: any input attribute whose value is a
/// template reference (`"{Node_1.output}"`) is an incoming edge.  Edge extraction
/// is therefore "walk every value in every node's `inputs` and resolve template
/// references to source-node names".
///
/// We deliberately keep this type in the App target (not ProjectModel) because
/// the model layer must remain agnostic about how the graph is *interpreted*.
/// A future Phase M3 may add ProjectModel-level edge APIs, but for the M2
/// viewer this lightweight read-only derivation is enough.
public struct GraphEdge: Hashable, Identifiable, Sendable {
    public var sourceNode: String
    public var sourceAttribute: String
    public var targetNode: String
    public var targetAttribute: String

    public var id: String {
        "\(sourceNode).\(sourceAttribute)->\(targetNode).\(targetAttribute)"
    }
}

public enum GraphLayout {
    /// Recursively walk an `MGJSONValue` and yield every contained
    /// `MGTemplateReference`.  Required because Meshroom permits references to
    /// appear *inside* nested objects and arrays: `featuresFolders` is a list
    /// of references, `advanced` is a dict whose leaves may be references, etc.
    public static func references(in value: MGJSONValue) -> [MGTemplateReference] {
        var out: [MGTemplateReference] = []
        walk(value, into: &out)
        return out
    }

    private static func walk(_ value: MGJSONValue, into out: inout [MGTemplateReference]) {
        switch value {
        case .string:
            if let ref = value.asTemplateReference {
                out.append(ref)
            }
        case .array(let arr):
            for v in arr { walk(v, into: &out) }
        case .object(let obj):
            for (_, v) in obj { walk(v, into: &out) }
        case .null, .bool, .int, .double:
            break
        }
    }

    /// All edges in the graph.  Edges whose source node does not exist (e.g.
    /// dangling references in a corrupt `.mg`) are *dropped* — the viewer is
    /// read-only and silently ignoring them matches Meshroom's behaviour for
    /// unresolved links.
    public static func edges(in project: MGProject) -> [GraphEdge] {
        let nodeSet = Set(project.graph.nodes.keys)
        var out: [GraphEdge] = []
        // Stable iteration order helps test assertions.
        for targetName in project.graph.sortedNodeNames {
            guard let node = project.graph[targetName] else { continue }
            guard let inputs = node.inputs else { continue }
            // Sorted input keys for stability; without this the edges array
            // order would depend on Dictionary hashing.
            for attrName in inputs.keys.sorted() {
                guard let value = inputs[attrName] else { continue }
                for ref in references(in: value) {
                    guard nodeSet.contains(ref.nodeName) else { continue }
                    out.append(GraphEdge(
                        sourceNode: ref.nodeName,
                        sourceAttribute: ref.attributePath.first ?? "",
                        targetNode: targetName,
                        targetAttribute: attrName
                    ))
                }
            }
        }
        return out
    }

    // MARK: - Geometry

    /// Bounding box of all node positions in canvas (model) coordinates.
    /// Returns `nil` if the graph has no positioned nodes.
    public static func bounds(of project: MGProject) -> CGRect? {
        var minX = Double.infinity
        var minY = Double.infinity
        var maxX = -Double.infinity
        var maxY = -Double.infinity
        var found = false
        for (_, node) in project.graph.nodes {
            guard let pos = node.position else { continue }
            found = true
            minX = min(minX, pos.x)
            minY = min(minY, pos.y)
            maxX = max(maxX, pos.x + NodeMetrics.width)
            maxY = max(maxY, pos.y + NodeMetrics.height)
        }
        guard found else { return nil }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }
}

/// Visual sizing constants used by both layout math (bounds) and the canvas
/// drawing code.  Centralised so a future M3 zoom-to-fit feature can use the
/// exact same dimensions the renderer uses.
public enum NodeMetrics {
    public static let width: CGFloat = 160
    public static let height: CGFloat = 60
    public static let cornerRadius: CGFloat = 8

    // MARK: - M6 pin metrics
    //
    // Pins are the small circular dots on the node's vertical edges used as
    // drag-to-connect handles.  Diameter is the rendered visual size; the hit
    // target is enlarged by the renderer's `.contentShape` so users don't have
    // to be pixel-perfect.
    public static let pinDiameter: CGFloat = 12
    public static let pinHitDiameter: CGFloat = 20
    /// Tolerance used by `ProjectViewModel.findInputPin(at:)` when matching a
    /// cursor location to an input pin centre, in model (pre-zoom) coordinates.
    public static let pinHitToleranceModel: CGFloat = 12
    /// Vertical reservation below the node header where the first input pin
    /// lives, in model coordinates.  Leaves the title row alone.
    public static let pinTopInset: CGFloat = 24
    /// Vertical spacing between consecutive input pins.
    public static let pinSpacing: CGFloat = 12
}

/// Identifier for a single pin (input or output) on a node.  Pulled out so the
/// canvas, the hit-tester, and the rubber-band renderer can all refer to the
/// same opaque shape.
public struct PinHandle: Hashable, Sendable {
    public let nodeName: String
    public let attrPath: String
    public let isInput: Bool
    public init(nodeName: String, attrPath: String, isInput: Bool) {
        self.nodeName = nodeName
        self.attrPath = attrPath
        self.isInput = isInput
    }
}

/// Pin geometry helpers.  Pin positions are derived in *model* coordinates
/// (the same space `MGNode.position` lives in) so a single computation works
/// for both rendering (NodeView overlays at node-local offsets) and
/// hit-testing (ProjectViewModel walks all nodes in model space).
///
/// The visible pin layout:
/// - One input pin per *top-level* input attribute, in alphabetical order,
///   stacked down the left edge.
/// - M7: One output pin per *declared output attribute* of the node's
///   `NodeBinary.Spec.outputs`, alphabetically sorted, stacked down the
///   right edge.  Nodes without a Spec (e.g. custom user types added via
///   the M9 palette before a spec is wired up) fall back to a single
///   `"output"` pin so connection editing keeps working.
public enum PinLayout {
    /// Sorted list of top-level input attribute names for a node.  Returned
    /// independently so renderers can iterate in the same order the hit-tester
    /// expects (otherwise a pin shown at slot N could resolve to attribute M).
    public static func inputAttrNames(of node: MGNode) -> [String] {
        (node.inputs ?? [:]).keys.sorted()
    }

    /// Per-input-pin Y offset *within the node's frame* (local coordinates).
    ///
    /// Returned y position is the visual centre, in node-local space.  This is
    /// the single source of truth for both the renderer (`NodeView` places the
    /// pin overlay at this y) and the hit-tester (`ProjectViewModel`
    /// converts to model space by adding `nodePosition.y`).  Without sharing
    /// the same function, a "pin at slot 5" could render at one y but be
    /// hit-tested at a different y.
    public static func inputPinLocalY(slot: Int) -> CGFloat {
        NodeMetrics.pinTopInset + CGFloat(slot) * NodeMetrics.pinSpacing
    }

    /// Per-input-pin centre in *model* coordinates.
    public static func inputPinModelCenter(
        node: MGNode,
        nodePosition: MGPosition,
        slot: Int
    ) -> CGPoint {
        // Left edge of the node, x = node.x; y = node.y + per-slot inset.
        // The renderer uses the exact same `inputPinLocalY` so no clamping is
        // applied here either — if a node has more inputs than fit in its
        // frame, the overflow pins visibly extend below the body, but the
        // hit-tester and the renderer agree on every centre.
        _ = node
        return CGPoint(
            x: nodePosition.x,
            y: nodePosition.y + inputPinLocalY(slot: slot)
        )
    }

    // MARK: - M7 multi-output pins

    /// Alphabetically-sorted list of output attribute names a node exposes.
    /// Pulled from `NodeBinary.spec(for:).outputs`; falls back to a single
    /// `"output"` for unknown node types so newly-created or custom nodes
    /// still get a working drag-from-output handle.
    public static func outputAttrNames(of node: MGNode) -> [String] {
        if let spec = NodeBinary.spec(for: node.nodeType) {
            return spec.outputs.sorted()
        }
        return [outputAttrName]
    }

    /// Per-output-pin Y offset *within the node's frame* (local coordinates).
    /// Mirrors `inputPinLocalY` so multi-output nodes stack their pins down
    /// the right edge with the same spacing as inputs on the left.
    public static func outputPinLocalY(slot: Int) -> CGFloat {
        NodeMetrics.pinTopInset + CGFloat(slot) * NodeMetrics.pinSpacing
    }

    /// Per-output-pin centre in *model* coordinates.  `slot` indexes into
    /// the alphabetical `outputAttrNames(of:)` list.
    public static func outputPinModelCenter(
        node: MGNode,
        nodePosition: MGPosition,
        slot: Int
    ) -> CGPoint {
        _ = node
        return CGPoint(
            x: nodePosition.x + NodeMetrics.width,
            y: nodePosition.y + outputPinLocalY(slot: slot)
        )
    }

    /// Convenience: centre of the output pin for the given attribute *name*.
    /// Returns `nil` if `attr` is not in the node's declared outputs.
    public static func outputPinModelCenter(
        node: MGNode,
        nodePosition: MGPosition,
        attr: String
    ) -> CGPoint? {
        let names = outputAttrNames(of: node)
        guard let slot = names.firstIndex(of: attr) else { return nil }
        return outputPinModelCenter(node: node, nodePosition: nodePosition, slot: slot)
    }

    /// Legacy single-output centre — kept for callers that still assume one
    /// output (e.g. fallback paths and the M6 rubber-band origin when only the
    /// node position is known).  Same geometry as the slot-0 multi-output
    /// pin for nodes whose primary output sorts alphabetically first; for
    /// the small number of nodes where it doesn't (e.g. StructureFromMotion,
    /// whose `extraInfoFolder` sorts before `output`), callers should
    /// prefer the slot-based overload.
    public static func outputPinModelCenter(nodePosition: MGPosition) -> CGPoint {
        CGPoint(
            x: nodePosition.x + NodeMetrics.width,
            y: nodePosition.y + NodeMetrics.height / 2
        )
    }

    /// Canonical output pin attribute name used when no Spec is registered.
    /// Meshroom nodes overwhelmingly expose their primary output under the
    /// literal name `"output"` — every fixture and template in the codebase
    /// uses it for cameraInit, featureExtraction, sfm, depthMap, etc.
    public static let outputAttrName = "output"

    // MARK: - M8 type compatibility

    /// Resolve the declared type of an input attribute on a node, returning
    /// `nil` for unknown attrs or unknown node types.  Unknown types short-
    /// circuit the type checker to "allow" (M8 spec rule #5).
    public static func inputType(of node: MGNode, attr: String) -> String? {
        guard let spec = NodeBinary.spec(for: node.nodeType) else { return nil }
        return spec.inputTypes[attr]
    }

    /// Resolve the declared type of an output attribute on a node.  Same
    /// "unknown = nil" semantics as `inputType(of:attr:)`.
    public static func outputType(of node: MGNode, attr: String) -> String? {
        guard let spec = NodeBinary.spec(for: node.nodeType) else { return nil }
        return spec.outputTypes[attr]
    }

    /// Check whether a connection from `sourceType` → `targetType` is allowed.
    ///
    /// Compatibility matrix (M8 spec §4):
    /// - `file → file` ✓
    /// - `file → fileArray` ✓ (Meshroom semantics: a single file is wrapped
    ///   into a one-element ListAttribute by `featuresFolders` etc.)
    /// - `int → int` ✓ and `int → intArray` ✓
    /// - `string → string` ✓
    /// - `bool → bool` ✓
    /// - `double → double` ✓ (also folded into the matrix for completeness)
    /// - **Anything not enumerated → ✗ unless either type is unknown**
    ///
    /// If `sourceType == nil` or `targetType == nil` the function returns
    /// `true` — we cannot prove incompatibility against an unknown type, so
    /// we err on the side of allow (rule #5).  The same applies to either
    /// type being the empty string, which we treat as "unspecified".
    public static func areCompatible(sourceType: String?, targetType: String?) -> Bool {
        guard let s = sourceType, !s.isEmpty,
              let t = targetType, !t.isEmpty
        else {
            return true
        }
        if s == t { return true }
        switch (s, t) {
        case ("file", "fileArray"): return true
        case ("int", "intArray"): return true
        case ("string", "stringArray"): return true
        default:
            return false
        }
    }
}
