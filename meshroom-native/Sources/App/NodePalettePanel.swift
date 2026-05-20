import SwiftUI
import ProjectModel
import UniformTypeIdentifiers

/// M9 — left-edge sidebar listing every node type the user can drag onto the
/// canvas.  Pairs with the right-edge inspector to give the user "library on
/// the left, properties on the right" — the convention every node-editor
/// (Houdini, Nuke, TouchDesigner, Origami, Blender Geometry Nodes) uses.
///
/// Each row is a drag source carrying a string payload `"avNodeType:<Type>"`.
/// The canvas's `.onDrop` consumes the payload, decodes the type, converts
/// the drop point into model coordinates, and calls
/// `ProjectViewModel.addNode(nodeType:at:undoManager:)`.
///
/// We deliberately stay decoupled from `NodeBinary.spec` order so palette
/// ordering matches the pipeline reading order rather than the dictionary's
/// hash order (which would shuffle between Swift runtime versions).
struct NodePalettePanel: View {
    @ObservedObject var viewModel: ProjectViewModel

    /// Ordering matches a typical photogrammetry pipeline reading top-down,
    /// so the user can drag CameraInit first, then FeatureExtraction, etc.
    /// Pulled out as a static constant so the palette is testable without a
    /// live ProjectViewModel — `NodePalettePanel.allNodeTypes` is the source
    /// of truth for the test "every NodeBinary spec is in the palette".
    static let allNodeTypes: [String] = [
        "CameraInit",
        "FeatureExtraction",
        "ImageMatching",
        "FeatureMatching",
        "StructureFromMotion",
        "PrepareDenseScene",
        "DepthMap",
        "DepthMapFilter",
        "Meshing",
        "MeshFiltering",
        "Texturing",
        "ImportMiddlebury",
    ]

    /// SF Symbol per node type.  Picked for rough semantic fit; chosen for
    /// availability across macOS 14+ rather than icon "correctness".  Falls
    /// back to a generic shape for unknown types.
    static func iconName(forType nodeType: String) -> String {
        switch nodeType {
        case "CameraInit":          return "camera"
        case "FeatureExtraction":   return "dot.scope"
        case "ImageMatching":       return "rectangle.on.rectangle"
        case "FeatureMatching":     return "arrow.left.arrow.right"
        case "StructureFromMotion": return "cube.transparent"
        case "PrepareDenseScene":   return "square.stack.3d.up"
        case "DepthMap":            return "square.layers"
        case "DepthMapFilter":      return "line.3.horizontal.decrease.circle"
        case "Meshing":             return "pyramid"
        case "MeshFiltering":       return "scissors"
        case "Texturing":           return "paintpalette"
        case "ImportMiddlebury":    return "tray.and.arrow.down"
        default:                    return "questionmark.app"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Nodes")
                    .font(.headline)
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.top, 8)
            .padding(.bottom, 6)
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(Self.allNodeTypes, id: \.self) { type in
                        NodePaletteRow(nodeType: type)
                    }
                }
                .padding(.vertical, 4)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

/// One row in the palette.  Carries a `NSItemProvider` for SwiftUI's drag-and-
/// drop pipeline; the payload is the literal node type string prefixed so
/// the canvas's `.onDrop` can unambiguously identify it as a palette drop
/// (rather than e.g. a file URL dropped from Finder).
struct NodePaletteRow: View {
    let nodeType: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: NodePalettePanel.iconName(forType: nodeType))
                .foregroundStyle(.secondary)
                .frame(width: 18)
            Text(nodeType)
                .font(.system(.body, design: .default))
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        // SwiftUI's drag-source modifier.  We hand back a plain-text item
        // provider whose payload is `"avNodeType:<Type>"`.  Drop targets
        // (the canvas) reject any string that doesn't start with that
        // prefix — keeps us from misinterpreting Finder file drags.
        .onDrag {
            NSItemProvider(object: ("avNodeType:" + nodeType) as NSString)
        }
        .help("Drag \(nodeType) onto the canvas")
    }
}

/// Extension: helper to parse a dropped item provider into the node type
/// it carries (or `nil` if the provider isn't a palette drag).  Pulled out
/// of the canvas so the test target can exercise the parsing rules without
/// firing SwiftUI gestures.
enum PalettePayload {
    static let prefix = "avNodeType:"

    /// Decode a raw payload string into a node type.  Returns `nil` if
    /// `raw` doesn't begin with the expected prefix or the trailing type
    /// name is empty.
    static func decode(_ raw: String) -> String? {
        guard raw.hasPrefix(prefix) else { return nil }
        let type = String(raw.dropFirst(prefix.count))
        return type.isEmpty ? nil : type
    }
}
