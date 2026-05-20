import Foundation

/// The `graph` block of a `.mg` file: a flat dictionary mapping unique node
/// names (e.g. `"FeatureExtraction_1"`) to their definitions.
///
/// JSON object key ordering is not guaranteed by RFC 8259, so the round-trip
/// test compares by *content*, not by serialized byte equality.
public struct MGGraph: Hashable, Sendable {
    public var nodes: [String: MGNode]

    public init(nodes: [String: MGNode] = [:]) {
        self.nodes = nodes
    }

    /// Convenience: node names sorted alphabetically (Meshroom does this too;
    /// see `GraphSerializer.serializeContent`).
    public var sortedNodeNames: [String] {
        nodes.keys.sorted()
    }

    public var count: Int { nodes.count }

    public subscript(name: String) -> MGNode? {
        get { nodes[name] }
        set { nodes[name] = newValue }
    }

    // MARK: - JSONSerialization bridge

    init(jsonObject obj: [String: Any]) throws {
        var out: [String: MGNode] = [:]
        out.reserveCapacity(obj.count)
        for (name, value) in obj {
            guard let nodeAny = value as? [String: Any] else {
                throw MGProjectError.unexpectedJSONValue("graph[\(name)] not an object")
            }
            out[name] = try MGNode(jsonObject: nodeAny)
        }
        self.nodes = out
    }

    func jsonObject() -> [String: Any] {
        var out: [String: Any] = [:]
        out.reserveCapacity(nodes.count)
        for (name, node) in nodes {
            out[name] = node.jsonObject()
        }
        return out
    }
}
