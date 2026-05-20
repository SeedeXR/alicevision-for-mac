import Foundation
import ProjectModel

/// Topological scheduling for a Meshroom node graph.
///
/// Edges are derived from input attribute *template references* — the same
/// mechanism `GraphLayout.edges(_:)` uses for visual edge rendering.  This
/// keeps the runtime execution order consistent with what the user sees on
/// the canvas:  a node visually downstream of another is also temporally
/// downstream of it.
///
/// We use Kahn's algorithm (BFS over in-degree) because its terminating
/// condition gives us a free cycle check:  if at the end of the loop we have
/// not emitted every node, the graph contains a cycle and we throw.  DFS-based
/// topo sort would also work but requires a separate colour-marking pass for
/// cycle detection; Kahn's is the simpler primitive for our needs.
public enum GraphScheduler {

    /// Errors raised when the graph cannot be linearised.  Currently only
    /// `cycle` is observable; a future `isolatedNode` case could surface
    /// disconnected sub-graphs if the executor ever wants to refuse them.
    public enum SchedulerError: Error, CustomStringConvertible {
        case cycle(remaining: [String])

        public var description: String {
            switch self {
            case .cycle(let remaining):
                return "Graph contains a cycle involving: \(remaining.joined(separator: ", "))"
            }
        }
    }

    /// Names of the source nodes referenced by any input of `node` — i.e.,
    /// the immediate predecessors in the execution DAG.
    ///
    /// The reference walker recurses into arrays and nested objects via
    /// `GraphLayout.references(in:)`, so dependencies inside `featuresFolders:
    /// ["{X.output}"]` or `advanced: { foo: "{Y.out}" }` are picked up.
    ///
    /// Dangling references (source node missing from the graph) are dropped
    /// silently — same policy as edge rendering, matching Meshroom's tolerance
    /// for partial graphs.
    public static func dependencies(of node: MGNode, in graph: MGGraph) -> [String] {
        guard let inputs = node.inputs else { return [] }
        var seen: Set<String> = []
        var out: [String] = []
        let nodeSet = Set(graph.nodes.keys)
        // Sorted iteration for stable order — makes tests deterministic and
        // produces predictable error messages when a cycle does exist.
        for key in inputs.keys.sorted() {
            guard let value = inputs[key] else { continue }
            for ref in GraphLayout.references(in: value) {
                guard nodeSet.contains(ref.nodeName) else { continue }
                if seen.insert(ref.nodeName).inserted {
                    out.append(ref.nodeName)
                }
            }
        }
        return out
    }

    /// Return all node names in a valid execution order, or throw `.cycle` if
    /// no such order exists.
    ///
    /// Tie-breaking: among nodes with equal current in-degree we emit them
    /// alphabetically by name.  This makes the order deterministic across
    /// runs, which both stabilises tests and gives the user a predictable
    /// "running list" in the UI.
    public static func topologicalOrder(_ graph: MGGraph) throws -> [String] {
        // Build incoming-edge counts and adjacency lists in one pass.
        var inDegree: [String: Int] = [:]
        var adjacency: [String: [String]] = [:]  // source -> [targets]
        for name in graph.nodes.keys {
            inDegree[name] = 0
            adjacency[name] = []
        }
        for (targetName, node) in graph.nodes {
            for source in dependencies(of: node, in: graph) {
                adjacency[source, default: []].append(targetName)
                inDegree[targetName, default: 0] += 1
            }
        }

        // Seed the queue with every zero-in-degree node, alphabetically for
        // deterministic emission order.
        var ready = inDegree.filter { $0.value == 0 }.map(\.key).sorted()
        var order: [String] = []
        order.reserveCapacity(graph.nodes.count)

        while !ready.isEmpty {
            // Pop alphabetically-smallest ready node to keep order stable.
            let next = ready.removeFirst()
            order.append(next)
            for dependent in (adjacency[next] ?? []).sorted() {
                inDegree[dependent, default: 0] -= 1
                if inDegree[dependent] == 0 {
                    // Insert preserving sorted order.
                    let pos = ready.firstIndex(where: { $0 > dependent }) ?? ready.endIndex
                    ready.insert(dependent, at: pos)
                }
            }
        }

        if order.count != graph.nodes.count {
            // Anything still with non-zero in-degree participates in (or
            // depends on) a cycle.  Report those so users can find the loop.
            let remaining = inDegree
                .filter { $0.value > 0 }
                .map(\.key)
                .sorted()
            throw SchedulerError.cycle(remaining: remaining)
        }
        return order
    }
}
