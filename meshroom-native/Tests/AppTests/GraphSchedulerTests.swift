import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// M5 unit tests for the topological scheduler.
///
/// Strategy: synthesise small graphs directly in code (no fixture files) so
/// each test pins exactly one structural property — linear chain, diamond,
/// cycle — without dragging in the noise of a real Meshroom project.  Process
/// invocation is *not* exercised here; XCTest + Foundation `Process` is well
/// known to be flaky on CI (forked child processes interact badly with the
/// XCTest dyld stub on darwin) and the scheduler's contract is purely
/// topological, so unit testing it standalone is the high-value coverage.
final class GraphSchedulerTests: XCTestCase {

    /// Build a node whose `inputs.dependsOn` references each of `deps` as a
    /// template string.  Used to construct synthetic DAGs without having to
    /// hand-write every input attribute.
    private func makeNode(deps: [String]) -> MGNode {
        if deps.isEmpty {
            return MGNode(nodeType: "TestNode", position: MGPosition(x: 0, y: 0))
        }
        // Each dep becomes a `{Name.output}` template reference inside an
        // array-valued input.  `MGTemplateReference.parse` rejects names that
        // aren't valid identifiers, so node names in the tests must stick to
        // `[A-Za-z_][A-Za-z0-9_]*`.
        let refStrings: [MGJSONValue] = deps.map { .string("{\($0).output}") }
        return MGNode(
            nodeType: "TestNode",
            position: MGPosition(x: 0, y: 0),
            inputs: ["dependsOn": .array(refStrings)]
        )
    }

    private func makeProject(_ nodes: [String: MGNode]) -> MGProject {
        MGProject(
            header: MGHeader(releaseVersion: "test", fileVersion: "2.0"),
            graph: MGGraph(nodes: nodes)
        )
    }

    // MARK: - dependencies(of:in:)

    /// Direct API smoke test: the helper must return upstream node names from
    /// a single template reference inside a nested array.
    func testDependenciesExtractsFromArrayedReferences() throws {
        let project = makeProject([
            "A": makeNode(deps: []),
            "B": makeNode(deps: ["A"]),
        ])
        let deps = GraphScheduler.dependencies(of: project.graph["B"]!, in: project.graph)
        XCTAssertEqual(deps, ["A"])
    }

    /// A node with no inputs at all has no dependencies — the lookup must not
    /// crash on the `nil` optional.
    func testDependenciesEmptyForNodeWithoutInputs() {
        let node = MGNode(nodeType: "TestNode", position: MGPosition(x: 0, y: 0))
        let graph = MGGraph(nodes: ["X": node])
        XCTAssertEqual(GraphScheduler.dependencies(of: node, in: graph), [])
    }

    /// Dangling refs (source node missing) must be dropped rather than
    /// reported, matching `GraphLayout.edges(in:)` policy.
    func testDependenciesDropsDanglingReferences() {
        let node = MGNode(
            nodeType: "TestNode",
            position: MGPosition(x: 0, y: 0),
            inputs: ["bad": .string("{NoSuchNode_1.output}")]
        )
        let graph = MGGraph(nodes: ["X": node])
        XCTAssertEqual(GraphScheduler.dependencies(of: node, in: graph), [])
    }

    // MARK: - topologicalOrder(_:)

    /// Linear chain: A -> B -> C must emit exactly [A, B, C].
    func testTopoLinearChain() throws {
        let project = makeProject([
            "A": makeNode(deps: []),
            "B": makeNode(deps: ["A"]),
            "C": makeNode(deps: ["B"]),
        ])
        let order = try GraphScheduler.topologicalOrder(project.graph)
        XCTAssertEqual(order, ["A", "B", "C"])
    }

    /// Diamond:  A -> B, A -> C, B+C -> D.  Both [A,B,C,D] and [A,C,B,D] are
    /// valid topological orders; we accept either but assert the partial
    /// order constraints (A first, D last) are honoured.
    func testTopoDiamond() throws {
        let project = makeProject([
            "A": makeNode(deps: []),
            "B": makeNode(deps: ["A"]),
            "C": makeNode(deps: ["A"]),
            "D": makeNode(deps: ["B", "C"]),
        ])
        let order = try GraphScheduler.topologicalOrder(project.graph)
        XCTAssertEqual(order.count, 4)
        XCTAssertEqual(order.first, "A", "A is the only zero-in-degree node")
        XCTAssertEqual(order.last, "D", "D depends on both B and C")
        // B and C may appear in either order, but both must precede D and
        // follow A — assert position invariants.
        let posA = order.firstIndex(of: "A")!
        let posB = order.firstIndex(of: "B")!
        let posC = order.firstIndex(of: "C")!
        let posD = order.firstIndex(of: "D")!
        XCTAssertLessThan(posA, posB)
        XCTAssertLessThan(posA, posC)
        XCTAssertLessThan(posB, posD)
        XCTAssertLessThan(posC, posD)
    }

    /// Cycle detection: A -> B -> A must throw.  We assert it via
    /// `XCTAssertThrowsError` and pattern-match the error case.
    func testTopoCycleThrows() {
        let project = makeProject([
            "A": makeNode(deps: ["B"]),
            "B": makeNode(deps: ["A"]),
        ])
        XCTAssertThrowsError(try GraphScheduler.topologicalOrder(project.graph)) { err in
            guard case GraphScheduler.SchedulerError.cycle(let remaining) = err else {
                XCTFail("expected .cycle, got \(err)")
                return
            }
            // Both nodes must be flagged as in the cycle.
            XCTAssertTrue(remaining.contains("A"))
            XCTAssertTrue(remaining.contains("B"))
        }
    }

    /// Multi-source DAG: independent zero-in-degree nodes A and X must both
    /// appear before their dependents and the order must remain deterministic
    /// (alphabetical tie-break).
    func testTopoTwoIndependentSources() throws {
        let project = makeProject([
            "Alpha": makeNode(deps: []),
            "Beta": makeNode(deps: ["Alpha"]),
            "Xeno": makeNode(deps: []),
            "Yang": makeNode(deps: ["Xeno"]),
        ])
        let order = try GraphScheduler.topologicalOrder(project.graph)
        XCTAssertEqual(order.count, 4)
        // Alphabetical tie-break: Alpha precedes Xeno; Beta precedes Yang;
        // Alpha precedes Beta and Xeno precedes Yang.
        let positions = Dictionary(uniqueKeysWithValues: order.enumerated().map { ($1, $0) })
        XCTAssertLessThan(positions["Alpha"]!, positions["Beta"]!)
        XCTAssertLessThan(positions["Xeno"]!, positions["Yang"]!)
        XCTAssertLessThan(positions["Alpha"]!, positions["Xeno"]!,
            "alphabetical tie-break on zero-in-degree seed")
    }

    // MARK: - End-to-end with the photogrammetryMini fixture

    /// A real `.mg` must topo-sort without throwing and place every node
    /// after its dependencies.  We don't pin a specific order (the fixture
    /// has multiple valid linearisations) but we verify the partial order is
    /// preserved.
    func testTopoOnRealFixture() throws {
        let bundle = Bundle.module
        guard let url = bundle.url(forResource: "photogrammetryMini", withExtension: "mg", subdirectory: "Fixtures")
            ?? bundle.url(forResource: "photogrammetryMini", withExtension: "mg") else {
            throw XCTSkip("photogrammetryMini fixture missing")
        }
        let project = try MGProject.load(from: url)
        let order = try GraphScheduler.topologicalOrder(project.graph)
        XCTAssertEqual(order.count, project.graph.nodes.count)
        let positions = Dictionary(uniqueKeysWithValues: order.enumerated().map { ($1, $0) })
        // Every dependency must come strictly before its dependent.
        for (name, node) in project.graph.nodes {
            for dep in GraphScheduler.dependencies(of: node, in: project.graph) {
                XCTAssertLessThan(
                    positions[dep] ?? -1,
                    positions[name] ?? -1,
                    "\(dep) must precede \(name) in topo order"
                )
            }
        }
    }
}
