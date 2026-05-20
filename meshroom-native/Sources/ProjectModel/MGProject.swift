import Foundation

/// Top-level representation of a Meshroom `.mg` project file.
///
/// The on-disk format is plain JSON with two top-level keys, `header` and
/// `graph`.  See `meshroom/core/graphIO.py::GraphSerializer.serialize()` for
/// the authoritative shape.
public struct MGProject: Hashable, Sendable {
    public var header: MGHeader
    public var graph: MGGraph

    public init(header: MGHeader, graph: MGGraph) {
        self.header = header
        self.graph = graph
    }

    // MARK: - Canonical decode/encode
    //
    // We route the canonical path through `JSONSerialization` (not
    // `JSONDecoder`) so the int-vs-double distinction inside heterogeneous
    // `inputs` values survives a round-trip.  `JSONDecoder` collapses
    // `NSNumber.float` into Swift `Double`, then re-emits as e.g. `45` (no
    // `.0`), so a re-decode would surface `.int(45)` instead of `.double(45)`.

    /// Decode a `.mg` file from raw JSON bytes.
    public static func decode(from data: Data) throws -> MGProject {
        let any = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        guard let root = any as? [String: Any] else {
            throw MGProjectError.invalidTopLevel
        }
        return try MGProject(jsonRoot: root)
    }

    /// Encode in Meshroom's canonical pretty-printed form: 4-space indent,
    /// sorted keys, and *floats are emitted with a decimal point* (`45.0`,
    /// not `45`).  This last detail is the reason we route through the
    /// hand-rolled `MGJSONWriter` instead of `JSONSerialization.data(...)`,
    /// which collapses `Double(45.0)` to `"45"` and breaks round-trip
    /// fidelity for any `.mg` containing float parameters.
    public func encode() throws -> Data {
        let root = jsonRoot()
        return try MGJSONWriter.write(root)
    }

    /// Load from disk.  Convenience for tests and the read-only viewer (M1).
    public static func load(from url: URL) throws -> MGProject {
        let data = try Data(contentsOf: url)
        return try decode(from: data)
    }

    // MARK: - JSONSerialization bridge

    init(jsonRoot root: [String: Any]) throws {
        guard let headerAny = root["header"] as? [String: Any] else {
            throw MGProjectError.missingField("header")
        }
        self.header = try MGHeader(jsonObject: headerAny)
        let graphAny = (root["graph"] as? [String: Any]) ?? [:]
        self.graph = try MGGraph(jsonObject: graphAny)
    }

    func jsonRoot() -> [String: Any] {
        [
            "header": header.jsonObject(),
            "graph": graph.jsonObject(),
        ]
    }

    // MARK: - Mutation API (M3)
    //
    // M2 was strictly read-only; M3 introduces interactive edits.  Mutation
    // is exposed as named methods rather than direct property writes so we can
    // centralise invariants (e.g. preserving `MGPosition.integerStorage`) and
    // keep a single point where view-models register undo actions.

    /// Update the position of a node by name.  No-op if the node is unknown.
    ///
    /// Preserves `MGPosition.integerStorage`:  if the original position was
    /// stored as integers (Meshroom's convention) we keep emitting integers
    /// on re-encode; if the file already used floats we keep floats.  Newly
    /// created `MGPosition` values default to `integerStorage = true` since
    /// the canvas snaps to integer pixels on drag.
    @discardableResult
    public mutating func updateNodePosition(name: String, position: MGPosition) -> Bool {
        guard var node = graph[name] else { return false }
        // Preserve integer-storage flag from the existing position so a file
        // that originally serialised positions as integers continues to do so.
        var newPos = position
        if let existing = node.position {
            newPos.integerStorage = existing.integerStorage
        }
        node.position = newPos
        graph[name] = node
        return true
    }

    /// Update a single attribute value inside a node's `inputs` dictionary,
    /// addressed by a dot-separated path.
    ///
    /// `attrPath` semantics (matching Meshroom's `Attribute.getName()` convention
    /// when walking nested ListAttribute/GroupAttribute trees):
    /// - `"focalLength"` — top-level scalar input.
    /// - `"foo.bar.baz"` — nested dict (`.object`) traversal.
    /// - `"foo.0.bar"` — array index traversal: integer-valued path components
    ///   step into `.array` values.
    ///
    /// Returns `true` on a successful update; `false` if the node is missing,
    /// the path is invalid (e.g. crosses a scalar leaf), or any intermediate
    /// container is missing.  The mutation is atomic — we build the replacement
    /// `inputs` dict in-memory before assigning, so a failed traversal leaves
    /// the project untouched.  This preserves crash-free Cmd-S after a no-op
    /// edit and is the contract `ProjectViewModel.commitInputValue` relies on.
    @discardableResult
    public mutating func updateAttribute(
        node nodeName: String,
        attrPath: String,
        newValue: MGJSONValue
    ) -> Bool {
        guard var node = graph[nodeName] else { return false }
        let components = attrPath.split(separator: ".").map(String.init)
        guard !components.isEmpty else { return false }
        // We update `inputs`; templates and saved projects both use this dict
        // for user-editable parameters.  `internalInputs` (label/color/comment)
        // are out of scope for M4 — those get their own editor in M5+.
        var inputs = node.inputs ?? [:]
        let topKey = components[0]
        let rest = Array(components.dropFirst())
        guard let topValue = inputs[topKey] else {
            // Direct write only when there are no further path components;
            // otherwise the intermediate container is missing and we refuse to
            // synthesise one — that's an attribute-schema concern, not a path
            // concern.
            if rest.isEmpty {
                inputs[topKey] = newValue
                node.inputs = inputs
                graph[nodeName] = node
                return true
            }
            return false
        }
        guard let replaced = Self.replaceAtPath(in: topValue, path: rest, with: newValue) else {
            return false
        }
        inputs[topKey] = replaced
        node.inputs = inputs
        graph[nodeName] = node
        return true
    }

    /// Recursive helper: produce a copy of `value` with the leaf at `path`
    /// replaced by `newValue`.  Returns `nil` if the path is invalid for the
    /// shape of `value` (e.g. indexing into a scalar, out-of-bounds array
    /// index, missing dict key).  An empty `path` simply returns `newValue`.
    private static func replaceAtPath(
        in value: MGJSONValue,
        path: [String],
        with newValue: MGJSONValue
    ) -> MGJSONValue? {
        guard let head = path.first else { return newValue }
        let tail = Array(path.dropFirst())
        switch value {
        case .object(var dict):
            guard let child = dict[head] else { return nil }
            guard let updated = replaceAtPath(in: child, path: tail, with: newValue) else {
                return nil
            }
            dict[head] = updated
            return .object(dict)
        case .array(var arr):
            // Integer-valued path component indexes into the array.  We accept
            // only decimal integers — no negative indices, no slicing.
            guard let idx = Int(head), idx >= 0, idx < arr.count else { return nil }
            guard let updated = replaceAtPath(in: arr[idx], path: tail, with: newValue) else {
                return nil
            }
            arr[idx] = updated
            return .array(arr)
        default:
            // Cannot descend further: any remaining path component over a
            // scalar leaf is an invalid traversal.
            return nil
        }
    }

    /// M9 — add a brand-new node to the graph.
    ///
    /// Refuses (returns `false`) when `name` is already taken — name
    /// uniqueness is the graph's primary key.  Callers are expected to
    /// generate non-colliding names via `nextAvailableNodeName(forType:)`.
    ///
    /// `uid` defaults to a freshly generated UUID; the M6 `NodeUIDHasher`
    /// will recompute a deterministic hash once the node's inputs are
    /// connected and the project is saved, so this UUID is just a stable
    /// placeholder.
    @discardableResult
    public mutating func addNode(
        name: String,
        nodeType: String,
        position: MGPosition,
        uid: String? = nil
    ) -> Bool {
        guard graph[name] == nil else { return false }
        let resolvedUid = uid ?? UUID().uuidString.lowercased()
        let node = MGNode(
            nodeType: nodeType,
            position: position,
            uid: resolvedUid,
            inputs: [:]
        )
        graph[name] = node
        return true
    }

    /// Remove a node by name.  No-op if missing.  M9 doesn't expose this
    /// in the UI, but pairing addNode with a deletion path keeps the test
    /// surface symmetric and lets future milestones (M10+) wire a Delete
    /// shortcut without re-touching the model layer.
    @discardableResult
    public mutating func removeNode(name: String) -> Bool {
        guard graph[name] != nil else { return false }
        graph.nodes.removeValue(forKey: name)
        return true
    }

    /// Compute the smallest unused `<NodeType>_<n>` name (n ≥ 1).  Walks every
    /// existing node name with that prefix, parses the trailing suffix, and
    /// returns one above the maximum seen — or `_1` when no node of the type
    /// exists yet.  Linear in graph size, which is fine for graphs <100 nodes.
    public func nextAvailableNodeName(forType nodeType: String) -> String {
        let prefix = nodeType + "_"
        var maxSuffix = 0
        for key in graph.nodes.keys where key.hasPrefix(prefix) {
            let suffix = key.dropFirst(prefix.count)
            if let n = Int(suffix), n > maxSuffix {
                maxSuffix = n
            }
        }
        return "\(prefix)\(maxSuffix + 1)"
    }

    /// Convenience: write the canonical encoded form atomically to a URL.
    ///
    /// "Atomic" here means we write to `<url>.tmp` and `rename(2)` it onto
    /// `<url>`.  POSIX guarantees rename within a single filesystem is atomic,
    /// so a crash mid-save can never leave a half-written `.mg` on disk.
    public func writeAtomically(to url: URL) throws {
        let data = try encode()
        let tmpURL = url.appendingPathExtension("tmp")
        // If a stale .tmp exists from a previous crash, remove it first; the
        // rename below cannot overwrite a directory or non-regular file.
        if FileManager.default.fileExists(atPath: tmpURL.path) {
            try? FileManager.default.removeItem(at: tmpURL)
        }
        try data.write(to: tmpURL, options: [.atomic])
        // `replaceItemAt` is the documented macOS-safe way to swap files; it
        // handles cross-volume cases by falling back to copy+delete.  We
        // ignore the `resultingItemURL` because we want the canonical URL.
        _ = try FileManager.default.replaceItemAt(url, withItemAt: tmpURL)
    }
}

/// `.mg` header block.
///
/// `releaseVersion` is the Meshroom application version that wrote the file.
/// `fileVersion` is the schema version — currently `"2.0"`.
/// `nodesVersions` maps node-type name to its individual version string.
/// `template` is `true` for `.mg` template files (where node `uid`/`outputs`
/// are omitted); `nil` or `false` for ordinary saved projects.
public struct MGHeader: Hashable, Sendable {
    public var releaseVersion: String
    public var fileVersion: String
    public var nodesVersions: [String: String]
    public var template: Bool?

    public init(
        releaseVersion: String,
        fileVersion: String,
        nodesVersions: [String: String] = [:],
        template: Bool? = nil
    ) {
        self.releaseVersion = releaseVersion
        self.fileVersion = fileVersion
        self.nodesVersions = nodesVersions
        self.template = template
    }

    init(jsonObject obj: [String: Any]) throws {
        self.releaseVersion = (obj["releaseVersion"] as? String) ?? ""
        self.fileVersion = (obj["fileVersion"] as? String) ?? ""
        if let nv = obj["nodesVersions"] as? [String: String] {
            self.nodesVersions = nv
        } else if let nvAny = obj["nodesVersions"] as? [String: Any] {
            // Tolerate non-string values by stringifying — matches Meshroom
            // which historically allowed Version() objects to be serialised.
            self.nodesVersions = nvAny.compactMapValues { ($0 as? String) ?? "\($0)" }
        } else {
            self.nodesVersions = [:]
        }
        if let t = obj["template"] as? Bool {
            self.template = t
        } else {
            self.template = nil
        }
    }

    func jsonObject() -> [String: Any] {
        var out: [String: Any] = [
            "releaseVersion": releaseVersion,
            "fileVersion": fileVersion,
            "nodesVersions": nodesVersions,
        ]
        if let t = template {
            out["template"] = NSNumber(value: t)
        }
        return out
    }
}
