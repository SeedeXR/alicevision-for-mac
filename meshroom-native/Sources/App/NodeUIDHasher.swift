import Foundation
import CryptoKit
import ProjectModel

/// Computes Meshroom-compatible UIDs for nodes in a `.mg` graph.
///
/// Contract (M6):
///
/// Python Meshroom's actual UID is `sha1(str([(attr, attr_uid), ...] + [nodeType]))`
/// where each `attr_uid` is itself a recursive hash of the attribute's value tree
/// (see `meshroom-mac/meshroom/core/node.py::_computeUid` + `attribute.py::uid`).
/// Re-implementing that byte-equivalently in Swift is a substantial undertaking —
/// it depends on the exact `repr()` of Python tuples and `str()` of `Attribute`
/// objects, both of which embed implementation details we'd have to mirror
/// exactly.
///
/// Instead, M6 ships a **simpler approximation** described in the M6 brief:
///   `uid = sha1(json.dumps(resolvedInputs, sort_keys=True, default=str))`
///
/// This is not byte-equivalent to Python Meshroom's `_computeUid`, so a
/// freshly-edited graph that *we* run will produce different cache-dir names
/// than a graph that Python Meshroom runs end-to-end.  But:
///
/// 1. The hash is **deterministic** across our own runs: same inputs → same
///    cache dir → warm-cache reuse.
/// 2. When the user opens a `.mg` saved by Python Meshroom, that file already
///    carries an explicit `"uid": "<sha1-hex>"` per node (see `MGNode.uid`).
///    `GraphExecutor` prefers `node.uid` over the recomputed value, so existing
///    Python-Meshroom cache directories are still consumable by our app.
/// 3. Position-only changes don't invalidate the cache because position lives
///    in `internalInputs`, not `inputs` — the hash function only sees `inputs`.
///
/// Format details (must byte-match Python's `json.dumps(..., sort_keys=True,
/// default=str)`):
/// - Object keys sorted lexicographically.
/// - `": "` between key and value, `", "` between items (default separators).
/// - `ensure_ascii=True` default: non-ASCII → `\uXXXX`.
/// - `/` is **not** escaped.  `"`, `\`, `\n`, `\r`, `\t`, `\b`, `\f` escaped;
///   other control chars `< 0x20` → `\u00XX` (lowercase hex).
/// - Floats via Swift's `String(Double)` which matches CPython `repr(float)`
///   for the values we typically see in Meshroom inputs (`1.0`, `0.5`,
///   `1e-05`, …).
/// - Booleans: `true`/`false`; null: `null`.
/// - SHA-1 from CryptoKit's `Insecure.SHA1`.  SHA-1 is required for Meshroom
///   compatibility even though CryptoKit flags it as deprecated for crypto.
public enum NodeUIDHasher {

    /// Computes the UID for a node, given its resolved input values
    /// (template references already substituted with upstream resolved paths).
    ///
    /// - Parameter resolvedInputs: the `inputs` dictionary with `{Node.attr}`
    ///   references rewritten to concrete strings.  Values that are not template
    ///   strings (numbers, bools, nested arrays/objects) are passed through.
    /// - Returns: lowercase 40-char SHA-1 hex string.
    public static func uid(for resolvedInputs: [String: MGJSONValue]) -> String {
        let json = canonicalJSON(.object(resolvedInputs))
        return sha1Hex(json)
    }

    /// Computes UIDs for every node in a graph, in topological order.
    ///
    /// Reference resolution: when a node's input contains a template like
    /// `"{Upstream.output}"`, the source attribute's resolved value
    /// (== the upstream node's cache-path containing its own UID) is
    /// substituted before hashing.  This produces a Merkle-like dependency:
    /// any change upstream invalidates every downstream UID.
    ///
    /// The output path used for substitution is recomputed from the upstream
    /// node's declared `outputs` template, with `{nodeCacheFolder}` expanded
    /// to `MeshroomCache/<UpstreamName>_<UpstreamUID>/`.  Note this uses a
    /// **logical** cache root — the function doesn't know the on-disk workdir.
    /// `GraphExecutor` re-derives the same logical path using its own workdir
    /// and ends up at the same UID because the workdir prefix appears
    /// **identically** in both upstream and downstream substitutions, so
    /// they wash out symmetrically.  To remove any chance of skew we use a
    /// fixed sentinel cache root here; downstream-resolved input strings
    /// then carry only the relative `<NodeName>_<UID>` segment that is
    /// genuinely dependency-bearing.
    ///
    /// - Parameters:
    ///   - graph: the graph to hash.
    ///   - topoOrder: a topological order of `graph.nodes.keys`.  Caller is
    ///     responsible for producing this (use `GraphScheduler.topologicalOrder`).
    /// - Returns: `[nodeName: uid]` for every node in the graph.
    public static func uids(for graph: MGGraph, topoOrder: [String]) -> [String: String] {
        // Logical cache root used to make upstream paths content-deterministic.
        // The literal value doesn't matter as long as it's stable across nodes:
        // every node sees the same prefix string, so two graphs with the same
        // structural inputs hash to the same UIDs regardless of where the user
        // actually placed `MeshroomCache/` on disk.
        let logicalCacheRoot = "<cache>"

        var uids: [String: String] = [:]
        var upstreamOutputs: [String: [String: String]] = [:]

        for nodeName in topoOrder {
            guard let node = graph[nodeName] else { continue }
            let inputs = node.inputs ?? [:]
            let resolved = resolveInputs(inputs, against: upstreamOutputs)
            let nodeUID = uid(for: resolved)
            uids[nodeName] = nodeUID

            // Pre-publish this node's resolved outputs so downstream nodes
            // can substitute against them.  Output templates contain
            // `{nodeCacheFolder}` which we expand to
            // `<cache>/MeshroomCache/<Node>_<UID>/`.
            let cacheDir = "\(logicalCacheRoot)/MeshroomCache/\(nodeName)_\(nodeUID)"
            var resolvedOuts: [String: String] = [:]
            if let outs = node.outputs {
                for (k, v) in outs {
                    if case let .string(template) = v {
                        resolvedOuts[k] = template.replacingOccurrences(
                            of: "{nodeCacheFolder}",
                            with: cacheDir
                        )
                    }
                }
            }
            upstreamOutputs[nodeName] = resolvedOuts
        }
        return uids
    }

    // MARK: - Reference resolution

    /// Walk an inputs dict, substituting `{Node.attr}` strings against
    /// `upstreamOutputs`.  Recurses into arrays and nested objects so
    /// references inside `featuresFolders:[…]` resolve correctly.
    private static func resolveInputs(
        _ inputs: [String: MGJSONValue],
        against upstreamOutputs: [String: [String: String]]
    ) -> [String: MGJSONValue] {
        var out: [String: MGJSONValue] = [:]
        out.reserveCapacity(inputs.count)
        for (k, v) in inputs {
            out[k] = resolveValue(v, against: upstreamOutputs)
        }
        return out
    }

    private static func resolveValue(
        _ value: MGJSONValue,
        against upstreamOutputs: [String: [String: String]]
    ) -> MGJSONValue {
        switch value {
        case .string(let s):
            if let ref = MGTemplateReference.parse(s),
               let outs = upstreamOutputs[ref.nodeName],
               let attr = ref.attributePath.first,
               let resolved = outs[attr] {
                return .string(resolved)
            }
            return value
        case .array(let arr):
            return .array(arr.map { resolveValue($0, against: upstreamOutputs) })
        case .object(let obj):
            var out: [String: MGJSONValue] = [:]
            out.reserveCapacity(obj.count)
            for (k, v) in obj {
                out[k] = resolveValue(v, against: upstreamOutputs)
            }
            return .object(out)
        case .null, .bool, .int, .double:
            return value
        }
    }

    // MARK: - SHA-1

    private static func sha1Hex(_ s: String) -> String {
        let data = Data(s.utf8)
        let digest = Insecure.SHA1.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Canonical JSON (Python json.dumps default-compatible)

    /// Render `value` to a JSON string byte-equivalent to Python's
    /// `json.dumps(v, sort_keys=True, default=str)` for the value types
    /// representable as `MGJSONValue` (no `Path`/datetime/etc. so `default=str`
    /// never fires).
    static func canonicalJSON(_ value: MGJSONValue) -> String {
        var buf = ""
        appendJSON(value, into: &buf)
        return buf
    }

    private static func appendJSON(_ value: MGJSONValue, into buf: inout String) {
        switch value {
        case .null:
            buf.append("null")
        case .bool(let b):
            buf.append(b ? "true" : "false")
        case .int(let i):
            buf.append(String(i))
        case .double(let d):
            appendDouble(d, into: &buf)
        case .string(let s):
            appendString(s, into: &buf)
        case .array(let arr):
            appendArray(arr, into: &buf)
        case .object(let obj):
            appendObject(obj, into: &buf)
        }
    }

    private static func appendObject(_ obj: [String: MGJSONValue], into buf: inout String) {
        if obj.isEmpty {
            buf.append("{}")
            return
        }
        buf.append("{")
        var first = true
        for key in obj.keys.sorted() {
            if !first { buf.append(", ") }
            first = false
            appendString(key, into: &buf)
            buf.append(": ")
            appendJSON(obj[key]!, into: &buf)
        }
        buf.append("}")
    }

    private static func appendArray(_ arr: [MGJSONValue], into buf: inout String) {
        if arr.isEmpty {
            buf.append("[]")
            return
        }
        buf.append("[")
        for (i, v) in arr.enumerated() {
            if i > 0 { buf.append(", ") }
            appendJSON(v, into: &buf)
        }
        buf.append("]")
    }

    /// Float formatting matching Python's `json.dumps`, which uses
    /// `float.__repr__` (since CPython 3.1).  Swift's `String(Double)`
    /// uses the same shortest-roundtrip algorithm and produces the same
    /// output for every finite value we've observed (`1.0`, `0.5`, `0.0001`,
    /// `1e-05`, `1e+16`, `3.141592653589793`).  NaN/Infinity are not
    /// representable in JSON; we emit `NaN`/`Infinity`/`-Infinity` to match
    /// Python's default `allow_nan=True` behaviour (which would otherwise
    /// raise) — practical Meshroom inputs never contain these.
    private static func appendDouble(_ d: Double, into buf: inout String) {
        if d.isNaN {
            buf.append("NaN")
            return
        }
        if d.isInfinite {
            buf.append(d < 0 ? "-Infinity" : "Infinity")
            return
        }
        buf.append(String(d))
    }

    /// String escaping matching Python's `json.dumps` default
    /// (`ensure_ascii=True`).  Escape table:
    /// - `"` → `\"`, `\` → `\\`
    /// - `\b` → `\b`, `\f` → `\f`, `\n` → `\n`, `\r` → `\r`, `\t` → `\t`
    /// - other ASCII control chars `< 0x20` → `\u00xx` (lowercase hex)
    /// - non-ASCII (code point ≥ 0x80) → `\uXXXX`; for code points beyond
    ///   the BMP, Python emits a UTF-16 surrogate pair, matching the JSON
    ///   spec.
    /// - `/` is **not** escaped (Python default).
    private static func appendString(_ s: String, into buf: inout String) {
        buf.append("\"")
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"":
                buf.append("\\\"")
            case "\\":
                buf.append("\\\\")
            case "\u{08}":
                buf.append("\\b")
            case "\u{0C}":
                buf.append("\\f")
            case "\n":
                buf.append("\\n")
            case "\r":
                buf.append("\\r")
            case "\t":
                buf.append("\\t")
            default:
                let v = scalar.value
                // Python `ensure_ascii=True` escapes everything outside
                // 0x20..=0x7e: that's ASCII control chars (< 0x20), DEL
                // (0x7f), and non-ASCII (>= 0x80).
                if v >= 0x20 && v <= 0x7E {
                    buf.unicodeScalars.append(scalar)
                } else if v <= 0xFFFF {
                    buf.append(String(format: "\\u%04x", v))
                } else {
                    // Non-BMP: encode as UTF-16 surrogate pair.
                    let adjusted = v - 0x10000
                    let high = 0xD800 + (adjusted >> 10)
                    let low = 0xDC00 + (adjusted & 0x3FF)
                    buf.append(String(format: "\\u%04x\\u%04x", high, low))
                }
            }
        }
        buf.append("\"")
    }
}
