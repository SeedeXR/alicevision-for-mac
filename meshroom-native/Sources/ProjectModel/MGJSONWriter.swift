import Foundation

/// Hand-rolled JSON writer for the `MGProject` tree.
///
/// **Why not just use `JSONSerialization.data(withJSONObject:)`?**
///
/// `JSONSerialization` on Apple platforms collapses `NSNumber(value: 45.0)`
/// to the textual form `"45"` (no decimal point), losing the information that
/// the source was a float.  Re-parsed, `"45"` becomes `NSNumber` of integer
/// type — which would round-trip as `MGJSONValue.int(45)` instead of
/// `.double(45.0)`, breaking value equality and (more importantly) breaking
/// fidelity with Meshroom's own writer which emits `45.0` for floats.
///
/// This writer accepts the `[String: Any]` tree produced by
/// `MGProject.jsonRoot()`, where numerics are typed `NSNumber`s.  We
/// inspect the underlying `CFNumber` type to write floats with `.0`.
///
/// Output format mirrors Python's `json.dumps(obj, indent=4, sort_keys=False)`
/// as a target (Meshroom's exact format).  Specifically:
/// - 4-space indent, newlines between elements
/// - key ordering preserved as-given (we sort upstream in
///   `MGGraph.jsonObject()` to match Meshroom's deterministic sort)
/// - ASCII escapes for control characters
/// - floats emitted via Swift's `Double` printf-style description, with `.0`
///   appended for whole-number values
enum MGJSONWriter {
    static func write(_ root: [String: Any], indent: Int = 4) throws -> Data {
        var buf = String()
        try writeAny(root, into: &buf, level: 0, indent: indent)
        guard let data = buf.data(using: .utf8) else {
            throw MGProjectError.unexpectedJSONValue("UTF-8 encoding failed")
        }
        return data
    }

    private static func writeAny(_ value: Any, into buf: inout String, level: Int, indent: Int) throws {
        if let n = value as? NSNumber {
            writeNumber(n, into: &buf)
            return
        }
        if value is NSNull {
            buf.append("null")
            return
        }
        if let s = value as? String {
            writeString(s, into: &buf)
            return
        }
        if let arr = value as? [Any] {
            try writeArray(arr, into: &buf, level: level, indent: indent)
            return
        }
        if let obj = value as? [String: Any] {
            try writeObject(obj, into: &buf, level: level, indent: indent)
            return
        }
        throw MGProjectError.unexpectedJSONValue(String(describing: type(of: value)))
    }

    private static func writeNumber(_ n: NSNumber, into buf: inout String) {
        if CFGetTypeID(n) == CFBooleanGetTypeID() {
            buf.append(n.boolValue ? "true" : "false")
            return
        }
        let cfn = n as CFNumber
        if CFNumberIsFloatType(cfn) {
            let d = n.doubleValue
            if d.isNaN || d.isInfinite {
                // JSON cannot represent these; emit null (matches python json).
                buf.append("null")
                return
            }
            // Swift's default Double description gives e.g. "45.0" for 45.0,
            // "0.5" for 0.5, "1.5e-07" for small numbers.  This is the same
            // format Python's json.dumps produces for floats.
            let s = String(d)
            buf.append(s)
            // If the textual form is missing a decimal point (shouldn't happen
            // for Double but defensively), append ".0".  Currently
            // unreachable on Apple platforms — Swift always emits .0 for
            // whole-number doubles in `String(Double)`.
            if !s.contains(".") && !s.contains("e") && !s.contains("E") && !s.contains("n") {
                buf.append(".0")
            }
            return
        }
        buf.append(String(n.int64Value))
    }

    private static func writeString(_ s: String, into buf: inout String) {
        buf.append("\"")
        for ch in s.unicodeScalars {
            switch ch {
            case "\"":   buf.append("\\\"")
            case "\\":   buf.append("\\\\")
            case "\n":   buf.append("\\n")
            case "\r":   buf.append("\\r")
            case "\t":   buf.append("\\t")
            case "\u{08}": buf.append("\\b")
            case "\u{0C}": buf.append("\\f")
            default:
                if ch.value < 0x20 {
                    buf.append(String(format: "\\u%04x", ch.value))
                } else {
                    buf.unicodeScalars.append(ch)
                }
            }
        }
        buf.append("\"")
    }

    private static func writeArray(_ arr: [Any], into buf: inout String, level: Int, indent: Int) throws {
        if arr.isEmpty {
            buf.append("[]")
            return
        }
        buf.append("[\n")
        let inner = String(repeating: " ", count: (level + 1) * indent)
        let outer = String(repeating: " ", count: level * indent)
        for (i, v) in arr.enumerated() {
            buf.append(inner)
            try writeAny(v, into: &buf, level: level + 1, indent: indent)
            if i < arr.count - 1 {
                buf.append(",")
            }
            buf.append("\n")
        }
        buf.append(outer)
        buf.append("]")
    }

    private static func writeObject(_ obj: [String: Any], into buf: inout String, level: Int, indent: Int) throws {
        if obj.isEmpty {
            buf.append("{}")
            return
        }
        buf.append("{\n")
        let inner = String(repeating: " ", count: (level + 1) * indent)
        let outer = String(repeating: " ", count: level * indent)
        // Sort keys for deterministic output (matches Meshroom's behaviour
        // for the graph block and most internal dicts).
        let keys = obj.keys.sorted()
        for (i, k) in keys.enumerated() {
            buf.append(inner)
            writeString(k, into: &buf)
            buf.append(": ")
            try writeAny(obj[k]!, into: &buf, level: level + 1, indent: indent)
            if i < keys.count - 1 {
                buf.append(",")
            }
            buf.append("\n")
        }
        buf.append(outer)
        buf.append("}")
    }
}
