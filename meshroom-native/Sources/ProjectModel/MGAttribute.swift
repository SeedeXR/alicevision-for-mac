import Foundation

/// A recursive, fully-typed JSON value used as the storage for the heterogeneous
/// `inputs`, `internalInputs`, and `outputs` dictionaries inside a Meshroom node.
///
/// `int` and `double` are kept as separate cases because the Meshroom on-disk
/// format distinguishes integer parameters (`maxNbFeatures: 0`) from float
/// parameters (`relativeFocalLength: 0.5`).  Collapsing them would round-trip
/// the format losslessly *in value* but rewrites every `45.0` as `45`, which
/// makes diffs against the original `.mg` and Meshroom interop noisy.
///
/// `null` is its own case so an explicit `"attr": null` in the JSON survives a
/// round-trip rather than being dropped to `nil`.
public indirect enum MGJSONValue: Hashable, Sendable {
    case null
    case bool(Bool)
    case int(Int64)
    case double(Double)
    case string(String)
    case array([MGJSONValue])
    case object([String: MGJSONValue])

    /// Build an `MGJSONValue` from the `Any` tree produced by
    /// `JSONSerialization.jsonObject(with:)`.  This is the *only* supported
    /// construction path that preserves the int-vs-double distinction:
    /// `JSONSerialization` returns `NSNumber` for numerics, and `NSNumber`
    /// records the originating CFNumber type so we can tell `42` from `42.0`.
    public init(jsonSerializationValue value: Any) throws {
        // Order matters: NSNumber bridges to many Swift types, so we must
        // dispatch on NSNumber identity (and its CFNumber sub-type) first.
        if let n = value as? NSNumber {
            // CFBooleanRef is a distinct CFType from CFNumberRef but bridges
            // to NSNumber.  We disambiguate by comparing the underlying type
            // identifier with kCFBooleanTypeID.
            if CFGetTypeID(n) == CFBooleanGetTypeID() {
                self = .bool(n.boolValue)
                return
            }
            let cfn = n as CFNumber
            if CFNumberIsFloatType(cfn) {
                self = .double(n.doubleValue)
            } else {
                self = .int(n.int64Value)
            }
            return
        }
        if value is NSNull {
            self = .null
            return
        }
        if let s = value as? String {
            self = .string(s)
            return
        }
        if let arr = value as? [Any] {
            self = .array(try arr.map { try MGJSONValue(jsonSerializationValue: $0) })
            return
        }
        if let obj = value as? [String: Any] {
            var out: [String: MGJSONValue] = [:]
            out.reserveCapacity(obj.count)
            for (k, v) in obj {
                out[k] = try MGJSONValue(jsonSerializationValue: v)
            }
            self = .object(out)
            return
        }
        throw MGProjectError.unexpectedJSONValue(String(describing: type(of: value)))
    }

    /// Convert back to the `Any` tree expected by
    /// `JSONSerialization.data(withJSONObject:)`.
    public var jsonSerializationValue: Any {
        switch self {
        case .null:
            return NSNull()
        case .bool(let b):
            return NSNumber(value: b)
        case .int(let i):
            return NSNumber(value: i)
        case .double(let d):
            return NSNumber(value: d)
        case .string(let s):
            return s
        case .array(let arr):
            return arr.map { $0.jsonSerializationValue }
        case .object(let obj):
            var out: [String: Any] = [:]
            out.reserveCapacity(obj.count)
            for (k, v) in obj {
                out[k] = v.jsonSerializationValue
            }
            return out
        }
    }
}

// MARK: - Codable

/// `Codable` conformance lets `MGJSONValue` participate in synthesised
/// `Codable` for `MGNode`/`MGProject`.  For numbers we lose the int-vs-double
/// distinction on this path because the public `Decoder` API hides the
/// underlying `NSNumber` type — so the canonical decode/encode entry points
/// on `MGProject` go through `JSONSerialization` instead (see `MGProject`).
extension MGJSONValue: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
            return
        }
        if let b = try? container.decode(Bool.self) {
            self = .bool(b)
            return
        }
        if let i = try? container.decode(Int64.self) {
            self = .int(i)
            return
        }
        if let d = try? container.decode(Double.self) {
            self = .double(d)
            return
        }
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        if let arr = try? container.decode([MGJSONValue].self) {
            self = .array(arr)
            return
        }
        if let obj = try? container.decode([String: MGJSONValue].self) {
            self = .object(obj)
            return
        }
        throw DecodingError.dataCorruptedError(
            in: container,
            debugDescription: "MGJSONValue could not decode value"
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let b):
            try container.encode(b)
        case .int(let i):
            try container.encode(i)
        case .double(let d):
            try container.encode(d)
        case .string(let s):
            try container.encode(s)
        case .array(let arr):
            try container.encode(arr)
        case .object(let obj):
            try container.encode(obj)
        }
    }
}

// MARK: - Template extraction

public extension MGJSONValue {
    /// If this value is a string of the form `{NodeName.attr}`, return the
    /// parsed reference; otherwise nil.  Lists and dicts are *not* recursed —
    /// callers that need a deep scan should walk the structure themselves.
    var asTemplateReference: MGTemplateReference? {
        guard case let .string(s) = self else { return nil }
        return MGTemplateReference.parse(s)
    }
}

/// Errors thrown by the model layer.  Kept in this file because
/// `MGJSONValue` is the only type that can throw outside of the standard
/// `DecodingError` / `EncodingError` flows.
public enum MGProjectError: Error, CustomStringConvertible {
    case unexpectedJSONValue(String)
    case invalidTopLevel
    case missingField(String)

    public var description: String {
        switch self {
        case .unexpectedJSONValue(let t):
            return "Unexpected JSON value of type \(t)"
        case .invalidTopLevel:
            return "Top-level JSON is not an object"
        case .missingField(let f):
            return "Missing required field: \(f)"
        }
    }
}
