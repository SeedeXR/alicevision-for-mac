import Foundation

/// A single entry inside `graph` of a `.mg` file.
///
/// Source of truth: `meshroom/core/node.py::BaseNode.toDict()` (Meshroom 2025.1).
/// Observed shape:
/// ```
/// {
///   "nodeType": "FeatureExtraction",
///   "position": [189, 8],
///   "parallelization": { "blockSize": 0, "size": 0, "split": 0 },  // optional
///   "uid": "abcdef...",                                            // optional
///   "inputs":         { "input": "{CameraInit_1.output}", ... },    // optional
///   "internalInputs": { "label": "..." },                           // optional
///   "outputs":        { "output": "{nodeCacheFolder}/..." }         // optional
/// }
/// ```
///
/// Templates (`header.template == true`) intentionally omit `uid`, `outputs`,
/// and `parallelization`; we model those as optional to round-trip both forms.
public struct MGNode: Hashable, Sendable {
    public var nodeType: String
    public var position: MGPosition?
    public var parallelization: MGParallelization?
    public var uid: String?
    /// Input attribute values, keyed by attribute name.  Values may be
    /// literals (any JSON type) or template references like `"{Node_1.out}"`.
    public var inputs: [String: MGJSONValue]?
    /// Computed-at-runtime input attributes (Meshroom's `internalAttributes`).
    /// Includes node label, color, comment, etc.
    public var internalInputs: [String: MGJSONValue]?
    /// Output attribute values, keyed by attribute name.  Templates omit this.
    public var outputs: [String: MGJSONValue]?

    public init(
        nodeType: String,
        position: MGPosition? = nil,
        parallelization: MGParallelization? = nil,
        uid: String? = nil,
        inputs: [String: MGJSONValue]? = nil,
        internalInputs: [String: MGJSONValue]? = nil,
        outputs: [String: MGJSONValue]? = nil
    ) {
        self.nodeType = nodeType
        self.position = position
        self.parallelization = parallelization
        self.uid = uid
        self.inputs = inputs
        self.internalInputs = internalInputs
        self.outputs = outputs
    }

    // MARK: - JSONSerialization bridge

    init(jsonObject obj: [String: Any]) throws {
        guard let nodeType = obj["nodeType"] as? String else {
            throw MGProjectError.missingField("nodeType")
        }
        self.nodeType = nodeType
        self.position = try MGPosition(jsonAny: obj["position"])
        self.parallelization = MGParallelization(jsonAny: obj["parallelization"])
        self.uid = obj["uid"] as? String
        self.inputs = try Self.decodeAttrDict(obj["inputs"])
        self.internalInputs = try Self.decodeAttrDict(obj["internalInputs"])
        self.outputs = try Self.decodeAttrDict(obj["outputs"])
    }

    private static func decodeAttrDict(_ any: Any?) throws -> [String: MGJSONValue]? {
        guard let any = any else { return nil }
        guard let dict = any as? [String: Any] else {
            throw MGProjectError.unexpectedJSONValue("attribute dict not an object")
        }
        var out: [String: MGJSONValue] = [:]
        out.reserveCapacity(dict.count)
        for (k, v) in dict {
            out[k] = try MGJSONValue(jsonSerializationValue: v)
        }
        return out
    }

    func jsonObject() -> [String: Any] {
        var out: [String: Any] = [
            "nodeType": nodeType,
        ]
        if let position { out["position"] = position.jsonAny() }
        if let parallelization { out["parallelization"] = parallelization.jsonObject() }
        if let uid { out["uid"] = uid }
        if let inputs { out["inputs"] = inputs.mapValues { $0.jsonSerializationValue } }
        if let internalInputs { out["internalInputs"] = internalInputs.mapValues { $0.jsonSerializationValue } }
        if let outputs { out["outputs"] = outputs.mapValues { $0.jsonSerializationValue } }
        return out
    }
}

/// Graph-canvas coordinates of a node, stored as `[x, y]` integer pixels.
///
/// Meshroom uses a 2-element JSON array, not an object — see
/// `node.py::Node._position` and `graphIO.py`.  We expose `x`/`y` accessors
/// while still encoding as the array form.
public struct MGPosition: Hashable, Sendable {
    public var x: Double
    public var y: Double
    /// Whether the source JSON used integer literals (`[189, 8]`) or doubles
    /// (`[189.5, 8.0]`).  Preserved across round-trip.
    public var integerStorage: Bool

    public init(x: Double, y: Double, integerStorage: Bool = true) {
        self.x = x
        self.y = y
        self.integerStorage = integerStorage
    }

    init?(jsonAny any: Any?) throws {
        guard let any = any, !(any is NSNull) else { return nil }
        guard let arr = any as? [Any], arr.count >= 2 else {
            throw MGProjectError.unexpectedJSONValue("position not a 2-element array")
        }
        let xNum = arr[0] as? NSNumber
        let yNum = arr[1] as? NSNumber
        guard let xN = xNum, let yN = yNum else {
            throw MGProjectError.unexpectedJSONValue("position elements not numeric")
        }
        let xIsFloat = CFNumberIsFloatType(xN as CFNumber)
        let yIsFloat = CFNumberIsFloatType(yN as CFNumber)
        self.x = xN.doubleValue
        self.y = yN.doubleValue
        self.integerStorage = !(xIsFloat || yIsFloat)
    }

    func jsonAny() -> Any {
        if integerStorage {
            return [NSNumber(value: Int64(x)), NSNumber(value: Int64(y))]
        }
        return [NSNumber(value: x), NSNumber(value: y)]
    }
}

/// Parallel-execution metadata Meshroom writes for non-template graphs.
/// All three values default to 0 for non-parallelized nodes.
public struct MGParallelization: Hashable, Sendable {
    public var blockSize: Int
    public var size: Int
    public var split: Int

    public init(blockSize: Int = 0, size: Int = 0, split: Int = 0) {
        self.blockSize = blockSize
        self.size = size
        self.split = split
    }

    init?(jsonAny any: Any?) {
        guard let any = any, !(any is NSNull) else { return nil }
        guard let dict = any as? [String: Any] else { return nil }
        self.blockSize = (dict["blockSize"] as? NSNumber)?.intValue ?? 0
        self.size = (dict["size"] as? NSNumber)?.intValue ?? 0
        self.split = (dict["split"] as? NSNumber)?.intValue ?? 0
    }

    func jsonObject() -> [String: Any] {
        [
            "blockSize": NSNumber(value: blockSize),
            "size": NSNumber(value: size),
            "split": NSNumber(value: split),
        ]
    }
}
