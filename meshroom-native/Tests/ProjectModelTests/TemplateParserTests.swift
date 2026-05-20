import XCTest
@testable import ProjectModel

final class TemplateParserTests: XCTestCase {

    func testSimpleReference() {
        let ref = MGTemplateReference.parse("{Node_1.output}")
        XCTAssertEqual(ref?.nodeName, "Node_1")
        XCTAssertEqual(ref?.attributePath, ["output"])
        XCTAssertEqual(ref?.rendered, "{Node_1.output}")
    }

    func testFrameworkNodeReference() {
        // CopyFiles_1 lives in meshroom/core, not aliceVision/.  It still uses
        // the same {Name.attr} reference syntax.
        let ref = MGTemplateReference.parse("{CopyFiles_1.output}")
        XCTAssertEqual(ref?.nodeName, "CopyFiles_1")
        XCTAssertEqual(ref?.attributePath, ["output"])
    }

    func testDottedAttributePath() {
        let ref = MGTemplateReference.parse("{CameraInit_1.viewpoints.0.path}")
        XCTAssertEqual(ref?.nodeName, "CameraInit_1")
        XCTAssertEqual(ref?.attributePath, ["viewpoints", "0", "path"])
    }

    func testRejectsBareString() {
        XCTAssertNil(MGTemplateReference.parse("Node_1.output"))
        XCTAssertNil(MGTemplateReference.parse(""))
        XCTAssertNil(MGTemplateReference.parse("{}"))
    }

    func testRejectsEnvVarStyle() {
        // ${ALICEVISION_SENSOR_DB} starts with $, must not be parsed as a ref.
        XCTAssertNil(MGTemplateReference.parse("${ALICEVISION_SENSOR_DB}"))
    }

    func testRejectsNodeCacheFolder() {
        // {nodeCacheFolder} is a Meshroom-builtin placeholder, not a node
        // reference (no dot).  Must not parse.
        XCTAssertNil(MGTemplateReference.parse("{nodeCacheFolder}"))
    }

    func testRejectsWhitespaceInside() {
        XCTAssertNil(MGTemplateReference.parse("{ Node.attr }"))
    }

    func testRejectsInvalidIdentifier() {
        XCTAssertNil(MGTemplateReference.parse("{1Node.output}"))
    }

    func testMGJSONValueExposesReference() {
        let v = MGJSONValue.string("{Node_1.output}")
        let ref = v.asTemplateReference
        XCTAssertEqual(ref?.nodeName, "Node_1")
        // Non-string values cannot be references.
        XCTAssertNil(MGJSONValue.int(42).asTemplateReference)
        XCTAssertNil(MGJSONValue.null.asTemplateReference)
    }
}
