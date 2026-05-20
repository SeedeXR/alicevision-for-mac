import XCTest
@testable import ProjectModel

final class RoundTripTests: XCTestCase {

    // MARK: - Fixture discovery

    /// Locate every `.mg` file shipped in the test bundle.
    ///
    /// Tries multiple discovery paths because SPM's resource layout varies
    /// across Swift toolchains:
    /// - `Bundle.module.url(forResource: "X", withExtension: "mg")` — works
    ///   when `.copy("Fixtures")` flattened the directory.
    /// - `Bundle.module.url(forResource: "Fixtures", withExtension: nil)` —
    ///   works when the directory was preserved verbatim.
    /// - `Bundle.module.resourceURL` + `/Fixtures` — last-resort manual probe.
    private func fixtureURLs() throws -> [URL] {
        let bundle = Bundle.module
        // First try the directory-preserving form.
        if let dir = bundle.url(forResource: "Fixtures", withExtension: nil),
           let attrs = try? FileManager.default.attributesOfItem(atPath: dir.path),
           attrs[.type] as? FileAttributeType == .typeDirectory {
            return try enumerateMG(in: dir)
        }
        // Then try resourceURL/Fixtures.
        if let resRoot = bundle.resourceURL {
            let candidate = resRoot.appendingPathComponent("Fixtures", isDirectory: true)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return try enumerateMG(in: candidate)
            }
            // Some SPM layouts flatten .copy() resources into the bundle root.
            // Detect by scanning for .mg files directly.
            let flat = try FileManager.default.contentsOfDirectory(
                at: resRoot, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
            )
            let mgs = flat.filter { $0.pathExtension == "mg" }
            if !mgs.isEmpty {
                return mgs.sorted { $0.lastPathComponent < $1.lastPathComponent }
            }
        }
        XCTFail("Fixtures directory not found in test bundle")
        return []
    }

    private func enumerateMG(in dir: URL) throws -> [URL] {
        let urls = try FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        return urls.filter { $0.pathExtension == "mg" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    // MARK: - Tests

    func testFixturesArePresent() throws {
        let fixtures = try fixtureURLs()
        XCTAssertGreaterThan(fixtures.count, 0, "no .mg fixtures shipped with the test target")
    }

    /// Load every shipped `.mg` file, encode it, decode the encoded form, and
    /// assert the two decoded representations are equal.  This catches both
    /// silent data loss on decode and structural drift on encode.
    func testRoundTripAllFixtures() throws {
        let fixtures = try fixtureURLs()
        XCTAssertGreaterThan(fixtures.count, 0, "no fixtures")
        for url in fixtures {
            let data = try Data(contentsOf: url)
            let project: MGProject
            do {
                project = try MGProject.decode(from: data)
            } catch {
                XCTFail("Failed to decode \(url.lastPathComponent): \(error)")
                continue
            }
            let encoded: Data
            do {
                encoded = try project.encode()
            } catch {
                XCTFail("Failed to encode \(url.lastPathComponent): \(error)")
                continue
            }
            let reDecoded: MGProject
            do {
                reDecoded = try MGProject.decode(from: encoded)
            } catch {
                XCTFail("Failed to re-decode encoded \(url.lastPathComponent): \(error)")
                continue
            }
            XCTAssertEqual(
                project, reDecoded,
                "Round-trip mismatch for \(url.lastPathComponent)"
            )
        }
    }

    /// Sanity check on the structural minimum: every fixture has a header with
    /// a fileVersion and a (possibly empty) graph.
    func testFixtureHeadersPopulated() throws {
        let fixtures = try fixtureURLs()
        for url in fixtures {
            let project = try MGProject.load(from: url)
            XCTAssertFalse(
                project.header.fileVersion.isEmpty,
                "fileVersion missing in \(url.lastPathComponent)"
            )
            XCTAssertFalse(
                project.header.releaseVersion.isEmpty,
                "releaseVersion missing in \(url.lastPathComponent)"
            )
        }
    }

    /// Fixture-specific assertions to lock down the schema interpretation.
    func testAppendTextAndFilesFixtureShape() throws {
        let url = try XCTUnwrap(try fixtureURLs().first { $0.lastPathComponent == "appendTextAndFiles.mg" })
        let project = try MGProject.load(from: url)
        XCTAssertEqual(project.header.fileVersion, "2.0")
        XCTAssertEqual(project.header.template, true)
        XCTAssertEqual(project.graph.count, 3)
        let appendFiles = try XCTUnwrap(project.graph["AppendFiles_1"])
        XCTAssertEqual(appendFiles.nodeType, "AppendFiles")
        XCTAssertEqual(appendFiles.position?.x, 189)
        XCTAssertEqual(appendFiles.position?.y, 8)
        let input1 = try XCTUnwrap(appendFiles.inputs?["input"])
        let ref = try XCTUnwrap(input1.asTemplateReference)
        XCTAssertEqual(ref.nodeName, "AppendText_1")
        XCTAssertEqual(ref.attributePath, ["output"])
    }

    func testEmptyTemplateFixture() throws {
        let url = try XCTUnwrap(try fixtureURLs().first { $0.lastPathComponent == "sharedTemplate.mg" })
        let project = try MGProject.load(from: url)
        XCTAssertEqual(project.graph.count, 0)
        XCTAssertEqual(project.header.template, true)
    }

    func testPhotogrammetryMiniFixture() throws {
        let url = try XCTUnwrap(try fixtureURLs().first { $0.lastPathComponent == "photogrammetryMini.mg" })
        let project = try MGProject.load(from: url)
        // Has both framework (CopyFiles) and AliceVision nodes.
        XCTAssertNotNil(project.graph["CopyFiles_1"])
        XCTAssertNotNil(project.graph["FeatureExtraction_1"])
        // Non-template form: uid + outputs + parallelization populated.
        let fe = try XCTUnwrap(project.graph["FeatureExtraction_1"])
        XCTAssertNotNil(fe.uid)
        XCTAssertEqual(fe.parallelization?.blockSize, 40)
        XCTAssertNotNil(fe.outputs?["output"])
        // Group attribute (nested dict) survives.
        let meshing = try XCTUnwrap(project.graph["Meshing_1"])
        guard case .object(let advanced) = meshing.inputs?["advanced"] ?? .null else {
            XCTFail("advanced should be a nested object")
            return
        }
        XCTAssertEqual(advanced["helperPointsGridSize"], .int(10))
        // double 20.0 must survive as .double, not be downgraded to .int(20).
        XCTAssertEqual(advanced["densifyScale"], .double(20.0))
        // List of dicts inside CopyFiles_1.inputFiles round-trips.
        guard case .array(let copyList) = project.graph["CopyFiles_1"]?.inputs?["inputFiles"] ?? .null else {
            XCTFail("inputFiles should be an array")
            return
        }
        XCTAssertEqual(copyList.count, 1)
        // CameraInit_1 has a float (defaultFieldOfView: 45.0).  Ensure the
        // int-vs-double distinction is preserved.
        let cam = try XCTUnwrap(project.graph["CameraInit_1"])
        XCTAssertEqual(cam.inputs?["defaultFieldOfView"], .double(45.0))
    }
}
