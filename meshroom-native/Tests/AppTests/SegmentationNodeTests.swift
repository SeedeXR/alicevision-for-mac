import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// SegmentationBiRefNet — Python-only AI segmentation node (rembg +
/// BiRefNet via ONNX Runtime + CoreML).  See
/// `instructions/ai_instruction.md` for the upstream design.
///
/// Unlike the 12 native `aliceVision_*` nodes, SegmentationBiRefNet
/// has no CLI binary.  Instead, the executor invokes
/// `scripts/run_python_node.sh`, which activates `meshroom-venv/` and
/// dispatches to a Python entry point with `--nodeType SegmentationBiRefNet`.
///
/// These tests pin the integration surface:
/// 1. The Spec is registered in `NodeBinary.specs`.
/// 2. Input / output type maps drive the M8 type-checker correctly.
/// 3. The palette lists the node so it can be dragged onto the canvas.
/// 4. The wrapper script exists, has the right shebang, and is executable
///    (so `Process.run()` doesn't fail with EPERM at runtime).
@MainActor
final class SegmentationNodeTests: XCTestCase {

    // MARK: - Spec registration

    func testSpecIsRegistered() {
        XCTAssertNotNil(
            NodeBinary.specs["SegmentationBiRefNet"],
            "SegmentationBiRefNet must be in NodeBinary.specs so the executor can run it"
        )
    }

    func testSpecPointsAtPythonWrapperNotAliceVisionBinary() {
        guard let spec = NodeBinary.spec(for: "SegmentationBiRefNet") else {
            return XCTFail("missing spec")
        }
        // The wrapper script is intentionally referenced as a path containing
        // `/` so `Process` resolves it relative to `binDir`, not as an
        // `aliceVision_*` binary in the same directory.
        XCTAssertTrue(
            spec.executable.contains("run_python_node.sh"),
            "executable should be the Python wrapper, got '\(spec.executable)'"
        )
        XCTAssertFalse(
            spec.executable.hasPrefix("aliceVision_"),
            "SegmentationBiRefNet has no aliceVision_* binary; got '\(spec.executable)'"
        )
    }

    func testSpecCarriesNodeTypeConstantFlag() {
        guard let spec = NodeBinary.spec(for: "SegmentationBiRefNet") else {
            return XCTFail("missing spec")
        }
        // The wrapper script disambiguates which Python node to invoke
        // by reading `--nodeType <Name>` from argv.  Without this, the
        // generic wrapper has no idea which Python class to dispatch to.
        XCTAssertTrue(
            spec.constantFlags.contains("--nodeType"),
            "constantFlags must include '--nodeType' so the wrapper can route to the right Python node"
        )
        XCTAssertTrue(
            spec.constantFlags.contains("SegmentationBiRefNet"),
            "constantFlags must include the node-type name as the value of --nodeType"
        )
    }

    func testSpecHasNoParallelization() {
        guard let spec = NodeBinary.spec(for: "SegmentationBiRefNet") else {
            return XCTFail("missing spec")
        }
        // The Python implementation loops over views in-process; we don't
        // want the Swift executor to fan out chunks at the Process level
        // because every chunk would have to re-import the (heavy) ONNX
        // model and re-warm the CoreML graph.  Single Process invocation.
        XCTAssertNil(
            spec.parallelization,
            "SegmentationBiRefNet must run as a single Process (per-image loop is Python-side)"
        )
    }

    func testSpecHasNoExpectedOutputFile() {
        guard let spec = NodeBinary.spec(for: "SegmentationBiRefNet") else {
            return XCTFail("missing spec")
        }
        // Per-view masks (e.g. `<viewId>_mask.png`) are keyed by SfMData
        // view IDs — there's no single canonical artefact to gate the
        // warm-cache on, so we leave expectedOutputFile nil and accept
        // that this node always re-runs unless every output path exists.
        XCTAssertNil(
            spec.expectedOutputFile,
            "Per-view mask outputs preclude a single canonical filename for warm-cache"
        )
    }

    // MARK: - Input / output type maps

    func testInputTypesCoverAllDeclaredInputFlags() {
        guard let spec = NodeBinary.spec(for: "SegmentationBiRefNet") else {
            return XCTFail("missing spec")
        }
        // Every declared inputFlag must have a type in inputTypes so the M8
        // type-checker can validate connections from upstream nodes.  An
        // empty map would force the type-checker into "unknown = ALLOW"
        // and let through e.g. a string-into-bool mistake.
        for flag in spec.inputFlags {
            XCTAssertNotNil(
                spec.inputTypes[flag],
                "inputTypes missing entry for inputFlag '\(flag)'"
            )
        }
    }

    func testInputTypesUseRecognisedTokens() {
        guard let spec = NodeBinary.spec(for: "SegmentationBiRefNet") else {
            return XCTFail("missing spec")
        }
        // The type-checker only understands a small set of strings — see
        // `NodeBinary.Spec.inputTypes` doc.  Catch typos like "float" or
        // "boolean" that would silently fall back to "unknown = ALLOW".
        let valid: Set<String> = [
            "file", "int", "double", "string", "bool",
            "fileArray", "intArray", "stringArray",
        ]
        for (name, typ) in spec.inputTypes {
            XCTAssertTrue(
                valid.contains(typ),
                "inputTypes['\(name)'] = '\(typ)' is not a recognised type token"
            )
        }
    }

    func testInputAttributeTypes() {
        guard let spec = NodeBinary.spec(for: "SegmentationBiRefNet") else {
            return XCTFail("missing spec")
        }
        XCTAssertEqual(spec.inputTypes["input"], "file",
                       "`input` is the upstream SfMData reference — must be a file")
        XCTAssertEqual(spec.inputTypes["modelVariant"], "string",
                       "`modelVariant` is a ChoiceParam (birefnet-general/dis/lite) — string")
        XCTAssertEqual(spec.inputTypes["outputResolution"], "string",
                       "`outputResolution` is a ChoiceParam (512/1024/2048) — string")
        XCTAssertEqual(spec.inputTypes["alphaMatting"], "bool",
                       "`alphaMatting` is a BoolParam — bool")
        XCTAssertEqual(spec.inputTypes["maskFormat"], "string",
                       "`maskFormat` chooses PNG/EXR — string")
    }

    func testOutputTypes() {
        guard let spec = NodeBinary.spec(for: "SegmentationBiRefNet") else {
            return XCTFail("missing spec")
        }
        XCTAssertEqual(spec.outputs, ["output"],
                       "single primary output exposes one pin on the right edge")
        XCTAssertEqual(spec.outputTypes["output"], "file",
                       "`output` is a folder of masks — modelled as 'file' like other folder outputs")
    }

    // MARK: - Palette listing

    func testPaletteListsSegmentationBiRefNet() {
        XCTAssertTrue(
            NodePalettePanel.allNodeTypes.contains("SegmentationBiRefNet"),
            "palette must list SegmentationBiRefNet so users can drag it onto the canvas"
        )
    }

    func testPaletteIconResolves() {
        let icon = NodePalettePanel.iconName(forType: "SegmentationBiRefNet")
        XCTAssertFalse(icon.isEmpty)
        XCTAssertNotEqual(
            icon, "questionmark.app",
            "SegmentationBiRefNet must have its own SF Symbol, not the unknown-type fallback"
        )
    }

    // MARK: - Wrapper script

    /// Resolve the wrapper script relative to the package root.  Walk
    /// upwards from this file until we find `Package.swift` so the test
    /// works whether SwiftPM placed the test binary under `.build/` or
    /// invoked it from a custom path.
    private func packageRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        while url.path != "/" {
            let candidate = url.appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: candidate.path) {
                return url
            }
            url.deleteLastPathComponent()
        }
        XCTFail("Could not locate Package.swift starting from \(#filePath)")
        return url
    }

    func testWrapperScriptExistsAtExpectedPath() {
        let script = packageRoot()
            .appendingPathComponent("scripts")
            .appendingPathComponent("run_python_node.sh")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: script.path),
            "wrapper script missing at \(script.path)"
        )
    }

    func testWrapperScriptIsExecutable() {
        let script = packageRoot()
            .appendingPathComponent("scripts")
            .appendingPathComponent("run_python_node.sh")
        XCTAssertTrue(
            FileManager.default.isExecutableFile(atPath: script.path),
            "wrapper script not executable at \(script.path) — run `chmod +x` after checkout"
        )
    }

    func testWrapperScriptStartsWithBashShebang() {
        let script = packageRoot()
            .appendingPathComponent("scripts")
            .appendingPathComponent("run_python_node.sh")
        guard let data = try? Data(contentsOf: script),
              let head = String(data: data.prefix(64), encoding: .utf8) else {
            return XCTFail("could not read wrapper script")
        }
        XCTAssertTrue(
            head.hasPrefix("#!/bin/bash") || head.hasPrefix("#!/usr/bin/env bash"),
            "wrapper script must start with a bash shebang; got: \(head.prefix(32))"
        )
    }
}
