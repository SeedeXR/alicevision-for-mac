import XCTest
import ProjectModel
@testable import MeshroomNativeApp

/// S53 — plugin system: tests that the manifest-driven discovery pipeline
/// (`NodeBinary.loadPluginSpecs()` + `PluginRegistry`) finds the bundled
/// `ai-segmentation` plugin and that the resulting `Spec` matches the
/// shape the previously-hardcoded `SegmentationBiRefNet` entry had.
///
/// These tests pin the contract that lets third-party plugins ship
/// `plugin.json` manifests without patching the Swift binary: any
/// regression here means a new plugin author cannot rely on the system.
@MainActor
final class PluginRegistryTests: XCTestCase {

    // MARK: - Discovery

    func testLoadPluginSpecsDiscoversAiSegmentation() {
        // `loadPluginSpecs()` returns the merged plugin->Spec map; calling
        // it also repopulates `PluginRegistry.shared` as a side effect.
        let plugins = NodeBinary.loadPluginSpecs()
        XCTAssertNotNil(
            plugins["SegmentationBiRefNet"],
            "ai-segmentation plugin must register SegmentationBiRefNet"
        )
        let manifest = PluginRegistry.shared.manifest(named: "ai-segmentation")
        XCTAssertNotNil(
            manifest,
            "PluginRegistry must hold the ai-segmentation manifest after discovery"
        )
        XCTAssertEqual(manifest?.version, "1.0.0",
                       "plugin manifest version must round-trip through JSON")
    }

    func testMergedSpecsExposesSegmentationBiRefNet() {
        // `NodeBinary.specs` is the union of `coreSpecs` + plugin specs.
        // Tests, palette, and executor all read from this union.
        XCTAssertNotNil(
            NodeBinary.specs["SegmentationBiRefNet"],
            "merged specs must contain plugin-supplied SegmentationBiRefNet"
        )
        // Sanity check that the existing 12 native binaries are still there.
        XCTAssertNotNil(NodeBinary.specs["CameraInit"])
        XCTAssertNotNil(NodeBinary.specs["Texturing"])
    }

    // MARK: - Spec shape

    func testLoadedSpecMatchesPreviouslyHardcodedShape() {
        // The hardcoded entry that was removed had:
        //   executable contains "run_python_node.sh"
        //   constantFlags: ["--nodeType", "SegmentationBiRefNet"]
        //   outputs: ["output"]
        //   parallelization == nil
        //   expectedOutputFile == nil
        // We pin every one of those so a future plugin.json edit can't
        // silently break the executor contract.
        guard let spec = NodeBinary.spec(for: "SegmentationBiRefNet") else {
            return XCTFail("plugin spec missing")
        }
        XCTAssertTrue(spec.executable.contains("run_python_node.sh"),
                      "plugin spec executable must point at the Python wrapper")
        XCTAssertTrue(spec.constantFlags.contains("--nodeType"),
                      "plugin spec must carry --nodeType so the wrapper routes correctly")
        XCTAssertTrue(spec.constantFlags.contains("SegmentationBiRefNet"),
                      "plugin spec must include the node type name as the --nodeType value")
        XCTAssertEqual(spec.outputs, ["output"])
        XCTAssertNil(spec.parallelization,
                     "plugin spec must not enable Process-level chunking")
        XCTAssertNil(spec.expectedOutputFile,
                     "plugin spec must leave expectedOutputFile nil — per-view masks")

        // Input types must match the M8 type-checker's vocabulary.
        let valid: Set<String> = [
            "file", "int", "double", "string", "bool",
            "fileArray", "intArray", "stringArray",
        ]
        for (name, typ) in spec.inputTypes {
            XCTAssertTrue(valid.contains(typ),
                          "inputTypes['\(name)'] = '\(typ)' is not recognised")
        }
        XCTAssertEqual(spec.inputTypes["input"], "file")
        XCTAssertEqual(spec.inputTypes["modelVariant"], "string")
        XCTAssertEqual(spec.inputTypes["alphaMatting"], "bool")
        XCTAssertEqual(spec.outputTypes["output"], "file")
    }

    // MARK: - Icon and category from manifest

    func testIconComesFromManifest() {
        // The palette icon for plugin-supplied nodes is read from the
        // manifest's `icon` field, not from any hardcoded switch in
        // NodePalettePanel.swift.  This pins that contract so a plugin
        // author can change the icon by editing plugin.json alone.
        let icon = NodePalettePanel.iconName(forType: "SegmentationBiRefNet")
        XCTAssertEqual(
            icon, "wand.and.stars.inverse",
            "icon must come from ai-segmentation plugin.json"
        )
        let manifest = PluginRegistry.shared.manifest(named: "ai-segmentation")
        let node = manifest?.nodes.first(where: { $0.name == "SegmentationBiRefNet" })
        XCTAssertEqual(node?.icon, "wand.and.stars.inverse")
        XCTAssertEqual(node?.category, "Utils",
                       "category must be read from manifest, not hardcoded")
    }

    // MARK: - Core specs still intact

    func testRemovingHardcodedEntryDidNotShrinkCoreSpecs() {
        // The 12 native AliceVision binaries must still live in `coreSpecs`.
        // SegmentationBiRefNet must NOT — it now arrives via the plugin.
        XCTAssertEqual(
            NodeBinary.coreSpecs.count, 12,
            "coreSpecs must hold exactly the 12 native aliceVision_* binaries"
        )
        XCTAssertNil(
            NodeBinary.coreSpecs["SegmentationBiRefNet"],
            "SegmentationBiRefNet must NOT be in coreSpecs — it's a plugin node"
        )
    }
}
