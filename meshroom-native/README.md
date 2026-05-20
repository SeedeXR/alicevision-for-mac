# meshroom-native

Swift Package Manager scaffold for the native macOS / SwiftUI replacement of
Meshroom's PyQt5/QML UI.  Tracks the M1–M4 roadmap defined in
`../memory/native_ui_design.md`.

This package contains **no UI code yet**.  The deliverable is a strictly
Foundation-only model layer that can read and write Meshroom's `.mg` project
files losslessly.

## Status: Phase 15 M1 scaffold

| Milestone | Deliverable | Status |
| --- | --- | --- |
| **M1** | `ProjectModel`: `.mg` round-trip + template ref parser | **DONE (this commit)** |
| M2 | `NodeSchema`: `--help-json` loader, schema-driven parameter forms | TODO |
| M3 | `Orchestrator`: actor-based Process runner with progress streaming | TODO |
| M4 | `GraphEditor`, `Inspector`, `Viewport3D`, `AssetLibrary` SwiftUI modules | TODO |

The full design rationale (107 node types, Apple Silicon Metal path, EDR, etc.)
lives in `../memory/native_ui_design.md`.

## Layout

```
meshroom-native/
├── Package.swift                  swift-tools 5.9, macOS 14+ baseline
├── Sources/
│   └── ProjectModel/              UI-free, Foundation-only
│       ├── MGProject.swift        top-level .mg representation + header
│       ├── MGGraph.swift          graph block (dict of nodes)
│       ├── MGNode.swift           one node + MGPosition + MGParallelization
│       ├── MGAttribute.swift      MGJSONValue recursive enum
│       └── MGTemplate.swift       {NodeName.attr} reference parser
└── Tests/
    └── ProjectModelTests/
        ├── RoundTripTests.swift   load+encode+re-decode for every fixture
        ├── TemplateParserTests.swift
        └── Fixtures/
            ├── appendTextAndFiles.mg    real upstream fixture (template, 3 nodes)
            ├── sharedTemplate.mg        real upstream fixture (empty template)
            └── photogrammetryMini.mg    synthetic, exercises uid/outputs/parallel/groups/CopyFiles
```

## Running the tests

```bash
cd /Users/alexmkwizu/Documents/SoftwareProjects/alicevision-mac/alicevision-for-mac/meshroom-native
swift test
```

There are no external dependencies — `swift test` builds and runs against the
toolchain that ships with the active Xcode (Swift 5.9+ required for the
`platforms:` syntax used in `Package.swift`).

## Schema coverage

The model mirrors the `.mg` schema as observed in
`upstream/meshroom-windows/meshroom/core/graphIO.py` (`GraphSerializer.serialize`,
`TemplateGraphSerializer.serializeNode`) and
`meshroom/core/node.py::BaseNode.toDict`:

| `.mg` field | Swift type | Notes |
| --- | --- | --- |
| `header.releaseVersion` | `String` | required |
| `header.fileVersion` | `String` | required, currently `"2.0"` |
| `header.nodesVersions` | `[String: String]` | per-node-type version |
| `header.template` | `Bool?` | absent on saved projects |
| `graph` | `MGGraph` (dict) | one entry per node |
| `graph.<Name>.nodeType` | `String` | required |
| `graph.<Name>.position` | `MGPosition?` (`[x, y]`) | stored as 2-element array; integer-preserving |
| `graph.<Name>.parallelization` | `MGParallelization?` | `{ blockSize, size, split }` |
| `graph.<Name>.uid` | `String?` | omitted in template form |
| `graph.<Name>.inputs` | `[String: MGJSONValue]?` | heterogeneous (literal / template ref / nested) |
| `graph.<Name>.internalInputs` | `[String: MGJSONValue]?` | label/color/comment |
| `graph.<Name>.outputs` | `[String: MGJSONValue]?` | omitted in template form |

`MGJSONValue` is a recursive enum with cases `null`, `bool`, `int`, `double`,
`string`, `array`, `object`.  Integers are kept distinct from doubles so
round-trips don't silently rewrite `0` as `0.0` *or* `45.0` as `45`.

Critically, the canonical decode/encode path goes through `JSONSerialization`
(read) and a hand-rolled `MGJSONWriter` (write) rather than `JSONEncoder` /
`JSONDecoder`.  The reason: `JSONEncoder` collapses `Double(45.0)` to the
textual form `"45"` (no decimal point), which would re-parse as
`MGJSONValue.int(45)` and break round-trip equality.  `MGJSONWriter` emits
`"45.0"` explicitly for any value that was originally a float, matching
Meshroom's own Python `json.dumps` output.  See the rationale at the top of
`Sources/ProjectModel/MGJSONWriter.swift`.

The `{NodeName.attr}` template reference syntax — including dotted sub-paths
like `{CameraInit_1.viewpoints.0.path}` — is parsed by `MGTemplateReference`.
The parser correctly rejects:

- `${ENV_VAR}` (env-var substitution; Meshroom passes these through verbatim)
- `{nodeCacheFolder}` (zero-arg placeholder, not a node reference)
- `{ Node.attr }` (whitespace not produced by upstream)
- `{1Node.output}` (invalid identifier)

## Explicitly NOT covered yet

These are documented now so M2 has a clean punchlist; none affect round-trip
fidelity because they are JSON-level pass-through values today.

- **Resolution of template references** against the graph (Phase M2; needed
  for the GraphEditor canvas).  Today we only parse the textual form.
- **`enabledWhen` / predicate language** on attributes (Phase M2/M3 — the
  schema where this lives is loaded from `aliceVision --help-json`, not from
  the `.mg` file).
- **CompatibilityNode flagging.** Meshroom marks nodes whose `nodesVersions`
  entry disagrees with the registered node desc.  We decode them as plain
  `MGNode`s — version comparison is a M2 responsibility.
- **`ui.json` sidecar** described in the design doc §1.3.  Not part of the
  Meshroom canonical format; will land as a separate `Sources/ProjectModel/UI*`
  type when M2 introduces the `.av` bundle.
- **Pretty-print formatting parity** with Meshroom's exact whitespace.  Our
  encoder uses `.sortedKeys + .prettyPrinted` (2-space indent) which is *not*
  byte-identical to Meshroom's 4-space output but is structurally equivalent
  and re-decodes cleanly.

## Fixture provenance

| Fixture | Source |
| --- | --- |
| `appendTextAndFiles.mg` | copied verbatim from `../meshroom-mac/tests/appendTextAndFiles.mg` (mirror of upstream Meshroom test) |
| `sharedTemplate.mg` | copied verbatim from `../meshroom-mac/tests/plugins/meshroom/sharedTemplate.mg` |
| `photogrammetryMini.mg` | **synthetic**, hand-authored to exercise the photogrammetry pipeline shape — `CopyFiles_1` framework node, nested group attribute (`Meshing_1.advanced.*`), list-of-dicts (`CopyFiles_1.inputFiles`), `uid`/`outputs`/`parallelization` (non-template form) |

Upstream Meshroom does not ship full `.mg` pipelines in its tree (the stock
pipelines live in the `meshroom_bin` install layout, not the source repo), so a
synthetic fixture is the only way to exercise the non-template encoding path at
this stage of the port.  When Phase 11 lands the real macOS Meshroom install,
swap `photogrammetryMini.mg` for a real `photogrammetry.mg` export from a
Monstree run.

## Deviations from the design doc

- The design doc §1.3 introduces a `NodeInput` enum with cases `.literal`,
  `.reference`, `.list`.  We instead store every input as `MGJSONValue` and
  expose `.asTemplateReference` as a *view* over the string case.  Reason:
  references can appear *inside* lists and groups (e.g. `featuresFolders:
  ["{FeatureExtraction_1.output}"]`), so the enum-at-the-top-level shape
  doesn't compose.  The chosen design lets a single recursive type handle every
  level uniformly.  Downstream code that wants strongly-typed edges can do a
  post-decode walk.

## Next concrete step toward M2 (node-graph editor UI)

1. Add `Sources/NodeSchema/` with a `NodeSchema` `Codable` struct matching the
   `--help-json` shape described in `native_ui_design.md` §2.4.
2. Implement an `MGGraph.edges()` derived view that walks every
   `MGJSONValue` in every node's `inputs`, locating `MGTemplateReference`s and
   yielding `(srcNode, srcAttr, dstNode, dstAttr)` tuples.  This is the
   integration point between the parser shipped here and the SwiftUI canvas.
3. Begin a `Sources/GraphEditor/` SwiftUI target with `NavigationSplitView`
   + a `Canvas`-based edges layer + plain `View`-per-node layout, driven by
   `@Observable` wrapping `MGProject`.
