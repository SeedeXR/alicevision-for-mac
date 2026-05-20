import SwiftUI
import ProjectModel

/// View shown once a project is loaded.  Hosts the canvas and a right-side
/// inspector populated when the user selects a node.
///
/// We use `HSplitView` rather than the newer `.inspector` modifier because
/// `.inspector` was added in macOS 14.0 *but* its behaviour (auto-show on
/// selection, gear icon, etc.) is intentionally more opinionated than what the
/// M2 viewer needs.  `HSplitView` gives a plain resizable split we can fully
/// control.
///
/// M3 introduces editing.  The view now owns a `ProjectViewModel` (a reference
/// type, see comment in that file for why) and forwards Cmd-S to it.  The
/// hidden Save button is wired with `.keyboardShortcut` so users get the
/// expected macOS Save key even before we wrap the menu via `CommandGroup`
/// at the App scene level.
struct ProjectView: View {
    /// `@StateObject` ownership: this view *creates* the view-model and is
    /// responsible for its lifecycle.  Using `@StateObject` (rather than
    /// `@ObservedObject`) prevents SwiftUI from reconstructing the model on
    /// every parent re-render, which would lose the dirty flag and undo state.
    @StateObject private var viewModel: ProjectViewModel
    @State private var selectedNodeName: String?
    @Environment(\.undoManager) private var undoManager

    init(project: MGProject, projectURL: URL?) {
        // `_viewModel = StateObject(wrappedValue:)` is the documented escape
        // hatch for initialising a `@StateObject` with constructor-injected
        // data.  Constructing `ProjectViewModel(project:projectURL:)` in the
        // view's `init` runs at most once per view identity.
        _viewModel = StateObject(wrappedValue: ProjectViewModel(
            project: project,
            projectURL: projectURL
        ))
    }

    var body: some View {
        HSplitView {
            // M9: leftmost column is the node-creation palette.  Sandwiching
            // it inside the same HSplitView (rather than overlaying it) lets
            // the user resize the column down to a few pixels if they want
            // more canvas room — matching every other docked-tools pattern
            // in macOS pro apps.
            NodePalettePanel(viewModel: viewModel)
                .frame(minWidth: 140, idealWidth: 180, maxWidth: 260)
            // Centre: canvas + pipeline panel below it.  VStack lets us
            // reserve the bottom strip for the pipeline regardless of
            // node-selection state on the right.
            VStack(spacing: 0) {
                NodeGraphCanvas(
                    viewModel: viewModel,
                    selectedNodeName: $selectedNodeName
                )
                .frame(minWidth: 400, minHeight: 300)
                if !viewModel.nodeStatuses.isEmpty {
                    Divider()
                    PipelinePanel(viewModel: viewModel)
                        .frame(maxHeight: 180)
                }
            }
            inspector
                .frame(minWidth: 260, idealWidth: 320, maxWidth: 480)
        }
        .navigationTitle(navigationTitle)
        // Hidden button: gives us a Cmd-S keyboard shortcut without needing
        // to declare a Commands scene.  SwiftUI binds the shortcut to whatever
        // button is in scope; placing the button inside `ProjectView` means
        // the shortcut is only live while a project is open, which is the
        // correct semantic (no save with no project).
        .background(
            Button("Save", action: { viewModel.saveProject() })
                .keyboardShortcut("s", modifiers: .command)
                .disabled(!viewModel.isDirty || viewModel.projectURL == nil)
                .hidden()
        )
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                runOrStopButton
            }
        }
        .alert(
            "Could not save project",
            isPresented: Binding(
                get: { viewModel.saveError != nil },
                set: { if !$0 { viewModel.saveError = nil } }
            ),
            actions: { Button("OK", role: .cancel) {} },
            message: { Text(viewModel.saveError ?? "Unknown error") }
        )
        // Failure sheet: when a node fails, surface the stderr tail so the
        // user can debug without digging into the log dir.
        .sheet(
            isPresented: Binding(
                get: { viewModel.failedNodeLog != nil && !viewModel.isRunning },
                set: { if !$0 { viewModel.failedNodeLog = nil; viewModel.failedNode = nil } }
            )
        ) {
            failureSheet
        }
    }

    /// Toolbar button that flips between "Run" and "Stop" based on
    /// `isRunning`.  Disabled until the project has both a binDir and root
    /// resolvable from the project URL.
    @ViewBuilder
    private var runOrStopButton: some View {
        if viewModel.isRunning {
            Button(role: .destructive) {
                viewModel.cancelPipeline()
            } label: {
                Label("Stop", systemImage: "stop.fill")
            }
            .help("Stop the running pipeline")
        } else {
            Button {
                guard let config = makeRunConfig() else { return }
                viewModel.runPipeline(config: config)
            } label: {
                Label("Run", systemImage: "play.fill")
            }
            .disabled(makeRunConfig() == nil)
            .help("Execute the graph through the native aliceVision binaries")
        }
    }

    /// Failure diagnostics sheet.  Shows the failed node's name + the captured
    /// log tail; user can copy the text and dismiss.
    private var failureSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "exclamationmark.octagon.fill")
                    .foregroundStyle(.red)
                Text("Pipeline failed")
                    .font(.headline)
                if let name = viewModel.failedNode {
                    Text("at \(name)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }
            ScrollView {
                Text(viewModel.failedNodeLog ?? "")
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
                    .background(Color(NSColor.textBackgroundColor))
            }
            .frame(minWidth: 540, minHeight: 240)
            HStack {
                Spacer()
                Button("Close") {
                    viewModel.failedNodeLog = nil
                    viewModel.failedNode = nil
                }
                .keyboardShortcut(.cancelAction)
            }
        }
        .padding()
        .frame(idealWidth: 600, idealHeight: 360)
    }

    /// Resolve the executor configuration from the project's directory layout.
    ///
    /// Convention (matching `scripts/run_meshroom.sh`):
    /// - The project file sits anywhere; we treat its parent as the working
    ///   directory and look for `build/aliceVision_*` siblings relative to
    ///   a few candidate roots.  In production builds the user will be able
    ///   to set these via Preferences (M6); for M5 we walk up the dir tree
    ///   looking for a `build/` containing `aliceVision_cameraInit`.
    private func makeRunConfig() -> GraphExecutor.RunConfig? {
        guard let projectURL = viewModel.projectURL else { return nil }
        let workDir = projectURL.deletingLastPathComponent()
        // Walk up from the project until we find the build/ siblings.
        var probe = workDir
        for _ in 0..<6 {
            let buildDir = probe.appendingPathComponent("build", isDirectory: true)
            let probeBin = buildDir.appendingPathComponent("aliceVision_cameraInit")
            if FileManager.default.fileExists(atPath: probeBin.path) {
                let avRoot = buildDir.appendingPathComponent("alicevision_root", isDirectory: true)
                return GraphExecutor.RunConfig(
                    workDir: workDir,
                    binDir: buildDir,
                    aliceVisionRoot: FileManager.default.fileExists(atPath: avRoot.path)
                        ? avRoot : buildDir
                )
            }
            let parent = probe.deletingLastPathComponent()
            if parent.path == probe.path { break }
            probe = parent
        }
        return nil
    }

    /// macOS convention: append an "— Edited" marker (or the leading bullet
    /// in the close button) when the document has unsaved changes.  We use
    /// the text form because `WindowGroup` doesn't expose
    /// `documentBrowserStyle` document state.
    private var navigationTitle: String {
        let base = viewModel.projectURL?.lastPathComponent ?? "Untitled project"
        return viewModel.isDirty ? "\(base) — Edited" : base
    }

    // MARK: - Inspector

    @ViewBuilder
    private var inspector: some View {
        if let name = selectedNodeName, let node = viewModel.project.graph[name] {
            NodeInspector(
                nodeName: name,
                node: node,
                viewModel: viewModel,
                undoManager: undoManager
            )
            // Re-create the inspector on selection change so its `@State`
            // (in-flight edit buffers) resets cleanly between nodes — otherwise
            // typing in one node's `focalLength` then clicking another would
            // carry the stale edit buffer across.
            .id(name)
        } else {
            VStack {
                Spacer()
                Text("Select a node")
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

/// Right-side panel showing the attributes of the selected node.
///
/// M4 makes input attribute leaves editable in-place.  Each scalar value
/// (bool/int/double/string) gets a type-aware editor; arrays and dicts
/// remain read-only for this milestone (nested editing is M5+).  Template
/// references stay editable as raw strings — structured connection editing
/// also lives in M5+.
///
/// `viewModel` is the same project view-model the canvas drives; passing it
/// through (rather than re-resolving from the environment) keeps the editor
/// rows pure functions of `(key, value)` for testability later.
struct NodeInspector: View {
    let nodeName: String
    let node: MGNode
    @ObservedObject var viewModel: ProjectViewModel
    let undoManager: UndoManager?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                Divider()
                inputsSection
                if let outputs = node.outputs, !outputs.isEmpty {
                    Divider()
                    outputsSection(outputs)
                }
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(nodeName)
                .font(.title3)
                .bold()
            Text(node.nodeType)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if let pos = node.position {
                Text("Position: (\(Int(pos.x)), \(Int(pos.y)))")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private var inputsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Inputs")
                .font(.headline)
            if let inputs = node.inputs, !inputs.isEmpty {
                ForEach(inputs.keys.sorted(), id: \.self) { key in
                    if let v = inputs[key] {
                        AttributeRow(
                            attrKey: key,
                            attrPath: key,
                            value: v,
                            editable: true,
                            nodeName: nodeName,
                            viewModel: viewModel,
                            undoManager: undoManager
                        )
                    }
                }
            } else {
                Text("(no inputs)")
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func outputsSection(_ outputs: [String: MGJSONValue]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Outputs")
                .font(.headline)
            ForEach(outputs.keys.sorted(), id: \.self) { key in
                if let v = outputs[key] {
                    // Outputs are computed by the running graph engine; they
                    // are never user-editable, so we pass `editable: false`
                    // to render the read-only display form.
                    AttributeRow(
                        attrKey: key,
                        attrPath: key,
                        value: v,
                        editable: false,
                        nodeName: nodeName,
                        viewModel: viewModel,
                        undoManager: undoManager
                    )
                }
            }
        }
    }
}

/// Single row in the inspector: key + (editor or read-only label) + optional
/// `link` badge for template references.  Pulled out of `NodeInspector` so the
/// in-flight edit buffer (`@State editText`) lives on a per-row View identity
/// — selecting a different node tears down its rows and rebuilds them, which
/// is the simplest way to keep edit buffers from leaking between selections.
struct AttributeRow: View {
    let attrKey: String
    let attrPath: String
    let value: MGJSONValue
    let editable: Bool
    let nodeName: String
    @ObservedObject var viewModel: ProjectViewModel
    let undoManager: UndoManager?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text(attrKey)
                    .font(.system(.body, design: .monospaced))
                    .bold()
                if value.asTemplateReference != nil {
                    Text("link")
                        .font(.caption2)
                        .padding(.horizontal, 4)
                        .padding(.vertical, 1)
                        .background(Color.accentColor.opacity(0.2))
                        .clipShape(RoundedRectangle(cornerRadius: 3))
                }
            }
            editorOrReadOnly
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var editorOrReadOnly: some View {
        if editable {
            switch value {
            case .bool(let b):
                BoolEditor(
                    initial: b,
                    onCommit: { newBool in commit(.bool(newBool)) }
                )
            case .int(let i):
                IntEditor(
                    initial: i,
                    onCommit: { newInt in commit(.int(newInt)) }
                )
            case .double(let d):
                DoubleEditor(
                    initial: d,
                    onCommit: { newDouble in commit(.double(newDouble)) }
                )
            case .string(let s):
                StringEditor(
                    initial: s,
                    onCommit: { newString in commit(.string(newString)) }
                )
            case .null:
                // M4 leaves `.null` as a read-only sentinel; promoting null
                // to a typed value via a "set type" action is M5+ territory.
                Text("null")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            case .array, .object:
                // Nested editors are out of scope for M4.  The compact pretty
                // description gives a one-glance preview of structure.
                Text(prettyDescription(value))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineLimit(4)
                Text("complex value (read-only)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        } else {
            Text(prettyDescription(value))
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .lineLimit(4)
        }
    }

    private func commit(_ newValue: MGJSONValue) {
        viewModel.commitInputValue(
            node: nodeName,
            attrPath: attrPath,
            oldValue: value,
            newValue: newValue,
            undoManager: undoManager
        )
    }
}

// MARK: - Type-specific editors
//
// Each scalar editor is its own little View so it owns the in-flight buffer
// privately.  Two reasons:
//
// 1. `TextField` needs a `Binding<String>` (or `Binding<Value>` + format),
//    which means we need somewhere to hold the pre-commit text.  Putting the
//    buffer on the row would let one row's edit leak into the next.
// 2. We commit on Enter (`.onSubmit`) and revert on Escape (`.onExitCommand`).
//    The revert is just a buffer reset, which is trivially local to the editor.

/// Boolean editor: a plain `Toggle`.  Toggles commit on click (no separate
/// Enter step), so we forward the value to `onCommit` immediately.
struct BoolEditor: View {
    let initial: Bool
    let onCommit: (Bool) -> Void
    @State private var current: Bool

    init(initial: Bool, onCommit: @escaping (Bool) -> Void) {
        self.initial = initial
        self.onCommit = onCommit
        _current = State(initialValue: initial)
    }

    var body: some View {
        Toggle("", isOn: Binding(
            get: { current },
            set: { newValue in
                current = newValue
                onCommit(newValue)
            }
        ))
        .toggleStyle(.switch)
        .labelsHidden()
    }
}

/// Integer editor.  Uses `IntegerFormatStyle` so non-numeric input is rejected
/// at parse time (the field reverts to the last valid value).  Commits on
/// Enter; Escape reverts the in-flight buffer back to `initial`.
struct IntEditor: View {
    let initial: Int64
    let onCommit: (Int64) -> Void
    @State private var current: Int64

    init(initial: Int64, onCommit: @escaping (Int64) -> Void) {
        self.initial = initial
        self.onCommit = onCommit
        _current = State(initialValue: initial)
    }

    var body: some View {
        TextField("", value: $current, format: .number.grouping(.never))
            .textFieldStyle(.roundedBorder)
            .font(.system(.caption, design: .monospaced))
            .onSubmit {
                onCommit(current)
            }
            .onExitCommand {
                current = initial
            }
    }
}

/// Floating-point editor.  Same shape as `IntEditor` but with a
/// `FloatingPointFormatStyle` so `.0` survives the round-trip.  We pass
/// `.number` (not `.percent`/`.currency`) which produces plain decimals.
struct DoubleEditor: View {
    let initial: Double
    let onCommit: (Double) -> Void
    @State private var current: Double

    init(initial: Double, onCommit: @escaping (Double) -> Void) {
        self.initial = initial
        self.onCommit = onCommit
        _current = State(initialValue: initial)
    }

    var body: some View {
        TextField("", value: $current, format: .number.grouping(.never))
            .textFieldStyle(.roundedBorder)
            .font(.system(.caption, design: .monospaced))
            .onSubmit {
                onCommit(current)
            }
            .onExitCommand {
                current = initial
            }
    }
}

/// Plain-string editor.  Also the editor for template references (e.g.
/// `"{CameraInit_1.output}"`): we deliberately edit those as raw strings in
/// M4 — a structured "pick an output to connect" UI is M5+ work.
struct StringEditor: View {
    let initial: String
    let onCommit: (String) -> Void
    @State private var current: String

    init(initial: String, onCommit: @escaping (String) -> Void) {
        self.initial = initial
        self.onCommit = onCommit
        _current = State(initialValue: initial)
    }

    var body: some View {
        TextField("", text: $current)
            .textFieldStyle(.roundedBorder)
            .font(.system(.caption, design: .monospaced))
            .onSubmit {
                onCommit(current)
            }
            .onExitCommand {
                current = initial
            }
    }
}

/// Pipeline panel: a sequential list of every node with status + last log line.
///
/// Displayed under the canvas during/after a run.  We deliberately use the
/// topological order (not the alphabetical `sortedNodeNames`) so the list
/// reads top-down in the same direction as execution.  If topo sort fails
/// (cycle), we fall back to alphabetical — the failure event will already
/// have surfaced via the diagnostics sheet.
struct PipelinePanel: View {
    @ObservedObject var viewModel: ProjectViewModel

    private var orderedNames: [String] {
        // Reuse the scheduler so the order shown matches what would actually
        // execute.  Throwing means cycle — fall back to alphabetical.
        if let order = try? GraphScheduler.topologicalOrder(viewModel.project.graph) {
            return order
        }
        return viewModel.project.graph.sortedNodeNames
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Pipeline")
                    .font(.headline)
                if viewModel.isRunning {
                    ProgressView()
                        .controlSize(.small)
                }
                Spacer()
            }
            .padding(.horizontal, 8)
            .padding(.top, 6)
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(orderedNames, id: \.self) { name in
                        // Disclose chunk children for parallelized nodes,
                        // otherwise show the row standalone.  We key on
                        // presence in `nodeChunks` so non-chunked nodes
                        // don't render an empty disclosure caret.
                        if let chunks = viewModel.nodeChunks[name], !chunks.isEmpty {
                            PipelineNodeWithChunks(
                                name: name,
                                chunks: chunks,
                                statuses: viewModel.nodeStatuses,
                                logTails: viewModel.nodeLogTails
                            )
                        } else {
                            PipelineRow(
                                name: name,
                                status: viewModel.nodeStatuses[name] ?? .queued,
                                lastLine: viewModel.nodeLogTails[name]?.last ?? ""
                            )
                        }
                    }
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 6)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor))
    }
}

/// One row in the pipeline panel: status badge + node name + last stdout line.
struct PipelineRow: View {
    let name: String
    let status: NodeRunStatus
    let lastLine: String

    var body: some View {
        HStack(spacing: 8) {
            statusBadge
            Text(name)
                .font(.system(.body, design: .monospaced))
                .frame(minWidth: 160, alignment: .leading)
            Text(lastLine)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer()
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch status {
        case .queued:
            Image(systemName: "circle")
                .foregroundStyle(.secondary)
                .frame(width: 16)
        case .running:
            ProgressView()
                .controlSize(.small)
                .frame(width: 16)
        case .succeeded:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .frame(width: 16)
        case .failed:
            Image(systemName: "xmark.octagon.fill")
                .foregroundStyle(.red)
                .frame(width: 16)
        case .skipped:
            Image(systemName: "arrow.right.circle")
                .foregroundStyle(.orange)
                .frame(width: 16)
        }
    }
}

/// Pipeline row for a parallelized node: shows the parent row and folds chunk
/// child rows underneath via a `DisclosureGroup`.  Default-expanded so a long
/// running DepthMap is visible without the user having to click — collapsing
/// is a manual action once the user wants to focus elsewhere.
struct PipelineNodeWithChunks: View {
    let name: String
    let chunks: [String]
    let statuses: [String: NodeRunStatus]
    let logTails: [String: [String]]
    @State private var expanded: Bool = true

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(chunks, id: \.self) { chunkKey in
                    PipelineRow(
                        name: chunkLabel(chunkKey),
                        status: statuses[chunkKey] ?? .queued,
                        lastLine: logTails[chunkKey]?.last ?? ""
                    )
                    .padding(.leading, 12)
                }
            }
        } label: {
            PipelineRow(
                name: name,
                status: statuses[name] ?? .queued,
                lastLine: logTails[name]?.last ?? ""
            )
        }
    }

    /// Strip the parent prefix from a chunk key so the disclosure child reads
    /// "chunk 1/5" rather than the full "DepthMap_1[chunk 1/5]".  Keeps the
    /// row compact under the parent which already shows the node name.
    private func chunkLabel(_ key: String) -> String {
        if let r = key.range(of: "[chunk ") {
            // "chunk 1/5]" -> drop trailing `]`
            let raw = String(key[r.lowerBound...])
            return raw
                .replacingOccurrences(of: "[", with: "")
                .replacingOccurrences(of: "]", with: "")
        }
        return key
    }
}

/// Produce a compact, human-readable rendering of an `MGJSONValue`.  We
/// avoid `String(describing:)` here because it leaks Swift enum syntax
/// (`MGJSONValue.int(0)`), which is unhelpful in an end-user inspector.
/// Lifted out of `NodeInspector` so the read-only fallbacks in `AttributeRow`
/// can share it without depending on the inspector's identity.
fileprivate func prettyDescription(_ value: MGJSONValue) -> String {
    switch value {
    case .null: return "null"
    case .bool(let b): return b ? "true" : "false"
    case .int(let i): return String(i)
    case .double(let d): return String(d)
    case .string(let s): return s
    case .array(let arr):
        if arr.isEmpty { return "[]" }
        return "[" + arr.map(prettyDescription).joined(separator: ", ") + "]"
    case .object(let obj):
        if obj.isEmpty { return "{}" }
        let pairs = obj.keys.sorted().map { k -> String in
            "\(k): \(prettyDescription(obj[k] ?? .null))"
        }
        return "{" + pairs.joined(separator: ", ") + "}"
    }
}
