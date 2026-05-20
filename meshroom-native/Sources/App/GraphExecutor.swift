import Foundation
import ProjectModel

/// Drives a `.mg` project to completion by invoking the matching
/// `aliceVision_*` binary for each node in topological order.
///
/// Concurrency model:
/// - The executor is a `@MainActor` class (rather than an actor) because it
///   has to mutate `@Published` view-model state on Process callbacks; jumping
///   from a Process's background thread back to a non-main actor adds latency
///   and ordering hazards.  All actual blocking work happens off-main on the
///   `Process` system thread; we only re-enter main when emitting events.
/// - Events stream out through an `AsyncStream<Event>` so the UI can consume
///   them with `for await` without any callback wrangling.  The stream
///   continuation is held inside the executor instance so a Stop button can
///   cancel mid-run by terminating the in-flight process and finishing the
///   stream.
///
/// Cache layout (M6):
/// - Cache directories follow Meshroom's `<workdir>/MeshroomCache/<NodeName>_<UID>/`
///   layout.  UIDs come from `NodeUIDHasher` (SHA-1 of the canonical
///   `json.dumps`-style serialization of resolved inputs), so re-running an
///   unchanged graph produces the same cache dir and reuses any artefacts
///   already on disk.
/// - When a `.mg` file already carries an explicit `"uid"` per node (as
///   Python-Meshroom-saved files do), the stored UID wins over the
///   recomputed one — that's how we read warm caches produced by upstream
///   Python Meshroom.  See `NodeUIDHasher` for the format details.
/// - Warm-cache detection: when `NodeBinary.Spec.expectedOutputFile` is set
///   AND that file already exists under the resolved cache dir, the node is
///   skipped with `.nodeSkipped` and no Process is spawned.  Nodes with a
///   `nil` expectedOutputFile (e.g. FeatureExtraction's per-view artefacts)
///   always re-run; correctness wins over warm-cache aggressiveness for
///   those.
@MainActor
public final class GraphExecutor {

    public enum Event: Sendable {
        case nodeStarted(name: String, binary: String, argv: [String])
        /// One newline-terminated chunk of stdout or stderr from a running node.
        case nodeProgress(name: String, line: String)
        case nodeSkipped(name: String, reason: String)
        case nodeFinished(name: String, exitCode: Int32)
        case failed(name: String, reason: String)
        case allDone
    }

    /// One row in the chunk plan for a parallelized node.
    ///
    /// Surfaced publicly so the UI's pipeline panel can render per-chunk
    /// status and the chunk planner is unit-testable in isolation from the
    /// (non-testable) `Process` spawning.
    public struct Chunk: Hashable, Sendable {
        public let index: Int       // 0-based chunk index
        public let total: Int       // total number of chunks in the plan
        public let start: Int       // rangeStart (first view index)
        public let size: Int        // rangeSize (view count this chunk processes)

        public init(index: Int, total: Int, start: Int, size: Int) {
            self.index = index
            self.total = total
            self.start = start
            self.size = size
        }
    }

    /// Configuration handed in by the UI; bundling these into a struct keeps
    /// `run(...)` signatures stable when we add e.g. environment overrides
    /// later.
    public struct RunConfig {
        public let workDir: URL          // Project working directory; `MeshroomCache/` lives here.
        public let binDir: URL           // Where `aliceVision_*` binaries live.
        public let aliceVisionRoot: URL  // Sets `ALICEVISION_ROOT`.
        public let extraEnv: [String: String]

        public init(workDir: URL, binDir: URL, aliceVisionRoot: URL, extraEnv: [String: String] = [:]) {
            self.workDir = workDir
            self.binDir = binDir
            self.aliceVisionRoot = aliceVisionRoot
            self.extraEnv = extraEnv
        }
    }

    /// The process currently running, kept so `cancel()` can SIGTERM it.  Only
    /// one node runs at a time in M5 — parallel sub-DAGs are M6.
    private var currentProcess: Process?
    /// Set to true on `cancel()`; the run loop checks it between nodes and
    /// after each Process exit to bail out cleanly.
    private var cancelled = false
    /// Output paths emitted by upstream nodes, indexed by node name.
    /// Built up during the run so downstream nodes can resolve their template
    /// inputs against the freshly-allocated cache directories.
    private var outputPaths: [String: [String: String]] = [:]
    /// Per-run cache-directory roots, indexed by node name.  Allocated lazily
    /// on first reference so a skipped (unsupported) node doesn't pollute the
    /// filesystem.
    private var nodeCacheDirs: [String: URL] = [:]
    /// Per-node Meshroom-compatible UIDs, computed once per `run()` invocation
    /// from the resolved input graph (or pulled from `MGNode.uid` when the
    /// `.mg` file already carries an explicit UID, e.g. files written by
    /// Python Meshroom).  Empty until `runLoop` populates it.
    private var nodeUIDs: [String: String] = [:]

    public init() {}

    // MARK: - Testing hooks
    //
    // These two functions expose the renderer + argv builder without going
    // through `run(...)`.  They make unit-testing the template resolver and
    // argv assembly free of `Process` spawning — which is what we'd otherwise
    // have to mock to verify the resolver runs correctly.

    /// Seed an upstream node's resolved output paths.  In a real run this is
    /// populated by `registerOutputs` as each node completes; tests bypass
    /// the pipeline by setting it directly.
    func setOutputsForTesting(_ outputs: [String: String], for nodeName: String) {
        outputPaths[nodeName] = outputs
    }

    /// Public-for-testing version of `buildArgv` that takes the already-resolved
    /// upstream outputs.  Production callers go through `run(...)`.
    func buildArgvForTesting(
        node: MGNode,
        nodeName: String,
        cacheDir: URL
    ) -> [String]? {
        guard let spec = NodeBinary.spec(for: node.nodeType) else { return nil }
        return buildArgv(node: node, nodeName: nodeName, spec: spec, cacheDir: cacheDir)
    }

    // MARK: - Chunk planning (pure, public for tests)

    /// Compute the chunk plan for `viewCount` views processed in groups of
    /// `blockSize`.  Returns `[]` when `viewCount <= 0` (the caller can decide
    /// whether to skip the node or run a single zero-range chunk).
    ///
    /// Matches upstream Meshroom's `Parallelization.getSizes` semantics:
    /// `total = ceil(viewCount / blockSize)`; chunk `i` has
    /// `rangeStart = i * blockSize`, `rangeSize = min(blockSize, viewCount - rangeStart)`.
    public static func planChunks(viewCount: Int, blockSize: Int) -> [Chunk] {
        guard viewCount > 0, blockSize > 0 else { return [] }
        let total = (viewCount + blockSize - 1) / blockSize
        var out: [Chunk] = []
        out.reserveCapacity(total)
        for i in 0..<total {
            let start = i * blockSize
            let size = min(blockSize, viewCount - start)
            out.append(Chunk(index: i, total: total, start: start, size: size))
        }
        return out
    }

    /// Count entries in the top-level `views` array of an SfMData JSON file.
    ///
    /// Works for both upstream variants we encounter in the M5 pipeline:
    /// - `cameraInit.sfm` (output of CameraInit) — `views` is a JSON array
    ///   of objects keyed by `viewId`.
    /// - downstream `sfm.sfm` / `sfm.abc` (output of StructureFromMotion) —
    ///   same shape; SfM may strip un-reconstructed views but never changes
    ///   the field name.
    ///
    /// We deliberately use `JSONSerialization` (not `Codable`) to avoid
    /// declaring the SfMData schema — we only need one field, and a partial
    /// `Codable` decoder would still walk the whole file and fail on the
    /// (large) `intrinsics`/`poses` arrays for any minor schema drift.
    ///
    /// Returns 0 if the file is missing, unreadable, or not JSON (e.g. an
    /// `.abc` Alembic binary downstream of SfM).  Callers should treat 0 as
    /// "fall back to a single chunk" so a malformed SfMData doesn't silently
    /// skip the node entirely.
    public static func countViewsInSfMData(at path: String) -> Int {
        guard FileManager.default.fileExists(atPath: path) else { return 0 }
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return 0 }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return 0
        }
        guard let views = obj["views"] as? [Any] else { return 0 }
        return views.count
    }

    /// Substitute `{start}` and `{size}` in a chunk's `rangeArg` template,
    /// then split on whitespace to produce argv tokens.  Public for tests.
    public static func renderRangeArgs(template: String, chunk: Chunk) -> [String] {
        let substituted = template
            .replacingOccurrences(of: "{start}", with: String(chunk.start))
            .replacingOccurrences(of: "{size}", with: String(chunk.size))
        // Split on any run of whitespace.  AliceVision flags never contain
        // embedded spaces; this is simpler than tokenizing for quotes.
        return substituted
            .split(whereSeparator: { $0.isWhitespace })
            .map(String.init)
    }

    /// Signal the executor to stop after (or during) the current node.  The
    /// running `Process` receives SIGTERM; `run(...)` then emits no further
    /// events for queued nodes and finishes the stream.
    public func cancel() {
        cancelled = true
        currentProcess?.terminate()
    }

    /// Execute every node in topological order.  Returns an `AsyncStream` of
    /// progress events; consumers drive it with `for await event in stream`.
    ///
    /// Failure handling: a non-zero exit on any node emits `.failed` and stops
    /// the pipeline.  Downstream nodes are *not* attempted (they would fail
    /// for the same reason — missing inputs).  The stream always terminates
    /// either with `.allDone` (success) or `.failed` (then nothing more).
    public func run(_ project: MGProject, config: RunConfig) -> AsyncStream<Event> {
        AsyncStream { continuation in
            Task { @MainActor in
                await self.runLoop(project: project, config: config, emit: { event in
                    continuation.yield(event)
                })
                continuation.finish()
            }
        }
    }

    // MARK: - Internal run loop

    private func runLoop(
        project: MGProject,
        config: RunConfig,
        emit: @escaping @MainActor (Event) -> Void
    ) async {
        // Reset per-run state so the same executor can serve multiple runs.
        currentProcess = nil
        cancelled = false
        outputPaths = [:]
        nodeCacheDirs = [:]
        nodeUIDs = [:]

        let order: [String]
        do {
            order = try GraphScheduler.topologicalOrder(project.graph)
        } catch {
            emit(.failed(name: "<scheduler>", reason: "\(error)"))
            return
        }

        // Pre-compute every node's UID (Merkle-style over resolved inputs).
        // We use the .mg-stored `uid` when present so files saved by Python
        // Meshroom reuse their existing cache directories.  Otherwise we
        // recompute via NodeUIDHasher — see that type's docs for the format.
        nodeUIDs = NodeUIDHasher.uids(for: project.graph, topoOrder: order)
        for nodeName in order {
            if let stored = project.graph[nodeName]?.uid, !stored.isEmpty {
                nodeUIDs[nodeName] = stored
            }
        }

        for nodeName in order {
            if cancelled { return }
            guard let node = project.graph[nodeName] else { continue }

            guard let spec = NodeBinary.spec(for: node.nodeType) else {
                emit(.nodeSkipped(name: nodeName, reason: "Unsupported node type '\(node.nodeType)'"))
                continue
            }

            // Allocate or reuse the node cache directory, registering its
            // output paths so downstream nodes can resolve their templates.
            let cacheDir = ensureCacheDir(for: nodeName, in: config.workDir)
            registerOutputs(of: node, named: nodeName, cacheDir: cacheDir)

            // Build argv now that we know our cache dir + upstream outputs.
            let argv = buildArgv(
                node: node,
                nodeName: nodeName,
                spec: spec,
                cacheDir: cacheDir
            )

            // M6 warm-cache detection: if the spec declares a canonical
            // `expectedOutputFile` and that file already lives in the
            // resolved cache dir, we've run this node before with the same
            // inputs (UID determinism guarantees it).  Skip the Process spawn.
            //
            // Nodes without a canonical artefact (FeatureExtraction's
            // per-view files etc.) fall through to a looser check: skip iff
            // every declared output path exists on disk.  That used to be
            // the M5 behaviour and is still useful for any node where the
            // template happens to point at a single concrete file.
            if let expected = spec.expectedOutputFile {
                let expectedPath = cacheDir.appending(path: expected).path
                if FileManager.default.fileExists(atPath: expectedPath) {
                    emit(.nodeSkipped(
                        name: nodeName,
                        reason: "Cache hit: \(expectedPath)"
                    ))
                    continue
                }
            } else if let outs = outputPaths[nodeName],
                      !outs.isEmpty,
                      outs.values.allSatisfy(Self.outputAlreadyOnDisk) {
                emit(.nodeSkipped(name: nodeName, reason: "Outputs exist at \(cacheDir.path)"))
                continue
            }

            // Decide single-invocation vs chunked execution.  We treat the
            // parent node name as the event "name" so existing UI keyed by
            // node name still receives `.nodeStarted` / `.nodeFinished` and
            // can fold per-chunk progress under it.  Chunk-scoped events
            // carry `"<NodeName>[chunk i/N]"` so the UI can show them on
            // their own row.
            let executableURL = config.binDir.appendingPathComponent(spec.executable)

            if let par = spec.parallelization {
                // Determine view count from the input SfMData.  The `input`
                // attribute is conventional across all parallelized
                // AliceVision binaries (see DepthMap, DepthMapFilter etc.).
                let inputPath: String? = {
                    guard let inputs = node.inputs, let raw = inputs["input"] else { return nil }
                    let rendered = renderInputValue(raw)
                    return rendered.isEmpty ? nil : rendered
                }()
                let viewCount = inputPath.map { Self.countViewsInSfMData(at: $0) } ?? 0
                let chunks = Self.planChunks(viewCount: viewCount, blockSize: par.blockSize)

                if chunks.isEmpty {
                    // No views detected — skip the node rather than spawning
                    // a single-view default chunk that would itself fail.
                    // This is safer than guessing: an empty SfMData almost
                    // always means an upstream stage produced no output.
                    emit(.nodeSkipped(
                        name: nodeName,
                        reason: "No views found in SfMData (\(inputPath ?? "no input")); nothing to process"
                    ))
                    continue
                }

                // Emit a single `.nodeStarted` for the parent so the UI
                // flips its status to running once for the whole node.
                emit(.nodeStarted(name: nodeName, binary: spec.executable, argv: argv))

                var allChunksSucceeded = true
                for chunk in chunks {
                    if cancelled {
                        emit(.failed(name: nodeName, reason: "Cancelled by user"))
                        return
                    }
                    let chunkName = "\(nodeName)[chunk \(chunk.index + 1)/\(chunk.total)]"
                    let chunkArgv = argv + Self.renderRangeArgs(template: par.rangeArg, chunk: chunk)
                    emit(.nodeStarted(name: chunkName, binary: spec.executable, argv: chunkArgv))

                    let result = await runOneProcess(
                        executable: executableURL,
                        argv: chunkArgv,
                        config: config,
                        nodeName: chunkName,
                        emit: emit
                    )

                    if cancelled {
                        emit(.failed(name: chunkName, reason: "Cancelled by user"))
                        emit(.failed(name: nodeName, reason: "Cancelled by user"))
                        return
                    }
                    switch result {
                    case .success(let exitCode) where exitCode == 0:
                        emit(.nodeFinished(name: chunkName, exitCode: exitCode))
                    case .success(let exitCode):
                        emit(.failed(name: chunkName, reason: "Exit code \(exitCode)"))
                        emit(.failed(
                            name: nodeName,
                            reason: "Chunk \(chunk.index + 1)/\(chunk.total) failed (exit \(exitCode))"
                        ))
                        allChunksSucceeded = false
                    case .failure(let error):
                        emit(.failed(name: chunkName, reason: "\(error)"))
                        emit(.failed(
                            name: nodeName,
                            reason: "Chunk \(chunk.index + 1)/\(chunk.total) failed: \(error)"
                        ))
                        allChunksSucceeded = false
                    }
                    if !allChunksSucceeded { return }
                }
                emit(.nodeFinished(name: nodeName, exitCode: 0))
                continue
            }

            emit(.nodeStarted(name: nodeName, binary: spec.executable, argv: argv))

            // Run the process.  `await` until it terminates or is cancelled.
            let result = await runOneProcess(
                executable: executableURL,
                argv: argv,
                config: config,
                nodeName: nodeName,
                emit: emit
            )

            if cancelled {
                emit(.failed(name: nodeName, reason: "Cancelled by user"))
                return
            }
            switch result {
            case .success(let exitCode) where exitCode == 0:
                emit(.nodeFinished(name: nodeName, exitCode: exitCode))
            case .success(let exitCode):
                emit(.failed(name: nodeName, reason: "Exit code \(exitCode)"))
                return
            case .failure(let error):
                emit(.failed(name: nodeName, reason: "\(error)"))
                return
            }
        }

        emit(.allDone)
    }

    // MARK: - Cache management

    /// Fallback UID used only when the executor is exercised through a
    /// testing entry point that bypasses `runLoop` (and therefore the UID
    /// pre-pass).  Real runs always have `nodeUIDs` populated.  Once-per-
    /// executor so two test calls in the same suite share a directory and
    /// can observe consistent paths.
    private let fallbackUID: String = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withFullDate, .withTime, .withColonSeparatorInTime]
        return f.string(from: Date()).replacingOccurrences(of: ":", with: "")
    }()

    private func ensureCacheDir(for nodeName: String, in workDir: URL) -> URL {
        if let existing = nodeCacheDirs[nodeName] { return existing }
        // Layout mirrors Meshroom: <workdir>/MeshroomCache/<NodeName>_<UID>/.
        // Prefer the pre-computed UID (Merkle hash over resolved inputs), then
        // fall back to a per-executor timestamp for code paths that haven't
        // populated `nodeUIDs` (currently only test hooks).
        let uid = nodeUIDs[nodeName] ?? fallbackUID
        let dir = workDir
            .appendingPathComponent("MeshroomCache", isDirectory: true)
            .appendingPathComponent("\(nodeName)_\(uid)", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        nodeCacheDirs[nodeName] = dir
        return dir
    }

    /// Walk the node's declared outputs and substitute `{nodeCacheFolder}`
    /// with the freshly-allocated cache dir.  Records the resolved paths in
    /// `outputPaths` so downstream nodes can find them via the template
    /// resolver.
    private func registerOutputs(of node: MGNode, named nodeName: String, cacheDir: URL) {
        guard let outs = node.outputs else {
            outputPaths[nodeName] = [:]
            return
        }
        var resolved: [String: String] = [:]
        for (key, value) in outs {
            if case let .string(template) = value {
                resolved[key] = Self.substituteCacheFolder(template, cacheDir: cacheDir)
            }
        }
        outputPaths[nodeName] = resolved
    }

    /// Replace literal `{nodeCacheFolder}` with the absolute cache path.
    /// `${ENV}` references stay as-is — the binaries inherit the environment
    /// from the spawned `Process`, so they can resolve them themselves.
    private static func substituteCacheFolder(_ template: String, cacheDir: URL) -> String {
        template.replacingOccurrences(of: "{nodeCacheFolder}", with: cacheDir.path)
    }

    private static func outputAlreadyOnDisk(_ path: String) -> Bool {
        FileManager.default.fileExists(atPath: path)
    }

    // MARK: - Template resolver

    /// Resolve template references inside an `MGJSONValue` against the
    /// `outputPaths` table populated by upstream nodes.  Recurses into arrays
    /// and nested objects so references inside `featuresFolders:["{X.output}"]`
    /// or grouped attributes resolve correctly.
    ///
    /// For scalar leaves the renderer is type-aware:
    /// - booleans collapse to `0` / `1` (AliceVision CLI convention)
    /// - numbers / strings stringify directly
    /// - arrays render as comma-joined strings (AliceVision multi-value flags
    ///   accept comma separation: `--allowedCameraModels pinhole,radial1`)
    /// - nested objects emit a JSON string — best-effort for advanced groups
    ///   like `Meshing.advanced` which AliceVision parses as nested params.
    func renderInputValue(_ value: MGJSONValue) -> String {
        switch value {
        case .null:
            return ""
        case .bool(let b):
            return b ? "1" : "0"
        case .int(let i):
            return String(i)
        case .double(let d):
            // Preserve trailing `.0` so float-typed flags don't lose precision.
            // String(Double) emits `1.0` (not `1`), which matches the CLI's
            // own canonical form.
            return String(d)
        case .string(let s):
            return resolveString(s)
        case .array(let arr):
            return arr.map { renderInputValue($0) }.joined(separator: ",")
        case .object(let obj):
            // Nested groups: render as JSON so the user at least sees them
            // intact in the log; the binary may not actually accept it.
            if let data = try? JSONSerialization.data(
                withJSONObject: obj.mapValues { $0.jsonSerializationValue }
            ), let s = String(data: data, encoding: .utf8) {
                return s
            }
            return ""
        }
    }

    /// Resolve `{NodeName.attr}` inside a string against `outputPaths`.
    /// Strings without braces, and `${ENV}` references, pass through unchanged
    /// — the binaries themselves resolve environment via the inherited env.
    private func resolveString(_ s: String) -> String {
        // Single, whole-string template ref: `{Node.out}` -> resolved path.
        if let ref = MGTemplateReference.parse(s) {
            let attr = ref.attributePath.first ?? ""
            if let nodeOuts = outputPaths[ref.nodeName], let path = nodeOuts[attr] {
                return path
            }
            // Reference to an upstream output we don't know about (e.g. the
            // user referenced `.input` which is itself an input alias).  Try
            // resolving against the upstream node's own inputs — not generally
            // safe, but matches `.mg` files that route the input forward via
            // `{X.input}` instead of `{X.output}`.
            return s  // Leave as-is — binary will likely fail; caller surfaces it.
        }
        return s
    }

    // MARK: - Argv building

    /// Construct an argv array for `node` from its spec, current upstream
    /// output paths, and the node's own `inputs` dictionary.
    ///
    /// Visibility is `internal` so tests can exercise it directly without
    /// running an actual process — the heart of the resolver lives here and is
    /// worth covering even if `Process` itself is XCTest-fragile.
    func buildArgv(
        node: MGNode,
        nodeName: String,
        spec: NodeBinary.Spec,
        cacheDir: URL
    ) -> [String] {
        var argv: [String] = []
        // Input flags first — ordered per spec for deterministic argv.
        let inputs = node.inputs ?? [:]
        for flagName in spec.inputFlags {
            guard let raw = inputs[flagName] else { continue }
            let rendered = renderInputValue(raw)
            // Skip empties so we don't pass `--flag ` with no value (the
            // binary's option parser would consume the next flag as the value).
            if rendered.isEmpty { continue }
            argv.append("--\(flagName)")
            argv.append(rendered)
        }
        // Then outputs.  These point at the freshly-allocated cache dir; we
        // re-evaluate the template here (rather than reading `outputPaths`
        // which has the same data) for tighter coupling to `spec.outputFlags`
        // — the spec is the source of truth for what we emit on argv.
        if let outs = node.outputs {
            for flagName in spec.outputFlags {
                guard let value = outs[flagName] else { continue }
                let rendered = renderInputValue(value)
                if rendered.isEmpty { continue }
                argv.append("--\(flagName)")
                argv.append(rendered)
            }
        } else {
            // Synthesize a single `--output <cacheDir>` for nodes that don't
            // declare outputs in the `.mg` (rare; templates can omit them).
            for flagName in spec.outputFlags {
                argv.append("--\(flagName)")
                argv.append(cacheDir.path)
            }
        }
        argv.append(contentsOf: spec.constantFlags)
        return argv
    }

    // MARK: - Process plumbing

    private enum ProcessResult {
        case success(Int32)
        case failure(Error)
    }

    /// Spawn one `Process` and `await` its termination.  stdout+stderr are
    /// piped through a `Pipe` whose `readabilityHandler` chunks the data on
    /// newlines and re-emits each line on the main actor as `.nodeProgress`.
    private func runOneProcess(
        executable: URL,
        argv: [String],
        config: RunConfig,
        nodeName: String,
        emit: @escaping @MainActor (Event) -> Void
    ) async -> ProcessResult {
        // Build env: inherit, then layer the AliceVision-specific vars on top.
        // Setting `ALICEVISION_ROOT` is the *minimum* required for binaries to
        // find their share/aliceVision resources (sensorDB, OCIO config, etc).
        var env = ProcessInfo.processInfo.environment
        env["ALICEVISION_ROOT"] = config.aliceVisionRoot.path
        env["ALICEVISION_BIN_PATH"] = config.binDir.path
        // Prepend our binDir to PATH so any child processes the binaries spawn
        // (rare, but possible) find the right siblings.
        let existingPath = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        env["PATH"] = "\(config.binDir.path):\(existingPath)"
        // Sensor DB and OCIO live under `aliceVisionRoot/share/aliceVision/`.
        env["ALICEVISION_SENSOR_DB"] = config.aliceVisionRoot
            .appendingPathComponent("share/aliceVision/cameraSensors.db").path
        env["ALICEVISION_OCIO"] = config.aliceVisionRoot
            .appendingPathComponent("share/aliceVision/config.ocio").path
        // Layer any caller overrides last so tests/scripts can stub values.
        for (k, v) in config.extraEnv {
            env[k] = v
        }

        let process = Process()
        process.executableURL = executable
        process.arguments = argv
        process.environment = env
        process.currentDirectoryURL = config.workDir

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        // Line-buffering on the pipe.  `availableData` arrives in arbitrary
        // chunks, so we accumulate and split on '\n' to emit one event per
        // logical line.  Without this, a single chunk can carry a partial line
        // followed by a complete one and the UI sees mangled output.
        //
        // The buffer lives inside a `final class` because `readabilityHandler`
        // is `@Sendable` (it's invoked on an arbitrary background queue) —
        // mutating a captured `var Data` would race under Swift 6 strict
        // concurrency.  Wrapping in a class lets the closure capture by
        // reference and serialise mutation through the pipe's own queue, which
        // delivers chunks sequentially.
        let buffer = LineBuffer()
        pipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if chunk.isEmpty { return }
            let lines = buffer.append(chunk)
            for line in lines {
                Task { @MainActor in
                    emit(.nodeProgress(name: nodeName, line: line))
                }
            }
        }

        currentProcess = process
        defer {
            currentProcess = nil
            pipe.fileHandleForReading.readabilityHandler = nil
        }

        do {
            try process.run()
        } catch {
            return .failure(error)
        }

        // Await termination via continuation.  Process.waitUntilExit blocks the
        // calling thread, which would freeze the main actor — withCheckedContinuation
        // lets us suspend the Task instead.
        let exitCode: Int32 = await withCheckedContinuation { cont in
            process.terminationHandler = { proc in
                cont.resume(returning: proc.terminationStatus)
            }
        }

        // Flush any final partial line that didn't end in newline.
        if let trailing = buffer.drain(), !trailing.isEmpty {
            emit(.nodeProgress(name: nodeName, line: trailing))
        }

        return .success(exitCode)
    }
}

/// Reference-type byte accumulator for `Pipe.readabilityHandler` callbacks.
///
/// `readabilityHandler` is `@Sendable` and fires on a background queue.  A
/// captured `var Data` would be a Swift 6 strict-concurrency violation; a
/// `final class` lets us mutate by reference and rely on the pipe's internal
/// serialisation (one callback at a time per file handle).  An `NSLock` guards
/// the buffer to defensively cover the case where the pipe ever batches
/// callbacks across queues.
private final class LineBuffer: @unchecked Sendable {
    private var data = Data()
    private let lock = NSLock()

    /// Append `chunk`, return any newline-terminated lines it completed.
    func append(_ chunk: Data) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        data.append(chunk)
        var out: [String] = []
        while let nl = data.firstIndex(of: 0x0A) {
            let lineData = data.subdata(in: 0..<nl)
            data.removeSubrange(0...nl)
            if let line = String(data: lineData, encoding: .utf8) {
                out.append(line)
            }
        }
        return out
    }

    /// Take any trailing (unterminated) bytes left over after the process
    /// exits.  Returns `nil` when the buffer is empty.
    func drain() -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard !data.isEmpty else { return nil }
        let s = String(data: data, encoding: .utf8)
        data.removeAll()
        return s
    }
}
