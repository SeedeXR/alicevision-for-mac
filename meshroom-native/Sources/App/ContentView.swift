import SwiftUI
import UniformTypeIdentifiers
import ProjectModel

/// The `.mg` UTI.  Meshroom does not register a system-wide UTI on macOS, so
/// we declare one locally for the `.fileImporter` filter.  Using
/// `UTType(filenameExtension: "mg")` returns a dynamic type bound to that
/// extension, which is sufficient to make `fileImporter` show `.mg` files.
extension UTType {
    static var meshroomGraph: UTType {
        // Prefer a dynamic UTType so we work without Info.plist registration.
        UTType(filenameExtension: "mg") ?? .json
    }
}

/// Root view.  Switches between an empty state (no project open) and the
/// `ProjectView` once a `.mg` file is loaded.  Errors are surfaced via an
/// `.alert` so they don't leave the user staring at a blank window.
struct ContentView: View {
    @State private var project: MGProject?
    @State private var projectURL: URL?
    @State private var isImporting = false
    @State private var errorMessage: String?
    @State private var showingError = false

    var body: some View {
        Group {
            if let project = project {
                ProjectView(project: project, projectURL: projectURL)
            } else {
                emptyState
            }
        }
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [.meshroomGraph],
            allowsMultipleSelection: false,
            onCompletion: handleImport
        )
        .alert("Could not open project", isPresented: $showingError, actions: {
            Button("OK", role: .cancel) {}
        }, message: {
            Text(errorMessage ?? "Unknown error")
        })
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    isImporting = true
                } label: {
                    Label("Open…", systemImage: "folder")
                }
                .help("Open a Meshroom .mg project file")
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 20) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(.system(size: 64))
                .foregroundStyle(.secondary)
            Text("No project open")
                .font(.title2)
            Text("Open a Meshroom .mg project file to view its graph.")
                .foregroundStyle(.secondary)
            Button {
                isImporting = true
            } label: {
                Label("Open .mg project…", systemImage: "folder")
            }
            .keyboardShortcut("o", modifiers: .command)
            .controlSize(.large)
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private func handleImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            loadProject(at: url)
        case .failure(let error):
            // User-cancel surfaces as a NSCocoaError with code 3072; suppress it.
            // The .fileImporter API documents that user-cancel is reported as a
            // CocoaError.userCancelled, so check that case explicitly.
            let nsError = error as NSError
            if nsError.code == NSUserCancelledError { return }
            errorMessage = error.localizedDescription
            showingError = true
        }
    }

    private func loadProject(at url: URL) {
        // .fileImporter URLs require sandboxed scoped access.  In a non-
        // sandboxed `swift run` executable this is a no-op, but doing it here
        // keeps the code correct for an eventually-sandboxed App Store build.
        let didStart = url.startAccessingSecurityScopedResource()
        defer {
            if didStart { url.stopAccessingSecurityScopedResource() }
        }
        do {
            let loaded = try MGProject.load(from: url)
            self.project = loaded
            self.projectURL = url
        } catch {
            errorMessage = "Failed to parse \(url.lastPathComponent): \(error)"
            showingError = true
        }
    }
}
