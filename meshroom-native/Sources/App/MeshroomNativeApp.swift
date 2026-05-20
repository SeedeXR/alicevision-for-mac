import SwiftUI

/// Top-level `@main` for the Phase 15 M2 read-only viewer.
///
/// We host a single `WindowGroup` that owns a `ContentView`.  The state machine
/// (empty → loaded → error) lives one level down inside `ContentView` so the
/// app shell stays trivial and we keep all SwiftUI lifecycle wiring co-located
/// with the views that need it.
@main
struct MeshroomNativeApp: App {
    var body: some Scene {
        WindowGroup("Meshroom Native") {
            ContentView()
                .frame(minWidth: 900, minHeight: 600)
        }
        // .windowStyle(.hiddenTitleBar) is intentionally NOT used — we want
        // a normal macOS chrome with traffic lights for the M2 milestone.
    }
}
