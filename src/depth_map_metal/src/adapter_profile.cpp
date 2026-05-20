// adapter_profile.cpp — S43 Phase 14 perf profiling impl.
//
// Only compiled into the static lib when -DAV_PROFILE_ADAPTER=ON.
// (See CMakeLists.txt: this TU is added to av_depth_map_metal only
// when the option is on, so disabled builds have zero overhead and
// zero new symbols.)

#if AV_PROFILE_ADAPTER

#include "av/depth_map/adapter_profile.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace av::depth_map::profile {

namespace {

struct Entry {
    std::uint64_t calls = 0;
    std::uint64_t total_us = 0;
    std::uint64_t max_us = 0;
};

struct Table {
    std::mutex mu;
    // Key: const char* — must be a string literal / process-lifetime
    // pointer. We hash the pointer itself (cheap; safe given the
    // restriction). For a forwarder-name table of size <=15 this is
    // perfectly fine.
    std::unordered_map<const char*, Entry> entries;
    std::atomic<bool> dumped{false};

    ~Table() {
        // RAII dump at process exit.
        dump_locked();
    }

    void dump_locked() {
        // We accept that this can be called twice (once explicitly, once
        // via ~Table). Use an atomic guard to no-op on re-entry.
        bool expected = false;
        if (!dumped.compare_exchange_strong(expected, true)) return;

        std::lock_guard lk(mu);
        if (entries.empty()) {
            std::fprintf(stderr,
                "[AV_PROFILE_ADAPTER] no forwarder calls recorded.\n");
            return;
        }

        // Snapshot + sort by total_us descending.
        std::vector<std::pair<const char*, Entry>> rows(
            entries.begin(), entries.end());
        std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) {
                return a.second.total_us > b.second.total_us;
            });

        std::uint64_t grand_total_us = 0;
        for (const auto& r : rows) grand_total_us += r.second.total_us;
        const double grand_total_ms =
            static_cast<double>(grand_total_us) / 1000.0;

        std::fprintf(stderr,
            "\n"
            "============================================================\n"
            "AV_PROFILE_ADAPTER — adapter forwarder timing summary\n"
            "============================================================\n"
            "Grand total wall time inside adapter: %.3f ms across %zu fns\n"
            "\n"
            "| Function                                  |   Calls |     Total (ms) |  Mean (ms) |   Max (ms) | %% Total |\n"
            "|-------------------------------------------|---------|----------------|------------|------------|---------|\n",
            grand_total_ms, rows.size());

        for (const auto& [name, e] : rows) {
            const double total_ms = static_cast<double>(e.total_us) / 1000.0;
            const double mean_ms  = static_cast<double>(e.total_us) /
                                    static_cast<double>(e.calls) / 1000.0;
            const double max_ms   = static_cast<double>(e.max_us)   / 1000.0;
            const double pct      = grand_total_us == 0
                                  ? 0.0
                                  : 100.0 * static_cast<double>(e.total_us) /
                                    static_cast<double>(grand_total_us);
            std::fprintf(stderr,
                "| %-41s | %7llu | %14.3f | %10.3f | %10.3f | %6.2f%% |\n",
                name,
                static_cast<unsigned long long>(e.calls),
                total_ms, mean_ms, max_ms, pct);
        }
        std::fprintf(stderr,
            "============================================================\n");
        std::fflush(stderr);
    }
};

// Function-local static = ordered destruction relative to other
// function-local statics. We avoid global static-init order pitfalls.
Table& table() {
    static Table t;
    return t;
}

// Register an atexit hook in addition to the ~Table dtor — atexit fires
// EARLIER than function-local-static destruction in some toolchains
// (e.g. when objects are torn down across DSOs). Whichever runs first
// will print, the other is a no-op via the `dumped` flag.
struct AtExitInstaller {
    AtExitInstaller() {
        std::atexit([] {
            table().dump_locked();
        });
    }
};
AtExitInstaller _installer;

}  // namespace

void record(const char* name, double ms) {
    // Convert to integer microseconds for accumulation precision.
    const std::uint64_t us = static_cast<std::uint64_t>(ms * 1000.0 + 0.5);
    auto& t = table();
    std::lock_guard lk(t.mu);
    auto& e = t.entries[name];
    e.calls += 1;
    e.total_us += us;
    if (us > e.max_us) e.max_us = us;
}

void dump() {
    table().dump_locked();
}

}  // namespace av::depth_map::profile

#endif  // AV_PROFILE_ADAPTER
