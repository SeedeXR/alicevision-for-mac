#pragma once

// adapter_profile.hpp — S43 Phase 14 perf profiling.
//
// Thread-safe per-function timing accumulator for the 15 `cuda_*`
// forwarders in `upstream_adapter.cpp`. Enabled only when the project
// is configured with `-DAV_PROFILE_ADAPTER=ON`; otherwise the macros
// expand to no-ops and there is zero runtime cost.
//
// Design:
//   * One global hash map (name → {count, total_us, max_us}) guarded by
//     a single mutex. Contention is irrelevant: the forwarders take
//     hundreds of microseconds to milliseconds each, so the per-call
//     locking overhead is in the noise.
//   * Times are accumulated in microseconds (integer) to avoid double
//     rounding; printed in milliseconds.
//   * On program exit a `static` sentinel object prints a sorted table
//     to stderr (sort key: total descending).

#if AV_PROFILE_ADAPTER

#include <chrono>
#include <cstdint>

namespace av::depth_map::profile {

// Record one call: `name` MUST be a string literal or a process-lifetime
// string (we store the pointer, no copy). `ms` is the elapsed wall-clock
// time in milliseconds.
void record(const char* name, double ms);

// Force-print the sorted table to stderr right now. (Also fires
// automatically at program exit.)
void dump();

// RAII scope timer. One per forwarder; records elapsed wall-clock time
// from construction → destruction.
struct ScopeTimer {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit ScopeTimer(const char* n)
        : name(n), t0(std::chrono::steady_clock::now()) {}
    ~ScopeTimer() {
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(
                        t1 - t0).count() / 1000.0;
        record(name, ms);
    }
};

}  // namespace av::depth_map::profile

// Two-step token paste: needed so `__LINE__` is expanded before being
// glued to `_av_profile_scope_`. Without the helper macros you get the
// literal identifier `_av_profile_scope___LINE__` repeated on every
// forwarder → all 15 declarations collide → compile error.
#define AV__PROFILE_CAT2(a, b) a##b
#define AV__PROFILE_CAT(a, b)  AV__PROFILE_CAT2(a, b)
#define AV_ADAPTER_PROFILE_SCOPE(name)                                        \
    ::av::depth_map::profile::ScopeTimer                                      \
        AV__PROFILE_CAT(_av_profile_scope_, __LINE__)(name)

#else  // AV_PROFILE_ADAPTER

#define AV_ADAPTER_PROFILE_SCOPE(name)  do { } while (0)

#endif  // AV_PROFILE_ADAPTER
