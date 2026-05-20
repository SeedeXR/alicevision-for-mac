// test_cuda_host_shim.cpp — sanity checks for the four upstream-
// facing shim header/source pairs created in S38 W1:
//
//   * aliceVision/depthMap/cuda/host/DeviceCache.hpp
//   * aliceVision/depthMap/cuda/host/DeviceStreamManager.hpp
//   * aliceVision/depthMap/cuda/host/patchPattern.hpp
//   * aliceVision/depthMap/cuda/host/utils.hpp
//
// The goal is to prove each shim symbol resolves at link time and
// behaves sanely without requiring upstream's depthMap host code
// itself (which is staged in a follow-up). Each block is small —
// the heavy lifting is exercised by the existing
// `test_device_cache`, `test_device_stream_manager`, etc. which
// already cover the underlying `av::depth_map::*` impls.
//
// Linkage: this test pulls in `av_cuda_host_shim`, which transitively
// pulls in `av::depth_map_metal` and the upstream `aliceVision_*`
// libs. The shim include path is on `PUBLIC`, so the upstream-
// style includes below resolve to our shims.

#include "aliceVision/depthMap/cuda/host/DeviceCache.hpp"
#include "aliceVision/depthMap/cuda/host/DeviceStreamManager.hpp"
#include "aliceVision/depthMap/cuda/host/patchPattern.hpp"
#include "aliceVision/depthMap/cuda/host/utils.hpp"
#include "aliceVision/depthMap/cuda/host/memory.hpp"   // set_adapter_device

#include "av/depth_map/upstream_adapter.hpp"           // current_patch_pattern
#include "av/depth_map/DevicePatchPattern.hpp"
#include "av/gpu/Device.hpp"

#include <aliceVision/depthMap/CustomPatchPatternParams.hpp>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

int g_failed = 0;

#define EXPECT(expr, msg) do {                                     \
    if (!(expr)) {                                                 \
        std::fprintf(stderr, "[fail] %s (line %d): %s\n",          \
                     msg, __LINE__, #expr);                        \
        ++g_failed;                                                \
    }                                                              \
} while (0)

void test_utils() {
    using namespace aliceVision::depthMap;

    const int n = listCudaDevices();
    EXPECT(n == 1, "listCudaDevices should return 1 on Apple Silicon");

    EXPECT(getCudaDeviceId() == 0,    "getCudaDeviceId should return 0");
    EXPECT(testCudaDeviceId(0) == true,  "testCudaDeviceId(0) should be true");
    EXPECT(testCudaDeviceId(1) == false, "testCudaDeviceId(1) should be false");

    setCudaDeviceId(0);  // no-op
    setCudaDeviceId(7);  // logs warning, no-op

    double a = 0, u = 0, t = 0;
    getDeviceMemoryInfo(a, u, t);
    EXPECT(t > 0.0,           "total memory should be > 0 MB");
    EXPECT(a > 0.0,           "available memory should be > 0 MB");
    EXPECT(a + u <= t * 1.05, "avail + used should be ~= total (5% slack)");
    EXPECT(t >  1024.0,       "total memory should exceed 1 GB on a dev Mac");
    std::printf("[info] memory : avail=%.1f MB used=%.1f MB total=%.1f MB\n",
                a, u, t);

    logDeviceMemoryInfo();  // prints to stderr
}

void test_device_cache() {
    using aliceVision::depthMap::DeviceCache;

    DeviceCache& c = DeviceCache::getInstance();
    c.build(/*maxMipmap=*/2, /*maxCamera=*/2);

    // Idempotency of clear() — no entries to drop, must not crash.
    c.clear();

    // Re-build to leave the singleton in a sane state for the rest
    // of the test process.
    c.build(2, 2);
}

void test_device_stream_manager() {
    using aliceVision::depthMap::DeviceStreamManager;

    DeviceStreamManager mgr(4);
    EXPECT(mgr.getNbStreams() == 4, "nb streams mismatch");

    auto s0 = mgr.getStream(0);
    EXPECT(s0 != nullptr, "getStream(0) returned null");

    auto s_wrap = mgr.getStream(4);
    EXPECT(s_wrap == s0, "modular indexing should wrap: getStream(4) == getStream(0)");

    mgr.waitStream(0);  // must return without blocking forever
    mgr.waitStream(7);  // wraps to slot 3; still must return
}

void test_patch_pattern() {
    using namespace aliceVision::depthMap;
    namespace ad = av::depth_map::upstream_adapter;

    CustomPatchPatternParams p;
    p.groupSubpartsPerLevel = false;
    CustomPatchPatternParams::SubpartParams s;
    s.isCircle      = true;
    s.level         = 0;
    s.nbCoordinates = 12;
    s.radius        = 2.5f;
    s.weight        = 1.0f;
    p.subpartsParams.push_back(s);

    buildCustomPatchPattern(p);

    const av::depth_map::DevicePatchPattern& pat = ad::current_patch_pattern();
    EXPECT(pat.nbSubparts == 1, "nbSubparts should be 1");
    EXPECT(pat.subparts[0].nbCoordinates == 12, "subpart 0 nbCoordinates");
    EXPECT(pat.subparts[0].isCircle == 1,        "subpart 0 isCircle");
    EXPECT(pat.subparts[0].weight   == 1.0f,     "subpart 0 weight");

    // A second build with two subparts should replace, not append.
    CustomPatchPatternParams p2;
    p2.groupSubpartsPerLevel = false;
    CustomPatchPatternParams::SubpartParams sA = s; sA.nbCoordinates = 8;
    CustomPatchPatternParams::SubpartParams sB = s; sB.level = 1; sB.nbCoordinates = 6;
    p2.subpartsParams.push_back(sA);
    p2.subpartsParams.push_back(sB);
    buildCustomPatchPattern(p2);

    const auto& pat2 = ad::current_patch_pattern();
    EXPECT(pat2.nbSubparts == 2, "second build: nbSubparts should be 2");
    EXPECT(pat2.subparts[0].nbCoordinates == 8, "second build subpart 0 coords");
    EXPECT(pat2.subparts[1].nbCoordinates == 6, "second build subpart 1 coords");

    // Validation: empty params should throw.
    bool threw = false;
    try {
        CustomPatchPatternParams empty;
        buildCustomPatchPattern(empty);
    } catch (const std::exception&) {
        threw = true;
    }
    EXPECT(threw, "empty CustomPatchPatternParams should throw");
}

}  // namespace

int main() try {
    // Required: every shim that allocates GPU resources needs the
    // process-global Device wired in.
    av::gpu::Device dev = av::gpu::Device::default_device();
    dev.load_library({});
    aliceVision::depthMap::set_adapter_device(dev);
    std::printf("[info] device : %s\n", dev.name().c_str());

    test_utils();
    test_device_cache();
    test_device_stream_manager();
    test_patch_pattern();

    if (g_failed) {
        std::fprintf(stderr, "[FAIL] %d assertion(s) failed\n", g_failed);
        return 1;
    }
    std::printf("[pass] test_cuda_host_shim\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "[exception] %s\n", e.what());
    return 2;
}
