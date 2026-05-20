// test_device_cache.cpp — validation of `DeviceCache` (Phase 8).
//
// Coverage:
//   * Mipmap pool: insert until full, then insert one more →
//     LRU eviction. Verify the evicted entry is gone and the
//     surviving entries are still queryable.
//   * Camera-params pool: same pattern, keyed by (camera_id,
//     downscale). Verify the cached params struct round-trips
//     bit-exactly.
//   * `clear()` drops everything.
//   * Error paths: requesting an un-cached entry throws.
//
// We separately test the `LRUCache<int>` template's invariants
// (insert order → slot reuse, eviction signaling) since they're
// the core machinery DeviceCache builds on.

#include "av/depth_map/DeviceCache.hpp"
#include "av/depth_map/LRUCache.hpp"
#include "av/depth_map/PatchOps.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

using av::depth_map::DeviceCache;
using av::depth_map::DeviceCameraParams;
using av::depth_map::LRUCache;
using av::depth_map::CameraPair;

std::vector<float> make_rgba(std::uint32_t w, std::uint32_t h, int seed) {
    std::vector<float> px(std::size_t(w) * h * 4);
    for (std::uint32_t j = 0; j < h; ++j)
        for (std::uint32_t i = 0; i < w; ++i) {
            const float u = float(i) / float(w);
            const float v = float(j) / float(h);
            const std::size_t k = (j * w + i) * 4;
            px[k+0] = 128.0f + 60.0f * std::sin(float(seed) + 11.0f*u);
            px[k+1] = 128.0f + 40.0f * std::cos(float(seed) + 13.0f*v);
            px[k+2] = 128.0f + 50.0f * std::sin(float(seed*7) + 17.0f*(u+v));
            px[k+3] = 255.0f;
        }
    return px;
}

DeviceCameraParams make_cam(int seed) {
    DeviceCameraParams cp{};
    // Fill with deterministic but distinguishable values.
    for (int i = 0; i < 12; ++i) cp.P[i]      = 0.1f * (i + seed);
    for (int i = 0; i <  9; ++i) cp.iP[i]     = 0.2f * (i + seed);
    for (int i = 0; i <  9; ++i) cp.R[i]      = 0.3f * (i + seed);
    for (int i = 0; i <  9; ++i) cp.iR[i]     = 0.4f * (i + seed);
    for (int i = 0; i <  9; ++i) cp.K[i]      = 0.5f * (i + seed);
    for (int i = 0; i <  9; ++i) cp.iK[i]     = 0.6f * (i + seed);
    cp.C[0]      = 1.0f * seed;  cp.C[1]      = 2.0f * seed;  cp.C[2]      = 3.0f * seed;
    cp.XVect[0]  = 0.7f * seed;  cp.XVect[1]  = 0.8f;        cp.XVect[2]  = 0.0f;
    cp.YVect[0]  = 0.0f;         cp.YVect[1]  = 0.9f * seed; cp.YVect[2]  = 0.0f;
    cp.ZVect[0]  = 0.0f;         cp.ZVect[1]  = 0.0f;        cp.ZVect[2]  = 1.1f * seed;
    return cp;
}

bool cam_eq(const DeviceCameraParams& a, const DeviceCameraParams& b) {
    auto eqf = [](float x, float y) {
        return std::bit_cast<std::uint32_t>(x) ==
               std::bit_cast<std::uint32_t>(y);
    };
    for (int i = 0; i < 12; ++i) if (!eqf(a.P[i],  b.P[i]))  return false;
    for (int i = 0; i <  9; ++i) if (!eqf(a.iP[i], b.iP[i])) return false;
    for (int i = 0; i <  9; ++i) if (!eqf(a.R[i],  b.R[i]))  return false;
    for (int i = 0; i <  9; ++i) if (!eqf(a.iR[i], b.iR[i])) return false;
    for (int i = 0; i <  9; ++i) if (!eqf(a.K[i],  b.K[i]))  return false;
    for (int i = 0; i <  9; ++i) if (!eqf(a.iK[i], b.iK[i])) return false;
    for (int i = 0; i <  3; ++i) {
        if (!eqf(a.C[i],     b.C[i])     ||
            !eqf(a.XVect[i], b.XVect[i]) ||
            !eqf(a.YVect[i], b.YVect[i]) ||
            !eqf(a.ZVect[i], b.ZVect[i])) return false;
    }
    return true;
}

int test_lru_template() {
    int failed = 0;
    LRUCache<int> c(3);
    int pos = -1; int old_val = -1;

    // Insert into empty cache: should return true, no eviction.
    if (!c.insert(10, pos, old_val) || pos != 0 || old_val != -1) {
        std::fprintf(stderr, "LRU: insert(10) bad\n"); ++failed;
    }
    if (!c.insert(20, pos, old_val) || pos != 1 || old_val != -1) {
        std::fprintf(stderr, "LRU: insert(20) bad\n"); ++failed;
    }
    if (!c.insert(30, pos, old_val) || pos != 2 || old_val != -1) {
        std::fprintf(stderr, "LRU: insert(30) bad\n"); ++failed;
    }
    if (c.size() != 3) { std::fprintf(stderr, "LRU: size wrong\n"); ++failed; }

    // Re-insert existing: returns false, no eviction. This also
    // moves 10 to MRU position.
    if (c.insert(10, pos, old_val) || pos != 0 || old_val != -1) {
        std::fprintf(stderr, "LRU: re-insert(10) bad\n"); ++failed;
    }
    // Now LRU order is [20, 30, 10]; 20 is least recent.

    // Insert 40: should evict 20.
    if (!c.insert(40, pos, old_val) || old_val != 20) {
        std::fprintf(stderr, "LRU: insert(40) didn't evict 20, got %d\n", old_val);
        ++failed;
    }
    if (c.get_index(20) != -1) {
        std::fprintf(stderr, "LRU: 20 still cached after eviction\n"); ++failed;
    }
    if (c.get_index(10) < 0 || c.get_index(30) < 0 || c.get_index(40) < 0) {
        std::fprintf(stderr, "LRU: surviving entries missing\n"); ++failed;
    }
    return failed;
}

}  // namespace

int main() try {
    using namespace av::gpu;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    int failed = 0;

    // ---- LRU template ----
    failed += test_lru_template();
    std::printf("[lru ] template invariants ok\n");

    // ---- DeviceCache: mipmap pool ----
    constexpr int kMaxMipmaps     = 3;
    constexpr int kMaxCameras     = 4;
    DeviceCache cache(dev, kMaxMipmaps, kMaxCameras);

    constexpr std::uint32_t kW = 32, kH = 24;
    const auto img_a = make_rgba(kW, kH, 1);
    const auto img_b = make_rgba(kW, kH, 2);
    const auto img_c = make_rgba(kW, kH, 3);
    const auto img_d = make_rgba(kW, kH, 4);

    cache.add_mipmap_image(/*camera_id=*/10, /*min=*/2, /*max=*/8,
        std::span<const float>(img_a), kW, kH);
    cache.add_mipmap_image(11, 2, 8, std::span<const float>(img_b), kW, kH);
    cache.add_mipmap_image(12, 2, 8, std::span<const float>(img_c), kW, kH);

    if (cache.mipmap_count() != 3) {
        std::fprintf(stderr, "mipmap_count after 3 adds: %d (expected 3)\n",
                     cache.mipmap_count());
        ++failed;
    }
    // All three should be retrievable.
    for (int id : {10, 11, 12}) {
        try {
            (void)cache.request_mipmap_image(id);
        } catch (...) {
            std::fprintf(stderr,
                "request_mipmap_image(%d) threw unexpectedly\n", id);
            ++failed;
        }
    }
    // Refresh 10 (move to MRU), then insert 13 (overflow → 11 evicted).
    cache.add_mipmap_image(10, 2, 8, std::span<const float>(img_a), kW, kH);
    cache.add_mipmap_image(13, 2, 8, std::span<const float>(img_d), kW, kH);
    bool threw_for_11 = false;
    try { (void)cache.request_mipmap_image(11); }
    catch (const std::out_of_range&) { threw_for_11 = true; }
    if (!threw_for_11) {
        std::fprintf(stderr,
            "mipmap 11 should have been evicted after refresh(10)+insert(13)\n");
        ++failed;
    }
    // 10, 12, 13 should still be cached.
    for (int id : {10, 12, 13}) {
        try { (void)cache.request_mipmap_image(id); }
        catch (...) {
            std::fprintf(stderr,
                "request_mipmap_image(%d) threw after eviction expected to spare it\n", id);
            ++failed;
        }
    }
    std::printf("[mip ] LRU eviction ok\n");

    // ---- DeviceCache: camera-params pool ----
    const DeviceCameraParams cam_10_4 = make_cam(10);
    const DeviceCameraParams cam_11_2 = make_cam(11);
    const DeviceCameraParams cam_12_2 = make_cam(12);
    const DeviceCameraParams cam_13_8 = make_cam(13);
    const DeviceCameraParams cam_14_4 = make_cam(14);

    cache.add_camera_params(10, 4, cam_10_4);
    cache.add_camera_params(11, 2, cam_11_2);
    cache.add_camera_params(12, 2, cam_12_2);
    cache.add_camera_params(13, 8, cam_13_8);
    if (cache.camera_param_count() != 4) {
        std::fprintf(stderr, "camera_param_count after 4 adds: %d (expected 4)\n",
                     cache.camera_param_count());
        ++failed;
    }
    // All retrievable + bit-exact.
    struct Q { int id, ds; const DeviceCameraParams* expected; };
    const Q queries[] = {
        { 10, 4, &cam_10_4 },
        { 11, 2, &cam_11_2 },
        { 12, 2, &cam_12_2 },
        { 13, 8, &cam_13_8 },
    };
    for (const auto& q : queries) {
        const auto& got = cache.request_camera_params(q.id, q.ds);
        if (!cam_eq(got, *q.expected)) {
            std::fprintf(stderr,
                "cam (%d,%d) mismatch\n", q.id, q.ds);
            ++failed;
        }
    }
    // Refresh (10, 4), then insert (14, 4) → evicts (11, 2).
    cache.add_camera_params(10, 4, cam_10_4);
    cache.add_camera_params(14, 4, cam_14_4);
    bool threw_for_11_2 = false;
    try { (void)cache.request_camera_params(11, 2); }
    catch (const std::out_of_range&) { threw_for_11_2 = true; }
    if (!threw_for_11_2) {
        std::fprintf(stderr,
            "cam (11, 2) should have been evicted after refresh+insert\n");
        ++failed;
    }
    // (10, 4), (12, 2), (13, 8), (14, 4) survive — and same params still.
    if (!cam_eq(cache.request_camera_params(10, 4), cam_10_4) ||
        !cam_eq(cache.request_camera_params(12, 2), cam_12_2) ||
        !cam_eq(cache.request_camera_params(13, 8), cam_13_8) ||
        !cam_eq(cache.request_camera_params(14, 4), cam_14_4)) {
        std::fprintf(stderr,
            "surviving camera params don't match\n");
        ++failed;
    }
    // (10, 4) and (10, 2) are different keys — adding (10, 2)
    // should NOT collide with (10, 4).
    const auto cam_10_2 = make_cam(99);
    cache.add_camera_params(10, 2, cam_10_2);
    // Cache may now have evicted whatever was LRU. We just verify
    // that the new (10, 2) is queryable AND that (10, 4) hasn't
    // morphed into it (because it was just refreshed above so it
    // shouldn't be the LRU victim).
    if (!cam_eq(cache.request_camera_params(10, 2), cam_10_2)) {
        std::fprintf(stderr, "(10, 2) not stored correctly\n");
        ++failed;
    }
    if (!cam_eq(cache.request_camera_params(10, 4), cam_10_4)) {
        std::fprintf(stderr,
            "(10, 4) collided with (10, 2) — pair keying broken\n");
        ++failed;
    }
    std::printf("[cam ] LRU eviction + (id, downscale) keying ok\n");

    // ---- clear() ----
    cache.clear();
    if (cache.mipmap_count() != 0 || cache.camera_param_count() != 0) {
        std::fprintf(stderr, "clear() didn't reset counts\n");
        ++failed;
    }
    // Any prior request now throws.
    try {
        (void)cache.request_mipmap_image(10);
        std::fprintf(stderr, "request after clear() didn't throw\n");
        ++failed;
    } catch (const std::out_of_range&) { /* expected */ }
    std::printf("[clr ] clear() ok\n");

    if (failed) {
        std::fprintf(stderr, "FAIL: %d issues\n", failed);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
catch (const av::gpu::GpuError& e) {
    std::fprintf(stderr, "GpuError: %s\n", e.what());
    return 2;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "exception: %s\n", e.what());
    return 2;
}
