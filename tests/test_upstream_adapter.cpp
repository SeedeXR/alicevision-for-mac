// test_upstream_adapter.cpp — end-to-end exercise of the Phase 8
// `cuda_*` adapter forwarders defined in
// `src/depth_map_metal/src/upstream_adapter.cpp`.
//
// We don't link upstream's depthMap (yet); instead we construct
// the upstream-typed arguments via the type-shim
// (`CudaDeviceMemoryPitched`, `CudaSize`, `DeviceMipmapImage`),
// invoke a representative subset of forwarders, and verify the
// result by reading back `gpu_buffer().data()`.
//
// Coverage (3 of 12):
//   1. `cuda_volumeInitialize<TSim>`         — the simplest; pure
//                                              buffer write.
//   2. `cuda_depthSimMapCopyDepthOnly`       — float2 copy, drops
//                                              the .y channel and
//                                              substitutes `defaultSim`.
//   3. `cuda_normalMapUpscale`               — packed_float3 nearest-
//                                              neighbor upscale, mirrors
//                                              `test_normal_map_upscale`.
//
// The remaining 9 forwarders are mechanical (param packing + a
// call to our `Volume::*` / `DepthSimMap::*`); they get exercised
// when upstream's depthMap target lands. For S35 this test
// proves the pattern compiles and works end-to-end.

#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"
#include "av/depth_map/DeviceMipmapImage.hpp"

// Pull the type-shim in directly (same pattern as
// `test_cuda_memory_shim.cpp`).
#include "../cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/memory.hpp"
#include "../cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/DeviceMipmapImage.hpp"

// And the matching local definitions for Range / mvsData::ROI /
// SgmParams / RefineParams that the adapter .cpp uses.
#include "../src/depth_map_metal/src/upstream_adapter_types.hpp"

// The adapter header.
#include "av/depth_map/upstream_adapter.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// CPU reference for cuda_normalMapUpscale (mirrors
// test_normal_map_upscale.cpp's `cpu_reference`).
void cpu_normal_upscale_ref(const std::vector<float>& in,
                            std::vector<float>&       out,
                            std::uint32_t in_w, std::uint32_t in_h,
                            std::uint32_t out_w, std::uint32_t out_h)
{
    const float ratio = float(in_w) / float(out_w);
    out.assign(std::size_t(out_w) * std::size_t(out_h) * 3, 0.0f);
    const int max_x = int(in_w) - 1;
    const int max_y = int(in_h) - 1;
    for (std::uint32_t y = 0; y < out_h; ++y)
        for (std::uint32_t x = 0; x < out_w; ++x) {
            const float ox = (float(x) - 0.5f) * ratio;
            const float oy = (float(y) - 0.5f) * ratio;
            int xp = int(std::floor(ox + 0.5f));
            int yp = int(std::floor(oy + 0.5f));
            if (xp > max_x) xp = max_x;
            if (yp > max_y) yp = max_y;
            const std::size_t ok = (std::size_t(y) * out_w + x) * 3;
            const std::size_t ik = (std::size_t(yp) * in_w + xp) * 3;
            out[ok + 0] = in[ik + 0];
            out[ok + 1] = in[ik + 1];
            out[ok + 2] = in[ik + 2];
        }
}

}  // namespace

int main() try {
    using namespace av::gpu;
    namespace av_dm = av::depth_map;
    namespace ali   = aliceVision::depthMap;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    // Wire the adapter device — required before any
    // CudaDeviceMemoryPitched / Volume / DepthSimMap singleton
    // construction.
    ali::set_adapter_device(dev);

    int failed = 0;

    // ===================================================================
    // 1. cuda_volumeInitialize<TSim>
    // ===================================================================
    {
        constexpr std::size_t X = 32, Y = 24, Z = 8;
        ali::CudaSize<3> sz(X, Y, Z);
        ali::CudaDeviceMemoryPitched<ali::TSim, 3> vol(sz);

        // Pre-fill with a contrasting sentinel.
        {
            auto* p = vol.getBuffer();
            std::memset(p, 0xAA, X * Y * Z);
        }

        constexpr ali::TSim kVal = 0x42;
        ali::cuda_volumeInitialize(vol, kVal, /*stream=*/nullptr);

        // Read back.
        const auto* gpu = vol.getBuffer();
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < X * Y * Z; ++i)
            if (gpu[i] != kVal) ++mismatches;
        if (mismatches) {
            std::fprintf(stderr,
                "[fail] cuda_volumeInitialize<TSim>: %zu of %zu mismatched\n",
                mismatches, X * Y * Z);
            ++failed;
        } else {
            std::printf("[ok]  cuda_volumeInitialize<TSim>: "
                        "%zu×%zu×%zu = 0x%02x\n",
                        X, Y, Z, kVal);
        }
    }

    // ===================================================================
    // 2. cuda_depthSimMapCopyDepthOnly
    // ===================================================================
    {
        constexpr std::uint32_t W = 41;   // non-multiple of 16 deliberately
        constexpr std::uint32_t H = 27;
        constexpr float kDefaultSim = -0.1234f;

        ali::CudaSize<2> dim(W, H);
        ali::CudaDeviceMemoryPitched<float2, 2> in_dmp (dim);
        ali::CudaDeviceMemoryPitched<float2, 2> out_dmp(dim);

        // Fingerprint the input depth + sim.
        auto* in_p = in_dmp.getBuffer();
        for (std::size_t k = 0; k < W * H; ++k) {
            in_p[k].x = static_cast<float>(k) * 0.25f + 1.0f;
            in_p[k].y = static_cast<float>(k) * 0.5f  - 7.0f;   // should be IGNORED
        }
        // Sentinel-fill the output so anything unwritten shows up.
        {
            auto* out_p = out_dmp.getBuffer();
            for (std::size_t k = 0; k < W * H; ++k) {
                out_p[k].x = -7777.0f;
                out_p[k].y = -7777.0f;
            }
        }

        ali::cuda_depthSimMapCopyDepthOnly(out_dmp, in_dmp, kDefaultSim,
                                            /*stream=*/nullptr);

        const auto* gpu = out_dmp.getBuffer();
        int bad = 0;
        for (std::size_t k = 0; k < W * H; ++k) {
            const std::uint32_t want_d = std::bit_cast<std::uint32_t>(in_p[k].x);
            const std::uint32_t got_d  = std::bit_cast<std::uint32_t>(gpu[k].x);
            const std::uint32_t want_s = std::bit_cast<std::uint32_t>(kDefaultSim);
            const std::uint32_t got_s  = std::bit_cast<std::uint32_t>(gpu[k].y);
            if (want_d != got_d || want_s != got_s) {
                if (bad < 4) {
                    std::fprintf(stderr,
                        "[depthCopy] k=%zu in.d=%g got.d=%g got.s=%g (want.s=%g)\n",
                        k, double(in_p[k].x), double(gpu[k].x),
                        double(gpu[k].y), double(kDefaultSim));
                }
                ++bad;
            }
        }
        if (bad) {
            std::fprintf(stderr,
                "[fail] cuda_depthSimMapCopyDepthOnly: %d mismatches\n", bad);
            ++failed;
        } else {
            std::printf("[ok]  cuda_depthSimMapCopyDepthOnly: %u×%u\n", W, H);
        }
    }

    // ===================================================================
    // 3. cuda_normalMapUpscale
    // ===================================================================
    {
        constexpr std::uint32_t in_w = 4,  in_h = 3;
        constexpr std::uint32_t out_w = 8, out_h = 6;

        ali::CudaSize<2> in_dim (in_w,  in_h);
        ali::CudaSize<2> out_dim(out_w, out_h);
        ali::CudaDeviceMemoryPitched<float3, 2> in_dmp (in_dim);
        ali::CudaDeviceMemoryPitched<float3, 2> out_dmp(out_dim);

        // Unique fingerprint per input pixel.
        std::vector<float> in_flat(std::size_t(in_w) * in_h * 3);
        for (std::uint32_t y = 0; y < in_h; ++y)
            for (std::uint32_t x = 0; x < in_w; ++x) {
                const std::size_t k = (std::size_t(y) * in_w + x) * 3;
                in_flat[k + 0] = float(x) * 1000.0f + float(y);
                in_flat[k + 1] = float(x) - float(y) * 0.25f;
                in_flat[k + 2] = -float(x) * 0.5f + float(y) * 7.0f + 0.125f;
            }
        {
            // Copy into the shim-backed input buffer.
            auto* p = reinterpret_cast<float*>(in_dmp.getBuffer());
            std::memcpy(p, in_flat.data(), in_flat.size() * sizeof(float));
        }
        // Sentinel-fill the output.
        {
            auto* p = reinterpret_cast<float*>(out_dmp.getBuffer());
            for (std::size_t k = 0; k < std::size_t(out_w) * out_h * 3; ++k)
                p[k] = -7777.0f;
        }

        aliceVision::ROI roi{};
        roi.x.begin = 0; roi.x.end = out_w;
        roi.y.begin = 0; roi.y.end = out_h;

        ali::cuda_normalMapUpscale(out_dmp, in_dmp, roi, /*stream=*/nullptr);

        // Reference.
        std::vector<float> ref_flat;
        cpu_normal_upscale_ref(in_flat, ref_flat,
                               in_w, in_h, out_w, out_h);

        const auto* gpu = reinterpret_cast<const float*>(out_dmp.getBuffer());
        int bad = 0;
        for (std::size_t k = 0; k < ref_flat.size(); ++k) {
            const std::uint32_t a = std::bit_cast<std::uint32_t>(gpu[k]);
            const std::uint32_t b = std::bit_cast<std::uint32_t>(ref_flat[k]);
            if (a != b) ++bad;
        }
        if (bad) {
            std::fprintf(stderr,
                "[fail] cuda_normalMapUpscale: %d / %zu mismatches\n",
                bad, ref_flat.size());
            ++failed;
        } else {
            std::printf("[ok]  cuda_normalMapUpscale: %u×%u → %u×%u\n",
                        in_w, in_h, out_w, out_h);
        }
    }

    if (failed) {
        std::fprintf(stderr, "FAIL: %d cases\n", failed);
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
