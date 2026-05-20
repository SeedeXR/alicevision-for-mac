// test_depth_sim_map_copy.cpp — validation of
// `depthSimMapCopyDepthOnly_kernel`.
//
// The kernel reads (depth, _) per pixel from `in_map`, writes
// (depth, default_sim) to `out_map`. Pure copy — no cameras, no
// textures, no arithmetic beyond the per-pixel write. Expected:
// bit-exact agreement with a trivial CPU reference.
//
// Test coverage:
//   * Width/height are non-multiples of the dispatch threadgroup
//     size (16) — exercises the bounds check.
//   * Input depth values include valid (>0), invalid (-1), and
//     NaN — verifies the kernel doesn't filter anything out
//     (it shouldn't — that's another kernel's job).
//   * Input sim is non-zero — verifies it is *replaced*, not
//     OR-ed or added.

#include "av/depth_map/DepthSimMap.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::uint32_t kW = 173;   // intentionally non-multiple-of-16
constexpr std::uint32_t kH =  91;
constexpr float         kDefaultSim = -0.42f;

}  // namespace

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    const std::size_t pix = std::size_t(kW) * std::size_t(kH);

    // Synthesize input float2 map. Mix valid / invalid / NaN /
    // huge / tiny values to make sure the kernel never inspects
    // the data — pure copy.
    std::mt19937_64 rng(0xfeedu);
    std::uniform_real_distribution<float> depth_dist(0.1f, 50.0f);
    std::uniform_real_distribution<float> sim_dist(-1.0f, 1.0f);

    std::vector<float> in(pix * 2);
    for (std::size_t k = 0; k < pix; ++k) {
        float depth;
        if ((k % 37) == 0)       depth = -1.0f;
        else if ((k % 53) == 0)  depth = std::nanf("");
        else if ((k % 79) == 0)  depth = 1e30f;
        else if ((k % 97) == 0)  depth = 0.0f;
        else                     depth = depth_dist(rng);
        in[k * 2 + 0] = depth;
        in[k * 2 + 1] = sim_dist(rng);
    }

    Buffer in_buf (dev, in.size() * sizeof(float));
    Buffer out_buf(dev, in.size() * sizeof(float));
    in_buf.upload(std::span<const float>(in));

    // Pre-fill the output buffer with a sentinel — the kernel
    // should overwrite every byte inside the ROI. Anything outside
    // the ROI (none here, ROI = full map) would remain at sentinel.
    std::vector<float> sentinel_fill(in.size(), -7777.0f);
    out_buf.upload(std::span<const float>(sentinel_fill));

    DepthSimMap dsm(dev);
    dsm.copy_depth_only(out_buf, in_buf, kW, kH, kDefaultSim);

    // ---- validate ----
    const auto* gpu = static_cast<const float*>(out_buf.data());

    int bad = 0;
    for (std::size_t k = 0; k < pix; ++k) {
        const float in_depth  = in[k * 2 + 0];
        const float gpu_depth = gpu[k * 2 + 0];
        const float gpu_sim   = gpu[k * 2 + 1];

        // Bit-exact equality except for NaN (NaN != NaN).
        bool depth_ok;
        if (std::isnan(in_depth)) {
            depth_ok = std::isnan(gpu_depth);
        } else {
            depth_ok = (std::bit_cast<std::uint32_t>(in_depth) ==
                        std::bit_cast<std::uint32_t>(gpu_depth));
        }
        const bool sim_ok =
            (std::bit_cast<std::uint32_t>(kDefaultSim) ==
             std::bit_cast<std::uint32_t>(gpu_sim));

        if (!depth_ok || !sim_ok) {
            if (bad < 4) std::fprintf(stderr,
                "k=%zu: in_depth=%g gpu=(%g, %g)  (depth_ok=%d sim_ok=%d)\n",
                k, static_cast<double>(in_depth),
                static_cast<double>(gpu_depth),
                static_cast<double>(gpu_sim),
                int(depth_ok), int(sim_ok));
            ++bad;
        }
    }

    std::printf("[info] roi          : %u × %u = %zu pixels\n",
                kW, kH, pix);
    std::printf("[info] default_sim  : %g\n", double(kDefaultSim));
    std::printf("[info] mismatches   : %d\n", bad);

    if (bad) {
        std::fprintf(stderr, "FAIL: %d mismatches\n", bad);
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
