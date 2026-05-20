// test_smooth_thickness.cpp — validation of
// `av_depth_thickness_smooth_thickness`.
//
// The kernel smooths the thickness channel (.y) of a (depth,
// thickness) map by averaging clamped depth-distances over the
// 3×3 neighborhood. Pixels with `depth <= 0` are skipped (no
// write). Cells with < 3 valid neighbors are also skipped.
//
// Validation: replicate the kernel logic bit-exactly on the CPU
// in FP32. Expect bit-exact agreement modulo neighbor read order
// (the kernel is in-place but only writes `.y` and only reads
// neighbors' `.x` — the per-thread inputs are well-defined).
//
// Test fixtures:
//   * Mixed valid + invalid pixels.
//   * Step in depth so the depth-distance is non-trivial.
//   * Pixels at the ROI edge (only 5 neighbors instead of 8).
//   * Pixels with exactly 2 valid neighbors (no-write case).
//   * Non-multiple-of-16 dimensions to exercise bounds.

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

constexpr std::uint32_t kW = 47;
constexpr std::uint32_t kH = 31;
constexpr float         kMinInflate = 0.5f;
constexpr float         kMaxInflate = 4.0f;

}  // namespace

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    const std::size_t pix = std::size_t(kW) * std::size_t(kH);

    // ---- synthesize input ----
    // Strategy:
    //   * Background gradient: depth = 4 + 0.05 * (x + y)
    //   * Invalid stripe: every 19th column has depth = -1
    //   * Isolated valid pixels (force nb_valid < 3 path):
    //     near the center, mark a 3x3 patch all-invalid except one.
    //   * Thickness: random in [0.02, 0.08]
    std::mt19937_64 rng(0xc0fee);
    std::uniform_real_distribution<float> thick_dist(0.02f, 0.08f);

    std::vector<float> in(pix * 2);
    for (std::uint32_t y = 0; y < kH; ++y)
        for (std::uint32_t x = 0; x < kW; ++x) {
            const std::size_t k = (std::size_t(y) * kW + x) * 2;
            float depth = 4.0f + 0.05f * float(x + y);
            if ((x % 19) == 5) depth = -1.0f;            // invalid stripe
            in[k + 0] = depth;
            in[k + 1] = thick_dist(rng);
        }
    // Force a "fewer-than-3-valid-neighbors" case at (kW/2, kH/2).
    {
        const int cx = int(kW / 2), cy = int(kH / 2);
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                if (dx == 1 && dy == 0) continue;        // keep this one valid
                const std::size_t k = (std::size_t(cy + dy) * kW + (cx + dx)) * 2;
                in[k + 0] = -1.0f;
            }
    }

    // ---- CPU reference (bit-exact replica of the kernel) ----
    std::vector<float> ref = in;
    for (std::uint32_t y = 0; y < kH; ++y)
        for (std::uint32_t x = 0; x < kW; ++x) {
            const std::size_t k = (std::size_t(y) * kW + x) * 2;
            const float cd = in[k + 0];
            const float ct = in[k + 1];
            if (cd <= 0.0f) continue;
            const float min_t = kMinInflate * ct;
            const float max_t = kMaxInflate * ct;
            float sum = 0.0f;
            int nb = 0;
            for (int yp = -1; yp <= 1; ++yp)
                for (int xp = -1; xp <= 1; ++xp) {
                    if (xp == 0 && yp == 0) continue;
                    const int nx = int(x) + xp;
                    const int ny = int(y) + yp;
                    if (nx < 0 || nx >= int(kW) ||
                        ny < 0 || ny >= int(kH)) continue;
                    const std::size_t nk =
                        (std::size_t(ny) * kW + nx) * 2;
                    if (in[nk + 0] > 0.0f) {
                        const float d = std::fabs(cd - in[nk + 0]);
                        sum += std::fmax(min_t, std::fmin(max_t, d));
                        ++nb;
                    }
                }
            if (nb < 3) continue;
            ref[k + 1] = sum / float(nb);
        }

    // ---- GPU ----
    Buffer dev_buf(dev, in.size() * sizeof(float));
    dev_buf.upload(std::span<const float>(in));

    DepthSimMap dsm(dev);
    dsm.smooth_thickness(dev_buf, kW, kH, kMinInflate, kMaxInflate);

    const auto* gpu = static_cast<const float*>(dev_buf.data());

    int bad             = 0;
    int touched         = 0;
    int preserved       = 0;
    int invalid_skipped = 0;
    int few_neighbors   = 0;
    for (std::size_t k = 0; k < pix; ++k) {
        const float in_d = in[k * 2 + 0];
        const float gpu_d = gpu[k * 2 + 0];
        const float gpu_t = gpu[k * 2 + 1];
        const float ref_d = ref[k * 2 + 0];
        const float ref_t = ref[k * 2 + 1];

        // Depth must never change (kernel only writes .y).
        const std::uint32_t in_d_bits  = std::bit_cast<std::uint32_t>(in_d);
        const std::uint32_t gpu_d_bits = std::bit_cast<std::uint32_t>(gpu_d);
        if (in_d_bits != gpu_d_bits) {
            if (bad < 4) std::fprintf(stderr,
                "k=%zu: depth changed (in=%g gpu=%g)\n",
                k, double(in_d), double(gpu_d));
            ++bad; continue;
        }

        // Thickness should agree with the FP32 reference. The
        // kernel uses MSL `max(min_t, min(max_t, d))`; the Metal
        // compiler may fuse this into a `clamp` intrinsic with
        // sub-ULP-level rounding differences vs std::fmax/fmin.
        // We accept a few-ULP relative error.
        const float a_t = gpu_t, b_t = ref_t;
        const float diff = std::fabs(a_t - b_t);
        const float rel  = diff / std::fmax(std::fabs(a_t) + std::fabs(b_t), 1e-30f);
        if (rel > 1e-6f) {
            if (bad < 4) std::fprintf(stderr,
                "k=%zu: thickness mismatch (gpu=%.9g ref=%.9g  diff=%.3e rel=%.3e)\n",
                k, double(gpu_t), double(ref_t),
                double(diff), double(rel));
            ++bad;
        }

        // Bookkeeping.
        if (in_d <= 0.0f) ++invalid_skipped;
        else if (std::bit_cast<std::uint32_t>(in[k * 2 + 1]) ==
                 std::bit_cast<std::uint32_t>(gpu_t)) ++preserved;
        else ++touched;
        (void)ref_d;  // unused but mirrors structure
    }

    // Sanity: count the few-neighbor case at the synthetic site.
    {
        const int cx = int(kW / 2), cy = int(kH / 2);
        const std::size_t k = std::size_t(cy) * kW + cx;
        if (std::bit_cast<std::uint32_t>(in[k * 2 + 1]) ==
            std::bit_cast<std::uint32_t>(gpu[k * 2 + 1])) {
            ++few_neighbors;
        }
    }

    std::printf("[info] roi          : %u × %u = %zu pixels\n",
                kW, kH, pix);
    std::printf("[info] touched      : %d  preserved : %d  invalid : %d\n",
                touched, preserved, invalid_skipped);
    std::printf("[info] center few-neighbor preserved : %s\n",
                few_neighbors ? "yes" : "no");
    std::printf("[info] mismatches   : %d\n", bad);

    if (bad) {
        std::fprintf(stderr, "FAIL: %d mismatches\n", bad);
        return 1;
    }
    if (touched == 0) {
        std::fprintf(stderr,
            "FAIL: no pixels were actually smoothed — test fixture too weak\n");
        return 1;
    }
    if (!few_neighbors) {
        std::fprintf(stderr,
            "FAIL: center few-neighbor case was modified — preservation broken\n");
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
