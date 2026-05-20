// test_normal_map_upscale.cpp — validation of
// `av_map_upscale_float3` (used by upstream's `cuda_normalMapUpscale`).
//
// Nearest-neighbor upscale. The kernel uses upstream's exact
// formula: `xp = floor((x - 0.5) * ratio + 0.5)`, `yp` likewise,
// clamped to `in_dim - 1`. We replicate that arithmetic on the
// CPU and compare bit-exactly.
//
// Coverage:
//   * Integer upscales (2×, 3×) where every output cell maps to
//     a well-defined input cell.
//   * Non-integer ratio (9×6 ← 4×3 = 2.25× horiz / 2.0× vert)
//     to exercise the FP rounding path.
//   * Dispatch dimensions that are *not* multiples of 16 to
//     stress the bounds-check on partial threadgroups.
//   * Unique per-pixel fingerprint values so any indexing error
//     surfaces.

#include "av/depth_map/DepthSimMap.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

struct Case {
    std::uint32_t in_w, in_h, out_w, out_h;
    const char*   name;
};

// Generate a unique per-pixel fingerprint. Each component encodes
// (x, y) in a way no other pixel could produce.
void fill_fingerprint(std::vector<float>& buf,
                      std::uint32_t w, std::uint32_t h)
{
    buf.resize(std::size_t(w) * std::size_t(h) * 3);
    for (std::uint32_t y = 0; y < h; ++y)
        for (std::uint32_t x = 0; x < w; ++x) {
            const std::size_t k = (std::size_t(y) * w + x) * 3;
            buf[k + 0] = float(x) * 1000.0f + float(y);
            buf[k + 1] = float(x) - float(y) * 0.25f;
            buf[k + 2] = -float(x) * 0.5f + float(y) * 7.0f + 0.125f;
        }
}

// Replicate upstream's nearest-neighbor formula bit-exactly.
void cpu_reference(const std::vector<float>& in,
                   std::vector<float>&       out,
                   std::uint32_t in_w,  std::uint32_t in_h,
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
            const std::size_t out_k = (std::size_t(y) * out_w + x) * 3;
            const std::size_t in_k  = (std::size_t(yp) * in_w + xp) * 3;
            out[out_k + 0] = in[in_k + 0];
            out[out_k + 1] = in[in_k + 1];
            out[out_k + 2] = in[in_k + 2];
        }
}

bool run_case(av::gpu::Device&         dev,
              av::depth_map::DepthSimMap& dsm,
              const Case&              c)
{
    using namespace av::gpu;
    using namespace av::depth_map;

    std::vector<float> in_buf, ref_buf;
    fill_fingerprint(in_buf, c.in_w, c.in_h);
    cpu_reference   (in_buf, ref_buf,
                     c.in_w, c.in_h, c.out_w, c.out_h);

    // Pre-fill output with a sentinel to detect un-written pixels.
    std::vector<float> sentinel(
        std::size_t(c.out_w) * std::size_t(c.out_h) * 3, -7777.0f);

    Buffer in_dev (dev, in_buf  .size() * sizeof(float));
    Buffer out_dev(dev, ref_buf .size() * sizeof(float));
    in_dev .upload(std::span<const float>(in_buf));
    out_dev.upload(std::span<const float>(sentinel));

    dsm.normal_map_upscale(out_dev, in_dev,
                           c.out_w, c.out_h, c.in_w, c.in_h);

    const auto* gpu = static_cast<const float*>(out_dev.data());

    int bad = 0;
    for (std::size_t k = 0; k < ref_buf.size(); ++k) {
        const std::uint32_t a = std::bit_cast<std::uint32_t>(gpu[k]);
        const std::uint32_t b = std::bit_cast<std::uint32_t>(ref_buf[k]);
        if (a != b) {
            if (bad < 4) {
                const std::size_t pix = k / 3;
                const std::uint32_t x = std::uint32_t(pix % c.out_w);
                const std::uint32_t y = std::uint32_t(pix / c.out_w);
                std::fprintf(stderr,
                    "  [%s] pix (%u, %u) ch%zu: gpu=%g ref=%g\n",
                    c.name, x, y, k % 3,
                    static_cast<double>(gpu[k]),
                    static_cast<double>(ref_buf[k]));
            }
            ++bad;
        }
    }

    std::printf("  case %-32s in=%ux%u  out=%ux%u  ratio=%.4f  diffs=%d\n",
                c.name, c.in_w, c.in_h, c.out_w, c.out_h,
                double(c.in_w) / double(c.out_w), bad);
    return bad == 0;
}

}  // namespace

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    DepthSimMap dsm(dev);

    const Case cases[] = {
        // Plain 2× integer upscale.
        {  4,  3,   8,   6, "2x integer"           },
        // 3× integer upscale.
        {  4,  3,  12,   9, "3x integer"           },
        // Non-integer ratio: in_w/out_w = 4/9 ≠ in_h/out_h = 3/6.
        {  4,  3,   9,   6, "non-integer ratio"    },
        // Realistic SGM→Refine resolution bridge (5× horiz, 5× vert)
        // with dispatch dims not aligned to 16.
        { 11,  7,  55,  35, "5x non-multiple-of-16"},
        // Stress: large output, modest input.
        { 32, 24, 256, 192, "256x192 from 32x24"   },
    };

    int failed = 0;
    for (const auto& c : cases) {
        if (!run_case(dev, dsm, c)) ++failed;
    }

    if (failed) {
        std::fprintf(stderr, "FAIL: %d / %zu cases\n",
                     failed, sizeof(cases) / sizeof(cases[0]));
        return 1;
    }
    std::printf("PASS (%zu cases)\n", sizeof(cases) / sizeof(cases[0]));
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
