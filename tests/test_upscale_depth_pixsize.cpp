// test_upscale_depth_pixsize.cpp — validation of
// `av_compute_sgm_upscaled_depth_pix_size_map_{nearest,bilinear}`.
//
// The kernels read from an SGM-resolution (depth, thickness)
// buffer and write a Refine-resolution (depth, pixSize) buffer
// where `pixSize = thickness / half_nb_depths`. An rc mipmap
// texture is sampled for alpha-masking — masked pixels write
// (-2, 0).
//
// Coverage:
//   * Both variants (nearest, bilinear) on the same synthetic
//     scene.
//   * Stripes in the rc texture with low alpha → masked pixels
//     write (-2, 0).
//   * Invalid cells in the SGM map → bilinear corner-fallback
//     (averages valid corners; writes (-1, 1) if all 4 invalid).
//   * Non-multiple-of-16 dispatch dims.
//
// Validation strategy: CPU reference replicates each kernel's
// arithmetic in FP32. For the texture sampling, we replicate the
// expected behavior of `filter::linear` at level 0 with pixel-
// center UV: a bit-exact sample of the source pixel. We accept
// a small relative tolerance to absorb sub-ULP rounding from any
// MSL-specific FP fusion.

#include "av/depth_map/DepthSimMap.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::uint32_t kInW  = 16;
constexpr std::uint32_t kInH  = 12;
constexpr std::uint32_t kOutW = 67;       // non-multiple-of-16
constexpr std::uint32_t kOutH = 49;       // non-multiple-of-16
constexpr std::int32_t  kStepXY        = 1;
constexpr std::int32_t  kHalfNbDepths  = 4;
constexpr std::uint32_t kRcW   = kOutW * kStepXY;
constexpr std::uint32_t kRcH   = kOutH * kStepXY;
// S43: alpha convention harmonized to [0, 1] to match the kernel's
// LEGACY 0.9 threshold (which mirrors real EXR-preserved alpha). With
// kAlphaPass=1.0 and mask=0.5, bilinear-sampled edge pixels average
// to ~0.75 → correctly fails the 0.9 threshold (matching CPU reference).
constexpr float         kAlphaPass    =   1.0f;
constexpr float         kAlphaMaskNN  =   0.5f;  // < 0.9f (nearest threshold)
// S43: bilinear threshold harmonized to LEGACY (0.9) per the same EXR-
// [0, 1] alpha convention as nearest. Mask value must now be < 0.9.
constexpr float         kAlphaMaskBL  =   0.5f;  // < 0.9f (bilinear threshold)

// Synthesize an SGM depth/thickness map with a known pattern +
// a few invalid cells.
void make_sgm_map(std::vector<float>& m) {
    m.assign(std::size_t(kInW) * std::size_t(kInH) * 2, 0.0f);
    for (std::uint32_t y = 0; y < kInH; ++y)
        for (std::uint32_t x = 0; x < kInW; ++x) {
            const std::size_t k = (std::size_t(y) * kInW + x) * 2;
            // Smooth gradient depth + thickness.
            float depth = 4.0f + 0.05f * float(x) + 0.04f * float(y);
            float thick = 0.10f + 0.005f * float(x + y);
            // Sprinkle a few invalid cells (depth <= 0).
            // Choose a region NOT covered by the rc-mask in either
            // variant — at output res, mask is on rows [5, 9) for
            // nearest, [12, 20) for bilinear, which correspond to
            // SGM rows ~[1.0, 2.2) and ~[2.8, 4.7) respectively.
            // Put invalid cells around (8, 8) so they're well clear.
            if ((x == 8 && y == 8) || (x == 9 && y == 8) ||
                (x == 8 && y == 9)) depth = -1.0f;
            m[k + 0] = depth;
            m[k + 1] = thick;
        }
}

// Synthesize a refine-resolution rc image. Most pixels have
// alpha = 255 (pass mask). A horizontal stripe has low alpha
// (chosen per variant to exercise the threshold).
void make_rc_image(std::vector<float>& px, float alpha_mask_value,
                   std::uint32_t mask_y_begin, std::uint32_t mask_y_end) {
    px.resize(std::size_t(kRcW) * std::size_t(kRcH) * 4);
    for (std::uint32_t y = 0; y < kRcH; ++y)
        for (std::uint32_t x = 0; x < kRcW; ++x) {
            const std::size_t k = (std::size_t(y) * kRcW + x) * 4;
            px[k + 0] = 128.0f;
            px[k + 1] = 128.0f;
            px[k + 2] = 128.0f;
            px[k + 3] = (y >= mask_y_begin && y < mask_y_end)
                ? alpha_mask_value : kAlphaPass;
        }
}

// CPU reference for the nearest variant.
void cpu_ref_nearest(const std::vector<float>& sgm,
                     const std::vector<float>& rc,
                     std::vector<float>&       out)
{
    out.assign(std::size_t(kOutW) * std::size_t(kOutH) * 2, 0.0f);
    const float ratio = float(kInW) / float(kOutW);
    for (std::uint32_t roiY = 0; roiY < kOutH; ++roiY)
        for (std::uint32_t roiX = 0; roiX < kOutW; ++roiX) {
            const std::size_t k = (std::size_t(roiY) * kOutW + roiX) * 2;
            const std::uint32_t x = roiX * std::uint32_t(kStepXY);
            const std::uint32_t y = roiY * std::uint32_t(kStepXY);
            const float alpha = rc[(std::size_t(y) * kRcW + x) * 4 + 3];
            if (alpha < 0.9f) {
                out[k + 0] = -2.0f;
                out[k + 1] =  0.0f;
                continue;
            }
            const float ox = (float(roiX) - 0.5f) * ratio;
            const float oy = (float(roiY) - 0.5f) * ratio;
            int xp = int(std::floor(ox + 0.5f));
            int yp = int(std::floor(oy + 0.5f));
            xp = std::min(xp, int(kInW) - 1);
            yp = std::min(yp, int(kInH) - 1);
            const std::size_t in_k =
                (std::size_t(yp) * kInW + xp) * 2;
            const float depth = sgm[in_k + 0];
            const float thick = sgm[in_k + 1];
            out[k + 0] = depth;
            out[k + 1] = thick / float(kHalfNbDepths);
        }
}

// CPU reference for the bilinear variant.
void cpu_ref_bilinear(const std::vector<float>& sgm,
                      const std::vector<float>& rc,
                      std::vector<float>&       out)
{
    out.assign(std::size_t(kOutW) * std::size_t(kOutH) * 2, 0.0f);
    const float ratio = float(kInW) / float(kOutW);
    const float min_alpha = 0.9f;  // S43: harmonized with kernel
    for (std::uint32_t roiY = 0; roiY < kOutH; ++roiY)
        for (std::uint32_t roiX = 0; roiX < kOutW; ++roiX) {
            const std::size_t k = (std::size_t(roiY) * kOutW + roiX) * 2;
            const std::uint32_t x = roiX * std::uint32_t(kStepXY);
            const std::uint32_t y = roiY * std::uint32_t(kStepXY);
            const float alpha = rc[(std::size_t(y) * kRcW + x) * 4 + 3];
            if (alpha < min_alpha) {
                out[k + 0] = -2.0f;
                out[k + 1] =  0.0f;
                continue;
            }
            const float ox = (float(roiX) - 0.5f) * ratio;
            const float oy = (float(roiY) - 0.5f) * ratio;
            int xp = int(std::floor(ox));
            int yp = int(std::floor(oy));
            // S48: clamp from BOTH ends. Mirror the kernel's
            // clamp_to_edge bilinear stencil — without the lower
            // clamp, roiX=0 produces xp=-1 and the CPU reference
            // OOB-reads the sgm vector. That OOB read picks up
            // heap-state-dependent garbage, which was the root
            // cause of the -j8 flakiness (the kernel itself also
            // OOB'd, but its garbage differed from the CPU's).
            xp = std::max(0, std::min(xp, int(kInW) - 2));
            yp = std::max(0, std::min(yp, int(kInH) - 2));
            auto load = [&](int gx, int gy) {
                const std::size_t kk =
                    (std::size_t(gy) * kInW + gx) * 2;
                return std::pair<float, float>(sgm[kk + 0], sgm[kk + 1]);
            };
            auto [lu_d, lu_t] = load(xp,     yp);
            auto [ru_d, ru_t] = load(xp + 1, yp);
            auto [rd_d, rd_t] = load(xp + 1, yp + 1);
            auto [ld_d, ld_t] = load(xp,     yp + 1);
            float d, t;
            if (lu_d <= 0.0f || ru_d <= 0.0f ||
                rd_d <= 0.0f || ld_d <= 0.0f) {
                float sd = 0.0f, st = 0.0f;
                int cnt = 0;
                if (lu_d > 0.0f) { sd += lu_d; st += lu_t; ++cnt; }
                if (ru_d > 0.0f) { sd += ru_d; st += ru_t; ++cnt; }
                if (rd_d > 0.0f) { sd += rd_d; st += rd_t; ++cnt; }
                if (ld_d > 0.0f) { sd += ld_d; st += ld_t; ++cnt; }
                if (cnt == 0) {
                    out[k + 0] = -1.0f;
                    out[k + 1] =  1.0f;
                    continue;
                }
                d = sd / float(cnt);
                t = st / float(cnt);
            } else {
                // S48: clamp weights to [0, 1] for edge clamping.
                const float ui = std::clamp(ox - float(xp), 0.0f, 1.0f);
                const float vi = std::clamp(oy - float(yp), 0.0f, 1.0f);
                const float ud = lu_d + (ru_d - lu_d) * ui;
                const float ut = lu_t + (ru_t - lu_t) * ui;
                const float dd = ld_d + (rd_d - ld_d) * ui;
                const float dt = ld_t + (rd_t - ld_t) * ui;
                d = ud + (dd - ud) * vi;
                t = ut + (dt - ut) * vi;
            }
            out[k + 0] = d;
            out[k + 1] = t / float(kHalfNbDepths);
        }
}

struct Stats {
    int   bad             = 0;
    int   masked          = 0;
    int   corner_fallback = 0;
    int   all_invalid     = 0;
    float worst_d = 0.0f, worst_p = 0.0f;
};

Stats compare(const float* gpu, const std::vector<float>& ref,
              const std::vector<float>* sgm_map,
              float in_to_out_ratio,
              bool bilinear)
{
    Stats s;
    const std::size_t pix = std::size_t(kOutW) * std::size_t(kOutH);
    for (std::size_t k = 0; k < pix; ++k) {
        const float gpu_d = gpu[k * 2 + 0];
        const float gpu_p = gpu[k * 2 + 1];
        const float ref_d = ref[k * 2 + 0];
        const float ref_p = ref[k * 2 + 1];
        if (ref_d == -2.0f) ++s.masked;
        else if (ref_d == -1.0f) ++s.all_invalid;
        // Detect bilinear corner-fallback path by sampling the
        // SGM neighbors as the kernel would.
        else if (bilinear && sgm_map) {
            const std::uint32_t roiX = std::uint32_t(k % kOutW);
            const std::uint32_t roiY = std::uint32_t(k / kOutW);
            const float ox = (float(roiX) - 0.5f) * in_to_out_ratio;
            const float oy = (float(roiY) - 0.5f) * in_to_out_ratio;
            int xp = int(std::floor(ox));
            int yp = int(std::floor(oy));
            // S48: clamp from both ends to match the kernel/CPU-ref
            // edge-clamp behavior.
            xp = std::max(0, std::min(xp, int(kInW) - 2));
            yp = std::max(0, std::min(yp, int(kInH) - 2));
            auto load_d = [&](int gx, int gy) {
                return (*sgm_map)[(std::size_t(gy) * kInW + gx) * 2];
            };
            const float a = load_d(xp,     yp);
            const float b = load_d(xp + 1, yp);
            const float c = load_d(xp + 1, yp + 1);
            const float d = load_d(xp,     yp + 1);
            if (a <= 0.0f || b <= 0.0f || c <= 0.0f || d <= 0.0f) {
                ++s.corner_fallback;
            }
        }
        auto rel = [](float a, float b) {
            const float diff = std::fabs(a - b);
            return diff / std::fmax(std::fabs(a) + std::fabs(b), 1e-30f);
        };
        const float rd = rel(gpu_d, ref_d);
        const float rp = rel(gpu_p, ref_p);
        s.worst_d = std::max(s.worst_d, rd);
        s.worst_p = std::max(s.worst_p, rp);
        if (rd > 1e-5f || rp > 1e-5f) {
            if (s.bad < 4) std::fprintf(stderr,
                "  pix %zu: gpu=(%g, %g)  ref=(%g, %g)  rel_d=%.3e rel_p=%.3e\n",
                k,
                double(gpu_d), double(gpu_p),
                double(ref_d), double(ref_p),
                double(rd), double(rp));
            ++s.bad;
        }
    }
    return s;
}

}  // namespace

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    // ---- SGM map ----
    std::vector<float> sgm;
    make_sgm_map(sgm);
    Buffer sgm_buf(dev, sgm.size() * sizeof(float));
    sgm_buf.upload(std::span<const float>(sgm));

    // ---- output buffer (re-used across variants) ----
    Buffer out_buf(dev,
        std::size_t(kOutW) * std::size_t(kOutH) * 2 * sizeof(float));

    DepthSimMap dsm(dev);

    int total_bad = 0;

    // ---- nearest variant ----
    {
        std::vector<float> rc_img;
        // Mask rows [5, 9) with alpha = 0.5 (< 0.9 nearest threshold).
        make_rc_image(rc_img, kAlphaMaskNN, 5, 9);

        Texture rc_tex(dev, Texture::Descriptor{
            kRcW, kRcH, 0, PixelFormat::RGBA32Float });
        rc_tex.upload(std::span<const float>(rc_img));
        rc_tex.generate_mipmaps();

        std::vector<float> ref;
        cpu_ref_nearest(sgm, rc_img, ref);

        // sentinel-fill the GPU output to detect un-written pixels.
        std::vector<float> sentinel(
            std::size_t(kOutW) * std::size_t(kOutH) * 2, -7777.0f);
        out_buf.upload(std::span<const float>(sentinel));

        DepthSimMap::ComputeUpscaledDepthPixSizeMapParams p{};
        p.out_width       = kOutW;  p.out_height      = kOutH;
        p.in_width        = kInW;   p.in_height       = kInH;
        p.roi_x_begin     = 0;      p.roi_y_begin     = 0;
        p.rc_level_width  = kRcW;   p.rc_level_height = kRcH;
        p.rc_mipmap_level = 0.0f;
        p.step_xy         = kStepXY;
        p.half_nb_depths  = kHalfNbDepths;
        p.bilinear        = false;
        dsm.compute_sgm_upscaled_depth_pix_size_map(
            out_buf, sgm_buf, rc_tex, p);

        const auto* gpu = static_cast<const float*>(out_buf.data());
        const auto s = compare(gpu, ref, nullptr,
                               float(kInW) / float(kOutW), false);
        std::printf("  nearest : masked=%d  diffs=%d  worst rel d=%.2e  p=%.2e\n",
                    s.masked, s.bad,
                    double(s.worst_d), double(s.worst_p));
        total_bad += s.bad;
    }

    // ---- bilinear variant ----
    {
        std::vector<float> rc_img;
        // Mask rows [12, 20) with alpha = 100 (< 229.5 bilinear threshold).
        make_rc_image(rc_img, kAlphaMaskBL, 12, 20);

        Texture rc_tex(dev, Texture::Descriptor{
            kRcW, kRcH, 0, PixelFormat::RGBA32Float });
        rc_tex.upload(std::span<const float>(rc_img));
        rc_tex.generate_mipmaps();

        std::vector<float> ref;
        cpu_ref_bilinear(sgm, rc_img, ref);

        std::vector<float> sentinel(
            std::size_t(kOutW) * std::size_t(kOutH) * 2, -7777.0f);
        out_buf.upload(std::span<const float>(sentinel));

        DepthSimMap::ComputeUpscaledDepthPixSizeMapParams p{};
        p.out_width       = kOutW;  p.out_height      = kOutH;
        p.in_width        = kInW;   p.in_height       = kInH;
        p.roi_x_begin     = 0;      p.roi_y_begin     = 0;
        p.rc_level_width  = kRcW;   p.rc_level_height = kRcH;
        p.rc_mipmap_level = 0.0f;
        p.step_xy         = kStepXY;
        p.half_nb_depths  = kHalfNbDepths;
        p.bilinear        = true;
        dsm.compute_sgm_upscaled_depth_pix_size_map(
            out_buf, sgm_buf, rc_tex, p);

        const auto* gpu = static_cast<const float*>(out_buf.data());
        const auto s = compare(gpu, ref, &sgm,
                               float(kInW) / float(kOutW), true);
        std::printf("  bilinear: masked=%d  corner_fallback=%d  all_invalid=%d\n"
                    "            diffs=%d  worst rel d=%.2e  p=%.2e\n",
                    s.masked, s.corner_fallback, s.all_invalid,
                    s.bad, double(s.worst_d), double(s.worst_p));

        // The corner-fallback path must trigger — we placed 3 invalid
        // SGM cells near (8, 8) deliberately outside the rc mask.
        if (s.corner_fallback == 0) {
            std::fprintf(stderr,
                "FAIL: corner-fallback path was not exercised\n");
            ++total_bad;
        }
        total_bad += s.bad;
    }

    if (total_bad) {
        std::fprintf(stderr, "FAIL: %d total mismatches\n", total_bad);
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
