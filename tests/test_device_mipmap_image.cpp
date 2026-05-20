// test_device_mipmap_image.cpp — validation of the host class
// `DeviceMipmapImage` (Phase 8 opener).
//
// Verifies:
//   * Metadata (min/max downscale, levels count, original dims).
//   * get_level() returns log2(downscale / min_downscale).
//   * get_dimensions() matches ceil(orig / downscale).
//   * Level 0 of the texture contains Lab values (L roughly
//     within [0, 100], a/b within [-128, 128]). This sanity-
//     checks the downscale + RGB→Lab steps inside fill().
//   * Deeper mip levels are also Lab-ranged (sanity check that
//     generate_mipmaps() ran over the Lab content, not the RGB).
//   * Higher mip levels have correct dimensions.

#include "av/depth_map/DeviceMipmapImage.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kW = 128;
constexpr std::uint32_t kH =  96;
constexpr std::uint32_t kMinDownscale =  2;
constexpr std::uint32_t kMaxDownscale = 16;

std::vector<float> make_rgba_image() {
    std::vector<float> px(std::size_t(kW) * kH * 4);
    for (std::uint32_t j = 0; j < kH; ++j)
        for (std::uint32_t i = 0; i < kW; ++i) {
            const float u = float(i) / float(kW);
            const float v = float(j) / float(kH);
            const float r = 128.0f + 60.0f * std::sin(15.0f*u + 2.0f*v);
            const float g = 128.0f + 50.0f * std::cos(13.0f*v);
            const float b = 128.0f + 45.0f * std::sin(17.0f*(u + v));
            const std::size_t k = (j * kW + i) * 4;
            px[k+0] = std::clamp(r, 0.0f, 255.0f);
            px[k+1] = std::clamp(g, 0.0f, 255.0f);
            px[k+2] = std::clamp(b, 0.0f, 255.0f);
            px[k+3] = 255.0f;
        }
    return px;
}

// Our port's `xyz2lab` multiplies the classical Lab output by
// 2.55 (upstream convention — fits uchar [0, 255] scale). So
// "classical L in [0, 100]" becomes "L in [0, ~255]", and a/b
// in [-128, 128] become roughly [-256, 256].
bool sane_lab_range(const std::vector<float>& level_pixels,
                    std::uint32_t pix_count,
                    const char* label,
                    float L_lo, float L_hi,
                    float ab_max)
{
    float L_min = 1e9f, L_max = -1e9f;
    float a_min = 1e9f, a_max = -1e9f;
    float b_min = 1e9f, b_max = -1e9f;
    for (std::uint32_t k = 0; k < pix_count; ++k) {
        const float L = level_pixels[k * 4 + 0];
        const float a = level_pixels[k * 4 + 1];
        const float b = level_pixels[k * 4 + 2];
        L_min = std::min(L_min, L); L_max = std::max(L_max, L);
        a_min = std::min(a_min, a); a_max = std::max(a_max, a);
        b_min = std::min(b_min, b); b_max = std::max(b_max, b);
    }
    std::printf("[info] %-7s L [%.2f, %.2f]  a [%.2f, %.2f]  b [%.2f, %.2f]\n",
                label, double(L_min), double(L_max),
                double(a_min), double(a_max),
                double(b_min), double(b_max));
    if (L_min < L_lo || L_max > L_hi) {
        std::fprintf(stderr,
            "FAIL: %s L outside [%.2f, %.2f]\n",
            label, double(L_lo), double(L_hi));
        return false;
    }
    if (std::fabs(a_min) > ab_max || std::fabs(a_max) > ab_max ||
        std::fabs(b_min) > ab_max || std::fabs(b_max) > ab_max) {
        std::fprintf(stderr,
            "FAIL: %s a/b outside [-%.2f, %.2f]\n",
            label, double(ab_max), double(ab_max));
        return false;
    }
    return true;
}

}  // namespace

// Run the full metadata + Lab-range battery on a DeviceMipmapImage
// instance. Returns the number of failures and, if `out_lv0` /
// `out_lv2` are non-null, fills them with the downloaded level
// pixels for later cross-path comparison.
static int run_battery(av::depth_map::DeviceMipmapImage& mip,
                       const char* path_label,
                       std::vector<float>* out_lv0,
                       std::vector<float>* out_lv2,
                       std::uint32_t* out_lv2_w,
                       std::uint32_t* out_lv2_h)
{
    using namespace av::gpu;
    using namespace av::depth_map;

    int failed = 0;

    // ---- metadata ----
    if (mip.width() != kW || mip.height() != kH) {
        std::fprintf(stderr, "FAIL [%s]: width/height mismatch\n", path_label);
        ++failed;
    }
    if (mip.min_downscale() != kMinDownscale ||
        mip.max_downscale() != kMaxDownscale) {
        std::fprintf(stderr, "FAIL [%s]: downscale metadata\n", path_label);
        ++failed;
    }

    // log2(max/min) + 1 = 4 levels.
    const Texture& tex = mip.texture();
    if (tex.mip_levels() != 4) {
        std::fprintf(stderr,
            "FAIL [%s]: mip_levels = %u (expected 4)\n",
            path_label, tex.mip_levels());
        ++failed;
    }
    if (tex.width() != kW / kMinDownscale ||
        tex.height() != kH / kMinDownscale) {
        std::fprintf(stderr,
            "FAIL [%s]: texture %u × %u (expected %u × %u)\n",
            path_label, tex.width(), tex.height(),
            kW / kMinDownscale, kH / kMinDownscale);
        ++failed;
    }
    std::printf("[info] [%s] tex     : %u × %u, %u levels\n",
                path_label,
                tex.width(), tex.height(), tex.mip_levels());

    // ---- get_level / get_dimensions ----
    struct Expect {
        std::uint32_t downscale;
        float         level;
        std::uint32_t w, h;
    };
    const Expect cases[] = {
        {  2, 0.0f, 64, 48 },
        {  4, 1.0f, 32, 24 },
        {  8, 2.0f, 16, 12 },
        { 16, 3.0f,  8,  6 },
    };
    for (const auto& c : cases) {
        const float L = mip.get_level(c.downscale);
        if (std::fabs(L - c.level) > 1e-5f) {
            std::fprintf(stderr,
                "FAIL [%s]: get_level(%u) = %g (expected %g)\n",
                path_label,
                c.downscale, double(L), double(c.level));
            ++failed;
        }
        const auto [w, h] = mip.get_dimensions(c.downscale);
        if (w != c.w || h != c.h) {
            std::fprintf(stderr,
                "FAIL [%s]: get_dimensions(%u) = (%u, %u) (expected (%u, %u))\n",
                path_label,
                c.downscale, w, h, c.w, c.h);
            ++failed;
        }
    }

    // ---- read back level 0 + check Lab range ----
    {
        const std::uint32_t lw = tex.width();
        const std::uint32_t lh = tex.height();
        std::vector<float> px(std::size_t(lw) * lh * 4, -7.0f);
        tex.download_level(
            std::span<std::byte>(
                reinterpret_cast<std::byte*>(px.data()),
                px.size() * sizeof(float)),
            /*mip_level=*/0);
        if (!sane_lab_range(px, lw * lh, "level 0",
                            -3.0f, 280.0f,
                            330.0f)) {
            ++failed;
        }
        if (out_lv0) *out_lv0 = std::move(px);
    }

    // ---- read back level 2 (deeper in the cascade) ----
    {
        const auto [lw, lh] = mip.get_dimensions(8);  // downscale 8 → level 2
        std::vector<float> px(std::size_t(lw) * lh * 4, -7.0f);
        tex.download_level(
            std::span<std::byte>(
                reinterpret_cast<std::byte*>(px.data()),
                px.size() * sizeof(float)),
            /*mip_level=*/2);
        bool any_written = false;
        for (std::uint32_t k = 0; k < lw * lh; ++k) {
            if (px[k * 4 + 0] != -7.0f) { any_written = true; break; }
        }
        if (!any_written) {
            std::fprintf(stderr, "FAIL [%s]: level 2 was not generated\n",
                         path_label);
            ++failed;
        } else if (!sane_lab_range(px, lw * lh, "level 2",
                                   -3.0f, 280.0f, 330.0f)) {
            ++failed;
        }
        if (out_lv2)   *out_lv2   = std::move(px);
        if (out_lv2_w) *out_lv2_w = lw;
        if (out_lv2_h) *out_lv2_h = lh;
    }

    return failed;
}

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    const auto rgba = make_rgba_image();

    DeviceMipmapImage mip(dev);
    mip.fill(std::span<const float>(rgba), kW, kH,
             kMinDownscale, kMaxDownscale);

    int failed = 0;

    // ---- Default path: Metal built-in generate_mipmaps. This is
    // the baseline that S26 established (level 0 L in [157, 222]).
    std::printf("[info] === path: built-in (default) ===\n");
    std::vector<float> default_lv0, default_lv2;
    std::uint32_t lv2_w = 0, lv2_h = 0;
    failed += run_battery(mip, "built-in",
                          &default_lv0, &default_lv2, &lv2_w, &lv2_h);

    // ---- Upstream-ported path: 5×5 Gaussian-weighted bilinear
    // stencil from the CUDA original. Run on a fresh instance.
    std::printf("[info] === path: upstream-ported ===\n");
    DeviceMipmapImage mip2(dev);
    mip2.set_use_upstream_mipgen(true);
    if (!mip2.use_upstream_mipgen()) {
        std::fprintf(stderr, "FAIL: set_use_upstream_mipgen did not stick\n");
        ++failed;
    }
    mip2.fill(std::span<const float>(rgba), kW, kH,
              kMinDownscale, kMaxDownscale);

    std::vector<float> upstream_lv0, upstream_lv2;
    std::uint32_t lv2_w2 = 0, lv2_h2 = 0;
    failed += run_battery(mip2, "upstream",
                          &upstream_lv0, &upstream_lv2, &lv2_w2, &lv2_h2);

    // ---- Cross-path sanity: level 0 is identical (both paths
    // populate level 0 the same way; only the cascade differs).
    if (default_lv0.size() != upstream_lv0.size()) {
        std::fprintf(stderr,
            "FAIL: level-0 sizes differ (%zu vs %zu)\n",
            default_lv0.size(), upstream_lv0.size());
        ++failed;
    } else {
        float max_diff_lv0 = 0.0f;
        for (std::size_t k = 0; k < default_lv0.size(); ++k) {
            max_diff_lv0 = std::max(max_diff_lv0,
                                    std::fabs(default_lv0[k] - upstream_lv0[k]));
        }
        std::printf("[info] level-0 max |built-in - upstream|: %.4f\n",
                    double(max_diff_lv0));
        if (max_diff_lv0 > 1e-3f) {
            std::fprintf(stderr,
                "FAIL: level-0 differs between paths "
                "(max diff %.4f > 1e-3)\n", double(max_diff_lv0));
            ++failed;
        }
    }

    // ---- Cross-path sanity: level 2 differs but stays in a
    // bounded range. We measure the max channelwise delta as a
    // diagnostic and require it stays under a generous budget
    // (50.0). The two cascades have fundamentally different
    // weighting (textbook box vs. Gaussian-weighted bilinear), so
    // they will not match bit-for-bit — but they should produce
    // values in a similar neighborhood.
    if (lv2_w != lv2_w2 || lv2_h != lv2_h2) {
        std::fprintf(stderr,
            "FAIL: level-2 dims differ between paths "
            "(%u×%u vs %u×%u)\n",
            lv2_w, lv2_h, lv2_w2, lv2_h2);
        ++failed;
    } else if (default_lv2.size() == upstream_lv2.size()) {
        float max_diff_lv2 = 0.0f;
        for (std::size_t k = 0; k < default_lv2.size(); ++k) {
            max_diff_lv2 = std::max(max_diff_lv2,
                                    std::fabs(default_lv2[k] - upstream_lv2[k]));
        }
        std::printf("[info] level-2 max |built-in - upstream|: %.4f\n",
                    double(max_diff_lv2));
        if (max_diff_lv2 > 50.0f) {
            std::fprintf(stderr,
                "FAIL: level-2 unexpectedly far from built-in "
                "(max diff %.4f > 50)\n", double(max_diff_lv2));
            ++failed;
        }
    }

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
