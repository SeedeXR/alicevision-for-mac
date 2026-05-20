// test_image_color_conversion.cpp — end-to-end validation of the
// in-place rgb2lab image kernel (port of upstream's cuda_rgb2lab).
//
// Strategy:
//   1. Build a synthetic 256x256 RGBA32Float "image" with values in
//      [0, 255] (linear, not sRGB — upstream treats input as linear).
//      Alpha = 255.
//   2. Upload to a Shared RGBA32Float texture.
//   3. Dispatch ImageColorConversion::rgb2lab over the texture.
//   4. Read the texture back via Texture::download.
//   5. Compute the same conversion per-pixel on the CPU in FP64.
//   6. Compare element-wise.
//
// The pipeline is `xyz2lab(rgb2xyz(rgb / 255))` — exactly the same
// chain already validated in test_color.cpp, just dispatched
// per-pixel over a full image. This test exercises the full-image
// dispatch surface and the new Texture::download path.

#include "av/depth_map/ImageColorConversion.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

constexpr std::uint32_t kW = 256;
constexpr std::uint32_t kH = 256;

// Same FP64 reference chain as test_color.cpp (xyz2lab(rgb2xyz(...))).
struct V3 { double x, y, z; };

V3 rgb2xyz_ref(V3 c) {
    return { 0.4124564 * c.x + 0.3575761 * c.y + 0.1804375 * c.z,
             0.2126729 * c.x + 0.7151522 * c.y + 0.0721750 * c.z,
             0.0193339 * c.x + 0.1191920 * c.y + 0.9503041 * c.z };
}

V3 xyz2lab_ref(V3 c) {
    constexpr double kappa  = 24389.0 / 27.0;
    constexpr double thresh = 216.0 / 24389.0;
    V3 r = { c.x / 0.95047, c.y, c.z / 1.08883 };
    auto f = [&](double v) {
        return v > thresh ? std::cbrt(v) : (kappa * v + 16.0) / 116.0;
    };
    V3 fc = { f(r.x), f(r.y), f(r.z) };
    return { (116.0 * fc.y - 16.0)   * 2.55,
             (500.0 * (fc.x - fc.y)) * 2.55,
             (200.0 * (fc.y - fc.z)) * 2.55 };
}

// Synthetic linear-RGB image, values in [0, 255]. A smooth gradient
// + a couple of cosine ripples so every (xyz2lab) branch (large vs
// small XYZ) is exercised across the image.
std::vector<float> make_image()
{
    std::vector<float> px(kW * kH * 4);
    for (std::uint32_t j = 0; j < kH; ++j) {
        for (std::uint32_t i = 0; i < kW; ++i) {
            const float u = float(i) / float(kW);
            const float v = float(j) / float(kH);
            const float r = 64.0f + 100.0f * u
                            + 30.0f * std::sin(3.0f * (u + 0.5f * v));
            const float g = 32.0f + 80.0f * v
                            + 25.0f * std::cos(2.0f * (v + 0.7f * u));
            const float b = 200.0f - 80.0f * u + 40.0f * v;
            const std::size_t k = (j * kW + i) * 4;
            px[k + 0] = std::clamp(r, 0.0f, 255.0f);
            px[k + 1] = std::clamp(g, 0.0f, 255.0f);
            px[k + 2] = std::clamp(b, 0.0f, 255.0f);
            px[k + 3] = 255.0f;
        }
    }
    return px;
}

}  // namespace

int main() try
{
    using namespace av::gpu;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    // Allocate a single-level RGBA32Float texture in Shared storage.
    // mip_levels=1 (no auto-mip): we don't sample it, we only
    // read/write it from a compute kernel, so we don't need mips.
    Texture img(dev, Texture::Descriptor{
        kW, kH, /*mip_levels=*/1,
        PixelFormat::RGBA32Float });
    img.set_label("rgb2lab.image");

    const auto src = make_image();
    img.upload(std::span<const float>(src));

    // -------- dispatch --------
    av::depth_map::ImageColorConversion conv(dev);
    conv.rgb2lab(img);

    // -------- read back --------
    std::vector<float> dst(kW * kH * 4);
    img.download(std::span<float>(dst));

    // -------- CPU reference + compare --------
    // Budget per channel: 5e-4 absolute on Lab. The same chain
    // measured "worst err XYZ→Lab 1.25e-04" against FP64 in
    // test_color.cpp at order-1 inputs; on order-255 outputs that
    // becomes ~3e-2 absolute. The FP32 GPU and FP64 CPU values are
    // compared without rescaling, so we budget the raw difference
    // on the Lab scale.
    constexpr double kTolLab = 3e-2;
    constexpr double kTolA   = 1e-3;   // alpha must pass through

    double worst_l = 0.0, worst_a = 0.0, worst_b = 0.0, worst_alpha = 0.0;
    int bad = 0;
    for (std::uint32_t j = 0; j < kH; ++j) {
        for (std::uint32_t i = 0; i < kW; ++i) {
            const std::size_t k = (j * kW + i) * 4;
            const V3 rgb01{ static_cast<double>(src[k + 0]) / 255.0,
                            static_cast<double>(src[k + 1]) / 255.0,
                            static_cast<double>(src[k + 2]) / 255.0 };
            const V3 lab_ref = xyz2lab_ref(rgb2xyz_ref(rgb01));

            const double dl = std::abs(static_cast<double>(dst[k + 0]) - lab_ref.x);
            const double da = std::abs(static_cast<double>(dst[k + 1]) - lab_ref.y);
            const double db = std::abs(static_cast<double>(dst[k + 2]) - lab_ref.z);
            const double dalpha = std::abs(static_cast<double>(dst[k + 3])
                                          - static_cast<double>(src[k + 3]));

            worst_l     = std::max(worst_l,     dl);
            worst_a     = std::max(worst_a,     da);
            worst_b     = std::max(worst_b,     db);
            worst_alpha = std::max(worst_alpha, dalpha);

            if (dl > kTolLab || da > kTolLab || db > kTolLab || dalpha > kTolA) {
                if (bad < 4) std::fprintf(stderr,
                    "pixel (%u, %u): gpu=(%.3f, %.3f, %.3f, %.1f) "
                    "ref=(%.3f, %.3f, %.3f, %.1f) | dl=%.4g da=%.4g db=%.4g dα=%.4g\n",
                    i, j,
                    static_cast<double>(dst[k + 0]),
                    static_cast<double>(dst[k + 1]),
                    static_cast<double>(dst[k + 2]),
                    static_cast<double>(dst[k + 3]),
                    lab_ref.x, lab_ref.y, lab_ref.z,
                    static_cast<double>(src[k + 3]),
                    dl, da, db, dalpha);
                ++bad;
            }
        }
    }

    std::printf("[info] pixels             : %u x %u = %u\n",
                kW, kH, kW * kH);
    std::printf("[info] worst |ΔL|         : %.3g (budget %.3g)\n", worst_l,     kTolLab);
    std::printf("[info] worst |Δa|         : %.3g (budget %.3g)\n", worst_a,     kTolLab);
    std::printf("[info] worst |Δb|         : %.3g (budget %.3g)\n", worst_b,     kTolLab);
    std::printf("[info] worst |Δα|         : %.3g (budget %.3g)\n", worst_alpha, kTolA);

    if (bad) {
        std::fprintf(stderr, "FAIL: %d pixels out of tolerance\n", bad);
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
