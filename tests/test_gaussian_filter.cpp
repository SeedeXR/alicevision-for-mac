// test_gaussian_filter.cpp — validation of the Metal ports of
//   * downscaleWithGaussianBlur_kernel
//   * medianFilter3_kernel
// from upstream's depthMap/cuda/imageProcessing/deviceGaussianFilter.cu.
//
// Strategy: build the Gaussian LUT host-side, run each kernel
// against a CPU FP64 reference doing identical math (bilinear
// sampling + same weight lookups + same averaging / median
// selection), compare per-pixel.

#include "av/depth_map/GaussianFilter.hpp"
#include "av/depth_map/GaussianTable.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// ---------------- Gaussian LUT (CPU reference) ----------------

struct LutCpu {
    std::array<int, 10>          offsets{};
    std::vector<float>           weights;

    LutCpu() {
        int sum = 0;
        for (int s = 0; s < 10; ++s) {
            offsets[s] = sum;
            sum += 2 * (s + 1) + 1;
        }
        weights.assign(sum, 0.0f);
        for (int s = 0; s < 10; ++s) {
            const int   r = s + 1;
            const int   n = 2 * r + 1;
            const float two_d2 = 2.0f;
            for (int idx = 0; idx < n; ++idx) {
                const int x = idx - r;
                weights[offsets[s] + idx] = std::exp(-float(x * x) / two_d2);
            }
        }
    }

    float at(int scale, int idx) const {
        return weights[offsets[scale] + idx];
    }
};

// ---------------- CPU bilinear sample, RGBA32Float ----------------

struct Rgba { double r, g, b, a; };

Rgba bilin_rgba(const std::vector<float>& tex,
                std::uint32_t W, std::uint32_t H,
                double px, double py)
{
    const double cx = px - 0.5;
    const double cy = py - 0.5;
    const int    ix0 = int(std::floor(cx));
    const int    iy0 = int(std::floor(cy));
    const double fx  = cx - ix0;
    const double fy  = cy - iy0;
    auto cl = [](int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    };
    const int W_i = int(W), H_i = int(H);
    const int x0 = cl(ix0,     0, W_i - 1);
    const int x1 = cl(ix0 + 1, 0, W_i - 1);
    const int y0 = cl(iy0,     0, H_i - 1);
    const int y1 = cl(iy0 + 1, 0, H_i - 1);

    auto load = [&](int x, int y) {
        const std::size_t k = (std::size_t(y) * W + x) * 4;
        return Rgba{ tex[k + 0], tex[k + 1], tex[k + 2], tex[k + 3] };
    };
    Rgba a = load(x0, y0);
    Rgba b = load(x1, y0);
    Rgba c = load(x0, y1);
    Rgba d = load(x1, y1);
    auto mix = [](Rgba u, Rgba v, double t) {
        return Rgba{
            u.r + (v.r - u.r) * t,
            u.g + (v.g - u.g) * t,
            u.b + (v.b - u.b) * t,
            u.a + (v.a - u.a) * t,
        };
    };
    return mix(mix(a, b, fx), mix(c, d, fx), fy);
}

// Single-channel (R32Float) read at integer texel.
double load_r(const std::vector<float>& tex,
              std::uint32_t W, std::uint32_t H,
              int x, int y)
{
    auto cl = [](int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    };
    const int xc = cl(x, 0, int(W) - 1);
    const int yc = cl(y, 0, int(H) - 1);
    return tex[std::size_t(yc) * W + xc];
}

// ---------------- CPU reference: downscaleWithGaussianBlur ---------

void downscale_ref(const std::vector<float>& in_tex,
                   std::uint32_t W_in,  std::uint32_t H_in,
                   std::uint32_t W_out, std::uint32_t H_out,
                   int downscale, int radius,
                   const LutCpu& lut,
                   std::vector<float>& out)
{
    const int scale = downscale - 1;
    const double s = double(downscale) * 0.5;
    out.assign(W_out * H_out * 4, 0.0f);
    for (std::uint32_t y = 0; y < H_out; ++y) {
        for (std::uint32_t x = 0; x < W_out; ++x) {
            Rgba acc{ 0, 0, 0, 0 };
            double sumF = 0.0;
            for (int i = -radius; i <= radius; ++i) {
                for (int j = -radius; j <= radius; ++j) {
                    const double fx = double(int(x) * downscale + j) + s;
                    const double fy = double(int(y) * downscale + i) + s;
                    const Rgba   c  = bilin_rgba(in_tex, W_in, H_in, fx, fy);
                    const double f  = double(lut.at(scale, i + radius))
                                    * double(lut.at(scale, j + radius));
                    acc.r += c.r * f; acc.g += c.g * f;
                    acc.b += c.b * f; acc.a += c.a * f;
                    sumF  += f;
                }
            }
            const std::size_t k = (std::size_t(y) * W_out + x) * 4;
            out[k + 0] = float(acc.r / sumF);
            out[k + 1] = float(acc.g / sumF);
            out[k + 2] = float(acc.b / sumF);
            out[k + 3] = float(acc.a / sumF);
        }
    }
}

// ---------------- CPU reference: medianFilter3 ----------------

void median3_ref(const std::vector<float>& in_tex,
                 std::uint32_t W, std::uint32_t H,
                 std::vector<float>& out)
{
    constexpr int radius = 3;
    constexpr int n      = (2 * radius + 1) * (2 * radius + 1);
    out.assign(W * H, 0.0f);  // border default value matches GPU output (untouched, zero)

    for (std::uint32_t y = radius; y + radius < H; ++y) {
        for (std::uint32_t x = radius; x + radius < W; ++x) {
            std::array<float, n> buf{};
            int k = 0;
            for (int yi = -radius; yi <= radius; ++yi)
                for (int xi = -radius; xi <= radius; ++xi)
                    buf[k++] = float(load_r(in_tex, W, H,
                                             int(x) + xi, int(y) + yi));
            // Median via nth_element (semantic match for the
            // upstream's "sort then take middle" algorithm).
            std::nth_element(buf.begin(), buf.begin() + n / 2, buf.end());
            out[std::size_t(y) * W + x] = buf[n / 2];
        }
    }
}

// ---------------- synthetic images ----------------

std::vector<float> make_rgba_image(std::uint32_t W, std::uint32_t H,
                                   std::uint64_t seed)
{
    std::vector<float> px(W * H * 4);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    for (std::uint32_t j = 0; j < H; ++j) {
        for (std::uint32_t i = 0; i < W; ++i) {
            const float u = float(i) / float(W);
            const float v = float(j) / float(H);
            // Smooth gradient + a tiny noise term so the bilinear
            // sampler exercises non-trivial interpolation patterns.
            px[(j * W + i) * 4 + 0] = 0.7f * u + 0.05f * uni(rng);
            px[(j * W + i) * 4 + 1] = 0.7f * v + 0.05f * uni(rng);
            px[(j * W + i) * 4 + 2] = 0.5f + 0.3f * std::sin(8.0f * (u + v));
            px[(j * W + i) * 4 + 3] = 1.0f;
        }
    }
    return px;
}

std::vector<float> make_r_image(std::uint32_t W, std::uint32_t H,
                                std::uint64_t seed)
{
    std::vector<float> px(W * H);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    for (std::uint32_t j = 0; j < H; ++j) {
        for (std::uint32_t i = 0; i < W; ++i) {
            // Smooth value plus a few "salt & pepper" outliers so
            // the median filter has something to suppress.
            const float u = float(i) / float(W);
            const float v = float(j) / float(H);
            float val = 0.4f + 0.3f * std::sin(6.0f * u)
                             + 0.3f * std::cos(4.0f * v);
            const float r = uni(rng);
            if (r < 0.02f)      val = 0.0f;
            else if (r > 0.98f) val = 1.0f;
            px[j * W + i] = val;
        }
    }
    return px;
}

}  // namespace

int main() try
{
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    GaussianTable  lut(dev);
    GaussianFilter gf(dev, lut);

    int bad = 0;

    // ============== downscaleWithGaussianBlur ==================
    {
        constexpr std::uint32_t W_in       = 128;
        constexpr std::uint32_t H_in       = 128;
        constexpr int           downscale  = 2;
        constexpr int           radius     = 2;   // scale = downscale - 1 = 1
        constexpr std::uint32_t W_out      = W_in / downscale;
        constexpr std::uint32_t H_out      = H_in / downscale;
        constexpr double        kTol       = 1e-4;

        const auto src = make_rgba_image(W_in, H_in, 0xbed015);

        Texture in_tex(dev, Texture::Descriptor{
            W_in, H_in, /*mip_levels=*/1, PixelFormat::RGBA32Float });
        in_tex.upload(std::span<const float>(src));

        Buffer out_buf(dev, W_out * H_out * 4 * sizeof(float));
        out_buf.set_label("downscale.out");

        gf.downscale_with_gaussian_blur(in_tex, out_buf,
                                        W_out, H_out,
                                        downscale, radius);

        std::vector<float> gpu(W_out * H_out * 4);
        std::memcpy(gpu.data(), out_buf.data(),
                    gpu.size() * sizeof(float));

        LutCpu lut_cpu;
        std::vector<float> ref;
        downscale_ref(src, W_in, H_in, W_out, H_out,
                      downscale, radius, lut_cpu, ref);

        double worst = 0.0;
        for (std::size_t k = 0; k < gpu.size(); ++k) {
            const double e = std::abs(double(gpu[k]) - double(ref[k]));
            worst = std::max(worst, e);
            if (e > kTol && bad < 3) {
                std::fprintf(stderr,
                    "downscale pixel %zu: gpu=%g ref=%g err=%g\n",
                    k, double(gpu[k]), double(ref[k]), e);
                ++bad;
            } else if (e > kTol) ++bad;
        }
        std::printf("[info] downscale: W=%u H=%u downscale=%d r=%d, worst |Δ|=%.3g (budget %.3g)\n",
                    W_out, H_out, downscale, radius, worst, kTol);
    }

    // ============== medianFilter3 ==============================
    {
        constexpr std::uint32_t W = 128;
        constexpr std::uint32_t H = 128;
        constexpr double        kTol = 1e-6;  // exact, modulo FP32 representation

        const auto src = make_r_image(W, H, 0xfeedfade);

        Texture in_tex(dev, Texture::Descriptor{
            W, H, /*mip_levels=*/1, PixelFormat::R32Float });
        in_tex.upload(std::span<const float>(src));

        Buffer out_buf(dev, W * H * sizeof(float));
        out_buf.set_label("median3.out");

        gf.median_filter_3(in_tex, out_buf, W, H);

        std::vector<float> gpu(W * H);
        std::memcpy(gpu.data(), out_buf.data(),
                    gpu.size() * sizeof(float));

        std::vector<float> ref;
        median3_ref(src, W, H, ref);

        double worst = 0.0;
        std::uint32_t worst_x = 0, worst_y = 0;
        for (std::uint32_t y = 0; y < H; ++y) {
            for (std::uint32_t x = 0; x < W; ++x) {
                const std::size_t k = std::size_t(y) * W + x;
                const double e = std::abs(double(gpu[k]) - double(ref[k]));
                if (e > worst) {
                    worst = e; worst_x = x; worst_y = y;
                }
                if (e > kTol) {
                    if (bad < 3) std::fprintf(stderr,
                        "median (%u, %u): gpu=%g ref=%g err=%g\n",
                        x, y, double(gpu[k]), double(ref[k]), e);
                    ++bad;
                }
            }
        }
        std::printf("[info] median3 : W=%u H=%u, worst |Δ|=%.3g @ (%u, %u) (budget %.3g)\n",
                    W, H, worst, worst_x, worst_y, kTol);
    }

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
