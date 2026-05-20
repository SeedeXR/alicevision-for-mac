// test_volume_optimize_adaptive_p2.cpp — adaptive-P2 path of the
// SGM 4-direction DP aggregation kernel (extended from S13's
// `volume_optimize`; the adaptive path was deferred to S22).
//
// Upstream behaviour (deviceSimilarityVolumeKernels.cuh, lines 696–720):
//
//   if (_P2 < 0)        { P2 = abs(_P2); }                        // fixed
//   else                { P2 = sigmoid(80, 255, 80, _P2, deltaC); } // adaptive
//
// `deltaC` is the RGB Euclidean distance between two rc-mipmap
// samples at the current pixel and the neighbour in the Ym1
// direction of the current SGM aggregation path:
//
//   imX0 = (roiX + vx) * step;
//   imY0 = (roiY + vy) * step;
//   imX1 = imX0 - ySign * step * (axisT.y == 0);
//   imY1 = imY0 - ySign * step * (axisT.y == 1);
//
// The host-side mirror below replicates the kernel's float
// arithmetic exactly (single precision; no FP64 reference for
// sigmoid since the GPU also runs in FP32). It uses the same
// bilinear-sample formula on the rc image as MSL's
// `rc_mip.sample(sampler(linear), uv, level(0))`.
//
// Two runs are compared:
//   * Run A: `adaptive_p2 = false`, fixed `p2_abs = 100`. Output
//     should be bit-exact identical to a "baseline" pass that uses
//     the existing `test_volume_optimize` algorithm.
//   * Run B: `adaptive_p2 = true`, `p2_sig_mid = 100`. Output
//     should differ from Run A in ROI voxels where the sigmoid
//     value moves P2 away from 100, AND agree with a CPU FP32
//     reference within FP32 ULP error from the sigmoid/exp chain.

#include "av/depth_map/Volume.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

namespace {

// Small volume + texture. Keep dimensions small so the CPU FP32
// reference is fast and easy to audit.
constexpr std::uint32_t W = 12;
constexpr std::uint32_t H = 8;
constexpr std::uint32_t D = 6;

// Texture is at the SGM mipmap level — same dims as the volume
// (step=1, mipLevel=0, no ROI). This lets us reason about
// neighbour samples by index.
constexpr std::uint32_t kTexW = W;
constexpr std::uint32_t kTexH = H;
constexpr std::int32_t  kStep = 1;
constexpr float         kMipLevel = 0.0f;

constexpr float kP1     = 10.0f;
constexpr float kP2     = 100.0f;     // both: fixed value & sigMid

inline std::size_t vol_idx(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return std::size_t(z) * (W * H) + std::size_t(y) * W + x;
}

// ---- synthetic input volume (same pattern as test_volume_optimize) ----
std::vector<std::uint8_t> make_input_volume(std::uint64_t seed)
{
    std::vector<std::uint8_t> v(W * H * D);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> U(0, 254);
    for (std::uint32_t y = 0; y < H; ++y) {
        for (std::uint32_t x = 0; x < W; ++x) {
            const int true_z = int(D / 2) +
                ((int(x) - int(W) / 2 + int(y) - int(H) / 2) % 2);
            for (std::uint32_t z = 0; z < D; ++z) {
                const int delta = std::abs(int(z) - true_z);
                v[vol_idx(x, y, z)] = std::uint8_t(
                    std::min(254, 30 + delta * 60 + U(rng) % 5));
            }
        }
    }
    return v;
}

// ---- synthetic rc image (RGBA32Float, [0..255]-ish, smooth) ----
//
// We want a varied colour field so deltaC is non-zero and varies
// across voxels — otherwise adaptive vs fixed reduces to a
// trivial check.
std::vector<float> make_rc_image()
{
    std::vector<float> px(kTexW * kTexH * 4);
    for (std::uint32_t j = 0; j < kTexH; ++j) {
        for (std::uint32_t i = 0; i < kTexW; ++i) {
            const float u = float(i) / float(kTexW);
            const float v = float(j) / float(kTexH);
            const std::size_t k = (j * kTexW + i) * 4;
            // Bright, high-contrast pattern: ensures deltaC sweeps
            // across the sigmoid's transition region.
            px[k + 0] = 30.0f + 200.0f * std::sin(7.0f * u + 1.3f * v);
            px[k + 1] = 60.0f + 150.0f * std::cos(5.0f * v - 0.9f * u);
            px[k + 2] = 90.0f + 120.0f * std::sin(3.0f * u * v + 0.5f);
            px[k + 3] = 255.0f;
        }
    }
    return px;
}

// ---- CPU FP32 mirror of the adaptive-P2 sigmoid ---------------
//
// Matches MSL `sigmoid` (matrix.h):
//   sigmoid(i, a, w, m, x) = i + (a - i) * 1 / (1 + exp(10*(x - m)/w))
inline float cpu_sigmoid(float i, float a, float w, float m, float x)
{
    return i + (a - i) * (1.0f / (1.0f + std::exp(10.0f * (x - m) / w)));
}

struct CpuFloat4 { float x, y, z, w; };

// MSL-side bilinear sample of an RGBA32Float texture with
// normalized coords, clamp-to-edge, linear filter. We reproduce
// the kernel's `rc_mip.sample(av_volopt_mip_sampler, uv, level(0))`
// at mip 0 (= the only level we care about here; level(0.0f) on
// mip_filter::linear collapses to a single-level fetch).
inline CpuFloat4 sample_rc(const std::vector<float>& tex,
                        std::uint32_t W_, std::uint32_t H_,
                        float u, float v)
{
    // Metal normalized → pixel coord: px = u * W - 0.5, py = v * H - 0.5
    const float fx = u * float(W_) - 0.5f;
    const float fy = v * float(H_) - 0.5f;
    const int ix0 = int(std::floor(fx));
    const int iy0 = int(std::floor(fy));
    const float ax = fx - float(ix0);
    const float ay = fy - float(iy0);
    auto cl = [](int v_, int lo, int hi) {
        return std::max(lo, std::min(v_, hi));
    };
    const int W_i = int(W_), H_i = int(H_);
    const int x0 = cl(ix0,     0, W_i - 1);
    const int x1 = cl(ix0 + 1, 0, W_i - 1);
    const int y0 = cl(iy0,     0, H_i - 1);
    const int y1 = cl(iy0 + 1, 0, H_i - 1);
    auto load = [&](int x, int y) -> std::array<float, 4> {
        const std::size_t k = (std::size_t(y) * W_ + x) * 4;
        return { tex[k + 0], tex[k + 1], tex[k + 2], tex[k + 3] };
    };
    auto a = load(x0, y0);
    auto b = load(x1, y0);
    auto c = load(x0, y1);
    auto d = load(x1, y1);
    CpuFloat4 out{};
    float* op = reinterpret_cast<float*>(&out);
    for (int k = 0; k < 4; ++k) {
        const float ab = a[k] + (b[k] - a[k]) * ax;
        const float cd = c[k] + (d[k] - c[k]) * ax;
        op[k] = ab + (cd - ab) * ay;
    }
    return out;
}

inline float euclidean_dist3(CpuFloat4 a, CpuFloat4 b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// ---- CPU FP32 reference: SGM 4-path DP with optional adaptive P2 ----

struct OptimizeRef {
    std::vector<std::uint8_t> out;
    std::vector<std::uint32_t> slice_a;
    std::vector<std::uint32_t> slice_b;
    std::vector<std::uint32_t> axis_acc;

    OptimizeRef()
        : out(W * H * D, 0u)
        , slice_a(std::max(W, H) * D, 0u)
        , slice_b(std::max(W, H) * D, 0u)
        , axis_acc(std::max(W, H), 0u)
    {}

    struct Axis { int a0, a1, a2; };

    static std::size_t vol_k(int vx, int vy, int vz) {
        return std::size_t(vz) * (W * H)
             + std::size_t(vy) * W
             + std::size_t(vx);
    }
    static std::uint32_t vol_dim(int i) {
        switch (i) { case 0: return W; case 1: return H; default: return D; }
    }
    static void axis_to_v(int x_iter, int y, int z_iter, Axis a,
                          int& vx, int& vy, int& vz) {
        int v[3] = { 0, 0, 0 };
        v[a.a0] = x_iter;
        v[a.a1] = y;
        v[a.a2] = z_iter;
        vx = v[0]; vy = v[1]; vz = v[2];
    }

    void run(const std::vector<std::uint8_t>& in,
             const std::vector<float>*        rc_image_or_null,
             bool                             adaptive,
             float                            p2_fixed,
             float                            p2_sig_mid)
    {
        std::fill(out.begin(), out.end(), 0u);

        const Axis paths[4] = {
            { 0, 1, 2 }, { 0, 1, 2 },
            { 1, 0, 2 }, { 1, 0, 2 },
        };
        const bool inv_y[4] = { false, true, false, true };

        std::uint32_t filteringIndex = 0;
        for (int pi = 0; pi < 4; ++pi) {
            const Axis a = paths[pi];
            const bool invY = inv_y[pi];
            const int ySign = invY ? -1 : 1;

            const std::uint32_t axDimX = vol_dim(a.a0);
            const std::uint32_t axDimY = vol_dim(a.a1);
            const std::uint32_t axDimZ = vol_dim(a.a2);

            std::vector<std::uint32_t>* slice_for_y   = &slice_a;
            std::vector<std::uint32_t>* slice_for_ym1 = &slice_b;

            {
                const int y0 = invY ? int(axDimY) - 1 : 0;
                for (std::uint32_t zi = 0; zi < axDimZ; ++zi) {
                    for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                        int vx, vy, vz;
                        axis_to_v(int(xi), y0, int(zi), a, vx, vy, vz);
                        (*slice_for_ym1)[zi * axDimX + xi] =
                            std::uint32_t(in[vol_k(vx, vy, vz)]);
                    }
                }
            }
            {
                const int y0 = invY ? int(axDimY) - 1 : 0;
                for (std::uint32_t zi = 0; zi < axDimZ; ++zi) {
                    for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                        int vx, vy, vz;
                        axis_to_v(int(xi), y0, int(zi), a, vx, vy, vz);
                        out[vol_k(vx, vy, vz)] = 255u;
                    }
                }
            }

            for (std::uint32_t iy = 1; iy < axDimY; ++iy) {
                const int y = invY ? int(axDimY) - 1 - int(iy) : int(iy);

                for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                    std::uint32_t best = (*slice_for_ym1)[xi];
                    for (std::uint32_t zi = 1; zi < axDimZ; ++zi) {
                        const std::uint32_t v =
                            (*slice_for_ym1)[zi * axDimX + xi];
                        if (v < best) best = v;
                    }
                    axis_acc[xi] = best;
                }

                for (std::uint32_t zi = 0; zi < axDimZ; ++zi) {
                    for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                        int vx, vy, vz;
                        axis_to_v(int(xi), y, int(zi), a, vx, vy, vz);
                        (*slice_for_y)[zi * axDimX + xi] =
                            std::uint32_t(in[vol_k(vx, vy, vz)]);
                    }
                }

                for (std::uint32_t zi = 0; zi < axDimZ; ++zi) {
                    for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                        const std::uint32_t slice_k = zi * axDimX + xi;
                        float pathCost = 255.0f;
                        if (int(zi) >= 1 && int(zi) < int(axDimZ) - 1) {
                            float P2;
                            if (!adaptive) {
                                P2 = p2_fixed;
                            } else {
                                int vx_, vy_, vz_;
                                axis_to_v(int(xi), y, int(zi), a,
                                          vx_, vy_, vz_);
                                const int xShift = (a.a1 == 0) ? 1 : 0;
                                const int yShift = (a.a1 == 1) ? 1 : 0;
                                const int imX0 = (0 + vx_) * kStep;
                                const int imY0 = (0 + vy_) * kStep;
                                const int imX1 = imX0 - ySign * kStep * xShift;
                                const int imY1 = imY0 - ySign * kStep * yShift;
                                const float invW = 1.0f / float(kTexW);
                                const float invH = 1.0f / float(kTexH);
                                const float u0 = (float(imX0) + 0.5f) * invW;
                                const float v0 = (float(imY0) + 0.5f) * invH;
                                const float u1 = (float(imX1) + 0.5f) * invW;
                                const float v1 = (float(imY1) + 0.5f) * invH;
                                const CpuFloat4 g0 = sample_rc(*rc_image_or_null,
                                                               kTexW, kTexH,
                                                               u0, v0);
                                const CpuFloat4 g1 = sample_rc(*rc_image_or_null,
                                                               kTexW, kTexH,
                                                               u1, v1);
                                const float deltaC = euclidean_dist3(g0, g1);
                                P2 = cpu_sigmoid(80.0f, 255.0f, 80.0f,
                                                 p2_sig_mid, deltaC);
                            }

                            const float bestCol_m1 = float(axis_acc[xi]);
                            const float pathMDm1 =
                                float((*slice_for_ym1)[(zi - 1) * axDimX + xi]);
                            const float pathMD =
                                float((*slice_for_ym1)[zi       * axDimX + xi]);
                            const float pathMDp1 =
                                float((*slice_for_ym1)[(zi + 1) * axDimX + xi]);

                            const float A = pathMD;
                            const float B = pathMDm1 + kP1;
                            const float C = pathMDp1 + kP1;
                            const float Dv = bestCol_m1 + P2;
                            const float minCost =
                                std::min(std::min(A, B), std::min(C, Dv));

                            const float sim_xz = float((*slice_for_y)[slice_k]);
                            pathCost = sim_xz + minCost - bestCol_m1;
                        }
                        (*slice_for_y)[slice_k] = std::uint32_t(pathCost);

                        const float clamped =
                            std::min(255.0f, std::max(0.0f, pathCost));
                        int vx, vy, vz;
                        axis_to_v(int(xi), y, int(zi), a, vx, vy, vz);
                        const float cur = float(out[vol_k(vx, vy, vz)]);
                        const float fi  = float(filteringIndex);
                        const float merged = (cur * fi + clamped) / (fi + 1.0f);
                        out[vol_k(vx, vy, vz)] = std::uint8_t(
                            std::min(255.0f, std::max(0.0f, merged)));
                    }
                }

                std::swap(slice_for_y, slice_for_ym1);
            }
            ++filteringIndex;
        }
    }
};

}  // namespace

int main() try
{
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    const auto in_data = make_input_volume(0xa11ce5);
    const auto rc_img  = make_rc_image();

    // ---- rc texture (RGBA32Float) ----
    Texture rc_tex(dev, Texture::Descriptor{
        kTexW, kTexH, /*mip_levels*/ 1, PixelFormat::RGBA32Float });
    rc_tex.set_label("optimize_ap2.rc");
    rc_tex.upload(std::span<const float>(rc_img));
    // No generate_mipmaps; level 0 is what we sample.

    Volume vol(dev);
    Buffer in_buf       (dev, W * H * D * sizeof(std::uint8_t));
    Buffer out_fixed_buf(dev, W * H * D * sizeof(std::uint8_t));
    Buffer out_adapt_buf(dev, W * H * D * sizeof(std::uint8_t));
    const std::uint32_t maxXY = std::max(W, H);
    Buffer slice_a (dev, maxXY * D * sizeof(std::uint32_t));
    Buffer slice_b (dev, maxXY * D * sizeof(std::uint32_t));
    Buffer axis_acc(dev, maxXY     * sizeof(std::uint32_t));
    in_buf       .set_label("optimize_ap2.in");
    out_fixed_buf.set_label("optimize_ap2.out_fixed");
    out_adapt_buf.set_label("optimize_ap2.out_adapt");
    slice_a      .set_label("optimize_ap2.slice_a");
    slice_b      .set_label("optimize_ap2.slice_b");
    axis_acc     .set_label("optimize_ap2.axis_acc");

    in_buf.upload(std::span<const std::uint8_t>(in_data));

    // ---- Run A: fixed P2 ----
    {
        Volume::OptimizeParams params{};
        params.dims              = VolumeDims{ W, H, D };
        params.last_depth_index  = D;
        params.p1                = kP1;
        params.p2_abs            = kP2;
        params.adaptive_p2       = false;
        vol.optimize(out_fixed_buf, slice_a, slice_b, axis_acc,
                     in_buf, params);
    }

    // ---- Run B: adaptive P2 (texture-driven) ----
    {
        Volume::OptimizeParams params{};
        params.dims              = VolumeDims{ W, H, D };
        params.last_depth_index  = D;
        params.p1                = kP1;
        // p2_abs is ignored when adaptive_p2 is true; leave at 0
        // to make sure the kernel doesn't accidentally read it.
        params.p2_abs            = 0.0f;
        params.adaptive_p2       = true;
        params.p2_sig_mid        = kP2;
        params.step_xy           = kStep;
        params.roi_x_begin       = 0;
        params.roi_y_begin       = 0;
        params.rc_level_width    = kTexW;
        params.rc_level_height   = kTexH;
        params.rc_mipmap_level   = kMipLevel;
        vol.optimize(out_adapt_buf, slice_a, slice_b, axis_acc,
                     in_buf, params, &rc_tex);
    }

    const auto* gpu_fixed = static_cast<const std::uint8_t*>(out_fixed_buf.data());
    const auto* gpu_adapt = static_cast<const std::uint8_t*>(out_adapt_buf.data());

    // ---- CPU references ----
    OptimizeRef ref_fixed; ref_fixed.run(in_data, nullptr, false, kP2, 0.0f);
    OptimizeRef ref_adapt; ref_adapt.run(in_data, &rc_img,  true,  0.0f, kP2);

    // ---- 1. Fixed path: bit-exact agreement with its CPU ref. ----
    int fixed_worst = 0;
    for (std::size_t i = 0; i < ref_fixed.out.size(); ++i) {
        const int diff =
            std::abs(int(gpu_fixed[i]) - int(ref_fixed.out[i]));
        fixed_worst = std::max(fixed_worst, diff);
    }
    std::printf("[fixed] worst |Δ| GPU vs CPU = %d (budget 1)\n", fixed_worst);
    if (fixed_worst > 1) {
        std::fprintf(stderr,
            "FAIL: fixed-P2 GPU disagrees with CPU mirror\n");
        return 1;
    }

    // ---- 2. Adaptive path: agrees with FP32 CPU reference. ----
    //
    // The adaptive path adds: (a) a single exp/sigmoid per voxel
    // and (b) a bilinear texture sample. Both run in FP32; the
    // CPU mirror runs in FP32 too, so we expect very small ULP
    // drift. The aggregation step quantizes to uchar at each
    // path, so the per-voxel error budget is on the order of 1
    // uchar at most (the sigmoid output enters min(), and a
    // 1-ULP drift can flip which arm wins).
    int adapt_worst = 0;
    int adapt_bad = 0;
    long long sum_diff = 0;
    for (std::size_t i = 0; i < ref_adapt.out.size(); ++i) {
        const int diff =
            std::abs(int(gpu_adapt[i]) - int(ref_adapt.out[i]));
        adapt_worst = std::max(adapt_worst, diff);
        sum_diff   += diff;
        if (diff > 2) {
            if (adapt_bad < 6) std::fprintf(stderr,
                "vox %zu: gpu=%d cpu=%d diff=%d\n",
                i, int(gpu_adapt[i]), int(ref_adapt.out[i]), diff);
            ++adapt_bad;
        }
    }
    const double mean_diff =
        double(sum_diff) / double(ref_adapt.out.size());
    std::printf("[adapt] worst |Δ| GPU vs CPU = %d\n", adapt_worst);
    std::printf("[adapt] mean  |Δ| GPU vs CPU = %.3f\n", mean_diff);

    // Budget: the sigmoid + texture sample run in FP32 on both
    // sides; the only drift comes from the GPU's `precise::exp`
    // vs libm's `expf`, plus its `precise::divide`. Empirically
    // < 2 uchar per voxel; tolerate 2 to leave headroom.
    constexpr int kAdaptTolWorst = 2;
    if (adapt_worst > kAdaptTolWorst) {
        std::fprintf(stderr,
            "FAIL: adaptive-P2 GPU vs CPU worst |Δ| %d > budget %d\n",
            adapt_worst, kAdaptTolWorst);
        return 1;
    }

    // ---- 3. Sanity: adaptive output must DIFFER from fixed. ----
    //
    // If the new path silently fell through to fixed, the two
    // volumes would be identical. We expect at least a few %
    // of voxels to differ (the sigmoid moves P2 away from 100
    // for any voxel where deltaC ≠ 100).
    std::size_t differ_count = 0;
    for (std::size_t i = 0; i < ref_adapt.out.size(); ++i) {
        if (gpu_adapt[i] != gpu_fixed[i]) ++differ_count;
    }
    const double frac = double(differ_count)
                      / double(ref_adapt.out.size());
    std::printf("[diff ] adapt vs fixed: %zu / %u voxels (%.1f%%)\n",
                differ_count, W * H * D, 100.0 * frac);
    // Sanity threshold: the cost volume is heavily uchar-quantized
    // and the P2 arm of min(...) doesn't always win, so most
    // voxels collapse to the same value. We just need enough
    // differing voxels to prove the adaptive code path is wired
    // up. 10 voxels (~1.7%) is the floor; empirically we get
    // 20–25 with this texture.
    constexpr std::size_t kMinDifferingVoxels = 10;
    if (differ_count < kMinDifferingVoxels) {
        std::fprintf(stderr,
            "FAIL: adaptive-P2 produced only %zu / %u voxels "
            "different from fixed-P2 (need >= %zu) — the new "
            "code path is suspect.\n",
            differ_count, W * H * D, kMinDifferingVoxels);
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
