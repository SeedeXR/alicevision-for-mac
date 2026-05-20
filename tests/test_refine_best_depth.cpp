// test_refine_best_depth.cpp — validation of
// volume_refineBestDepth (Refine pipeline exit).
//
// The kernel is pure scalar arithmetic (no camera, no texture):
// Gaussian-weighted convolution over the FP16 refinement-volume's
// Z axis, find the sub-sample with the strongest signal, convert
// to depth.
//
// Strategy: build a synthetic half volume with a known, sharp
// best-Z peak per pixel. Run the kernel. CPU FP64 reference does
// the same arithmetic. Expect bit-exact-ish agreement (the only
// drift sources are FP32 exp() vs FP64 exp() and the half cast).

#include "av/depth_map/Volume.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using av::depth_map::Volume;
using av::depth_map::VolumeDims;

constexpr std::uint32_t W = 16;
constexpr std::uint32_t H = 12;
constexpr std::uint32_t D = 9;             // volDimZ
constexpr std::int32_t  kSamplesPerPixSize = 4;
constexpr std::int32_t  kHalfNbSamples     = 12;
constexpr std::int32_t  kHalfNbDepths      = (int(D) - 1) / 2;   // 4
constexpr float         kSigma             = 4.0f;
constexpr float         kTwoSigmaSq        = 2.0f * kSigma * kSigma;

// IEEE binary16 float ↔ bit-pattern helpers (re-used from S10 etc.).
std::uint16_t float_to_half_bits(float f) {
    const std::uint32_t fi = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (fi >> 16) & 0x8000u;
    std::int32_t        exp  = static_cast<std::int32_t>((fi >> 23) & 0xffu) - 127 + 15;
    std::uint32_t       mant =  fi & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
        const std::uint32_t round_bit = 1u << (shift - 1);
        std::uint32_t       half_mant = mant >> shift;
        if ((mant & ((1u << shift) - 1u)) > round_bit ||
            ((mant & ((1u << shift) - 1u)) == round_bit && (half_mant & 1u))) {
            half_mant += 1u;
        }
        return static_cast<std::uint16_t>(sign | half_mant);
    }
    if (exp >= 31) {
        if (((fi >> 23) & 0xffu) == 0xffu && mant != 0)
            return static_cast<std::uint16_t>(sign | 0x7c00u | (mant >> 13) | 1u);
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    const std::uint32_t round_bit = 1u << 12;
    std::uint32_t       half_mant = mant >> 13;
    if ((mant & 0x1fffu) > round_bit ||
        ((mant & 0x1fffu) == round_bit && (half_mant & 1u))) {
        half_mant += 1u;
    }
    if (half_mant == 0x400u) {
        half_mant = 0;
        exp += 1;
        if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | half_mant);
}
float half_bits_to_float(std::uint16_t h) {
    const std::uint32_t sign  = (h & 0x8000u) << 16;
    const std::uint32_t exp16 = (h >> 10) & 0x1fu;
    const std::uint32_t mant  =  h & 0x3ffu;
    if (exp16 == 0) {
        if (mant == 0) return std::bit_cast<float>(sign);
        int e = -1;
        std::uint32_t m = mant;
        do { m <<= 1; ++e; } while ((m & 0x400u) == 0);
        const std::uint32_t fi = sign
            | (static_cast<std::uint32_t>(127 - 15 - e) << 23)
            | ((m & 0x3ffu) << 13);
        return std::bit_cast<float>(fi);
    }
    if (exp16 == 31) {
        const std::uint32_t fi = sign | 0x7f800000u | (mant << 13);
        return std::bit_cast<float>(fi);
    }
    const std::uint32_t fi = sign
        | ((exp16 + 127 - 15) << 23)
        | (mant << 13);
    return std::bit_cast<float>(fi);
}

}  // namespace

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    const VolumeDims dims{ W, H, D };
    const std::size_t pix = std::size_t(W) * std::size_t(H);

    // ---- synthesize an FP16 refinement volume ----
    // Upstream's convention: best = HIGHEST in the input volume
    // (the kernel flips with -invSimSum). Build a per-pixel sim
    // pattern where the max is at a known relative-Z. To exercise
    // both "middle wins" and "off-center wins" cases, randomly
    // pick a per-pixel target Z.
    std::mt19937_64 rng(0xd00d);
    std::uniform_int_distribution<int> tZdist(0, int(D) - 1);
    std::uniform_real_distribution<float> sgm_pix_dist(0.005f, 0.020f);
    std::uniform_real_distribution<float> sgm_depth_jitter(-0.05f, 0.05f);

    std::vector<std::uint16_t> vol(W * H * D, 0);
    std::vector<float> sgm_dp(pix * 2);
    std::vector<int> truth_z(pix);

    // A few pixels are invalid (sgm depth < 0) to exercise the
    // bypass branch.
    constexpr int kInvalidEveryN = 23;

    for (std::uint32_t k = 0; k < pix; ++k) {
        const float sgm_pix_size = sgm_pix_dist(rng);
        if (int(k) % kInvalidEveryN == 0) {
            sgm_dp[k * 2 + 0] = -1.0f;
            sgm_dp[k * 2 + 1] = sgm_pix_size;
            truth_z[k] = -1;
            continue;
        }
        sgm_dp[k * 2 + 0] = 4.0f + sgm_depth_jitter(rng);   // ≈ 4
        sgm_dp[k * 2 + 1] = sgm_pix_size;
        const int true_z = tZdist(rng);
        truth_z[k] = true_z;
        for (std::uint32_t vz = 0; vz < D; ++vz) {
            const int rz = int(vz) - kHalfNbDepths;
            const int tz = true_z - kHalfNbDepths;
            const int delta = rz - tz;
            // Triangular peak around the truth, decaying with |delta|.
            const float sim = std::max(0.0f, 0.9f - 0.18f * float(std::abs(delta)));
            vol[std::size_t(vz) * pix + k] = float_to_half_bits(sim);
        }
    }

    // ---- GPU buffers ----
    Buffer vol_buf  (dev, vol.size() * sizeof(std::uint16_t));
    Buffer sgm_buf  (dev, sgm_dp.size() * sizeof(float));
    Buffer out_buf  (dev, pix * 2 * sizeof(float));
    vol_buf.upload(std::span<const std::uint16_t>(vol));
    sgm_buf.upload(std::span<const float>(sgm_dp));

    Volume vol_runner(dev);
    Volume::RefineBestDepthParams p{};
    p.dims                    = dims;
    p.samples_per_pix_size    = kSamplesPerPixSize;
    p.half_nb_samples         = kHalfNbSamples;
    p.half_nb_depths          = kHalfNbDepths;
    p.two_times_sigma_pow_two = kTwoSigmaSq;
    p.roi_width               = W;
    p.roi_height              = H;
    vol_runner.refine_best_depth(out_buf, sgm_buf, vol_buf, p);

    const auto* gpu = static_cast<const float*>(out_buf.data());

    // ---- CPU FP64 reference ----
    int bad = 0;
    int invalid_count = 0;
    int valid_count = 0;
    double worst_depth_err = 0.0;
    double worst_sim_err   = 0.0;

    for (std::uint32_t k = 0; k < pix; ++k) {
        const float gpu_depth = gpu[k * 2 + 0];
        const float gpu_sim   = gpu[k * 2 + 1];

        if (sgm_dp[k * 2 + 0] <= 0.0f) {
            ++invalid_count;
            // Expected: passthrough (-1, 1.0).
            if (gpu_depth != sgm_dp[k * 2 + 0] || gpu_sim != 1.0f) {
                if (bad < 3) std::fprintf(stderr,
                    "invalid k=%u: gpu=(%g, %g) expected=(%g, %g)\n",
                    k,
                    static_cast<double>(gpu_depth),
                    static_cast<double>(gpu_sim),
                    static_cast<double>(sgm_dp[k * 2 + 0]),
                    1.0);
                ++bad;
            }
            continue;
        }
        ++valid_count;

        // Replicate the kernel arithmetic in FP64.
        double best_sample_sim = 0.0;
        int    best_sample_off = 0;
        for (int sample = -kHalfNbSamples; sample <= kHalfNbSamples; ++sample) {
            double sample_sim = 0.0;
            for (std::uint32_t vz = 0; vz < D; ++vz) {
                const int rz = int(vz) - kHalfNbDepths;
                const int zs = rz * kSamplesPerPixSize;
                const double invSimSum =
                    double(half_bits_to_float(vol[std::size_t(vz) * pix + k]));
                const double simSum = -invSimSum;
                const double d = double(zs - sample);
                sample_sim += simSum * std::exp(-(d * d) / double(kTwoSigmaSq));
            }
            if (sample_sim < best_sample_sim) {
                best_sample_sim = sample_sim;
                best_sample_off = sample;
            }
        }
        const double sample_size = double(sgm_dp[k * 2 + 1])
                                 / double(kSamplesPerPixSize);
        const double ref_depth   = double(sgm_dp[k * 2 + 0])
                                 + double(best_sample_off) * sample_size;
        const double ref_sim     = best_sample_sim;

        const double d_err = std::abs(double(gpu_depth) - ref_depth);
        const double s_err = std::abs(double(gpu_sim)   - ref_sim);
        worst_depth_err = std::max(worst_depth_err, d_err);
        worst_sim_err   = std::max(worst_sim_err,   s_err);

        // Budget:
        //   * depth: depends on sample_size (~ 0.001..0.005) and
        //     whether GPU & CPU pick the same best_sample_off.
        //     Allow up to one sample_size for cases where multiple
        //     sub-samples have nearly-equal weighted sums and FP32
        //     vs FP64 noise flips the winner.
        //   * sim: small FP32 / FP64 drift in the Gaussian sum.
        //     Budget 5e-3.
        const double depth_budget = double(sgm_dp[k * 2 + 1])
                                  / double(kSamplesPerPixSize)
                                  * 1.1;   // ≈ 1 sample_size
        constexpr double kSimBudget = 5e-3;
        if (d_err > depth_budget || s_err > kSimBudget) {
            if (bad < 4) std::fprintf(stderr,
                "k=%u: gpu=(%g, %g) ref=(%g, %g) Δdepth=%g Δsim=%g "
                "(budget depth=%g sim=%g)\n",
                k,
                static_cast<double>(gpu_depth), static_cast<double>(gpu_sim),
                ref_depth, ref_sim, d_err, s_err,
                depth_budget, kSimBudget);
            ++bad;
        }
    }

    std::printf("[info] pixels        : %zu (valid=%d, invalid=%d)\n",
                pix, valid_count, invalid_count);
    std::printf("[info] worst |Δdepth| : %.3g\n", worst_depth_err);
    std::printf("[info] worst |Δsim|   : %.3g\n", worst_sim_err);

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
