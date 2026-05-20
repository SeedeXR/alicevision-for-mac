// test_simstat.cpp — numerical validation of the Metal port of
// depthMap/cuda/device/SimStat.cuh against an FP64 CPU reference.
//
// Each test case feeds the GPU's per-thread `simStat` accumulator
// 64 (x, y, w) samples. The CPU reproduces the exact arithmetic
// in FP64 and we tolerate FP32 noise — a 64-term weighted sum at
// operands of order 1 should preserve ~5 decimal digits.

#include "av/depth_map/SimStatOps.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kCases     = 1024;
constexpr std::size_t kSamples   = av::depth_map::SimStatOps::kSamples;
constexpr std::size_t kInPer     = av::depth_map::SimStatOps::kInPerCase;
constexpr std::size_t kOutPer    = av::depth_map::SimStatOps::kOutPerCase;

// Mixed abs/rel error budgets:
//   |g − r| / max(|r|, 1.0) — for variances tiny |r| stays raw.
constexpr double kTolVar = 5e-5;
constexpr double kTolSim = 5e-5;

struct RefStat {
    double xsum = 0, ysum = 0, xxsum = 0, yysum = 0, xysum = 0, wsum = 0, count = 0;
    void update(double x, double y, double w) {
        wsum  += w;
        count += 1.0;
        xsum  += w * x;
        ysum  += w * y;
        xxsum += w * x * x;
        yysum += w * y * y;
        xysum += w * x * y;
    }
    double varXW () const { return (xxsum - xsum * xsum / wsum) / wsum; }
    double varYW () const { return (yysum - ysum * ysum / wsum) / wsum; }
    double varXYW() const { return (xysum - xsum * ysum / wsum) / wsum; }
    double wSim()  const {
        const double raw = varXYW() / std::sqrt(varXW() * varYW());
        return std::isfinite(raw) ? -raw : 1.0;
    }
};

double abs_rel(double g, double r) {
    return std::abs(g - r) / std::max(std::abs(r), 1.0);
}

}  // namespace

int main() try {
    auto dev = av::gpu::Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    av::depth_map::SimStatOps ops(dev);

    std::mt19937_64 rng(0x515145741);
    // y is correlated with x (via a fixed slope + Gaussian noise) so
    // computeWSim produces a meaningfully non-zero answer per case.
    std::uniform_real_distribution<double> X(0.1, 1.0);
    std::uniform_real_distribution<double> slope(-1.5, 1.5);
    std::normal_distribution<double>       Noise(0.0, 0.05);
    std::uniform_real_distribution<double> W(0.1, 1.0);

    std::vector<float>   in (kCases * kInPer);
    std::vector<RefStat> ref(kCases);

    for (std::size_t k = 0; k < kCases; ++k) {
        const double a = slope(rng);
        const double b = Noise(rng) * 5.0;
        for (std::size_t i = 0; i < kSamples; ++i) {
            const double x = X(rng);
            const double y = a * x + b + Noise(rng);
            const double w = W(rng);
            ref[k].update(x, y, w);

            float* p = in.data() + k * kInPer + i * 3;
            p[0] = static_cast<float>(x);
            p[1] = static_cast<float>(y);
            p[2] = static_cast<float>(w);
        }
    }

    std::vector<float> out(kCases * kOutPer);
    ops.validate(in, out);

    double worst_varxw  = 0.0;
    double worst_varyw  = 0.0;
    double worst_varxyw = 0.0;
    double worst_sim    = 0.0;
    int bad = 0;

    for (std::size_t k = 0; k < kCases; ++k) {
        const float* o = out.data() + k * kOutPer;
        const RefStat& r = ref[k];

        const double e_xw  = abs_rel(static_cast<double>(o[0]), r.varXW());
        const double e_yw  = abs_rel(static_cast<double>(o[1]), r.varYW());
        const double e_xyw = abs_rel(static_cast<double>(o[2]), r.varXYW());
        const double e_sim = abs_rel(static_cast<double>(o[3]), r.wSim());

        worst_varxw  = std::max(worst_varxw,  e_xw);
        worst_varyw  = std::max(worst_varyw,  e_yw);
        worst_varxyw = std::max(worst_varxyw, e_xyw);
        worst_sim    = std::max(worst_sim,    e_sim);

        if (e_xw > kTolVar || e_yw > kTolVar || e_xyw > kTolVar) {
            if (bad < 3) std::fprintf(stderr,
                "k=%zu variances: xw=%g yw=%g xyw=%g\n",
                k, e_xw, e_yw, e_xyw);
            ++bad;
        }
        if (e_sim > kTolSim) {
            if (bad < 3) std::fprintf(stderr,
                "k=%zu sim: gpu=%g ref=%g err=%g\n",
                k, static_cast<double>(o[3]), r.wSim(), e_sim);
            ++bad;
        }
    }

    std::printf("[info] cases             : %zu\n", kCases);
    std::printf("[info] worst err varXW   : %.3g (budget %.3g)\n", worst_varxw,  kTolVar);
    std::printf("[info] worst err varYW   : %.3g (budget %.3g)\n", worst_varyw,  kTolVar);
    std::printf("[info] worst err varXYW  : %.3g (budget %.3g)\n", worst_varxyw, kTolVar);
    std::printf("[info] worst err sim     : %.3g (budget %.3g)\n", worst_sim,    kTolSim);

    if (bad) {
        std::fprintf(stderr, "FAIL: %d failures\n", bad);
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
