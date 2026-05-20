// test_color.cpp — numerical validation of the Metal port of
// depthMap/cuda/device/color.cuh against an FP64 CPU reference.
//
// The reference implements the same formulas in double precision
// and we tolerate FP32 drift through the algorithm chain. Notable
// budgets:
//   * srgb2rgb / rgb2xyz / rgb2hsl: 1e-5  abs/rel
//   * xyz2lab: 1e-3  (cbrt + chained mults amplify FP32 noise)
//   * euclideanDist3: 1e-6
//   * CostYKfromLab (both arities): 1e-3 (fast-exp at the tail)
//
// Inputs cover the full sRGB unit cube + spatial separations of
// (-7..+7) px and γ values in physically reasonable ranges.

#include "av/depth_map/ColorOps.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kCases = 4096;

constexpr double kTolLinear = 1e-5;   // srgb2rgb
constexpr double kTolXYZ    = 1e-5;
constexpr double kTolHSL    = 1e-5;
constexpr double kTolLAB    = 1e-3;
constexpr double kTolDist   = 1e-6;
constexpr double kTolCost6  = 1e-3;
constexpr double kTolCost3  = 1e-3;

constexpr std::size_t kInPer  = av::depth_map::ColorOps::kInPerCase;
constexpr std::size_t kOutPer = av::depth_map::ColorOps::kOutPerCase;

// ---------------- FP64 CPU reference ----------------

struct V3 { double x, y, z; };

V3 srgb2rgb_ref(V3 c) {
    auto e = [](double v) {
        return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
    };
    return { e(c.x), e(c.y), e(c.z) };
}

V3 rgb2xyz_ref(V3 c) {
    return { 0.4124564 * c.x + 0.3575761 * c.y + 0.1804375 * c.z,
             0.2126729 * c.x + 0.7151522 * c.y + 0.0721750 * c.z,
             0.0193339 * c.x + 0.1191920 * c.y + 0.9503041 * c.z };
}

V3 rgb2hsl_ref(V3 c) {
    const double cmin = std::min({ c.x, c.y, c.z });
    const double cmax = std::max({ c.x, c.y, c.z });

    double h = 0.0;
    if (cmin == cmax) {
        // h = 0
    } else if (cmax == c.x) {
        h = ((c.y - c.z) / (cmax - cmin) + 6.0) / 6.0;
        if (h >= 1.0) h -= 1.0;
    } else if (cmax == c.y) {
        h = ((c.z - c.x) / (cmax - cmin) + 2.0) / 6.0;
    } else {
        h = ((c.x - c.y) / (cmax - cmin) + 4.0) / 6.0;
    }

    const double l = 0.5 * (cmin + cmax);
    double s = 0.0;
    if (cmin != cmax) {
        s = (l <= 0.5) ? (cmax - cmin) / (2.0 * l)
                       : (cmax - cmin) / (2.0 - 2.0 * l);
    }
    return { h, s, l };
}

V3 xyz2lab_ref(V3 c) {
    constexpr double kappa  = 24389.0 / 27.0;
    constexpr double thresh = 216.0 / 24389.0;
    V3 r = { c.x / 0.95047, c.y, c.z / 1.08883 };
    auto f = [&](double v) {
        return v > thresh ? std::cbrt(v) : (kappa * v + 16.0) / 116.0;
    };
    V3 fc = { f(r.x), f(r.y), f(r.z) };
    return { (116.0 * fc.y - 16.0)       * 2.55,
             (500.0 * (fc.x - fc.y))     * 2.55,
             (200.0 * (fc.y - fc.z))     * 2.55 };
}

double euclidean_dist_ref(V3 a, V3 b) {
    const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double cost6_ref(int dx, int dy, V3 c0, V3 c1,
                 double invGammaC, double invGammaP) {
    double deltaC = euclidean_dist_ref(c0, c1) * invGammaC;
    double deltaP = std::sqrt(double(dx * dx + dy * dy)) * invGammaP;
    return std::exp(-(deltaC + deltaP));
}

double cost3_ref(V3 c0, V3 c1, double invGammaC) {
    return std::exp(-(euclidean_dist_ref(c0, c1) * invGammaC));
}

double abs_rel(double g, double r) {
    return std::abs(g - r) / std::max(std::abs(r), 1.0);
}

}  // namespace

int main() try {
    auto dev = av::gpu::Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    av::depth_map::ColorOps ops(dev);

    std::mt19937_64 rng(0xc01023);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Dxy(-7, 7);
    std::uniform_real_distribution<double> Gc(1.0 / 50.0,   1.0 / 5.0);
    std::uniform_real_distribution<double> Gp(1.0 / 16.0,   1.0 / 2.0);

    std::vector<float> in (kCases * kInPer);
    std::vector<V3>    c0r(kCases), c1r(kCases);
    std::vector<int>   dxr(kCases), dyr(kCases);
    std::vector<double> gcr(kCases), gpr(kCases);

    for (std::size_t k = 0; k < kCases; ++k) {
        c0r[k] = { U01(rng), U01(rng), U01(rng) };
        c1r[k] = { U01(rng), U01(rng), U01(rng) };
        dxr[k] = Dxy(rng);
        dyr[k] = Dxy(rng);
        gcr[k] = Gc(rng);
        gpr[k] = Gp(rng);

        float* p = in.data() + k * kInPer;
        p[0] = static_cast<float>(c0r[k].x);
        p[1] = static_cast<float>(c0r[k].y);
        p[2] = static_cast<float>(c0r[k].z);
        p[3] = static_cast<float>(c1r[k].x);
        p[4] = static_cast<float>(c1r[k].y);
        p[5] = static_cast<float>(c1r[k].z);
        p[6] = static_cast<float>(dxr[k]);
        p[7] = static_cast<float>(dyr[k]);
        p[8] = static_cast<float>(gcr[k]);
        p[9] = static_cast<float>(gpr[k]);
        p[10] = p[11] = 0.0f;
    }

    std::vector<float> out(kCases * kOutPer);
    ops.validate(in, out);

    double worst_lin  = 0.0;
    double worst_xyz  = 0.0;
    double worst_hsl  = 0.0;
    double worst_lab  = 0.0;
    double worst_dist = 0.0;
    double worst_c6   = 0.0;
    double worst_c3   = 0.0;
    int bad = 0;

    for (std::size_t k = 0; k < kCases; ++k) {
        const float* o = out.data() + k * kOutPer;

        const V3 lin_ref = srgb2rgb_ref(c0r[k]);
        const V3 xyz_ref = rgb2xyz_ref(lin_ref);
        const V3 hsl_ref = rgb2hsl_ref(lin_ref);
        const V3 lab_ref = xyz2lab_ref(xyz_ref);

        // 1. sRGB → linear
        for (int i = 0; i < 3; ++i) {
            const double r = (&lin_ref.x)[i];
            const double e = abs_rel(static_cast<double>(o[i]), r);
            worst_lin = std::max(worst_lin, e);
            if (e > kTolLinear && bad < 3) {
                std::fprintf(stderr,
                    "k=%zu lin[%d]: gpu=%g ref=%g err=%g\n", k, i,
                    static_cast<double>(o[i]), r, e);
                ++bad;
            } else if (e > kTolLinear) { ++bad; }
        }

        // 2. linear → XYZ
        for (int i = 0; i < 3; ++i) {
            const double r = (&xyz_ref.x)[i];
            const double e = abs_rel(static_cast<double>(o[3 + i]), r);
            worst_xyz = std::max(worst_xyz, e);
            if (e > kTolXYZ) ++bad;
        }

        // 3. linear → HSL
        for (int i = 0; i < 3; ++i) {
            const double r = (&hsl_ref.x)[i];
            const double e = abs_rel(static_cast<double>(o[6 + i]), r);
            worst_hsl = std::max(worst_hsl, e);
            if (e > kTolHSL) ++bad;
        }

        // 4. XYZ → Lab
        for (int i = 0; i < 3; ++i) {
            const double r = (&lab_ref.x)[i];
            const double e = abs_rel(static_cast<double>(o[9 + i]), r);
            worst_lab = std::max(worst_lab, e);
            if (e > kTolLAB) ++bad;
        }

        // 5. euclideanDist3 (on the raw sRGB inputs c0, c1)
        {
            const double r = euclidean_dist_ref(c0r[k], c1r[k]);
            const double e = abs_rel(static_cast<double>(o[12]), r);
            worst_dist = std::max(worst_dist, e);
            if (e > kTolDist) ++bad;
        }

        // 6. CostYKfromLab (6-arg)
        {
            const double r = cost6_ref(dxr[k], dyr[k],
                                       c0r[k], c1r[k],
                                       gcr[k], gpr[k]);
            const double e = abs_rel(static_cast<double>(o[13]), r);
            worst_c6 = std::max(worst_c6, e);
            if (e > kTolCost6) ++bad;
        }

        // 7. CostYKfromLab (3-arg)
        {
            const double r = cost3_ref(c0r[k], c1r[k], gcr[k]);
            const double e = abs_rel(static_cast<double>(o[14]), r);
            worst_c3 = std::max(worst_c3, e);
            if (e > kTolCost3) ++bad;
        }
    }

    std::printf("[info] cases             : %zu\n", kCases);
    std::printf("[info] worst err sRGB→lin: %.3g (budget %.3g)\n", worst_lin,  kTolLinear);
    std::printf("[info] worst err lin→XYZ : %.3g (budget %.3g)\n", worst_xyz,  kTolXYZ);
    std::printf("[info] worst err lin→HSL : %.3g (budget %.3g)\n", worst_hsl,  kTolHSL);
    std::printf("[info] worst err XYZ→Lab : %.3g (budget %.3g)\n", worst_lab,  kTolLAB);
    std::printf("[info] worst err eucDist : %.3g (budget %.3g)\n", worst_dist, kTolDist);
    std::printf("[info] worst err cost6   : %.3g (budget %.3g)\n", worst_c6,   kTolCost6);
    std::printf("[info] worst err cost3   : %.3g (budget %.3g)\n", worst_c3,   kTolCost3);

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
