// test_matrix.cpp — numerical validation of the Metal port of
// depthMap/cuda/device/matrix.cuh against an Eigen FP64 reference.
//
// Per case, the kernel exercises six load-bearing helpers:
//   M3x3mulV3, M3x3mulM3x3, M3x4mulV3, project3DPoint,
//   outerMultiply, lineLineIntersect.
//
// For each, we compute the expected output in Eigen (FP64),
// compare element-wise against the FP32 GPU output, and fail if
// any element exceeds an absolute+relative tolerance budget.
//
// Tolerance: matched to FP32 ulp scale on values of order O(1)
// after the operations above; bumped for lineLineIntersect which
// has a division and is sensitive to near-parallel inputs.

#include "av/depth_map/MatrixOps.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kCases       = 4096;

// Tolerances are MIXED absolute/relative error metrics: each is
//     err = |gpu - ref| / max(|ref|, 1.0)
// so for |ref| >= 1 the budget is interpreted as relative error,
// and for |ref| < 1 it's interpreted as absolute error. This is
// the right metric for FP32-vs-FP64 differences on random N(0,1)
// inputs, where some output components naturally land near zero.
//
// Budgets are calibrated against FP32 ULP×N for an N-term sum
// at operands of order 1:
//   1 mul   →  ~1.2e-7
//   3 terms →  ~4e-7
//   4 terms →  ~5e-7
//   9 mul + 6 add (mat-mat element) → ~2e-6
// The lineLineIntersect operation includes two divisions and
// chained dot products on differences; its budget is larger by
// roughly the conditioning of the involved cross terms.
constexpr double kTolMV    = 5e-6;   // M3x3 * v        (3-term sum)
constexpr double kTolMM    = 1e-5;   // M3x3 * M3x3     (each element: 3-term)
constexpr double kTolP     = 5e-6;   // M3x4 * v        (4-term sum)
constexpr double kTolProj  = 5e-5;   // project3DPoint  (4 terms + div)
constexpr double kTolOuter = 1e-6;   // outer (single multiply)
constexpr double kTolLLI   = 5e-3;   // lineLineIntersect (div-heavy)

// Layout
constexpr std::size_t kInPer  = av::depth_map::MatrixOps::kInPerCase;
constexpr std::size_t kOutPer = av::depth_map::MatrixOps::kOutPerCase;

struct CaseRef {
    Eigen::Matrix3d  A;
    Eigen::Matrix3d  B;
    Eigen::Matrix<double, 3, 4> P;
    Eigen::Vector3d  v3;
    Eigen::Vector3d  u3;
    Eigen::Vector3d  p1, p2, p3, p4;
};

void make_cases(std::size_t n,
                std::vector<float>&    in,
                std::vector<CaseRef>&  ref,
                std::uint64_t          seed)
{
    in.resize(n * kInPer);
    ref.resize(n);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> N(0.0, 1.0);

    for (std::size_t k = 0; k < n; ++k) {
        CaseRef& r = ref[k];

        // Random 3x3 matrices, modest scale.
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                r.A(i, j) = N(rng);
                r.B(i, j) = N(rng);
            }
        // 3x4 camera-like matrix.
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                r.P(i, j) = N(rng);
        // Bias P's z-row so perspective denominator is well above 0
        // (avoids near-singularities — we're testing the helper, not
        // its numerical-stability budget at the cliff).
        r.P(2, 3) += 5.0;

        for (int i = 0; i < 3; ++i) {
            r.v3(i) = N(rng);
            r.u3(i) = N(rng);
            r.p1(i) = N(rng);
            r.p2(i) = N(rng) + 1.0;  // bias to avoid p2==p1
            r.p3(i) = N(rng);
            r.p4(i) = N(rng) + 1.0;
        }

        // Pack into the float input span — column-major flat for
        // matrices, matching matrix.h conventions.
        float* row = in.data() + k * kInPer;
        // A
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                row[j * 3 + i] = static_cast<float>(r.A(i, j));
        // B
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                row[9 + j * 3 + i] = static_cast<float>(r.B(i, j));
        // P
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 3; ++i)
                row[18 + j * 3 + i] = static_cast<float>(r.P(i, j));
        // v3, u3
        row[30] = static_cast<float>(r.v3(0));
        row[31] = static_cast<float>(r.v3(1));
        row[32] = static_cast<float>(r.v3(2));
        row[33] = static_cast<float>(r.u3(0));
        row[34] = static_cast<float>(r.u3(1));
        row[35] = static_cast<float>(r.u3(2));
        // p1..p4
        for (int t = 0; t < 4; ++t) {
            const Eigen::Vector3d& p = (t == 0) ? r.p1
                                     : (t == 1) ? r.p2
                                     : (t == 2) ? r.p3 : r.p4;
            row[36 + t * 3 + 0] = static_cast<float>(p(0));
            row[36 + t * 3 + 1] = static_cast<float>(p(1));
            row[36 + t * 3 + 2] = static_cast<float>(p(2));
        }
    }
}

// Compare a GPU float result against an Eigen FP64 reference,
// element-wise. Returns the worst mixed abs/rel error observed:
//
//     err = |g - r| / max(|r|, 1.0)
//
// - For |r| >= 1: this is the *relative* error.
// - For |r| <  1: this is the *absolute* error.
//
// This is the right metric for outputs that can legitimately land
// near zero by cancellation (random matrix products), where a pure
// relative metric falsely amplifies negligible absolute errors.
template <class M>
double abs_rel_err(const M& gpu, const M& ref)
{
    double worst = 0.0;
    for (Eigen::Index i = 0; i < gpu.size(); ++i) {
        const double r = ref(i);
        const double g = static_cast<double>(gpu(i));
        const double scale = std::max(std::abs(r), 1.0);
        const double err   = std::abs(g - r) / scale;
        worst = std::max(worst, err);
    }
    return worst;
}

}  // namespace

int main() try
{
    auto dev = av::gpu::Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    av::depth_map::MatrixOps ops(dev);

    std::vector<float>   in;
    std::vector<CaseRef> refs;
    make_cases(kCases, in, refs, 0xdeadbeefULL);

    std::vector<float> out(kCases * kOutPer);
    ops.validate(in, out);

    double worst_mv    = 0;
    double worst_mm    = 0;
    double worst_p     = 0;
    double worst_proj  = 0;
    double worst_outer = 0;
    double worst_lli   = 0;
    int bad = 0;

    for (std::size_t k = 0; k < kCases; ++k) {
        const float* o = out.data() + k * kOutPer;
        const CaseRef& r = refs[k];

        // 1. Av
        {
            Eigen::Vector3d ref = r.A * r.v3;
            Eigen::Vector3f gpu(o[0], o[1], o[2]);
            const double e = abs_rel_err(gpu.cast<double>().eval(), ref);
            worst_mv = std::max(worst_mv, e);
            if (e > kTolMV) {
                if (bad < 3) std::fprintf(stderr,
                    "k=%zu Av err=%g\n", k, e);
                ++bad;
            }
        }

        // 2. AB
        {
            Eigen::Matrix3d ref = r.A * r.B;
            Eigen::Matrix3f gpu;
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    gpu(i, j) = o[3 + j * 3 + i];
            const double e = abs_rel_err(gpu.cast<double>().eval(), ref);
            worst_mm = std::max(worst_mm, e);
            if (e > kTolMM) {
                if (bad < 3) std::fprintf(stderr,
                    "k=%zu AB err=%g\n", k, e);
                ++bad;
            }
        }

        // 3. Pv = P * (v3, 1)   (affine, before perspective divide)
        {
            Eigen::Vector4d vh(r.v3(0), r.v3(1), r.v3(2), 1.0);
            Eigen::Vector3d ref = r.P * vh;
            Eigen::Vector3f gpu(o[12], o[13], o[14]);
            const double e = abs_rel_err(gpu.cast<double>().eval(), ref);
            worst_p = std::max(worst_p, e);
            if (e > kTolP) {
                if (bad < 3) std::fprintf(stderr,
                    "k=%zu Pv err=%g\n", k, e);
                ++bad;
            }
        }

        // 4. project3DPoint
        {
            Eigen::Vector4d vh(r.v3(0), r.v3(1), r.v3(2), 1.0);
            Eigen::Vector3d pv = r.P * vh;
            Eigen::Vector2d ref(pv(0) / pv(2), pv(1) / pv(2));
            Eigen::Vector2f gpu(o[15], o[16]);
            const double e = abs_rel_err(gpu.cast<double>().eval(), ref);
            worst_proj = std::max(worst_proj, e);
            if (e > kTolProj) {
                if (bad < 3) std::fprintf(stderr,
                    "k=%zu proj err=%g\n", k, e);
                ++bad;
            }
        }

        // 5. outer(v3, u3) — column-major flat
        {
            Eigen::Matrix3d ref = r.v3 * r.u3.transpose();
            Eigen::Matrix3f gpu;
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    gpu(i, j) = o[17 + j * 3 + i];
            const double e = abs_rel_err(gpu.cast<double>().eval(), ref);
            worst_outer = std::max(worst_outer, e);
            if (e > kTolOuter) {
                if (bad < 3) std::fprintf(stderr,
                    "k=%zu outer err=%g\n", k, e);
                ++bad;
            }
        }

        // 6. lineLineIntersect — reference via the same Bourke formula.
        {
            Eigen::Vector3d p13 = r.p1 - r.p3;
            Eigen::Vector3d p43 = r.p4 - r.p3;
            Eigen::Vector3d p21 = r.p2 - r.p1;

            double d1343 = p13.dot(p43);
            double d4321 = p43.dot(p21);
            double d1321 = p13.dot(p21);
            double d4343 = p43.dot(p43);
            double d2121 = p21.dot(p21);

            double denom = d2121 * d4343 - d4321 * d4321;
            double numer = d1343 * d4321 - d1321 * d4343;

            double mua = numer / denom;
            double mub = (d1343 + d4321 * mua) / d4343;

            Eigen::Vector3d pa = r.p1 + p21 * mua;
            Eigen::Vector3d pb = r.p3 + p43 * mub;
            Eigen::Vector3d mid_ref = (pa + pb) * 0.5;

            Eigen::Vector3f mid_gpu (o[26], o[27], o[28]);
            Eigen::Vector3f lli1_gpu(o[29], o[30], o[31]);
            Eigen::Vector3f lli2_gpu(o[32], o[33], o[34]);

            const double e_mid  = abs_rel_err(mid_gpu .cast<double>().eval(), mid_ref);
            const double e_lli1 = abs_rel_err(lli1_gpu.cast<double>().eval(), pa);
            const double e_lli2 = abs_rel_err(lli2_gpu.cast<double>().eval(), pb);
            const double e = std::max({ e_mid, e_lli1, e_lli2 });
            worst_lli = std::max(worst_lli, e);
            if (e > kTolLLI) {
                if (bad < 3) std::fprintf(stderr,
                    "k=%zu lli err mid=%g lli1=%g lli2=%g\n",
                    k, e_mid, e_lli1, e_lli2);
                ++bad;
            }
        }
    }

    std::printf("[info] cases             : %zu\n", kCases);
    std::printf("[info] worst err M3x3*v  : %.3g (budget %.3g)\n", worst_mv,    kTolMV);
    std::printf("[info] worst err M3x3*M  : %.3g (budget %.3g)\n", worst_mm,    kTolMM);
    std::printf("[info] worst err M3x4*v  : %.3g (budget %.3g)\n", worst_p,     kTolP);
    std::printf("[info] worst err project : %.3g (budget %.3g)\n", worst_proj,  kTolProj);
    std::printf("[info] worst err outer   : %.3g (budget %.3g)\n", worst_outer, kTolOuter);
    std::printf("[info] worst err lineLine: %.3g (budget %.3g)\n", worst_lli,   kTolLLI);

    if (bad) {
        std::fprintf(stderr, "FAIL: %d cases out of tolerance\n", bad);
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
