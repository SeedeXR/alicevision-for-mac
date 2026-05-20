// test_patch.cpp — numerical validation of the Metal port of
// depthMap/cuda/device/Patch.cuh's geometry helpers against an
// Eigen FP64 reference.
//
// Helpers covered this session:
//   * triangulateMatchRef
//   * computeRotCSEpip
//   * computeHomography
//   * refineDepthSubPixel
//
// The texture-coupled compNCCby3DptsYK kernels are not in this
// test — they need color.cuh (CostYKfromLab) and SimStat.cuh
// (simStat) ports first.
//
// Setup: construct a pair of randomly-posed cameras with identical
// pinhole intrinsics, generate 3D points in front of both cameras,
// project them into each view to get the (refpix, tarpix) pairs,
// then verify the kernel reconstructs the original 3D points and
// the geometry helpers behave as expected.

#include "av/depth_map/PatchOps.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using av::depth_map::DeviceCameraParams;
using av::depth_map::PatchOps;

constexpr std::size_t kCases = 1024;

// Tolerances. Mixed abs/rel error |g - r| / max(|r|, 1):
constexpr double kTolTri   = 5e-3;   // back-projection chain + lineLine div
constexpr double kTolBasis = 1e-4;   // cross products + 3 normalizations
constexpr double kTolHomog = 1e-3;   // K·(R-tn/d)·iK chain (many mat-muls + div)
constexpr double kTolDepth = 1e-5;   // single rational formula

// Pack an Eigen column-major matrix to a float[N] flat buffer.
template <int R, int C>
void pack(const Eigen::Matrix<double, R, C>& m, float* out) {
    for (int j = 0; j < C; ++j)
        for (int i = 0; i < R; ++i)
            out[j * R + i] = static_cast<float>(m(i, j));
}

void pack_vec3(const Eigen::Vector3d& v, float* out) {
    out[0] = static_cast<float>(v.x());
    out[1] = static_cast<float>(v.y());
    out[2] = static_cast<float>(v.z());
}

// Build a DeviceCameraParams from intrinsic K (3x3), rotation R
// (3x3, world→cam), and camera center C (world). Fills all derived
// members consistently with how upstream constructs them.
DeviceCameraParams make_cam(const Eigen::Matrix3d& K,
                            const Eigen::Matrix3d& R,
                            const Eigen::Vector3d& C)
{
    // P = K * [R | -R*C]  (3x4, world → image-homogeneous)
    Eigen::Matrix<double, 3, 4> P;
    P.block<3, 3>(0, 0) = R;
    P.block<3, 1>(0, 3) = -R * C;
    P = K * P;

    // iP = R^T * K^-1  (3x3 ray-direction inverse projection)
    const Eigen::Matrix3d iK_d = K.inverse();
    const Eigen::Matrix3d iP_d = R.transpose() * iK_d;

    DeviceCameraParams cp{};
    pack<3, 4>(P,   cp.P);
    pack<3, 3>(iP_d, cp.iP);
    pack<3, 3>(R,    cp.R);
    pack<3, 3>(R.transpose(), cp.iR);
    pack<3, 3>(K,    cp.K);
    pack<3, 3>(iK_d, cp.iK);
    pack_vec3(C, cp.C);

    // Camera axes in world frame are rows of R^T (= columns of R).
    pack_vec3(R.row(0).transpose(), cp.XVect);
    pack_vec3(R.row(1).transpose(), cp.YVect);
    pack_vec3(R.row(2).transpose(), cp.ZVect);

    return cp;
}

// Reconstruct an Eigen P (3x4) from a DeviceCameraParams (column-major).
Eigen::Matrix<double, 3, 4> P_of(const DeviceCameraParams& cp) {
    Eigen::Matrix<double, 3, 4> P;
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 3; ++i)
            P(i, j) = static_cast<double>(cp.P[j * 3 + i]);
    return P;
}

Eigen::Vector2d project(const Eigen::Matrix<double, 3, 4>& P,
                        const Eigen::Vector3d& X)
{
    Eigen::Vector4d Xh; Xh << X, 1.0;
    Eigen::Vector3d ph = P * Xh;
    return { ph(0) / ph(2), ph(1) / ph(2) };
}

template <class V>
double abs_rel_err(const V& gpu, const V& ref) {
    double worst = 0.0;
    for (Eigen::Index i = 0; i < gpu.size(); ++i) {
        const double r = ref(i);
        const double g = static_cast<double>(gpu(i));
        const double scale = std::max(std::abs(r), 1.0);
        worst = std::max(worst, std::abs(g - r) / scale);
    }
    return worst;
}

}  // namespace

int main() try {
    auto dev = av::gpu::Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    // -------- camera pair --------
    Eigen::Matrix3d K;
    K << 800.0,   0.0, 320.0,
           0.0, 800.0, 240.0,
           0.0,   0.0,   1.0;

    Eigen::Matrix3d Rrc = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Crc(0.0, 0.0, 0.0);

    // Target camera: small rotation around y plus a baseline along x.
    Eigen::AngleAxisd aa(0.10, Eigen::Vector3d::UnitY());
    Eigen::Matrix3d Rtc = aa.toRotationMatrix();
    Eigen::Vector3d Ctc(0.5, 0.0, 0.0);

    const DeviceCameraParams rc = make_cam(K, Rrc, Crc);
    const DeviceCameraParams tc = make_cam(K, Rtc, Ctc);
    const Eigen::Matrix<double, 3, 4> Pref = P_of(rc);
    const Eigen::Matrix<double, 3, 4> Ptar = P_of(tc);

    PatchOps ops(dev, rc, tc);

    // -------- generate test cases --------
    std::vector<float> in_buf (kCases * PatchOps::kInPerCase);
    std::vector<float> out_buf(kCases * PatchOps::kOutPerCase);

    std::mt19937_64 rng(0x600d5);
    std::uniform_real_distribution<double> Z(2.0, 10.0);
    std::uniform_real_distribution<double> XY(-2.0, 2.0);
    std::normal_distribution<double>       N(0.0, 1.0);

    // Reference data per case (kept on the CPU for cross-checks).
    std::vector<Eigen::Vector3d> X_ref(kCases);
    std::vector<Eigen::Vector3d> N_ref(kCases);
    std::vector<Eigen::Vector2d> ref_pix_v(kCases);
    std::vector<Eigen::Vector2d> tar_pix_v(kCases);
    std::vector<Eigen::Vector3d> depths_ref(kCases);
    std::vector<Eigen::Vector3d> sims_ref(kCases);

    for (std::size_t k = 0; k < kCases; ++k) {
        const Eigen::Vector3d X(XY(rng), XY(rng), Z(rng));
        const Eigen::Vector3d n_world = Eigen::Vector3d(N(rng), N(rng), N(rng)).normalized();
        const Eigen::Vector2d rp = project(Pref, X);
        const Eigen::Vector2d tp = project(Ptar, X);

        X_ref[k]      = X;
        N_ref[k]      = n_world;
        ref_pix_v[k]  = rp;
        tar_pix_v[k]  = tp;

        // Quadratic minimum: f(x) = a*x^2 + b*x + c with minimum
        // at xs and value sims_min. We pick xs, sims_min, then
        // sample at -1, 0, +1.
        const double xs       = 0.3 * N(rng);             // sub-pixel shift
        const double sims_min = -0.9;                     // best similarity
        const double curv     = 0.05;                     // positive curvature
        auto f = [&](double x) {
            return sims_min + curv * (x - xs) * (x - xs);
        };
        const Eigen::Vector3d depths(9.0, 10.0, 11.0);    // synthetic depths
        const Eigen::Vector3d sims(f(-1.0), f(0.0), f(+1.0));
        depths_ref[k] = depths;
        sims_ref[k]   = sims;

        // Pack inputs.
        float* row = in_buf.data() + k * PatchOps::kInPerCase;
        row[0]  = static_cast<float>(X.x());
        row[1]  = static_cast<float>(X.y());
        row[2]  = static_cast<float>(X.z());
        row[3]  = static_cast<float>(n_world.x());
        row[4]  = static_cast<float>(n_world.y());
        row[5]  = static_cast<float>(n_world.z());
        row[6]  = static_cast<float>(rp.x());
        row[7]  = static_cast<float>(rp.y());
        row[8]  = static_cast<float>(tp.x());
        row[9]  = static_cast<float>(tp.y());
        row[10] = static_cast<float>(depths.x());
        row[11] = static_cast<float>(depths.y());
        row[12] = static_cast<float>(depths.z());
        row[13] = static_cast<float>(sims.x());
        row[14] = static_cast<float>(sims.y());
        row[15] = static_cast<float>(sims.z());
    }

    // -------- dispatch --------
    ops.validate(in_buf, out_buf);

    // -------- compare --------
    double worst_tri   = 0.0;
    double worst_basis = 0.0;
    double worst_homog = 0.0;
    double worst_depth = 0.0;
    int bad = 0;

    for (std::size_t k = 0; k < kCases; ++k) {
        const float* o = out_buf.data() + k * PatchOps::kOutPerCase;

        // 1. triangulateMatchRef should recover X within FP32 noise
        //    of the back-projection chain. Reference: the same
        //    midpoint formula in FP64 — which equals X up to
        //    numerical noise because the cameras observe X exactly.
        {
            const Eigen::Vector3d ref = X_ref[k];
            Eigen::Vector3f g_f(o[0], o[1], o[2]);
            const double e = abs_rel_err(g_f.cast<double>().eval(), ref);
            worst_tri = std::max(worst_tri, e);
            if (e > kTolTri) {
                if (bad < 3) std::fprintf(stderr,
                    "k=%zu tri: gpu=(%g,%g,%g) ref=(%g,%g,%g) err=%g\n", k,
                    static_cast<double>(g_f.x()), static_cast<double>(g_f.y()),
                    static_cast<double>(g_f.z()),
                    ref.x(), ref.y(), ref.z(), e);
                ++bad;
            }
        }

        // 2. computeRotCSEpip — reference: y = (vrc × vtc)/||·||,
        //    n = ((vrc + vtc)/2) / ||·||, x = (y × n) / ||·||,
        //    where vrc = (Crc - p) / ||·||, vtc = (Ctc - p) / ||·||.
        {
            const Eigen::Vector3d p = X_ref[k];
            const Eigen::Vector3d Crc_v(static_cast<double>(rc.C[0]),
                                        static_cast<double>(rc.C[1]),
                                        static_cast<double>(rc.C[2]));
            const Eigen::Vector3d Ctc_v(static_cast<double>(tc.C[0]),
                                        static_cast<double>(tc.C[1]),
                                        static_cast<double>(tc.C[2]));
            const Eigen::Vector3d v1 = (Crc_v - p).normalized();
            const Eigen::Vector3d v2 = (Ctc_v - p).normalized();
            const Eigen::Vector3d y_ref = v1.cross(v2).normalized();
            const Eigen::Vector3d n_ref = ((v1 + v2) * 0.5).normalized();
            const Eigen::Vector3d x_ref = y_ref.cross(n_ref).normalized();

            Eigen::Vector3f y_gpu(o[3], o[4], o[5]);
            Eigen::Vector3f n_gpu(o[6], o[7], o[8]);
            Eigen::Vector3f x_gpu(o[9], o[10], o[11]);

            // The basis is defined up to sign on `y` (the cross
            // product orientation depends on the order of v1, v2).
            // Allow either sign by flipping if cosine is negative.
            auto sign_align = [](Eigen::Vector3f& g, const Eigen::Vector3d& r) {
                if (g.cast<double>().dot(r) < 0) g = -g;
            };
            sign_align(y_gpu, y_ref);
            sign_align(n_gpu, n_ref);
            sign_align(x_gpu, x_ref);

            const double ey = abs_rel_err(y_gpu.cast<double>().eval(), y_ref);
            const double en = abs_rel_err(n_gpu.cast<double>().eval(), n_ref);
            const double ex = abs_rel_err(x_gpu.cast<double>().eval(), x_ref);
            const double e  = std::max({ ey, en, ex });
            worst_basis = std::max(worst_basis, e);
            if (e > kTolBasis) {
                if (bad < 3) std::fprintf(stderr,
                    "k=%zu epip basis err: y=%g n=%g x=%g\n", k, ey, en, ex);
                ++bad;
            }
        }

        // 3. computeHomography — mirror upstream's formula exactly.
        //    Note: upstream's "tr" = R_t · (C_r - C_t), expressed
        //    in the target-cam frame. This is NOT the H-Z standard
        //    "t = R*C_r − C_t". We replicate upstream's path so the
        //    test validates kernel correctness against the upstream
        //    behavior, not against H-Z's canonical form.
        {
            const Eigen::Vector3d p_world = X_ref[k];
            const Eigen::Vector3d n_world = N_ref[k];
            const Eigen::Matrix3d Kd  = K;
            const Eigen::Matrix3d iKd = Kd.inverse();

            const Eigen::Vector3d _tl   = -Rrc * Crc;
            const Eigen::Vector3d _tr   = -Rtc * Ctc;
            const Eigen::Matrix3d tmpRr = Rtc * Rrc.transpose();
            const Eigen::Vector3d tr    = _tr - tmpRr * _tl;

            Eigen::Vector3d n_cam = Rrc * n_world;
            n_cam.normalize();
            const Eigen::Vector3d p_cam = Rrc * (p_world - Crc);
            const double d_ref = -n_cam.dot(p_cam);

            const Eigen::Matrix3d H_ref =
                Kd * (tmpRr - (tr * n_cam.transpose()) / d_ref) * iKd;

            Eigen::Matrix3f H_gpu;
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    H_gpu(i, j) = o[12 + j * 3 + i];

            // Homographies are defined up to scale; compare after
            // normalizing both to unit Frobenius norm (and aligning
            // signs).
            Eigen::Matrix3d H_gpu_d = H_gpu.cast<double>();
            const double s_g = H_gpu_d.norm();
            const double s_r = H_ref.norm();
            if (s_g > 1e-12 && s_r > 1e-12) {
                H_gpu_d /= s_g;
                Eigen::Matrix3d H_ref_n = H_ref / s_r;
                if (H_gpu_d.cwiseProduct(H_ref_n).sum() < 0)
                    H_gpu_d = -H_gpu_d;
                const double e = abs_rel_err(H_gpu_d, H_ref_n);
                worst_homog = std::max(worst_homog, e);
                if (e > kTolHomog) {
                    if (bad < 3) std::fprintf(stderr,
                        "k=%zu homog err=%g\n", k, e);
                    ++bad;
                }
            }
        }

        // 4. refineDepthSubPixel — verify formula matches the
        //    CPU-side reproduction.
        {
            float simM1 = static_cast<float>(sims_ref[k].x());
            float sim   = static_cast<float>(sims_ref[k].y());
            float simP1 = static_cast<float>(sims_ref[k].z());
            simM1 = (simM1 + 1.0f) * 0.5f;
            sim   = (sim   + 1.0f) * 0.5f;
            simP1 = (simP1 + 1.0f) * 0.5f;
            float refined;
            if (simM1 < sim || simP1 < sim) {
                refined = static_cast<float>(depths_ref[k].y());
            } else {
                const float dispStep = -((simP1 - simM1) /
                                         (2.0f * (simP1 + simM1 - 2.0f * sim)));
                const float floatDepthM1 = static_cast<float>(depths_ref[k].x());
                const float floatDepthP1 = static_cast<float>(depths_ref[k].z());
                const float b = (floatDepthP1 + floatDepthM1) * 0.5f;
                const float a = b - floatDepthM1;
                refined = a * dispStep + b;
                if (!std::isfinite(refined) || refined <= 0.0f)
                    refined = static_cast<float>(depths_ref[k].y());
            }
            const float gpu = o[21];
            const double e = std::abs(static_cast<double>(gpu - refined)) /
                             std::max<double>(std::abs(static_cast<double>(refined)),
                                              1.0);
            worst_depth = std::max(worst_depth, e);
            if (e > kTolDepth) {
                if (bad < 3) std::fprintf(stderr,
                    "k=%zu depth: gpu=%g ref=%g err=%g\n", k,
                    static_cast<double>(gpu),
                    static_cast<double>(refined), e);
                ++bad;
            }
        }
    }

    std::printf("[info] cases             : %zu\n", kCases);
    std::printf("[info] worst err triang  : %.3g (budget %.3g)\n", worst_tri,   kTolTri);
    std::printf("[info] worst err epip    : %.3g (budget %.3g)\n", worst_basis, kTolBasis);
    std::printf("[info] worst err homog   : %.3g (budget %.3g)\n", worst_homog, kTolHomog);
    std::printf("[info] worst err refine  : %.3g (budget %.3g)\n", worst_depth, kTolDepth);

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
