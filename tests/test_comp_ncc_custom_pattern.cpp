// test_comp_ncc_custom_pattern.cpp — end-to-end validation of
// compNCCby3DptsYK_customPatchPattern<TInvertAndFilter>.
//
// Sibling of test_comp_ncc.cpp. Same R+T camera setup and image
// content, but instead of a fixed (2*wsh+1)^2 block we drive the
// spatial sampling from a small `DevicePatchPattern` containing
// multiple subparts:
//   - subpart 0: a 5x5 "full" block at mipmap level 0, downscale 1.
//   - subpart 1: a 3x3 "full" block at mipmap level 0, downscale 2.
//   - subpart 2: a circle (8 evenly-spaced unit-radius samples) at
//                mipmap level 0.
//
// Per-subpart NCCs are folded by `weight`, exactly like upstream's
// CUDA implementation. The CPU FP64 reference re-runs the same
// arithmetic (bilinear sampling at mip 0 — we keep `mipmapLevel` at
// 0 and all `subpart.level` at 0, so the reference doesn't need a
// real CPU mipmap pyramid).
//
// Tolerance budget: per-axis.
//
//   no_filter : 5e-2 absolute — matches test_comp_ncc.cpp's
//               budget. Each subpart is a weighted-NCC of fewer
//               than 81 samples, so per-subpart FP32 drift is
//               <= the single-NCC drift. The weighted-mean fold
//               cannot amplify it beyond 1×.
//
//   filter    : 8e-2 absolute — the sigmoid invert-and-filter
//               step amplifies per-subpart NCC drift near its
//               inflection (sigMid = -0.7, sigwidth = 0.7) by a
//               factor of up to 10 / sigwidth × (endVal-zeroVal)
//               ≈ 3.57 at the steepest slope. With three subparts
//               each contributing ~2e-2 of pre-sigmoid drift,
//               the post-fold sum can reach ~3 × 3.57 × 2e-2
//               ≈ 7e-2. Empirically (64 cases, this build):
//               5.24e-2; 8e-2 absorbs the headroom without
//               admitting actually-broken kernels.

#include "av/depth_map/CompNCC.hpp"
#include "av/depth_map/DevicePatchPattern.hpp"
#include "av/depth_map/PatchOps.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using av::depth_map::DeviceCameraParams;
using av::depth_map::DevicePatchPattern;
using av::depth_map::DevicePatchPatternSubpart;
using av::depth_map::PatchCase;
using av::depth_map::CompNCCParams;

constexpr std::uint32_t kImgW = 256;
constexpr std::uint32_t kImgH = 256;
constexpr std::size_t   kCases = 64;

// See the header comment for the per-axis budget reasoning.
constexpr double kTolSimNoFilter = 5e-2;
constexpr double kTolSimFilter   = 8e-2;

constexpr float kRcMinAlpha = 255.0f * 0.9f;
constexpr float kTcMinAlpha = 255.0f * 0.4f;

// ---------------- camera utilities (mirrored from test_comp_ncc.cpp) ---

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

DeviceCameraParams make_cam(const Eigen::Matrix3d& K,
                            const Eigen::Matrix3d& R,
                            const Eigen::Vector3d& C)
{
    Eigen::Matrix<double, 3, 4> P;
    P.block<3, 3>(0, 0) = R;
    P.block<3, 1>(0, 3) = -R * C;
    P = K * P;

    const Eigen::Matrix3d iK_d = K.inverse();
    const Eigen::Matrix3d iP_d = R.transpose() * iK_d;

    DeviceCameraParams cp{};
    pack<3, 4>(P, cp.P);
    pack<3, 3>(iP_d, cp.iP);
    pack<3, 3>(R, cp.R);
    pack<3, 3>(R.transpose(), cp.iR);
    pack<3, 3>(K, cp.K);
    pack<3, 3>(iK_d, cp.iK);
    pack_vec3(C, cp.C);
    pack_vec3(R.row(0).transpose(), cp.XVect);
    pack_vec3(R.row(1).transpose(), cp.YVect);
    pack_vec3(R.row(2).transpose(), cp.ZVect);
    return cp;
}

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

// ---------------- synthetic image (same as test_comp_ncc.cpp) ------

struct ImageF4 { float r, g, b, a; };

ImageF4 make_pixel(std::uint32_t i, std::uint32_t j) {
    const float u = static_cast<float>(i) / static_cast<float>(kImgW);
    const float v = static_cast<float>(j) / static_cast<float>(kImgH);
    const float intensity =
        128.0f + 80.0f * std::sin(8.0f * u + 0.3f * v)
                + 40.0f * std::cos(3.0f * v - 0.7f * u);
    return ImageF4{ intensity,
                    127.5f + 60.0f * u,
                    127.5f + 60.0f * v,
                    255.0f };
}

std::vector<float> make_image() {
    std::vector<float> px(kImgW * kImgH * 4);
    for (std::uint32_t j = 0; j < kImgH; ++j) {
        for (std::uint32_t i = 0; i < kImgW; ++i) {
            const ImageF4 c = make_pixel(i, j);
            const std::size_t k = (j * kImgW + i) * 4;
            px[k + 0] = c.r;
            px[k + 1] = c.g;
            px[k + 2] = c.b;
            px[k + 3] = c.a;
        }
    }
    return px;
}

Eigen::Vector4d bilin_sample(const std::vector<float>& pixels,
                             std::uint32_t W, std::uint32_t H,
                             double px, double py)
{
    const double cx = px - 0.5;
    const double cy = py - 0.5;
    const int ix0_raw = static_cast<int>(std::floor(cx));
    const int iy0_raw = static_cast<int>(std::floor(cy));
    const double fx = cx - ix0_raw;
    const double fy = cy - iy0_raw;

    auto clamp_xy = [&](int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    };
    const int W_i = static_cast<int>(W);
    const int H_i = static_cast<int>(H);
    const int ix0 = clamp_xy(ix0_raw,     0, W_i - 1);
    const int ix1 = clamp_xy(ix0_raw + 1, 0, W_i - 1);
    const int iy0 = clamp_xy(iy0_raw,     0, H_i - 1);
    const int iy1 = clamp_xy(iy0_raw + 1, 0, H_i - 1);

    auto load = [&](int x, int y) {
        const std::size_t k = (static_cast<std::size_t>(y) * W + x) * 4;
        return Eigen::Vector4d(static_cast<double>(pixels[k + 0]),
                               static_cast<double>(pixels[k + 1]),
                               static_cast<double>(pixels[k + 2]),
                               static_cast<double>(pixels[k + 3]));
    };
    const Eigen::Vector4d a = load(ix0, iy0);
    const Eigen::Vector4d b = load(ix1, iy0);
    const Eigen::Vector4d c = load(ix0, iy1);
    const Eigen::Vector4d d = load(ix1, iy1);

    const Eigen::Vector4d ab = a + (b - a) * fx;
    const Eigen::Vector4d cd = c + (d - c) * fx;
    return ab + (cd - ab) * fy;
}

// 6-arg cost (color + spatial proximity), matching CostYKfromLab(int,int,...)
double cost_yk_full(int dx, int dy,
                    const Eigen::Vector4d& c1, const Eigen::Vector4d& c2,
                    double invGammaC, double invGammaP)
{
    const Eigen::Vector3d a = c1.head<3>();
    const Eigen::Vector3d b = c2.head<3>();
    const double deltaC = (a - b).norm() * invGammaC;
    const double deltaP = std::sqrt(double(dx * dx + dy * dy)) * invGammaP;
    return std::exp(-(deltaC + deltaP));
}

// 3-arg cost (color only), matching CostYKfromLab(float4,float4,float)
double cost_yk_color(const Eigen::Vector4d& c1, const Eigen::Vector4d& c2,
                     double invGammaC)
{
    const Eigen::Vector3d a = c1.head<3>();
    const Eigen::Vector3d b = c2.head<3>();
    const double deltaC = (a - b).norm() * invGammaC;
    return std::exp(-deltaC);
}

// Sigmoid mirror of matrix.h::sigmoid (FP64 reference).
double sigmoid_ref(double zeroVal, double endVal,
                   double sigwidth, double sigMid, double xval)
{
    return zeroVal + (endVal - zeroVal)
                   * (1.0 / (1.0 + std::exp(10.0 * ((xval - sigMid) / sigwidth))));
}

// CPU reference for compNCCby3DptsYK_customPatchPattern<TInvertAndFilter>.
// Mirrors the MSL kernel in Patch.h exactly (mipmapLevel == 0 path,
// no consistent-scale; this is what the test exercises).
//
// Returns INFINITY in the same cases as the kernel.
double comp_ncc_custom_ref(const std::vector<float>& rc_img,
                           const std::vector<float>& tc_img,
                           const Eigen::Matrix<double, 3, 4>& Pref,
                           const Eigen::Matrix<double, 3, 4>& Ptar,
                           const PatchCase& patch,
                           const DevicePatchPattern& pattern,
                           double invGammaC, double invGammaP,
                           bool   invertAndFilter)
{
    const Eigen::Vector3d p_world(static_cast<double>(patch.p[0]),
                                  static_cast<double>(patch.p[1]),
                                  static_cast<double>(patch.p[2]));
    const Eigen::Vector3d ax(static_cast<double>(patch.x[0]),
                             static_cast<double>(patch.x[1]),
                             static_cast<double>(patch.x[2]));
    const Eigen::Vector3d ay(static_cast<double>(patch.y[0]),
                             static_cast<double>(patch.y[1]),
                             static_cast<double>(patch.y[2]));
    const double d = static_cast<double>(patch.d);

    const Eigen::Vector2d rp = project(Pref, p_world);
    const Eigen::Vector2d tp = project(Ptar, p_world);

    // Kernel margin is a fixed 2.0 in this variant.
    constexpr double margin = 2.0;
    auto out_of_bounds = [&](double x, double y, std::uint32_t W, std::uint32_t H) {
        return x < margin || x > double(W - 1) - margin ||
               y < margin || y > double(H - 1) - margin;
    };
    if (out_of_bounds(rp.x(), rp.y(), kImgW, kImgH) ||
        out_of_bounds(tp.x(), tp.y(), kImgW, kImgH)) {
        return std::numeric_limits<double>::infinity();
    }

    const Eigen::Vector4d rcCenterBase = bilin_sample(rc_img, kImgW, kImgH,
                                                       rp.x() + 0.5,
                                                       rp.y() + 0.5);
    const Eigen::Vector4d tcCenterBase = bilin_sample(tc_img, kImgW, kImgH,
                                                       tp.x() + 0.5,
                                                       tp.y() + 0.5);
    if (rcCenterBase(3) < static_cast<double>(kRcMinAlpha) ||
        tcCenterBase(3) < static_cast<double>(kTcMinAlpha)) {
        return std::numeric_limits<double>::infinity();
    }

    double fsim = 0.0;
    double wsum = 0.0;

    for (int s = 0; s < pattern.nbSubparts; ++s) {
        const auto& subpart = pattern.subparts[s];

        // Test setup keeps every subpart.level at 0 so the center
        // color (from the base mip) is fine to reuse.
        const Eigen::Vector4d rcCenter = rcCenterBase;
        const Eigen::Vector4d tcCenter = tcCenterBase;

        double xsum = 0, ysum = 0, xxsum = 0, yysum = 0, xysum = 0;
        double wsumSub = 0;

        if (subpart.isCircle != 0) {
            for (int c = 0; c < subpart.nbCoordinates; ++c) {
                const double rx = static_cast<double>(subpart.coordinates[c][0]);
                const double ry = static_cast<double>(subpart.coordinates[c][1]);
                const Eigen::Vector3d pp = p_world + ax * (d * rx) + ay * (d * ry);
                const Eigen::Vector2d rpc = project(Pref, pp);
                const Eigen::Vector2d tpc = project(Ptar, pp);
                const Eigen::Vector4d rcCol = bilin_sample(rc_img, kImgW, kImgH,
                                                            rpc.x() + 0.5,
                                                            rpc.y() + 0.5);
                const Eigen::Vector4d tcCol = bilin_sample(tc_img, kImgW, kImgH,
                                                            tpc.x() + 0.5,
                                                            tpc.y() + 0.5);
                const double w = cost_yk_color(rcCenter, rcCol, invGammaC)
                               * cost_yk_color(tcCenter, tcCol, invGammaC);
                const double X = rcCol(0);
                const double Y = tcCol(0);
                wsumSub += w;
                xsum  += w * X;
                ysum  += w * Y;
                xxsum += w * X * X;
                yysum += w * Y * Y;
                xysum += w * X * Y;
            }
        } else {
            const double ds = static_cast<double>(subpart.downscale);
            const int wsh = subpart.wsh;
            for (int yp = -wsh; yp <= wsh; ++yp) {
                for (int xp = -wsh; xp <= wsh; ++xp) {
                    const Eigen::Vector3d pp = p_world
                        + ax * (d * double(xp) * ds)
                        + ay * (d * double(yp) * ds);
                    const Eigen::Vector2d rpc = project(Pref, pp);
                    const Eigen::Vector2d tpc = project(Ptar, pp);
                    const Eigen::Vector4d rcCol = bilin_sample(rc_img, kImgW, kImgH,
                                                                rpc.x() + 0.5,
                                                                rpc.y() + 0.5);
                    const Eigen::Vector4d tcCol = bilin_sample(tc_img, kImgW, kImgH,
                                                                tpc.x() + 0.5,
                                                                tpc.y() + 0.5);
                    const double w = cost_yk_full(xp, yp, rcCenter, rcCol,
                                                  invGammaC, invGammaP)
                                   * cost_yk_full(xp, yp, tcCenter, tcCol,
                                                  invGammaC, invGammaP);
                    const double X = rcCol(0);
                    const double Y = tcCol(0);
                    wsumSub += w;
                    xsum  += w * X;
                    ysum  += w * Y;
                    xxsum += w * X * X;
                    yysum += w * Y * Y;
                    xysum += w * X * Y;
                }
            }
        }

        // computeWSim equivalent (FP64).
        const double varXW  = (xxsum - xsum * xsum / wsumSub) / wsumSub;
        const double varYW  = (yysum - ysum * ysum / wsumSub) / wsumSub;
        const double varXYW = (xysum - xsum * ysum / wsumSub) / wsumSub;
        const double rawSim = varXYW / std::sqrt(varXW * varYW);
        const double fsimSubpart = std::isfinite(rawSim) ? -rawSim : 1.0;

        if (fsimSubpart < 0.0) {
            if (invertAndFilter) {
                const double fsimInverted =
                    sigmoid_ref(0.0, 1.0, 0.7, -0.7, fsimSubpart);
                fsim += fsimInverted * static_cast<double>(subpart.weight);
            } else {
                fsim += fsimSubpart * static_cast<double>(subpart.weight);
            }
            wsum += static_cast<double>(subpart.weight);
        }
    }

    if (wsum == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    if (invertAndFilter) return fsim;
    return fsim / wsum;
}

// Build the test pattern: 3 subparts, varied modes / weights.
DevicePatchPattern build_test_pattern() {
    DevicePatchPattern pp{};
    pp.nbSubparts = 3;

    // Subpart 0: 5x5 full block, downscale 1, weight 0.5.
    {
        auto& s = pp.subparts[0];
        std::memset(&s, 0, sizeof(s));
        s.nbCoordinates = 0;
        s.level         = 0.0f;
        s.downscale     = 1.0f;
        s.weight        = 0.5f;
        s.isCircle      = 0;
        s.wsh           = 2;        // 5x5
    }
    // Subpart 1: 3x3 full block, downscale 2, weight 0.3.
    {
        auto& s = pp.subparts[1];
        std::memset(&s, 0, sizeof(s));
        s.nbCoordinates = 0;
        s.level         = 0.0f;
        s.downscale     = 2.0f;
        s.weight        = 0.3f;
        s.isCircle      = 0;
        s.wsh           = 1;        // 3x3
    }
    // Subpart 2: 8-point circle (unit radius), weight 0.2.
    {
        auto& s = pp.subparts[2];
        std::memset(&s, 0, sizeof(s));
        const int N = 8;
        s.nbCoordinates = N;
        s.level         = 0.0f;
        s.downscale     = 1.0f;     // unused for circle path
        s.weight        = 0.2f;
        s.isCircle      = 1;
        s.wsh           = 0;        // unused for circle path
        for (int k = 0; k < N; ++k) {
            const double th = (2.0 * M_PI * k) / static_cast<double>(N);
            s.coordinates[k][0] = static_cast<float>(std::cos(th));
            s.coordinates[k][1] = static_cast<float>(std::sin(th));
        }
    }
    return pp;
}

}  // namespace

int main() try {
    auto dev = av::gpu::Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    // ---------------- cameras (same as test_comp_ncc.cpp) ------------
    Eigen::Matrix3d K;
    K << 400.0,   0.0, double(kImgW) * 0.5,
           0.0, 400.0, double(kImgH) * 0.5,
           0.0,   0.0,   1.0;

    Eigen::Matrix3d Rrc = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Crc(0.0, 0.0, 0.0);
    Eigen::AngleAxisd aa(0.05, Eigen::Vector3d::UnitY());
    Eigen::Matrix3d Rtc = aa.toRotationMatrix();
    Eigen::Vector3d Ctc(0.3, 0.0, 0.0);

    const DeviceCameraParams rc_dcp = make_cam(K, Rrc, Crc);
    const DeviceCameraParams tc_dcp = make_cam(K, Rtc, Ctc);
    const auto Pref = P_of(rc_dcp);
    const auto Ptar = P_of(tc_dcp);

    // ---------------- textures ----------------
    const auto img = make_image();

    using namespace av::gpu;
    Texture rc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, /*mip_levels=auto*/ 0,
        PixelFormat::RGBA32Float });
    Texture tc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, /*mip_levels=auto*/ 0,
        PixelFormat::RGBA32Float });
    rc_tex.set_label("compncc_cp.rc");
    tc_tex.set_label("compncc_cp.tc");
    rc_tex.upload(std::span<const float>(img));
    tc_tex.upload(std::span<const float>(img));
    rc_tex.generate_mipmaps();
    tc_tex.generate_mipmaps();

    // ---------------- patches (same distribution as test_comp_ncc.cpp) --
    av::depth_map::CompNCC ncc(dev, rc_dcp, tc_dcp);

    std::vector<PatchCase> patches(kCases);
    std::mt19937_64 rng(0xc0ffeedface);
    std::uniform_real_distribution<double> XY(-1.5, 1.5);
    std::uniform_real_distribution<double> Z(3.5, 7.0);

    for (std::size_t k = 0; k < kCases; ++k) {
        const Eigen::Vector3d X(XY(rng), XY(rng), Z(rng));
        const Eigen::Vector3d n(0.0, 0.0, -1.0);
        const Eigen::Vector3d xa(1.0, 0.0, 0.0);
        const Eigen::Vector3d ya(0.0, 1.0, 0.0);
        const double d_world = X.z() / 400.0;

        PatchCase& pc = patches[k];
        std::memset(&pc, 0, sizeof(pc));
        pc.p[0] = float(X.x()); pc.p[1] = float(X.y()); pc.p[2] = float(X.z());
        pc.n[0] = float(n.x());  pc.n[1] = float(n.y());  pc.n[2] = float(n.z());
        pc.x[0] = float(xa.x()); pc.x[1] = float(xa.y()); pc.x[2] = float(xa.z());
        pc.y[0] = float(ya.x()); pc.y[1] = float(ya.y()); pc.y[2] = float(ya.z());
        pc.d    = float(d_world);
    }

    CompNCCParams params{};
    params.rcLevelWidth   = kImgW;
    params.rcLevelHeight  = kImgH;
    params.tcLevelWidth   = kImgW;
    params.tcLevelHeight  = kImgH;
    params.mipmapLevel    = 0.0f;
    params.wsh            = 0;          // ignored by the custom-pattern path
    params.invGammaC      = 1.0f / 20.0f;
    params.invGammaP      = 1.0f / 4.0f;
    params.useConsistentScale = 0;

    const DevicePatchPattern pattern = build_test_pattern();

    // ---------------- dispatch <false> ----------------
    std::vector<float> gpu_sim(kCases);
    ncc.run_no_filter_custom_pattern(patches, gpu_sim,
                                     rc_tex, tc_tex, params, pattern);

    // ---------------- dispatch <true> -----------------
    std::vector<float> gpu_sim_f(kCases);
    ncc.run_filter_custom_pattern(patches, gpu_sim_f,
                                  rc_tex, tc_tex, params, pattern);

    // ---------------- CPU reference -------------------
    auto evaluate_axis = [&](bool invertAndFilter,
                             const std::vector<float>& gpu,
                             const char* tag,
                             double tol,
                             std::size_t& n_inf_gpu_out,
                             std::size_t& n_inf_cpu_out,
                             std::size_t& n_match_out,
                             double& worst_err_out,
                             int&    bad_out)
    {
        std::size_t n_inf_gpu = 0, n_inf_cpu = 0, n_match = 0;
        double worst_finite_err = 0.0;
        int bad = 0;

        for (std::size_t k = 0; k < kCases; ++k) {
            const double ref = comp_ncc_custom_ref(
                img, img, Pref, Ptar, patches[k], pattern,
                double(params.invGammaC), double(params.invGammaP),
                invertAndFilter);
            const double g = static_cast<double>(gpu[k]);

            const bool gpu_inf = !std::isfinite(g);
            const bool cpu_inf = !std::isfinite(ref);
            if (gpu_inf) ++n_inf_gpu;
            if (cpu_inf) ++n_inf_cpu;

            if (gpu_inf != cpu_inf) {
                if (bad < 3) std::fprintf(stderr,
                    "[%s] k=%zu inf-mismatch: gpu=%g cpu=%g\n",
                    tag, k, g, ref);
                ++bad;
                continue;
            }
            if (gpu_inf) continue;

            const double err = std::abs(g - ref);
            worst_finite_err = std::max(worst_finite_err, err);
            ++n_match;
            if (err > tol) {
                if (bad < 3) std::fprintf(stderr,
                    "[%s] k=%zu sim err=%g gpu=%g ref=%g\n",
                    tag, k, err, g, ref);
                ++bad;
            }
        }

        std::printf("[info] %-19s: cases=%zu gpu_inf=%zu cpu_inf=%zu finite=%zu worst_err=%.3g (budget %.3g)\n",
                    tag, kCases, n_inf_gpu, n_inf_cpu, n_match,
                    worst_finite_err, tol);

        n_inf_gpu_out = n_inf_gpu;
        n_inf_cpu_out = n_inf_cpu;
        n_match_out   = n_match;
        worst_err_out = worst_finite_err;
        bad_out       = bad;
    };

    std::size_t n_inf_gpu_a, n_inf_cpu_a, n_match_a;
    double worst_a;
    int bad_a;
    evaluate_axis(false, gpu_sim,   "no_filter", kTolSimNoFilter,
                  n_inf_gpu_a, n_inf_cpu_a, n_match_a, worst_a, bad_a);

    std::size_t n_inf_gpu_b, n_inf_cpu_b, n_match_b;
    double worst_b;
    int bad_b;
    evaluate_axis(true,  gpu_sim_f, "filter",    kTolSimFilter,
                  n_inf_gpu_b, n_inf_cpu_b, n_match_b, worst_b, bad_b);

    // Sanity #1: at least 1/4 of the cases must produce a finite
    // similarity. Otherwise we are just testing the bounds path.
    if (n_match_a < kCases / 4 || n_match_b < kCases / 4) {
        std::fprintf(stderr,
            "FAIL: too few finite similarity scores (no_filter=%zu filter=%zu / %zu)\n",
            n_match_a, n_match_b, kCases);
        return 1;
    }

    // Sanity #2: with identical R and T images, the geometrically
    // correct unfiltered similarity should be strongly negative
    // (well below 0). For the filtered variant, the sigmoid maps
    // those scores into (0, 1) with the best matches near 1; check
    // that the median is comfortably positive.
    {
        std::vector<float> finite;
        finite.reserve(n_match_a);
        for (std::size_t k = 0; k < kCases; ++k) {
            if (std::isfinite(gpu_sim[k])) finite.push_back(gpu_sim[k]);
        }
        std::nth_element(finite.begin(),
                         finite.begin() + finite.size() / 2,
                         finite.end());
        const float median = finite[finite.size() / 2];
        std::printf("[info] median no_filter  : %.4f (expect well below 0)\n",
                    static_cast<double>(median));
        if (median > -0.4f) {
            std::fprintf(stderr,
                "FAIL: no_filter median similarity %.4f is not negative enough\n",
                static_cast<double>(median));
            return 1;
        }
    }
    {
        std::vector<float> finite;
        finite.reserve(n_match_b);
        for (std::size_t k = 0; k < kCases; ++k) {
            if (std::isfinite(gpu_sim_f[k])) finite.push_back(gpu_sim_f[k]);
        }
        std::nth_element(finite.begin(),
                         finite.begin() + finite.size() / 2,
                         finite.end());
        const float median = finite[finite.size() / 2];
        std::printf("[info] median filter     : %.4f (expect positive)\n",
                    static_cast<double>(median));
        if (median < 0.1f) {
            std::fprintf(stderr,
                "FAIL: filter median similarity %.4f is not positive enough\n",
                static_cast<double>(median));
            return 1;
        }
    }

    if (bad_a || bad_b) {
        std::fprintf(stderr,
            "FAIL: out-of-tolerance cases (no_filter=%d filter=%d)\n",
            bad_a, bad_b);
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
