// test_refine_pipeline.cpp — end-to-end Refine pipeline run on
// Apple Silicon. Wires `init_refine → refine_similarity →
// refine_best_depth` on the homography-warped 2-view scene from
// test_sgm_accuracy (T = Warp_H(R) for the plane at world Z=4).
//
// The SGM mid depth fed into Refine is set to the *per-pixel
// analytical truth* — i.e., we hand the Refine pass a correct
// initial guess. With that input:
//   * NCC<true> peaks at the middle Z slice (no sub-pixel offset
//     needed).
//   * refine_best_depth's Gaussian sweep should find the winner
//     near sub-sample offset 0.
//   * Output depth ≈ SGM mid ≈ truth (within a few sub-sample
//     widths).
//
// This validates *integration* of the three Refine kernels —
// data layouts, parameter passing, sub-sample geometry, FP16
// volume plumbing. Depth-recovery accuracy is bounded by the
// Refine sweep's sub-sample resolution
// (sgm_pix_size / samplesPerPixSize).

#include "av/depth_map/PatchOps.hpp"
#include "av/depth_map/Volume.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using av::depth_map::DeviceCameraParams;
using av::depth_map::Volume;
using av::depth_map::VolumeDims;

constexpr std::uint32_t kImgW   = 256;
constexpr std::uint32_t kImgH   = 192;
constexpr std::uint32_t kVolX   = 32;
constexpr std::uint32_t kVolY   = 24;
constexpr std::uint32_t kVolZ   = 9;            // halfNbDepths = 4
constexpr std::int32_t  kHalfNbDepths      = (int(kVolZ) - 1) / 2;
constexpr std::int32_t  kSamplesPerPixSize = 4;
constexpr std::int32_t  kHalfNbSamples     = 12;
constexpr float         kSigma             = 4.0f;
constexpr float         kTwoSigmaSq        = 2.0f * kSigma * kSigma;
constexpr int           kStep              = 8;
constexpr int           kWsh               = 3;

constexpr float kTruthZ = 4.0f;
constexpr float kFx = 400.0f;
constexpr float kFy = 400.0f;
constexpr float kCx = float(kImgW) * 0.5f;
constexpr float kCy = float(kImgH) * 0.5f;

// ---- camera utilities (mirrored from test_sgm_accuracy) ----

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

// ---- image, homography, warp (same recipe as test_sgm_accuracy) ----

std::vector<float> make_r_image() {
    std::vector<float> px(kImgW * kImgH * 4);
    for (std::uint32_t j = 0; j < kImgH; ++j)
        for (std::uint32_t i = 0; i < kImgW; ++i) {
            const float u = float(i) / float(kImgW);
            const float v = float(j) / float(kImgH);
            const float r = 128.0f
                + 60.0f * std::sin(15.0f * u + 2.0f * v)
                + 40.0f * std::cos(11.0f * v - 3.0f * u)
                + 25.0f * std::sin(23.0f * (u + 0.4f * v));
            const float g = 128.0f
                + 50.0f * std::cos(13.0f * v)
                + 30.0f * std::sin(19.0f * u);
            const float b = 128.0f
                + 45.0f * std::sin(17.0f * (u + v));
            const std::size_t k = (j * kImgW + i) * 4;
            px[k + 0] = std::clamp(r, 0.0f, 255.0f);
            px[k + 1] = std::clamp(g, 0.0f, 255.0f);
            px[k + 2] = std::clamp(b, 0.0f, 255.0f);
            px[k + 3] = 255.0f;
        }
    return px;
}

Eigen::Matrix3d plane_homography(
    const Eigen::Matrix3d& K,
    const Eigen::Matrix3d& Rrc, const Eigen::Vector3d& Crc,
    const Eigen::Matrix3d& Rtc, const Eigen::Vector3d& Ctc,
    const Eigen::Vector3d& plane_point_world,
    const Eigen::Vector3d& plane_normal_world_unit)
{
    const Eigen::Vector3d _tl   = -Rrc * Crc;
    const Eigen::Vector3d _tr   = -Rtc * Ctc;
    const Eigen::Matrix3d tmpRr = Rtc * Rrc.transpose();
    const Eigen::Vector3d tr    = _tr - tmpRr * _tl;
    Eigen::Vector3d n_cam = Rrc * plane_normal_world_unit;
    n_cam.normalize();
    const Eigen::Vector3d p_cam = Rrc * (plane_point_world - Crc);
    const double d_ref = -n_cam.dot(p_cam);
    return K * (tmpRr - (tr * n_cam.transpose()) / d_ref) * K.inverse();
}

struct V4 { float r, g, b, a; };
V4 bilin(const std::vector<float>& tex, double px, double py) {
    const double cx = px - 0.5, cy = py - 0.5;
    const int ix0 = int(std::floor(cx)), iy0 = int(std::floor(cy));
    const double fx = cx - ix0, fy = cy - iy0;
    auto cl = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };
    const int Wi = int(kImgW), Hi = int(kImgH);
    const int x0 = cl(ix0, 0, Wi-1), x1 = cl(ix0+1, 0, Wi-1);
    const int y0 = cl(iy0, 0, Hi-1), y1 = cl(iy0+1, 0, Hi-1);
    auto ld = [&](int x, int y) {
        const std::size_t k = (std::size_t(y) * kImgW + x) * 4;
        return V4{ tex[k], tex[k+1], tex[k+2], tex[k+3] };
    };
    V4 a = ld(x0, y0), b = ld(x1, y0), c = ld(x0, y1), d = ld(x1, y1);
    auto mix1 = [](float u, float v, double t) { return float(u + (v - u) * t); };
    V4 ab{ mix1(a.r,b.r,fx), mix1(a.g,b.g,fx), mix1(a.b,b.b,fx), mix1(a.a,b.a,fx) };
    V4 cd{ mix1(c.r,d.r,fx), mix1(c.g,d.g,fx), mix1(c.b,d.b,fx), mix1(c.a,d.a,fx) };
    return V4{ mix1(ab.r,cd.r,fy), mix1(ab.g,cd.g,fy), mix1(ab.b,cd.b,fy), mix1(ab.a,cd.a,fy) };
}

std::vector<float> warp(const std::vector<float>& r, const Eigen::Matrix3d& H_inv) {
    std::vector<float> t(kImgW * kImgH * 4);
    for (std::uint32_t j = 0; j < kImgH; ++j)
        for (std::uint32_t i = 0; i < kImgW; ++i) {
            const Eigen::Vector3d th(double(i) + 0.5, double(j) + 0.5, 1.0);
            const Eigen::Vector3d rh = H_inv * th;
            const V4 c = bilin(r, rh(0)/rh(2), rh(1)/rh(2));
            const std::size_t k = (j * kImgW + i) * 4;
            t[k] = c.r; t[k+1] = c.g; t[k+2] = c.b; t[k+3] = c.a;
        }
    return t;
}

// Analytical true euclidean depth at pixel (px, py) for plane Z = Z_plane.
double true_depth(double px, double py, double Z_plane) {
    const double X = (px - double(kCx)) * Z_plane / double(kFx);
    const double Y = (py - double(kCy)) * Z_plane / double(kFy);
    return std::sqrt(X * X + Y * Y + Z_plane * Z_plane);
}

}  // namespace

int main() try
{
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    static_assert(kVolX * kStep == kImgW, "geometry");
    static_assert(kVolY * kStep == kImgH, "geometry");

    // ---- cameras ----
    Eigen::Matrix3d K;
    K << kFx, 0.0, double(kCx),
         0.0, kFy, double(kCy),
         0.0, 0.0, 1.0;
    const Eigen::Matrix3d Rrc = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d Crc(0.0, 0.0, 0.0);
    const Eigen::AngleAxisd aa(0.03, Eigen::Vector3d::UnitY());
    const Eigen::Matrix3d Rtc = aa.toRotationMatrix();
    const Eigen::Vector3d Ctc(0.25, 0.0, 0.0);
    const DeviceCameraParams rc = make_cam(K, Rrc, Crc);
    const DeviceCameraParams tc = make_cam(K, Rtc, Ctc);

    // ---- plane homography & T = Warp_H(R) ----
    const Eigen::Vector3d plane_point (0.0, 0.0, double(kTruthZ));
    const Eigen::Vector3d plane_normal(0.0, 0.0, -1.0);
    const Eigen::Matrix3d H     = plane_homography(K, Rrc, Crc, Rtc, Ctc,
                                                    plane_point, plane_normal);
    const Eigen::Matrix3d H_inv = H.inverse();

    const auto r_img = make_r_image();
    const auto t_img = warp(r_img, H_inv);

    Texture rc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, 0, PixelFormat::RGBA32Float });
    Texture tc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, 0, PixelFormat::RGBA32Float });
    rc_tex.upload(std::span<const float>(r_img));
    tc_tex.upload(std::span<const float>(t_img));
    rc_tex.generate_mipmaps();
    tc_tex.generate_mipmaps();

    // ---- SGM depth-pix-size map: per-pixel analytical truth ----
    // The SGM mid depth at each pixel equals the analytical truth
    // for the plane Z = kTruthZ. The SGM pix-size is set so the
    // Refine sub-sample sweep covers a reasonable neighborhood
    // around the truth.
    constexpr double kSgmPixSize = double(kTruthZ) / double(kFx);  // ~ 0.01
    const std::size_t pix = std::size_t(kVolX) * std::size_t(kVolY);
    std::vector<float> sgm_dp(pix * 2);
    for (std::uint32_t vy = 0; vy < kVolY; ++vy)
        for (std::uint32_t vx = 0; vx < kVolX; ++vx) {
            const double cx = (double(vx) + 0.5) * double(kStep);
            const double cy = (double(vy) + 0.5) * double(kStep);
            const std::size_t k = std::size_t(vy) * kVolX + vx;
            sgm_dp[k * 2 + 0] = float(true_depth(cx, cy, double(kTruthZ)));
            sgm_dp[k * 2 + 1] = float(kSgmPixSize);
        }
    Buffer sgm_buf(dev, sgm_dp.size() * sizeof(float));
    sgm_buf.upload(std::span<const float>(sgm_dp));

    // ---- FP16 refinement volume ----
    const VolumeDims dims{ kVolX, kVolY, kVolZ };
    Buffer vol_buf(dev, dims.voxel_count() * sizeof(std::uint16_t));

    // ---- depth-sim output ----
    Buffer dsim_buf(dev, pix * 2 * sizeof(float));

    Volume vol(dev);

    // ===========================================================
    // 1. init_refine (zero the half volume)
    // ===========================================================
    vol.init_refine(vol_buf, dims, 0.0f);

    // ===========================================================
    // 2. refine_similarity (one R, T call — accumulate into half)
    // ===========================================================
    Volume::RefineSimilarityParams rs{};
    rs.dims                   = dims;
    rs.rc_refine_level_width  = kImgW;
    rs.rc_refine_level_height = kImgH;
    rs.tc_refine_level_width  = kImgW;
    rs.tc_refine_level_height = kImgH;
    rs.rc_mipmap_level        = 0.0f;
    rs.step_xy                = kStep;
    rs.wsh                    = kWsh;
    rs.inv_gamma_c            = 1.0f / 20.0f;
    rs.inv_gamma_p            = 1.0f / 4.0f;
    rs.use_consistent_scale   = 0;
    rs.depth_range_begin      = 0;
    rs.depth_range_end        = kVolZ;
    rs.roi_x_begin            = 0;
    rs.roi_y_begin            = 0;
    rs.roi_width              = kVolX;
    rs.roi_height             = kVolY;
    vol.refine_similarity(vol_buf, sgm_buf, rc_tex, tc_tex, rc, tc, rs);

    // ===========================================================
    // 3. refine_best_depth
    // ===========================================================
    Volume::RefineBestDepthParams rb{};
    rb.dims                    = dims;
    rb.samples_per_pix_size    = kSamplesPerPixSize;
    rb.half_nb_samples         = kHalfNbSamples;
    rb.half_nb_depths          = kHalfNbDepths;
    rb.two_times_sigma_pow_two = kTwoSigmaSq;
    rb.roi_width               = kVolX;
    rb.roi_height              = kVolY;
    vol.refine_best_depth(dsim_buf, sgm_buf, vol_buf, rb);

    // ---- analytics ----
    const auto* dsim = static_cast<const float*>(dsim_buf.data());

    // Compute per-pixel (gpu_depth, truth_depth, |Δ|).
    std::vector<double> errs;
    int invalid = 0;
    int finite_count = 0;
    errs.reserve(pix);
    for (std::uint32_t vy = 0; vy < kVolY; ++vy)
        for (std::uint32_t vx = 0; vx < kVolX; ++vx) {
            const std::size_t k = std::size_t(vy) * kVolX + vx;
            const float gpu_depth = dsim[k * 2 + 0];
            if (gpu_depth <= 0.0f) { ++invalid; continue; }
            ++finite_count;
            const double cx = (double(vx) + 0.5) * double(kStep);
            const double cy = (double(vy) + 0.5) * double(kStep);
            const double truth = true_depth(cx, cy, double(kTruthZ));
            errs.push_back(std::abs(double(gpu_depth) - truth));
        }
    auto pctl = [](std::vector<double>& v, double q) {
        if (v.empty()) return 0.0;
        const std::size_t k = std::min<std::size_t>(
            v.size() - 1, std::size_t(double(v.size()) * q));
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(k), v.end());
        return v[k];
    };
    std::vector<double> es = errs;
    const double median = pctl(es, 0.50);
    const double p90    = pctl(es, 0.90);
    const double p99    = pctl(es, 0.99);
    const double worst  = es.empty() ? 0.0 : *std::max_element(es.begin(), es.end());

    // Sub-sample resolution = sgm_pix_size / samplesPerPixSize ≈ 0.0025.
    const double sample_size = kSgmPixSize / double(kSamplesPerPixSize);

    std::printf("[info] roi             : %u × %u = %zu pixels\n",
                kVolX, kVolY, pix);
    std::printf("[info] sub-sample size : %.5f (sgm_pix_size %.4f / %d samples)\n",
                sample_size, double(kSgmPixSize), kSamplesPerPixSize);
    std::printf("[info] finite/invalid  : %d / %d\n", finite_count, invalid);
    std::printf("[info] |Δ truth| dist  : median=%.4f p90=%.4f p99=%.4f worst=%.4f\n",
                median, p90, p99, worst);

    // Budgets reflecting integration-test scope:
    //   * finite pixels ≥ 70% (boundary loss from the homography
    //     warp + the NCC kernel's dd=wsh+2 margin).
    //   * median |Δ|: ≤ 2 sub-sample widths. With SGM mid set to
    //     truth, the optimal Refine output is offset 0; FP32 noise
    //     in the NCC chain can push it by 1-2 sub-samples.
    //   * p90 |Δ|: ≤ 6 sub-sample widths.
    bool ok = true;
    const int min_finite = int(pix) * 70 / 100;
    if (finite_count < min_finite) {
        std::fprintf(stderr, "FAIL: only %d / %zu pixels finite (need %d)\n",
                     finite_count, pix, min_finite);
        ok = false;
    }
    if (median > 2.0 * sample_size) {
        std::fprintf(stderr,
            "FAIL: median |Δ| %.4f > %.4f (2 sub-sample widths)\n",
            median, 2.0 * sample_size);
        ok = false;
    }
    if (p90 > 6.0 * sample_size) {
        std::fprintf(stderr,
            "FAIL: p90 |Δ| %.4f > %.4f (6 sub-sample widths)\n",
            p90, 6.0 * sample_size);
        ok = false;
    }

    if (!ok) return 1;
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
