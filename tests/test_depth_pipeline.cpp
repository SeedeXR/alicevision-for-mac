// test_depth_pipeline.cpp — end-to-end SGM → Refine → Optimize
// run on Apple Silicon. The depthMap CUDA kernel surface is now
// fully ported; this test wires all three stages together on the
// S15 plane-induced-homography scene and produces a fused depth
// map.
//
// Stages:
//   1. SGM       — init_sim → compute_similarity → optimize →
//                  retrieve_best_depth.  Output: SGM-resolution
//                  (depth, thickness) map.
//   2. Bridge    — compute_sgm_upscaled_depth_pix_size_map.
//                  Output: Refine-resolution (depth, pixSize) map.
//   3. Refine    — init_refine → refine_similarity →
//                  refine_best_depth.
//                  Output: Refine-resolution (depth, sim) map.
//   4. Optimize  — optimize_depth_sim_map.
//                  Output: fused Refine-resolution (depth, sim).
//
// Validation: per-pixel |opt_depth − truth| vs analytical plane
// depth.  Mirrors S15's accuracy budgeting (lock-on rate within
// 1.5 SGM depth-plane steps; median/p90 of the absolute error).

#include "av/depth_map/DepthSimMap.hpp"
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
using av::depth_map::DepthSimMap;

// ---- geometry ----
constexpr std::uint32_t kImgW = 128;
constexpr std::uint32_t kImgH =  96;

// SGM ROI = 16 × 12 → SGM step in image = 8.
constexpr std::uint32_t kSgmX   = 16;
constexpr std::uint32_t kSgmY   = 12;
constexpr std::uint32_t kSgmZ   = 11;        // 11 depth planes
constexpr int           kSgmStep = 8;        // kSgmX * kSgmStep == kImgW
constexpr int           kSgmWsh  = 4;

// Refine ROI = 32 × 24 → Refine step in image = 4.
constexpr std::uint32_t kRfX   = 32;
constexpr std::uint32_t kRfY   = 24;
constexpr std::uint32_t kRfZ   = 9;          // 9 sub-depth slices
constexpr int           kRfStep = 4;         // kRfX * kRfStep == kImgW
constexpr int           kRfWsh  = 3;
constexpr std::int32_t  kHalfNbDepths      = (int(kRfZ) - 1) / 2;   // 4
constexpr std::int32_t  kSamplesPerPixSize = 4;
constexpr std::int32_t  kHalfNbSamples     = 12;
constexpr float         kSigma             = 4.0f;
constexpr float         kTwoSigmaSq        = 2.0f * kSigma * kSigma;

// 11 planes covering [3.5, 4.5] at step 0.1, truth at idx 5.
constexpr float kTruthZ   = 4.0f;
constexpr float kDepth0   = 3.5f;
constexpr float kDepthDz  = 0.10f;

constexpr float kFx = 200.0f, kFy = 200.0f;
constexpr float kCx = float(kImgW) * 0.5f;
constexpr float kCy = float(kImgH) * 0.5f;

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

std::vector<float> make_r_image() {
    std::vector<float> px(std::size_t(kImgW) * kImgH * 4);
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
            const float b = 128.0f + 45.0f * std::sin(17.0f * (u + v));
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
    return V4{ mix1(ab.r,cd.r,fy), mix1(ab.g,cd.g,fy),
               mix1(ab.b,cd.b,fy), mix1(ab.a,cd.a,fy) };
}

std::vector<float> warp(const std::vector<float>& r, const Eigen::Matrix3d& H_inv) {
    std::vector<float> t(std::size_t(kImgW) * kImgH * 4);
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

double true_depth(double px, double py, double Z) {
    const double X = (px - double(kCx)) * Z / double(kFx);
    const double Y = (py - double(kCy)) * Z / double(kFy);
    return std::sqrt(X * X + Y * Y + Z * Z);
}

double pct(std::vector<double>& v, double q) {
    if (v.empty()) return 0.0;
    const std::size_t k = std::min<std::size_t>(
        v.size() - 1, std::size_t(q * double(v.size())));
    std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(k), v.end());
    return v[k];
}

}  // namespace

int main() try
{
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());
    std::printf("[info] image        : %u × %u\n", kImgW, kImgH);
    std::printf("[info] sgm roi      : %u × %u (step=%d, %u planes)\n",
                kSgmX, kSgmY, kSgmStep, kSgmZ);
    std::printf("[info] refine roi   : %u × %u (step=%d, %u z-slices)\n",
                kRfX, kRfY, kRfStep, kRfZ);

    static_assert(kSgmX * kSgmStep == kImgW, "sgm step");
    static_assert(kSgmY * kSgmStep == kImgH, "sgm step");
    static_assert(kRfX  * kRfStep  == kImgW, "rf step");
    static_assert(kRfY  * kRfStep  == kImgH, "rf step");

    // ---------------- cameras + plane homography ----------------
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

    // ==============================================================
    //  Stage 1 — SGM
    // ==============================================================
    std::vector<float> depths(kSgmZ);
    for (std::uint32_t z = 0; z < kSgmZ; ++z)
        depths[z] = kDepth0 + float(z) * kDepthDz;
    Buffer dep_buf(dev, depths.size() * sizeof(float));
    dep_buf.upload(std::span<const float>(depths));

    const VolumeDims sgm_dims{ kSgmX, kSgmY, kSgmZ };
    const std::size_t sgm_pix    = std::size_t(kSgmX) * std::size_t(kSgmY);
    const std::size_t vol_bytes  = sgm_dims.voxel_count() * sizeof(std::uint8_t);
    Buffer best     (dev, vol_bytes);
    Buffer second   (dev, vol_bytes);
    Buffer filtered (dev, vol_bytes);
    const std::uint32_t maxXY    = std::max(kSgmX, kSgmY);
    Buffer slice_a (dev, maxXY * kSgmZ * sizeof(std::uint32_t));
    Buffer slice_b (dev, maxXY * kSgmZ * sizeof(std::uint32_t));
    Buffer axis_acc(dev, maxXY *           sizeof(std::uint32_t));
    Buffer sgm_dt_buf  (dev, sgm_pix * 2 * sizeof(float));  // (depth, thickness)
    Buffer sgm_dsim_buf(dev, sgm_pix * 2 * sizeof(float));  // (depth, sim)

    Volume vol(dev);
    vol.init_sim(best,   sgm_dims, 255);
    vol.init_sim(second, sgm_dims, 255);

    Volume::ComputeSimilarityParams cs{};
    cs.dims                  = sgm_dims;
    cs.rc_sgm_level_width    = kImgW;  cs.rc_sgm_level_height = kImgH;
    cs.tc_sgm_level_width    = kImgW;  cs.tc_sgm_level_height = kImgH;
    cs.rc_mipmap_level       = 0.0f;
    cs.step_xy               = kSgmStep;
    cs.wsh                   = kSgmWsh;
    cs.inv_gamma_c           = 1.0f / 20.0f;
    cs.inv_gamma_p           = 1.0f / 4.0f;
    cs.use_consistent_scale  = 0;
    cs.depth_range_begin     = 0;
    cs.depth_range_end       = kSgmZ;
    cs.roi_x_begin           = 0;       cs.roi_y_begin           = 0;
    cs.roi_width             = kSgmX;   cs.roi_height            = kSgmY;
    vol.compute_similarity(best, second, dep_buf, rc_tex, tc_tex, rc, tc, cs);

    Volume::OptimizeParams op{};
    op.dims = sgm_dims;
    op.last_depth_index = kSgmZ;
    op.p1 = 10.0f;
    op.p2_abs = 100.0f;
    vol.optimize(filtered, slice_a, slice_b, axis_acc, best, op);

    Volume::RetrieveBestDepthParams rp{};
    rp.dims                  = sgm_dims;
    rp.depth_range_begin     = 0;
    rp.depth_range_end       = kSgmZ;
    rp.roi_x_begin           = 0;       rp.roi_y_begin = 0;
    rp.scale_step            = kSgmStep;
    rp.thickness_mult_factor = 1.0f;
    rp.max_similarity        = 220.0f;
    vol.retrieve_best_depth(sgm_dt_buf, sgm_dsim_buf, dep_buf, filtered, rc, rp);

    // Quick sanity report on SGM output.
    {
        const auto* dt = static_cast<const float*>(sgm_dt_buf.data());
        std::vector<double> errs;
        int valid = 0, invalid = 0;
        for (std::uint32_t y = 0; y < kSgmY; ++y)
            for (std::uint32_t x = 0; x < kSgmX; ++x) {
                const std::size_t k = std::size_t(y) * kSgmX + x;
                const float d = dt[k * 2];
                if (d < 0.0f) { ++invalid; continue; }
                ++valid;
                const double px = (double(x) + 0.5) * double(kSgmStep);
                const double py = (double(y) + 0.5) * double(kSgmStep);
                const double truth = true_depth(px, py, double(kTruthZ));
                errs.push_back(std::abs(double(d) - truth));
            }
        std::printf("[sgm ] valid=%d invalid=%d ", valid, invalid);
        if (!errs.empty()) {
            std::vector<double> e = errs;
            std::printf("median=%.4f p90=%.4f\n",
                        pct(e, 0.5), pct(e, 0.9));
        } else {
            std::printf("\n");
        }
    }

    // ==============================================================
    //  Stage 2 — Bridge: SGM(depth, thickness) → Refine(depth, pixSize)
    // ==============================================================
    Buffer rf_sgm_dp_buf(dev,
        std::size_t(kRfX) * std::size_t(kRfY) * 2 * sizeof(float));

    DepthSimMap dsm(dev);
    DepthSimMap::ComputeUpscaledDepthPixSizeMapParams up{};
    up.out_width       = kRfX;  up.out_height       = kRfY;
    up.in_width        = kSgmX; up.in_height        = kSgmY;
    up.roi_x_begin     = 0;     up.roi_y_begin      = 0;
    up.rc_level_width  = kImgW; up.rc_level_height  = kImgH;
    up.rc_mipmap_level = 0.0f;
    up.step_xy         = kRfStep;
    up.half_nb_depths  = kHalfNbDepths;
    up.bilinear        = true;
    dsm.compute_sgm_upscaled_depth_pix_size_map(
        rf_sgm_dp_buf, sgm_dt_buf, rc_tex, up);

    // Sanity: at least some valid pixels.
    {
        const auto* p = static_cast<const float*>(rf_sgm_dp_buf.data());
        int valid = 0, masked = 0, all_inv = 0;
        for (std::uint32_t k = 0; k < kRfX * kRfY; ++k) {
            const float d = p[k * 2];
            if (d == -2.0f) ++masked;
            else if (d == -1.0f) ++all_inv;
            else ++valid;
        }
        std::printf("[brdg] valid=%d masked=%d all_invalid=%d\n",
                    valid, masked, all_inv);
    }

    // ==============================================================
    //  Stage 3 — Refine
    // ==============================================================
    const VolumeDims rf_dims{ kRfX, kRfY, kRfZ };
    Buffer rf_vol_buf(dev,
        rf_dims.voxel_count() * sizeof(std::uint16_t));
    const std::size_t rf_pix = std::size_t(kRfX) * std::size_t(kRfY);
    Buffer rf_dsim_buf(dev, rf_pix * 2 * sizeof(float));

    vol.init_refine(rf_vol_buf, rf_dims, 0.0f);

    Volume::RefineSimilarityParams rs{};
    rs.dims                   = rf_dims;
    rs.rc_refine_level_width  = kImgW; rs.rc_refine_level_height = kImgH;
    rs.tc_refine_level_width  = kImgW; rs.tc_refine_level_height = kImgH;
    rs.rc_mipmap_level        = 0.0f;
    rs.step_xy                = kRfStep;
    rs.wsh                    = kRfWsh;
    rs.inv_gamma_c            = 1.0f / 20.0f;
    rs.inv_gamma_p            = 1.0f / 4.0f;
    rs.use_consistent_scale   = 0;
    rs.depth_range_begin      = 0;
    rs.depth_range_end        = kRfZ;
    rs.roi_x_begin            = 0;     rs.roi_y_begin            = 0;
    rs.roi_width              = kRfX;  rs.roi_height             = kRfY;
    vol.refine_similarity(rf_vol_buf, rf_sgm_dp_buf,
                          rc_tex, tc_tex, rc, tc, rs);

    Volume::RefineBestDepthParams rb{};
    rb.dims                    = rf_dims;
    rb.samples_per_pix_size    = kSamplesPerPixSize;
    rb.half_nb_samples         = kHalfNbSamples;
    rb.half_nb_depths          = kHalfNbDepths;
    rb.two_times_sigma_pow_two = kTwoSigmaSq;
    rb.roi_width               = kRfX; rb.roi_height = kRfY;
    vol.refine_best_depth(rf_dsim_buf, rf_sgm_dp_buf, rf_vol_buf, rb);

    {
        const auto* p = static_cast<const float*>(rf_dsim_buf.data());
        std::vector<double> errs;
        int valid = 0, invalid = 0;
        for (std::uint32_t y = 0; y < kRfY; ++y)
            for (std::uint32_t x = 0; x < kRfX; ++x) {
                const std::size_t k = std::size_t(y) * kRfX + x;
                const float d = p[k * 2];
                if (d <= 0.0f) { ++invalid; continue; }
                ++valid;
                const double px = (double(x) + 0.5) * double(kRfStep);
                const double py = (double(y) + 0.5) * double(kRfStep);
                const double truth = true_depth(px, py, double(kTruthZ));
                errs.push_back(std::abs(double(d) - truth));
            }
        std::printf("[refn] valid=%d invalid=%d ", valid, invalid);
        if (!errs.empty()) {
            std::vector<double> e = errs;
            std::printf("median=%.4f p90=%.4f\n",
                        pct(e, 0.5), pct(e, 0.9));
        } else {
            std::printf("\n");
        }
    }

    // ==============================================================
    //  Stage 4 — Optimize (gradient-descent fusion)
    // ==============================================================
    Buffer opt_buf(dev, rf_pix * 2 * sizeof(float));
    Texture variance_tex (dev, Texture::Descriptor{
        kRfX, kRfY, 1, PixelFormat::R32Float });
    Texture tmp_depth_tex(dev, Texture::Descriptor{
        kRfX, kRfY, 1, PixelFormat::R32Float });

    DepthSimMap::OptimizeGradientDescentParams ogp{};
    ogp.width            = kRfX;
    ogp.height           = kRfY;
    ogp.roi_x_begin      = 0;
    ogp.roi_y_begin      = 0;
    ogp.rc_level_width   = kImgW;
    ogp.rc_level_height  = kImgH;
    ogp.rc_mipmap_level  = 0.0f;
    ogp.step_xy          = kRfStep;
    ogp.nb_iterations    = 20;
    dsm.optimize_depth_sim_map(
        opt_buf, rf_sgm_dp_buf, rf_dsim_buf, rc_tex,
        variance_tex, tmp_depth_tex, rc, ogp);

    // ==============================================================
    //  Final validation: opt depth vs analytical truth.
    // ==============================================================
    const auto* opt = static_cast<const float*>(opt_buf.data());
    std::vector<double> errs;
    int valid = 0, invalid = 0;
    int locked = 0;
    constexpr double kLockSteps = 1.5;
    constexpr double kLockTol   = kLockSteps * double(kDepthDz);  // 1.5 plane steps
    for (std::uint32_t y = 0; y < kRfY; ++y)
        for (std::uint32_t x = 0; x < kRfX; ++x) {
            const std::size_t k = std::size_t(y) * kRfX + x;
            const float d = opt[k * 2];
            if (d <= 0.0f) { ++invalid; continue; }
            ++valid;
            const double px = (double(x) + 0.5) * double(kRfStep);
            const double py = (double(y) + 0.5) * double(kRfStep);
            const double truth = true_depth(px, py, double(kTruthZ));
            const double err = std::abs(double(d) - truth);
            errs.push_back(err);
            if (err <= kLockTol) ++locked;
        }

    std::vector<double> e = errs;
    const double median = pct(e, 0.5);
    const double p90    = pct(e, 0.9);
    const double p99    = pct(e, 0.99);
    const double worst  = errs.empty() ? 0.0
        : *std::max_element(errs.begin(), errs.end());
    const double lock_rate = errs.empty() ? 0.0
        : 100.0 * double(locked) / double(errs.size());

    std::printf("[opt ] valid=%d invalid=%d (of %u)\n",
                valid, invalid, kRfX * kRfY);
    std::printf("[opt ] |Δ truth| median=%.4f p90=%.4f p99=%.4f worst=%.4f\n",
                median, p90, p99, worst);
    std::printf("[opt ] lock-on (≤ %.2f) %.1f%%  (%d/%zu)\n",
                kLockTol, lock_rate, locked, errs.size());

    // Budgets reflect the quantization-limited regime: SGM picks
    // discrete depth planes (0.1 step) but the Refine sub-sample
    // sweep covers only ±0.075 around the SGM seed (= ±halfNbSamples
    // × sgm_pix_size / samplesPerPixSize, with sgm_pix_size ≈ 0.025).
    // So pixels where SGM picks the wrong plane (offset = 0.10 from
    // truth) sit outside Refine's correction range; their final
    // |Δ| stays near 1 SGM step. The right validation metric is the
    // lock-on rate at 1.5 SGM steps, not strict median.
    bool ok = true;
    if (valid < int(kRfX * kRfY) * 60 / 100) {
        std::fprintf(stderr, "FAIL: only %d / %u valid pixels\n",
                     valid, kRfX * kRfY);
        ok = false;
    }
    if (median > 1.2 * double(kDepthDz)) {
        std::fprintf(stderr,
            "FAIL: median |Δ| %.4f > 1.2 SGM steps (%.4f)\n",
            median, 1.2 * double(kDepthDz));
        ok = false;
    }
    if (lock_rate < 70.0) {
        std::fprintf(stderr,
            "FAIL: lock-on rate %.1f%% < 70%% (within 1.5 SGM steps)\n",
            lock_rate);
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
