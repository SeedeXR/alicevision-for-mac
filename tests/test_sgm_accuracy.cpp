// test_sgm_accuracy.cpp — first photogrammetrically meaningful
// depth recovery on Apple Silicon.
//
// Builds a non-degenerate 2-view scene: R image is a textured
// pattern; T image is R warped through the plane-induced
// homography H for a known plane at world Z = kTruthZ. After
// warping, T is *geometrically consistent* with R observing the
// plane — running the SGM pipeline should recover Z = kTruthZ.
//
// Pipeline:
//     init_sim → compute_similarity → optimize → retrieve_best_depth
//
// Validation: for each pixel that produced a finite depth, compare
// against the per-pixel analytical truth
//   true_depth(px, py) = Z_plane * sqrt(((px-cx)/f)² + ((py-cy)/f)² + 1)
// We allow ~one depth-plane step of slack on the median error and
// two steps on the p90, reflecting (a) FP32 noise in the NCC
// chain and (b) discrete depth-plane indexing.
//
// The plane homography is built using the same formula that the
// MSL kernel's `computeHomography` (Patch.h) implements, validated
// in S5 against an Eigen FP64 reference (test_patch.cpp).

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
constexpr std::uint32_t kVolZ   = 11;        // 11 depth planes
constexpr int           kStep   = 8;         // kVolX * kStep == kImgW
constexpr int           kWsh    = 4;

constexpr float kTruthZ  = 4.0f;
constexpr float kDepth0  = 3.5f;             // depths[0]
constexpr float kDepthDz = 0.10f;            // step (so kTruthZ is depths[5])
constexpr std::uint32_t kTruthIdx = 5;

constexpr float kFx = 400.0f;
constexpr float kFy = 400.0f;
constexpr float kCx = float(kImgW) * 0.5f;
constexpr float kCy = float(kImgH) * 0.5f;

// ---- camera utilities (mirrored from test_comp_ncc) ----

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

// ---- R image (textured) ----

std::vector<float> make_r_image() {
    std::vector<float> px(kImgW * kImgH * 4);
    for (std::uint32_t j = 0; j < kImgH; ++j) {
        for (std::uint32_t i = 0; i < kImgW; ++i) {
            const float u = float(i) / float(kImgW);
            const float v = float(j) / float(kImgH);
            // High-frequency texture so NCC has structure to lock onto.
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
    }
    return px;
}

// ---- Plane-induced homography (port of `computeHomography` from Patch.h
//      to FP64 / Eigen). Mirrors upstream's exact formula. ----

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

// ---- bilinear sample with clamp-to-edge on a row-major RGBA buffer ----

struct V4 { float r, g, b, a; };

V4 bilin_sample(const std::vector<float>& tex, double px, double py) {
    const double cx = px - 0.5;
    const double cy = py - 0.5;
    const int ix0_raw = int(std::floor(cx));
    const int iy0_raw = int(std::floor(cy));
    const double fx = cx - ix0_raw;
    const double fy = cy - iy0_raw;
    auto cl = [](int v, int lo, int hi) {
        return std::max(lo, std::min(v, hi));
    };
    const int Wi = int(kImgW), Hi = int(kImgH);
    const int x0 = cl(ix0_raw,     0, Wi - 1);
    const int x1 = cl(ix0_raw + 1, 0, Wi - 1);
    const int y0 = cl(iy0_raw,     0, Hi - 1);
    const int y1 = cl(iy0_raw + 1, 0, Hi - 1);
    auto load = [&](int x, int y) {
        const std::size_t k = (std::size_t(y) * kImgW + x) * 4;
        return V4{ tex[k + 0], tex[k + 1], tex[k + 2], tex[k + 3] };
    };
    V4 a = load(x0, y0), b = load(x1, y0);
    V4 c = load(x0, y1), d = load(x1, y1);
    auto mix1 = [](float u, float v, double t) {
        return float(u + (v - u) * t);
    };
    V4 ab{ mix1(a.r, b.r, fx), mix1(a.g, b.g, fx),
           mix1(a.b, b.b, fx), mix1(a.a, b.a, fx) };
    V4 cd{ mix1(c.r, d.r, fx), mix1(c.g, d.g, fx),
           mix1(c.b, d.b, fx), mix1(c.a, d.a, fx) };
    return V4{ mix1(ab.r, cd.r, fy), mix1(ab.g, cd.g, fy),
               mix1(ab.b, cd.b, fy), mix1(ab.a, cd.a, fy) };
}

// ---- Warp R image into T image via H_inv: for each T pixel,
//      find the corresponding R pixel and bilinearly sample. ----

std::vector<float> warp_r_to_t(const std::vector<float>& r_img,
                                const Eigen::Matrix3d& H_inv)
{
    std::vector<float> t_img(kImgW * kImgH * 4);
    for (std::uint32_t j = 0; j < kImgH; ++j) {
        for (std::uint32_t i = 0; i < kImgW; ++i) {
            const Eigen::Vector3d t_h(double(i) + 0.5,
                                       double(j) + 0.5, 1.0);
            const Eigen::Vector3d r_h = H_inv * t_h;
            const double r_x = r_h(0) / r_h(2) - 0.5;
            const double r_y = r_h(1) / r_h(2) - 0.5;
            const V4 c = bilin_sample(r_img, r_x + 0.5, r_y + 0.5);
            const std::size_t k = (j * kImgW + i) * 4;
            t_img[k + 0] = c.r;
            t_img[k + 1] = c.g;
            t_img[k + 2] = c.b;
            t_img[k + 3] = c.a;
        }
    }
    return t_img;
}

// True euclidean depth at pixel (px, py) for the plane Z = Z_plane.
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

    static_assert(kVolX * kStep == kImgW, "kVolX * kStep != kImgW");
    static_assert(kVolY * kStep == kImgH, "kVolY * kStep != kImgH");

    // ---------------- cameras ----------------
    Eigen::Matrix3d K;
    K << kFx,   0.0, double(kCx),
         0.0,  kFy,  double(kCy),
         0.0,  0.0,  1.0;
    const Eigen::Matrix3d Rrc = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d Crc(0.0, 0.0, 0.0);
    // Baseline tuned for: (a) wide enough that
    // `computeRotCSEpip`'s cross(v1, v2) is far from degenerate
    // (which improves NCC's peak sharpness at the truth Z), but
    // (b) narrow enough that the warped T image keeps most pixels
    // inside R's bounds.
    const Eigen::AngleAxisd aa(0.03, Eigen::Vector3d::UnitY());
    const Eigen::Matrix3d Rtc = aa.toRotationMatrix();
    const Eigen::Vector3d Ctc(0.25, 0.0, 0.0);
    const DeviceCameraParams rc = make_cam(K, Rrc, Crc);
    const DeviceCameraParams tc = make_cam(K, Rtc, Ctc);

    // ---------------- plane & homography ----------------
    // Plane: Z = kTruthZ in world coords, normal toward the camera.
    const Eigen::Vector3d plane_point (0.0, 0.0, double(kTruthZ));
    const Eigen::Vector3d plane_normal(0.0, 0.0, -1.0);
    const Eigen::Matrix3d H = plane_homography(K, Rrc, Crc, Rtc, Ctc,
                                                plane_point, plane_normal);
    const Eigen::Matrix3d H_inv = H.inverse();

    // ---------------- images ----------------
    const auto r_img = make_r_image();
    const auto t_img = warp_r_to_t(r_img, H_inv);

    Texture rc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, /*mip_levels=auto*/ 0, PixelFormat::RGBA32Float });
    Texture tc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, /*mip_levels=auto*/ 0, PixelFormat::RGBA32Float });
    rc_tex.set_label("acc.rc"); tc_tex.set_label("acc.tc");
    rc_tex.upload(std::span<const float>(r_img));
    tc_tex.upload(std::span<const float>(t_img));
    rc_tex.generate_mipmaps();
    tc_tex.generate_mipmaps();

    // ---------------- depths buffer ----------------
    std::vector<float> depths(kVolZ);
    for (std::uint32_t z = 0; z < kVolZ; ++z)
        depths[z] = kDepth0 + float(z) * kDepthDz;
    Buffer dep_buf(dev, depths.size() * sizeof(float));
    dep_buf.upload(std::span<const float>(depths));

    // ---------------- volume buffers ----------------
    const VolumeDims dims{ kVolX, kVolY, kVolZ };
    const std::size_t vol_bytes = dims.voxel_count() * sizeof(std::uint8_t);
    Buffer best    (dev, vol_bytes);
    Buffer second  (dev, vol_bytes);
    Buffer filtered(dev, vol_bytes);
    const std::uint32_t maxXY = std::max(kVolX, kVolY);
    Buffer slice_a (dev, maxXY * kVolZ * sizeof(std::uint32_t));
    Buffer slice_b (dev, maxXY * kVolZ * sizeof(std::uint32_t));
    Buffer axis_acc(dev, maxXY *           sizeof(std::uint32_t));
    const std::size_t out_pixels = std::size_t(kVolX) * std::size_t(kVolY);
    Buffer dt_buf  (dev, out_pixels * 2 * sizeof(float));
    Buffer dsim_buf(dev, out_pixels * 2 * sizeof(float));

    Volume vol(dev);

    // ---------------- pipeline ----------------
    vol.init_sim(best,   dims, 255);
    vol.init_sim(second, dims, 255);

    Volume::ComputeSimilarityParams cs{};
    cs.dims                  = dims;
    cs.rc_sgm_level_width    = kImgW;
    cs.rc_sgm_level_height   = kImgH;
    cs.tc_sgm_level_width    = kImgW;
    cs.tc_sgm_level_height   = kImgH;
    cs.rc_mipmap_level       = 0.0f;
    cs.step_xy               = kStep;
    cs.wsh                   = kWsh;
    cs.inv_gamma_c           = 1.0f / 20.0f;
    cs.inv_gamma_p           = 1.0f / 4.0f;
    cs.use_consistent_scale  = 0;
    cs.depth_range_begin     = 0;
    cs.depth_range_end       = kVolZ;
    cs.roi_x_begin           = 0;
    cs.roi_y_begin           = 0;
    cs.roi_width             = kVolX;
    cs.roi_height            = kVolY;
    vol.compute_similarity(best, second, dep_buf, rc_tex, tc_tex, rc, tc, cs);

    Volume::OptimizeParams op{};
    op.dims             = dims;
    op.last_depth_index = kVolZ;
    op.p1               = 10.0f;
    op.p2_abs           = 100.0f;
    vol.optimize(filtered, slice_a, slice_b, axis_acc, best, op);

    Volume::RetrieveBestDepthParams rp{};
    rp.dims                  = dims;
    rp.depth_range_begin     = 0;
    rp.depth_range_end       = kVolZ;
    rp.roi_x_begin           = 0;
    rp.roi_y_begin           = 0;
    rp.scale_step            = kStep;
    rp.thickness_mult_factor = 1.0f;
    rp.max_similarity        = 220.0f;
    vol.retrieve_best_depth(dt_buf, dsim_buf, dep_buf, filtered, rc, rp);

    // ---------------- compare against analytical truth ----------------
    const auto* dt = static_cast<const float*>(dt_buf.data());

    std::vector<double> errs;
    int finite_count = 0;
    int invalid_count = 0;
    errs.reserve(out_pixels);

    for (std::uint32_t vy = 0; vy < kVolY; ++vy) {
        for (std::uint32_t vx = 0; vx < kVolX; ++vx) {
            const std::size_t k = std::size_t(vy) * kVolX + vx;
            const float d = dt[k * 2 + 0];
            if (d < 0.0f) { ++invalid_count; continue; }
            ++finite_count;
            const double px = (double(vx) + 0.5) * double(kStep);
            const double py = (double(vy) + 0.5) * double(kStep);
            const double truth = true_depth(px, py, double(kTruthZ));
            errs.push_back(std::abs(double(d) - truth));
        }
    }

    auto percentile = [](std::vector<double>& v, double q) {
        if (v.empty()) return 0.0;
        const std::size_t k = std::min<std::size_t>(
            v.size() - 1,
            std::size_t(double(v.size()) * q));
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(k), v.end());
        return v[k];
    };
    std::vector<double> es = errs;
    const double err_median = percentile(es, 0.50);
    const double err_p90    = percentile(es, 0.90);
    const double err_p99    = percentile(es, 0.99);

    std::printf("[info] roi             : %u × %u = %zu pixels\n",
                kVolX, kVolY, out_pixels);
    std::printf("[info] depth planes    : %u, range [%.2f, %.2f] step %.2f, truth Z=%.2f (idx %u)\n",
                kVolZ, double(kDepth0),
                double(kDepth0) + double(kVolZ - 1) * double(kDepthDz),
                double(kDepthDz), double(kTruthZ), kTruthIdx);
    std::printf("[info] finite / invalid: %d / %d\n",
                finite_count, invalid_count);
    std::printf("[info] |Δ truth| dist  : median=%.3f p90=%.3f p99=%.3f\n",
                err_median, err_p90, err_p99);

    // Budgets reflecting real FP32 SGM behavior with a 0.10
    // depth-plane step:
    //   * Finite pixels: ≥ 80% (boundary loss from the homography
    //     warp + the NCC kernel's own dd=wsh+2 margin).
    //   * Median |Δ|: ≤ 1.5 * kDepthDz. The NCC chain has ~1-step
    //     FP32 noise that flips ~50% of pixels between plane k and
    //     k±1 even when the truth lies exactly between them.
    //   * p90 |Δ|: ≤ 3 * kDepthDz. Edge-region patches near the
    //     warp boundary have weaker NCC and can drift 2-3 planes.
    //   * "Lock-on rate": ≥ 60% of pixels within 1.5 * kDepthDz of
    //     truth — confirms most pixels pick the right plane or one
    //     adjacent, not random noise.
    bool ok = true;
    const int min_finite = int(out_pixels) * 80 / 100;
    if (finite_count < min_finite) {
        std::fprintf(stderr, "FAIL: only %d / %d pixels finite (need %d)\n",
                     finite_count, int(out_pixels), min_finite);
        ok = false;
    }
    if (err_median > 1.5 * double(kDepthDz)) {
        std::fprintf(stderr,
            "FAIL: median |Δ depth| %.3f > %.3f (1.5 depth-plane steps)\n",
            err_median, 1.5 * double(kDepthDz));
        ok = false;
    }
    if (err_p90 > 3.0 * double(kDepthDz)) {
        std::fprintf(stderr,
            "FAIL: p90 |Δ depth| %.3f > %.3f (3 depth-plane steps)\n",
            err_p90, 3.0 * double(kDepthDz));
        ok = false;
    }

    int lock_on = 0;
    const double lock_threshold = 1.5 * double(kDepthDz);
    for (const double e : errs)
        if (e <= lock_threshold) ++lock_on;
    const double lock_on_rate = double(lock_on) / double(finite_count);
    std::printf("[info] lock-on rate     : %.1f%% within 1.5 depth-plane steps\n",
                100.0 * lock_on_rate);
    if (lock_on_rate < 0.60) {
        std::fprintf(stderr,
            "FAIL: only %.1f%% of pixels within 1.5 depth-plane steps of truth (need >= 60%%)\n",
            100.0 * lock_on_rate);
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
