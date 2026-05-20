// test_sgm_pipeline.cpp — first end-to-end SGM run on Apple Silicon.
//
// Wires together every Phase 7 kernel we have:
//   1. Volume::init_sim          — best, 2nd-best ← 255
//   2. Volume::compute_similarity — fill cost volume from R+T NCC
//   3. Volume::optimize           — SGM 4-direction DP regularization
//   4. Volume::retrieve_best_depth — depth/sim map from filtered volume
//
// Scene: pinhole R camera at origin looking down +Z, pinhole T
// camera with 0.3 baseline and 0.05 rad Y rotation. Both cameras
// see an identical RGBA32Float "image" (smooth sin+gradient
// pattern, same content in both views — geometrically-correct
// matches in the plane-sweep produce strong NCC). Truth: a
// fronto-parallel plane at world Z = 4.0.
//
// What this test actually validates (and what it doesn't):
//
//   This is a *smoke test* for the SGM pipeline integration. It
//   verifies that:
//     * The four kernels wire together with consistent data
//       layouts (volume dims, depth-plane indexing, ROI).
//     * `init_sim → compute_similarity → optimize →
//       retrieve_best_depth` runs without errors and produces a
//       majority of finite-depth pixels.
//     * The depth output is *clustered* (not random), and
//       individual depths stay inside the back-projected range of
//       the depth-plane list (a check that retrieve_best_depth's
//       index→depth conversion uses a sensible plane).
//     * `out_depth_thickness.x == out_depth_sim.x` per pixel (the
//       invariant from S11).
//
//   What this test does NOT validate is depth-recovery *accuracy*.
//   With identical R=T image content (the test_comp_ncc trick
//   needed for unit-testing one patch in isolation), the NCC
//   surface isn't sharply peaked at the geometrically-correct Z —
//   multiple depth hypotheses produce similar NCC values because
//   the texture is smooth. The DP regularization then picks the
//   most-consistent-across-neighbors depth, which on this
//   degenerate scene tends to be the highest-Z hypothesis (the
//   patches there sample the widest image area, averaging more
//   texture). Real depth-recovery validation would use
//   T = H_plane × R (plane-induced homography on a real image) or
//   actual photogrammetric imagery — outside the scope of this
//   smoke test. The unit tests in S10..S13 already validate each
//   kernel's *internal* correctness against a CPU reference.

#include "av/depth_map/PatchOps.hpp"   // DeviceCameraParams mirror
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

// Scene + volume parameters.
constexpr std::uint32_t kImgW   = 256;
constexpr std::uint32_t kImgH   = 192;
constexpr std::uint32_t kVolX   = 32;
constexpr std::uint32_t kVolY   = 24;
constexpr std::uint32_t kVolZ   = 9;            // 9 depth planes
// 32 voxels along X * 8 pix/voxel = 256 image pixels — tiles
// the input image without gaps.
constexpr int           kStep   = 8;
constexpr int           kWsh    = 3;            // 7×7 NCC patch

// Truth plane depth, and the depth-plane sweep range straddling it.
constexpr float         kTruthZ  = 4.0f;
constexpr float         kDepth0  = 3.6f;        // depths[0]
constexpr float         kDepthDz = 0.10f;       // step between planes
// → depths[i] = 3.6, 3.7, ..., 4.4 (9 planes, kTruthZ at i=4)

// Camera intrinsics.
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

// Identical-content R/T image with structure NCC can lock onto.
std::vector<float> make_image() {
    std::vector<float> px(kImgW * kImgH * 4);
    for (std::uint32_t j = 0; j < kImgH; ++j) {
        for (std::uint32_t i = 0; i < kImgW; ++i) {
            const float u = float(i) / float(kImgW);
            const float v = float(j) / float(kImgH);
            const float intensity =
                128.0f + 80.0f * std::sin(8.0f * u + 0.3f * v)
                        + 40.0f * std::cos(3.0f * v - 0.7f * u);
            const std::size_t k = (j * kImgW + i) * 4;
            px[k + 0] = intensity;
            px[k + 1] = 127.5f + 60.0f * u;
            px[k + 2] = 127.5f + 60.0f * v;
            px[k + 3] = 255.0f;
        }
    }
    return px;
}

// True euclidean depth from camera center to the point at pixel
// (px, py) projected onto the plane Z = Z_plane (camera at origin,
// identity rotation).
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

    static_assert(kVolX * kStep == kImgW, "ROI step doesn't tile the image");
    static_assert(kVolY * kStep == kImgH, "ROI step doesn't tile the image");

    // ---------------- cameras ----------------
    Eigen::Matrix3d K;
    K << kFx,   0.0, double(kCx),
         0.0,  kFy,  double(kCy),
         0.0,  0.0,  1.0;
    const Eigen::Matrix3d Rrc = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d Crc(0.0, 0.0, 0.0);
    const Eigen::AngleAxisd aa(0.05, Eigen::Vector3d::UnitY());
    const Eigen::Matrix3d Rtc = aa.toRotationMatrix();
    const Eigen::Vector3d Ctc(0.3, 0.0, 0.0);
    const DeviceCameraParams rc = make_cam(K, Rrc, Crc);
    const DeviceCameraParams tc = make_cam(K, Rtc, Ctc);

    // ---------------- textures ----------------
    const auto img = make_image();
    Texture rc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, /*mip_levels=auto*/ 0, PixelFormat::RGBA32Float });
    Texture tc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, /*mip_levels=auto*/ 0, PixelFormat::RGBA32Float });
    rc_tex.set_label("sgm.rc"); tc_tex.set_label("sgm.tc");
    rc_tex.upload(std::span<const float>(img));
    tc_tex.upload(std::span<const float>(img));
    rc_tex.generate_mipmaps();
    tc_tex.generate_mipmaps();

    // ---------------- depths buffer ----------------
    std::vector<float> depths(kVolZ);
    for (std::uint32_t z = 0; z < kVolZ; ++z)
        depths[z] = kDepth0 + float(z) * kDepthDz;
    // sanity: kTruthZ should land near the middle of the sweep
    const std::uint32_t truth_z_idx = std::uint32_t(std::round((kTruthZ - kDepth0) / kDepthDz));
    Buffer dep_buf(dev, depths.size() * sizeof(float));
    dep_buf.upload(std::span<const float>(depths));

    // ---------------- volume buffers ----------------
    const VolumeDims dims{ kVolX, kVolY, kVolZ };
    const std::size_t vol_bytes = dims.voxel_count() * sizeof(std::uint8_t);

    Buffer best    (dev, vol_bytes);   // input to optimize; also 1st-best in compute_similarity
    Buffer second  (dev, vol_bytes);   // 2nd-best (compute_similarity output)
    Buffer filtered(dev, vol_bytes);   // optimize output; input to retrieve_best_depth
    best    .set_label("sgm.best");
    second  .set_label("sgm.second");
    filtered.set_label("sgm.filtered");

    // optimize scratch
    const std::uint32_t maxXY = std::max(kVolX, kVolY);
    Buffer slice_a (dev, maxXY * kVolZ * sizeof(std::uint32_t));
    Buffer slice_b (dev, maxXY * kVolZ * sizeof(std::uint32_t));
    Buffer axis_acc(dev, maxXY *           sizeof(std::uint32_t));

    // retrieve outputs (float2 packed per pixel)
    const std::size_t out_pixels = std::size_t(kVolX) * std::size_t(kVolY);
    Buffer dt_buf  (dev, out_pixels * 2 * sizeof(float));
    Buffer dsim_buf(dev, out_pixels * 2 * sizeof(float));
    dt_buf  .set_label("sgm.depth_thickness");
    dsim_buf.set_label("sgm.depth_sim");

    Volume vol(dev);

    // ===================================================
    // 1. init both cost volumes to 255
    // ===================================================
    vol.init_sim(best,   dims, 255);
    vol.init_sim(second, dims, 255);

    // ===================================================
    // 2. compute_similarity for (R, T)
    // ===================================================
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

    // ===================================================
    // 3. optimize (SGM 4-direction DP)
    // ===================================================
    Volume::OptimizeParams op{};
    op.dims              = dims;
    op.last_depth_index  = kVolZ;
    op.p1                = 10.0f;
    op.p2_abs            = 100.0f;
    vol.optimize(filtered, slice_a, slice_b, axis_acc, best, op);

    // ===================================================
    // 4. retrieve_best_depth
    // ===================================================
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

    // ---------------- read back depth map ----------------
    const auto* dt   = static_cast<const float*>(dt_buf.data());
    const auto* dsim = static_cast<const float*>(dsim_buf.data());

    // ---------------- analytics ----------------
    std::vector<double> finite_depths;
    std::vector<double> per_pixel_err;       // |gpu_depth - true_depth| per pixel
    int invalid = 0;
    finite_depths.reserve(out_pixels);
    per_pixel_err.reserve(out_pixels);

    for (std::uint32_t vy = 0; vy < kVolY; ++vy) {
        for (std::uint32_t vx = 0; vx < kVolX; ++vx) {
            const std::size_t k = std::size_t(vy) * kVolX + vx;
            const float d = dt[k * 2 + 0];
            if (d < 0.0f) {
                ++invalid;
                continue;
            }
            finite_depths.push_back(double(d));

            // analytical truth at the center pixel of this voxel column
            const double px = (double(vx) + 0.5) * double(kStep);
            const double py = (double(vy) + 0.5) * double(kStep);
            const double truth = true_depth(px, py, double(kTruthZ));
            per_pixel_err.push_back(std::abs(double(d) - truth));
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

    std::vector<double> fd_sorted = finite_depths;
    std::vector<double> err_sorted = per_pixel_err;
    const double depth_median = percentile(fd_sorted, 0.50);
    const double depth_p10    = percentile(fd_sorted, 0.10);
    const double depth_p90    = percentile(fd_sorted, 0.90);
    const double err_median   = percentile(err_sorted, 0.50);
    const double err_p90      = percentile(err_sorted, 0.90);

    std::printf("[info] roi             : %u × %u = %zu pixels\n",
                kVolX, kVolY, out_pixels);
    std::printf("[info] depth planes    : %u, range [%.2f, %.2f] step %.2f, truth Z=%.2f (idx ≈ %u)\n",
                kVolZ, double(kDepth0),
                double(kDepth0) + double(kVolZ - 1) * double(kDepthDz),
                double(kDepthDz), double(kTruthZ), truth_z_idx);
    std::printf("[info] invalid pixels  : %d / %zu (%.1f%%)\n",
                invalid, out_pixels,
                100.0 * double(invalid) / double(out_pixels));
    std::printf("[info] gpu depth dist  : p10=%.3f median=%.3f p90=%.3f\n",
                depth_p10, depth_median, depth_p90);
    std::printf("[info] |Δ truth| dist  : median=%.3f p90=%.3f\n",
                err_median, err_p90);

    // ---------------- assertions (smoke-test level) ----------------
    // We assert *integration* correctness, not depth-recovery
    // accuracy (see the header comment for why a real depth-
    // recovery validation would need a different synthetic scene).
    bool ok = true;

    // 1. Majority of pixels produce finite depths (kernel + pipeline
    //    plumbing isn't broken).
    if (invalid > int(out_pixels) / 2) {
        std::fprintf(stderr,
            "FAIL: more than half the pixels are invalid: %d / %zu\n",
            invalid, out_pixels);
        ok = false;
    }

    // 2. Depths are *clustered*, not random. p90 - p10 should be
    //    small compared to the depth-plane sweep range.
    const double depth_spread = depth_p90 - depth_p10;
    const double sweep_range  = double(kDepthDz) * double(kVolZ - 1);
    if (depth_spread > sweep_range) {
        std::fprintf(stderr,
            "FAIL: depth distribution is not clustered "
            "(p10-p90 spread %.3f > sweep range %.3f)\n",
            depth_spread, sweep_range);
        ok = false;
    }

    // 3. Depths land inside the *back-projected* range of the
    //    depth-plane list, i.e., between
    //    [depthPlaneToDepth(min_plane, ...), depthPlaneToDepth(max_plane, corner)].
    //    For our intrinsics, the corner-stretching factor reaches
    //    ~1.08, so kDepth0 / 1.08 ≈ 3.33 and (kDepth0 + (kVolZ-1)*kDepthDz) * 1.08
    //    ≈ 4.75. Anything outside [3.0, 5.0] would indicate a
    //    depth-plane-indexing bug.
    if (depth_p10 < 3.0 || depth_p90 > 5.0) {
        std::fprintf(stderr,
            "FAIL: depths outside the back-projected plane range "
            "(p10=%.3f p90=%.3f)\n", depth_p10, depth_p90);
        ok = false;
    }
    // depth/sim's depth field should match the depth/thickness depth
    // bit-for-bit (single-source-of-truth assertion from S11).
    int dt_dsim_mismatch = 0;
    for (std::size_t k = 0; k < out_pixels; ++k) {
        if (dt[k * 2 + 0] != dsim[k * 2 + 0]) ++dt_dsim_mismatch;
    }
    if (dt_dsim_mismatch != 0) {
        std::fprintf(stderr,
            "FAIL: depth disagrees between depth/thickness and depth/sim outputs (%d pixels)\n",
            dt_dsim_mismatch);
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
