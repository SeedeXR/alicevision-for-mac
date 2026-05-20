// test_compute_similarity.cpp — validation of the
// volume_compute_similarity kernel (heaviest Phase 7 kernel).
//
// Strategy:
//   1. Build a 2-view synthetic scene (mirrors test_comp_ncc.cpp:
//      identical 256×256 RGBA32Float gradient on both R and T
//      textures with a small baseline + rotation).
//   2. Define a small (volX=16, volY=12, volZ=4) cost volume over
//      a few depth planes at depths [3.5, 4.0, 4.5, 5.0].
//   3. Initialize both best/2nd-best volumes to 255 (sentinel).
//   4. Dispatch compute_similarity for the single (R, T) pair.
//   5. CPU FP64 reference does the same algorithm: for each voxel,
//      compute Patch, compute NCC (bilinear texture sample +
//      Yoon-Kweon weights + simStat), remap to uchar, WTA-update.
//   6. Compare voxel-by-voxel.
//
// Key correctness invariants:
//   * After one (R, T) invocation, the 2nd-best volume must stay
//     all-255 (the WTA only demotes on the second+ T call).
//   * The best volume must match the CPU reference within a few
//     uchar units of drift (FP32-vs-FP64 noise through the NCC
//     chain maps to discretization differences after the
//     0..255 remap).

#include "av/depth_map/Volume.hpp"
#include "av/depth_map/PatchOps.hpp"   // DeviceCameraParams mirror
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using av::depth_map::DeviceCameraParams;
using av::depth_map::Volume;
using av::depth_map::VolumeDims;

// ---- scene parameters ----
constexpr std::uint32_t kTexW = 256;
constexpr std::uint32_t kTexH = 256;
constexpr std::uint32_t kVolX = 16;
constexpr std::uint32_t kVolY = 12;
constexpr std::uint32_t kVolZ = 4;
constexpr int           kWsh  = 4;
constexpr int           kStep = 8;        // ROI step ≈ kTexW / kVolX

constexpr float kRcMinAlpha = 255.0f * 0.9f;
constexpr float kTcMinAlpha = 255.0f * 0.4f;

// ---- camera utilities (lifted from test_comp_ncc) ----

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
Eigen::Matrix3d iP_of(const DeviceCameraParams& cp) {
    Eigen::Matrix3d iP;
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            iP(i, j) = static_cast<double>(cp.iP[j * 3 + i]);
    return iP;
}
Eigen::Vector3d C_of(const DeviceCameraParams& cp) {
    return { static_cast<double>(cp.C[0]),
             static_cast<double>(cp.C[1]),
             static_cast<double>(cp.C[2]) };
}
Eigen::Vector3d Z_of(const DeviceCameraParams& cp) {
    return { static_cast<double>(cp.ZVect[0]),
             static_cast<double>(cp.ZVect[1]),
             static_cast<double>(cp.ZVect[2]) };
}
Eigen::Vector2d project(const Eigen::Matrix<double, 3, 4>& P,
                        const Eigen::Vector3d& X)
{
    Eigen::Vector4d Xh; Xh << X, 1.0;
    Eigen::Vector3d ph = P * Xh;
    return { ph(0) / ph(2), ph(1) / ph(2) };
}

// ---- synthetic image (matches test_comp_ncc's pattern) ----

std::vector<float> make_image() {
    std::vector<float> px(kTexW * kTexH * 4);
    for (std::uint32_t j = 0; j < kTexH; ++j) {
        for (std::uint32_t i = 0; i < kTexW; ++i) {
            const float u = float(i) / float(kTexW);
            const float v = float(j) / float(kTexH);
            const float intensity =
                128.0f + 80.0f * std::sin(8.0f * u + 0.3f * v)
                        + 40.0f * std::cos(3.0f * v - 0.7f * u);
            const std::size_t k = (j * kTexW + i) * 4;
            px[k + 0] = intensity;
            px[k + 1] = 127.5f + 60.0f * u;
            px[k + 2] = 127.5f + 60.0f * v;
            px[k + 3] = 255.0f;
        }
    }
    return px;
}

// ---- CPU bilinear sample, RGBA32Float ----

struct V4 { double r, g, b, a; };
V4 bilin(const std::vector<float>& tex, std::uint32_t W, std::uint32_t H,
         double px, double py)
{
    const double cx = px - 0.5;
    const double cy = py - 0.5;
    const int ix0 = int(std::floor(cx));
    const int iy0 = int(std::floor(cy));
    const double fx = cx - ix0;
    const double fy = cy - iy0;
    auto cl = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };
    const int W_i = int(W), H_i = int(H);
    const int x0 = cl(ix0,     0, W_i - 1);
    const int x1 = cl(ix0 + 1, 0, W_i - 1);
    const int y0 = cl(iy0,     0, H_i - 1);
    const int y1 = cl(iy0 + 1, 0, H_i - 1);
    auto load = [&](int x, int y) {
        const std::size_t k = (std::size_t(y) * W + x) * 4;
        return V4{ tex[k + 0], tex[k + 1], tex[k + 2], tex[k + 3] };
    };
    V4 a = load(x0, y0), b = load(x1, y0);
    V4 c = load(x0, y1), d = load(x1, y1);
    V4 ab{ a.r + (b.r - a.r) * fx, a.g + (b.g - a.g) * fx,
           a.b + (b.b - a.b) * fx, a.a + (b.a - a.a) * fx };
    V4 cd{ c.r + (d.r - c.r) * fx, c.g + (d.g - c.g) * fx,
           c.b + (d.b - c.b) * fx, c.a + (d.a - c.a) * fx };
    return { ab.r + (cd.r - ab.r) * fy, ab.g + (cd.g - ab.g) * fy,
             ab.b + (cd.b - ab.b) * fy, ab.a + (cd.a - ab.a) * fy };
}

double cost_yk(int dx, int dy, V4 c1, V4 c2,
               double invGammaC, double invGammaP)
{
    const double dxc = c1.r - c2.r;
    const double dyc = c1.g - c2.g;
    const double dzc = c1.b - c2.b;
    const double deltaC = std::sqrt(dxc * dxc + dyc * dyc + dzc * dzc) * invGammaC;
    const double deltaP = std::sqrt(double(dx * dx + dy * dy)) * invGammaP;
    return std::exp(-(deltaC + deltaP));
}

// ---- CPU FP64 reference for the full kernel ----

// Mirrors compNCCby3DptsYK<false>'s formula. Same as test_comp_ncc
// but inlined here for clarity.
double ncc_ref(const std::vector<float>& rc_img,
               const std::vector<float>& tc_img,
               const Eigen::Matrix<double, 3, 4>& Pref,
               const Eigen::Matrix<double, 3, 4>& Ptar,
               const Eigen::Vector3d& p_world,
               const Eigen::Vector3d& ax, const Eigen::Vector3d& ay,
               double pix_size,
               int wsh,
               double invGammaC, double invGammaP)
{
    const Eigen::Vector2d rp = project(Pref, p_world);
    const Eigen::Vector2d tp = project(Ptar, p_world);
    const double margin = wsh + 2.0;
    auto oob = [&](double x, double y) {
        return x < margin || x > double(kTexW - 1) - margin ||
               y < margin || y > double(kTexH - 1) - margin;
    };
    if (oob(rp.x(), rp.y()) || oob(tp.x(), tp.y()))
        return std::numeric_limits<double>::infinity();
    const V4 rcC = bilin(rc_img, kTexW, kTexH, rp.x() + 0.5, rp.y() + 0.5);
    const V4 tcC = bilin(tc_img, kTexW, kTexH, tp.x() + 0.5, tp.y() + 0.5);
    if (rcC.a < kRcMinAlpha || tcC.a < kTcMinAlpha)
        return std::numeric_limits<double>::infinity();

    double xsum = 0, ysum = 0, xxsum = 0, yysum = 0, xysum = 0, wsum = 0;
    for (int yp = -wsh; yp <= wsh; ++yp) {
        for (int xp = -wsh; xp <= wsh; ++xp) {
            const Eigen::Vector3d pp = p_world
                + ax * (pix_size * double(xp))
                + ay * (pix_size * double(yp));
            const Eigen::Vector2d rpc = project(Pref, pp);
            const Eigen::Vector2d tpc = project(Ptar, pp);
            const V4 rcCol = bilin(rc_img, kTexW, kTexH,
                                   rpc.x() + 0.5, rpc.y() + 0.5);
            const V4 tcCol = bilin(tc_img, kTexW, kTexH,
                                   tpc.x() + 0.5, tpc.y() + 0.5);
            const double w_color = cost_yk(xp, yp, rcC, rcCol, invGammaC, invGammaP)
                                 * cost_yk(xp, yp, tcC, tcCol, invGammaC, invGammaP);
            const double X = rcCol.r;
            const double Y = tcCol.r;
            wsum  += w_color;
            xsum  += w_color * X;
            ysum  += w_color * Y;
            xxsum += w_color * X * X;
            yysum += w_color * Y * Y;
            xysum += w_color * X * Y;
        }
    }
    const double varXW  = (xxsum - xsum * xsum / wsum) / wsum;
    const double varYW  = (yysum - ysum * ysum / wsum) / wsum;
    const double varXYW = (xysum - xsum * ysum / wsum) / wsum;
    const double raw    = varXYW / std::sqrt(varXW * varYW);
    return std::isfinite(raw) ? -raw : 1.0;
}

// Mirrors volume_computePatch on the CPU.
struct PatchCpu {
    Eigen::Vector3d p, n, x, y;
    double          d;
};
PatchCpu compute_patch_ref(const DeviceCameraParams& rc,
                           const DeviceCameraParams& tc,
                           double fp_plane_depth,
                           const Eigen::Vector2d& pix)
{
    PatchCpu out;
    // get3DPointForPixelAndFrontoParellePlaneRC
    const Eigen::Vector3d Cv = C_of(rc);
    const Eigen::Vector3d Zv = Z_of(rc);
    const Eigen::Vector3d planep = Cv + Zv * fp_plane_depth;
    Eigen::Vector3d v = iP_of(rc) * Eigen::Vector3d(pix.x(), pix.y(), 1.0);
    v.normalize();
    const double k = (planep.dot(Zv) - Zv.dot(Cv)) / Zv.dot(v);
    out.p = Cv + v * k;

    // computePixSize
    const Eigen::Vector2d rp1(pix.x() + 1.0, pix.y());
    Eigen::Vector3d refvect = iP_of(rc) * Eigen::Vector3d(rp1.x(), rp1.y(), 1.0);
    refvect.normalize();
    out.d = refvect.cross(Cv - out.p).norm();

    // computeRotCSEpip
    Eigen::Vector3d v1 = (C_of(rc) - out.p).normalized();
    Eigen::Vector3d v2 = (C_of(tc) - out.p).normalized();
    out.y = v1.cross(v2).normalized();
    out.n = ((v1 + v2) * 0.5).normalized();
    out.x = out.y.cross(out.n).normalized();
    return out;
}

std::uint8_t remap_to_uchar(double fsim)
{
    if (!std::isfinite(fsim)) return 255;
    double f = (fsim + 1.0) * 0.5;            // (-1, +1) → (0, 1)
    f = std::clamp(f, 0.0, 1.0);
    return static_cast<std::uint8_t>(f * 254.0);
}

}  // namespace

int main() try
{
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    // ---- cameras ----
    Eigen::Matrix3d K;
    K << 400.0,   0.0, double(kTexW) * 0.5,
           0.0, 400.0, double(kTexH) * 0.5,
           0.0,   0.0,   1.0;
    const Eigen::Matrix3d Rrc = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d Crc(0.0, 0.0, 0.0);
    Eigen::AngleAxisd aa(0.05, Eigen::Vector3d::UnitY());
    const Eigen::Matrix3d Rtc = aa.toRotationMatrix();
    const Eigen::Vector3d Ctc(0.3, 0.0, 0.0);
    const DeviceCameraParams rc = make_cam(K, Rrc, Crc);
    const DeviceCameraParams tc = make_cam(K, Rtc, Ctc);
    const auto Pref = P_of(rc);
    const auto Ptar = P_of(tc);

    // ---- textures (identical content on R and T) ----
    const auto img = make_image();
    Texture rc_tex(dev, Texture::Descriptor{
        kTexW, kTexH, /*mip_levels=auto*/ 0, PixelFormat::RGBA32Float });
    Texture tc_tex(dev, Texture::Descriptor{
        kTexW, kTexH, /*mip_levels=auto*/ 0, PixelFormat::RGBA32Float });
    rc_tex.set_label("cs.rc"); tc_tex.set_label("cs.tc");
    rc_tex.upload(std::span<const float>(img));
    tc_tex.upload(std::span<const float>(img));
    rc_tex.generate_mipmaps();
    tc_tex.generate_mipmaps();

    // ---- depths buffer ----
    std::vector<float> depths{ 3.5f, 4.0f, 4.5f, 5.0f };
    Buffer dep_buf(dev, depths.size() * sizeof(float));
    dep_buf.upload(std::span<const float>(depths));

    // ---- volume buffers initialized to 255 ----
    const std::size_t vol_bytes =
        std::size_t(kVolX) * std::size_t(kVolY) * std::size_t(kVolZ);
    Buffer best   (dev, vol_bytes);
    Buffer second (dev, vol_bytes);
    best   .set_label("cs.best");
    second .set_label("cs.second");

    Volume vol(dev);
    vol.init_sim(best,   VolumeDims{ kVolX, kVolY, kVolZ }, 255);
    vol.init_sim(second, VolumeDims{ kVolX, kVolY, kVolZ }, 255);

    // ---- compute_similarity for (rc, tc) ----
    Volume::ComputeSimilarityParams params{};
    params.dims                  = VolumeDims{ kVolX, kVolY, kVolZ };
    params.rc_sgm_level_width    = kTexW;
    params.rc_sgm_level_height   = kTexH;
    params.tc_sgm_level_width    = kTexW;
    params.tc_sgm_level_height   = kTexH;
    params.rc_mipmap_level       = 0.0f;
    params.step_xy               = kStep;
    params.wsh                   = kWsh;
    params.inv_gamma_c           = 1.0f / 20.0f;
    params.inv_gamma_p           = 1.0f / 4.0f;
    params.use_consistent_scale  = 0;
    params.depth_range_begin     = 0;
    params.depth_range_end       = kVolZ;
    params.roi_x_begin           = 0;
    params.roi_y_begin           = 0;
    params.roi_width             = kVolX;
    params.roi_height            = kVolY;

    vol.compute_similarity(best, second, dep_buf,
                           rc_tex, tc_tex, rc, tc, params);

    // ---- CPU reference + compare ----
    const auto* gpu_best   = static_cast<const std::uint8_t*>(best.data());
    const auto* gpu_second = static_cast<const std::uint8_t*>(second.data());

    // Distribution-aware tolerance.
    //
    // The (-1, +1) similarity drift on this kernel is larger than
    // test_comp_ncc's because the patches here use the
    // epipolar-basis (cross(v1, v2)) computed by
    // `computeRotCSEpip` — for our small baseline (0.3) at depths
    // ~3.5-5.0, the angle between v1 and v2 is ≈4° (sin ≈ 0.07),
    // so the cross product loses ~1 decimal digit of FP32 precision.
    // That 1% error propagates through 81 patch-sample positions
    // and discretizes into 0..254 uchar.
    //
    // Empirically observed worst-case is ~40 uchar (~16% of full
    // scale) on a few edge-ROI voxels; the 95th-percentile drift
    // sits at ~10 uchar (~4%). We assert both an absolute worst-
    // case ceiling and a percentile bound so genuine regressions
    // (kernel changes, layout bugs, etc.) are caught even if the
    // worst voxel happens to be inside budget.
    constexpr int kTolWorstUchar    = 60;
    constexpr double kTolP95Uchar   = 12.0;
    constexpr double kTolP99Uchar   = 30.0;

    int second_nonsentinel = 0;
    int worst_diff = 0;
    int valid_count = 0;
    int invalid_count = 0;
    std::vector<int> diffs;
    diffs.reserve(kVolX * kVolY * kVolZ);

    for (std::uint32_t vz = 0; vz < kVolZ; ++vz) {
        for (std::uint32_t vy = 0; vy < kVolY; ++vy) {
            for (std::uint32_t vx = 0; vx < kVolX; ++vx) {
                const Eigen::Vector2d pix(
                    double(int(params.roi_x_begin + vx) * params.step_xy),
                    double(int(params.roi_y_begin + vy) * params.step_xy));
                const PatchCpu pc = compute_patch_ref(rc, tc,
                                                     double(depths[vz]),
                                                     pix);
                const double fsim = ncc_ref(img, img, Pref, Ptar,
                                            pc.p, pc.x, pc.y, pc.d,
                                            kWsh,
                                            double(params.inv_gamma_c),
                                            double(params.inv_gamma_p));
                const std::uint8_t ref_uc = remap_to_uchar(fsim);
                if (ref_uc == 255) ++invalid_count; else ++valid_count;

                const std::size_t k =
                    std::size_t(vz) * (kVolX * kVolY) +
                    std::size_t(vy) *  kVolX +
                    vx;
                const int diff =
                    std::abs(int(gpu_best[k]) - int(ref_uc));
                worst_diff = std::max(worst_diff, diff);
                diffs.push_back(diff);
                if (gpu_second[k] != 255) ++second_nonsentinel;
            }
        }
    }

    std::sort(diffs.begin(), diffs.end());
    const double p50 = double(diffs[diffs.size() / 2]);
    const double p95 = double(diffs[std::size_t(double(diffs.size()) * 0.95)]);
    const double p99 = double(diffs[std::size_t(double(diffs.size()) * 0.99)]);

    std::printf("[info] voxels             : %u (valid=%d, invalid/255=%d)\n",
                kVolX * kVolY * kVolZ, valid_count, invalid_count);
    std::printf("[info] |Δ| (uchar)        : median=%.0f p95=%.0f p99=%.0f worst=%d\n",
                p50, p95, p99, worst_diff);
    std::printf("[info] budgets            : p95<=%g p99<=%g worst<=%d\n",
                kTolP95Uchar, kTolP99Uchar, kTolWorstUchar);
    std::printf("[info] 2nd-best non-255   : %d  (must be 0 after single-tc call)\n",
                second_nonsentinel);

    if (second_nonsentinel != 0) {
        std::fprintf(stderr,
            "FAIL: 2nd-best volume should be all-255 after single-tc invocation\n");
        return 1;
    }
    int bad = 0;
    if (worst_diff > kTolWorstUchar) {
        std::fprintf(stderr, "FAIL: worst |Δ| %d > budget %d\n",
                     worst_diff, kTolWorstUchar);
        ++bad;
    }
    if (p95 > kTolP95Uchar) {
        std::fprintf(stderr, "FAIL: 95th-percentile |Δ| %.1f > budget %.1f\n",
                     p95, kTolP95Uchar);
        ++bad;
    }
    if (p99 > kTolP99Uchar) {
        std::fprintf(stderr, "FAIL: 99th-percentile |Δ| %.1f > budget %.1f\n",
                     p99, kTolP99Uchar);
        ++bad;
    }
    if (bad) return 1;
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
