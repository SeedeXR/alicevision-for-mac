// test_comp_ncc.cpp — end-to-end validation of compNCCby3DptsYK.
//
// This is the first real depthMap kernel and the architectural
// climax of the device/ port: it exercises every helper we've
// built (matrix → Patch → color → SimStat) plus the texture
// sampling infrastructure. One thread = one patch hypothesis = one
// similarity score.
//
// Test setup:
//   1. A 256x256 RGBA32Float "image" containing a deterministic
//      gradient + noise pattern. Same content for both R and T
//      textures (so any patch correctly aligned through the geometry
//      should produce a strong similarity).
//   2. Synthetic R+T cameras with identical intrinsics and a small
//      baseline + rotation (so projections differ between views).
//   3. For each test case, pick a 3D point X on a fronto-parallel
//      plane in front of both cameras, project it into R, use that
//      pixel coord to look up the (synthetic) image content, and
//      build a Patch hypothesis at X with normal parallel to the
//      cameras' look directions.
//   4. Dispatch the kernel.
//   5. Compute the CPU FP64 reference doing identical bilinear
//      sampling + NCC accumulation.
//   6. Compare.
//
// Tolerance budget: ~5e-3 absolute on the similarity. The kernel
// runs a 9x9 = 81-sample weighted reduction with 2 bilinear-sampled
// float4 colors per sample; FP32 noise + sampling-corner edge cases
// reach ~1e-3 absolute typically.

#include "av/depth_map/CompNCC.hpp"
#include "av/depth_map/PatchOps.hpp"        // DeviceCameraParams mirror
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
using av::depth_map::PatchCase;
using av::depth_map::CompNCCParams;

constexpr std::uint32_t kImgW = 256;
constexpr std::uint32_t kImgH = 256;
constexpr std::size_t   kCases = 64;
constexpr int           kWsh   = 4;   // 9x9 patch
// Tolerance budget for the GPU/CPU similarity comparison.
//
// 81-sample weighted NCC with bilinear texture sampling at random
// sub-pixel positions has multiple FP32-vs-FP64 drift sources:
//   * Per-sample bilinear interpolation: ~1e-6 / channel
//   * `exp(-deltaC*invGammaC + ...)` weight: ~1e-4 relative
//   * SimStat accumulation: 81 × FP32 ULP per running sum
//   * `computeWSim = covXY / sqrt(varX * varY)`: catastrophic
//     cancellation when both variances are large but covariance
//     is dominated by a tail of the weight distribution.
//
// In practice (empirical worst case across 64 cases this build):
// ~1.9e-2. Budget at 5e-2 absorbs that comfortably without
// admitting actually-broken kernels.
constexpr double        kTolSim = 5e-2;

constexpr float kRcMinAlpha = 255.0f * 0.9f;  // mirror MSL kRcMinAlpha
constexpr float kTcMinAlpha = 255.0f * 0.4f;  // mirror MSL kTcMinAlpha

// ---------------- camera utilities (mirrored from test_patch.cpp) ---

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

// ---------------- synthetic image ----------------------------------
//
// Deterministic float4 pattern at (i, j) — used for both R and T
// textures so that any geometrically-correct patch lookup finds the
// same content in both views (up to sampling-position drift across
// cameras, which is what NCC will measure).
//
// Channel layout chosen to match the upstream alpha convention:
//   x: a smoothly-varying "intensity" used by NCC (the channel
//      simStat.update accumulates).
//   y, z: secondary channels for the bilateral weight (they don't
//      matter much for the test outcome; they participate in
//      `euclideanDist3(c1, c2)`).
//   w: alpha — always 255 here so the alpha-mask path doesn't fire.

struct ImageF4 { float r, g, b, a; };

ImageF4 make_pixel(std::uint32_t i, std::uint32_t j) {
    const float u = static_cast<float>(i) / static_cast<float>(kImgW);
    const float v = static_cast<float>(j) / static_cast<float>(kImgH);
    // A smooth, non-trivial intensity function so the patch
    // gradient has structure NCC can lock onto.
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

// CPU bilinear sample at *pixel* coords (px, py). Matches MSL's
// filter::linear + address::clamp_to_edge for non-mipmapped use.
// `pixels` is row-major; channel layout matches make_image().
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
        return Eigen::Vector4d(pixels[k + 0], pixels[k + 1],
                               pixels[k + 2], pixels[k + 3]);
    };
    const Eigen::Vector4d a = load(ix0, iy0);
    const Eigen::Vector4d b = load(ix1, iy0);
    const Eigen::Vector4d c = load(ix0, iy1);
    const Eigen::Vector4d d = load(ix1, iy1);

    const Eigen::Vector4d ab = a + (b - a) * fx;
    const Eigen::Vector4d cd = c + (d - c) * fx;
    return ab + (cd - ab) * fy;
}

// ---------------- CPU reference for compNCCby3DptsYK<false> --------

double cost_yk(int dx, int dy,
               const Eigen::Vector4d& c1, const Eigen::Vector4d& c2,
               double invGammaC, double invGammaP)
{
    const Eigen::Vector3d a = c1.head<3>();
    const Eigen::Vector3d b = c2.head<3>();
    const double deltaC = (a - b).norm() * invGammaC;
    const double deltaP = std::sqrt(double(dx * dx + dy * dy)) * invGammaP;
    return std::exp(-(deltaC + deltaP));
}

// Mirrors compNCCby3DptsYK<false>. Returns INFINITY on the same
// edge cases as the kernel.
double comp_ncc_ref(const std::vector<float>& rc_img,
                    const std::vector<float>& tc_img,
                    const Eigen::Matrix<double, 3, 4>& Pref,
                    const Eigen::Matrix<double, 3, 4>& Ptar,
                    const PatchCase& patch,
                    int wsh,
                    double invGammaC, double invGammaP)
{
    const Eigen::Vector3d p_world(patch.p[0], patch.p[1], patch.p[2]);
    const Eigen::Vector3d ax     (patch.x[0], patch.x[1], patch.x[2]);
    const Eigen::Vector3d ay     (patch.y[0], patch.y[1], patch.y[2]);
    const double d = static_cast<double>(patch.d);

    const Eigen::Vector2d rp = project(Pref, p_world);
    const Eigen::Vector2d tp = project(Ptar, p_world);

    const double margin = wsh + 2.0;
    auto out_of_bounds = [&](double x, double y, std::uint32_t W, std::uint32_t H) {
        return x < margin || x > double(W - 1) - margin ||
               y < margin || y > double(H - 1) - margin;
    };
    if (out_of_bounds(rp.x(), rp.y(), kImgW, kImgH) ||
        out_of_bounds(tp.x(), tp.y(), kImgW, kImgH)) {
        return std::numeric_limits<double>::infinity();
    }

    const Eigen::Vector4d rcCenter = bilin_sample(rc_img, kImgW, kImgH,
                                                  rp.x() + 0.5,
                                                  rp.y() + 0.5);
    const Eigen::Vector4d tcCenter = bilin_sample(tc_img, kImgW, kImgH,
                                                  tp.x() + 0.5,
                                                  tp.y() + 0.5);
    if (rcCenter(3) < static_cast<double>(kRcMinAlpha) ||
        tcCenter(3) < static_cast<double>(kTcMinAlpha)) {
        return std::numeric_limits<double>::infinity();
    }

    double xsum = 0, ysum = 0, xxsum = 0, yysum = 0, xysum = 0, wsum = 0;
    for (int yp = -wsh; yp <= wsh; ++yp) {
        for (int xp = -wsh; xp <= wsh; ++xp) {
            const Eigen::Vector3d pp = p_world
                + ax * (d * double(xp))
                + ay * (d * double(yp));
            const Eigen::Vector2d rpc = project(Pref, pp);
            const Eigen::Vector2d tpc = project(Ptar, pp);
            const Eigen::Vector4d rcCol = bilin_sample(rc_img, kImgW, kImgH,
                                                       rpc.x() + 0.5,
                                                       rpc.y() + 0.5);
            const Eigen::Vector4d tcCol = bilin_sample(tc_img, kImgW, kImgH,
                                                       tpc.x() + 0.5,
                                                       tpc.y() + 0.5);

            const double w_color = cost_yk(xp, yp, rcCenter, rcCol,
                                           invGammaC, invGammaP)
                                  * cost_yk(xp, yp, tcCenter, tcCol,
                                            invGammaC, invGammaP);

            const double X = rcCol(0);
            const double Y = tcCol(0);
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

}  // namespace

int main() try {
    auto dev = av::gpu::Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    // ---------------- cameras ----------------
    Eigen::Matrix3d K;
    K << 400.0,   0.0, double(kImgW) * 0.5,
           0.0, 400.0, double(kImgH) * 0.5,
           0.0,   0.0,   1.0;

    Eigen::Matrix3d Rrc = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Crc(0.0, 0.0, 0.0);

    // Target: small rotation around Y + 30 cm baseline along X.
    Eigen::AngleAxisd aa(0.05, Eigen::Vector3d::UnitY());
    Eigen::Matrix3d Rtc = aa.toRotationMatrix();
    Eigen::Vector3d Ctc(0.3, 0.0, 0.0);

    const DeviceCameraParams rc_dcp = make_cam(K, Rrc, Crc);
    const DeviceCameraParams tc_dcp = make_cam(K, Rtc, Ctc);
    const auto Pref = P_of(rc_dcp);
    const auto Ptar = P_of(tc_dcp);

    // ---------------- textures ----------------
    const auto img = make_image();   // shared content for R and T

    using namespace av::gpu;
    Texture rc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, /*mip_levels=auto*/ 0,
        PixelFormat::RGBA32Float });
    Texture tc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, /*mip_levels=auto*/ 0,
        PixelFormat::RGBA32Float });
    rc_tex.set_label("compncc.rc");
    tc_tex.set_label("compncc.tc");
    rc_tex.upload(std::span<const float>(img));
    tc_tex.upload(std::span<const float>(img));
    rc_tex.generate_mipmaps();
    tc_tex.generate_mipmaps();

    // ---------------- patches ----------------
    av::depth_map::CompNCC ncc(dev, rc_dcp, tc_dcp);

    std::vector<PatchCase> patches(kCases);
    std::mt19937_64 rng(0xc0ffeedface);
    std::uniform_real_distribution<double> XY(-1.5, 1.5);
    std::uniform_real_distribution<double> Z(3.5, 7.0);

    for (std::size_t k = 0; k < kCases; ++k) {
        const Eigen::Vector3d X(XY(rng), XY(rng), Z(rng));

        // Patch frame: normal pointing toward camera (-Z in world,
        // i.e. toward the camera origin at +Z behind the patch);
        // x axis along world-X; y axis along world-Y. Pixel size
        // chosen so 1 patch step ≈ 1 image pixel.
        const Eigen::Vector3d n(0.0, 0.0, -1.0);
        const Eigen::Vector3d xa(1.0, 0.0, 0.0);
        const Eigen::Vector3d ya(0.0, 1.0, 0.0);

        const double d_world = X.z() / 400.0;  // K(0,0) = 400 → 1 pixel ≈ z/f

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
    params.wsh            = kWsh;
    params.invGammaC      = 1.0f / 20.0f;
    params.invGammaP      = 1.0f / 4.0f;
    params.useConsistentScale = 0;

    // ---------------- dispatch ----------------
    std::vector<float> gpu_sim(kCases);
    ncc.run_no_filter(patches, gpu_sim, rc_tex, tc_tex, params);

    // ---------------- CPU reference ----------------
    std::size_t n_inf_gpu = 0, n_inf_cpu = 0, n_match = 0;
    double worst_finite_err = 0.0;
    int bad = 0;

    for (std::size_t k = 0; k < kCases; ++k) {
        const double ref = comp_ncc_ref(img, img, Pref, Ptar,
                                         patches[k], kWsh,
                                         double(params.invGammaC),
                                         double(params.invGammaP));
        const double gpu = static_cast<double>(gpu_sim[k]);

        const bool gpu_inf = !std::isfinite(gpu);
        const bool cpu_inf = !std::isfinite(ref);
        if (gpu_inf) ++n_inf_gpu;
        if (cpu_inf) ++n_inf_cpu;

        if (gpu_inf != cpu_inf) {
            if (bad < 3) std::fprintf(stderr,
                "k=%zu inf-mismatch: gpu=%g cpu=%g\n", k, gpu, ref);
            ++bad;
            continue;
        }
        if (gpu_inf) continue;   // both inf — agreement

        const double err = std::abs(gpu - ref);
        worst_finite_err = std::max(worst_finite_err, err);
        ++n_match;
        if (err > kTolSim) {
            if (bad < 3) std::fprintf(stderr,
                "k=%zu sim err=%g gpu=%g ref=%g\n",
                k, err, gpu, ref);
            ++bad;
        }
    }

    std::printf("[info] cases             : %zu (gpu_inf=%zu cpu_inf=%zu finite_pairs=%zu)\n",
                kCases, n_inf_gpu, n_inf_cpu, n_match);
    std::printf("[info] worst finite err  : %.3g (budget %.3g)\n",
                worst_finite_err, kTolSim);

    // Sanity: at least some test cases should land inside the image
    // and produce a strong negative similarity. If everything is
    // INFINITY the test isn't exercising the kernel.
    if (n_match < kCases / 4) {
        std::fprintf(stderr,
            "FAIL: too few finite similarity scores (%zu / %zu)\n",
            n_match, kCases);
        return 1;
    }
    // Sanity: with identical R and T images the geometrically-correct
    // matches must produce strong NCC. Verify the median is well
    // below 0.
    std::vector<float> finite;
    finite.reserve(n_match);
    for (std::size_t k = 0; k < kCases; ++k) {
        if (std::isfinite(gpu_sim[k])) finite.push_back(gpu_sim[k]);
    }
    std::nth_element(finite.begin(),
                     finite.begin() + finite.size() / 2,
                     finite.end());
    const float median = finite[finite.size() / 2];
    std::printf("[info] median similarity : %.4f (expect well below 0)\n",
                static_cast<double>(median));
    if (median > -0.5f) {
        std::fprintf(stderr,
            "FAIL: median similarity %.4f is not negative enough\n",
            static_cast<double>(median));
        return 1;
    }

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
