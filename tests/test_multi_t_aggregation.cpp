// test_multi_t_aggregation.cpp — multi-T-camera aggregation
// validation for `compute_similarity` and `refine_similarity`.
//
// Both kernels accumulate across multiple (R, T) calls. The
// integration tests so far have only run single-T pairs. This
// test exercises the multi-call aggregation paths.
//
// Setup: R + 3 T cameras viewing the same plane (different
// baselines/rotations); per T, generate T_i = Warp_H_i(R).
//
// ===== compute_similarity (WTA) ==================================
//
// Invariant after K T calls with per-voxel sims s_1..s_K:
//     best[v] == min(s_1, ..., s_K)
//     2nd [v] == 2nd-smallest({s_1, ..., s_K, 255})
//
// Verification: run each T alone (capturing best_T_i / 2nd_T_i),
// then run all three sequentially and check the invariant
// voxel-by-voxel.
//
// ===== refine_similarity (additive) ==============================
//
// Per call, the kernel does promote-add-demote on the half
// volume:  half ← FP16(float(half) + float(sim_new)).
// After K calls: half_K = FP16(... FP16(FP16(0 + s_1) + s_2) ...).
// The FP16 rounding chain is order-dependent; we replicate it on
// the CPU using the IEEE 754 binary16 round-to-nearest-even
// helpers from the existing test_refine_similarity / test_volume_easy.

#include "av/depth_map/PatchOps.hpp"
#include "av/depth_map/Volume.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using av::depth_map::DeviceCameraParams;
using av::depth_map::Volume;
using av::depth_map::VolumeDims;

constexpr std::uint32_t kImgW = 128;
constexpr std::uint32_t kImgH = 96;
constexpr std::uint32_t kVolX = 16;
constexpr std::uint32_t kVolY = 12;
constexpr std::uint32_t kVolZ = 7;
constexpr int           kStep = 8;
constexpr int           kWsh  = 3;
constexpr float         kTruthZ  = 4.0f;
constexpr float         kDepth0  = 3.7f;
constexpr float         kDepthDz = 0.10f;
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
            const float r = 128.0f + 60.0f * std::sin(15.0f*u + 2.0f*v)
                                   + 40.0f * std::cos(11.0f*v - 3.0f*u);
            const float g = 128.0f + 50.0f * std::cos(13.0f*v);
            const float b = 128.0f + 45.0f * std::sin(17.0f*(u+v));
            const std::size_t k = (j * kImgW + i) * 4;
            px[k+0] = std::clamp(r, 0.0f, 255.0f);
            px[k+1] = std::clamp(g, 0.0f, 255.0f);
            px[k+2] = std::clamp(b, 0.0f, 255.0f);
            px[k+3] = 255.0f;
        }
    return px;
}

Eigen::Matrix3d plane_homography(
    const Eigen::Matrix3d& K,
    const Eigen::Matrix3d& Rrc, const Eigen::Vector3d& Crc,
    const Eigen::Matrix3d& Rtc, const Eigen::Vector3d& Ctc,
    const Eigen::Vector3d& pp,  const Eigen::Vector3d& pn)
{
    const Eigen::Vector3d _tl = -Rrc * Crc;
    const Eigen::Vector3d _tr = -Rtc * Ctc;
    const Eigen::Matrix3d Rr  = Rtc * Rrc.transpose();
    const Eigen::Vector3d tr  = _tr - Rr * _tl;
    Eigen::Vector3d n_cam = Rrc * pn; n_cam.normalize();
    const Eigen::Vector3d p_cam = Rrc * (pp - Crc);
    const double d_ref = -n_cam.dot(p_cam);
    return K * (Rr - (tr * n_cam.transpose()) / d_ref) * K.inverse();
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

// IEEE binary16 helpers (reused pattern from earlier tests).
float half_bits_to_float(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    const std::uint32_t e    = (h >> 10) & 0x1fu;
    const std::uint32_t m    =  h & 0x3ffu;
    if (e == 0) {
        if (m == 0) return std::bit_cast<float>(sign);
        int ex = -1; std::uint32_t mm = m;
        do { mm <<= 1; ++ex; } while ((mm & 0x400u) == 0);
        const std::uint32_t fi = sign
            | (std::uint32_t(127 - 15 - ex) << 23)
            | ((mm & 0x3ffu) << 13);
        return std::bit_cast<float>(fi);
    }
    if (e == 31) return std::bit_cast<float>(sign | 0x7f800000u | (m << 13));
    const std::uint32_t fi = sign | ((e + 127 - 15) << 23) | (m << 13);
    return std::bit_cast<float>(fi);
}
std::uint16_t float_to_half_bits(float f) {
    const std::uint32_t fi = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t s  = (fi >> 16) & 0x8000u;
    std::int32_t       ex  = std::int32_t((fi >> 23) & 0xffu) - 127 + 15;
    std::uint32_t      mant = fi & 0x7fffffu;
    if (ex <= 0) {
        if (ex < -10) return std::uint16_t(s);
        mant |= 0x800000u;
        const std::uint32_t shift = std::uint32_t(14 - ex);
        const std::uint32_t round_bit = 1u << (shift - 1);
        std::uint32_t hm = mant >> shift;
        if ((mant & ((1u << shift) - 1u)) > round_bit ||
            ((mant & ((1u << shift) - 1u)) == round_bit && (hm & 1u))) ++hm;
        return std::uint16_t(s | hm);
    }
    if (ex >= 31) {
        if (((fi >> 23) & 0xffu) == 0xffu && mant != 0)
            return std::uint16_t(s | 0x7c00u | (mant >> 13) | 1u);
        return std::uint16_t(s | 0x7c00u);
    }
    const std::uint32_t round_bit = 1u << 12;
    std::uint32_t hm = mant >> 13;
    if ((mant & 0x1fffu) > round_bit ||
        ((mant & 0x1fffu) == round_bit && (hm & 1u))) ++hm;
    if (hm == 0x400u) { hm = 0; ex += 1;
                        if (ex >= 31) return std::uint16_t(s | 0x7c00u); }
    return std::uint16_t(s | (std::uint32_t(ex) << 10) | hm);
}

}  // namespace

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    static_assert(kVolX * kStep == kImgW, "vol step");
    static_assert(kVolY * kStep == kImgH, "vol step");

    // ---- cameras ----
    Eigen::Matrix3d K;
    K << kFx, 0.0, double(kCx),
         0.0, kFy, double(kCy),
         0.0, 0.0, 1.0;
    const Eigen::Matrix3d Rrc = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d Crc(0.0, 0.0, 0.0);
    const DeviceCameraParams rc = make_cam(K, Rrc, Crc);

    // Three T cameras at different baselines/rotations.
    struct TC { Eigen::Vector3d C; double y_rad; };
    const std::array<TC, 3> tcs = {{
        { Eigen::Vector3d( 0.25, 0.0, 0.0),  0.03 },
        { Eigen::Vector3d( 0.40, 0.0, 0.0), -0.02 },
        { Eigen::Vector3d(-0.30, 0.0, 0.0),  0.04 },
    }};

    // ---- R + T images ----
    const auto r_img = make_r_image();
    Texture rc_tex(dev, Texture::Descriptor{
        kImgW, kImgH, 0, PixelFormat::RGBA32Float });
    rc_tex.upload(std::span<const float>(r_img));
    rc_tex.generate_mipmaps();

    std::array<Texture, 3> tc_texs = {
        Texture(dev, Texture::Descriptor{kImgW, kImgH, 0, PixelFormat::RGBA32Float}),
        Texture(dev, Texture::Descriptor{kImgW, kImgH, 0, PixelFormat::RGBA32Float}),
        Texture(dev, Texture::Descriptor{kImgW, kImgH, 0, PixelFormat::RGBA32Float})
    };
    std::array<DeviceCameraParams, 3> tcps;

    const Eigen::Vector3d plane_pt(0.0, 0.0, double(kTruthZ));
    const Eigen::Vector3d plane_nm(0.0, 0.0, -1.0);
    for (int i = 0; i < 3; ++i) {
        const Eigen::AngleAxisd aa(tcs[i].y_rad, Eigen::Vector3d::UnitY());
        const Eigen::Matrix3d Rtc = aa.toRotationMatrix();
        tcps[i] = make_cam(K, Rtc, tcs[i].C);
        const Eigen::Matrix3d H = plane_homography(K, Rrc, Crc,
            Rtc, tcs[i].C, plane_pt, plane_nm);
        const auto t_img = warp(r_img, H.inverse());
        tc_texs[i].upload(std::span<const float>(t_img));
        tc_texs[i].generate_mipmaps();
    }

    // ---- depths buffer ----
    std::vector<float> depths(kVolZ);
    for (std::uint32_t z = 0; z < kVolZ; ++z)
        depths[z] = kDepth0 + float(z) * kDepthDz;
    Buffer dep_buf(dev, depths.size() * sizeof(float));
    dep_buf.upload(std::span<const float>(depths));

    const VolumeDims dims{ kVolX, kVolY, kVolZ };
    const std::size_t vox = dims.voxel_count();

    Volume vol(dev);

    // =================================================================
    //  Test 1 — compute_similarity WTA
    // =================================================================

    // Per-T isolated runs.
    std::array<std::vector<std::uint8_t>, 3> per_t_best;
    std::array<std::vector<std::uint8_t>, 3> per_t_2nd;
    for (int i = 0; i < 3; ++i) {
        Buffer best  (dev, vox);
        Buffer second(dev, vox);
        vol.init_sim(best,   dims, 255);
        vol.init_sim(second, dims, 255);
        Volume::ComputeSimilarityParams cs{};
        cs.dims                  = dims;
        cs.rc_sgm_level_width    = kImgW; cs.rc_sgm_level_height = kImgH;
        cs.tc_sgm_level_width    = kImgW; cs.tc_sgm_level_height = kImgH;
        cs.rc_mipmap_level       = 0.0f;
        cs.step_xy               = kStep;
        cs.wsh                   = kWsh;
        cs.inv_gamma_c           = 1.0f / 20.0f;
        cs.inv_gamma_p           = 1.0f / 4.0f;
        cs.use_consistent_scale  = 0;
        cs.depth_range_begin     = 0;
        cs.depth_range_end       = kVolZ;
        cs.roi_x_begin           = 0;     cs.roi_y_begin = 0;
        cs.roi_width             = kVolX; cs.roi_height  = kVolY;
        vol.compute_similarity(best, second, dep_buf,
            rc_tex, tc_texs[i], rc, tcps[i], cs);
        per_t_best[i].assign(static_cast<const std::uint8_t*>(best.data()),
                              static_cast<const std::uint8_t*>(best.data()) + vox);
        per_t_2nd [i].assign(static_cast<const std::uint8_t*>(second.data()),
                              static_cast<const std::uint8_t*>(second.data()) + vox);
    }

    // Multi-T aggregated run.
    std::vector<std::uint8_t> multi_best, multi_2nd;
    {
        Buffer best  (dev, vox);
        Buffer second(dev, vox);
        vol.init_sim(best,   dims, 255);
        vol.init_sim(second, dims, 255);
        Volume::ComputeSimilarityParams cs{};
        cs.dims                  = dims;
        cs.rc_sgm_level_width    = kImgW; cs.rc_sgm_level_height = kImgH;
        cs.tc_sgm_level_width    = kImgW; cs.tc_sgm_level_height = kImgH;
        cs.rc_mipmap_level       = 0.0f;
        cs.step_xy               = kStep;
        cs.wsh                   = kWsh;
        cs.inv_gamma_c           = 1.0f / 20.0f;
        cs.inv_gamma_p           = 1.0f / 4.0f;
        cs.use_consistent_scale  = 0;
        cs.depth_range_begin     = 0;
        cs.depth_range_end       = kVolZ;
        cs.roi_x_begin           = 0;     cs.roi_y_begin = 0;
        cs.roi_width             = kVolX; cs.roi_height  = kVolY;
        for (int i = 0; i < 3; ++i) {
            vol.compute_similarity(best, second, dep_buf,
                rc_tex, tc_texs[i], rc, tcps[i], cs);
        }
        multi_best.assign(static_cast<const std::uint8_t*>(best.data()),
                           static_cast<const std::uint8_t*>(best.data()) + vox);
        multi_2nd .assign(static_cast<const std::uint8_t*>(second.data()),
                           static_cast<const std::uint8_t*>(second.data()) + vox);
    }

    // Validate WTA invariant.
    int wta_bad = 0;
    for (std::size_t v = 0; v < vox; ++v) {
        // Combine per-T values + the 255 sentinel.
        std::array<int, 4> vals = {
            int(per_t_best[0][v]),
            int(per_t_best[1][v]),
            int(per_t_best[2][v]),
            255
        };
        std::sort(vals.begin(), vals.end());
        const int exp_best = vals[0];
        const int exp_2nd  = vals[1];
        if (int(multi_best[v]) != exp_best || int(multi_2nd[v]) != exp_2nd) {
            if (wta_bad < 4) std::fprintf(stderr,
                "WTA: v=%zu got=(%d, %d) expected=(%d, %d) per_t=(%d, %d, %d)\n",
                v, int(multi_best[v]), int(multi_2nd[v]),
                exp_best, exp_2nd,
                int(per_t_best[0][v]), int(per_t_best[1][v]), int(per_t_best[2][v]));
            ++wta_bad;
        }
    }
    std::printf("[wta ] voxels=%zu  bad=%d\n", vox, wta_bad);

    // =================================================================
    //  Test 2 — refine_similarity additive
    // =================================================================
    // Build a synthetic per-pixel SGM (depth, pix_size) map and run
    // refine_similarity once per T, accumulating into the half volume.
    const std::size_t pix = std::size_t(kVolX) * std::size_t(kVolY);
    std::vector<float> sgm_dp(pix * 2);
    for (std::uint32_t y = 0; y < kVolY; ++y)
        for (std::uint32_t x = 0; x < kVolX; ++x) {
            const std::size_t k = std::size_t(y) * kVolX + x;
            // Set SGM mid depth to the analytical truth at this pixel
            // for the plane Z=4.0 — same trick as test_refine_pipeline.
            const double px = (double(x) + 0.5) * double(kStep);
            const double py = (double(y) + 0.5) * double(kStep);
            const double X = (px - double(kCx)) * double(kTruthZ) / double(kFx);
            const double Y = (py - double(kCy)) * double(kTruthZ) / double(kFy);
            const double truth = std::sqrt(X*X + Y*Y + kTruthZ*kTruthZ);
            sgm_dp[k * 2 + 0] = float(truth);
            sgm_dp[k * 2 + 1] = 0.01f;
        }
    Buffer sgm_buf(dev, sgm_dp.size() * sizeof(float));
    sgm_buf.upload(std::span<const float>(sgm_dp));

    // Per-T isolated half volumes.
    std::array<std::vector<std::uint16_t>, 3> per_t_half;
    for (int i = 0; i < 3; ++i) {
        Buffer half_buf(dev, vox * sizeof(std::uint16_t));
        vol.init_refine(half_buf, dims, 0.0f);
        Volume::RefineSimilarityParams rs{};
        rs.dims                   = dims;
        rs.rc_refine_level_width  = kImgW; rs.rc_refine_level_height = kImgH;
        rs.tc_refine_level_width  = kImgW; rs.tc_refine_level_height = kImgH;
        rs.rc_mipmap_level        = 0.0f;
        rs.step_xy                = kStep;
        rs.wsh                    = kWsh;
        rs.inv_gamma_c            = 1.0f / 20.0f;
        rs.inv_gamma_p            = 1.0f / 4.0f;
        rs.use_consistent_scale   = 0;
        rs.depth_range_begin      = 0;
        rs.depth_range_end        = kVolZ;
        rs.roi_x_begin            = 0;     rs.roi_y_begin           = 0;
        rs.roi_width              = kVolX; rs.roi_height            = kVolY;
        vol.refine_similarity(half_buf, sgm_buf,
            rc_tex, tc_texs[i], rc, tcps[i], rs);
        per_t_half[i].assign(
            static_cast<const std::uint16_t*>(half_buf.data()),
            static_cast<const std::uint16_t*>(half_buf.data()) + vox);
    }

    // Multi-T aggregated half volume.
    std::vector<std::uint16_t> multi_half;
    {
        Buffer half_buf(dev, vox * sizeof(std::uint16_t));
        vol.init_refine(half_buf, dims, 0.0f);
        Volume::RefineSimilarityParams rs{};
        rs.dims                   = dims;
        rs.rc_refine_level_width  = kImgW; rs.rc_refine_level_height = kImgH;
        rs.tc_refine_level_width  = kImgW; rs.tc_refine_level_height = kImgH;
        rs.rc_mipmap_level        = 0.0f;
        rs.step_xy                = kStep;
        rs.wsh                    = kWsh;
        rs.inv_gamma_c            = 1.0f / 20.0f;
        rs.inv_gamma_p            = 1.0f / 4.0f;
        rs.use_consistent_scale   = 0;
        rs.depth_range_begin      = 0;
        rs.depth_range_end        = kVolZ;
        rs.roi_x_begin            = 0;     rs.roi_y_begin            = 0;
        rs.roi_width              = kVolX; rs.roi_height             = kVolY;
        for (int i = 0; i < 3; ++i) {
            vol.refine_similarity(half_buf, sgm_buf,
                rc_tex, tc_texs[i], rc, tcps[i], rs);
        }
        multi_half.assign(
            static_cast<const std::uint16_t*>(half_buf.data()),
            static_cast<const std::uint16_t*>(half_buf.data()) + vox);
    }

    // Validate: half volume should equal the FP16-stepped sum of the
    // per-T half volumes. Mirror upstream's promote-add-demote pattern.
    int   add_bad = 0;
    float worst_add_diff = 0.0f;
    for (std::size_t v = 0; v < vox; ++v) {
        // Accumulate in FP32 then re-quantize to FP16 between adds
        // to match the kernel's per-call rounding chain.
        float acc = 0.0f;
        for (int i = 0; i < 3; ++i) {
            const float contrib = half_bits_to_float(per_t_half[i][v]);
            acc = half_bits_to_float(float_to_half_bits(acc + contrib));
        }
        const float gpu_f = half_bits_to_float(multi_half[v]);
        const float diff  = std::fabs(gpu_f - acc);
        worst_add_diff = std::max(worst_add_diff, diff);
        // Tolerate up to 2 half ULPs (~0.001 at magnitude ~1).
        // The per-T half stored values are the *output* of the kernel
        // (which may have a small per-T FP32 NCC noise vs the kernel's
        // own FP32 accumulator that holds the same chained sum). So 2
        // ULP is a sane budget.
        if (diff > 2e-3f) {
            if (add_bad < 4) std::fprintf(stderr,
                "ADD: v=%zu gpu=%g ref=%g diff=%.3g\n",
                v, gpu_f, acc, diff);
            ++add_bad;
        }
    }
    std::printf("[add ] voxels=%zu  worst |Δ|=%.3g  bad=%d\n",
                vox, worst_add_diff, add_bad);

    int failed = wta_bad + add_bad;
    if (failed) {
        std::fprintf(stderr, "FAIL: WTA=%d ADD=%d\n", wta_bad, add_bad);
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
