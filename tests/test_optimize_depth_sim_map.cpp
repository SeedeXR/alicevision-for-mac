// test_optimize_depth_sim_map.cpp — validation of the
// gradient-descent fusion pipeline (`optimize_*` trio in
// `deviceDepthSimilarityMapKernels.cuh`).
//
// Strategy: run the full host orchestration (variance map +
// N-iter loop) on a small synthetic scene and compare the
// per-pixel result against a CPU FP32 reference replicating the
// same arithmetic. A small relative budget absorbs sub-ULP drift
// from the GPU's fast-math `exp()` inside the sigmoid path.
//
// Scene:
//   * 32 × 24 ROI.
//   * SGM map: uniform (depth=4.0, pixSize=0.01).
//   * Refine map: uniform (depth=4.05, sim=-0.95) — slightly
//     ahead of SGM so the fine_sim_weight contributes.
//   * rc texture: uniform L=128 → gradient = 0 → img_var = 0.
//   * Camera: simple K (fx=200, principal at center), identity R,
//     origin C.
//   * 5 iterations.
//
// Expected behavior:
//   * close_to_rough_weight ≈ 0 (depth_opt stays near SGM).
//   * (1 − close_to_rough_weight) ≈ 1 → the smooth/fine blend
//     contributes.
//   * With smoothStep ≈ 0 (uniform plane) and step_to_fine
//     clamped to +pixSize/10, depth_opt drifts slowly toward
//     refine_depth.
//   * No NaN; no divergence.

#include "av/depth_map/DepthSimMap.hpp"
#include "av/depth_map/PatchOps.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Texture.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using av::depth_map::DeviceCameraParams;

constexpr std::uint32_t kW    = 32;
constexpr std::uint32_t kH    = 24;
constexpr float         kFx   = 200.0f, kFy = 200.0f;
constexpr float         kCx   = float(kW) * 0.5f;
constexpr float         kCy   = float(kH) * 0.5f;
constexpr float         kSgmDepth     = 4.0f;
constexpr float         kSgmPixSize   = 0.01f;
constexpr float         kRefineDepth  = 4.05f;
constexpr float         kRefineSim    = -0.95f;
constexpr int           kNbIter       = 5;

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
DeviceCameraParams make_cam() {
    Eigen::Matrix3d K;
    K << kFx, 0.0, double(kCx),
         0.0, kFy, double(kCy),
         0.0, 0.0, 1.0;
    const Eigen::Matrix3d R   = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d C   (0.0, 0.0, 0.0);
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

// CPU-side FP32 mirror of the kernel arithmetic.
//
// We only need to replicate the parts the test fixture exercises:
//   * Variance is 0 everywhere (constant rc texture).
//   * Depth texture = current out_opt[i].x (nearest pixel sample).
//   * smoothStep is ~0 (uniform-depth plane), energy is 180 by
//     default since with all neighbors at the same depth, the
//     angle is 180 (collinear) → 180 − 180 = 0, but we have
//     n=2 valid neighbor pairs so energy = 0 (not 180).

float sigmoid_cpu(float zero, float endv, float w, float mid, float x) {
    return zero + (endv - zero) * (1.0f / (1.0f + std::exp(10.0f * ((x - mid) / w))));
}
float sigmoid2_cpu(float zero, float endv, float w, float mid, float x) {
    return zero + (endv - zero) * (1.0f / (1.0f + std::exp(10.0f * ((mid - x) / w))));
}

struct V3 { float x, y, z; };
inline V3 add(V3 a, V3 b)         { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline V3 sub(V3 a, V3 b)         { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline V3 mul(V3 a, float s)      { return {a.x*s, a.y*s, a.z*s}; }
inline float dot3(V3 a, V3 b)     { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float len3(V3 a)           { return std::sqrt(dot3(a, a)); }
inline V3 nrm3(V3 a)              { float l = len3(a); return {a.x/l, a.y/l, a.z/l}; }

inline V3 mat3xv2(const float* iP, float u, float v) {
    // iP is column-major 3x3 stored as 9 floats.
    return {
        iP[0]*u + iP[3]*v + iP[6],
        iP[1]*u + iP[4]*v + iP[7],
        iP[2]*u + iP[5]*v + iP[8]
    };
}

V3 get_3d(const DeviceCameraParams& cam, float u, float v, float depth) {
    V3 dir = mat3xv2(cam.iP, u, v);
    dir = nrm3(dir);
    V3 C = { cam.C[0], cam.C[1], cam.C[2] };
    return add(C, mul(dir, depth));
}

float angle_deg(V3 A, V3 B, V3 C) {
    V3 v1 = nrm3(sub(B, A));
    V3 v2 = nrm3(sub(C, A));
    float c = std::max(-1.0f, std::min(1.0f, dot3(v1, v2)));
    return std::fabs(std::acos(c)) * (180.0f / float(M_PI));
}

float copysign_f(float a, float b) { return std::copysign(a, b); }

}  // namespace

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    DeviceCameraParams cam = make_cam();

    const std::size_t pix = std::size_t(kW) * std::size_t(kH);

    // ---- SGM map ----
    std::vector<float> sgm(pix * 2);
    std::vector<float> refn(pix * 2);
    for (std::size_t k = 0; k < pix; ++k) {
        sgm [k * 2 + 0] = kSgmDepth;
        sgm [k * 2 + 1] = kSgmPixSize;
        refn[k * 2 + 0] = kRefineDepth;
        refn[k * 2 + 1] = kRefineSim;
    }

    Buffer sgm_buf  (dev, sgm.size()  * sizeof(float));
    Buffer ref_buf  (dev, refn.size() * sizeof(float));
    Buffer out_buf  (dev, pix * 2 * sizeof(float));
    sgm_buf.upload(std::span<const float>(sgm));
    ref_buf.upload(std::span<const float>(refn));

    // ---- rc mipmap texture: uniform L=128 → gradient 0 → variance 0
    std::vector<float> rc_img(pix * 4);
    for (std::size_t k = 0; k < pix; ++k) {
        rc_img[k * 4 + 0] = 128.0f;
        rc_img[k * 4 + 1] = 128.0f;
        rc_img[k * 4 + 2] = 128.0f;
        rc_img[k * 4 + 3] = 255.0f;
    }
    Texture rc_tex(dev, Texture::Descriptor{
        kW, kH, 0, PixelFormat::RGBA32Float });
    rc_tex.upload(std::span<const float>(rc_img));
    rc_tex.generate_mipmaps();

    // ---- variance + depth scratch textures ----
    Texture variance_tex(dev, Texture::Descriptor{
        kW, kH, 1, PixelFormat::R32Float });
    Texture tmp_depth_tex(dev, Texture::Descriptor{
        kW, kH, 1, PixelFormat::R32Float });

    DepthSimMap dsm(dev);
    DepthSimMap::OptimizeGradientDescentParams p{};
    p.width            = kW;
    p.height           = kH;
    p.roi_x_begin      = 0;
    p.roi_y_begin      = 0;
    p.rc_level_width   = kW;
    p.rc_level_height  = kH;
    p.rc_mipmap_level  = 0.0f;
    p.step_xy          = 1;
    p.nb_iterations    = kNbIter;
    dsm.optimize_depth_sim_map(
        out_buf, sgm_buf, ref_buf, rc_tex,
        variance_tex, tmp_depth_tex, cam, p);

    const auto* gpu = static_cast<const float*>(out_buf.data());

    // ---- CPU reference (single iteration block per iter) ----
    std::vector<float> opt(pix * 2);
    for (std::size_t k = 0; k < pix; ++k) {
        opt[k * 2 + 0] = kSgmDepth;
        opt[k * 2 + 1] = kSgmPixSize;
    }

    for (int iter = 0; iter < kNbIter; ++iter) {
        std::vector<float> tmp_depth(pix);
        for (std::size_t k = 0; k < pix; ++k) tmp_depth[k] = opt[k * 2];

        std::vector<float> next_opt = opt;

        for (std::uint32_t roiY = 0; roiY < kH; ++roiY)
            for (std::uint32_t roiX = 0; roiX < kW; ++roiX) {
                const std::size_t k = std::size_t(roiY) * kW + roiX;

                float sgm_d = sgm[k * 2 + 0];
                float sgm_p = sgm[k * 2 + 1];
                float ref_d = refn[k * 2 + 0];
                float ref_s = refn[k * 2 + 1];

                float opt_d, opt_s;
                if (iter == 0) {
                    opt_d = sgm_d;
                    opt_s = ref_s;
                } else {
                    opt_d = opt[k * 2 + 0];
                    opt_s = opt[k * 2 + 1];
                }
                float out_d = opt_d, out_s = opt_s;
                if (opt_d > 0.0f) {
                    // getCellSmoothStepEnergy
                    auto sample_depth = [&](int x, int y) {
                        x = std::clamp(x, 0, int(kW) - 1);
                        y = std::clamp(y, 0, int(kH) - 1);
                        return tmp_depth[std::size_t(y) * kW + x];
                    };
                    float d0 = sample_depth(int(roiX), int(roiY));
                    float dL = sample_depth(int(roiX),     int(roiY) - 1);
                    float dR = sample_depth(int(roiX),     int(roiY) + 1);
                    float dU = sample_depth(int(roiX) - 1, int(roiY));
                    float dB = sample_depth(int(roiX) + 1, int(roiY));
                    float smooth_energy_y = 180.0f;
                    float smooth_step = 0.0f;
                    if (d0 > 0.0f) {
                        V3 p0 = get_3d(cam,
                            float(roiX) + float(p.roi_x_begin),
                            float(roiY) + float(p.roi_y_begin), d0);
                        V3 pL = get_3d(cam,
                            float(roiX) + float(p.roi_x_begin),
                            float(roiY) - 1.0f + float(p.roi_y_begin), dL);
                        V3 pR = get_3d(cam,
                            float(roiX) + float(p.roi_x_begin),
                            float(roiY) + 1.0f + float(p.roi_y_begin), dR);
                        V3 pU = get_3d(cam,
                            float(roiX) - 1.0f + float(p.roi_x_begin),
                            float(roiY) + float(p.roi_y_begin), dU);
                        V3 pB = get_3d(cam,
                            float(roiX) + 1.0f + float(p.roi_x_begin),
                            float(roiY) + float(p.roi_y_begin), dB);
                        V3 cg = {0,0,0};
                        float n = 0;
                        if (dL > 0) { cg = add(cg, pL); n += 1; }
                        if (dR > 0) { cg = add(cg, pR); n += 1; }
                        if (dU > 0) { cg = add(cg, pU); n += 1; }
                        if (dB > 0) { cg = add(cg, pB); n += 1; }
                        if (n > 1.0f) {
                            cg = mul(cg, 1.0f / n);
                            V3 C = { cam.C[0], cam.C[1], cam.C[2] };
                            V3 vcn = nrm3(sub(C, p0));
                            // closestPointToLine3D(cg, p0, vcn) =
                            //   p0 + vcn * dot(vcn, cg - p0)
                            V3 pS = add(p0, mul(vcn, dot3(vcn, sub(cg, p0))));
                            smooth_step = len3(sub(C, pS)) - d0;
                        }
                        float e = 0;
                        float n2 = 0;
                        if (dL > 0 && dR > 0) {
                            e = std::max(e, 180.0f - angle_deg(p0, pL, pR));
                            n2 += 1;
                        }
                        if (dU > 0 && dB > 0) {
                            e = std::max(e, 180.0f - angle_deg(p0, pU, pB));
                            n2 += 1;
                        }
                        if (n2 > 0.0f) smooth_energy_y = e;
                    }
                    float step_to_smooth = smooth_step;
                    step_to_smooth = copysign_f(
                        std::fmin(std::fabs(step_to_smooth), sgm_p / 10.0f),
                        step_to_smooth);
                    float depth_energy = smooth_energy_y;

                    float step_to_fine = ref_d - opt_d;
                    step_to_fine = copysign_f(
                        std::fmin(std::fabs(step_to_fine), sgm_p / 10.0f),
                        step_to_fine);

                    float step_to_rough = sgm_d - opt_d;
                    float img_var = 0.0f;   // constant rc image
                    float weighted_color_var =
                        sigmoid2_cpu(5.0f, 30.0f, 40.0f, 20.0f, img_var);
                    float fine_sim_w =
                        sigmoid_cpu(0.0f, 1.0f, 0.7f, -0.7f, ref_s);
                    float energy_lower_w =
                        sigmoid_cpu(0.0f, 1.0f, 30.0f, weighted_color_var, depth_energy);
                    float close_to_rough_w =
                        1.0f - sigmoid_cpu(0.0f, 1.0f, 10.0f, 17.0f,
                                           std::fabs(step_to_rough / sgm_p));
                    float depth_opt_step =
                        close_to_rough_w * step_to_rough +
                        (1.0f - close_to_rough_w) *
                            (energy_lower_w * fine_sim_w * step_to_fine +
                             (1.0f - energy_lower_w) * step_to_smooth);

                    out_d = opt_d + depth_opt_step;
                    out_s = (1.0f - close_to_rough_w) *
                            (energy_lower_w * fine_sim_w * ref_s +
                             (1.0f - energy_lower_w) * (depth_energy / 20.0f));
                }
                next_opt[k * 2 + 0] = out_d;
                next_opt[k * 2 + 1] = out_s;
            }
        opt = next_opt;
    }

    // ---- compare ----
    int bad = 0;
    double worst_d = 0.0, worst_s = 0.0;
    for (std::size_t k = 0; k < pix; ++k) {
        const float gd = gpu[k * 2 + 0];
        const float gs = gpu[k * 2 + 1];
        const float rd = opt[k * 2 + 0];
        const float rs = opt[k * 2 + 1];
        auto rel = [](float a, float b) {
            const float d = std::fabs(a - b);
            return d / std::fmax(std::fabs(a) + std::fabs(b), 1e-30f);
        };
        const double r_d = rel(gd, rd);
        const double r_s = rel(gs, rs);
        worst_d = std::max(worst_d, r_d);
        worst_s = std::max(worst_s, r_s);
        // Budgets:
        //   * depth: tight (1e-5 rel) — depth path is essentially
        //     scalar adds + a copysign(min(...)) clamp. No
        //     sigmoid in the depth-update for our test fixture.
        //   * sim: 1e-3 rel — the sim output is a chained product
        //     of 4 sigmoids `exp()`-based. Metal compiles with
        //     `-ffast-math` (set in `cmake/Metal.cmake`), which
        //     loosens `exp()` semantics vs `std::exp()`. The
        //     observed drift is ~2e-4 rel (~2e-4 absolute on a
        //     [-1, 1] sim).
        if (r_d > 1e-5 || r_s > 1e-3) {
            if (bad < 4) std::fprintf(stderr,
                "k=%zu gpu=(%.7g, %.7g) ref=(%.7g, %.7g) rel=(%.2e, %.2e)\n",
                k, double(gd), double(gs), double(rd), double(rs),
                r_d, r_s);
            ++bad;
        }
        if (!std::isfinite(gd) || !std::isfinite(gs)) {
            std::fprintf(stderr, "k=%zu non-finite gpu=(%g, %g)\n",
                k, double(gd), double(gs));
            return 1;
        }
    }

    std::printf("[info] roi              : %u × %u = %zu pixels\n", kW, kH, pix);
    std::printf("[info] iterations       : %d\n", kNbIter);
    std::printf("[info] sample gpu (mid) : (%.7g, %.7g)\n",
                double(gpu[(kH/2 * kW + kW/2) * 2 + 0]),
                double(gpu[(kH/2 * kW + kW/2) * 2 + 1]));
    std::printf("[info] worst rel        : d=%.2e  s=%.2e\n", worst_d, worst_s);
    std::printf("[info] budgets          : d_rel ≤ 1e-5  s_rel ≤ 1e-3\n");
    std::printf("[info] mismatches       : %d\n", bad);

    if (bad) {
        std::fprintf(stderr, "FAIL: %d mismatches\n", bad);
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
