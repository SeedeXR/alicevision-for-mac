// test_retrieve_best_depth.cpp — validation of the
// volume_retrieve_best_depth kernel (Phase 7 SGM exit point).
//
// Strategy:
//   1. Build a synthetic uchar cost volume on the CPU where each
//      pixel (vx, vy) has a known "true" best-Z index with a low
//      similarity score, and progressively worse scores at
//      neighboring Z's, and the sentinel 255 at far Z's.
//   2. Build a depth-plane list `depths[z]` of monotonically
//      increasing plane depths in front of the camera.
//   3. Build a synthetic R camera (pinhole, identity rotation,
//      origin at world origin, looking down +Z) and convert to
//      DeviceCameraParams via the same helper used in test_patch /
//      test_comp_ncc.
//   4. Dispatch retrieve_best_depth.
//   5. CPU reference does the same WTA scan + depthPlaneToDepth
//      conversion + thickness computation in FP64.
//   6. Compare per-pixel: depth, thickness, sim. Also stress-test
//      with a stripe of "all-255" pixels (invalid path).

#include "av/depth_map/Volume.hpp"
#include "av/depth_map/PatchOps.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using av::depth_map::DeviceCameraParams;
using av::depth_map::Volume;
using av::depth_map::VolumeDims;

constexpr std::uint32_t W = 48;
constexpr std::uint32_t H = 32;
constexpr std::uint32_t D = 24;
constexpr av::depth_map::VolumeDims kDims{ W, H, D };

// ---- mirror of test_patch's make_cam (FP64 → DeviceCameraParams) ----

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

// ---- CPU FP64 reference for depthPlaneToDepth + the full kernel ----

double depth_plane_to_depth_ref(const DeviceCameraParams& cp,
                                double fp_plane_depth,
                                Eigen::Vector2d pix)
{
    const Eigen::Vector3d C_v(static_cast<double>(cp.C[0]),
                              static_cast<double>(cp.C[1]),
                              static_cast<double>(cp.C[2]));
    const Eigen::Vector3d Z_v(static_cast<double>(cp.ZVect[0]),
                              static_cast<double>(cp.ZVect[1]),
                              static_cast<double>(cp.ZVect[2]));
    const Eigen::Vector3d planep = C_v + Z_v * fp_plane_depth;

    // M3x3mulV2(iP, pix) = iP * (pix.x, pix.y, 1)^T  (column-major iP)
    Eigen::Matrix3d iP_d;
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            iP_d(i, j) = static_cast<double>(cp.iP[j * 3 + i]);
    Eigen::Vector3d ph(pix.x(), pix.y(), 1.0);
    Eigen::Vector3d v = iP_d * ph;
    v.normalize();

    // linePlaneIntersect(linePoint=C_v, lineVect=v, planePoint=planep, planeNormal=Z_v)
    // k = (dot(planep, Z_v) - dot(Z_v, C_v)) / dot(Z_v, v)
    const double k = (planep.dot(Z_v) - Z_v.dot(C_v)) / Z_v.dot(v);
    const Eigen::Vector3d p_on_plane = C_v + v * k;
    return (C_v - p_on_plane).norm();
}

struct OutPair { float depth, thickness, sim; bool invalid; };

OutPair kernel_ref(const std::vector<std::uint8_t>& vol,
                   const std::vector<float>&         depths,
                   const DeviceCameraParams&         cam,
                   const Volume::RetrieveBestDepthParams& p,
                   std::uint32_t vx, std::uint32_t vy)
{
    const Eigen::Vector2d pix(double(int(p.roi_x_begin + vx) * p.scale_step),
                              double(int(p.roi_y_begin + vy) * p.scale_step));

    double bestSim   = 255.0;
    int    bestZIdx  = -1;
    const std::uint32_t zb = p.depth_range_begin;
    const std::uint32_t ze = (p.depth_range_end == 0) ? p.dims.z : p.depth_range_end;
    for (std::uint32_t vz = zb; vz < ze; ++vz) {
        const std::size_t k = std::size_t(vz) * (p.dims.x * p.dims.y)
                            + std::size_t(vy) * p.dims.x + vx;
        const double s = double(vol[k]);
        if (s < bestSim) {
            bestSim  = s;
            bestZIdx = int(vz);
        }
    }

    if (bestZIdx < 0 || bestSim > double(p.max_similarity)) {
        return { -1.0f, -1.0f, 1.0f, true };
    }

    const int zm1 = std::max(0,                  bestZIdx - 1);
    const int zp1 = std::min(int(p.dims.z) - 1, bestZIdx + 1);

    const double bd    = depth_plane_to_depth_ref(cam, double(depths[bestZIdx]), pix);
    const double bd_m1 = depth_plane_to_depth_ref(cam, double(depths[zm1]),       pix);
    const double bd_p1 = depth_plane_to_depth_ref(cam, double(depths[zp1]),       pix);

    const double thickness =
        std::max(bd_p1 - bd, bd - bd_m1) * double(p.thickness_mult_factor);
    const double sim = (bestSim / 255.0) * 2.0 - 1.0;

    return { float(bd), float(thickness), float(sim), false };
}

}  // namespace

int main() try
{
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    // ---------------- camera ----------------
    Eigen::Matrix3d K;
    K << 400.0,   0.0, double(W) * 0.5,
           0.0, 400.0, double(H) * 0.5,
           0.0,   0.0,   1.0;
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d C(0.0, 0.0, 0.0);
    const DeviceCameraParams cam = make_cam(K, R, C);

    // ---------------- synthetic cost volume ----------------
    std::vector<std::uint8_t> vol(W * H * D, 255);
    std::mt19937_64 rng(0x511);
    std::uniform_int_distribution<int> trueZdist(2, int(D) - 3);

    // Per-pixel: pick a true best-Z; set vol[*, *, bestZ] = 40,
    // make the immediate neighbors 80/120, and leave the rest at 255.
    // Reserve a small "invalid stripe" near vy=0 where all Zs stay 255.
    std::vector<int> true_best(W * H, -1);
    for (std::uint32_t vy = 0; vy < H; ++vy) {
        for (std::uint32_t vx = 0; vx < W; ++vx) {
            if (vy < 2) continue;   // invalid stripe → all 255
            const int z = trueZdist(rng);
            true_best[vy * W + vx] = z;
            const auto put = [&](int zi, std::uint8_t v) {
                if (zi < 0 || zi >= int(D)) return;
                vol[std::size_t(zi) * (W * H) + vy * W + vx] = v;
            };
            put(z,     40);
            put(z - 1, 80);
            put(z + 1, 80);
            put(z - 2, 120);
            put(z + 2, 120);
        }
    }
    // A second stripe with bestSim above the maxSimilarity threshold,
    // so the over-threshold path is also exercised.
    for (std::uint32_t vy = H - 2; vy < H; ++vy) {
        for (std::uint32_t vx = 0; vx < W; ++vx) {
            const int z = trueZdist(rng);
            true_best[vy * W + vx] = z;
            vol[std::size_t(z) * (W * H) + vy * W + vx] = 220;  // > maxSim
        }
    }

    // ---------------- depth-plane list ----------------
    std::vector<float> depths(D);
    for (std::uint32_t z = 0; z < D; ++z) {
        depths[z] = 2.0f + float(z) * 0.15f;   // 2.0 .. 5.45 along +Z
    }

    // ---------------- params ----------------
    Volume::RetrieveBestDepthParams params{};
    params.dims                  = kDims;
    params.depth_range_begin     = 0;
    params.depth_range_end       = 0;   // → defaults to dims.z on host
    params.roi_x_begin           = 0;
    params.roi_y_begin           = 0;
    params.scale_step            = 1;
    params.thickness_mult_factor = 1.5f;
    params.max_similarity        = 200.0f;

    // ---------------- GPU buffers ----------------
    Volume volume_runner(dev);

    Buffer vol_buf  (dev, W * H * D * sizeof(std::uint8_t));
    Buffer dep_buf  (dev, D * sizeof(float));
    Buffer dt_buf   (dev, W * H * 2 * sizeof(float));   // float2
    Buffer dsim_buf (dev, W * H * 2 * sizeof(float));   // float2
    vol_buf .set_label("retrieve.vol");
    dep_buf .set_label("retrieve.depths");
    dt_buf  .set_label("retrieve.depth_thickness");
    dsim_buf.set_label("retrieve.depth_sim");

    vol_buf.upload(std::span<const std::uint8_t>(vol));
    dep_buf.upload(std::span<const float>(depths));

    volume_runner.retrieve_best_depth(dt_buf, dsim_buf, dep_buf, vol_buf,
                                      cam, params);

    // ---------------- compare ----------------
    const auto* dt   = static_cast<const float*>(dt_buf.data());
    const auto* dsim = static_cast<const float*>(dsim_buf.data());

    // Tolerances:
    //   * Invalid path: bit-exact equality with (-1, -1) / (-1, 1).
    //   * Valid path: FP32 noise through depthPlaneToDepth +
    //     thickness multiply. Budget 1e-3 abs on depth/thickness;
    //     1e-7 abs on sim (it's a single (bestSim/255)*2-1 ops chain).
    constexpr float kTolDepth     = 1e-3f;
    constexpr float kTolThickness = 1e-3f;
    constexpr float kTolSim       = 1e-6f;

    int bad = 0;
    int invalid_count = 0;
    float worst_depth = 0.0f, worst_thick = 0.0f, worst_sim = 0.0f;

    for (std::uint32_t vy = 0; vy < H; ++vy) {
        for (std::uint32_t vx = 0; vx < W; ++vx) {
            const OutPair ref = kernel_ref(vol, depths, cam, params, vx, vy);
            const std::size_t k = std::size_t(vy) * W + vx;
            const float gpu_depth     = dt[k * 2 + 0];
            const float gpu_thickness = dt[k * 2 + 1];
            const float gpu_sim_depth = dsim[k * 2 + 0];
            const float gpu_sim       = dsim[k * 2 + 1];

            if (ref.invalid) {
                ++invalid_count;
                const bool ok =
                    gpu_depth     == -1.0f &&
                    gpu_thickness == -1.0f &&
                    gpu_sim_depth == -1.0f &&
                    gpu_sim       ==  1.0f;
                if (!ok) {
                    if (bad < 3) std::fprintf(stderr,
                        "invalid (%u, %u): got dt=(%g, %g) dsim=(%g, %g)\n",
                        vx, vy,
                        static_cast<double>(gpu_depth),
                        static_cast<double>(gpu_thickness),
                        static_cast<double>(gpu_sim_depth),
                        static_cast<double>(gpu_sim));
                    ++bad;
                }
                continue;
            }
            const float ed = std::abs(gpu_depth     - ref.depth);
            const float et = std::abs(gpu_thickness - ref.thickness);
            const float es = std::abs(gpu_sim       - ref.sim);
            worst_depth = std::max(worst_depth, ed);
            worst_thick = std::max(worst_thick, et);
            worst_sim   = std::max(worst_sim,   es);
            if (gpu_sim_depth != gpu_depth) {
                if (bad < 3) std::fprintf(stderr,
                    "depth disagree between out_dt and out_dsim @ (%u, %u): %g vs %g\n",
                    vx, vy,
                    static_cast<double>(gpu_depth),
                    static_cast<double>(gpu_sim_depth));
                ++bad;
            }
            if (ed > kTolDepth || et > kTolThickness || es > kTolSim) {
                if (bad < 4) std::fprintf(stderr,
                    "(%u, %u): depth gpu=%g ref=%g (Δ=%g)  "
                    "thick gpu=%g ref=%g (Δ=%g)  "
                    "sim gpu=%g ref=%g (Δ=%g)\n",
                    vx, vy,
                    static_cast<double>(gpu_depth),     static_cast<double>(ref.depth),     static_cast<double>(ed),
                    static_cast<double>(gpu_thickness), static_cast<double>(ref.thickness), static_cast<double>(et),
                    static_cast<double>(gpu_sim),       static_cast<double>(ref.sim),       static_cast<double>(es));
                ++bad;
            }
        }
    }

    std::printf("[info] pixels             : %u (invalid path: %d)\n",
                W * H, invalid_count);
    std::printf("[info] worst |Δdepth|     : %.3g (budget %.3g)\n",
                static_cast<double>(worst_depth), static_cast<double>(kTolDepth));
    std::printf("[info] worst |Δthickness| : %.3g (budget %.3g)\n",
                static_cast<double>(worst_thick), static_cast<double>(kTolThickness));
    std::printf("[info] worst |Δsim|       : %.3g (budget %.3g)\n",
                static_cast<double>(worst_sim), static_cast<double>(kTolSim));

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
