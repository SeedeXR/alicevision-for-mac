// test_sgm_pipeline_via_adapter.cpp — Phase 8 milestone:
// "fake-Sgm pipeline" that invokes the 12 `cuda_*` adapter
// forwarders in the exact same order/parameters as upstream's
// `Sgm::sgmRc` (in `upstream/.../depthMap/Sgm.cpp`) — but
// without compiling any of upstream's host code. Validates
// the adapter end-to-end + produces a real depth map.
//
// Mirrors the S24 `test_depth_pipeline` scene (plane-induced
// homography, 128×96 image, SGM ROI 16×12 / 11 planes, Refine
// ROI 32×24 / 9 z-slices) so we can compare lock-on rates.
//
// Call sequence (per `Sgm.cpp` / `Refine.cpp` lines 211–314 / 125–286):
//   SGM:
//     1.  cuda_volumeInitialize<TSim>(best,   255)
//     2.  cuda_volumeInitialize<TSim>(2nd,    255)
//     3.  for each T: cuda_volumeComputeSimilarity(best, 2nd, ...)
//     4.  cuda_volumeUpdateUninitializedSimilarity(best, 2nd)
//     5.  cuda_volumeOptimize(filtered, slice_a, slice_b, axis_acc,
//                              best, rcMip, sgmParams, lastDepthIndex,
//                              roi, stream)
//     6.  cuda_volumeRetrieveBestDepth(out_depth_thickness,
//                                      out_depth_sim, depths,
//                                      filtered, rcId, sgmParams,
//                                      range, roi, stream)
//   Bridge:
//     7.  cuda_computeSgmUpscaledDepthPixSizeMap(...)
//   Refine:
//     8.  cuda_volumeInitialize<TSimRefine>(rfvol, 0.f)
//     9.  for each T: cuda_volumeRefineSimilarity(rfvol, ...)
//    10.  cuda_volumeRefineBestDepth(out_refine_depth_sim_map,
//                                    sgm_dp, rfvol, refineParams,
//                                    roi, stream)
//   Optimize:
//    11.  cuda_depthSimMapOptimizeGradientDescent(out_opt_dsm,
//                                                 imgVar, tmpDepth,
//                                                 sgm_dp, refine_ds,
//                                                 rcId, rcMip,
//                                                 refineParams, roi,
//                                                 stream)
//
// Validation: read back out_opt_dsm.x channel; compare per pixel
// to analytical truth on the plane Z=4. Expect lock-on rate
// comparable to test_depth_pipeline (S24: 94% within 1.5 SGM
// steps).

#include "av/depth_map/upstream_adapter.hpp"
#include "av/depth_map/DeviceMipmapImage.hpp"
#include "av/depth_map/PatchOps.hpp"   // DeviceCameraParams
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

// Pull the shim definitions; we use the SAME paths upstream
// `Sgm.cpp` would use, so the shim's CudaDeviceMemoryPitched and
// DeviceMipmapImage flow naturally.
#include "../cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/memory.hpp"
#include "../cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/DeviceMipmapImage.hpp"
#include "../src/depth_map_metal/src/upstream_adapter_types.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using av::depth_map::DeviceCameraParams;

constexpr std::uint32_t kImgW = 128;
constexpr std::uint32_t kImgH =  96;

constexpr std::uint32_t kSgmX = 16;
constexpr std::uint32_t kSgmY = 12;
constexpr std::uint32_t kSgmZ = 11;
constexpr int           kSgmStep = 8;
constexpr int           kSgmWsh  = 4;

constexpr std::uint32_t kRfX = 32;
constexpr std::uint32_t kRfY = 24;
constexpr std::uint32_t kRfZ = 9;
constexpr int           kRfStep = 4;
constexpr int           kRfWsh  = 3;
constexpr std::int32_t  kHalfNbDepths      = (int(kRfZ) - 1) / 2;
constexpr std::int32_t  kSamplesPerPixSize = 4;

constexpr float kTruthZ   = 4.0f;
constexpr float kDepth0   = 3.5f;
constexpr float kDepthDz  = 0.10f;

constexpr float kFx = 200.0f, kFy = 200.0f;
constexpr float kCx = float(kImgW) * 0.5f;
constexpr float kCy = float(kImgH) * 0.5f;

// ----- camera + image helpers (cloned from test_depth_pipeline) -----

template <int R, int C>
void pack(const Eigen::Matrix<double, R, C>& m, float* out) {
    for (int j = 0; j < C; ++j)
        for (int i = 0; i < R; ++i)
            out[j * R + i] = static_cast<float>(m(i, j));
}
void pack_vec3(const Eigen::Vector3d& v, float* out) {
    out[0] = float(v.x()); out[1] = float(v.y()); out[2] = float(v.z());
}
DeviceCameraParams make_cam(const Eigen::Matrix3d& K,
                            const Eigen::Matrix3d& R,
                            const Eigen::Vector3d& C)
{
    Eigen::Matrix<double, 3, 4> P;
    P.block<3, 3>(0, 0) = R;
    P.block<3, 1>(0, 3) = -R * C;
    P = K * P;
    const Eigen::Matrix3d iK = K.inverse();
    const Eigen::Matrix3d iP = R.transpose() * iK;
    DeviceCameraParams cp{};
    pack<3, 4>(P, cp.P);
    pack<3, 3>(iP, cp.iP);
    pack<3, 3>(R, cp.R);
    pack<3, 3>(R.transpose(), cp.iR);
    pack<3, 3>(K, cp.K);
    pack<3, 3>(iK, cp.iK);
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
            const float b = 128.0f + 45.0f * std::sin(17.0f*(u + v));
            const std::size_t k = (j * kImgW + i) * 4;
            px[k+0] = std::clamp(r, 0.0f, 255.0f);
            px[k+1] = std::clamp(g, 0.0f, 255.0f);
            px[k+2] = std::clamp(b, 0.0f, 255.0f);
            px[k+3] = 255.0f;
        }
    return px;
}

Eigen::Matrix3d plane_homography(const Eigen::Matrix3d& K,
                                  const Eigen::Matrix3d& Rrc, const Eigen::Vector3d& Crc,
                                  const Eigen::Matrix3d& Rtc, const Eigen::Vector3d& Ctc,
                                  const Eigen::Vector3d& pp, const Eigen::Vector3d& pn)
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
    using namespace aliceVision::depthMap;
    using av::depth_map::upstream_adapter::set_camera_param;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    // CRITICAL: wire the type-shim's process-global device for all
    // subsequent CudaDeviceMemoryPitched allocations.
    set_adapter_device(dev);

    static_assert(kSgmX * kSgmStep == kImgW, "sgm step");
    static_assert(kRfX  * kRfStep  == kImgW, "rf step");

    // ===== camera + homography =====
    Eigen::Matrix3d K;
    K << kFx, 0.0, double(kCx),
         0.0, kFy, double(kCy),
         0.0, 0.0, 1.0;
    const Eigen::Matrix3d Rrc = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d Crc(0, 0, 0);
    const Eigen::AngleAxisd aa(0.03, Eigen::Vector3d::UnitY());
    const Eigen::Matrix3d Rtc = aa.toRotationMatrix();
    const Eigen::Vector3d Ctc(0.25, 0, 0);

    const DeviceCameraParams rc_cam = make_cam(K, Rrc, Crc);
    const DeviceCameraParams tc_cam = make_cam(K, Rtc, Ctc);

    constexpr int kRcId = 0;
    constexpr int kTcId = 1;
    set_camera_param(kRcId, rc_cam);
    set_camera_param(kTcId, tc_cam);

    const Eigen::Vector3d plane_pt(0, 0, double(kTruthZ));
    const Eigen::Vector3d plane_nm(0, 0, -1);
    const Eigen::Matrix3d H = plane_homography(K, Rrc, Crc, Rtc, Ctc,
                                                plane_pt, plane_nm);
    const Eigen::Matrix3d H_inv = H.inverse();

    const auto r_img = make_r_image();
    const auto t_img = warp(r_img, H_inv);

    // Mipmap images via the shim — adapter forwarders use the
    // `aliceVision::depthMap::DeviceMipmapImage` shim's
    // `.av_texture()` / `.av_impl()` bridges.
    av::depth_map::DeviceMipmapImage rc_av(dev);
    rc_av.fill(std::span<const float>(r_img), kImgW, kImgH, 1, 8);
    av::depth_map::DeviceMipmapImage tc_av(dev);
    tc_av.fill(std::span<const float>(t_img), kImgW, kImgH, 1, 8);
    aliceVision::depthMap::DeviceMipmapImage rc_mip{rc_av};
    aliceVision::depthMap::DeviceMipmapImage tc_mip{tc_av};

    // ===== params =====
    SgmParams sgmParams;
    sgmParams.scale  = 1;            // mipmap level 0
    sgmParams.stepXY = kSgmStep;
    sgmParams.wsh    = kSgmWsh;
    sgmParams.gammaC = 20.0;
    sgmParams.gammaP = 4.0;
    sgmParams.p1     = 10.0;
    sgmParams.p2Weighting = -100.0;  // negative → fixed P2 = 100
    sgmParams.depthThicknessInflate = 1.0;
    sgmParams.maxSimilarity         = 220.0;
    sgmParams.useConsistentScale    = false;

    RefineParams refineParams;
    refineParams.scale  = 1;
    refineParams.stepXY = kRfStep;
    refineParams.wsh    = kRfWsh;
    refineParams.gammaC = 20.0;
    refineParams.gammaP = 4.0;
    refineParams.halfNbDepths   = kHalfNbDepths;
    refineParams.nbSubsamples   = kSamplesPerPixSize;
    refineParams.sigma          = 4.0;
    refineParams.interpolateMiddleDepth = true;     // bilinear
    refineParams.useConsistentScale     = false;
    refineParams.useSgmNormalMap        = false;
    refineParams.optimizationNbIterations = 20;

    aliceVision::ROI sgm_roi;
    sgm_roi.x.begin = 0; sgm_roi.x.end = kSgmX;
    sgm_roi.y.begin = 0; sgm_roi.y.end = kSgmY;
    aliceVision::ROI rf_roi;
    rf_roi.x.begin = 0; rf_roi.x.end = kRfX;
    rf_roi.y.begin = 0; rf_roi.y.end = kRfY;

    aliceVision::Range depthRange;
    depthRange.begin = 0; depthRange.end = kSgmZ;

    // ===== SGM allocations =====
    CudaSize<3> sgmVolDim(kSgmX, kSgmY, kSgmZ);
    CudaDeviceMemoryPitched<TSim, 3> volumeBestSim_dmp(sgmVolDim);
    CudaDeviceMemoryPitched<TSim, 3> volumeSecBestSim_dmp(sgmVolDim);

    // Depths buffer (CudaDeviceMemoryPitched<float, 2> mimicking
    // upstream's 1×N "depths" layout — see Sgm.cpp:49).
    CudaSize<2> depthsDim(kSgmZ, 1);
    CudaDeviceMemoryPitched<float, 2> depths_dmp(depthsDim);
    {
        float* p = depths_dmp.getBuffer();
        for (std::uint32_t z = 0; z < kSgmZ; ++z) p[z] = kDepth0 + float(z) * kDepthDz;
    }

    // Output buffers from retrieveBestDepth.
    CudaSize<2> sgmMapDim(kSgmX, kSgmY);
    CudaDeviceMemoryPitched<float2, 2> depthThicknessMap_dmp(sgmMapDim);
    CudaDeviceMemoryPitched<float2, 2> depthSimMap_dmp(sgmMapDim);

    // Optimize scratch (Sgm.cpp:77–79).
    const std::uint32_t maxXY = std::max(kSgmX, kSgmY);
    CudaSize<2> sliceADim(maxXY, kSgmZ);
    CudaSize<2> sliceBDim(maxXY, kSgmZ);
    CudaSize<2> axisAccDim(maxXY, 1);
    CudaDeviceMemoryPitched<TSimAcc, 2> volumeSliceAccA_dmp(sliceADim);
    CudaDeviceMemoryPitched<TSimAcc, 2> volumeSliceAccB_dmp(sliceBDim);
    CudaDeviceMemoryPitched<TSimAcc, 2> volumeAxisAcc_dmp(axisAccDim);

    // ===== SGM pipeline =====
    cuda_volumeInitialize(volumeBestSim_dmp,   TSim(255), nullptr);
    cuda_volumeInitialize(volumeSecBestSim_dmp, TSim(255), nullptr);

    cuda_volumeComputeSimilarity(volumeBestSim_dmp, volumeSecBestSim_dmp,
                                 depths_dmp,
                                 kRcId, kTcId,
                                 rc_mip, tc_mip,
                                 sgmParams,
                                 depthRange,
                                 sgm_roi,
                                 nullptr);

    cuda_volumeUpdateUninitializedSimilarity(volumeBestSim_dmp,
                                              volumeSecBestSim_dmp,
                                              nullptr);

    cuda_volumeOptimize(volumeBestSim_dmp,   // out (reuse best)
                        volumeSliceAccA_dmp,
                        volumeSliceAccB_dmp,
                        volumeAxisAcc_dmp,
                        volumeBestSim_dmp,   // in (same volume!
                                              // upstream actually uses
                                              // best as both in and out)
                        rc_mip,
                        sgmParams,
                        int(kSgmZ),
                        sgm_roi,
                        nullptr);

    cuda_volumeRetrieveBestDepth(depthThicknessMap_dmp,
                                 depthSimMap_dmp,
                                 depths_dmp,
                                 volumeBestSim_dmp,
                                 kRcId,
                                 sgmParams,
                                 depthRange,
                                 sgm_roi,
                                 nullptr);

    // ---- snapshot SGM result ----
    int sgm_valid = 0, sgm_invalid = 0;
    std::vector<double> sgm_errs;
    {
        const float2* dt = depthThicknessMap_dmp.getBuffer();
        for (std::uint32_t y = 0; y < kSgmY; ++y)
            for (std::uint32_t x = 0; x < kSgmX; ++x) {
                const float d = dt[std::size_t(y) * kSgmX + x].x;
                if (d <= 0.0f) { ++sgm_invalid; continue; }
                ++sgm_valid;
                const double px = (double(x) + 0.5) * double(kSgmStep);
                const double py = (double(y) + 0.5) * double(kSgmStep);
                sgm_errs.push_back(std::abs(double(d) - true_depth(px, py, kTruthZ)));
            }
    }
    {
        std::vector<double> e = sgm_errs;
        std::printf("[sgm ] valid=%d invalid=%d median=%.4f p90=%.4f\n",
                    sgm_valid, sgm_invalid,
                    pct(e, 0.5), pct(e, 0.9));
    }

    // ===== Refine bridge =====
    CudaSize<2> rfMapDim(kRfX, kRfY);
    CudaDeviceMemoryPitched<float2, 2> sgmDepthPixSizeMap_dmp(rfMapDim);

    cuda_computeSgmUpscaledDepthPixSizeMap(sgmDepthPixSizeMap_dmp,
                                            depthThicknessMap_dmp,
                                            kRcId,
                                            rc_mip,
                                            refineParams,
                                            rf_roi,
                                            nullptr);

    // ===== Refine =====
    CudaSize<3> rfVolDim(kRfX, kRfY, kRfZ);
    CudaDeviceMemoryPitched<TSimRefine, 3> volumeRefineSim_dmp(rfVolDim);

    cuda_volumeInitialize(volumeRefineSim_dmp, TSimRefine(0.0f), nullptr);

    aliceVision::Range rfRange;
    rfRange.begin = 0; rfRange.end = kRfZ;

    cuda_volumeRefineSimilarity(volumeRefineSim_dmp,
                                 sgmDepthPixSizeMap_dmp,
                                 nullptr,           // no sgm normal map
                                 kRcId, kTcId,
                                 rc_mip, tc_mip,
                                 refineParams,
                                 rfRange,
                                 rf_roi,
                                 nullptr);

    CudaDeviceMemoryPitched<float2, 2> refinedDepthSimMap_dmp(rfMapDim);
    cuda_volumeRefineBestDepth(refinedDepthSimMap_dmp,
                                sgmDepthPixSizeMap_dmp,
                                volumeRefineSim_dmp,
                                refineParams,
                                rf_roi,
                                nullptr);

    // ===== Optimize =====
    CudaDeviceMemoryPitched<float2, 2> optimizedDepthSimMap_dmp(rfMapDim);
    CudaDeviceMemoryPitched<float, 2>  optImgVariance_dmp(rfMapDim);
    CudaDeviceMemoryPitched<float, 2>  optTmpDepthMap_dmp(rfMapDim);

    cuda_depthSimMapOptimizeGradientDescent(optimizedDepthSimMap_dmp,
                                              optImgVariance_dmp,
                                              optTmpDepthMap_dmp,
                                              sgmDepthPixSizeMap_dmp,
                                              refinedDepthSimMap_dmp,
                                              kRcId,
                                              rc_mip,
                                              refineParams,
                                              rf_roi,
                                              nullptr);

    // ===== validate against analytical truth =====
    const float2* opt = optimizedDepthSimMap_dmp.getBuffer();
    std::vector<double> errs;
    int valid = 0, invalid = 0, locked = 0;
    constexpr double kLockTol = 1.5 * double(kDepthDz);
    for (std::uint32_t y = 0; y < kRfY; ++y)
        for (std::uint32_t x = 0; x < kRfX; ++x) {
            const float d = opt[std::size_t(y) * kRfX + x].x;
            if (d <= 0.0f) { ++invalid; continue; }
            ++valid;
            const double px = (double(x) + 0.5) * double(kRfStep);
            const double py = (double(y) + 0.5) * double(kRfStep);
            const double err = std::abs(double(d) - true_depth(px, py, kTruthZ));
            errs.push_back(err);
            if (err <= kLockTol) ++locked;
        }

    std::vector<double> e = errs;
    const double median = pct(e, 0.5);
    const double p90    = pct(e, 0.9);
    const double worst  = errs.empty() ? 0.0 : *std::max_element(errs.begin(), errs.end());
    const double lock_rate = errs.empty() ? 0.0 :
        100.0 * double(locked) / double(errs.size());

    std::printf("[opt ] valid=%d invalid=%d (of %u)\n",
                valid, invalid, kRfX * kRfY);
    std::printf("[opt ] |Δ truth| median=%.4f p90=%.4f worst=%.4f\n",
                median, p90, worst);
    std::printf("[opt ] lock-on (≤ %.2f) %.1f%%  (%d/%zu)\n",
                kLockTol, lock_rate, locked, errs.size());

    // Budgets mirror S24 (test_depth_pipeline) — the adapter is
    // expected to produce equivalent results, lock-on ≥ 70%.
    bool ok = true;
    if (valid < int(kRfX * kRfY) * 60 / 100) {
        std::fprintf(stderr, "FAIL: only %d / %u valid\n", valid, kRfX * kRfY);
        ok = false;
    }
    if (lock_rate < 70.0) {
        std::fprintf(stderr,
            "FAIL: lock-on rate %.1f%% < 70%%\n", lock_rate);
        ok = false;
    }
    if (!ok) return 1;

    // Cleanup the camera-param table.
    av::depth_map::upstream_adapter::clear_camera_params();

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
