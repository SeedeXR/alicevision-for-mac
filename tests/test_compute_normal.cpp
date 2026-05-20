// test_compute_normal.cpp — validation of
// `av_depth_sim_map_compute_normal`.
//
// Generate a depth map for a tilted plane (analytical truth),
// run the normal-from-depth kernel, compare every valid output
// normal against the true plane normal.
//
// Precision note: the kernel uses an FP32 stat3d covariance
// accumulator + an FP32 eig33 eigendecomposition. Upstream CUDA
// uses FP64 accumulators (no FP64 on Apple Silicon GPUs). The
// budget tolerates the expected FP32 drift:
//   * Median cosine deviation ≤ 1e-3
//   * p90 cosine deviation     ≤ 1e-2
// (cosine deviation = 1 − |dot(gpu_n, true_n)|)
//
// Coverage:
//   * Tilted plane with a moderate but non-axis-aligned normal.
//   * Interior pixels (≥ wsh from each edge) where all 49
//     neighbors are valid.
//   * Edge pixels with truncated neighborhoods (kernel uses
//     `continue` when `roiX/Y + xp/yp` falls outside the ROI).
//   * Invalid-depth pixels written as a stripe → expect normals
//     of (-1, -1, -1).

#include "av/depth_map/DepthSimMap.hpp"
#include "av/depth_map/PatchOps.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using av::depth_map::DeviceCameraParams;

constexpr std::uint32_t kW    = 64;
constexpr std::uint32_t kH    = 48;
constexpr std::int32_t  kStep = 1;
constexpr float kFx = 200.0f, kFy = 200.0f;
constexpr float kCx = float(kW) * 0.5f;
constexpr float kCy = float(kH) * 0.5f;

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

// Compute the depth (along the camera ray) of the intersection
// of the pixel's ray with a plane.
// Plane: n · X = d (n unit; in world coords)
// Ray:   X = C + t · D    (D = normalized world ray direction)
// → t   = (d − n·C) / (n·D)
// Depth (Euclidean from C) = t × |D| = t (since D is unit).
double ray_plane_depth(const Eigen::Matrix3d& iP_d,
                       const Eigen::Vector3d& C,
                       double pix_x, double pix_y,
                       const Eigen::Vector3d& n_unit, double d_plane)
{
    Eigen::Vector3d v(pix_x, pix_y, 1.0);
    Eigen::Vector3d D = iP_d * v;
    D.normalize();
    const double denom = n_unit.dot(D);
    if (std::abs(denom) < 1e-12) return -1.0;
    const double t = (d_plane - n_unit.dot(C)) / denom;
    if (t <= 0.0) return -1.0;
    return t;
}

}  // namespace

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    // ---- camera ----
    Eigen::Matrix3d K;
    K << kFx, 0.0, double(kCx),
         0.0, kFy, double(kCy),
         0.0, 0.0, 1.0;
    const Eigen::Matrix3d R   = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d C   (0.0, 0.0, 0.0);
    const DeviceCameraParams cam = make_cam(K, R, C);
    const Eigen::Matrix3d iP_d   = R.transpose() * K.inverse();

    // ---- tilted plane in world coords ----
    Eigen::Vector3d n_world(0.15, 0.10, -1.0);
    n_world.normalize();
    constexpr double kPlanePointZ = 4.0;
    // Plane passes through (0, 0, kPlanePointZ).
    const double d_plane = n_world.dot(Eigen::Vector3d(0.0, 0.0, kPlanePointZ));

    // ---- synthesize depth map ----
    std::vector<float> dsm(std::size_t(kW) * kH * 2);
    for (std::uint32_t y = 0; y < kH; ++y)
        for (std::uint32_t x = 0; x < kW; ++x) {
            const std::size_t k = (std::size_t(y) * kW + x) * 2;
            double depth = ray_plane_depth(iP_d, C,
                double(x) + 0.5, double(y) + 0.5,
                n_world, d_plane);
            // Mark an invalid stripe at y in [3, 6) to exercise
            // the (-1, -1, -1) output path.
            if (y >= 3 && y < 6) depth = -1.0;
            dsm[k + 0] = float(depth);
            dsm[k + 1] = 0.0f;
        }
    Buffer dsm_buf(dev, dsm.size() * sizeof(float));
    dsm_buf.upload(std::span<const float>(dsm));

    // ---- output (packed_float3 = 3 floats per pixel) ----
    Buffer normal_buf(dev,
        std::size_t(kW) * std::size_t(kH) * 3 * sizeof(float));
    std::vector<float> sentinel(
        std::size_t(kW) * kH * 3, -7777.0f);
    normal_buf.upload(std::span<const float>(sentinel));

    DepthSimMap dsm_runner(dev);
    DepthSimMap::ComputeNormalParams p{};
    p.width       = kW;
    p.height      = kH;
    p.roi_x_begin = 0;
    p.roi_y_begin = 0;
    p.step_xy     = kStep;
    p.wsh         = 3;
    dsm_runner.compute_normal(normal_buf, dsm_buf, cam, p);

    const auto* gpu = static_cast<const float*>(normal_buf.data());

    // ---- analyze ----
    int valid_pixels   = 0;
    int invalid_match  = 0;  // (-1, -1, -1) where expected
    int invalid_wrong  = 0;
    int corner_wsh     = 0;  // edge pixels with truncated neighborhood
    int interior       = 0;
    std::vector<double> cos_devs;
    cos_devs.reserve(kW * kH);

    for (std::uint32_t y = 0; y < kH; ++y) {
        for (std::uint32_t x = 0; x < kW; ++x) {
            const std::size_t k    = std::size_t(y) * kW + x;
            const float       in_d = dsm[k * 2];
            const float       gx   = gpu[k * 3 + 0];
            const float       gy   = gpu[k * 3 + 1];
            const float       gz   = gpu[k * 3 + 2];

            if (in_d <= 0.0f) {
                // Expect (-1, -1, -1).
                if (gx == -1.0f && gy == -1.0f && gz == -1.0f) {
                    ++invalid_match;
                } else {
                    ++invalid_wrong;
                }
                continue;
            }
            // Even valid-depth pixels may get (-1, -1, -1) if the
            // PCA fails (< 3 valid neighbors). Edge pixels at the
            // very corners might trigger this — count them.
            if (gx == -1.0f && gy == -1.0f && gz == -1.0f) {
                ++corner_wsh;
                continue;
            }
            ++valid_pixels;
            const bool is_interior =
                (x >= 3u && x + 3u < kW) &&
                (y >= 6u && y + 3u < kH);   // y >= 6 to clear invalid stripe
            if (is_interior) ++interior;

            // Cosine deviation from |n_world|. Brace-init avoids
            // most-vexing-parse with the float→double conversions.
            const Eigen::Vector3d gpu_n{double(gx), double(gy), double(gz)};
            const double dot_abs = std::abs(gpu_n.dot(n_world));
            const double cos_dev = 1.0 - std::min(1.0, dot_abs);
            cos_devs.push_back(cos_dev);
        }
    }

    auto pct = [](std::vector<double>& v, double q) -> double {
        if (v.empty()) return 0.0;
        const std::size_t k = std::min<std::size_t>(
            v.size() - 1, std::size_t(q * double(v.size())));
        std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(k), v.end());
        return v[k];
    };
    std::vector<double> tmp = cos_devs;
    const double median = pct(tmp, 0.50);
    const double p90    = pct(tmp, 0.90);
    const double p99    = pct(tmp, 0.99);
    const double worst  = cos_devs.empty()
        ? 0.0 : *std::max_element(cos_devs.begin(), cos_devs.end());

    std::printf("[info] roi              : %u × %u = %u pixels\n",
                kW, kH, kW * kH);
    std::printf("[info] valid pixels     : %d  (interior=%d)\n",
                valid_pixels, interior);
    std::printf("[info] invalid (-1,-1,-1): match=%d  wrong=%d  corner=%d\n",
                invalid_match, invalid_wrong, corner_wsh);
    std::printf("[info] cosine-deviation : median=%.2e  p90=%.2e  p99=%.2e  worst=%.2e\n",
                median, p90, p99, worst);

    int failed = 0;
    if (invalid_wrong != 0) {
        std::fprintf(stderr,
            "FAIL: %d invalid-input pixels did not yield (-1,-1,-1)\n",
            invalid_wrong);
        ++failed;
    }
    if (valid_pixels < int(kW * kH) / 2) {
        std::fprintf(stderr,
            "FAIL: only %d valid normals from %u pixels\n",
            valid_pixels, kW * kH);
        ++failed;
    }
    if (median > 1e-3) {
        std::fprintf(stderr,
            "FAIL: median cosine deviation %.3e > 1e-3\n", median);
        ++failed;
    }
    if (p90 > 1e-2) {
        std::fprintf(stderr,
            "FAIL: p90 cosine deviation %.3e > 1e-2\n", p90);
        ++failed;
    }
    if (failed) return 1;
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
