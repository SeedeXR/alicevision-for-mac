// test_gaussian_filter_volume.cpp — validation of the Metal ports of
//   * gaussianBlurVolumeZ_kernel
//   * gaussianBlurVolumeXYZ_kernel
// from upstream's depthMap/cuda/imageProcessing/deviceGaussianFilter.cu
// (host wrappers cuda_gaussianBlurVolumeZ / cuda_gaussianBlurVolumeXYZ).
//
// Strategy: build a synthetic packed float volume, run each kernel
// against a FP64 CPU reference doing the same convolution (including
// the upstream `iz > 0` quirk in the Z-blur). Worst |Δ| per voxel
// must be within 1e-4.

#include "av/depth_map/GaussianFilter.hpp"
#include "av/depth_map/GaussianTable.hpp"
#include "av/depth_map/Volume.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

// ---------------- Gaussian LUT (CPU reference) ----------------

struct LutCpu {
    std::array<int, 10>          offsets{};
    std::vector<double>          weights;

    LutCpu() {
        int sum = 0;
        for (int s = 0; s < 10; ++s) {
            offsets[s] = sum;
            sum += 2 * (s + 1) + 1;
        }
        weights.assign(sum, 0.0);
        // Match upstream's float-precision `expf(-x²/(2·δ²))` with
        // δ = 1, then store as double. (We use float exp to keep the
        // weight values bit-identical to those uploaded to the GPU.)
        for (int s = 0; s < 10; ++s) {
            const int r = s + 1;
            const int n = 2 * r + 1;
            for (int idx = 0; idx < n; ++idx) {
                const int x = idx - r;
                weights[offsets[s] + idx] =
                    double(std::exp(-float(x * x) / 2.0f));
            }
        }
    }

    double at(int scale, int idx) const {
        return weights[offsets[scale] + idx];
    }
};

// ---------------- Volume layout helpers ----------------

inline std::size_t lin_idx(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                           std::uint32_t X, std::uint32_t Y)
{
    return std::size_t(z) * (std::size_t(X) * std::size_t(Y))
         + std::size_t(y) * std::size_t(X)
         + std::size_t(x);
}

// ---------------- CPU reference: gaussianBlurVolumeZ ----------------
//
// Mirrors upstream exactly, including the `(iz < volDimZ) && (iz > 0)`
// quirk (iz==0 excluded from the convolution support).

void blur_z_ref(const std::vector<float>& in,
                std::vector<float>&       out,
                std::uint32_t X, std::uint32_t Y, std::uint32_t Z,
                int gaussRadius,
                const LutCpu& lut)
{
    const int gaussScale = gaussRadius - 1;
    out.assign(std::size_t(X) * Y * Z, 0.0f);

    for (std::uint32_t vz = 0; vz < Z; ++vz) {
        for (std::uint32_t vy = 0; vy < Y; ++vy) {
            for (std::uint32_t vx = 0; vx < X; ++vx) {
                double sum = 0.0;
                double sumF = 0.0;
                for (int rz = -gaussRadius; rz <= gaussRadius; ++rz) {
                    const int iz = int(vz) + rz;
                    if (iz < int(Z) && iz > 0) {  // upstream quirk
                        const double value =
                            double(in[lin_idx(vx, vy, std::uint32_t(iz), X, Y)]);
                        const double f = lut.at(gaussScale, rz + gaussRadius);
                        sum  += value * f;
                        sumF += f;
                    }
                }
                out[lin_idx(vx, vy, vz, X, Y)] = float(sum / sumF);
            }
        }
    }
}

// ---------------- CPU reference: gaussianBlurVolumeXYZ ----------------

void blur_xyz_ref(const std::vector<float>& in,
                  std::vector<float>&       out,
                  std::uint32_t X, std::uint32_t Y, std::uint32_t Z,
                  int gaussRadius,
                  const LutCpu& lut)
{
    const int gaussScale = gaussRadius - 1;
    out.assign(std::size_t(X) * Y * Z, 0.0f);

    for (std::uint32_t vz = 0; vz < Z; ++vz) {
        for (std::uint32_t vy = 0; vy < Y; ++vy) {
            for (std::uint32_t vx = 0; vx < X; ++vx) {
                const int xMin = std::max(-gaussRadius, -int(vx));
                const int yMin = std::max(-gaussRadius, -int(vy));
                const int zMin = std::max(-gaussRadius, -int(vz));
                const int xMax = std::min( gaussRadius, int(X) - int(vx) - 1);
                const int yMax = std::min( gaussRadius, int(Y) - int(vy) - 1);
                const int zMax = std::min( gaussRadius, int(Z) - int(vz) - 1);

                double sum = 0.0;
                double sumF = 0.0;
                for (int rx = xMin; rx <= xMax; ++rx) {
                    const int ix = int(vx) + rx;
                    const double fx = lut.at(gaussScale, rx + gaussRadius);
                    for (int ry = yMin; ry <= yMax; ++ry) {
                        const int iy = int(vy) + ry;
                        const double fy = lut.at(gaussScale, ry + gaussRadius);
                        for (int rz = zMin; rz <= zMax; ++rz) {
                            const int iz = int(vz) + rz;
                            const double fz = lut.at(gaussScale, rz + gaussRadius);
                            const double value =
                                double(in[lin_idx(std::uint32_t(ix),
                                                  std::uint32_t(iy),
                                                  std::uint32_t(iz), X, Y)]);
                            const double f = fx * fy * fz;
                            sum  += value * f;
                            sumF += f;
                        }
                    }
                }
                out[lin_idx(vx, vy, vz, X, Y)] = float(sum / sumF);
            }
        }
    }
}

// ---------------- Synthetic volume ----------------

std::vector<float> make_volume(std::uint32_t X, std::uint32_t Y, std::uint32_t Z,
                               std::uint64_t seed)
{
    std::vector<float> v(std::size_t(X) * Y * Z);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> noise(-0.05f, 0.05f);
    for (std::uint32_t z = 0; z < Z; ++z) {
        for (std::uint32_t y = 0; y < Y; ++y) {
            for (std::uint32_t x = 0; x < X; ++x) {
                const float u = float(x) / float(X);
                const float w = float(y) / float(Y);
                const float t = float(z) / float(Z);
                const float val =
                      0.30f * std::sin(6.0f * u)
                    + 0.30f * std::cos(5.0f * w)
                    + 0.30f * std::sin(7.0f * t)
                    + noise(rng);
                v[lin_idx(x, y, z, X, Y)] = val;
            }
        }
    }
    return v;
}

}  // namespace

int main() try
{
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    GaussianTable  lut(dev);
    GaussianFilter gf(dev, lut);

    constexpr std::uint32_t X = 32;
    constexpr std::uint32_t Y = 24;
    constexpr std::uint32_t Z = 20;
    constexpr int           radius = 2;   // gaussScale = radius - 1 = 1
    constexpr double        kTol  = 1e-4;

    const auto src = make_volume(X, Y, Z, 0xfeed12ULL);
    LutCpu lut_cpu;
    int bad = 0;

    // ============== gaussianBlurVolumeZ ==================
    {
        Buffer vol(dev, src.size() * sizeof(float));
        vol.set_label("blur_z.vol");
        vol.upload(std::span<const float>(src));

        VolumeDims dims{ X, Y, Z };
        gf.gaussian_blur_volume_z(vol, dims, radius);

        std::vector<float> gpu(src.size());
        std::memcpy(gpu.data(), vol.data(), gpu.size() * sizeof(float));

        std::vector<float> ref;
        blur_z_ref(src, ref, X, Y, Z, radius, lut_cpu);

        double worst = 0.0;
        std::size_t worst_k = 0;
        for (std::size_t k = 0; k < gpu.size(); ++k) {
            const double e = std::abs(double(gpu[k]) - double(ref[k]));
            if (e > worst) { worst = e; worst_k = k; }
            if (e > kTol) {
                if (bad < 3) std::fprintf(stderr,
                    "blur_z voxel %zu: gpu=%g ref=%g err=%g\n",
                    k, double(gpu[k]), double(ref[k]), e);
                ++bad;
            }
        }
        std::printf(
            "[info] blur_z : X=%u Y=%u Z=%u r=%d, voxels=%zu, worst |Δ|=%.3g @ k=%zu (budget %.3g)\n",
            X, Y, Z, radius, gpu.size(), worst, worst_k, kTol);
    }

    // ============== gaussianBlurVolumeXYZ ==================
    {
        Buffer vol(dev, src.size() * sizeof(float));
        vol.set_label("blur_xyz.vol");
        vol.upload(std::span<const float>(src));

        VolumeDims dims{ X, Y, Z };
        gf.gaussian_blur_volume_xyz(vol, dims, radius);

        std::vector<float> gpu(src.size());
        std::memcpy(gpu.data(), vol.data(), gpu.size() * sizeof(float));

        std::vector<float> ref;
        blur_xyz_ref(src, ref, X, Y, Z, radius, lut_cpu);

        double worst = 0.0;
        std::size_t worst_k = 0;
        for (std::size_t k = 0; k < gpu.size(); ++k) {
            const double e = std::abs(double(gpu[k]) - double(ref[k]));
            if (e > worst) { worst = e; worst_k = k; }
            if (e > kTol) {
                if (bad < 3) std::fprintf(stderr,
                    "blur_xyz voxel %zu: gpu=%g ref=%g err=%g\n",
                    k, double(gpu[k]), double(ref[k]), e);
                ++bad;
            }
        }
        std::printf(
            "[info] blur_xyz: X=%u Y=%u Z=%u r=%d, voxels=%zu, worst |Δ|=%.3g @ k=%zu (budget %.3g)\n",
            X, Y, Z, radius, gpu.size(), worst, worst_k, kTol);
    }

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
