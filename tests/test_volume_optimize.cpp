// test_volume_optimize.cpp — validation of the SGM 4-direction DP
// aggregation (volume_optimize) ported in Session 13.
//
// Strategy:
//   1. Build a synthetic uchar cost volume (small: 12×8×6 = 576
//      voxels) with a smooth-but-non-trivial pattern that has a
//      clear "winner" Z per pixel.
//   2. Run our `Volume::optimize` over it.
//   3. CPU FP64 reference re-implements the same 4-path DP
//      aggregation algorithm verbatim from upstream:
//        path order: (Y forward, Y reverse, X forward, X reverse)
//        per path:   init out[Y=y0] = 255, copy Ym1 = in[Y=y0],
//                     for iy=1..axDimY-1:
//                       bestSimInYm1[x] = min over Z of Ym1[x, z]
//                       Y = in[Y=y]
//                       per (x, z) thread DP update + write to slice
//                                                    + aggregate into out
//                       swap Y ↔ Ym1
//   4. Compare GPU vs CPU per-voxel (bit-exact agreement expected
//      — both use the same fixed-P2 path and identical arithmetic).

#include "av/depth_map/Volume.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::uint32_t W = 12;
constexpr std::uint32_t H = 8;
constexpr std::uint32_t D = 6;

constexpr float kP1     = 10.0f;
constexpr float kP2_abs = 100.0f;

// ---- linear index helpers ----
inline std::size_t vol_idx(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return std::size_t(z) * (W * H) + std::size_t(y) * W + x;
}

// ---- synthetic input ----
std::vector<std::uint8_t> make_input_volume(std::uint64_t seed)
{
    std::vector<std::uint8_t> v(W * H * D);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> U(0, 254);
    // Place a low-cost stripe at a per-pixel random Z, with higher
    // costs elsewhere. This gives the DP something to do (it should
    // smooth the per-pixel-best-Z surface).
    for (std::uint32_t y = 0; y < H; ++y) {
        for (std::uint32_t x = 0; x < W; ++x) {
            const int true_z = int(D / 2) + ((int(x) - int(W) / 2 + int(y) - int(H) / 2) % 2);
            for (std::uint32_t z = 0; z < D; ++z) {
                const int delta = std::abs(int(z) - true_z);
                v[vol_idx(x, y, z)] = std::uint8_t(std::min(254, 30 + delta * 60 + U(rng) % 5));
            }
        }
    }
    return v;
}

// ---- CPU reference: SGM 4-direction DP, verbatim port of MSL kernel ----

struct OptimizeRef {
    std::vector<std::uint8_t> out;
    std::vector<std::uint32_t> slice_a;
    std::vector<std::uint32_t> slice_b;
    std::vector<std::uint32_t> axis_acc;

    OptimizeRef()
        : out(W * H * D, 0u)
        , slice_a(std::max(W, H) * D, 0u)
        , slice_b(std::max(W, H) * D, 0u)
        , axis_acc(std::max(W, H), 0u)
    {}

    // axisT swizzle: axis[0/1/2] map iteration coords to volume coords.
    struct Axis { int a0, a1, a2; };

    static std::size_t vol_k(int vx, int vy, int vz) {
        return std::size_t(vz) * (W * H)
             + std::size_t(vy) * W
             + std::size_t(vx);
    }

    static std::uint32_t vol_dim(int i) {
        switch (i) {
            case 0:  return W;
            case 1:  return H;
            default: return D;
        }
    }

    // Apply axisT swizzle: out (vx,vy,vz) given iteration (x_iter, y, z_iter).
    static void axis_to_v(int x_iter, int y, int z_iter, Axis a,
                          int& vx, int& vy, int& vz) {
        int v[3] = { 0, 0, 0 };
        v[a.a0] = x_iter;
        v[a.a1] = y;
        v[a.a2] = z_iter;
        vx = v[0]; vy = v[1]; vz = v[2];
    }

    void run(const std::vector<std::uint8_t>& in) {
        // Initial state of `out` is irrelevant (the algorithm writes
        // every voxel). Match the GPU buffer's pre-fill: any value.
        std::fill(out.begin(), out.end(), 0u);

        const Axis paths[4] = {
            { 0, 1, 2 },   // Y forward
            { 0, 1, 2 },   // Y reverse
            { 1, 0, 2 },   // X forward
            { 1, 0, 2 },   // X reverse
        };
        const bool inv_y[4] = { false, true, false, true };

        std::uint32_t filteringIndex = 0;
        for (int pi = 0; pi < 4; ++pi) {
            const Axis a = paths[pi];
            const bool invY = inv_y[pi];

            const std::uint32_t axDimX = vol_dim(a.a0);
            const std::uint32_t axDimY = vol_dim(a.a1);
            const std::uint32_t axDimZ = vol_dim(a.a2);

            std::vector<std::uint32_t>* slice_for_y   = &slice_a;
            std::vector<std::uint32_t>* slice_for_ym1 = &slice_b;

            // Step 1: copy in_vol's Y=y0 plane into ym1 slice.
            {
                const int y0 = invY ? int(axDimY) - 1 : 0;
                for (std::uint32_t zi = 0; zi < axDimZ; ++zi) {
                    for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                        int vx, vy, vz;
                        axis_to_v(int(xi), y0, int(zi), a, vx, vy, vz);
                        (*slice_for_ym1)[zi * axDimX + xi] =
                            std::uint32_t(in[vol_k(vx, vy, vz)]);
                    }
                }
            }

            // Step 2: set out[Y=y0] plane to 255.
            {
                const int y0 = invY ? int(axDimY) - 1 : 0;
                for (std::uint32_t zi = 0; zi < axDimZ; ++zi) {
                    for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                        int vx, vy, vz;
                        axis_to_v(int(xi), y0, int(zi), a, vx, vy, vz);
                        out[vol_k(vx, vy, vz)] = 255u;
                    }
                }
            }

            // Step 3: Y-loop.
            for (std::uint32_t iy = 1; iy < axDimY; ++iy) {
                const int y = invY ? int(axDimY) - 1 - int(iy) : int(iy);

                // 3a. Best Z in Ym1 column.
                for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                    std::uint32_t best = (*slice_for_ym1)[xi];
                    for (std::uint32_t zi = 1; zi < axDimZ; ++zi) {
                        const std::uint32_t v = (*slice_for_ym1)[zi * axDimX + xi];
                        if (v < best) best = v;
                    }
                    axis_acc[xi] = best;
                }

                // 3b. Copy in_vol's Y=y plane into slice_for_y.
                for (std::uint32_t zi = 0; zi < axDimZ; ++zi) {
                    for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                        int vx, vy, vz;
                        axis_to_v(int(xi), y, int(zi), a, vx, vy, vz);
                        (*slice_for_y)[zi * axDimX + xi] =
                            std::uint32_t(in[vol_k(vx, vy, vz)]);
                    }
                }

                // 3c. Aggregate DP step.
                for (std::uint32_t zi = 0; zi < axDimZ; ++zi) {
                    for (std::uint32_t xi = 0; xi < axDimX; ++xi) {
                        const std::uint32_t slice_k = zi * axDimX + xi;
                        float pathCost = 255.0f;
                        if (int(zi) >= 1 && int(zi) < int(axDimZ) - 1) {
                            const float bestCol_m1 = float(axis_acc[xi]);
                            const float pathMDm1 =
                                float((*slice_for_ym1)[(zi - 1) * axDimX + xi]);
                            const float pathMD =
                                float((*slice_for_ym1)[zi       * axDimX + xi]);
                            const float pathMDp1 =
                                float((*slice_for_ym1)[(zi + 1) * axDimX + xi]);

                            const float A = pathMD;
                            const float B = pathMDm1 + kP1;
                            const float C = pathMDp1 + kP1;
                            const float Dv = bestCol_m1 + kP2_abs;
                            const float minCost = std::min(std::min(A, B), std::min(C, Dv));

                            const float sim_xz = float((*slice_for_y)[slice_k]);
                            pathCost = sim_xz + minCost - bestCol_m1;
                        }
                        (*slice_for_y)[slice_k] = std::uint32_t(pathCost);

                        const float clamped = std::min(255.0f, std::max(0.0f, pathCost));
                        int vx, vy, vz;
                        axis_to_v(int(xi), y, int(zi), a, vx, vy, vz);
                        const float cur = float(out[vol_k(vx, vy, vz)]);
                        const float fi  = float(filteringIndex);
                        const float merged = (cur * fi + clamped) / (fi + 1.0f);
                        out[vol_k(vx, vy, vz)] =
                            std::uint8_t(std::min(255.0f, std::max(0.0f, merged)));
                    }
                }

                std::swap(slice_for_y, slice_for_ym1);
            }
            ++filteringIndex;
        }
    }
};

}  // namespace

int main() try
{
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    const auto in_data = make_input_volume(0xa11ce5);

    // GPU run.
    Volume vol(dev);

    Buffer in_buf (dev, W * H * D * sizeof(std::uint8_t));
    Buffer out_buf(dev, W * H * D * sizeof(std::uint8_t));
    const std::uint32_t maxXY = std::max(W, H);
    Buffer slice_a (dev, maxXY * D * sizeof(std::uint32_t));
    Buffer slice_b (dev, maxXY * D * sizeof(std::uint32_t));
    Buffer axis_acc(dev, maxXY     * sizeof(std::uint32_t));
    in_buf  .set_label("optimize.in");
    out_buf .set_label("optimize.out");
    slice_a .set_label("optimize.slice_a");
    slice_b .set_label("optimize.slice_b");
    axis_acc.set_label("optimize.axis_acc");

    in_buf.upload(std::span<const std::uint8_t>(in_data));

    Volume::OptimizeParams params{};
    params.dims              = VolumeDims{ W, H, D };
    params.last_depth_index  = D;
    params.p1                = kP1;
    params.p2_abs            = kP2_abs;

    vol.optimize(out_buf, slice_a, slice_b, axis_acc, in_buf, params);

    // CPU reference.
    OptimizeRef ref;
    ref.run(in_data);

    // Compare voxel-by-voxel.
    const auto* gpu = static_cast<const std::uint8_t*>(out_buf.data());
    int bad = 0;
    int worst = 0;
    int sum_diff = 0;
    for (std::size_t i = 0; i < ref.out.size(); ++i) {
        const int diff = std::abs(int(gpu[i]) - int(ref.out[i]));
        worst = std::max(worst, diff);
        sum_diff += diff;
        if (diff > 1) {
            if (bad < 4) std::fprintf(stderr,
                "vox %zu: gpu=%d cpu=%d diff=%d\n",
                i, int(gpu[i]), int(ref.out[i]), diff);
            ++bad;
        }
    }
    const double mean_diff = double(sum_diff) / double(ref.out.size());

    std::printf("[info] voxels             : %u\n", W * H * D);
    std::printf("[info] worst |Δ| (uchar)  : %d\n", worst);
    std::printf("[info] mean  |Δ| (uchar)  : %.3f\n", mean_diff);

    // Both implementations use the same fixed-P2 path with identical
    // integer/float arithmetic. The only drift sources are:
    //   * sequential aggregation order (host loop dispatches each
    //     path; the kernel's output volume reads-modifies-writes
    //     across the 4 paths sequentially via the host swap)
    //   * float division at the aggregation step
    // Expectation: bit-exact in 99%+ of voxels; isolated 1-ULP
    // discretization rounding around boundary voxels. Budget worst
    // = 1 uchar.
    constexpr int kTolWorst = 1;
    if (worst > kTolWorst) {
        std::fprintf(stderr,
            "FAIL: worst |Δ| %d > budget %d  (bad voxels: %d)\n",
            worst, kTolWorst, bad);
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
