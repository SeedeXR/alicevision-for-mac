#pragma once

// PatchOps — host driver for the Patch.h geometry validation
// kernel.
//
// Per the upstream layout, DeviceCameraParams packs 6 matrices
// (P, iP, R, iR, K, iK) and 4 column vectors (C, X, Y, Z) into
// 276 bytes (3 floats per packed vector). The CPU mirror struct
// below matches that layout exactly so it can be passed straight
// through `set_bytes` to a `constant` MSL binding.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace av::gpu { class Device; }

namespace av::depth_map {

// CPU mirror of MSL `DeviceCameraParams`. Field order, sizes, and
// packing must match the MSL struct in src/shaders/depth_map/Patch.h.
struct DeviceCameraParams {
    float P [12];   // 3x4 camera matrix      (column-major)
    float iP[9];    // inverse intrinsic·R    (column-major)
    float R [9];    // rotation               (column-major)
    float iR[9];    // R transpose            (column-major)
    float K [9];    // intrinsic              (column-major)
    float iK[9];    // inverse intrinsic      (column-major)
    float C [3];    // camera center
    float XVect[3]; // camera x axis
    float YVect[3]; // camera y axis
    float ZVect[3]; // camera z axis (look)
};
static_assert(sizeof(DeviceCameraParams) == (12 + 9 + 9 + 9 + 9 + 9 + 3 + 3 + 3 + 3) * sizeof(float),
              "DeviceCameraParams layout must be tightly packed to match MSL packed_float3.");

class PatchOps {
public:
    static constexpr std::size_t kInPerCase  = 16;
    static constexpr std::size_t kOutPerCase = 22;

    PatchOps(av::gpu::Device& dev,
             const DeviceCameraParams& rc,
             const DeviceCameraParams& tc);

    PatchOps(const PatchOps&)            = delete;
    PatchOps& operator=(const PatchOps&) = delete;
    PatchOps(PatchOps&&) noexcept;
    PatchOps& operator=(PatchOps&&) noexcept;
    ~PatchOps();

    void validate(std::span<const float> inputs,
                  std::span<float>        outputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
