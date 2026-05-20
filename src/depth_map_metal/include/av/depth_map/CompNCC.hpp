#pragma once

// CompNCC — host driver for the patch-NCC kernel (the first real
// depthMap kernel). Per dispatch:
//   * 2 DeviceCameraParams (constant inputs)
//   * 2 mipmapped textures (rc, tc)
//   * 1 Patch per case
//   * 1 CompNCCParams (level widths, mipmap level, patch half-width,
//                      γ_color, γ_proximity, consistent-scale flag)
//   * 1 float output per case (similarity)

#include "av/depth_map/PatchOps.hpp"           // for DeviceCameraParams mirror
#include "av/depth_map/DevicePatchPattern.hpp" // for the custom-pattern variant

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace av::gpu { class Device; class Texture; }

namespace av::depth_map {

// Per-thread Patch hypothesis. Must match MSL's `Patch` struct in
// src/shaders/depth_map/Patch.h byte-for-byte. MSL uses float3
// (16-byte aligned) for vec3 members; CPU mirror uses
// simd-style 16-byte-aligned wrappers via explicit padding.
struct PatchCase {
    float p [3]; float _pad0;
    float n [3]; float _pad1;
    float x [3]; float _pad2;
    float y [3]; float _pad3;
    float d;
    float _pad4[3];
};
static_assert(sizeof(PatchCase) == 80,
              "PatchCase must mirror MSL Patch with 16B vec3 alignment.");

struct CompNCCParams {
    std::uint32_t rcLevelWidth   = 0;
    std::uint32_t rcLevelHeight  = 0;
    std::uint32_t tcLevelWidth   = 0;
    std::uint32_t tcLevelHeight  = 0;
    float         mipmapLevel    = 0.0f;
    std::int32_t  wsh            = 4;
    float         invGammaC      = 1.0f / 20.0f;
    float         invGammaP      = 1.0f / 4.0f;
    std::uint32_t useConsistentScale = 0;
};

class CompNCC {
public:
    CompNCC(av::gpu::Device& dev,
            const DeviceCameraParams& rc,
            const DeviceCameraParams& tc);

    CompNCC(const CompNCC&)            = delete;
    CompNCC& operator=(const CompNCC&) = delete;
    CompNCC(CompNCC&&) noexcept;
    CompNCC& operator=(CompNCC&&) noexcept;
    ~CompNCC();

    // Run `patches.size()` cases through the unfiltered variant
    // (compNCCby3DptsYK<false>). Output similarity scores in `out`
    // (one float per case).
    void run_no_filter(std::span<const PatchCase> patches,
                       std::span<float>           out,
                       const av::gpu::Texture&    rc_mipmap,
                       const av::gpu::Texture&    tc_mipmap,
                       const CompNCCParams&       params);

    // Same but uses compNCCby3DptsYK<true> (sigmoid post-processing).
    void run_filter(std::span<const PatchCase> patches,
                    std::span<float>           out,
                    const av::gpu::Texture&    rc_mipmap,
                    const av::gpu::Texture&    tc_mipmap,
                    const CompNCCParams&       params);

    // Custom-patch-pattern variants. Same I/O contract as
    // `run_no_filter` / `run_filter`, but the spatial sampling is
    // driven by `pattern` (uploaded as constant bytes), which carries
    // its own per-subpart `wsh`, `level`, `downscale`, `weight`,
    // and circle/full mode. The `wsh` field of `params` is ignored
    // by this path; everything else (mipmapLevel, γs, etc.) still
    // applies.
    void run_no_filter_custom_pattern(
        std::span<const PatchCase> patches,
        std::span<float>           out,
        const av::gpu::Texture&    rc_mipmap,
        const av::gpu::Texture&    tc_mipmap,
        const CompNCCParams&       params,
        const DevicePatchPattern&  pattern);

    void run_filter_custom_pattern(
        std::span<const PatchCase> patches,
        std::span<float>           out,
        const av::gpu::Texture&    rc_mipmap,
        const av::gpu::Texture&    tc_mipmap,
        const CompNCCParams&       params,
        const DevicePatchPattern&  pattern);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
