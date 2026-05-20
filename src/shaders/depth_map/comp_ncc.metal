// comp_ncc.metal — kernel wrapper around compNCCby3DptsYK from
// Patch.h. One thread = one Patch hypothesis = one similarity score.
//
// Layout:
//   buffer(0) : rc DeviceCameraParams (constant)
//   buffer(1) : tc DeviceCameraParams (constant)
//   buffer(2) : per-case Patch array
//   buffer(3) : per-case similarity output (float)
//   buffer(4) : CompNCCParams (level widths, mipmap level, wsh, γ, flags)
//   buffer(5) : count
//   texture(0): rc mipmapped image
//   texture(1): tc mipmapped image
//
// The host bakes both PSO variants (filter on / off) into the same
// metallib; the test exercises the unfiltered variant first.

#include <metal_stdlib>
#include "Patch.h"
using namespace metal;
using namespace av_depthmap;

struct CompNCCParams {
    uint  rcLevelWidth;
    uint  rcLevelHeight;
    uint  tcLevelWidth;
    uint  tcLevelHeight;
    float mipmapLevel;
    int   wsh;
    float invGammaC;
    float invGammaP;
    uint  useConsistentScale;   // 0/1, packed bool
};

kernel void av_compNCC_validate_no_filter(
    constant DeviceCameraParams&     rc       [[buffer(0)]],
    constant DeviceCameraParams&     tc       [[buffer(1)]],
    device const Patch*              patches  [[buffer(2)]],
    device       float*              out      [[buffer(3)]],
    constant CompNCCParams&          params   [[buffer(4)]],
    constant uint&                   count    [[buffer(5)]],
    texture2d<float, access::sample> rcMipmap [[texture(0)]],
    texture2d<float, access::sample> tcMipmap [[texture(1)]],
    uint                             gid      [[thread_position_in_grid]])
{
    constexpr sampler smp(coord::normalized,
                          address::clamp_to_edge,
                          filter::linear,
                          mip_filter::linear);
    if (gid >= count) return;
    out[gid] = compNCCby3DptsYK<false>(
        rc, tc, rcMipmap, tcMipmap, smp,
        params.rcLevelWidth, params.rcLevelHeight,
        params.tcLevelWidth, params.tcLevelHeight,
        params.mipmapLevel, params.wsh,
        params.invGammaC, params.invGammaP,
        params.useConsistentScale != 0u,
        patches[gid]);
}

kernel void av_compNCC_validate_filter(
    constant DeviceCameraParams&     rc       [[buffer(0)]],
    constant DeviceCameraParams&     tc       [[buffer(1)]],
    device const Patch*              patches  [[buffer(2)]],
    device       float*              out      [[buffer(3)]],
    constant CompNCCParams&          params   [[buffer(4)]],
    constant uint&                   count    [[buffer(5)]],
    texture2d<float, access::sample> rcMipmap [[texture(0)]],
    texture2d<float, access::sample> tcMipmap [[texture(1)]],
    uint                             gid      [[thread_position_in_grid]])
{
    constexpr sampler smp(coord::normalized,
                          address::clamp_to_edge,
                          filter::linear,
                          mip_filter::linear);
    if (gid >= count) return;
    out[gid] = compNCCby3DptsYK<true>(
        rc, tc, rcMipmap, tcMipmap, smp,
        params.rcLevelWidth, params.rcLevelHeight,
        params.tcLevelWidth, params.tcLevelHeight,
        params.mipmapLevel, params.wsh,
        params.invGammaC, params.invGammaP,
        params.useConsistentScale != 0u,
        patches[gid]);
}

// ----------------------------------------------------------------
// Custom-patch-pattern variants. Same as above but `wsh` becomes
// per-subpart and the spatial sampling is driven by a constant
// DevicePatchPattern bound at buffer(6).
// ----------------------------------------------------------------

kernel void av_compNCC_customPattern_no_filter(
    constant DeviceCameraParams&     rc       [[buffer(0)]],
    constant DeviceCameraParams&     tc       [[buffer(1)]],
    device const Patch*              patches  [[buffer(2)]],
    device       float*              out      [[buffer(3)]],
    constant CompNCCParams&          params   [[buffer(4)]],
    constant uint&                   count    [[buffer(5)]],
    constant DevicePatchPattern&     pattern  [[buffer(6)]],
    texture2d<float, access::sample> rcMipmap [[texture(0)]],
    texture2d<float, access::sample> tcMipmap [[texture(1)]],
    uint                             gid      [[thread_position_in_grid]])
{
    constexpr sampler smp(coord::normalized,
                          address::clamp_to_edge,
                          filter::linear,
                          mip_filter::linear);
    if (gid >= count) return;
    out[gid] = compNCCby3DptsYK_customPatchPattern<false>(
        rc, tc, rcMipmap, tcMipmap, smp,
        params.rcLevelWidth, params.rcLevelHeight,
        params.tcLevelWidth, params.tcLevelHeight,
        params.mipmapLevel,
        params.invGammaC, params.invGammaP,
        params.useConsistentScale != 0u,
        pattern,
        patches[gid]);
}

kernel void av_compNCC_customPattern_filter(
    constant DeviceCameraParams&     rc       [[buffer(0)]],
    constant DeviceCameraParams&     tc       [[buffer(1)]],
    device const Patch*              patches  [[buffer(2)]],
    device       float*              out      [[buffer(3)]],
    constant CompNCCParams&          params   [[buffer(4)]],
    constant uint&                   count    [[buffer(5)]],
    constant DevicePatchPattern&     pattern  [[buffer(6)]],
    texture2d<float, access::sample> rcMipmap [[texture(0)]],
    texture2d<float, access::sample> tcMipmap [[texture(1)]],
    uint                             gid      [[thread_position_in_grid]])
{
    constexpr sampler smp(coord::normalized,
                          address::clamp_to_edge,
                          filter::linear,
                          mip_filter::linear);
    if (gid >= count) return;
    out[gid] = compNCCby3DptsYK_customPatchPattern<true>(
        rc, tc, rcMipmap, tcMipmap, smp,
        params.rcLevelWidth, params.rcLevelHeight,
        params.tcLevelWidth, params.tcLevelHeight,
        params.mipmapLevel,
        params.invGammaC, params.invGammaP,
        params.useConsistentScale != 0u,
        pattern,
        patches[gid]);
}
