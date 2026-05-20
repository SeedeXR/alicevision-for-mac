// patch_kernels.metal — validation kernel for Patch.h geometry
// helpers. Exercises four load-bearing routines per thread:
//
//   triangulateMatchRef
//   computeRotCSEpip
//   computeHomography
//   refineDepthSubPixel
//
// Per-case input layout (16 floats):
//   [0..2]   patchPoint p (used as a 3D world point)
//   [3..5]   patchNormal n (unit, used by computeHomography)
//   [6..7]   refpix (used by triangulateMatchRef)
//   [8..9]   tarpix (used by triangulateMatchRef)
//   [10..12] depths (3 candidates: dM1, d, dP1)
//   [13..15] sims   (3 similarity scores at those depths)
//
// Per-case output layout (22 floats):
//   [0..2]   triangulated 3D point
//   [3..5]   epipolar y axis (after computeRotCSEpip)
//   [6..8]   epipolar n axis
//   [9..11]  epipolar x axis
//   [12..20] homography H (column-major 3x3 flat)
//   [21]     refined depth from refineDepthSubPixel

#include <metal_stdlib>
#include "Patch.h"
using namespace metal;
using namespace av_depthmap;

constant constexpr uint kPatchInPerCase  = 16;
constant constexpr uint kPatchOutPerCase = 22;

kernel void av_patch_validate(
    constant DeviceCameraParams& rc       [[buffer(0)]],
    constant DeviceCameraParams& tc       [[buffer(1)]],
    device   const float*        in_buf   [[buffer(2)]],
    device         float*        out_buf  [[buffer(3)]],
    constant uint&               count    [[buffer(4)]],
    uint                         gid      [[thread_position_in_grid]])
{
    if (gid >= count) return;

    const device float* in  = in_buf  + gid * kPatchInPerCase;
    device       float* out = out_buf + gid * kPatchOutPerCase;

    const float3 p      = float3(in[0], in[1], in[2]);
    const float3 n      = float3(in[3], in[4], in[5]);
    const float2 refpix = float2(in[6], in[7]);
    const float2 tarpix = float2(in[8], in[9]);
    const float3 depths = float3(in[10], in[11], in[12]);
    const float3 sims   = float3(in[13], in[14], in[15]);

    // 1. triangulateMatchRef
    const float3 tri = triangulateMatchRef(rc, tc, refpix, tarpix);
    out[0] = tri.x; out[1] = tri.y; out[2] = tri.z;

    // 2. computeRotCSEpip — needs a Patch object whose 3D point is `p`.
    Patch ptch;
    ptch.p = p;
    ptch.n = float3(0.0f);   // ignored
    ptch.x = float3(0.0f);   // ignored
    ptch.y = float3(0.0f);   // ignored
    ptch.d = 0.0f;
    computeRotCSEpip(ptch, rc, tc);
    out[3] = ptch.y.x; out[4] = ptch.y.y; out[5] = ptch.y.z;
    out[6] = ptch.n.x; out[7] = ptch.n.y; out[8] = ptch.n.z;
    out[9] = ptch.x.x; out[10] = ptch.x.y; out[11] = ptch.x.z;

    // 3. computeHomography
    thread float H[9];
    computeHomography(H, rc, tc, p, n);
    for (int i = 0; i < 9; ++i) out[12 + i] = H[i];

    // 4. refineDepthSubPixel
    out[21] = refineDepthSubPixel(depths, sims);
}
