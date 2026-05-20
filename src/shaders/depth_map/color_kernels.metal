// color_kernels.metal — validation kernel for color.h helpers.
//
// Per-case input layout (12 floats):
//   [0..2]   c0 (a sample RGB, used through the conversion chain)
//   [3..5]   c1 (a second RGB, partner for distance/cost)
//   [6]      dx (cast to int inside the kernel)
//   [7]      dy
//   [8]      invGammaC
//   [9]      invGammaP
//   [10..11] padding (reserved; keeps power-of-two alignment when
//                     we batch many cases)
//
// Per-case output layout (15 floats):
//   [0..2]   srgb2rgb(c0)
//   [3..5]   rgb2xyz(c0_linear) where c0_linear = srgb2rgb(c0)
//   [6..8]   rgb2hsl(c0_linear)
//   [9..11]  xyz2lab(rgb2xyz(c0_linear))     -- the full chain
//   [12]     euclideanDist3(c0, c1)
//   [13]     CostYKfromLab(dx, dy, (c0,1), (c1,1), invGammaC, invGammaP)
//   [14]     CostYKfromLab((c0,1), (c1,1), invGammaC)
//
// Note: CostYKfromLab takes float4 (the alpha component is unused
// by `euclideanDist3(float4, float4)` which reads only .xyz), so
// we synthesize alpha=1.

#include <metal_stdlib>
#include "color.h"
using namespace metal;
using namespace av_depthmap;

constant constexpr uint kColorInPerCase  = 12;
constant constexpr uint kColorOutPerCase = 15;

kernel void av_color_validate(
    device const float* in_buf  [[buffer(0)]],
    device       float* out_buf [[buffer(1)]],
    constant     uint&  count   [[buffer(2)]],
    uint                gid     [[thread_position_in_grid]])
{
    if (gid >= count) return;

    const device float* in  = in_buf  + gid * kColorInPerCase;
    device       float* out = out_buf + gid * kColorOutPerCase;

    const float3 c0 = float3(in[0], in[1], in[2]);
    const float3 c1 = float3(in[3], in[4], in[5]);
    const int    dx = int(in[6]);
    const int    dy = int(in[7]);
    const float  invGammaC = in[8];
    const float  invGammaP = in[9];

    const float3 c0_linear = srgb2rgb(c0);
    const float3 xyz       = rgb2xyz(c0_linear);
    const float3 hsl       = rgb2hsl(c0_linear);
    const float3 lab       = xyz2lab(xyz);

    out[0]  = c0_linear.x; out[1]  = c0_linear.y; out[2]  = c0_linear.z;
    out[3]  = xyz.x;       out[4]  = xyz.y;       out[5]  = xyz.z;
    out[6]  = hsl.x;       out[7]  = hsl.y;       out[8]  = hsl.z;
    out[9]  = lab.x;       out[10] = lab.y;       out[11] = lab.z;

    out[12] = euclideanDist3(c0, c1);

    const float4 c0h(c0.x, c0.y, c0.z, 1.0f);
    const float4 c1h(c1.x, c1.y, c1.z, 1.0f);
    out[13] = CostYKfromLab(dx, dy, c0h, c1h, invGammaC, invGammaP);
    out[14] = CostYKfromLab(c0h, c1h, invGammaC);
}
