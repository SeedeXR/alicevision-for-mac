// color_conversion.metal — Metal port of
// depthMap/cuda/imageProcessing/deviceColorConversion.cu.
//
// Single in-place per-pixel kernel: rgb2lab over an entire image.
// Input/output texture format: RGBA32Float (so we can preserve the
// out-of-range a*/b* values from xyz2lab without truncation, matching
// upstream's TODO note about needing float textures).
//
// Notably, upstream treats the input RGB as *linear*, not sRGB —
// `srgb2rgb` is intentionally skipped in the CUDA original. We
// preserve that behavior here.

#include <metal_stdlib>
#include "color.h"
using namespace metal;
using namespace av_depthmap;

kernel void av_rgb2lab(
    texture2d<float, access::read_write> img [[texture(0)]],
    uint2                                gid [[thread_position_in_grid]])
{
    const uint w = img.get_width();
    const uint h = img.get_height();
    if (gid.x >= w || gid.y >= h) return;

    const float4 rgb = img.read(gid);

    // Treat the input as linear RGB in [0, 255]. The CUDA original
    // applies `(1/255)` to bring it into [0, 1] before rgb2xyz.
    constexpr float kInv255 = 1.0f / 255.0f;
    const float3 lab = xyz2lab(rgb2xyz(float3(rgb.x * kInv255,
                                              rgb.y * kInv255,
                                              rgb.z * kInv255)));

    img.write(float4(lab.x, lab.y, lab.z, rgb.w), gid);
}
