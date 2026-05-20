// mipmap_array.metal — Metal port of the custom mipmap-construction
// kernel from depthMap/cuda/imageProcessing/deviceMipmappedArray.cu:
//
//   createMipmappedArrayLevel_kernel<TRadius=2>
//       For each output pixel (x, y) at level N, samples a
//       (2·TRadius+1)² stencil from level N-1 using bilinear texture
//       fetch, applies separable Gaussian weights at upstream scale
//       index 1 (radius=2, delta=1.0), and normalizes by the sum of
//       weights. The sampling pattern matches upstream verbatim:
//
//           u = (x + j + 0.5f) / outWidth
//           v = (y + i + 0.5f) / outHeight
//
//       where outWidth/outHeight are the *current* level dims. Each
//       horizontal/vertical neighbor j/i shifts by one *output* texel
//       (= two input texels), so the stencil covers 5 output-units
//       across, which the bilinear filter resolves against the
//       2×-larger source. This is intentionally NOT a textbook 2×
//       box-average — it's the upstream-specific weighting that we
//       reproduce for numerical parity.
//
// The companion debug-flat-image kernel (`createMipmappedArray
// DebugFlatImage_kernel`) is upstream-only debug machinery; we do
// not port it.
//
// Output goes to a flat float4 buffer (one float4 per output texel,
// row-major, no padding). The host then `replaceRegion`s that buffer
// into the destination mip level.

#include <metal_stdlib>
using namespace metal;

inline float av_mip_getGauss(constant const float* weights,
                             constant const int*   offsets,
                             int scale, int idx)
{
    return weights[offsets[scale] + idx];
}

struct MipmapLevelParams {
    uint width;   // output (level N) width
    uint height;  // output (level N) height
    int  radius;  // upstream hardcodes TRadius=2; passed in to allow
                  // future variants (used as scale index = radius-1).
};

kernel void av_create_mipmapped_array_level(
    texture2d<float, access::sample> in_prev_level [[texture(0)]],
    device   float4*                 out_level     [[buffer(0)]],
    constant const float*            gauss_w       [[buffer(1)]],
    constant const int*              gauss_off     [[buffer(2)]],
    constant MipmapLevelParams&      p             [[buffer(3)]],
    uint2                            gid           [[thread_position_in_grid]])
{
    // Bilinear, clamp-to-edge, normalized coords — matches upstream
    // CUDA texture descriptor (filterMode=Linear, addressMode=Clamp,
    // normalizedCoords=1).
    constexpr sampler smp(coord::normalized,
                          address::clamp_to_edge,
                          filter::linear);

    if (gid.x >= p.width || gid.y >= p.height)
        return;

    const float px = 1.0f / float(p.width);
    const float py = 1.0f / float(p.height);

    // Upstream uses scale index 1 (radius=2) which gives weights for
    // delta=1.0 of size 5. We pass it in as `radius` so the LUT
    // lookup is `scale = radius - 1`. For the upstream call this is
    // always 1, but we keep it general.
    const int R     = p.radius;
    const int scale = R - 1;

    float4 sumColor  = float4(0.0f);
    float  sumFactor = 0.0f;

    for (int i = -R; i <= R; ++i) {
        for (int j = -R; j <= R; ++j) {
            // domain factor (separable Gaussian)
            const float factor =
                av_mip_getGauss(gauss_w, gauss_off, scale, i + R) *
                av_mip_getGauss(gauss_w, gauss_off, scale, j + R);

            // normalized coords in the output grid — bilinear filter
            // pulls the corresponding location from the 2×-larger
            // previous level.
            const float u = (float(gid.x) + float(j) + 0.5f) * px;
            const float v = (float(gid.y) + float(i) + 0.5f) * py;

            const float4 color = in_prev_level.sample(smp, float2(u, v));

            sumColor  += color * factor;
            sumFactor += factor;
        }
    }

    out_level[gid.y * p.width + gid.x] = sumColor / sumFactor;
}
