// texture_smoke.metal — proves the MTLTexture + sampler + mipmap
// pipeline works end-to-end through av::gpu::Texture.
//
// The kernel samples a 2D texture at a list of (u, v, mip) probes
// and writes the float4 results to a buffer. The host verifies:
//   (1) bilinear interpolation on mip 0 matches the analytical
//       average of the 4 neighboring texels;
//   (2) sampling at mip level k recovers an average consistent
//       with the box-filter mipmap that `generateMipmaps` builds.
//
// The sampler is declared with `constexpr sampler` so we don't
// have to wire an MTLSamplerState from the CPU side — sufficient
// for the smoke test. Real depthMap kernels will follow the same
// pattern.

#include <metal_stdlib>
using namespace metal;

struct TexProbe {
    float u;      // normalized [0, 1]
    float v;      // normalized [0, 1]
    float level;  // mipmap level (continuous)
};

kernel void av_texture_sample(
    texture2d<float, access::sample> tex      [[texture(0)]],
    device const TexProbe*           probes   [[buffer(0)]],
    device       float4*             out      [[buffer(1)]],
    constant     uint&               count    [[buffer(2)]],
    uint                             gid      [[thread_position_in_grid]])
{
    constexpr sampler smp(coord::normalized,
                          address::clamp_to_edge,
                          filter::linear,
                          mip_filter::linear);

    if (gid >= count) return;
    const TexProbe p = probes[gid];
    out[gid] = tex.sample(smp, float2(p.u, p.v), level(p.level));
}
