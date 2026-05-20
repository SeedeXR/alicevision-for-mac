// hello.metal — smoke-test kernels for the av::gpu skeleton.
//
// Two kernels:
//   * av_hello_saxpy: classic Y = aX + Y to prove buffer plumbing
//                     and float compute.
//   * av_hello_simdsum: per-threadgroup reduction using simd_sum
//                       to prove SIMD-group functions work.
//
// Keep this file MSL-correct and conservative; it is the first
// kernel that proves our entire .metal → .air → .metallib →
// MTL::Device::newDefaultLibrary() pipeline.

#include <metal_stdlib>
using namespace metal;

struct SaxpyParams {
    uint  count;
    float a;
};

kernel void av_hello_saxpy(
    device       float*       y      [[buffer(0)]],
    device const float*       x      [[buffer(1)]],
    constant     SaxpyParams& params [[buffer(2)]],
    uint                      gid    [[thread_position_in_grid]])
{
    if (gid >= params.count) return;
    y[gid] = params.a * x[gid] + y[gid];
}

// Each threadgroup reduces its `tg_size` chunk via simd_sum and
// writes a single float to `partials[tg_id]`. Caller sums the
// partials on CPU (or in a second pass) to obtain the total.
//
// Threadgroup size MUST equal SIMD width (32) for this kernel to
// be a single-SIMD-group reduction. The host picks the group
// size accordingly.

kernel void av_hello_simdsum(
    device const float* x        [[buffer(0)]],
    device       float* partials [[buffer(1)]],
    constant     uint&   count    [[buffer(2)]],
    uint                 gid      [[thread_position_in_grid]],
    uint                 tg_id    [[threadgroup_position_in_grid]],
    uint                 tg_size  [[threads_per_threadgroup]])
{
    float v = (gid < count) ? x[gid] : 0.0f;
    float s = simd_sum(v);
    if ((gid % tg_size) == 0u) {
        partials[tg_id] = s;
    }
}
