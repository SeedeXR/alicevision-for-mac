// simstat_kernels.metal — validation kernel for SimStat.h.
//
// Each thread runs a complete weighted-NCC accumulation over a
// fixed-size sample buffer (kSimstatSamples = 64) and outputs the
// resulting variances and similarity.
//
// Per-thread input layout (kSimstatSamples * 3 floats):
//   each sample = (x, y, w)
//
// Per-thread output layout (4 floats):
//   [0] getVarianceXW()
//   [1] getVarianceYW()
//   [2] getVarianceXYW()
//   [3] computeWSim()

#include <metal_stdlib>
#include "SimStat.h"
using namespace metal;
using namespace av_depthmap;

constant constexpr uint kSimstatSamples    = 64;
constant constexpr uint kSimstatInPerCase  = kSimstatSamples * 3;
constant constexpr uint kSimstatOutPerCase = 4;

kernel void av_simstat_validate(
    device const float* in_buf  [[buffer(0)]],
    device       float* out_buf [[buffer(1)]],
    constant     uint&  count   [[buffer(2)]],
    uint                gid     [[thread_position_in_grid]])
{
    if (gid >= count) return;

    const device float* in  = in_buf  + gid * kSimstatInPerCase;
    device       float* out = out_buf + gid * kSimstatOutPerCase;

    simStat sst;
    sst.init_zero();
    for (uint i = 0; i < kSimstatSamples; ++i) {
        const float x = in[i * 3 + 0];
        const float y = in[i * 3 + 1];
        const float w = in[i * 3 + 2];
        sst.update(x, y, w);
    }

    out[0] = sst.getVarianceXW();
    out[1] = sst.getVarianceYW();
    out[2] = sst.getVarianceXYW();
    out[3] = sst.computeWSim();
}
