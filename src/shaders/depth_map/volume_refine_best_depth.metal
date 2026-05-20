// volume_refine_best_depth.metal — port of
// volume_refineBestDepth_kernel from upstream's
// deviceSimilarityVolumeKernels.cuh:515.
//
// Exit point of the Refine pipeline: per pixel (vx, vy), find
// the sub-sample offset around the SGM mid depth that maximizes
// a Gaussian-weighted sum of the FP16 invert-and-filtered NCC
// values from `refine_similarity`. Then convert the winning
// sample offset into an actual depth and write
// (best_depth, best_sample_sim) to the output map.
//
// Note: unlike `retrieve_best_depth` (S11), this kernel does
// *not* use any camera parameters. The depth is computed as
// `sgm_depth + sample_offset * (sgm_pix_size / samplesPerPixSize)`
// where sgm_depth is already a Euclidean depth from the SGM exit
// stage. No back-projection or texture work.
//
// Per-pixel work (mirrors upstream lines 525-593):
//   1. Read (sgm_depth, sgm_pix_size) at (vx, vy).
//   2. If sgm_depth ≤ 0, write the invalid value and return.
//   3. Sweep `sample = -halfNbSamples ... +halfNbSamples`. For
//      each sample:
//        for vz in [0, volDimZ):
//            zs = (vz - halfNbDepths) * samplesPerPixSize
//            simSum = -half2float(vol[vz])    // flip convention: best = lowest
//            sampleSim += simSum * exp(-(zs - sample)² / (2σ²))
//      Keep the sample with the lowest sampleSim.
//   4. sample_size = sgm_pix_size / samplesPerPixSize
//      best_depth  = sgm_depth + bestSampleOffsetIdx * sample_size
//      Write (best_depth, best_sample_sim).

#include <metal_stdlib>
using namespace metal;

struct RefineBestDepthParams {
    uint  volDimX;
    uint  volDimY;
    uint  volDimZ;
    int   samplesPerPixSize;
    int   halfNbSamples;        // sub-sample sweep range per side
    int   halfNbDepths;         // == (volDimZ - 1) / 2 typically
    float twoTimesSigmaPowerTwo; // = 2 σ²
    uint  roiWidth;
    uint  roiHeight;
};

kernel void av_volume_refine_best_depth(
    device       float2*           out_depth_sim_map [[buffer(0)]],
    device const float2*           in_sgm_dp_map     [[buffer(1)]],
    device const half*             in_vol_sim_half   [[buffer(2)]],
    constant RefineBestDepthParams& p                [[buffer(3)]],
    uint2                          gid               [[thread_position_in_grid]])
{
    if (gid.x >= p.roiWidth || gid.y >= p.roiHeight) return;

    const uint vx = gid.x;
    const uint vy = gid.y;
    const uint pix_k = vy * p.volDimX + vx;

    const float2 sgm = in_sgm_dp_map[pix_k];

    // Invalid / masked: pass-through.
    if (sgm.x <= 0.0f) {
        out_depth_sim_map[pix_k] = float2(sgm.x, 1.0f);
        return;
    }

    // Sliding-Gaussian sweep.
    float bestSampleSim     = 0.0f;   // all valid sums are ≤ 0
    int   bestSampleOffsetIdx = 0;    // default: middle depth (SGM)

    for (int sample = -p.halfNbSamples; sample <= p.halfNbSamples; ++sample) {
        float sampleSim = 0.0f;
        for (uint vz = 0; vz < p.volDimZ; ++vz) {
            const int rz = int(vz) - p.halfNbDepths;
            const int zs = rz * p.samplesPerPixSize;
            const uint vol_k = vz * (p.volDimX * p.volDimY) + pix_k;
            const float invSimSum = float(in_vol_sim_half[vol_k]);
            const float simSum    = -invSimSum;     // flip: best = lowest
            const float d         = float(zs - sample);
            sampleSim += simSum * exp(-(d * d) / p.twoTimesSigmaPowerTwo);
        }
        if (sampleSim < bestSampleSim) {
            bestSampleSim       = sampleSim;
            bestSampleOffsetIdx = sample;
        }
    }

    const float sample_size = sgm.y / float(p.samplesPerPixSize);
    const float offset      = float(bestSampleOffsetIdx) * sample_size;
    const float best_depth  = sgm.x + offset;
    out_depth_sim_map[pix_k] = float2(best_depth, bestSampleSim);
}
