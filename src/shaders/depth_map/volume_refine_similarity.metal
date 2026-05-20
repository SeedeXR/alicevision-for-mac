// volume_refine_similarity.metal — port of
// volume_refineSimilarity_kernel from upstream's
// deviceSimilarityVolumeKernels.cuh.
//
// Refine pass: operates on the FP16 (TSimRefine = half) cost
// volume. Unlike compute_similarity which sweeps an explicit list
// of depth planes, refine works around a per-pixel "middle depth"
// supplied by the upstream SGM pass — each Z slice of the volume
// corresponds to a sub-pixel offset (`(vz - center_z) * sgm_pix_size`)
// along the reference-camera ray.
//
// Algorithm (per thread = per (vx, vy, vz) voxel):
//   1. Look up `sgm_depth, sgm_pix_size` at (vx, vy).
//   2. Skip if sgm_depth ≤ 0 (invalid / masked).
//   3. Initialize p = `get3DPointForPixelAndDepthFromRC(rc, pix, sgm_depth)`.
//   4. If vz ≠ middle, slide p along the ray by
//      `(vz - middle) * sgm_pix_size`.
//   5. Build the patch: patch.p = p, patch.d = pix size,
//      patch.{n,x,y} from the epipolar basis (or supplied
//      normal map — not yet wired through this port).
//   6. Call compNCCby3DptsYK<true> (sigmoid invert-and-filter).
//   7. If finite, additively accumulate into the half volume
//      via the promote-add-demote pattern (matching upstream's
//      bit-exact behavior even though Apple GPUs have native
//      FP16 ALUs).
//   8. If invalid (NCC returned INFINITY), leave the slot alone.
//
// Deferred (matches the scope decisions made for compute_similarity):
//   * `useCustomPatchPattern` branch — needs DevicePatchPattern.
//   * `in_sgmNormalMap_d` — would replace the bisector-based
//     patch.n. Adding it later requires a buffer binding + flag.

#include <metal_stdlib>
#include "Patch.h"
#include "volume_helpers.h"
using namespace metal;
using namespace av_depthmap;

struct RefineSimilarityParams {
    uint  volDimX;
    uint  volDimY;
    uint  volDimZ;
    uint  rcRefineLevelWidth;
    uint  rcRefineLevelHeight;
    uint  tcRefineLevelWidth;
    uint  tcRefineLevelHeight;
    float rcMipmapLevel;
    int   stepXY;
    int   wsh;
    float invGammaC;
    float invGammaP;
    uint  useConsistentScale;
    uint  depthRangeBegin;
    uint  depthRangeEnd;
    uint  roiXBegin;
    uint  roiYBegin;
    uint  roiWidth;
    uint  roiHeight;
};

kernel void av_volume_refine_similarity(
    device half*                     inout_vol_sim    [[buffer(0)]],
    device const float2*             in_sgm_dp_map    [[buffer(1)]],
    constant DeviceCameraParams&     rc               [[buffer(2)]],
    constant DeviceCameraParams&     tc               [[buffer(3)]],
    constant RefineSimilarityParams& p                [[buffer(4)]],
    texture2d<float, access::sample> rcMipmap         [[texture(0)]],
    texture2d<float, access::sample> tcMipmap         [[texture(1)]],
    uint3                            gid              [[thread_position_in_grid]])
{
    if (gid.x >= p.roiWidth || gid.y >= p.roiHeight) return;

    constexpr sampler smp(coord::normalized,
                          address::clamp_to_edge,
                          filter::linear,
                          mip_filter::linear);

    const uint vx = gid.x;
    const uint vy = gid.y;
    const uint vz = p.depthRangeBegin + gid.z;

    const float2 pix = float2(float(int(p.roiXBegin + vx) * p.stepXY),
                              float(int(p.roiYBegin + vy) * p.stepXY));

    // Per-pixel SGM-supplied (mid-depth, sgm pixSize).
    const float2 sgm = in_sgm_dp_map[vy * p.volDimX + vx];
    if (sgm.x <= 0.0f) return;   // invalid / masked

    // Initialize the 3D point at the SGM mid depth.
    float3 p3 = get3DPointForPixelAndDepthFromRC(rc, pix, sgm.x);

    // Slide along the ray by the per-voxel sub-pixel offset.
    const int center_z = (int(p.volDimZ) - 1) / 2;
    const int rel_offset = int(vz) - center_z;
    if (rel_offset != 0) {
        const float dist = float(rel_offset) * sgm.y;
        move3DPointByRcPixSize(p3, rc, dist);
    }

    // Build the patch. The epipolar basis logic is identical to
    // computeRotCSEpip; inlined here so we can choose the normal
    // source explicitly (today: bisector; future: optional normal
    // map). Once the normal map lands, computeRotCSEpip can be
    // reused unchanged.
    Patch patch;
    patch.p = p3;
    patch.d = computePixSize(rc, p3);
    {
        const float3 v1 = normalize(float3(rc.C) - patch.p);
        const float3 v2 = normalize(float3(tc.C) - patch.p);
        patch.y = normalize(cross(v1, v2));
        // Default normal source: bisector of v1, v2.
        patch.n = normalize((v1 + v2) * 0.5f);
        patch.x = normalize(cross(patch.y, patch.n));
    }

    // NCC with sigmoid invert-and-filter → similarity in (0, 1)
    // or INFINITY when invalid.
    const float fsim = compNCCby3DptsYK<true>(
        rc, tc, rcMipmap, tcMipmap, smp,
        p.rcRefineLevelWidth, p.rcRefineLevelHeight,
        p.tcRefineLevelWidth, p.tcRefineLevelHeight,
        p.rcMipmapLevel,
        p.wsh,
        p.invGammaC, p.invGammaP,
        p.useConsistentScale != 0u,
        patch);

    if (!isfinite(fsim)) return;   // upstream contract: leave the slot

    // Additive accumulation, promote-add-demote (matches
    // av_volume_add_half / upstream's bit-exact behavior).
    const uint k = vz * (p.volDimX * p.volDimY)
                 + vy *  p.volDimX
                 + vx;
    const float prev = float(inout_vol_sim[k]);
    inout_vol_sim[k] = half(prev + fsim);
}
