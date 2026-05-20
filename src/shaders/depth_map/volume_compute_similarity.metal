// volume_compute_similarity.metal — port of
// volume_computeSimilarity_kernel from upstream's
// deviceSimilarityVolumeKernels.cuh.
//
// Per-voxel NCC over a 3D cost volume. One thread per (roiX, roiY,
// roiZ). For each thread:
//   1. Translate volume-relative (roiX, roiY) to image-space pix.
//   2. Look up the fronto-parallel plane depth at the corresponding
//      vz = depthRange.begin + roiZ.
//   3. Build a Patch via volume_computePatch (uses Patch.h helpers).
//   4. Call compNCCby3DptsYK<false> with the R and T mipmaps.
//   5. Remap sim ∈ (-1, +1) to uchar in (0, 254). INFINITY → 255
//      (sentinel for invalid).
//   6. Update best/2nd-best volume entries with the new score.
//
// Called once per T camera (the outer host loop iterates over the
// T cameras the R camera should be compared against). On the first
// T call after init, both volumes start at 255 (uninitialized);
// subsequent T calls progressively narrow toward the true winners.
//
// Notes vs upstream:
//   * The `useCustomPatchPattern` branch is omitted — the custom-
//     pattern path requires DevicePatchPattern which we haven't
//     ported yet. We only handle the regular 9×9 (or wsh×wsh)
//     box NCC.
//   * The TSIM_USE_FLOAT path is also omitted; we run uchar TSim
//     exclusively (matching upstream's default).

#include <metal_stdlib>
#include "Patch.h"
#include "volume_helpers.h"
using namespace metal;
using namespace av_depthmap;

struct ComputeSimilarityParams {
    // Volume dims (must match the buffer sizes of out_volume_1st/2nd).
    uint  volDimX;
    uint  volDimY;
    uint  volDimZ;
    // Image-space dimensions at the current SGM mipmap level.
    uint  rcSgmLevelWidth;
    uint  rcSgmLevelHeight;
    uint  tcSgmLevelWidth;
    uint  tcSgmLevelHeight;
    // Mipmap level at which compNCCby3DptsYK samples the textures.
    float rcMipmapLevel;
    // Image-coordinate scaling per ROI step (typically the downscale
    // factor used for the SGM pass).
    int   stepXY;
    // NCC patch half-width (the upstream "wsh").
    int   wsh;
    float invGammaC;
    float invGammaP;
    // Booleans packed as uints; non-zero = true.
    uint  useConsistentScale;
    // Range / ROI flattened.
    uint  depthRangeBegin;
    uint  depthRangeEnd;     // exclusive (informational; we infer from roiZ)
    uint  roiXBegin;
    uint  roiYBegin;
    uint  roiWidth;          // == volDimX in the common case
    uint  roiHeight;         // == volDimY in the common case
};

kernel void av_volume_compute_similarity(
    device uchar*                    out_volume_1st [[buffer(0)]],
    device uchar*                    out_volume_2nd [[buffer(1)]],
    device const float*              in_depths      [[buffer(2)]],
    constant DeviceCameraParams&     rc             [[buffer(3)]],
    constant DeviceCameraParams&     tc             [[buffer(4)]],
    constant ComputeSimilarityParams& p             [[buffer(5)]],
    texture2d<float, access::sample> rcMipmap       [[texture(0)]],
    texture2d<float, access::sample> tcMipmap       [[texture(1)]],
    uint3                            gid            [[thread_position_in_grid]])
{
    if (gid.x >= p.roiWidth || gid.y >= p.roiHeight) return;

    constexpr sampler smp(coord::normalized,
                          address::clamp_to_edge,
                          filter::linear,
                          mip_filter::linear);

    const uint vx = gid.x;
    const uint vy = gid.y;
    const uint vz = p.depthRangeBegin + gid.z;

    // Image-space pixel for this voxel column.
    const float2 pix = float2(float(int(p.roiXBegin + vx) * p.stepXY),
                              float(int(p.roiYBegin + vy) * p.stepXY));

    // Depth-plane lookup. `in_depths` is a flat float array of
    // length >= depthRangeEnd; we index by `vz`.
    const float depthPlane = in_depths[vz];

    Patch patch;
    volume_computePatch(patch, rc, tc, depthPlane, pix);

    // Compute NCC. INFINITY return = invalid (out of image / masked).
    const float fsim = compNCCby3DptsYK<false>(
        rc, tc, rcMipmap, tcMipmap, smp,
        p.rcSgmLevelWidth, p.rcSgmLevelHeight,
        p.tcSgmLevelWidth, p.tcSgmLevelHeight,
        p.rcMipmapLevel,
        p.wsh,
        p.invGammaC, p.invGammaP,
        p.useConsistentScale != 0u,
        patch);

    // Remap to uchar. Sentinel 255 for invalid.
    float fsim_remapped;
    if (!isfinite(fsim)) {
        fsim_remapped = 255.0f;
    } else {
        // (-1, +1) → (0, 1) → clamp → ×254.
        constexpr float fminVal     = -1.0f;
        constexpr float fmaxVal     =  1.0f;
        constexpr float fmultiplier = 1.0f / (fmaxVal - fminVal);
        float f = (fsim - fminVal) * fmultiplier;
        f = clamp(f, 0.0f, 1.0f);
        fsim_remapped = f * 254.0f;
    }

    // Min / second-min update against the existing volume entries.
    // Each thread is the sole writer for its (vx, vy, vz) slot, so
    // no atomics needed. Across multiple kernel invocations (one
    // per T camera), this accumulates the best/2nd-best NCC.
    const uint k = vz * (p.volDimX * p.volDimY)
                 + vy *  p.volDimX
                 + vx;

    const uchar fsim_uc = uchar(fsim_remapped);
    const uchar cur_1st = out_volume_1st[k];
    const uchar cur_2nd = out_volume_2nd[k];

    if (fsim_uc < cur_1st) {
        out_volume_2nd[k] = cur_1st;
        out_volume_1st[k] = fsim_uc;
    } else if (fsim_uc < cur_2nd) {
        out_volume_2nd[k] = fsim_uc;
    }
}
