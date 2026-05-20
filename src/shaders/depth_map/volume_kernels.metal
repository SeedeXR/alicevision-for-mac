// volume_kernels.metal — Phase 7 opener: the three "easy" volume
// kernels from upstream's deviceSimilarityVolumeKernels.cuh.
//
//   volume_init        : set all voxels in a 3D volume to a value
//   volume_add (half)  : in-place additive accumulation, FP16-safe
//   volume_update_uninitialized : where 2nd-best == 255 (sentinel),
//                       copy the 1st-best value into the slot
//
// Type system (mirrored from upstream's similarity.hpp):
//   * TSim       = uchar  (8-bit similarity, default)
//   * TSimRefine = half   (FP16 refinement volume)
//
// Layout convention for the 3D volume:
//   * Packed (no row padding). Linear index =
//       z * (volDimX * volDimY) + y * volDimX + x
//   * Upstream uses pitched memory with separate row/depth strides;
//     we keep packed for now (simplicity + correctness first;
//     re-evaluate via profiling if alignment hurts).
//
// FP16 add note: the upstream `__half + __half` path is documented
// as giving bad results on some GPUs ("not native FP16 ALU on some
// chips"); upstream therefore promotes to float, adds, demotes.
// On Apple GPUs FP16 ALUs are first-class, but we preserve the
// promote-add-demote pattern so the *numerical* result matches the
// upstream reference bit-for-bit.

#include <metal_stdlib>
#include "volume_helpers.h"
using namespace metal;
using namespace av_depthmap;

// Layout for both init/add/update. `volDim*` are the volume
// dimensions in voxels; the buffer must be at least
// `volDimX * volDimY * volDimZ * sizeof(T)` bytes.
struct VolumeDims {
    uint volDimX;
    uint volDimY;
    uint volDimZ;
};

// Init params carry a typed payload — we have two kernel variants,
// one per element type, with their own params struct.
struct VolumeInitUcharParams {
    VolumeDims dims;
    uint       value;   // packed into uint to keep 4-byte alignment
};

struct VolumeInitHalfParams {
    VolumeDims dims;
    // The host packs the desired `half` value into the low 16 bits
    // of a uint (via float-to-half bit conversion); the kernel
    // re-interprets it. This avoids needing a `half` field with
    // tricky alignment in the constant-memory blob.
    uint       value_half_bits;
};

inline uint av_volume_linear_index(uint3 gid, VolumeDims d)
{
    return gid.z * (d.volDimX * d.volDimY) + gid.y * d.volDimX + gid.x;
}

inline bool av_volume_in_bounds(uint3 gid, VolumeDims d)
{
    return gid.x < d.volDimX && gid.y < d.volDimY && gid.z < d.volDimZ;
}

// ----------------------------------------------------------------
// volume_init — uchar variant
// ----------------------------------------------------------------
kernel void av_volume_init_uchar(
    device   uchar*                       volume [[buffer(0)]],
    constant VolumeInitUcharParams&       p      [[buffer(1)]],
    uint3                                 gid    [[thread_position_in_grid]])
{
    if (!av_volume_in_bounds(gid, p.dims)) return;
    volume[av_volume_linear_index(gid, p.dims)] = uchar(p.value & 0xffu);
}

// ----------------------------------------------------------------
// volume_init — half variant
// ----------------------------------------------------------------
kernel void av_volume_init_half(
    device   half*                        volume [[buffer(0)]],
    constant VolumeInitHalfParams&        p      [[buffer(1)]],
    uint3                                 gid    [[thread_position_in_grid]])
{
    if (!av_volume_in_bounds(gid, p.dims)) return;
    const ushort bits = ushort(p.value_half_bits & 0xffffu);
    volume[av_volume_linear_index(gid, p.dims)] = as_type<half>(bits);
}

// ----------------------------------------------------------------
// volume_add — FP16 in-place additive accumulation.
// Promote-add-demote to match upstream's bit-equivalent behavior.
// ----------------------------------------------------------------
kernel void av_volume_add_half(
    device   half*           inout  [[buffer(0)]],
    device const half*       in     [[buffer(1)]],
    constant VolumeDims&     dims   [[buffer(2)]],
    uint3                    gid    [[thread_position_in_grid]])
{
    if (!av_volume_in_bounds(gid, dims)) return;
    const uint k = av_volume_linear_index(gid, dims);
    const float a = float(inout[k]);
    const float b = float(in[k]);
    inout[k] = half(a + b);
}

// ----------------------------------------------------------------
// volume_update_uninitialized — for the 8-bit similarity volume,
// where 255 is the "uninitialized" sentinel. Copy the 1st-best
// value into 2nd-best slots that are still at 255.
// ----------------------------------------------------------------
kernel void av_volume_update_uninitialized_uchar(
    device   uchar*          inout2nd [[buffer(0)]],
    device const uchar*      in1st    [[buffer(1)]],
    constant VolumeDims&     dims     [[buffer(2)]],
    uint3                    gid      [[thread_position_in_grid]])
{
    if (!av_volume_in_bounds(gid, dims)) return;
    const uint k = av_volume_linear_index(gid, dims);
    if (inout2nd[k] >= uchar(255)) {
        inout2nd[k] = in1st[k];
    }
}

// ----------------------------------------------------------------
// volume_retrieve_best_depth — SGM exit point.
//
// Per output pixel (vx, vy): scan Z planes in [depthRange.begin,
// depthRange.end), find min-similarity index. If no valid found
// or `bestSim > maxSimilarity` → write `(-1, -1)` to depth/thickness
// and `(-1, +1)` to depth/sim. Otherwise:
//   * Convert the 3 neighbor depth-planes (best ± 1, clamped) to
//     actual 3D distances via `depthPlaneToDepth`.
//   * Compute thickness = `max(bestDepth_p1 - best, best -
//     bestDepth_m1) * thicknessMultFactor`.
//   * Convert bestSim from (0, 255) to (-1, +1):
//     `out_bestSim = (bestSim / 255) * 2 - 1`.
//
// The 8-bit similarity sentinel is 255 (uninitialized). Best
// possible similarity is 0, worst valid is 254.
//
// Two output buffers (depth/thickness + depth/sim) are always
// written. The CUDA original made depth/sim nullable; MSL can't
// represent nullable device pointers in compute kernels, so we
// always populate both. The extra ~8 bytes/pixel is negligible.
//
// Volume layout convention (must match Volume.cpp): packed,
// linear index = z * (X * Y) + y * X + x.
// Output buffers are flat float2 in (vy * roiW + vx) order.
// `in_depths` is a flat float buffer of length `volDimZ` (one
// entry per depth-plane along the sweep).
// ----------------------------------------------------------------

struct RetrieveBestDepthParams {
    uint  volDimX;
    uint  volDimY;
    uint  volDimZ;        // for clamping bestZIdx ± 1
    uint  depthRangeBegin;
    uint  depthRangeEnd;
    uint  roiXBegin;      // for the pix coords passed to depthPlaneToDepth
    uint  roiYBegin;
    int   scaleStep;
    float thicknessMultFactor;
    float maxSimilarity;  // in 0..255 range (the uchar volume scale)
};

kernel void av_volume_retrieve_best_depth(
    device float2*                         out_depth_thickness [[buffer(0)]],
    device float2*                         out_depth_sim       [[buffer(1)]],
    device const float*                    in_depths           [[buffer(2)]],
    device const uchar*                    in_vol_sim          [[buffer(3)]],
    constant DeviceCameraParams&           rc                  [[buffer(4)]],
    constant RetrieveBestDepthParams&      p                   [[buffer(5)]],
    uint2                                  gid                 [[thread_position_in_grid]])
{
    if (gid.x >= p.volDimX || gid.y >= p.volDimY) return;

    const uint vx = gid.x;
    const uint vy = gid.y;

    // Corresponding image-space pixel for camera back-projection.
    const float2 pix = float2(float(int(p.roiXBegin + vx) * p.scaleStep),
                              float(int(p.roiYBegin + vy) * p.scaleStep));

    // ---------- WTA over the depth range ----------
    float bestSim = 255.0f;
    int   bestZIdx = -1;

    for (uint vz = p.depthRangeBegin; vz < p.depthRangeEnd; ++vz) {
        const uint  k = vz * (p.volDimX * p.volDimY) + vy * p.volDimX + vx;
        const float simAtZ = float(in_vol_sim[k]);
        if (simAtZ < bestSim) {
            bestSim  = simAtZ;
            bestZIdx = int(vz);
        }
    }

    const uint out_k = vy * p.volDimX + vx;

    // ---------- Invalid / above-threshold path ----------
    if (bestZIdx < 0 || bestSim > p.maxSimilarity) {
        out_depth_thickness[out_k] = float2(-1.0f, -1.0f);
        out_depth_sim       [out_k] = float2(-1.0f,  1.0f);
        return;
    }

    // ---------- Neighbor depth planes (clamped) ----------
    const int bestZ_m1 = max(0,                bestZIdx - 1);
    const int bestZ_p1 = min(int(p.volDimZ) - 1, bestZIdx + 1);

    const float dp_m1 = in_depths[bestZ_m1];
    const float dp    = in_depths[bestZIdx];
    const float dp_p1 = in_depths[bestZ_p1];

    const float bestDepth    = depthPlaneToDepth(rc, dp,    pix);
    const float bestDepth_m1 = depthPlaneToDepth(rc, dp_m1, pix);
    const float bestDepth_p1 = depthPlaneToDepth(rc, dp_p1, pix);

    // No sub-pixel interpolation (matches upstream's default —
    // ALICEVISION_DEPTHMAP_RETRIEVE_BEST_Z_INTERPOLATION is OFF by
    // default and we don't enable it here).
    const float out_bestDepth = bestDepth;
    const float out_bestSim   = (bestSim / 255.0f) * 2.0f - 1.0f;

    // Thickness: max gap to the two neighbor depths, inflated.
    const float out_thickness =
        max(bestDepth_p1 - out_bestDepth,
            out_bestDepth - bestDepth_m1) * p.thicknessMultFactor;

    out_depth_thickness[out_k] = float2(out_bestDepth, out_thickness);
    out_depth_sim       [out_k] = float2(out_bestDepth, out_bestSim);
}
