// volume_helpers.h — small inline helpers used by the planeSweeping
// volume kernels. Mirrors the top of upstream's
// deviceSimilarityVolumeKernels.cuh (the bits we actually need so
// far). Add to this header as more kernels land.
//
// Currently provides:
//   * depthPlaneToDepth(rc, fp_plane_depth, pix)
//       Given a fronto-parallel plane at `fp_plane_depth` in the
//       reference camera's Z direction, back-project pixel `pix`
//       onto that plane and return its distance from the camera
//       center. Used by retrieveBestDepth to convert from
//       "depth-plane index along the sweep" → actual depth.

#pragma once

#include <metal_stdlib>
#include "matrix.h"
#include "Patch.h"   // for DeviceCameraParams
using namespace metal;

namespace av_depthmap {

inline float depthPlaneToDepth(constant const DeviceCameraParams& rc,
                               float                              fp_plane_depth,
                               float2                             pix)
{
    const float3 planep = float3(rc.C) + float3(rc.ZVect) * fp_plane_depth;
    float3 v = M3x3mulV2(rc.iP, pix);
    v = normalize(v);
    const float3 p = linePlaneIntersect(float3(rc.C), v,
                                        planep, float3(rc.ZVect));
    return length(float3(rc.C) - p);
}

// Build a Patch on a fronto-parallel reference-camera plane.
// Mirrors upstream's volume_computePatch (deviceSimilarityVolume
// Kernels.cuh:26). Uses 3 helpers from Patch.h that already accept
// `constant`-address-space camera params.
inline void volume_computePatch(thread Patch&                      patch,
                                constant const DeviceCameraParams& rc,
                                constant const DeviceCameraParams& tc,
                                float                              fp_plane_depth,
                                float2                             pix)
{
    patch.p = get3DPointForPixelAndFrontoParellePlaneRC(rc, pix, fp_plane_depth);
    patch.d = computePixSize(rc, patch.p);
    computeRotCSEpip(patch, rc, tc);
}

// Slide a 3D point along the ray from `rc.C` through `p` by
// `rcPixSize` units (signed). Mirrors upstream's
// move3DPointByRcPixSize (deviceSimilarityVolumeKernels.cuh:17).
// Used by volume_refineSimilarity to step around the SGM mid depth.
inline void move3DPointByRcPixSize(thread float3&                     p,
                                   constant const DeviceCameraParams& rc,
                                   float                              rcPixSize)
{
    float3 rpv = p - float3(rc.C);
    rpv = normalize(rpv);
    p = p + rpv * rcPixSize;
}

}  // namespace av_depthmap
