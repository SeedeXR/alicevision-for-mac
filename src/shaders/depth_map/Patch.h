// Patch.h — Metal port of depthMap/cuda/device/Patch.cuh
// (geometry helpers only; NCC kernels deferred until color.cuh
// and SimStat.cuh are ported).
//
// CUDA → MSL translation notes:
//
//  * The CUDA original used `double` inside `rotPointAroundVect`
//    and `angleBetwUnitV1andUnitV2`. Apple GPUs have no FP64
//    units; all math here is `float`. `angleBetwUnitV1andUnitV2`
//    is hardened with a clamp to keep `acos` in domain. The
//    Rodrigues rotation in `rotPointAroundVect` retains its sin/cos
//    formulation and stays in float — observed numerical drift is
//    bounded by FP32 ULP × the (small) number of operations.
//
//  * `DeviceCameraParams` is mirrored from
//    depthMap/cuda/device/DeviceCameraParams.hpp. The CUDA file
//    placed it in `__constant__` memory; in MSL the equivalent is
//    the `constant` address space, which is what camera-param
//    buffers will be bound through in real kernels. The helpers
//    below accept the struct by `const&` and don't care about its
//    address space.
//
//  * The texture-using `compNCCby3DptsYK*` template kernels from
//    the CUDA original are NOT in this file. They depend on
//    color.cuh (CostYKfromLab) and SimStat.cuh (simStat). When
//    those ports land they'll join this header.

#pragma once

#include <metal_stdlib>
#include "matrix.h"
#include "color.h"
#include "SimStat.h"
#include "DevicePatchPattern.h"
using namespace metal;

namespace av_depthmap {

// ----------------------------------------------------------------
// DeviceCameraParams — mirror of upstream's CUDA struct.
//
// Layouts of P, iP, R, iR, K, iK are column-major 3x3 / 3x4 flat
// (see matrix.h for the indexing convention).
// ----------------------------------------------------------------

// CUDA's `float3` is 12 bytes (3 packed floats). MSL's `float3`
// is 16-byte aligned. To keep the CPU-side mirror struct binary-
// compatible with the upstream layout (and to fit four camera
// vectors into 48 bytes total instead of 64), we use
// `packed_float3` here — which is exactly 12 bytes. Conversions
// to `float3` for arithmetic happen implicitly in MSL.
struct DeviceCameraParams
{
    float         P[12];
    float         iP[9];
    float         R[9];
    float         iR[9];
    float         K[9];
    float         iK[9];
    packed_float3 C;
    packed_float3 XVect;
    packed_float3 YVect;
    packed_float3 ZVect;
};

// ----------------------------------------------------------------
// Patch — oriented planar patch.
// ----------------------------------------------------------------

struct Patch
{
    float3 p;     // 3d point (patch center)
    float3 n;     // patch normal
    float3 x;     // patch x axis
    float3 y;     // patch y axis
    float  d;     // pixel size at the patch
};

// ----------------------------------------------------------------
// Rodrigues rotation of point X around unit axis `vect` by `angle`
// degrees. CUDA source used FP64; we stay in FP32 — sufficient for
// the patch-rotation use case (small angular increments).
// ----------------------------------------------------------------

inline void rotPointAroundVect(thread float3& out, float3 X, float3 vect, int angle)
{
    const float sizeX = length(X);
    float xn = X.x / sizeX;
    float yn = X.y / sizeX;
    float zn = X.z / sizeX;
    const float u = vect.x;
    const float v = vect.y;
    const float w = vect.z;

    const float ux = u * xn, uy = u * yn, uz = u * zn;
    const float vx = v * xn, vy = v * yn, vz = v * zn;
    const float wx = w * xn, wy = w * yn, wz = w * zn;

    const float angle_rad = float(angle) * (M_PI_F / 180.0f);
    const float sa = sin(angle_rad);
    const float ca = cos(angle_rad);

    const float nx = u * (ux + vy + wz) + (xn * (v * v + w * w) - u * (vy + wz)) * ca + (-wy + vz) * sa;
    const float ny = v * (ux + vy + wz) + (yn * (u * u + w * w) - v * (ux + wz)) * ca + ( wx - uz) * sa;
    const float nz = w * (ux + vy + wz) + (zn * (u * u + v * v) - w * (ux + vy)) * ca + (-vx + uy) * sa;

    const float ren = sqrt(nx * nx + ny * ny + nz * nz);
    out = float3(nx / ren, ny / ren, nz / ren) * sizeX;
}

inline void rotatePatch(thread Patch& ptch, int rx, int ry)
{
    float3 n, y_, x_;
    rotPointAroundVect(n, ptch.n, ptch.x, rx);
    rotPointAroundVect(y_, ptch.y, ptch.x, rx);
    ptch.n = n;
    ptch.y = y_;

    rotPointAroundVect(n, ptch.n, ptch.y, ry);
    rotPointAroundVect(x_, ptch.x, ptch.y, ry);
    ptch.n = n;
    ptch.x = x_;
}

inline void movePatch(thread Patch& ptch, int pt)
{
    const float3 v = ptch.n;
    const float  d = ptch.d * float(pt);
    ptch.p = ptch.p + v * d;
}

// Build an arbitrary right-handed orthonormal basis from a unit
// normal `n`. Output: xax, yax both orthogonal to n; xax × yax = n
// (up to sign; the basis is not canonical, only consistent).
inline void computeRotCS(thread float3& xax, thread float3& yax, float3 n)
{
    float3 x = float3(-n.y + n.z,
                      +n.x + n.z,
                      -n.x - n.y);
    if (fabs(x.x) < 1e-7f && fabs(x.y) < 1e-7f && fabs(x.z) < 1e-7f) {
        x = float3(-n.y - n.z,
                   +n.x - n.z,
                   +n.x + n.y);
    }
    xax = normalize(x);
    yax = cross(n, xax);
}

// Epipolar-plane basis: given the patch center and the two camera
// centers, build a frame where:
//   - `y` is perpendicular to the epipolar plane.
//   - `n` lies on the epipolar plane (bisector of v1 and v2).
//   - `x` lies on the epipolar plane, x = y × n.
//
// The CUDA original normalizes ptch.n redundantly after `(v1+v2)/2`
// since v1 and v2 are unit vectors; we keep the explicit normalize
// for parity with upstream behavior on edge cases (degenerate
// bisector when v1 ≈ -v2).
inline void computeRotCSEpip(thread Patch& ptch,
                             constant const DeviceCameraParams& rc,
                             constant const DeviceCameraParams& tc)
{
    const float3 v1 = normalize(rc.C - ptch.p);
    const float3 v2 = normalize(tc.C - ptch.p);

    ptch.y = normalize(cross(v1, v2));
    ptch.n = normalize((v1 + v2) * 0.5f);
    ptch.x = normalize(cross(ptch.y, ptch.n));
}

// Pixel-footprint size at a 3D point, derived from a 1-pixel shift
// in image space backprojected to the world.
inline float computePixSize(constant const DeviceCameraParams& cam, float3 p)
{
    const float2 rp  = project3DPoint(cam.P, p);
    const float2 rp1 = rp + float2(1.0f, 0.0f);

    float3 refvect = M3x3mulV2(cam.iP, rp1);
    refvect = normalize(refvect);
    return pointLineDistance3D(p, cam.C, refvect);
}

// Project a 3D point to image space, returning (-1, -1) if the
// point lies behind the camera (z <= 0).
inline void getPixelFor3DPoint(thread float2& out,
                               constant const DeviceCameraParams& cam,
                               float3 X)
{
    const float3 p = M3x4mulV3(cam.P, X);
    if (p.z <= 0.0f) {
        out = float2(-1.0f, -1.0f);
    } else {
        out = float2(p.x / p.z, p.y / p.z);
    }
}

// Back-project a pixel to its intersection with a fronto-parallel
// plane at depth `fpPlaneDepth` from the reference camera.
inline float3 get3DPointForPixelAndFrontoParellePlaneRC(
    constant const DeviceCameraParams& cam,
    float2 pix,
    float  fpPlaneDepth)
{
    const float3 planep = cam.C + cam.ZVect * fpPlaneDepth;
    float3 v = M3x3mulV2(cam.iP, pix);
    v = normalize(v);
    return linePlaneIntersect(cam.C, v, planep, cam.ZVect);
}

// Back-project a pixel to a 3D point at given depth along the ray.
inline float3 get3DPointForPixelAndDepthFromRC(
    constant const DeviceCameraParams& cam,
    float2 pix,
    float  depth)
{
    float3 rpv = M3x3mulV2(cam.iP, pix);
    rpv = normalize(rpv);
    return cam.C + rpv * depth;
}

// Two-view triangulation by midpoint of the perpendicular-feet of
// the back-projected rays.
inline float3 triangulateMatchRef(constant const DeviceCameraParams& rc,
                                  constant const DeviceCameraParams& tc,
                                  float2 refpix,
                                  float2 tarpix)
{
    float3 refvect = M3x3mulV2(rc.iP, refpix);
    refvect = normalize(refvect);
    const float3 refpoint = refvect + rc.C;

    float3 tarvect = M3x3mulV2(tc.iP, tarpix);
    tarvect = normalize(tarvect);
    const float3 tarpoint = tarvect + tc.C;

    float k, l;
    float3 lli1, lli2;
    lineLineIntersect(&k, &l, &lli1, &lli2,
                      rc.C, refpoint, tc.C, tarpoint);
    return rc.C + refvect * k;
}

// Quadratic sub-pixel depth refinement from 3 cost samples.
// Matches Qingxiong PAMI 2008 (Stereo Matching with Color-Weighted
// Correlation + Hierarchical BP + Occlusion Handling).
inline float refineDepthSubPixel(float3 depths, float3 sims)
{
    float simM1 = sims.x;
    float sim   = sims.y;
    float simP1 = sims.z;
    simM1 = (simM1 + 1.0f) * 0.5f;
    sim   = (sim   + 1.0f) * 0.5f;
    simP1 = (simP1 + 1.0f) * 0.5f;

    if (simM1 < sim || simP1 < sim) return depths.y;

    const float dispStep = -((simP1 - simM1) /
                             (2.0f * (simP1 + simM1 - 2.0f * sim)));

    const float floatDepthM1 = depths.x;
    const float floatDepthP1 = depths.z;
    const float b = (floatDepthP1 + floatDepthM1) * 0.5f;
    const float a = b - floatDepthM1;
    const float interpDepth = a * dispStep + b;

    if (!isfinite(interpDepth) || interpDepth <= 0.0f) return depths.y;
    return interpDepth;
}

// Cross-camera mipmap level selection to keep patch comparison at
// a consistent world-scale across views with different resolutions
// w.r.t. the patch surface.
inline void computeRcTcMipmapLevels(thread float& out_rcMipmapLevel,
                                    thread float& out_tcMipmapLevel,
                                    float mipmapLevel,
                                    constant const DeviceCameraParams& rc,
                                    constant const DeviceCameraParams& tc,
                                    float2 rp0, float2 tp0,
                                    float3 p0)
{
    const float rcDepth = length(rc.C - p0);
    const float tcDepth = length(tc.C - p0);

    const float2 rp1 = rp0 + float2(1.0f, 0.0f);
    const float2 tp1 = tp0 + float2(1.0f, 0.0f);

    float3 rpv = M3x3mulV2(rc.iP, rp1);
    rpv = normalize(rpv);
    const float3 prp1 = rc.C + rpv * rcDepth;

    float3 tpv = M3x3mulV2(tc.iP, tp1);
    tpv = normalize(tpv);
    const float3 ptp1 = tc.C + tpv * tcDepth;

    const float rcDist = distance(p0, prp1);
    const float tcDist = distance(p0, ptp1);
    const float distFactor = rcDist / tcDist;

    out_rcMipmapLevel = mipmapLevel;
    out_tcMipmapLevel = mipmapLevel;
    if (distFactor < 1.0f) {
        out_tcMipmapLevel = mipmapLevel - log2(1.0f / distFactor);
        if (out_tcMipmapLevel < 0.0f) {
            out_rcMipmapLevel = mipmapLevel + fabs(out_tcMipmapLevel);
            out_tcMipmapLevel = 0.0f;
        }
    } else {
        out_tcMipmapLevel = mipmapLevel + log2(distFactor);
    }
}

inline int angleBetwUnitV1andUnitV2(float3 V1, float3 V2)
{
    const float c = clamp(dot(V1, V2), -1.0f, 1.0f);
    return int(fabs(acos(c)) / (M_PI_F / 180.0f));
}

// Plane-induced homography between two views (Hartley & Zisserman
// 2nd edition, Eq. 13.2):
//     H = K_t * (R_rt - t_rt * n^T / d) * K_r^-1
// where (R_rt, t_rt) is the relative pose from ref to target,
// `n` is the plane normal in the ref camera frame, and `d` is the
// distance from the ref camera center to the plane.
//
// Implementation note: matrix.h's M3x3mulM3x3 / M3x3transpose
// helpers only accept `thread const float*` arguments. The camera
// matrices live in `constant` address space, so we copy them once
// into thread-local arrays at function entry. 4 × 9 × 4 B = 144 B
// of stack per invocation — negligible. The alternative
// (proliferating address-space overloads of M3x3mulM3x3 to the
// full constant×thread×device cross-product) bloats matrix.h with
// no observed performance benefit.
inline void computeHomography(thread float* out_H,
                              constant const DeviceCameraParams& rc,
                              constant const DeviceCameraParams& tc,
                              float3 in_p,
                              float3 in_n)
{
    thread float rcR[9], tcR[9], tcK[9], rciK[9];
    for (int i = 0; i < 9; ++i) {
        rcR [i] = rc.R [i];
        tcR [i] = tc.R [i];
        tcK [i] = tc.K [i];
        rciK[i] = rc.iK[i];
    }

    const float3 _tl = float3(0.0f) - M3x3mulV3(rcR, float3(rc.C));
    const float3 _tr = float3(0.0f) - M3x3mulV3(tcR, float3(tc.C));

    const float3 p = M3x3mulV3(rcR, in_p - float3(rc.C));
    float3 n = M3x3mulV3(rcR, in_n);
    n = normalize(n);
    const float d = -dot(n, p);

    thread float RrT[9];
    M3x3transpose(RrT, rcR);

    thread float tmpRr[9];
    M3x3mulM3x3(tmpRr, tcR, RrT);
    const float3 tr = _tr - M3x3mulV3(tmpRr, _tl);

    thread float tmp[9];
    thread float tmp1[9];
    outerMultiply(tmp, tr, n / d);
    M3x3minusM3x3(tmp, tmpRr, tmp);
    M3x3mulM3x3(tmp1, tcK, tmp);
    M3x3mulM3x3(tmp, tmp1, rciK);

    for (int i = 0; i < 9; ++i) out_H[i] = tmp[i];
}

// ----------------------------------------------------------------
// compNCCby3DptsYK — patch-based Normalized Cross-Correlation
// across two mipmapped views.
//
// This is the first real depthMap kernel: it exercises every helper
// in the device/ tree — `project3DPoint` (matrix.h), the texture
// sample path, `CostYKfromLab` (color.h), and `simStat`
// (SimStat.h). Per-thread; one similarity score per dispatched patch.
//
// Returns:
//   - similarity in (-1, 0)         best=−1, worst≈0
//   - INFINITY if the patch falls outside the image margins or
//     either center pixel's alpha is below the per-camera threshold
//   - if TInvertAndFilter, the similarity is passed through
//     `sigmoid(0, 1, 0.7, -0.7, sim)` so the output range becomes
//     (0, 1) with best≈1.
//
// Caller contract: the textures must be in `clamp_to_edge` /
// `filter::linear` / `mip_filter::linear` configuration; the
// validation kernel below sets this up via a `constexpr sampler`.
//
// Translation notes vs CUDA:
//   * `tex2DLod<float4>(tex, u, v, level)` →
//     `tex.sample(smp, float2(u, v), level(L))`
//     where `smp` is a `sampler` argument (we pass it in instead of
//     binding inside the function because samplers can't be local).
//   * `CUDART_INF_F` → MSL `INFINITY` (from `<metal_stdlib>`).
//   * The `cudaTextureObject_t` parameter type is replaced by
//     `texture2d<float, access::sample>`; the sampler is a separate
//     argument.
// ----------------------------------------------------------------

template <bool TInvertAndFilter>
inline float compNCCby3DptsYK(
    constant const DeviceCameraParams& rc,
    constant const DeviceCameraParams& tc,
    texture2d<float, access::sample>   rcMipmap,
    texture2d<float, access::sample>   tcMipmap,
    sampler                            smp,
    uint   rcLevelWidth,  uint   rcLevelHeight,
    uint   tcLevelWidth,  uint   tcLevelHeight,
    float  mipmapLevel,
    int    wsh,
    float  invGammaC,
    float  invGammaP,
    bool   useConsistentScale,
    Patch  patch)
{
    const float2 rp = project3DPoint(rc.P, patch.p);
    const float2 tp = project3DPoint(tc.P, patch.p);

    // Margin around the patch in image space.
    const float dd = float(wsh) + 2.0f;
    if (rp.x < dd || rp.x > float(rcLevelWidth  - 1u) - dd ||
        tp.x < dd || tp.x > float(tcLevelWidth  - 1u) - dd ||
        rp.y < dd || rp.y > float(rcLevelHeight - 1u) - dd ||
        tp.y < dd || tp.y > float(tcLevelHeight - 1u) - dd)
    {
        return INFINITY;
    }

    const float rcInvLevelWidth  = 1.0f / float(rcLevelWidth);
    const float rcInvLevelHeight = 1.0f / float(rcLevelHeight);
    const float tcInvLevelWidth  = 1.0f / float(tcLevelWidth);
    const float tcInvLevelHeight = 1.0f / float(tcLevelHeight);

    float rcMipmapLevel = mipmapLevel;
    float tcMipmapLevel = mipmapLevel;
    if (useConsistentScale) {
        computeRcTcMipmapLevels(rcMipmapLevel, tcMipmapLevel,
                                mipmapLevel, rc, tc,
                                rp, tp, patch.p);
    }

    // Center colors for the bilateral support-weight reference.
    const float4 rcCenterColor = rcMipmap.sample(
        smp,
        float2((rp.x + 0.5f) * rcInvLevelWidth,
               (rp.y + 0.5f) * rcInvLevelHeight),
        level(rcMipmapLevel));
    const float4 tcCenterColor = tcMipmap.sample(
        smp,
        float2((tp.x + 0.5f) * tcInvLevelWidth,
               (tp.y + 0.5f) * tcInvLevelHeight),
        level(tcMipmapLevel));

    if (rcCenterColor.w < kRcMinAlpha ||
        tcCenterColor.w < kTcMinAlpha)
    {
        return INFINITY;
    }

    simStat sst;
    sst.init_zero();

    for (int yp = -wsh; yp <= wsh; ++yp) {
        for (int xp = -wsh; xp <= wsh; ++xp) {
            const float3 p = patch.p
                           + patch.x * (patch.d * float(xp))
                           + patch.y * (patch.d * float(yp));

            const float2 rpc = project3DPoint(rc.P, p);
            const float2 tpc = project3DPoint(tc.P, p);

            const float4 rcPatchColor = rcMipmap.sample(
                smp,
                float2((rpc.x + 0.5f) * rcInvLevelWidth,
                       (rpc.y + 0.5f) * rcInvLevelHeight),
                level(rcMipmapLevel));
            const float4 tcPatchColor = tcMipmap.sample(
                smp,
                float2((tpc.x + 0.5f) * tcInvLevelWidth,
                       (tpc.y + 0.5f) * tcInvLevelHeight),
                level(tcMipmapLevel));

            const float w = CostYKfromLab(xp, yp,
                                          rcCenterColor, rcPatchColor,
                                          invGammaC, invGammaP)
                          * CostYKfromLab(xp, yp,
                                          tcCenterColor, tcPatchColor,
                                          invGammaC, invGammaP);

            sst.update(rcPatchColor.x, tcPatchColor.x, w);
        }
    }

    const float sim = sst.computeWSim();
    if (TInvertAndFilter) {
        // Upstream comment: "best similarity value was -1, worst was 0.
        //                    best similarity value is 1, worst is still 0."
        // sigmoid params follow https://www.desmos.com/calculator/skmhf1gpyf
        return sigmoid(0.0f, 1.0f, 0.7f, -0.7f, sim);
    }
    return sim;
}

// ----------------------------------------------------------------
// compNCCby3DptsYK_customPatchPattern — same as compNCCby3DptsYK
// but the sample positions, mipmap-level offset, and weighting are
// driven by a `DevicePatchPattern` constant struct (1..N subparts).
//
// One similarity score per dispatched patch, computed as the
// `weight`-weighted average of per-subpart NCC scores. A subpart
// is either:
//   - a circle (use `coordinates[]` as relative offsets in the
//     patch's tangent plane), or
//   - a full square block of side `2*wsh+1` at the subpart's
//     `downscale` step in the tangent plane.
//
// Upstream CUDA file: depthMap/cuda/device/Patch.cuh, lines 599+.
//
// Translation notes vs CUDA:
//   * `constantPatchPattern_d` (CUDA constant-memory symbol) → a
//     plain `constant DevicePatchPattern&` argument; the host binds
//     the struct via `set_bytes` (it is < 4 KB).
//   * `CostYKfromLab(c1, c2, invGammaC)` (3-arg, color-only) is the
//     weight for circle subparts; `CostYKfromLab(xp,yp,c1,c2,gC,gP)`
//     (6-arg) is the weight for full subparts. Both are already in
//     color.h.
//   * Upstream uses `subpart.isCircle` (bool); we use `int != 0`
//     selector to match the layout-stable mirror in
//     DevicePatchPattern.h. Semantics are identical.
//   * The "uninitialized margin" upstream uses is `dd = 2.f` (a
//     fixed margin, NOT `wsh+2` like the simple kernel) — see the
//     comment in upstream around line 618.
//   * Final wsum==0 fold returns INFINITY, exactly like upstream.
// ----------------------------------------------------------------

template <bool TInvertAndFilter>
inline float compNCCby3DptsYK_customPatchPattern(
    constant const DeviceCameraParams& rc,
    constant const DeviceCameraParams& tc,
    texture2d<float, access::sample>   rcMipmap,
    texture2d<float, access::sample>   tcMipmap,
    sampler                            smp,
    uint   rcLevelWidth,  uint   rcLevelHeight,
    uint   tcLevelWidth,  uint   tcLevelHeight,
    float  mipmapLevel,
    float  invGammaC,
    float  invGammaP,
    bool   useConsistentScale,
    constant const DevicePatchPattern& pattern,
    Patch  patch)
{
    const float2 rp = project3DPoint(rc.P, patch.p);
    const float2 tp = project3DPoint(tc.P, patch.p);

    // Image 2D coordinates margin (upstream uses a fixed 2.f here,
    // not wsh+2 — wsh is per-subpart in this kernel).
    const float dd = 2.0f;

    if (rp.x < dd || rp.x > float(rcLevelWidth  - 1u) - dd ||
        tp.x < dd || tp.x > float(tcLevelWidth  - 1u) - dd ||
        rp.y < dd || rp.y > float(rcLevelHeight - 1u) - dd ||
        tp.y < dd || tp.y > float(tcLevelHeight - 1u) - dd)
    {
        return INFINITY;
    }

    const float rcInvLevelWidth  = 1.0f / float(rcLevelWidth);
    const float rcInvLevelHeight = 1.0f / float(rcLevelHeight);
    const float tcInvLevelWidth  = 1.0f / float(tcLevelWidth);
    const float tcInvLevelHeight = 1.0f / float(tcLevelHeight);

    // Center alpha gate (upstream samples only .w at the base level
    // for the gate, before subpart-level processing).
    const float rcAlpha = rcMipmap.sample(
        smp,
        float2((rp.x + 0.5f) * rcInvLevelWidth,
               (rp.y + 0.5f) * rcInvLevelHeight),
        level(mipmapLevel)).w;
    const float tcAlpha = tcMipmap.sample(
        smp,
        float2((tp.x + 0.5f) * tcInvLevelWidth,
               (tp.y + 0.5f) * tcInvLevelHeight),
        level(mipmapLevel)).w;

    if (rcAlpha < kRcMinAlpha || tcAlpha < kTcMinAlpha) {
        return INFINITY;
    }

    float rcMipmapLevel = mipmapLevel;
    float tcMipmapLevel = mipmapLevel;
    if (useConsistentScale) {
        computeRcTcMipmapLevels(rcMipmapLevel, tcMipmapLevel,
                                mipmapLevel, rc, tc,
                                rp, tp, patch.p);
    }

    float fsim = 0.0f;
    float wsum = 0.0f;

    for (int s = 0; s < pattern.nbSubparts; ++s) {
        const constant DevicePatchPatternSubpart& subpart =
            pattern.subparts[s];

        simStat sst;
        sst.init_zero();

        // Center colors at this subpart's mipmap level.
        const float4 rcCenterColor = rcMipmap.sample(
            smp,
            float2((rp.x + 0.5f) * rcInvLevelWidth,
                   (rp.y + 0.5f) * rcInvLevelHeight),
            level(rcMipmapLevel + subpart.level));
        const float4 tcCenterColor = tcMipmap.sample(
            smp,
            float2((tp.x + 0.5f) * tcInvLevelWidth,
                   (tp.y + 0.5f) * tcInvLevelHeight),
            level(tcMipmapLevel + subpart.level));

        if (subpart.isCircle != 0) {
            for (int c = 0; c < subpart.nbCoordinates; ++c) {
                const float2 relativeCoord = subpart.coordinates[c];

                const float3 p = patch.p
                               + patch.x * (patch.d * relativeCoord.x)
                               + patch.y * (patch.d * relativeCoord.y);

                const float2 rpc = project3DPoint(rc.P, p);
                const float2 tpc = project3DPoint(tc.P, p);

                const float4 rcPatchColor = rcMipmap.sample(
                    smp,
                    float2((rpc.x + 0.5f) * rcInvLevelWidth,
                           (rpc.y + 0.5f) * rcInvLevelHeight),
                    level(rcMipmapLevel + subpart.level));
                const float4 tcPatchColor = tcMipmap.sample(
                    smp,
                    float2((tpc.x + 0.5f) * tcInvLevelWidth,
                           (tpc.y + 0.5f) * tcInvLevelHeight),
                    level(tcMipmapLevel + subpart.level));

                // Color-only Yoon&Kweon weight (no dx/dy term for
                // circle samples — upstream's 3-arg variant).
                const float w =
                    CostYKfromLab(rcCenterColor, rcPatchColor, invGammaC)
                  * CostYKfromLab(tcCenterColor, tcPatchColor, invGammaC);

                sst.update(rcPatchColor.x, tcPatchColor.x, w);
            }
        } else {
            // Full patch block at subpart.downscale step.
            for (int yp = -subpart.wsh; yp <= subpart.wsh; ++yp) {
                for (int xp = -subpart.wsh; xp <= subpart.wsh; ++xp) {
                    const float3 p = patch.p
                        + patch.x * (patch.d * float(xp) * subpart.downscale)
                        + patch.y * (patch.d * float(yp) * subpart.downscale);

                    const float2 rpc = project3DPoint(rc.P, p);
                    const float2 tpc = project3DPoint(tc.P, p);

                    const float4 rcPatchColor = rcMipmap.sample(
                        smp,
                        float2((rpc.x + 0.5f) * rcInvLevelWidth,
                               (rpc.y + 0.5f) * rcInvLevelHeight),
                        level(rcMipmapLevel + subpart.level));
                    const float4 tcPatchColor = tcMipmap.sample(
                        smp,
                        float2((tpc.x + 0.5f) * tcInvLevelWidth,
                               (tpc.y + 0.5f) * tcInvLevelHeight),
                        level(tcMipmapLevel + subpart.level));

                    // Full 6-arg Yoon&Kweon weight (color + spatial
                    // proximity to subpart center).
                    const float w =
                        CostYKfromLab(xp, yp,
                                      rcCenterColor, rcPatchColor,
                                      invGammaC, invGammaP)
                      * CostYKfromLab(xp, yp,
                                      tcCenterColor, tcPatchColor,
                                      invGammaC, invGammaP);

                    sst.update(rcPatchColor.x, tcPatchColor.x, w);
                }
            }
        }

        const float fsimSubpart = sst.computeWSim();

        // Only accumulate finite valid-similarity subparts (raw
        // value < 0). `computeWSim` returns 1.0f for the degenerate
        // case (non-finite variance product), which is filtered
        // here exactly like upstream's `< 0.f` test.
        if (fsimSubpart < 0.0f) {
            if (TInvertAndFilter) {
                // Same sigmoid as in compNCCby3DptsYK<true>.
                const float fsimInverted =
                    sigmoid(0.0f, 1.0f, 0.7f, -0.7f, fsimSubpart);
                fsim += fsimInverted * subpart.weight;
            } else {
                fsim += fsimSubpart * subpart.weight;
            }
            wsum += subpart.weight;
        }
    }

    if (wsum == 0.0f) {
        return INFINITY;
    }
    if (TInvertAndFilter) {
        // Upstream: "for now, we do not average" in the inverted
        // branch.
        return fsim;
    }
    return fsim / wsum;
}

}  // namespace av_depthmap
