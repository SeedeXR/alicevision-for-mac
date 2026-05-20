// volume_optimize.metal — port of the 4 sub-kernels behind
// upstream's `cuda_volumeOptimize`:
//
//   * av_volume_init_y_slice              — set a 1D slice (X,Z plane at given Y)
//                                            to a constant value
//   * av_volume_get_xz_slice               — extract a 2D (X,Z) slice from the
//                                            3D uchar volume at a given Y,
//                                            promoting to uint (TSimAcc)
//   * av_volume_compute_best_z_in_slice    — per X column of the (X,Z) slice,
//                                            find the min over Z
//   * av_volume_aggregate_cost_at_x        — SGM DP step at one Y position:
//                                            combines (Ym1 slice, bestSimInYm1,
//                                            current XZ slice from input volume,
//                                            output aggregate volume) →
//                                            updates current slice + aggregates
//                                            into out_volAgr
//
// All four kernels honor the `axisT` swizzle (upstream's `int3 axisT`)
// that re-maps the (x, y, z) iteration coords to volume (vx, vy, vz)
// coords. Upstream uses an `int3` and pointer-indexes into it; MSL
// can't take `&v.x` so we use a `int v[3]` array.
//
// P2 mode (matches upstream's `_P2` signed convention):
//   * Fixed P2: `_P2 < 0` upstream → we take `P2 = abs(_P2)`.
//     This is the path we shipped in Session 13.
//   * Adaptive P2: `_P2 >= 0` upstream → P2 is derived per-thread
//     from a sigmoid over `deltaC = euclideanDist3(gcr0, gcr1)`,
//     where `gcr0` and `gcr1` are two adjacent rc-image samples
//     (current pixel + neighbor in the Ym1 direction at the SGM
//     mipmap level). Added in this session; gated by the
//     `adaptive_p2` flag in AggregateParams so the existing fixed
//     path is unchanged at the byte level.
//
// In the adaptive path we pass `_P2` (the signed value, with its
// sign preserved) into `sigmoid` as `sigMid`, matching upstream
// verbatim. Upstream computes:
//   P2 = sigmoid(80.f, 255.f, 80.f, _P2, deltaC)
//
// We expose `_P2` as an explicit `float p2_sig_mid` parameter to
// avoid baking sign games into the host-side OptimizeParams API.

#include <metal_stdlib>
#include "color.h"
#include "matrix.h"

using namespace metal;

// `axisT` swizzle as a packed int3 alias. The CPU mirror struct
// (ints, no padding) maps to this layout. Index meaning:
//   axisT[0] = volume-X position fed by iteration x
//   axisT[1] = volume-X position fed by iteration y (the slice index)
//   axisT[2] = volume-X position fed by iteration z
struct OptimizeAxis {
    int axis0;   // upstream axisT.x
    int axis1;   // upstream axisT.y
    int axis2;   // upstream axisT.z
};

// ----------------------------------------------------------------
// S48: function-constant specialization for the 4 SGM-DP sub-kernels.
//
// The axis-swizzle is the same for all dispatches within a single SGM
// path (2 of the 4 paths use {0,1,2}, the other 2 use {1,0,2}). Making
// axis0/axis1/axis2 function constants lets us:
//   * Eliminate the runtime `int v[3]; v[axis0] = x; ...` reindirection.
//   * Compile-time-resolve all `vol[axis0]`-style permuted-dim lookups.
//   * Compile-time-resolve the adaptive-P2 `if (axis1 == 0)` branch in
//     `aggregate_cost`.
// Host creates 2 PSO sets at init (one per axis configuration) and
// dispatches the right set per path.
//
// Function-constant slots (must be unique across the metallib):
//   0: kAxis0      — int, no default (must be supplied)
//   1: kAxis1      — int, no default
//   2: kAxis2      — int, no default
// ----------------------------------------------------------------
constant int kAxis0 [[function_constant(0)]];
constant int kAxis1 [[function_constant(1)]];
constant int kAxis2 [[function_constant(2)]];

inline uint av_optimize_linear_index(int vx, int vy, int vz,
                                     uint volX, uint volY)
{
    return uint(vz) * (volX * volY) + uint(vy) * volX + uint(vx);
}

// Given iteration coords (x, y, z) and an axisT swizzle, expand to
// the actual volume coordinates (vx, vy, vz). Mirrors upstream's
// `(&v.x)[axisT.x] = x; ...` pattern.
inline void av_optimize_axis_to_v(int x, int y, int z,
                                  OptimizeAxis a,
                                  thread int& vx,
                                  thread int& vy,
                                  thread int& vz)
{
    int v[3] = { 0, 0, 0 };
    v[a.axis0] = x;
    v[a.axis1] = y;
    v[a.axis2] = z;
    vx = v[0];
    vy = v[1];
    vz = v[2];
}

// S48: function-constant variant. Same body as av_optimize_axis_to_v
// but uses the PSO-baked kAxis0/1/2 instead of a runtime struct.
// With function constants, Metal's compiler can fully unroll this and
// eliminate the temporary `v[3]` array — the compiler sees a fixed
// permutation at PSO-compile time.
inline void av_optimize_axis_to_v_fc(int x, int y, int z,
                                     thread int& vx,
                                     thread int& vy,
                                     thread int& vz)
{
    int v[3] = { 0, 0, 0 };
    v[kAxis0] = x;
    v[kAxis1] = y;
    v[kAxis2] = z;
    vx = v[0];
    vy = v[1];
    vz = v[2];
}

// ----------------------------------------------------------------
// av_volume_init_y_slice — set the 2D XZ plane at given `y` to a
// constant. The iteration grid is `(volDim[axisT.x], volDim[axisT.z])`;
// each thread writes one (vx, vy, vz) voxel where `y` is the slice
// index in the axisT-permuted coords.
// ----------------------------------------------------------------

struct InitYSliceParams {
    uint         volDimX;     // volume X
    uint         volDimY;     // volume Y
    uint         volDimZ;     // volume Z
    OptimizeAxis axis;
    int          y;           // slice index in axisT coords
    uint         value;       // uchar value, packed in uint
};

kernel void av_volume_init_y_slice_uchar(
    device   uchar*                  volume [[buffer(0)]],
    constant InitYSliceParams&       p      [[buffer(1)]],
    uint2                            gid    [[thread_position_in_grid]])
{
    // grid.x iterates the volume axis at axisT[0],
    // grid.y iterates the volume axis at axisT[2].
    const int x_iter = int(gid.x);
    const int z_iter = int(gid.y);

    // Bound check against the *axisT-permuted* dimensions.
    int vol[3] = { int(p.volDimX), int(p.volDimY), int(p.volDimZ) };
    if (x_iter < 0 || x_iter >= vol[p.axis.axis0]) return;
    if (z_iter < 0 || z_iter >= vol[p.axis.axis2]) return;

    int vx, vy, vz;
    av_optimize_axis_to_v(x_iter, p.y, z_iter, p.axis, vx, vy, vz);

    const uint k = av_optimize_linear_index(vx, vy, vz,
                                            p.volDimX, p.volDimY);
    volume[k] = uchar(p.value & 0xffu);
}

// ----------------------------------------------------------------
// av_volume_get_xz_slice — extract the XZ plane at given `y` from
// the 3D uchar volume and write to a flat (X,Z) slice as uint
// (TSimAcc). The slice is packed row-major: idx = z * volDimXa + x.
// ----------------------------------------------------------------

struct GetXZSliceParams {
    uint         volDimX;
    uint         volDimY;
    uint         volDimZ;
    OptimizeAxis axis;
    int          y;
};

kernel void av_volume_get_xz_slice_uchar_to_uint(
    device   uint*                   out_slice [[buffer(0)]],
    device const uchar*              in_volume [[buffer(1)]],
    constant GetXZSliceParams&       p         [[buffer(2)]],
    uint2                            gid       [[thread_position_in_grid]])
{
    const int x_iter = int(gid.x);
    const int z_iter = int(gid.y);

    int vol[3] = { int(p.volDimX), int(p.volDimY), int(p.volDimZ) };
    const int axDimX = vol[p.axis.axis0];
    const int axDimZ = vol[p.axis.axis2];
    if (x_iter >= axDimX || z_iter >= axDimZ) return;

    int vx, vy, vz;
    av_optimize_axis_to_v(x_iter, p.y, z_iter, p.axis, vx, vy, vz);

    const uint vol_k = av_optimize_linear_index(vx, vy, vz,
                                                p.volDimX, p.volDimY);
    const uint slice_k = uint(z_iter) * uint(axDimX) + uint(x_iter);
    out_slice[slice_k] = uint(in_volume[vol_k]);
}

// ----------------------------------------------------------------
// av_volume_compute_best_z_in_slice — per X column of an XZ slice,
// find the min over Z. Each thread = one X. Slice layout is packed
// row-major: idx = z * volDimXa + x.
// ----------------------------------------------------------------

struct BestZParams {
    uint volDimXa;    // axisT-permuted X dim (== slice width)
    uint volDimZ;     // slice height (full Z extent of the volume)
};

kernel void av_volume_compute_best_z_in_slice(
    device const uint*            in_slice [[buffer(0)]],
    device       uint*            out_best [[buffer(1)]],   // length volDimXa
    constant BestZParams&         p        [[buffer(2)]],
    uint                          gid      [[thread_position_in_grid]])
{
    const uint x = gid;
    if (x >= p.volDimXa) return;

    uint best = in_slice[0u * p.volDimXa + x];
    for (uint z = 1; z < p.volDimZ; ++z) {
        const uint v = in_slice[z * p.volDimXa + x];
        if (v < best) best = v;
    }
    out_best[x] = best;
}

// ----------------------------------------------------------------
// av_volume_aggregate_cost_at_x — the DP step, fixed-P2 variant.
//
// Per (x_iter, z_iter) thread, at the host-supplied `y`:
//   pathCost = sim_xz + min(pathCostMD,
//                           pathCostMDM1 + P1,
//                           pathCostMDP1 + P1,
//                           bestCostInColM1 + P2)
//                     - bestCostInColM1
// Write pathCost to the current XZ slice (used as Ym1 for the next
// iteration via the host's slice-pointer swap). Aggregate into the
// output volume:
//   out_vol[v] = (out_vol[v] * filteringIndex + clamp(pathCost,0,255))
//              / (filteringIndex + 1)
//
// The kernel skips the boundary planes (z = 0 and z = volDim.z-1);
// those keep `pathCost = 255.f` per upstream convention.
// ----------------------------------------------------------------

struct AggregateParams {
    uint         volDimX;
    uint         volDimY;
    uint         volDimZ;
    OptimizeAxis axis;
    int          y;
    float        P1;
    float        P2_abs;       // upstream's `abs(_P2)` (fixed-P2 path)
    uint         filteringIndex;
    // -------- adaptive-P2 fields (Session 22) --------
    // When `adaptive_p2 != 0` the kernel ignores P2_abs and computes
    // P2 per-thread from the rc mipmap via
    //     deltaC = euclideanDist3(gcr0, gcr1)
    //     P2     = sigmoid(80, 255, 80, p2_sig_mid, deltaC)
    // matching upstream's `_P2 >= 0` branch byte-for-byte.
    uint         adaptive_p2;      // 0 = fixed, !=0 = adaptive
    int          ySign;            // -1 if the path runs Ym1→Y in reverse, else +1
    int          stepXY;           // sgmParams.stepXY (rc-image stride per voxel)
    int          roiXBegin;
    int          roiYBegin;
    uint         rcLevelWidth;
    uint         rcLevelHeight;
    float        rcMipmapLevel;
    float        p2_sig_mid;       // upstream's `_P2` (passed signed, NOT abs).
};

// Sampler mirrors `av_dsm_mip_sampler` (depth_sim_map.metal): normalized
// coords, clamp-to-edge, linear filter + linear mip filter. This matches
// upstream's `tex2DLod<float4>` semantics for the rc mipmap.
constexpr sampler av_volopt_mip_sampler(
    coord::normalized,
    address::clamp_to_edge,
    filter::linear,
    mip_filter::linear);

kernel void av_volume_aggregate_cost_at_x(
    device       uint*                xz_slice_for_y   [[buffer(0)]],
    device const uint*                xz_slice_for_ym1 [[buffer(1)]],
    device const uint*                best_sim_in_ym1  [[buffer(2)]],
    device       uchar*               vol_agr          [[buffer(3)]],
    constant AggregateParams&         p                [[buffer(4)]],
    texture2d<float, access::sample>  rc_mip           [[texture(0)]],
    uint2                             gid              [[thread_position_in_grid]])
{
    const int x_iter = int(gid.x);
    const int z_iter = int(gid.y);

    int vol[3] = { int(p.volDimX), int(p.volDimY), int(p.volDimZ) };
    const int axDimX = vol[p.axis.axis0];
    if (x_iter >= axDimX || z_iter >= int(p.volDimZ)) return;

    int vx, vy, vz;
    av_optimize_axis_to_v(x_iter, p.y, z_iter, p.axis, vx, vy, vz);

    const uint slice_k = uint(z_iter) * uint(axDimX) + uint(x_iter);
    float pathCost = 255.0f;

    if (z_iter >= 1 && z_iter < int(p.volDimZ) - 1) {
        float P2;
        if (p.adaptive_p2 == 0u) {
            // Fixed-P2 path (Session 13): host passes the
            // already-absolute value, matching upstream's
            // `if(_P2 < 0) P2 = abs(_P2);` branch.
            P2 = p.P2_abs;
        } else {
            // Adaptive-P2 path: sample the rc mipmap at the
            // current pixel + a neighbor offset by ±step in the
            // axis-permuted Ym1 direction. Convention copies
            // upstream's `_P2 >= 0` branch verbatim.
            const int imX0 = (p.roiXBegin + vx) * p.stepXY;
            const int imY0 = (p.roiYBegin + vy) * p.stepXY;
            const int xShift = (p.axis.axis1 == 0) ? 1 : 0;
            const int yShift = (p.axis.axis1 == 1) ? 1 : 0;
            const int imX1 = imX0 - p.ySign * p.stepXY * xShift;
            const int imY1 = imY0 - p.ySign * p.stepXY * yShift;

            const float invW = 1.0f / float(p.rcLevelWidth);
            const float invH = 1.0f / float(p.rcLevelHeight);
            const float2 uv0 = float2(
                (float(imX0) + 0.5f) * invW,
                (float(imY0) + 0.5f) * invH);
            const float2 uv1 = float2(
                (float(imX1) + 0.5f) * invW,
                (float(imY1) + 0.5f) * invH);
            const float4 gcr0 = rc_mip.sample(av_volopt_mip_sampler, uv0,
                                              level(p.rcMipmapLevel));
            const float4 gcr1 = rc_mip.sample(av_volopt_mip_sampler, uv1,
                                              level(p.rcMipmapLevel));
            const float deltaC = av_depthmap::euclideanDist3(gcr0, gcr1);

            // sigmoid(zeroVal=80, endVal=255, sigwidth=80,
            //         sigMid=_P2, xval=deltaC)
            //   = 80 + (255-80) * 1/(1 + exp(10*(deltaC - _P2)/80))
            P2 = sigmoid(80.0f, 255.0f, 80.0f, p.p2_sig_mid, deltaC);
        }

        const float bestCol_m1 = float(best_sim_in_ym1[uint(x_iter)]);
        const float pathMDm1 = float(xz_slice_for_ym1[uint(z_iter - 1) * uint(axDimX) + uint(x_iter)]);
        const float pathMD   = float(xz_slice_for_ym1[uint(z_iter    ) * uint(axDimX) + uint(x_iter)]);
        const float pathMDp1 = float(xz_slice_for_ym1[uint(z_iter + 1) * uint(axDimX) + uint(x_iter)]);

        const float a = pathMD;
        const float b = pathMDm1 + p.P1;
        const float c = pathMDp1 + p.P1;
        const float d = bestCol_m1 + P2;
        const float minCost = min(min(a, b), min(c, d));

        // The "current sim" for this voxel is the input similarity
        // sample copied into xz_slice_for_y by the host's getSlice
        // dispatch earlier this iteration.
        const float sim_xz = float(xz_slice_for_y[slice_k]);
        pathCost = sim_xz + minCost - bestCol_m1;
    }

    // Update slice for Y-1 of the next iteration.
    xz_slice_for_y[slice_k] = uint(pathCost);

    // Clamp for the uchar aggregation step.
    pathCost = min(255.0f, max(0.0f, pathCost));

    const uint vol_k = av_optimize_linear_index(vx, vy, vz,
                                                p.volDimX, p.volDimY);
    const float cur = float(vol_agr[vol_k]);
    const float fi  = float(p.filteringIndex);
    const float merged = (cur * fi + pathCost) / (fi + 1.0f);
    vol_agr[vol_k] = uchar(min(255.0f, max(0.0f, merged)));
}

// ================================================================
// S48: function-constant-specialized variants of the 4 sub-kernels.
//
// These are byte-identical to the runtime-axis kernels above, except
// the axis swizzle (axis0/1/2) is read from function constants
// kAxis0/kAxis1/kAxis2 instead of `p.axis`. At PSO-compile time the
// permuted-dim lookups `vol[kAxisN]` and the array-indexed assignment
// `v[kAxisN] = ...` collapse to constants; the adaptive-P2 branch
// `if (kAxis1 == 0)` becomes a compile-time selection.
//
// The host (Volume::optimize) creates 2 PSO sets for these — one with
// kAxis0/1/2 = (0,1,2) and one with (1,0,2) — and dispatches the
// right set per path.
//
// Kernel naming convention: `_fc` suffix for "function-constant".
// ================================================================

kernel void av_volume_init_y_slice_uchar_fc(
    device   uchar*                  volume [[buffer(0)]],
    constant InitYSliceParams&       p      [[buffer(1)]],
    uint2                            gid    [[thread_position_in_grid]])
{
    const int x_iter = int(gid.x);
    const int z_iter = int(gid.y);

    // vol[kAxisN] is a constant index at PSO-compile time → the compiler
    // emits a direct read from one of volDimX/volDimY/volDimZ.
    int vol[3] = { int(p.volDimX), int(p.volDimY), int(p.volDimZ) };
    if (x_iter < 0 || x_iter >= vol[kAxis0]) return;
    if (z_iter < 0 || z_iter >= vol[kAxis2]) return;

    int vx, vy, vz;
    av_optimize_axis_to_v_fc(x_iter, p.y, z_iter, vx, vy, vz);

    const uint k = av_optimize_linear_index(vx, vy, vz,
                                            p.volDimX, p.volDimY);
    volume[k] = uchar(p.value & 0xffu);
}

kernel void av_volume_get_xz_slice_uchar_to_uint_fc(
    device   uint*                   out_slice [[buffer(0)]],
    device const uchar*              in_volume [[buffer(1)]],
    constant GetXZSliceParams&       p         [[buffer(2)]],
    uint2                            gid       [[thread_position_in_grid]])
{
    const int x_iter = int(gid.x);
    const int z_iter = int(gid.y);

    int vol[3] = { int(p.volDimX), int(p.volDimY), int(p.volDimZ) };
    const int axDimX = vol[kAxis0];
    const int axDimZ = vol[kAxis2];
    if (x_iter >= axDimX || z_iter >= axDimZ) return;

    int vx, vy, vz;
    av_optimize_axis_to_v_fc(x_iter, p.y, z_iter, vx, vy, vz);

    const uint vol_k = av_optimize_linear_index(vx, vy, vz,
                                                p.volDimX, p.volDimY);
    const uint slice_k = uint(z_iter) * uint(axDimX) + uint(x_iter);
    out_slice[slice_k] = uint(in_volume[vol_k]);
}

// compute_best_z does NOT reference the axis swizzle directly — it
// operates on a flat (X, Z) slice. We still provide a `_fc` alias so
// the host's per-axis PSO table is uniform; the function constants
// are simply unused by this kernel (PSO creation still requires the
// constants because they're declared at module scope).
kernel void av_volume_compute_best_z_in_slice_fc(
    device const uint*            in_slice [[buffer(0)]],
    device       uint*            out_best [[buffer(1)]],
    constant BestZParams&         p        [[buffer(2)]],
    uint                          gid      [[thread_position_in_grid]])
{
    const uint x = gid;
    if (x >= p.volDimXa) return;

    uint best = in_slice[0u * p.volDimXa + x];
    for (uint z = 1; z < p.volDimZ; ++z) {
        const uint v = in_slice[z * p.volDimXa + x];
        if (v < best) best = v;
    }
    out_best[x] = best;
}

kernel void av_volume_aggregate_cost_at_x_fc(
    device       uint*                xz_slice_for_y   [[buffer(0)]],
    device const uint*                xz_slice_for_ym1 [[buffer(1)]],
    device const uint*                best_sim_in_ym1  [[buffer(2)]],
    device       uchar*               vol_agr          [[buffer(3)]],
    constant AggregateParams&         p                [[buffer(4)]],
    texture2d<float, access::sample>  rc_mip           [[texture(0)]],
    uint2                             gid              [[thread_position_in_grid]])
{
    const int x_iter = int(gid.x);
    const int z_iter = int(gid.y);

    int vol[3] = { int(p.volDimX), int(p.volDimY), int(p.volDimZ) };
    const int axDimX = vol[kAxis0];
    if (x_iter >= axDimX || z_iter >= int(p.volDimZ)) return;

    int vx, vy, vz;
    av_optimize_axis_to_v_fc(x_iter, p.y, z_iter, vx, vy, vz);

    const uint slice_k = uint(z_iter) * uint(axDimX) + uint(x_iter);
    float pathCost = 255.0f;

    if (z_iter >= 1 && z_iter < int(p.volDimZ) - 1) {
        float P2;
        if (p.adaptive_p2 == 0u) {
            P2 = p.P2_abs;
        } else {
            const int imX0 = (p.roiXBegin + vx) * p.stepXY;
            const int imY0 = (p.roiYBegin + vy) * p.stepXY;
            // S48: kAxis1 is a compile-time constant — these two
            // ternaries collapse to constants at PSO-compile time
            // (one of the two becomes 1 and the other 0).
            const int xShift = (kAxis1 == 0) ? 1 : 0;
            const int yShift = (kAxis1 == 1) ? 1 : 0;
            const int imX1 = imX0 - p.ySign * p.stepXY * xShift;
            const int imY1 = imY0 - p.ySign * p.stepXY * yShift;

            const float invW = 1.0f / float(p.rcLevelWidth);
            const float invH = 1.0f / float(p.rcLevelHeight);
            const float2 uv0 = float2(
                (float(imX0) + 0.5f) * invW,
                (float(imY0) + 0.5f) * invH);
            const float2 uv1 = float2(
                (float(imX1) + 0.5f) * invW,
                (float(imY1) + 0.5f) * invH);
            const float4 gcr0 = rc_mip.sample(av_volopt_mip_sampler, uv0,
                                              level(p.rcMipmapLevel));
            const float4 gcr1 = rc_mip.sample(av_volopt_mip_sampler, uv1,
                                              level(p.rcMipmapLevel));
            const float deltaC = av_depthmap::euclideanDist3(gcr0, gcr1);
            P2 = sigmoid(80.0f, 255.0f, 80.0f, p.p2_sig_mid, deltaC);
        }

        const float bestCol_m1 = float(best_sim_in_ym1[uint(x_iter)]);
        const float pathMDm1 = float(xz_slice_for_ym1[uint(z_iter - 1) * uint(axDimX) + uint(x_iter)]);
        const float pathMD   = float(xz_slice_for_ym1[uint(z_iter    ) * uint(axDimX) + uint(x_iter)]);
        const float pathMDp1 = float(xz_slice_for_ym1[uint(z_iter + 1) * uint(axDimX) + uint(x_iter)]);

        const float a = pathMD;
        const float b = pathMDm1 + p.P1;
        const float c = pathMDp1 + p.P1;
        const float d = bestCol_m1 + P2;
        const float minCost = min(min(a, b), min(c, d));

        const float sim_xz = float(xz_slice_for_y[slice_k]);
        pathCost = sim_xz + minCost - bestCol_m1;
    }

    xz_slice_for_y[slice_k] = uint(pathCost);
    pathCost = min(255.0f, max(0.0f, pathCost));

    const uint vol_k = av_optimize_linear_index(vx, vy, vz,
                                                p.volDimX, p.volDimY);
    const float cur = float(vol_agr[vol_k]);
    const float fi  = float(p.filteringIndex);
    const float merged = (cur * fi + pathCost) / (fi + 1.0f);
    vol_agr[vol_k] = uchar(min(255.0f, max(0.0f, merged)));
}
