// depth_sim_map.metal — post-processing kernels on the
// (depth, similarity) maps emitted by SGM and Refine.
//
// Phase 7 / `deviceDepthSimilarityMap.cu` port (Session 19+).
//
// Layout: depth-sim maps are float2 packed, row-major, no pitch.
// Linear index: `y * width + x`. Matches the rest of our port.

#include <metal_stdlib>
using namespace metal;

#include "matrix.h"    // sigmoid, sigmoid2, closestPointToLine3D, angleBetwABandAC
#include "Patch.h"     // DeviceCameraParams, get3DPointForPixelAndDepthFromRC
#include "eig33.h"     // eig33_decompose (used by Stat3d PCA)
using av_depthmap::DeviceCameraParams;
using av_depthmap::get3DPointForPixelAndDepthFromRC;

// ---------------------------------------------------------------
// av_depth_sim_map_copy_depth_only
// ---------------------------------------------------------------
// Port of `depthSimMapCopyDepthOnly_kernel` from
// `deviceDepthSimilarityMapKernels.cuh`.
//
// Read `(depth, _)` from `in_map`, write `(depth, default_sim)` to
// `out_map`. Pure copy — no cameras, no textures.

struct DepthSimMapCopyDepthOnlyParams {
    uint  width;
    uint  height;
    float default_sim;
};

kernel void av_depth_sim_map_copy_depth_only(
    device       float2*                            out_map [[buffer(0)]],
    device const float2*                            in_map  [[buffer(1)]],
    constant DepthSimMapCopyDepthOnlyParams&        p       [[buffer(2)]],
    uint2                                           gid     [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    const uint idx = gid.y * p.width + gid.x;
    out_map[idx] = float2(in_map[idx].x, p.default_sim);
}

// ---------------------------------------------------------------
// av_map_upscale_float3
// ---------------------------------------------------------------
// Port of `mapUpscale_kernel<float3>` from
// `deviceDepthSimilarityMapKernels.cuh`. The host method
// `DepthSimMap::normal_map_upscale` mirrors `cuda_normalMapUpscale`.
//
// Nearest-neighbor upscale. For each output pixel `(x, y)` the
// source coordinate is computed via the upstream formula:
//     ox = (x - 0.5) * ratio
//     oy = (y - 0.5) * ratio
//     xp = clamp(floor(ox + 0.5), 0, in_w - 1)
//     yp = clamp(floor(oy + 0.5), 0, in_h - 1)
// Then `out[y][x] = in[yp][xp]`.
//
// `ratio = in_width / out_width` (so ratio < 1 for upscale).
// `packed_float3` matches CUDA's `float3` (12-byte stride).

struct MapUpscaleFloat3Params {
    uint  out_width;
    uint  out_height;
    uint  in_width;     // row stride for the source map
    uint  in_height;
    float ratio;        // = float(in_width) / float(out_width)
};

kernel void av_map_upscale_float3(
    device       packed_float3*               out_map [[buffer(0)]],
    device const packed_float3*               in_map  [[buffer(1)]],
    constant MapUpscaleFloat3Params&          p       [[buffer(2)]],
    uint2                                     gid     [[thread_position_in_grid]])
{
    if (gid.x >= p.out_width || gid.y >= p.out_height) {
        return;
    }

    const float ox = (float(gid.x) - 0.5f) * p.ratio;
    const float oy = (float(gid.y) - 0.5f) * p.ratio;

    // Upstream clamps with `int(out_w * ratio) - 1`, which equals
    // `int(in_w) - 1` for ratio = in_w / out_w (modulo FP rounding).
    // We use the pre-computed `in_w - 1` to avoid the FP roundtrip.
    const int max_x = int(p.in_width)  - 1;
    const int max_y = int(p.in_height) - 1;
    const int xp = min(int(floor(ox + 0.5f)), max_x);
    const int yp = min(int(floor(oy + 0.5f)), max_y);

    const uint out_idx = gid.y * p.out_width  + gid.x;
    const uint in_idx  = uint(yp) * p.in_width + uint(xp);
    out_map[out_idx] = in_map[in_idx];
}

// ---------------------------------------------------------------
// av_depth_thickness_smooth_thickness
// ---------------------------------------------------------------
// Port of `depthThicknessMapSmoothThickness_kernel` from
// `deviceDepthSimilarityMapKernels.cuh`.
//
// Smooth the `thickness` channel of a (depth, thickness) map by
// averaging clamped depth-distances over the 3×3 neighborhood.
// Pixels with non-positive depth are skipped (in either the
// center or a neighbor).
//
// In-place: the kernel only writes `inout_map[idx].y`; `.x` is
// only read. Neighbor reads of `.x` are not affected by other
// threads' writes — the data race on `.y` is benign because no
// other thread reads `.y`.
//
// Inflate factors:
//   min_thickness = min_thickness_inflate × center.thickness
//   max_thickness = max_thickness_inflate × center.thickness
// Caller computes these from the SGM and Refine parameters
// (upstream's `cuda_depthThicknessSmoothThickness` does the
// computation; we expose the scalars directly).

struct DepthThicknessSmoothParams {
    uint  width;
    uint  height;
    float min_thickness_inflate;
    float max_thickness_inflate;
};

kernel void av_depth_thickness_smooth_thickness(
    device       float2*                             inout_map [[buffer(0)]],
    constant DepthThicknessSmoothParams&             p         [[buffer(1)]],
    uint2                                            gid       [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    const uint idx = gid.y * p.width + gid.x;
    const float2 center = inout_map[idx];

    // Invalid / masked center pixel.
    if (center.x <= 0.0f) {
        return;
    }

    const float min_t = p.min_thickness_inflate * center.y;
    const float max_t = p.max_thickness_inflate * center.y;

    float sum_center_dist = 0.0f;
    int   nb_valid        = 0;

    const int cx = int(gid.x);
    const int cy = int(gid.y);
    for (int yp = -1; yp <= 1; ++yp) {
        for (int xp = -1; xp <= 1; ++xp) {
            if (xp == 0 && yp == 0) {
                continue;
            }
            const int nx = cx + xp;
            const int ny = cy + yp;
            if (nx < 0 || nx >= int(p.width) ||
                ny < 0 || ny >= int(p.height)) {
                continue;
            }
            const float2 patch = inout_map[uint(ny) * p.width + uint(nx)];
            if (patch.x > 0.0f) {
                const float d = fabs(center.x - patch.x);
                sum_center_dist += max(min_t, min(max_t, d));
                ++nb_valid;
            }
        }
    }

    // Require at least 3 valid patch pixels (over 8).
    if (nb_valid < 3) {
        return;
    }
    inout_map[idx].y = sum_center_dist / float(nb_valid);
}

// ---------------------------------------------------------------
// av_compute_sgm_upscaled_depth_pix_size_map_{nearest,bilinear}
// ---------------------------------------------------------------
// Port of `computeSgmUpscaledDepthPixSizeMap_{nearestNeighbor,
// bilinear}_kernel` from `deviceDepthSimilarityMapKernels.cuh`.
//
// Both variants upscale the SGM-resolution (depth, thickness) map
// to the Refine resolution, with per-pixel pixSize computed from
// thickness (`pixSize = thickness / halfNbDepths`). The compute-
// pixsize-via-camera path (upstream's
// `ALICEVISION_DEPTHMAP_COMPUTE_PIXSIZEMAP`) is commented out
// upstream by default and is omitted here. Caller passes the
// rc mipmap texture for the alpha-mask pre-filter.
//
// Inputs:
//   * in_sgm_map (sgm-resolution float2 (depth, thickness) map)
//   * rc mipmap texture (sample at level rc_mipmap_level for the
//     alpha mask check at image-space coord (x, y))
//   * params
//
// Output:
//   * out_map (refine-resolution float2 (depth, pixSize) map)
//
// The masking threshold is upstream-bug-for-bug: the nearest
// variant uses `0.9f` (likely intended for [0, 1] textures);
// the bilinear variant uses `ALICEVISION_DEPTHMAP_RC_MIN_ALPHA
// = 255 * 0.9 = 229.5f` (correct for [0, 255] textures). We
// preserve both behaviors.

#define AV_RC_MIN_ALPHA_255    229.5f   // ALICEVISION_DEPTHMAP_RC_MIN_ALPHA
#define AV_RC_MIN_ALPHA_LEGACY 0.9f     // nearest-variant constant (upstream)

struct ComputeUpscaledDepthPixSizeMapParams {
    uint  out_width;
    uint  out_height;
    uint  in_width;
    uint  in_height;
    uint  roi_x_begin;
    uint  roi_y_begin;
    uint  rc_level_width;
    uint  rc_level_height;
    float rc_mipmap_level;
    int   step_xy;
    int   half_nb_depths;
    float ratio;
};

constexpr sampler av_dsm_mip_sampler(
    coord::normalized,
    address::clamp_to_edge,
    filter::linear,
    mip_filter::linear);

inline float2 av_load_sgm(device const float2* in_map,
                          int xp, int yp, uint in_width)
{
    return in_map[uint(yp) * in_width + uint(xp)];
}

// --- nearest-neighbor variant ---
kernel void av_compute_sgm_upscaled_depth_pix_size_map_nearest(
    device       float2*                                out_map [[buffer(0)]],
    device const float2*                                in_map  [[buffer(1)]],
    constant ComputeUpscaledDepthPixSizeMapParams&      p       [[buffer(2)]],
    texture2d<float, access::sample>                    rc_mip  [[texture(0)]],
    uint2                                               gid     [[thread_position_in_grid]])
{
    if (gid.x >= p.out_width || gid.y >= p.out_height) {
        return;
    }
    const uint roiX = gid.x;
    const uint roiY = gid.y;

    const uint x = (p.roi_x_begin + roiX) * uint(p.step_xy);
    const uint y = (p.roi_y_begin + roiY) * uint(p.step_xy);

    const uint out_idx = roiY * p.out_width + roiX;

    // Alpha-mask pre-filter at the rc mipmap level.
    const float2 uv = float2(
        (float(x) + 0.5f) / float(p.rc_level_width),
        (float(y) + 0.5f) / float(p.rc_level_height));
    const float4 rc_color =
        rc_mip.sample(av_dsm_mip_sampler, uv, level(p.rc_mipmap_level));
    // Upstream nearest variant uses bare `0.9f` (likely a bug;
    // textures are [0, 255]). Preserve faithfully.
    if (rc_color.w < AV_RC_MIN_ALPHA_LEGACY) {
        out_map[out_idx] = float2(-2.0f, 0.0f);
        return;
    }

    // Nearest-neighbor lookup into SGM map.
    const float ox = (float(roiX) - 0.5f) * p.ratio;
    const float oy = (float(roiY) - 0.5f) * p.ratio;
    int xp = int(floor(ox + 0.5f));
    int yp = int(floor(oy + 0.5f));
    xp = min(xp, int(p.in_width)  - 1);
    yp = min(yp, int(p.in_height) - 1);

    const float2 dth = av_load_sgm(in_map, xp, yp, p.in_width);
    const float  pixSize = dth.y / float(p.half_nb_depths);

    out_map[out_idx] = float2(dth.x, pixSize);
}

// --- bilinear variant ---
kernel void av_compute_sgm_upscaled_depth_pix_size_map_bilinear(
    device       float2*                                out_map [[buffer(0)]],
    device const float2*                                in_map  [[buffer(1)]],
    constant ComputeUpscaledDepthPixSizeMapParams&      p       [[buffer(2)]],
    texture2d<float, access::sample>                    rc_mip  [[texture(0)]],
    uint2                                               gid     [[thread_position_in_grid]])
{
    if (gid.x >= p.out_width || gid.y >= p.out_height) {
        return;
    }
    const uint roiX = gid.x;
    const uint roiY = gid.y;

    const uint x = (p.roi_x_begin + roiX) * uint(p.step_xy);
    const uint y = (p.roi_y_begin + roiY) * uint(p.step_xy);

    const uint out_idx = roiY * p.out_width + roiX;

    const float2 uv = float2(
        (float(x) + 0.5f) / float(p.rc_level_width),
        (float(y) + 0.5f) / float(p.rc_level_height));
    const float4 rc_color =
        rc_mip.sample(av_dsm_mip_sampler, uv, level(p.rc_mipmap_level));
    // Both variants now use the LEGACY 0.9 threshold (S40 fix). Our
    // textures preserve EXR alpha in [0, 1] — opaque pixels have
    // alpha = 1.0, masked pixels < 0.9. The upstream 229.5 constant
    // was a [0, 255]-uchar artifact we don't share. See mental_note §8i.
    if (rc_color.w < AV_RC_MIN_ALPHA_LEGACY) {
        out_map[out_idx] = float2(-2.0f, 0.0f);
        return;
    }

    // 4-tap bilinear into SGM map (with corner-fallback when any
    // corner depth is invalid).
    const float ox = (float(roiX) - 0.5f) * p.ratio;
    const float oy = (float(roiY) - 0.5f) * p.ratio;
    int xp = int(floor(ox));
    int yp = int(floor(oy));
    // S48: clamp from BOTH ends. The upper clamp ensures the 2x2
    // stencil fits within the SGM map. The lower clamp (max(., 0))
    // matches `clamp_to_edge` behavior at the top/left boundary —
    // without it, edge ROI pixels (e.g. roiX=0 with ratio<1) get
    // xp=-1 and OOB-load uninitialized device memory. That was the
    // root cause of `test_upscale_depth_pixsize`'s -j8 flakiness:
    // the OOB-read garbage varied with heap state across parallel
    // ctest processes, producing 62 mismatches ~1/3 of -j8 runs.
    xp = clamp(xp, 0, int(p.in_width)  - 2);
    yp = clamp(yp, 0, int(p.in_height) - 2);

    const float2 lu = av_load_sgm(in_map, xp,     yp,     p.in_width);
    const float2 ru = av_load_sgm(in_map, xp + 1, yp,     p.in_width);
    const float2 rd = av_load_sgm(in_map, xp + 1, yp + 1, p.in_width);
    const float2 ld = av_load_sgm(in_map, xp,     yp + 1, p.in_width);

    float2 dth;
    if (lu.x <= 0.0f || ru.x <= 0.0f || rd.x <= 0.0f || ld.x <= 0.0f) {
        // At least one corner invalid: average the valid ones.
        float2 sum = float2(0.0f, 0.0f);
        int count = 0;
        if (lu.x > 0.0f) { sum += lu; ++count; }
        if (ru.x > 0.0f) { sum += ru; ++count; }
        if (rd.x > 0.0f) { sum += rd; ++count; }
        if (ld.x > 0.0f) { sum += ld; ++count; }
        if (count == 0) {
            out_map[out_idx] = float2(-1.0f, 1.0f);
            return;
        }
        dth = sum / float(count);
    } else {
        // S48: clamp weights to [0, 1] so the edge-clamped stencil
        // does proper interpolation (not extrapolation) at the
        // boundary. Matches texture `clamp_to_edge` semantics.
        const float ui = clamp(ox - float(xp), 0.0f, 1.0f);
        const float vi = clamp(oy - float(yp), 0.0f, 1.0f);
        const float2 u = lu + (ru - lu) * ui;
        const float2 d = ld + (rd - ld) * ui;
        dth = u + (d - u) * vi;
    }

    const float pixSize = dth.y / float(p.half_nb_depths);
    out_map[out_idx] = float2(dth.x, pixSize);
}

// ---------------------------------------------------------------
// av_depth_sim_map_compute_normal
// ---------------------------------------------------------------
// Port of `depthSimMapComputeNormal_kernel<TWsh>` from
// `deviceDepthSimilarityMapKernels.cuh` (upstream instantiates
// only `<3>`; we hardcode wsh=3 here).
//
// For each output pixel:
//   * Read center depth from in_depth_sim_map (.x only).
//   * If invalid (depth ≤ 0), write (-1, -1, -1) and return.
//   * Get 3D point `p` + per-pixel pixSize (3D distance to the
//     +1-pixel neighbor at the same depth).
//   * Walk a (2·wsh+1)² = 7×7 patch:
//       - For each neighbor with valid depth and |Δdepth| <
//         30 × pixSize, get its 3D point and add to a Stat3d
//         accumulator.
//   * Fit a plane via PCA (eig33 on the covariance) → smallest-
//     eigenvalue eigenvector = surface normal `n`.
//   * Orient `n` toward the camera using
//     `orientedPointPlaneDistanceNormalizedNormal`.
//
// Precision: FP32 accumulators (the GPU has no FP64). Upstream
// CUDA uses FP64 accumulators in `cuda_stat3d` — see the S22
// handover for the precision trade-off analysis.

// Stat3d — FP32 accumulator for the PCA plane fit. Mirrors the
// CUDA `cuda_stat3d` struct but with `float` accumulators.
struct Stat3d {
    float xsum, ysum, zsum;
    float xxsum, yysum, zzsum;
    float xysum, xzsum, yzsum;
    float count;
};

inline void av_stat3d_init(thread Stat3d& s)
{
    s.xsum = s.ysum = s.zsum = 0.0f;
    s.xxsum = s.yysum = s.zzsum = 0.0f;
    s.xysum = s.xzsum = s.yzsum = 0.0f;
    s.count = 0.0f;
}

inline void av_stat3d_update(thread Stat3d& s, float3 p, float w)
{
    s.xxsum += p.x * p.x;
    s.yysum += p.y * p.y;
    s.zzsum += p.z * p.z;
    s.xysum += p.x * p.y;
    s.xzsum += p.x * p.z;
    s.yzsum += p.y * p.z;
    s.xsum  += p.x;
    s.ysum  += p.y;
    s.zsum  += p.z;
    s.count += w;
}

// Fit a plane via PCA. Mirrors upstream's `computePlaneByPCA`
// (eig33 on the covariance; the smallest-eigenvalue eigenvector
// is the surface normal). Returns false if count < 3.
inline bool av_stat3d_compute_plane(
    thread const Stat3d& s,
    thread float3&       plane_point,
    thread float3&       plane_normal)
{
    if (s.count < 3.0f) {
        return false;
    }
    const float xmean = s.xsum / s.count;
    const float ymean = s.ysum / s.count;
    const float zmean = s.zsum / s.count;

    // Build the covariance matrix using upstream's literal form
    // to preserve evaluation order. Mathematically this reduces
    // to `xxsum/count - xmean*xmean` for the diagonal etc., but
    // upstream's unsimplified form is what we match.
    float A[3][3];
    A[0][0] = (s.xxsum - s.xsum * xmean - s.xsum * xmean + xmean * xmean * s.count) / s.count;
    A[0][1] = (s.xysum - s.ysum * xmean - s.xsum * ymean + xmean * ymean * s.count) / s.count;
    A[0][2] = (s.xzsum - s.zsum * xmean - s.xsum * zmean + xmean * zmean * s.count) / s.count;
    A[1][0] = (s.xysum - s.xsum * ymean - s.ysum * xmean + ymean * xmean * s.count) / s.count;
    A[1][1] = (s.yysum - s.ysum * ymean - s.ysum * ymean + ymean * ymean * s.count) / s.count;
    A[1][2] = (s.yzsum - s.zsum * ymean - s.ysum * zmean + ymean * zmean * s.count) / s.count;
    A[2][0] = (s.xzsum - s.xsum * zmean - s.zsum * xmean + zmean * xmean * s.count) / s.count;
    A[2][1] = (s.yzsum - s.ysum * zmean - s.zsum * ymean + zmean * ymean * s.count) / s.count;
    A[2][2] = (s.zzsum - s.zsum * zmean - s.zsum * zmean + zmean * zmean * s.count) / s.count;

    float V[3][3], d[3];
    eig33_decompose(A, V, d);

    // d[] sorted ascending: smallest eigenvalue is d[0]; its
    // eigenvector is column 0 of V.
    plane_normal = float3(V[0][0], V[1][0], V[2][0]);
    plane_normal = normalize(plane_normal);
    plane_point  = float3(xmean, ymean, zmean);
    return true;
}

inline float av_oriented_point_plane_distance(
    float3 point, float3 plane_point, float3 plane_normal_unit)
{
    return dot(point, plane_normal_unit) - dot(plane_point, plane_normal_unit);
}

struct DepthSimMapComputeNormalParams {
    uint  width;          // output ROI width (dispatch dim)
    uint  height;         // output ROI height (dispatch dim)
    uint  roi_x_begin;    // image-space anchor
    uint  roi_y_begin;
    int   step_xy;
    int   wsh;            // patch half-width (=3 for the upstream instantiation)
};

kernel void av_depth_sim_map_compute_normal(
    device       packed_float3*                       out_normal_map  [[buffer(0)]],
    device const float2*                              in_depth_sim    [[buffer(1)]],
    constant DepthSimMapComputeNormalParams&          p               [[buffer(2)]],
    constant DeviceCameraParams&                      rc              [[buffer(3)]],
    uint2                                             gid             [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    const uint roiX = gid.x;
    const uint roiY = gid.y;

    // Image-space coords (full resolution).
    const uint x = (p.roi_x_begin + roiX) * uint(p.step_xy);
    const uint y = (p.roi_y_begin + roiY) * uint(p.step_xy);

    const uint out_idx = roiY * p.width + roiX;
    const float in_depth =
        in_depth_sim[roiY * p.width + roiX].x;   // .x channel = depth

    if (in_depth <= 0.0f) {
        out_normal_map[out_idx] = packed_float3(-1.0f, -1.0f, -1.0f);
        return;
    }

    // 3D point + per-pixel pixSize.
    const float3 P =
        get3DPointForPixelAndDepthFromRC(rc, float2(float(x), float(y)), in_depth);
    const float3 P_right =
        get3DPointForPixelAndDepthFromRC(rc, float2(float(x + 1), float(y)), in_depth);
    const float  pixSize = length(P - P_right);

    Stat3d s;
    av_stat3d_init(s);

    for (int yp = -p.wsh; yp <= p.wsh; ++yp) {
        const int roiYp = int(roiY) + yp;
        if (roiYp < 0) {
            continue;
        }
        if (roiYp >= int(p.height)) {
            continue;
        }
        for (int xp = -p.wsh; xp <= p.wsh; ++xp) {
            const int roiXp = int(roiX) + xp;
            if (roiXp < 0) {
                continue;
            }
            if (roiXp >= int(p.width)) {
                continue;
            }
            const float depth_p =
                in_depth_sim[uint(roiYp) * p.width + uint(roiXp)].x;

            if (depth_p > 0.0f &&
                fabs(depth_p - in_depth) < 30.0f * pixSize) {
                const float2 pix_p = float2(
                    float(int(x) + xp),
                    float(int(y) + yp));
                const float3 PP =
                    get3DPointForPixelAndDepthFromRC(rc, pix_p, depth_p);
                av_stat3d_update(s, PP, 1.0f);
            }
        }
    }

    float3 plane_point = P;
    float3 plane_normal = float3(-1.0f, -1.0f, -1.0f);

    if (!av_stat3d_compute_plane(s, plane_point, plane_normal)) {
        out_normal_map[out_idx] = packed_float3(-1.0f, -1.0f, -1.0f);
        return;
    }

    // Orient the normal toward the camera.
    float3 nc = rc.C - P;
    nc = normalize(nc);
    if (av_oriented_point_plane_distance(
            plane_point + plane_normal, plane_point, nc) < 0.0f) {
        plane_normal = -plane_normal;
    }

    out_normal_map[out_idx] = packed_float3(
        plane_normal.x, plane_normal.y, plane_normal.z);
}

// ===============================================================
//   optimize_*  —  gradient-descent fusion of SGM and Refine.
// ===============================================================
// Port of the three `optimize_*` kernels in
// `deviceDepthSimilarityMapKernels.cuh` plus the
// `getCellSmoothStepEnergy` helper.
//
// Host orchestration (in `DepthSimMap::optimize_depth_sim_map`):
//   1. Copy SGM (depth, pixSize) → out_opt (depth, sim).
//   2. Run `av_optimize_var_l_of_lab_to_w` once (writes the
//      variance map from the rc mipmap's L channel gradient).
//   3. Iterate N times:
//      a. `av_optimize_get_opt_depth_map`: copy out_opt.x →
//         depth texture for the next iteration's sampling.
//      b. `av_optimize_depth_sim_map`: read SGM, Refine, current
//         opt, variance, depth texture; gradient-descent update.

// Sampler used to read the variance + depth textures from
// `optimize_depth_sim_map`. Upstream wraps these as
// `CudaTexture<float, false /*normalized*/, false /*linear*/>` —
// pixel coords, nearest filter. The MSL equivalent:
constexpr sampler av_dsm_pixel_sampler(
    coord::pixel, address::clamp_to_edge, filter::nearest);

// -----------------------------------------------------------------
// av_get_cell_smooth_step_energy
// -----------------------------------------------------------------
// Port of `getCellSmoothStepEnergy` (line 25 of
// `deviceDepthSimilarityMapKernels.cuh`).
//
// Inputs:
//   rc        — R camera params.
//   depth_tex — texture holding the per-pixel current optimized
//               depth (.x channel from out_opt).
//   cell0     — float2(roiX, roiY); ROI-local integer coords.
//               (No 0.5 offset because nearest sampling.)
//   offset_roi — float2(roi_x_begin, roi_y_begin); to convert
//                ROI-local back to image-space coords for the
//                3D reconstruction.
//
// Returns:
//   (smoothStep, energy)
//     smoothStep — signed depth distance from p0 to the line
//                  passing through the camera and the neighbor
//                  centroid. 0 if no valid neighbors.
//     energy     — max of (180 - angle(A, B, C)) over the L-R
//                  and U-B neighbor pairs, in degrees.
//                  Defaults to 180 if no valid neighbor pair.
inline float2 av_get_cell_smooth_step_energy(
    constant DeviceCameraParams&     rc,
    texture2d<float, access::sample> depth_tex,
    float2                           cell0,
    float2                           offset_roi)
{
    float2 out = float2(0.0f, 180.0f);
    const float d0 = depth_tex.sample(av_dsm_pixel_sampler, cell0).x;
    if (d0 <= 0.0f) {
        return out;
    }
    const float2 cellL = cell0 + float2( 0.0f, -1.0f);
    const float2 cellR = cell0 + float2( 0.0f,  1.0f);
    const float2 cellU = cell0 + float2(-1.0f,  0.0f);
    const float2 cellB = cell0 + float2( 1.0f,  0.0f);
    const float dL = depth_tex.sample(av_dsm_pixel_sampler, cellL).x;
    const float dR = depth_tex.sample(av_dsm_pixel_sampler, cellR).x;
    const float dU = depth_tex.sample(av_dsm_pixel_sampler, cellU).x;
    const float dB = depth_tex.sample(av_dsm_pixel_sampler, cellB).x;

    const float3 p0 = get3DPointForPixelAndDepthFromRC(rc, cell0 + offset_roi, d0);
    const float3 pL = get3DPointForPixelAndDepthFromRC(rc, cellL + offset_roi, dL);
    const float3 pR = get3DPointForPixelAndDepthFromRC(rc, cellR + offset_roi, dR);
    const float3 pU = get3DPointForPixelAndDepthFromRC(rc, cellU + offset_roi, dU);
    const float3 pB = get3DPointForPixelAndDepthFromRC(rc, cellB + offset_roi, dB);

    float3 cg = float3(0.0f, 0.0f, 0.0f);
    float  n  = 0.0f;
    if (dL > 0.0f) { cg += pL; n += 1.0f; }
    if (dR > 0.0f) { cg += pR; n += 1.0f; }
    if (dU > 0.0f) { cg += pU; n += 1.0f; }
    if (dB > 0.0f) { cg += pB; n += 1.0f; }

    if (n > 1.0f) {
        cg = cg / n;
        const float3 vcn = normalize(rc.C - p0);
        const float3 pS = closestPointToLine3D(cg, p0, vcn);
        out.x = length(rc.C - pS) - d0;
    }

    float e = 0.0f;
    n = 0.0f;
    if (dL > 0.0f && dR > 0.0f) {
        e = max(e, (180.0f - angleBetwABandAC(p0, pL, pR)));
        n += 1.0f;
    }
    if (dU > 0.0f && dB > 0.0f) {
        e = max(e, (180.0f - angleBetwABandAC(p0, pU, pB)));
        n += 1.0f;
    }
    if (n > 0.0f) {
        out.y = e;
    }
    return out;
}

// -----------------------------------------------------------------
// av_optimize_var_l_of_lab_to_w
// -----------------------------------------------------------------
// Port of `optimize_varLofLABtoW_kernel`. Writes the per-pixel L
// gradient magnitude (in [0, 255] scale) from the rc mipmap to a
// single-channel float texture.

struct OptimizeVarLParams {
    uint  width;          // dispatch dim
    uint  height;
    uint  roi_x_begin;
    uint  roi_y_begin;
    uint  rc_level_width;
    uint  rc_level_height;
    float rc_mipmap_level;
    int   step_xy;
};

kernel void av_optimize_var_l_of_lab_to_w(
    texture2d<float, access::write>   out_var [[texture(0)]],
    texture2d<float, access::sample>  rc_mip  [[texture(1)]],
    constant OptimizeVarLParams&      p       [[buffer(0)]],
    uint2                             gid     [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    const float x = float(p.roi_x_begin + gid.x) * float(p.step_xy);
    const float y = float(p.roi_y_begin + gid.y) * float(p.step_xy);
    const float inv_w = 1.0f / float(p.rc_level_width);
    const float inv_h = 1.0f / float(p.rc_level_height);

    const float xM1 = rc_mip.sample(av_dsm_mip_sampler,
        float2(((x - 1.0f) + 0.5f) * inv_w, ((y + 0.0f) + 0.5f) * inv_h),
        level(p.rc_mipmap_level)).x;
    const float xP1 = rc_mip.sample(av_dsm_mip_sampler,
        float2(((x + 1.0f) + 0.5f) * inv_w, ((y + 0.0f) + 0.5f) * inv_h),
        level(p.rc_mipmap_level)).x;
    const float yM1 = rc_mip.sample(av_dsm_mip_sampler,
        float2(((x + 0.0f) + 0.5f) * inv_w, ((y - 1.0f) + 0.5f) * inv_h),
        level(p.rc_mipmap_level)).x;
    const float yP1 = rc_mip.sample(av_dsm_mip_sampler,
        float2(((x + 0.0f) + 0.5f) * inv_w, ((y + 1.0f) + 0.5f) * inv_h),
        level(p.rc_mipmap_level)).x;

    const float2 g = float2(xM1 - xP1, yM1 - yP1);
    const float grad = length(g);
    out_var.write(float4(grad, 0.0f, 0.0f, 0.0f), gid);
}

// -----------------------------------------------------------------
// av_optimize_get_opt_depth_map
// -----------------------------------------------------------------
// Port of `optimize_getOptDeptMapFromOptDepthSimMap_kernel`.
// Copies the .x channel of a float2 depth/sim map into a single-
// channel float texture (used as the depth sampler in the next
// iteration of optimize_depth_sim_map).

struct OptimizeGetOptDepthParams {
    uint width;
    uint height;
};

kernel void av_optimize_get_opt_depth_map(
    texture2d<float, access::write>           out_depth [[texture(0)]],
    device const float2*                      in_opt    [[buffer(0)]],
    constant OptimizeGetOptDepthParams&       p         [[buffer(1)]],
    uint2                                     gid       [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    const uint idx = gid.y * p.width + gid.x;
    const float d  = in_opt[idx].x;
    out_depth.write(float4(d, 0.0f, 0.0f, 0.0f), gid);
}

// -----------------------------------------------------------------
// av_optimize_depth_sim_map
// -----------------------------------------------------------------
// Port of `optimize_depthSimMap_kernel`. The gradient-descent step.

struct OptimizeDepthSimMapParams {
    uint  width;
    uint  height;
    uint  roi_x_begin;
    uint  roi_y_begin;
    int   iter;
};

kernel void av_optimize_depth_sim_map(
    device       float2*                          out_opt      [[buffer(0)]],
    device const float2*                          in_sgm       [[buffer(1)]],
    device const float2*                          in_refine    [[buffer(2)]],
    constant OptimizeDepthSimMapParams&           p            [[buffer(3)]],
    constant DeviceCameraParams&                  rc           [[buffer(4)]],
    texture2d<float, access::sample>              variance_tex [[texture(0)]],
    texture2d<float, access::sample>              depth_tex    [[texture(1)]],
    uint2                                         gid          [[thread_position_in_grid]])
{
    if (gid.x >= p.width || gid.y >= p.height) {
        return;
    }
    const uint idx = gid.y * p.width + gid.x;

    const float2 sgm_dp = in_sgm[idx];
    const float  sgm_depth   = sgm_dp.x;
    const float  sgm_pix_sz  = sgm_dp.y;

    const float2 ref_ds   = in_refine[idx];
    const float  ref_depth = ref_ds.x;
    const float  ref_sim   = ref_ds.y;

    float2 out_ds = (p.iter == 0)
        ? float2(sgm_depth, ref_sim)
        : out_opt[idx];
    const float depth_opt = out_ds.x;

    if (depth_opt > 0.0f) {
        const float2 smooth_energy = av_get_cell_smooth_step_energy(
            rc, depth_tex,
            float2(float(gid.x), float(gid.y)),
            float2(float(p.roi_x_begin), float(p.roi_y_begin)));
        float step_to_smooth = smooth_energy.x;
        step_to_smooth = copysign(min(fabs(step_to_smooth), sgm_pix_sz / 10.0f),
                                  step_to_smooth);
        const float depth_energy = smooth_energy.y;

        float step_to_fine = ref_depth - depth_opt;
        step_to_fine = copysign(min(fabs(step_to_fine), sgm_pix_sz / 10.0f),
                                step_to_fine);

        const float step_to_rough = sgm_depth - depth_opt;
        const float img_var = variance_tex.sample(
            av_dsm_pixel_sampler,
            float2(float(gid.x), float(gid.y))).x;
        constexpr float color_var_threshold = 20.0f;
        constexpr float angle_threshold     = 30.0f;

        const float weighted_color_var =
            sigmoid2(5.0f, angle_threshold, 40.0f, color_var_threshold, img_var);
        const float fine_sim_weight =
            sigmoid(0.0f, 1.0f, 0.7f, -0.7f, ref_sim);
        const float energy_lower_weight =
            sigmoid(0.0f, 1.0f, 30.0f, weighted_color_var, depth_energy);
        const float close_to_rough_weight =
            1.0f - sigmoid(0.0f, 1.0f, 10.0f, 17.0f,
                           fabs(step_to_rough / sgm_pix_sz));

        const float depth_opt_step =
            close_to_rough_weight * step_to_rough +
            (1.0f - close_to_rough_weight) *
                (energy_lower_weight * fine_sim_weight * step_to_fine +
                 (1.0f - energy_lower_weight) * step_to_smooth);

        out_ds.x = depth_opt + depth_opt_step;
        out_ds.y = (1.0f - close_to_rough_weight) *
                   (energy_lower_weight * fine_sim_weight * ref_sim +
                    (1.0f - energy_lower_weight) * (depth_energy / 20.0f));
    }
    out_opt[idx] = out_ds;
}
