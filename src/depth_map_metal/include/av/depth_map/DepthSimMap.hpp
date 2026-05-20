#pragma once

// DepthSimMap — host driver for the depthSimMap post-processing
// kernels in upstream's `deviceDepthSimilarityMap.cu`.
//
// Buffer layout: depth-sim maps are float2 packed, row-major, no
// pitch. Linear index `y * width + x`.

#include "av/depth_map/PatchOps.hpp"   // DeviceCameraParams mirror

#include <cstdint>
#include <memory>

namespace av::gpu {
    class Device;
    class Buffer;
    class Texture;
}

namespace av::depth_map {

class DepthSimMap {
public:
    explicit DepthSimMap(av::gpu::Device& dev);

    DepthSimMap(const DepthSimMap&)            = delete;
    DepthSimMap& operator=(const DepthSimMap&) = delete;
    DepthSimMap(DepthSimMap&&) noexcept;
    DepthSimMap& operator=(DepthSimMap&&) noexcept;
    ~DepthSimMap();

    // Copy `depth` (the .x channel) from `in_map` to `out_map`,
    // overwriting the .y channel with `default_sim`. Both buffers
    // are float2 packed (= width*height*2 floats), row-major.
    //
    // Matches upstream's `cuda_depthSimMapCopyDepthOnly`.
    void copy_depth_only(av::gpu::Buffer&        out_map,
                         const av::gpu::Buffer&  in_map,
                         std::uint32_t           width,
                         std::uint32_t           height,
                         float                   default_sim);

    // Nearest-neighbor upscale a `packed_float3` (= 3 floats per
    // pixel) map. Typically used for normal-map upscale from the
    // SGM resolution to the Refine resolution.
    //
    // Buffer sizing:
    //   * `out_map` ≥ out_width * out_height * 3 floats.
    //   * `in_map`  ≥ in_width  * in_height  * 3 floats.
    //
    // Mirrors upstream's `cuda_normalMapUpscale` semantics:
    // computes `ratio = in_width / out_width` internally, then
    // nearest-neighbor maps via `(x - 0.5) * ratio` per output
    // pixel. The output dispatch covers `out_width × out_height`
    // pixels; values outside that range are not written.
    void normal_map_upscale(av::gpu::Buffer&        out_map,
                            const av::gpu::Buffer&  in_map,
                            std::uint32_t           out_width,
                            std::uint32_t           out_height,
                            std::uint32_t           in_width,
                            std::uint32_t           in_height);

    // Smooth the thickness channel (.y) of a (depth, thickness)
    // map by averaging clamped depth-distances over the 3×3
    // neighborhood. In-place — only `.y` is written.
    //
    // Pixels with `depth <= 0` are skipped (no write). The output
    // thickness is also preserved when fewer than 3 of the 8
    // neighbors have positive depth.
    //
    // `min_thickness_inflate` and `max_thickness_inflate` are
    // typically derived from SGM and Refine parameters:
    //   sgmScaleStep    = sgmParams.scale    * sgmParams.stepXY
    //   refineScaleStep = refineParams.scale * refineParams.stepXY
    //   max_nb_refine_samples = max(sgmScaleStep / refineScaleStep, 2)
    //   min_thickness_inflate = refineParams.halfNbDepths / max_nb_refine_samples
    //   max_thickness_inflate = refineParams.halfNbDepths / 2
    void smooth_thickness(av::gpu::Buffer&       inout_map,
                          std::uint32_t          width,
                          std::uint32_t          height,
                          float                  min_thickness_inflate,
                          float                  max_thickness_inflate);

    // Upscale SGM-resolution (depth, thickness) map to the Refine
    // resolution, writing (depth, pixSize) to `out_map` where
    // `pixSize = thickness / half_nb_depths`. The rc mipmap is
    // sampled at level `rc_mipmap_level` for an alpha-mask
    // pre-filter; masked pixels write (-2, 0).
    //
    // Output buffer is `out_width * out_height * 2` floats.
    // Input  buffer is `in_width  * in_height  * 2` floats.
    //
    // The compute-pixsize-via-camera path (upstream's
    // `ALICEVISION_DEPTHMAP_COMPUTE_PIXSIZEMAP` — commented out
    // upstream by default) is omitted; only the
    // thickness-derived pixSize is implemented.
    struct ComputeUpscaledDepthPixSizeMapParams {
        std::uint32_t out_width      = 0;   // refine resolution (dispatch dims)
        std::uint32_t out_height     = 0;
        std::uint32_t in_width       = 0;   // sgm resolution
        std::uint32_t in_height      = 0;
        std::uint32_t roi_x_begin    = 0;   // image-space anchor
        std::uint32_t roi_y_begin    = 0;
        std::uint32_t rc_level_width  = 0;  // mipmap-level dims
        std::uint32_t rc_level_height = 0;
        float         rc_mipmap_level = 0.0f;
        std::int32_t  step_xy        = 1;
        std::int32_t  half_nb_depths = 1;
        bool          bilinear       = false;
    };

    void compute_sgm_upscaled_depth_pix_size_map(
        av::gpu::Buffer&        out_map,
        const av::gpu::Buffer&  in_map,
        const av::gpu::Texture& rc_mipmap,
        const ComputeUpscaledDepthPixSizeMapParams& params);

    // Compute a packed_float3 normal map from a (depth, sim) map.
    // Only the depth (.x) channel is used. For each output pixel:
    //   * If center depth ≤ 0, write (-1, -1, -1).
    //   * Otherwise fit a plane via PCA on the valid 3D points in a
    //     (2*wsh + 1)² neighborhood (gated by |Δdepth| < 30 * pixSize),
    //     return the unit normal oriented toward the camera.
    //
    // Buffer sizing:
    //   * `out_normal_map` ≥ width * height * 3 floats (packed_float3).
    //   * `in_depth_sim_map` ≥ width * height * 2 floats.
    //
    // Note: this is the FP32-accumulator equivalent of upstream's
    // FP64-accumulator `cuda_stat3d`. Apple Silicon GPUs have no
    // FP64; the precision floor is FP32 throughout.
    //
    // Upstream hardcodes wsh=3 (`depthSimMapComputeNormal_kernel<3>`)
    // — preserving that as the only currently-supported value.
    struct ComputeNormalParams {
        std::uint32_t width        = 0;   // dispatch dim = output ROI width
        std::uint32_t height       = 0;
        std::uint32_t roi_x_begin  = 0;   // image-space anchor
        std::uint32_t roi_y_begin  = 0;
        std::int32_t  step_xy      = 1;
        std::int32_t  wsh          = 3;
    };

    void compute_normal(av::gpu::Buffer&             out_normal_map,
                        const av::gpu::Buffer&       in_depth_sim_map,
                        const DeviceCameraParams&    rc_camera,
                        const ComputeNormalParams&   params);

    // Gradient-descent fusion of SGM and Refine depth maps. The
    // output buffer holds the fused (depth, sim) map. Two scratch
    // R32Float textures must be passed:
    //   * `variance_tex` — sized to (width, height); written once
    //     by the var-of-L-of-LAB pre-pass; read each iteration.
    //   * `tmp_depth_tex` — sized to (width, height); rewritten
    //     each iteration with the current out_opt.x channel.
    //
    // Buffer sizing:
    //   * `out_opt_depth_sim_map` ≥ width*height*2 floats.
    //   * `in_sgm_depth_pix_size_map` ≥ width*height*2 floats.
    //   * `in_refine_depth_sim_map`   ≥ width*height*2 floats.
    //
    // Mirrors upstream's `cuda_depthSimMapOptimizeGradientDescent`.
    // The kernel uses the rc mipmap texture at level
    // `rc_mipmap_level` for the variance map (via 4 neighbor
    // samples of the L channel).
    struct OptimizeGradientDescentParams {
        std::uint32_t width        = 0;
        std::uint32_t height       = 0;
        std::uint32_t roi_x_begin  = 0;
        std::uint32_t roi_y_begin  = 0;
        std::uint32_t rc_level_width  = 0;
        std::uint32_t rc_level_height = 0;
        float         rc_mipmap_level = 0.0f;
        std::int32_t  step_xy            = 1;
        std::int32_t  nb_iterations      = 100;
    };

    void optimize_depth_sim_map(
        av::gpu::Buffer&             out_opt_depth_sim_map,
        const av::gpu::Buffer&       in_sgm_depth_pix_size_map,
        const av::gpu::Buffer&       in_refine_depth_sim_map,
        const av::gpu::Texture&      rc_mipmap,
        av::gpu::Texture&            variance_tex,
        av::gpu::Texture&            tmp_depth_tex,
        const DeviceCameraParams&    rc_camera,
        const OptimizeGradientDescentParams& params);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
