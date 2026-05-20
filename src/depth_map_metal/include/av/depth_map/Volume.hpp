#pragma once

// Volume — host driver for the cost-volume housekeeping kernels
// (Phase 7 opener):
//
//   init_sim(volume, dims, value)
//   init_refine(volume, dims, value_float)   // host-side float → half conversion
//   add_refine(out, in, dims)                // FP16 promote-add-demote
//   update_uninitialized(volume2nd, volume1st, dims)
//
// Element types:
//   TSim       = uchar  (8-bit cost volume — matches upstream)
//   TSimRefine = half   (FP16 refinement volume — matches upstream)
//
// Buffer layout: packed (no row padding), z-major-of-y-major-of-x.
// Linear index: `z * (X * Y) + y * X + x`.

#include "av/depth_map/PatchOps.hpp"   // DeviceCameraParams mirror

#include <cstddef>
#include <cstdint>
#include <memory>

namespace av::gpu {
    class Device;
    class Buffer;
    class Texture;
}

namespace av::depth_map {

struct VolumeDims {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
    std::size_t voxel_count() const noexcept {
        return std::size_t(x) * std::size_t(y) * std::size_t(z);
    }
};

class Volume {
public:
    explicit Volume(av::gpu::Device& dev);

    Volume(const Volume&)            = delete;
    Volume& operator=(const Volume&) = delete;
    Volume(Volume&&) noexcept;
    Volume& operator=(Volume&&) noexcept;
    ~Volume();

    // Initialize an 8-bit cost volume to a single value.
    void init_sim(av::gpu::Buffer&  volume,
                  VolumeDims        dims,
                  std::uint8_t      value);

    // Initialize a half-precision refinement volume to a single
    // value. The host converts `value_float` to half (IEEE binary16)
    // and packs the bit pattern into the param buffer.
    void init_refine(av::gpu::Buffer& volume,
                     VolumeDims       dims,
                     float            value_float);

    // In-place `inout += in` on a half-precision volume. Both
    // buffers must be sized for `dims.voxel_count()` half elements.
    void add_refine(av::gpu::Buffer&       inout,
                    const av::gpu::Buffer& in,
                    VolumeDims             dims);

    // Where `volume2nd[i] == 255`, copy `volume1st[i]` into
    // `volume2nd[i]`. Both volumes are 8-bit, same dims.
    void update_uninitialized(av::gpu::Buffer&       volume2nd,
                              const av::gpu::Buffer& volume1st,
                              VolumeDims             dims);

    // ----------- SGM exit point: retrieve_best_depth ------------

    struct RetrieveBestDepthParams {
        VolumeDims    dims;
        std::uint32_t depth_range_begin     = 0;
        std::uint32_t depth_range_end       = 0;   // exclusive; defaults to dims.z if 0
        std::uint32_t roi_x_begin           = 0;
        std::uint32_t roi_y_begin           = 0;
        std::int32_t  scale_step            = 1;
        float         thickness_mult_factor = 1.0f;
        float         max_similarity        = 200.0f;   // uchar scale (0..254 valid; 255 sentinel)
    };

    // Per-pixel WTA + sub-pixel refinement on a filled uchar cost
    // volume. Writes `(depth, thickness)` to `out_depth_thickness`
    // (float2 packed, X*Y entries) and `(depth, sim ∈ [-1, 1])` to
    // `out_depth_sim` (float2 packed, X*Y entries).
    //
    //   * On invalid / over-threshold pixels, depth = -1.
    //   * `in_depths` is a flat float buffer of length `dims.z`
    //     holding the depth-plane values used during the sweep.
    //   * `rc_camera` is the reference-camera params (same struct
    //     used by Patch.h / PatchOps.hpp).
    void retrieve_best_depth(av::gpu::Buffer&             out_depth_thickness,
                             av::gpu::Buffer&             out_depth_sim,
                             const av::gpu::Buffer&       in_depths,
                             const av::gpu::Buffer&       volume,
                             const DeviceCameraParams&    rc_camera,
                             const RetrieveBestDepthParams& params);

    // ----------- compute_similarity (heaviest kernel) -----------

    struct ComputeSimilarityParams {
        VolumeDims    dims;                  // volDimX/Y/Z
        std::uint32_t rc_sgm_level_width  = 0;
        std::uint32_t rc_sgm_level_height = 0;
        std::uint32_t tc_sgm_level_width  = 0;
        std::uint32_t tc_sgm_level_height = 0;
        float         rc_mipmap_level     = 0.0f;
        std::int32_t  step_xy             = 1;
        std::int32_t  wsh                 = 4;     // NCC patch half-width
        float         inv_gamma_c         = 1.0f / 20.0f;
        float         inv_gamma_p         = 1.0f / 4.0f;
        std::uint32_t use_consistent_scale = 0;
        std::uint32_t depth_range_begin    = 0;
        std::uint32_t depth_range_end      = 0;   // exclusive; 0 → dims.z
        std::uint32_t roi_x_begin          = 0;
        std::uint32_t roi_y_begin          = 0;
        std::uint32_t roi_width            = 0;   // 0 → dims.x
        std::uint32_t roi_height           = 0;   // 0 → dims.y
    };

    // Per-voxel NCC for one (R, T) camera pair. Both output volumes
    // (best + 2nd-best) must be pre-initialized (typically to 255).
    // Re-invoke per T camera to aggregate.
    void compute_similarity(av::gpu::Buffer&             out_volume_1st,
                            av::gpu::Buffer&             out_volume_2nd,
                            const av::gpu::Buffer&       in_depths,
                            const av::gpu::Texture&      rc_mipmap,
                            const av::gpu::Texture&      tc_mipmap,
                            const DeviceCameraParams&    rc_camera,
                            const DeviceCameraParams&    tc_camera,
                            const ComputeSimilarityParams& params);

    // ----------- optimize: SGM 4-direction DP aggregation -------

    struct OptimizeParams {
        VolumeDims    dims;
        // Optional override; defaults to dims.z (== upstream's lastDepthIndex).
        std::uint32_t last_depth_index = 0;
        // SGM penalties.
        //   * `p1` matches upstream's `sgmParams.p1`.
        //   * `p2_abs` is the fixed P2 penalty (upstream's
        //     `abs(_P2)` when `_P2 < 0`).
        //   * `adaptive_p2 = true` activates the texture-driven
        //     adaptive-P2 path (upstream's `_P2 >= 0` branch). In
        //     that mode `p2_sig_mid` is fed straight into the
        //     sigmoid as `sigMid` (signed, NOT abs'd, matching
        //     upstream verbatim). When adaptive-P2 is enabled the
        //     caller MUST also supply `rc_mipmap`, `step_xy`,
        //     `roi_x_begin`, `roi_y_begin`, `rc_level_width`,
        //     `rc_level_height`, and `rc_mipmap_level`.
        float         p1               = 10.0f;
        float         p2_abs           = 100.0f;
        bool          adaptive_p2      = false;
        float         p2_sig_mid       = 100.0f;
        std::int32_t  step_xy          = 1;
        std::int32_t  roi_x_begin      = 0;
        std::int32_t  roi_y_begin      = 0;
        std::uint32_t rc_level_width   = 0;
        std::uint32_t rc_level_height  = 0;
        float         rc_mipmap_level  = 0.0f;
    };

    // Compute the SGM-aggregated similarity volume from
    // `in_volume`. Writes the aggregate result into `out_volume`.
    // The three buffers `slice_a`, `slice_b`, `axis_acc` are
    // scratch / "scratchpad" memory used internally and must be
    // sized at least:
    //   slice_a, slice_b: max(dims.x, dims.y) * dims.z * sizeof(uint)
    //   axis_acc:        max(dims.x, dims.y) * sizeof(uint)
    //
    // `out_volume` must be pre-allocated (size = dims.voxel_count()
    // bytes). Its initial content is irrelevant — `optimize` writes
    // every voxel.
    //
    // `rc_mipmap` is required only when `params.adaptive_p2` is
    // true; otherwise pass `nullptr`. The texture is the same
    // pre-Lab rc mipmap used by `compute_similarity`.
    void optimize(av::gpu::Buffer&        out_volume,
                  av::gpu::Buffer&        slice_a,
                  av::gpu::Buffer&        slice_b,
                  av::gpu::Buffer&        axis_acc,
                  const av::gpu::Buffer&  in_volume,
                  const OptimizeParams&   params,
                  const av::gpu::Texture* rc_mipmap = nullptr);

    // ----------- refine_similarity (FP16 Refine pass) -----------

    struct RefineSimilarityParams {
        VolumeDims    dims;
        std::uint32_t rc_refine_level_width  = 0;
        std::uint32_t rc_refine_level_height = 0;
        std::uint32_t tc_refine_level_width  = 0;
        std::uint32_t tc_refine_level_height = 0;
        float         rc_mipmap_level        = 0.0f;
        std::int32_t  step_xy                = 1;
        std::int32_t  wsh                    = 3;
        float         inv_gamma_c            = 1.0f / 20.0f;
        float         inv_gamma_p            = 1.0f / 4.0f;
        std::uint32_t use_consistent_scale   = 0;
        std::uint32_t depth_range_begin      = 0;
        std::uint32_t depth_range_end        = 0;   // exclusive; 0 → dims.z
        std::uint32_t roi_x_begin            = 0;
        std::uint32_t roi_y_begin            = 0;
        std::uint32_t roi_width              = 0;   // 0 → dims.x
        std::uint32_t roi_height             = 0;   // 0 → dims.y
    };

    // Per-voxel NCC<true> (sigmoid invert-and-filter) for one
    // (R, T) camera pair, around the per-pixel SGM mid depth.
    // The Z slice corresponds to a sub-pixel offset along the R
    // ray of `(vz - center_z) * sgm_pix_size`.
    // Half-volume accumulates via promote-add-demote; INFINITY
    // similarities leave the slot unchanged.
    //
    // `inout_vol_sim_half` is the FP16 refinement volume (size
    // dims.voxel_count() * 2 bytes). `in_sgm_depth_pix_size_map`
    // holds (sgm_depth, sgm_pix_size) per (vx, vy) — sized
    // dims.x * dims.y * 2 floats.
    void refine_similarity(av::gpu::Buffer&             inout_vol_sim_half,
                           const av::gpu::Buffer&       in_sgm_depth_pix_size_map,
                           const av::gpu::Texture&      rc_mipmap,
                           const av::gpu::Texture&      tc_mipmap,
                           const DeviceCameraParams&    rc_camera,
                           const DeviceCameraParams&    tc_camera,
                           const RefineSimilarityParams& params);

    // ----------- refine_best_depth (Refine exit) -----------

    struct RefineBestDepthParams {
        VolumeDims    dims;
        // Number of sub-samples between two depth planes. The full
        // sub-sample sweep covers [-halfNbSamples, +halfNbSamples].
        std::int32_t  samples_per_pix_size    = 4;
        std::int32_t  half_nb_samples         = 12;
        // Should typically equal (dims.z - 1) / 2.
        std::int32_t  half_nb_depths          = 0;
        // = 2 * σ². Picks the Gaussian kernel width.
        float         two_times_sigma_pow_two = 0.0f;
        std::uint32_t roi_width               = 0;   // 0 → dims.x
        std::uint32_t roi_height              = 0;   // 0 → dims.y
    };

    // Per-pixel: sweep sub-sample offsets around the SGM mid depth,
    // convolve with a Gaussian over the FP16 refinement volume's Z
    // axis, find the sub-sample with the strongest signal. Output
    // is `(best_depth, best_sample_sim)` per pixel — the sim value
    // is negative (the kernel flips the inverted convention back).
    //
    // `out_refine_depth_sim_map`: dims.x * dims.y * 2 floats.
    // `in_sgm_depth_pix_size_map`: dims.x * dims.y * 2 floats.
    // `in_vol_sim_half`: dims.voxel_count() half16 entries.
    void refine_best_depth(av::gpu::Buffer&             out_refine_depth_sim_map,
                           const av::gpu::Buffer&       in_sgm_depth_pix_size_map,
                           const av::gpu::Buffer&       in_vol_sim_half,
                           const RefineBestDepthParams& params);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
