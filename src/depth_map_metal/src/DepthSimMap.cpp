#include "av/depth_map/DepthSimMap.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Texture.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace av::depth_map {

namespace {

// MSL-side mirror of `DepthSimMapCopyDepthOnlyParams` in
// `depth_sim_map.metal`. Layout must match byte-for-byte.
struct CopyDepthOnlyParamsGpu {
    std::uint32_t width;
    std::uint32_t height;
    float         default_sim;
};

// Mirror of `MapUpscaleFloat3Params` in `depth_sim_map.metal`.
struct MapUpscaleFloat3ParamsGpu {
    std::uint32_t out_width;
    std::uint32_t out_height;
    std::uint32_t in_width;
    std::uint32_t in_height;
    float         ratio;
};

// Mirror of `DepthThicknessSmoothParams` in `depth_sim_map.metal`.
struct DepthThicknessSmoothParamsGpu {
    std::uint32_t width;
    std::uint32_t height;
    float         min_thickness_inflate;
    float         max_thickness_inflate;
};

// Mirror of `ComputeUpscaledDepthPixSizeMapParams`.
struct ComputeUpscaledDepthPixSizeMapParamsGpu {
    std::uint32_t out_width;
    std::uint32_t out_height;
    std::uint32_t in_width;
    std::uint32_t in_height;
    std::uint32_t roi_x_begin;
    std::uint32_t roi_y_begin;
    std::uint32_t rc_level_width;
    std::uint32_t rc_level_height;
    float         rc_mipmap_level;
    std::int32_t  step_xy;
    std::int32_t  half_nb_depths;
    float         ratio;
};

// Mirror of `DepthSimMapComputeNormalParams` in `depth_sim_map.metal`.
struct ComputeNormalParamsGpu {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t roi_x_begin;
    std::uint32_t roi_y_begin;
    std::int32_t  step_xy;
    std::int32_t  wsh;
};

// Mirror of `OptimizeVarLParams` in `depth_sim_map.metal`.
struct OptimizeVarLParamsGpu {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t roi_x_begin;
    std::uint32_t roi_y_begin;
    std::uint32_t rc_level_width;
    std::uint32_t rc_level_height;
    float         rc_mipmap_level;
    std::int32_t  step_xy;
};

// Mirror of `OptimizeGetOptDepthParams` in `depth_sim_map.metal`.
struct OptimizeGetOptDepthParamsGpu {
    std::uint32_t width;
    std::uint32_t height;
};

// Mirror of `OptimizeDepthSimMapParams` in `depth_sim_map.metal`.
struct OptimizeDepthSimMapParamsGpu {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t roi_x_begin;
    std::uint32_t roi_y_begin;
    std::int32_t  iter;
};

}  // namespace

struct DepthSimMap::Impl {
    av::gpu::Device&  device;
    av::gpu::Pipeline copy_depth_only;
    av::gpu::Pipeline map_upscale_float3;
    av::gpu::Pipeline smooth_thickness;
    av::gpu::Pipeline upscale_pixsize_nearest;
    av::gpu::Pipeline upscale_pixsize_bilinear;
    av::gpu::Pipeline compute_normal;
    av::gpu::Pipeline optimize_var_l;
    av::gpu::Pipeline optimize_get_opt_depth;
    av::gpu::Pipeline optimize_depth_sim_map;

    Impl(av::gpu::Device& d,
         av::gpu::Pipeline cd,
         av::gpu::Pipeline mu3,
         av::gpu::Pipeline st,
         av::gpu::Pipeline up_n,
         av::gpu::Pipeline up_b,
         av::gpu::Pipeline cn,
         av::gpu::Pipeline ovl,
         av::gpu::Pipeline ogd,
         av::gpu::Pipeline ods)
        : device(d),
          copy_depth_only(std::move(cd)),
          map_upscale_float3(std::move(mu3)),
          smooth_thickness(std::move(st)),
          upscale_pixsize_nearest(std::move(up_n)),
          upscale_pixsize_bilinear(std::move(up_b)),
          compute_normal(std::move(cn)),
          optimize_var_l(std::move(ovl)),
          optimize_get_opt_depth(std::move(ogd)),
          optimize_depth_sim_map(std::move(ods)) {}
};

DepthSimMap::DepthSimMap(av::gpu::Device& dev)
    : impl_(std::make_unique<Impl>(
          dev,
          dev.make_pipeline("av_depth_sim_map_copy_depth_only"),
          dev.make_pipeline("av_map_upscale_float3"),
          dev.make_pipeline("av_depth_thickness_smooth_thickness"),
          dev.make_pipeline("av_compute_sgm_upscaled_depth_pix_size_map_nearest"),
          dev.make_pipeline("av_compute_sgm_upscaled_depth_pix_size_map_bilinear"),
          dev.make_pipeline("av_depth_sim_map_compute_normal"),
          dev.make_pipeline("av_optimize_var_l_of_lab_to_w"),
          dev.make_pipeline("av_optimize_get_opt_depth_map"),
          dev.make_pipeline("av_optimize_depth_sim_map")))
{}

DepthSimMap::DepthSimMap(DepthSimMap&&) noexcept            = default;
DepthSimMap& DepthSimMap::operator=(DepthSimMap&&) noexcept = default;
DepthSimMap::~DepthSimMap()                                  = default;

void DepthSimMap::copy_depth_only(av::gpu::Buffer&       out_map,
                                  const av::gpu::Buffer& in_map,
                                  std::uint32_t          width,
                                  std::uint32_t          height,
                                  float                  default_sim)
{
    using namespace av::gpu;

    const std::size_t need =
        std::size_t(width) * std::size_t(height) * 2 * sizeof(float);
    if (out_map.size_bytes() < need || in_map.size_bytes() < need) {
        throw std::invalid_argument(
            "DepthSimMap::copy_depth_only: buffer too small");
    }

    CopyDepthOnlyParamsGpu p{ width, height, default_sim };

    CommandBuffer cb(impl_->device);
    cb.set_label  ("depth_sim_map.copy_depth_only")
      .set_pipeline(impl_->copy_depth_only)
      .set_buffer  (0, out_map)
      .set_buffer  (1, in_map)
      .set_bytes   (2, &p, sizeof(p))
      .dispatch    ({ width, height, 1u }, { 16u, 16u, 1u });
    cb.commit_and_wait();
}

void DepthSimMap::normal_map_upscale(av::gpu::Buffer&       out_map,
                                     const av::gpu::Buffer& in_map,
                                     std::uint32_t          out_width,
                                     std::uint32_t          out_height,
                                     std::uint32_t          in_width,
                                     std::uint32_t          in_height)
{
    using namespace av::gpu;

    if (out_width == 0 || in_width == 0) {
        throw std::invalid_argument(
            "DepthSimMap::normal_map_upscale: zero width");
    }

    // packed_float3 = 3 floats per pixel.
    const std::size_t out_need =
        std::size_t(out_width) * std::size_t(out_height) * 3 * sizeof(float);
    const std::size_t in_need =
        std::size_t(in_width) * std::size_t(in_height) * 3 * sizeof(float);
    if (out_map.size_bytes() < out_need || in_map.size_bytes() < in_need) {
        throw std::invalid_argument(
            "DepthSimMap::normal_map_upscale: buffer too small");
    }

    MapUpscaleFloat3ParamsGpu p{
        out_width,
        out_height,
        in_width,
        in_height,
        static_cast<float>(in_width) / static_cast<float>(out_width)
    };

    CommandBuffer cb(impl_->device);
    cb.set_label  ("depth_sim_map.normal_map_upscale")
      .set_pipeline(impl_->map_upscale_float3)
      .set_buffer  (0, out_map)
      .set_buffer  (1, in_map)
      .set_bytes   (2, &p, sizeof(p))
      .dispatch    ({ out_width, out_height, 1u }, { 16u, 16u, 1u });
    cb.commit_and_wait();
}

void DepthSimMap::smooth_thickness(av::gpu::Buffer& inout_map,
                                    std::uint32_t   width,
                                    std::uint32_t   height,
                                    float           min_thickness_inflate,
                                    float           max_thickness_inflate)
{
    using namespace av::gpu;

    const std::size_t need =
        std::size_t(width) * std::size_t(height) * 2 * sizeof(float);
    if (inout_map.size_bytes() < need) {
        throw std::invalid_argument(
            "DepthSimMap::smooth_thickness: buffer too small");
    }

    DepthThicknessSmoothParamsGpu p{
        width, height, min_thickness_inflate, max_thickness_inflate };

    CommandBuffer cb(impl_->device);
    cb.set_label  ("depth_sim_map.smooth_thickness")
      .set_pipeline(impl_->smooth_thickness)
      .set_buffer  (0, inout_map)
      .set_bytes   (1, &p, sizeof(p))
      .dispatch    ({ width, height, 1u }, { 16u, 16u, 1u });
    cb.commit_and_wait();
}

void DepthSimMap::compute_sgm_upscaled_depth_pix_size_map(
    av::gpu::Buffer&        out_map,
    const av::gpu::Buffer&  in_map,
    const av::gpu::Texture& rc_mipmap,
    const ComputeUpscaledDepthPixSizeMapParams& cp)
{
    using namespace av::gpu;

    if (cp.out_width == 0 || cp.in_width == 0 || cp.half_nb_depths == 0) {
        throw std::invalid_argument(
            "DepthSimMap::compute_sgm_upscaled_depth_pix_size_map: bad params");
    }
    const std::size_t out_need =
        std::size_t(cp.out_width) * std::size_t(cp.out_height) * 2 * sizeof(float);
    const std::size_t in_need  =
        std::size_t(cp.in_width)  * std::size_t(cp.in_height)  * 2 * sizeof(float);
    if (out_map.size_bytes() < out_need || in_map.size_bytes() < in_need) {
        throw std::invalid_argument(
            "DepthSimMap::compute_sgm_upscaled_depth_pix_size_map: buffer too small");
    }

    ComputeUpscaledDepthPixSizeMapParamsGpu p{
        cp.out_width, cp.out_height,
        cp.in_width,  cp.in_height,
        cp.roi_x_begin, cp.roi_y_begin,
        cp.rc_level_width, cp.rc_level_height,
        cp.rc_mipmap_level,
        cp.step_xy, cp.half_nb_depths,
        static_cast<float>(cp.in_width) / static_cast<float>(cp.out_width)
    };

    CommandBuffer cb(impl_->device);
    cb.set_label  (cp.bilinear
                   ? "depth_sim_map.upscale_pixsize_bilinear"
                   : "depth_sim_map.upscale_pixsize_nearest")
      .set_pipeline(cp.bilinear ? impl_->upscale_pixsize_bilinear
                                : impl_->upscale_pixsize_nearest)
      .set_buffer  (0, out_map)
      .set_buffer  (1, in_map)
      .set_bytes   (2, &p, sizeof(p))
      .set_texture (0, rc_mipmap)
      .dispatch    ({ cp.out_width, cp.out_height, 1u }, { 16u, 16u, 1u });
    cb.commit_and_wait();
}

void DepthSimMap::optimize_depth_sim_map(
    av::gpu::Buffer&             out_opt,
    const av::gpu::Buffer&       in_sgm,
    const av::gpu::Buffer&       in_refine,
    const av::gpu::Texture&      rc_mipmap,
    av::gpu::Texture&            variance_tex,
    av::gpu::Texture&            tmp_depth_tex,
    const DeviceCameraParams&    rc_camera,
    const OptimizeGradientDescentParams& cp)
{
    using namespace av::gpu;

    if (cp.width == 0 || cp.height == 0) {
        throw std::invalid_argument(
            "DepthSimMap::optimize_depth_sim_map: zero dims");
    }
    const std::size_t pix     = std::size_t(cp.width) * std::size_t(cp.height);
    const std::size_t f2_need = pix * 2 * sizeof(float);
    if (out_opt   .size_bytes() < f2_need ||
        in_sgm    .size_bytes() < f2_need ||
        in_refine .size_bytes() < f2_need) {
        throw std::invalid_argument(
            "DepthSimMap::optimize_depth_sim_map: buffer too small");
    }

    // Seed out_opt with the SGM buffer. The kernel's iter==0 branch
    // overwrites the value, but `optimize_get_opt_depth_map` is the
    // first thing each iteration runs and it reads out_opt.x — for
    // iter=0, that read needs the SGM depth.
    std::memcpy(out_opt.data(), in_sgm.data(), f2_need);

    // ---- pass 1: variance map (run once) ----
    OptimizeVarLParamsGpu vlp{
        cp.width, cp.height,
        cp.roi_x_begin, cp.roi_y_begin,
        cp.rc_level_width, cp.rc_level_height,
        cp.rc_mipmap_level,
        cp.step_xy };
    {
        CommandBuffer cb(impl_->device);
        cb.set_label  ("depth_sim_map.optimize_var_l")
          .set_pipeline(impl_->optimize_var_l)
          .set_bytes   (0, &vlp, sizeof(vlp))
          .set_texture (0, variance_tex)
          .set_texture (1, rc_mipmap)
          .dispatch    ({ cp.width, cp.height, 1u }, { 16u, 16u, 1u });
        cb.commit_and_wait();
    }

    // ---- iterations ----
    OptimizeGetOptDepthParamsGpu gp{ cp.width, cp.height };
    for (int iter = 0; iter < cp.nb_iterations; ++iter) {
        // (a) copy out_opt.x → tmp_depth_tex
        {
            CommandBuffer cb(impl_->device);
            cb.set_label  ("depth_sim_map.optimize_get_opt_depth")
              .set_pipeline(impl_->optimize_get_opt_depth)
              .set_texture (0, tmp_depth_tex)
              .set_buffer  (0, out_opt)
              .set_bytes   (1, &gp, sizeof(gp))
              .dispatch    ({ cp.width, cp.height, 1u }, { 16u, 16u, 1u });
            cb.commit_and_wait();
        }
        // (b) optimize step
        OptimizeDepthSimMapParamsGpu op{
            cp.width, cp.height,
            cp.roi_x_begin, cp.roi_y_begin,
            iter };
        {
            CommandBuffer cb(impl_->device);
            cb.set_label  ("depth_sim_map.optimize_depth_sim_map")
              .set_pipeline(impl_->optimize_depth_sim_map)
              .set_buffer  (0, out_opt)
              .set_buffer  (1, in_sgm)
              .set_buffer  (2, in_refine)
              .set_bytes   (3, &op, sizeof(op))
              .set_bytes   (4, &rc_camera, sizeof(DeviceCameraParams))
              .set_texture (0, variance_tex)
              .set_texture (1, tmp_depth_tex)
              .dispatch    ({ cp.width, cp.height, 1u }, { 16u, 16u, 1u });
            cb.commit_and_wait();
        }
    }
}

void DepthSimMap::compute_normal(av::gpu::Buffer&             out_normal_map,
                                 const av::gpu::Buffer&       in_depth_sim_map,
                                 const DeviceCameraParams&    rc_camera,
                                 const ComputeNormalParams&   cp)
{
    using namespace av::gpu;

    if (cp.width == 0 || cp.height == 0) {
        throw std::invalid_argument(
            "DepthSimMap::compute_normal: zero dims");
    }
    const std::size_t pix = std::size_t(cp.width) * std::size_t(cp.height);
    if (out_normal_map .size_bytes() < pix * 3 * sizeof(float) ||
        in_depth_sim_map.size_bytes() < pix * 2 * sizeof(float)) {
        throw std::invalid_argument(
            "DepthSimMap::compute_normal: buffer too small");
    }

    ComputeNormalParamsGpu p{
        cp.width, cp.height, cp.roi_x_begin, cp.roi_y_begin,
        cp.step_xy, cp.wsh };

    CommandBuffer cb(impl_->device);
    cb.set_label  ("depth_sim_map.compute_normal")
      .set_pipeline(impl_->compute_normal)
      .set_buffer  (0, out_normal_map)
      .set_buffer  (1, in_depth_sim_map)
      .set_bytes   (2, &p, sizeof(p))
      .set_bytes   (3, &rc_camera, sizeof(DeviceCameraParams))
      .dispatch    ({ cp.width, cp.height, 1u }, { 16u, 16u, 1u });
    cb.commit_and_wait();
}

}  // namespace av::depth_map
