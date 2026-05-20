#include "av/depth_map/GaussianFilter.hpp"
#include "av/depth_map/GaussianTable.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Texture.hpp"

#include <cstring>
#include <stdexcept>

namespace av::depth_map {

namespace {

// Mirror the MSL `DownscaleParams` struct layout exactly.
struct DownscaleParams {
    std::uint32_t downscaledWidth;
    std::uint32_t downscaledHeight;
    std::int32_t  downscale;
    std::int32_t  gaussRadius;
    std::uint32_t inputWidth;
    std::uint32_t inputHeight;
};

struct MedianParams {
    std::uint32_t width;
    std::uint32_t height;
};

// Mirror MSL's VolumeBlurParams.
struct VolumeBlurParams {
    std::uint32_t volDimX;
    std::uint32_t volDimY;
    std::uint32_t volDimZ;
    std::int32_t  gaussRadius;
};

}  // namespace

struct GaussianFilter::Impl {
    av::gpu::Device&    device;
    av::gpu::Pipeline   pipeline_downscale;
    av::gpu::Pipeline   pipeline_median3;
    av::gpu::Pipeline   pipeline_blur_z;
    av::gpu::Pipeline   pipeline_blur_xyz;
    GaussianTable&      table;

    Impl(av::gpu::Device& d,
         av::gpu::Pipeline pd,
         av::gpu::Pipeline pm,
         av::gpu::Pipeline pz,
         av::gpu::Pipeline pxyz,
         GaussianTable& t) noexcept
        : device(d),
          pipeline_downscale(std::move(pd)),
          pipeline_median3(std::move(pm)),
          pipeline_blur_z(std::move(pz)),
          pipeline_blur_xyz(std::move(pxyz)),
          table(t) {}
};

GaussianFilter::GaussianFilter(av::gpu::Device& dev, GaussianTable& table)
    : impl_(std::make_unique<Impl>(
          dev,
          dev.make_pipeline("av_downscale_with_gaussian_blur"),
          dev.make_pipeline("av_median_filter_3"),
          dev.make_pipeline("av_gaussian_blur_volume_z"),
          dev.make_pipeline("av_gaussian_blur_volume_xyz"),
          table))
{}

GaussianFilter::GaussianFilter(GaussianFilter&&) noexcept            = default;
GaussianFilter& GaussianFilter::operator=(GaussianFilter&&) noexcept = default;
GaussianFilter::~GaussianFilter()                                    = default;

void GaussianFilter::downscale_with_gaussian_blur(
    const av::gpu::Texture& in_tex,
    av::gpu::Buffer&        out_buf,
    std::uint32_t           downscaled_w,
    std::uint32_t           downscaled_h,
    std::int32_t            downscale,
    std::int32_t            gauss_radius)
{
    if (downscale <= 0 || gauss_radius <= 0)
        throw std::invalid_argument(
            "downscale_with_gaussian_blur: downscale and radius must be > 0");
    if (gauss_radius > downscale)
        throw std::invalid_argument(
            "downscale_with_gaussian_blur: gauss_radius must be <= downscale "
            "(Gaussian LUT scale = downscale - 1)");

    const std::size_t expected_bytes = std::size_t(downscaled_w)
                                     * std::size_t(downscaled_h)
                                     * 4 * sizeof(float);
    if (out_buf.size_bytes() < expected_bytes)
        throw std::invalid_argument(
            "downscale_with_gaussian_blur: out_buf too small");

    using namespace av::gpu;
    DownscaleParams params{};
    params.downscaledWidth  = downscaled_w;
    params.downscaledHeight = downscaled_h;
    params.downscale        = downscale;
    params.gaussRadius      = gauss_radius;
    params.inputWidth       = in_tex.width();
    params.inputHeight      = in_tex.height();

    CommandBuffer cb(impl_->device);
    cb.set_label  ("gaussian.downscale")
      .set_pipeline(impl_->pipeline_downscale)
      .set_texture (0, in_tex)
      .set_buffer  (0, out_buf)
      .set_buffer  (1, impl_->table.weights())
      .set_buffer  (2, impl_->table.offsets())
      .set_bytes   (3, &params, sizeof(params))
      .dispatch({ downscaled_w, downscaled_h, 1u },
                { 32u, 2u, 1u });
    cb.commit_and_wait();
}

namespace {

void run_volume_blur(av::gpu::Device&          device,
                     av::gpu::Pipeline&        pipeline,
                     GaussianTable&            table,
                     av::gpu::Buffer&          inout_volume,
                     VolumeDims                dims,
                     std::int32_t              gauss_radius,
                     const char*               label)
{
    using namespace av::gpu;

    if (gauss_radius < 1 || gauss_radius > GaussianTable::kMaxScales)
        throw std::invalid_argument(
            "gaussian_blur_volume: gauss_radius out of supported range");

    const std::size_t need_bytes = dims.voxel_count() * sizeof(float);
    if (inout_volume.size_bytes() < need_bytes)
        throw std::invalid_argument(
            "gaussian_blur_volume: inout_volume too small");

    // Upstream pattern: write into scratch, then copy back into the
    // input (the kernel needs un-aliased reads). Use a temporary
    // device buffer for the scratch.
    Buffer scratch(device, need_bytes);
    scratch.set_label("gaussian.blur_volume.scratch");

    VolumeBlurParams params{
        dims.x, dims.y, dims.z, gauss_radius,
    };

    CommandBuffer cb(device);
    cb.set_label  (label)
      .set_pipeline(pipeline)
      .set_buffer  (0, scratch)
      .set_buffer  (1, inout_volume)
      .set_buffer  (2, table.weights())
      .set_buffer  (3, table.offsets())
      .set_bytes   (4, &params, sizeof(params))
      .dispatch({ dims.x, dims.y, dims.z }, { 32u, 1u, 1u });
    cb.commit_and_wait();

    // Copy scratch → inout_volume on the host side (Shared storage:
    // both are CPU-visible on UMA). The upstream `copyFrom` uses
    // cudaMemcpy3D; for our packed layout a single memcpy is exact.
    std::memcpy(inout_volume.data(), scratch.data(), need_bytes);
}

}  // namespace

void GaussianFilter::gaussian_blur_volume_z(av::gpu::Buffer& inout_volume,
                                            VolumeDims       dims,
                                            std::int32_t     gauss_radius)
{
    run_volume_blur(impl_->device, impl_->pipeline_blur_z, impl_->table,
                    inout_volume, dims, gauss_radius,
                    "gaussian.blur_volume_z");
}

void GaussianFilter::gaussian_blur_volume_xyz(av::gpu::Buffer& inout_volume,
                                              VolumeDims       dims,
                                              std::int32_t     gauss_radius)
{
    run_volume_blur(impl_->device, impl_->pipeline_blur_xyz, impl_->table,
                    inout_volume, dims, gauss_radius,
                    "gaussian.blur_volume_xyz");
}

void GaussianFilter::median_filter_3(
    const av::gpu::Texture& in_tex,
    av::gpu::Buffer&        out_buf,
    std::uint32_t           width,
    std::uint32_t           height)
{
    const std::size_t expected_bytes = std::size_t(width)
                                     * std::size_t(height)
                                     * sizeof(float);
    if (out_buf.size_bytes() < expected_bytes)
        throw std::invalid_argument(
            "median_filter_3: out_buf too small");

    using namespace av::gpu;
    MedianParams params{ width, height };

    CommandBuffer cb(impl_->device);
    cb.set_label  ("gaussian.median3")
      .set_pipeline(impl_->pipeline_median3)
      .set_texture (0, in_tex)
      .set_buffer  (0, out_buf)
      .set_bytes   (1, &params, sizeof(params))
      .dispatch({ width, height, 1u },
                { 32u, 2u, 1u });
    cb.commit_and_wait();
}

}  // namespace av::depth_map
