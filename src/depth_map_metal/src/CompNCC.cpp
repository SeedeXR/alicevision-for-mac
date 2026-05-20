#include "av/depth_map/CompNCC.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Texture.hpp"

#include <stdexcept>

namespace av::depth_map {

struct CompNCC::Impl {
    av::gpu::Device&    device;
    av::gpu::Pipeline   pipeline_no_filter;
    av::gpu::Pipeline   pipeline_filter;
    av::gpu::Pipeline   pipeline_custom_pattern_no_filter;
    av::gpu::Pipeline   pipeline_custom_pattern_filter;
    DeviceCameraParams  rc;
    DeviceCameraParams  tc;

    Impl(av::gpu::Device& d,
         av::gpu::Pipeline pn,  av::gpu::Pipeline pf,
         av::gpu::Pipeline pcn, av::gpu::Pipeline pcf,
         const DeviceCameraParams& r, const DeviceCameraParams& t) noexcept
        : device(d),
          pipeline_no_filter(std::move(pn)),
          pipeline_filter(std::move(pf)),
          pipeline_custom_pattern_no_filter(std::move(pcn)),
          pipeline_custom_pattern_filter(std::move(pcf)),
          rc(r), tc(t) {}
};

CompNCC::CompNCC(av::gpu::Device& dev,
                 const DeviceCameraParams& rc,
                 const DeviceCameraParams& tc)
    : impl_(std::make_unique<Impl>(
          dev,
          dev.make_pipeline("av_compNCC_validate_no_filter"),
          dev.make_pipeline("av_compNCC_validate_filter"),
          dev.make_pipeline("av_compNCC_customPattern_no_filter"),
          dev.make_pipeline("av_compNCC_customPattern_filter"),
          rc, tc))
{}

CompNCC::CompNCC(CompNCC&&) noexcept            = default;
CompNCC& CompNCC::operator=(CompNCC&&) noexcept = default;
CompNCC::~CompNCC()                              = default;

namespace {

void run_impl(av::gpu::Device& dev,
              av::gpu::Pipeline& pipe,
              const DeviceCameraParams& rc,
              const DeviceCameraParams& tc,
              std::span<const PatchCase> patches,
              std::span<float>           out,
              const av::gpu::Texture&    rc_mipmap,
              const av::gpu::Texture&    tc_mipmap,
              const CompNCCParams&       params)
{
    if (out.size() != patches.size())
        throw std::invalid_argument(
            "CompNCC: out.size() must equal patches.size()");

    using namespace av::gpu;

    Buffer patch_buf(dev, patches.size_bytes());
    Buffer out_buf  (dev, out.size_bytes());
    patch_buf.set_label("compncc.patches");
    out_buf  .set_label("compncc.out");
    patch_buf.upload(patches);

    const std::uint32_t count_u32 =
        static_cast<std::uint32_t>(patches.size());

    CommandBuffer cb(dev);
    cb.set_pipeline(pipe)
      .set_bytes  (0, &rc,     sizeof(DeviceCameraParams))
      .set_bytes  (1, &tc,     sizeof(DeviceCameraParams))
      .set_buffer (2, patch_buf)
      .set_buffer (3, out_buf)
      .set_bytes  (4, &params, sizeof(CompNCCParams))
      .set_bytes  (5, &count_u32, sizeof(count_u32))
      .set_texture(0, rc_mipmap)
      .set_texture(1, tc_mipmap)
      .dispatch_1d(pipe, count_u32);
    cb.commit_and_wait();

    auto src = out_buf.as_span<const float>();
    std::copy(src.begin(), src.end(), out.begin());
}

}  // namespace

void CompNCC::run_no_filter(std::span<const PatchCase> patches,
                            std::span<float>           out,
                            const av::gpu::Texture&    rc_mipmap,
                            const av::gpu::Texture&    tc_mipmap,
                            const CompNCCParams&       params)
{
    run_impl(impl_->device, impl_->pipeline_no_filter,
             impl_->rc, impl_->tc, patches, out,
             rc_mipmap, tc_mipmap, params);
}

void CompNCC::run_filter(std::span<const PatchCase> patches,
                         std::span<float>           out,
                         const av::gpu::Texture&    rc_mipmap,
                         const av::gpu::Texture&    tc_mipmap,
                         const CompNCCParams&       params)
{
    run_impl(impl_->device, impl_->pipeline_filter,
             impl_->rc, impl_->tc, patches, out,
             rc_mipmap, tc_mipmap, params);
}

namespace {

void run_custom_impl(av::gpu::Device& dev,
                     av::gpu::Pipeline& pipe,
                     const DeviceCameraParams& rc,
                     const DeviceCameraParams& tc,
                     std::span<const PatchCase> patches,
                     std::span<float>           out,
                     const av::gpu::Texture&    rc_mipmap,
                     const av::gpu::Texture&    tc_mipmap,
                     const CompNCCParams&       params,
                     const DevicePatchPattern&  pattern)
{
    if (out.size() != patches.size())
        throw std::invalid_argument(
            "CompNCC: out.size() must equal patches.size()");

    using namespace av::gpu;

    Buffer patch_buf(dev, patches.size_bytes());
    Buffer out_buf  (dev, out.size_bytes());
    patch_buf.set_label("compncc.patches");
    out_buf  .set_label("compncc.out");
    patch_buf.upload(patches);

    const std::uint32_t count_u32 =
        static_cast<std::uint32_t>(patches.size());

    CommandBuffer cb(dev);
    cb.set_pipeline(pipe)
      .set_bytes  (0, &rc,      sizeof(DeviceCameraParams))
      .set_bytes  (1, &tc,      sizeof(DeviceCameraParams))
      .set_buffer (2, patch_buf)
      .set_buffer (3, out_buf)
      .set_bytes  (4, &params,  sizeof(CompNCCParams))
      .set_bytes  (5, &count_u32, sizeof(count_u32))
      .set_bytes  (6, &pattern, sizeof(DevicePatchPattern))
      .set_texture(0, rc_mipmap)
      .set_texture(1, tc_mipmap)
      .dispatch_1d(pipe, count_u32);
    cb.commit_and_wait();

    auto src = out_buf.as_span<const float>();
    std::copy(src.begin(), src.end(), out.begin());
}

}  // namespace

void CompNCC::run_no_filter_custom_pattern(
    std::span<const PatchCase> patches,
    std::span<float>           out,
    const av::gpu::Texture&    rc_mipmap,
    const av::gpu::Texture&    tc_mipmap,
    const CompNCCParams&       params,
    const DevicePatchPattern&  pattern)
{
    run_custom_impl(impl_->device, impl_->pipeline_custom_pattern_no_filter,
                    impl_->rc, impl_->tc, patches, out,
                    rc_mipmap, tc_mipmap, params, pattern);
}

void CompNCC::run_filter_custom_pattern(
    std::span<const PatchCase> patches,
    std::span<float>           out,
    const av::gpu::Texture&    rc_mipmap,
    const av::gpu::Texture&    tc_mipmap,
    const CompNCCParams&       params,
    const DevicePatchPattern&  pattern)
{
    run_custom_impl(impl_->device, impl_->pipeline_custom_pattern_filter,
                    impl_->rc, impl_->tc, patches, out,
                    rc_mipmap, tc_mipmap, params, pattern);
}

}  // namespace av::depth_map
