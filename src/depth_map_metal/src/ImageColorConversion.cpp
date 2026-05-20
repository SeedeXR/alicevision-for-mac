#include "av/depth_map/ImageColorConversion.hpp"

#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Texture.hpp"

namespace av::depth_map {

struct ImageColorConversion::Impl {
    av::gpu::Device&  device;
    av::gpu::Pipeline pipeline;
    Impl(av::gpu::Device& d, av::gpu::Pipeline p) noexcept
        : device(d), pipeline(std::move(p)) {}
};

ImageColorConversion::ImageColorConversion(av::gpu::Device& dev)
    : impl_(std::make_unique<Impl>(dev,
                                   dev.make_pipeline("av_rgb2lab")))
{}

ImageColorConversion::ImageColorConversion(ImageColorConversion&&) noexcept            = default;
ImageColorConversion& ImageColorConversion::operator=(ImageColorConversion&&) noexcept = default;
ImageColorConversion::~ImageColorConversion()                                          = default;

void ImageColorConversion::rgb2lab(av::gpu::Texture& tex)
{
    using namespace av::gpu;

    // Threadgroup shape mirrors upstream: 32 across × 2 down.
    // On Apple GPUs 32 along x maps to one SIMD group; the y count
    // is tunable. We keep upstream's choice as the starting point.
    constexpr std::uint32_t kTgX = 32;
    constexpr std::uint32_t kTgY = 2;

    CommandBuffer cb(impl_->device);
    cb.set_label("rgb2lab")
      .set_pipeline(impl_->pipeline)
      .set_texture (0, tex)
      .dispatch({ tex.width(), tex.height(), 1u },
                { kTgX, kTgY, 1u });
    cb.commit_and_wait();
}

}  // namespace av::depth_map
