#include "av/depth_map/ColorOps.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Pipeline.hpp"

#include <stdexcept>

namespace av::depth_map {

struct ColorOps::Impl {
    av::gpu::Device&  device;
    av::gpu::Pipeline pipeline;
    Impl(av::gpu::Device& d, av::gpu::Pipeline p) noexcept
        : device(d), pipeline(std::move(p)) {}
};

ColorOps::ColorOps(av::gpu::Device& dev)
    : impl_(std::make_unique<Impl>(dev,
                                   dev.make_pipeline("av_color_validate")))
{}

ColorOps::ColorOps(ColorOps&&) noexcept            = default;
ColorOps& ColorOps::operator=(ColorOps&&) noexcept = default;
ColorOps::~ColorOps()                              = default;

void ColorOps::validate(std::span<const float> inputs,
                        std::span<float>        outputs)
{
    if (inputs.size() % kInPerCase != 0)
        throw std::invalid_argument("ColorOps: inputs size mismatch");
    const std::size_t count = inputs.size() / kInPerCase;
    if (outputs.size() != count * kOutPerCase)
        throw std::invalid_argument("ColorOps: outputs size mismatch");

    using namespace av::gpu;
    auto& dev = impl_->device;

    Buffer in_buf (dev, inputs.size_bytes());
    Buffer out_buf(dev, outputs.size_bytes());
    in_buf .set_label("color.in");
    out_buf.set_label("color.out");
    in_buf.upload(inputs);

    const std::uint32_t count_u32 = static_cast<std::uint32_t>(count);

    CommandBuffer cb(dev);
    cb.set_pipeline(impl_->pipeline)
      .set_buffer (0, in_buf)
      .set_buffer (1, out_buf)
      .set_bytes  (2, &count_u32, sizeof(count_u32))
      .dispatch_1d(impl_->pipeline, count_u32);
    cb.commit_and_wait();

    auto src = out_buf.as_span<const float>();
    std::copy(src.begin(), src.end(), outputs.begin());
}

}  // namespace av::depth_map
