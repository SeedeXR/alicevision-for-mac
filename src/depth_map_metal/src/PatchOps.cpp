#include "av/depth_map/PatchOps.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Pipeline.hpp"

#include <stdexcept>

namespace av::depth_map {

struct PatchOps::Impl {
    av::gpu::Device&    device;
    av::gpu::Pipeline   pipeline;
    DeviceCameraParams  rc;
    DeviceCameraParams  tc;

    Impl(av::gpu::Device& d, av::gpu::Pipeline p,
         const DeviceCameraParams& r, const DeviceCameraParams& t) noexcept
        : device(d), pipeline(std::move(p)), rc(r), tc(t) {}
};

PatchOps::PatchOps(av::gpu::Device& dev,
                   const DeviceCameraParams& rc,
                   const DeviceCameraParams& tc)
    : impl_(std::make_unique<Impl>(dev,
                                   dev.make_pipeline("av_patch_validate"),
                                   rc, tc))
{}

PatchOps::PatchOps(PatchOps&&) noexcept            = default;
PatchOps& PatchOps::operator=(PatchOps&&) noexcept = default;
PatchOps::~PatchOps()                              = default;

void PatchOps::validate(std::span<const float> inputs,
                        std::span<float>        outputs)
{
    if (inputs.size() % kInPerCase != 0)
        throw std::invalid_argument("PatchOps: inputs size mismatch");
    const std::size_t count = inputs.size() / kInPerCase;
    if (outputs.size() != count * kOutPerCase)
        throw std::invalid_argument("PatchOps: outputs size mismatch");

    using namespace av::gpu;
    auto& dev = impl_->device;

    Buffer in_buf (dev, inputs.size_bytes());
    Buffer out_buf(dev, outputs.size_bytes());
    in_buf.upload(inputs);

    const std::uint32_t count_u32 = static_cast<std::uint32_t>(count);

    CommandBuffer cb(dev);
    cb.set_pipeline(impl_->pipeline)
      .set_bytes  (0, &impl_->rc, sizeof(DeviceCameraParams))
      .set_bytes  (1, &impl_->tc, sizeof(DeviceCameraParams))
      .set_buffer (2, in_buf)
      .set_buffer (3, out_buf)
      .set_bytes  (4, &count_u32, sizeof(count_u32))
      .dispatch_1d(impl_->pipeline, count_u32);
    cb.commit_and_wait();

    auto src = out_buf.as_span<const float>();
    std::copy(src.begin(), src.end(), outputs.begin());
}

}  // namespace av::depth_map
