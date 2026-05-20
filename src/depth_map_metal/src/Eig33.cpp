#include "av/depth_map/Eig33.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Pipeline.hpp"

#include <stdexcept>

namespace av::depth_map {

struct Eig33::Impl {
    av::gpu::Device&  device;
    av::gpu::Pipeline pipeline;

    Impl(av::gpu::Device& d, av::gpu::Pipeline p) noexcept
        : device(d), pipeline(std::move(p)) {}
};

Eig33::Eig33(av::gpu::Device& dev)
    : impl_(std::make_unique<Impl>(dev,
                                   dev.make_pipeline("av_eig33_decompose")))
{}

Eig33::Eig33(Eig33&&) noexcept            = default;
Eig33& Eig33::operator=(Eig33&&) noexcept = default;
Eig33::~Eig33()                           = default;

void Eig33::decompose(std::span<const float> matrices_in,
                      std::span<float>        values_out,
                      std::span<float>        vectors_out)
{
    if (matrices_in.size() % 9 != 0)
        throw std::invalid_argument("Eig33: matrices_in size must be a multiple of 9");
    const std::size_t count = matrices_in.size() / 9;
    if (values_out.size()  != count * 3 ||
        vectors_out.size() != count * 9)
        throw std::invalid_argument("Eig33: values/vectors output size mismatch");

    using namespace av::gpu;
    auto& dev = impl_->device;

    Buffer in_buf  (dev, matrices_in.size_bytes());
    Buffer val_buf (dev, values_out.size_bytes());
    Buffer vec_buf (dev, vectors_out.size_bytes());
    in_buf .set_label("eig33.matrices_in");
    val_buf.set_label("eig33.values_out");
    vec_buf.set_label("eig33.vectors_out");

    // Upload input (UMA — this is a memcpy, no DMA).
    in_buf.upload(matrices_in);

    const std::uint32_t count_u32 = static_cast<std::uint32_t>(count);

    CommandBuffer cb(dev);
    cb.set_label("eig33.decompose")
      .set_pipeline(impl_->pipeline)
      .set_buffer (0, in_buf)
      .set_buffer (1, val_buf)
      .set_buffer (2, vec_buf)
      .set_bytes  (3, &count_u32, sizeof(count_u32))
      .dispatch_1d(impl_->pipeline, count_u32);
    cb.commit_and_wait();

    // Read back (UMA — direct memcpy from the same buffer).
    auto val_src = val_buf.as_span<const float>();
    auto vec_src = vec_buf.as_span<const float>();
    std::copy(val_src.begin(), val_src.end(), values_out.begin());
    std::copy(vec_src.begin(), vec_src.end(), vectors_out.begin());
}

}  // namespace av::depth_map
