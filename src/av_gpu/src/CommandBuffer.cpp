#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Queue.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Texture.hpp"
#include "av/gpu/Errors.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <dispatch/dispatch.h>

namespace av::gpu {

struct CommandBuffer::Impl {
    NS::SharedPtr<MTL::CommandBuffer>         cb;
    NS::SharedPtr<MTL::ComputeCommandEncoder> enc;
    const Pipeline*                           bound_pipeline = nullptr;
};

CommandBuffer::CommandBuffer(const Device& dev)
    : impl_(std::make_unique<Impl>())
{
    auto* queue = dev.raw_command_queue();
    if (!queue) {
        throw GpuError("CommandBuffer: Device has no command queue.");
    }
    impl_->cb  = NS::RetainPtr(queue->commandBuffer());
    if (!impl_->cb) {
        throw GpuError("CommandBuffer: failed to create MTL::CommandBuffer.");
    }
    impl_->enc = NS::RetainPtr(impl_->cb->computeCommandEncoder());
    if (!impl_->enc) {
        throw GpuError("CommandBuffer: failed to create compute encoder.");
    }
}

CommandBuffer::CommandBuffer(const Queue& q)
    : impl_(std::make_unique<Impl>())
{
    auto* queue = q.raw_command_queue();
    if (!queue) {
        throw GpuError("CommandBuffer: Queue is empty.");
    }
    impl_->cb = NS::RetainPtr(queue->commandBuffer());
    if (!impl_->cb) {
        throw GpuError("CommandBuffer: failed to create MTL::CommandBuffer.");
    }
    impl_->enc = NS::RetainPtr(impl_->cb->computeCommandEncoder());
    if (!impl_->enc) {
        throw GpuError("CommandBuffer: failed to create compute encoder.");
    }
}

CommandBuffer::CommandBuffer(CommandBuffer&&) noexcept            = default;
CommandBuffer& CommandBuffer::operator=(CommandBuffer&&) noexcept = default;
CommandBuffer::~CommandBuffer()                                   = default;

CommandBuffer& CommandBuffer::set_pipeline(const Pipeline& p) {
    if (!p.raw()) throw GpuError("set_pipeline: empty Pipeline.");
    impl_->enc->setComputePipelineState(p.raw());
    impl_->bound_pipeline = &p;
    return *this;
}

CommandBuffer& CommandBuffer::set_buffer(std::uint32_t index, const Buffer& b,
                                        std::uint64_t offset)
{
    impl_->enc->setBuffer(b.raw(), offset, index);
    return *this;
}

CommandBuffer& CommandBuffer::set_bytes(std::uint32_t index, const void* data,
                                       std::size_t bytes)
{
    if (bytes > 4096) {
        throw GpuError("set_bytes: payload exceeds 4 KB; use a Buffer.");
    }
    impl_->enc->setBytes(data, bytes, index);
    return *this;
}

CommandBuffer& CommandBuffer::set_texture(std::uint32_t index, const Texture& t)
{
    impl_->enc->setTexture(t.raw(), index);
    return *this;
}

CommandBuffer& CommandBuffer::dispatch(GridSize grid, GroupSize group) {
    if (!impl_->bound_pipeline) {
        throw GpuError("dispatch: no pipeline bound.");
    }
    MTL::Size g  = MTL::Size::Make(grid.x,  grid.y,  grid.z);
    MTL::Size tg = MTL::Size::Make(group.x, group.y, group.z);
    impl_->enc->dispatchThreads(g, tg);
    return *this;
}

CommandBuffer& CommandBuffer::dispatch_1d(const Pipeline& p, std::uint32_t count) {
    set_pipeline(p);
    const std::uint32_t simd = static_cast<std::uint32_t>(
        std::max<std::size_t>(p.thread_execution_width(), 1));
    const std::uint32_t max  = static_cast<std::uint32_t>(
        std::max<std::size_t>(p.max_threads_per_threadgroup(), simd));
    // Pick a power-of-two threadgroup ≤ max, multiple of simd.
    std::uint32_t group = simd;
    while ((group * 2) <= max) group *= 2;
    if (group > count) group = std::max<std::uint32_t>(1, simd);
    return dispatch({ count, 1, 1 }, { group, 1, 1 });
}

CommandBuffer& CommandBuffer::set_label(const char* label) {
    if (label) {
        auto* nslabel = NS::String::string(label, NS::UTF8StringEncoding);
        impl_->cb ->setLabel(nslabel);
        impl_->enc->setLabel(nslabel);
    }
    return *this;
}

void CommandBuffer::commit_and_wait() {
    impl_->enc->endEncoding();
    impl_->cb->commit();
    impl_->cb->waitUntilCompleted();
    auto* err = impl_->cb->error();
    if (err) {
        throw_from_ns_error("Command buffer error", err);
    }
}

void CommandBuffer::commit_async(std::function<void()> done) {
    impl_->enc->endEncoding();
    if (done) {
        // The handler block retains state; copy into a heap-owned
        // function we delete inside the block.
        auto* heap_done = new std::function<void()>(std::move(done));
        impl_->cb->addCompletedHandler(
            ^(MTL::CommandBuffer* /*cb*/) {
                (*heap_done)();
                delete heap_done;
            });
    }
    impl_->cb->commit();
}

MTL::CommandBuffer* CommandBuffer::raw_command_buffer() const noexcept {
    return impl_->cb.get();
}
MTL::ComputeCommandEncoder* CommandBuffer::raw_encoder() const noexcept {
    return impl_->enc.get();
}

}  // namespace av::gpu
