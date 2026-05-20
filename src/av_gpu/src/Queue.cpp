#include "av/gpu/Queue.hpp"

#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstring>

namespace av::gpu {

struct Queue::Impl {
    NS::SharedPtr<MTL::CommandQueue> queue;
};

Queue::Queue(const Device& dev)
    : impl_(std::make_unique<Impl>())
{
    auto* mtl_dev = dev.raw_device();
    if (!mtl_dev) {
        throw GpuError("Queue: Device has no MTLDevice.");
    }
    impl_->queue = NS::TransferPtr(mtl_dev->newCommandQueue());
    if (!impl_->queue) {
        throw GpuError("Queue: failed to create MTLCommandQueue.");
    }
}

Queue::Queue(Queue&&) noexcept            = default;
Queue& Queue::operator=(Queue&&) noexcept = default;
Queue::~Queue()                            = default;

void Queue::wait_until_completed()
{
    if (!impl_->queue) return;
    // The standard "drain this queue" idiom: submit a no-op
    // command buffer + wait. Any work submitted to this queue
    // before us is serialized ahead and must complete first.
    auto cb = NS::RetainPtr(impl_->queue->commandBuffer());
    if (!cb) {
        throw GpuError("Queue::wait_until_completed: commandBuffer() failed.");
    }
    cb->commit();
    cb->waitUntilCompleted();
}

void Queue::set_label(const char* label)
{
    if (!impl_->queue || !label) return;
    auto* ns_label = NS::String::string(label, NS::UTF8StringEncoding);
    impl_->queue->setLabel(ns_label);
}

MTL::CommandQueue* Queue::raw_command_queue() const noexcept {
    return impl_->queue.get();
}

}  // namespace av::gpu
