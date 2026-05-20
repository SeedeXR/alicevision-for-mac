#pragma once

// Queue — RAII wrapper around an `MTL::CommandQueue`. Used by
// `DeviceStreamManager` to provide multiple independent
// dispatch streams.
//
// The default code path (most av_gpu users today) lives on the
// single queue owned by `Device`. A `Queue` is only needed when
// a caller wants to dispatch work that runs *concurrently* with
// other dispatches — i.e., the pipelined batch use case in
// upstream's `DeviceStreamManager`.

#include <memory>

namespace MTL {
    class CommandQueue;
}

namespace av::gpu {

class Device;

class Queue {
public:
    // Allocate a fresh MTLCommandQueue on the given Device. Throws
    // GpuError on allocation failure.
    explicit Queue(const Device& dev);

    Queue(const Queue&)            = delete;
    Queue& operator=(const Queue&) = delete;
    Queue(Queue&&) noexcept;
    Queue& operator=(Queue&&) noexcept;
    ~Queue();

    // Submit an empty command buffer on this queue and block until
    // it completes. Mirrors `cudaStreamSynchronize` semantics:
    // returns once all previously-submitted work on this queue is
    // done.
    void wait_until_completed();

    // Optional debug label (visible in Metal frame capture).
    void set_label(const char* label);

    // Internal: raw MTL handle. Used by `CommandBuffer(const Queue&)`.
    MTL::CommandQueue* raw_command_queue() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::gpu
