#pragma once

// DeviceStreamManager — owns a fixed-size pool of Metal command
// queues and exposes them as "streams" for pipelined dispatch.
// Mirrors upstream's
// `aliceVision::depthMap::DeviceStreamManager`.
//
// Upstream maps to CUDA streams (`cudaStream_t`). On Apple
// Silicon the analog is `MTLCommandQueue`: each queue serializes
// its own commands, but distinct queues run concurrently
// (subject to dependencies the runtime infers).
//
// API:
//   * `get_stream(i)` returns the queue at slot `i % nb_streams`.
//     This matches upstream's modular indexing — callers can
//     enqueue work on any integer index without worrying about
//     range.
//   * `wait_stream(i)` blocks until all previously-submitted
//     work on that queue completes (`cudaStreamSynchronize`
//     equivalent).
//   * `nb_streams()` returns the pool size.

#include <cstdint>
#include <memory>

namespace av::gpu {
    class Device;
    class Queue;
}

namespace av::depth_map {

class DeviceStreamManager {
public:
    DeviceStreamManager(av::gpu::Device& dev, int nb_streams);

    DeviceStreamManager(const DeviceStreamManager&)            = delete;
    DeviceStreamManager& operator=(const DeviceStreamManager&) = delete;
    DeviceStreamManager(DeviceStreamManager&&) noexcept;
    DeviceStreamManager& operator=(DeviceStreamManager&&) noexcept;
    ~DeviceStreamManager();

    int nb_streams() const noexcept;

    // Returns the queue at `stream_index % nb_streams`. The
    // returned reference is stable for the lifetime of `*this`.
    av::gpu::Queue& get_stream(int stream_index);

    // Block until all work submitted to `stream_index % nb_streams`
    // has completed.
    void wait_stream(int stream_index);

    // Block until all queues are drained.
    void wait_all();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
