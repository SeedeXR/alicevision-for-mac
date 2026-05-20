#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>

namespace MTL {
    class CommandBuffer;
    class ComputeCommandEncoder;
    class Texture;
}

namespace av::gpu {

class Device;
class Queue;
class Pipeline;
class Buffer;
class Texture;

// Logical grid size in 1D/2D/3D.
struct GridSize {
    std::uint32_t x = 1, y = 1, z = 1;
};

// Threadgroup (workgroup) size.
struct GroupSize {
    std::uint32_t x = 1, y = 1, z = 1;
};

// A command buffer + a single compute encoder. Construct, bind
// pipeline + buffers, dispatch, commit. Designed for one-shot use.
// For long pipelines hosting many kernels in one command buffer,
// future API will expose a scoped encoder primitive.
class CommandBuffer {
public:
    // Construct on the Device's default command queue. Suitable
    // for one-shot dispatches in single-stream code.
    explicit CommandBuffer(const Device& dev);

    // Construct on an explicit command queue. Used by code that
    // dispatches across multiple queues (e.g.,
    // `DeviceStreamManager`) so each queue's work is serialized
    // independently.
    explicit CommandBuffer(const Queue& queue);

    CommandBuffer(const CommandBuffer&)            = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;
    CommandBuffer(CommandBuffer&&) noexcept;
    CommandBuffer& operator=(CommandBuffer&&) noexcept;
    ~CommandBuffer();

    // Bind a compute pipeline state. Required before set_buffer/dispatch.
    CommandBuffer& set_pipeline(const Pipeline& p);

    // Bind a buffer at `index` in the [[buffer(index)]] table.
    CommandBuffer& set_buffer(std::uint32_t index, const Buffer& b,
                              std::uint64_t offset = 0);

    // Bind a small (<= 4 KB) chunk of immutable host data at
    // `index` in the [[buffer(index)]] table. Useful for kernel
    // params (image dims, radius, etc.) without allocating a Buffer.
    CommandBuffer& set_bytes(std::uint32_t index, const void* data,
                             std::size_t bytes);

    // Bind a texture at `index` in the [[texture(index)]] table.
    CommandBuffer& set_texture(std::uint32_t index, const Texture& t);

    // Dispatch the kernel. Uses `dispatchThreads:threadsPerThreadgroup:`
    // (non-uniform total) on Apple Silicon — no need to pad the grid.
    CommandBuffer& dispatch(GridSize grid, GroupSize group);

    // Convenience: pick a 1D group size near the device's thread-
    // execution-width and dispatch a 1D grid of `count` threads.
    CommandBuffer& dispatch_1d(const Pipeline& p, std::uint32_t count);

    // Set a debug label on the encoder / command buffer.
    CommandBuffer& set_label(const char* label);

    // Commit and block until completion. For dev / smoke tests.
    // In production paths prefer commit_async() + completion handlers.
    void commit_and_wait();

    // Commit; invoke `done` on completion. The handler runs on a
    // Metal-managed dispatch queue; do not call into other gpu objects
    // from inside it.
    void commit_async(std::function<void()> done = nullptr);

    MTL::CommandBuffer*           raw_command_buffer() const noexcept;
    MTL::ComputeCommandEncoder*   raw_encoder()        const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::gpu
