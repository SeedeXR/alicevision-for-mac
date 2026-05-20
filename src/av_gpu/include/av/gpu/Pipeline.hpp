#pragma once

#include <memory>
#include <string>

namespace MTL {
    class ComputePipelineState;
    class Function;
}

namespace av::gpu {

class Device;
class CommandBuffer;

// A compute pipeline state object (PSO). Construct via
// Device::make_pipeline(name). The PSO caches the GPU-compiled
// kernel; reuse across dispatches.
class Pipeline {
public:
    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    ~Pipeline();

    const std::string& function_name() const noexcept;

    // Max threads per threadgroup admitted by this PSO on this device.
    std::size_t max_threads_per_threadgroup() const noexcept;

    // SIMD-group width on this device (typically 32 on Apple GPUs).
    std::size_t thread_execution_width() const noexcept;

    MTL::ComputePipelineState* raw() const noexcept;

private:
    friend class Device;
    Pipeline();   // constructed by Device::make_pipeline()
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::gpu
