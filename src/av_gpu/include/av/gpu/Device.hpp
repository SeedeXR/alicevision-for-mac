#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace MTL {
    class Device;
    class CommandQueue;
    class Library;
}

namespace av::gpu {

class Pipeline;

// A small typed wrapper for an MTLFunctionConstantValues set, used by
// the function-constant-specialized `Device::make_pipeline` overload.
//
// MSL function constants are typed values bound at PSO-compile time
// to declared `[[function_constant(N)]]` variables. Specializing a
// kernel for known-at-compile-time values lets the Metal compiler
// inline, dead-code-eliminate, and unroll based on those values.
//
// Usage:
//   FunctionConstants fc;
//   fc.set_int(0, 1);   // kAxis0 = 1
//   fc.set_int(1, 0);   // kAxis1 = 0
//   fc.set_int(2, 2);   // kAxis2 = 2
//   auto pso = dev.make_pipeline("my_kernel_fc", fc);
//
// Only `int` is supported currently; add more setters as needed.
class FunctionConstants {
public:
    FunctionConstants() = default;

    // Bind an int32 value to the function-constant slot `index`. The
    // MSL declaration must be `constant int kName [[function_constant(index)]]`.
    void set_int(std::uint32_t index, std::int32_t value) {
        ints_.emplace_back(index, value);
    }

    // For introspection / Device::make_pipeline. Pairs are (slot_index, value).
    const std::vector<std::pair<std::uint32_t, std::int32_t>>& ints() const noexcept {
        return ints_;
    }

private:
    std::vector<std::pair<std::uint32_t, std::int32_t>> ints_;
};

// A thin RAII handle on a Metal device + its default command queue.
// Use Device::default_device() for the system default device.
//
// Ownership semantics: the Device retains the underlying MTL::Device.
// Moves transfer ownership; copies are forbidden.
class Device {
public:
    // Acquire the system default Metal device. Throws GpuError if no
    // Metal device is available on this host (no Apple GPU).
    static Device default_device();

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;
    ~Device();

    // Human-readable device name.
    std::string name() const;

    // Recommended working set size in bytes; this is what
    // `MTL::Device::recommendedMaxWorkingSetSize` reports.
    std::uint64_t recommended_working_set() const;

    // Whether this is a unified-memory device. Always true on Apple
    // Silicon; false on (unlikely) discrete configurations.
    bool has_unified_memory() const;

    // Load a `.metallib` file. Pass an empty path to load
    // `default.metallib` next to the running executable.
    void load_library(const std::filesystem::path& metallib);

    // Build a compute pipeline from a function name in the currently
    // loaded library.
    Pipeline make_pipeline(const std::string& function_name) const;

    // Build a function-constant-specialized compute pipeline. The
    // function constants must match the declared `[[function_constant(N)]]`
    // slots in the kernel. Each PSO created this way is independent;
    // create one per specialization at init time and dispatch the
    // right one per call site.
    Pipeline make_pipeline(const std::string& function_name,
                           const FunctionConstants& constants) const;

    // Raw access (use sparingly).
    MTL::Device*       raw_device()      const noexcept;
    MTL::CommandQueue* raw_command_queue() const noexcept;
    MTL::Library*      raw_library()     const noexcept;

private:
    Device();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::gpu
