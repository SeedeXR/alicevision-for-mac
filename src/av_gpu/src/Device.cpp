#include "av/gpu/Device.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Errors.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <mach-o/dyld.h>
#include <string>
#include <vector>

namespace av::gpu {

// ----------------------- error helper ------------------------------

[[noreturn]] void throw_from_ns_error(const char* what, const NS::Error* err) {
    std::string msg = what ? what : "Metal error";
    if (err) {
        if (auto* desc = err->localizedDescription()) {
            msg += ": ";
            msg += desc->utf8String();
        }
    }
    throw GpuError(msg);
}

// ----------------------- Pipeline impl -----------------------------
// (defined here to keep build TU count down; logically belongs in
// Pipeline.cpp but the Device <-> Pipeline coupling is tight.)

struct Pipeline::Impl {
    NS::SharedPtr<MTL::ComputePipelineState> pso;
    std::string                              name;
};

Pipeline::Pipeline()              : impl_(std::make_unique<Impl>()) {}
Pipeline::Pipeline(Pipeline&&) noexcept            = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;
Pipeline::~Pipeline()                              = default;

const std::string& Pipeline::function_name() const noexcept { return impl_->name; }

std::size_t Pipeline::max_threads_per_threadgroup() const noexcept {
    return impl_->pso ? impl_->pso->maxTotalThreadsPerThreadgroup() : 0;
}

std::size_t Pipeline::thread_execution_width() const noexcept {
    return impl_->pso ? impl_->pso->threadExecutionWidth() : 0;
}

MTL::ComputePipelineState* Pipeline::raw() const noexcept {
    return impl_->pso.get();
}

// ----------------------- Device impl -------------------------------

struct Device::Impl {
    NS::SharedPtr<MTL::Device>       device;
    NS::SharedPtr<MTL::CommandQueue> queue;
    NS::SharedPtr<MTL::Library>      library;  // may be null until load_library
};

Device::Device() : impl_(std::make_unique<Impl>()) {}
Device::Device(Device&&) noexcept            = default;
Device& Device::operator=(Device&&) noexcept = default;
Device::~Device()                            = default;

Device Device::default_device() {
    Device d;
    d.impl_->device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (!d.impl_->device) {
        throw GpuError("No Metal-capable GPU available on this host.");
    }
    d.impl_->queue = NS::TransferPtr(d.impl_->device->newCommandQueue());
    if (!d.impl_->queue) {
        throw GpuError("Failed to create MTLCommandQueue.");
    }
    return d;
}

std::string Device::name() const {
    if (!impl_->device) return {};
    auto* n = impl_->device->name();
    return n ? std::string(n->utf8String()) : std::string{};
}

std::uint64_t Device::recommended_working_set() const {
    return impl_->device ? impl_->device->recommendedMaxWorkingSetSize() : 0;
}

bool Device::has_unified_memory() const {
    return impl_->device ? impl_->device->hasUnifiedMemory() : false;
}

// Locate the running executable's directory so we can find
// `default.metallib` next to it without hard-coded paths.
static std::filesystem::path executable_dir() {
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buf(size + 1, 0);
    _NSGetExecutablePath(buf.data(), &size);
    std::filesystem::path p(buf.data());
    return std::filesystem::canonical(p).parent_path();
}

void Device::load_library(const std::filesystem::path& metallib) {
    if (!impl_->device) {
        throw GpuError("Device::load_library called on default-constructed Device.");
    }

    std::filesystem::path path = metallib;
    if (path.empty()) {
        path = executable_dir() / "default.metallib";
    }
    if (!std::filesystem::exists(path)) {
        throw GpuError("metallib not found: " + path.string());
    }

    auto* url = NS::URL::fileURLWithPath(
        NS::String::string(path.c_str(), NS::UTF8StringEncoding));
    NS::Error* err = nullptr;
    auto lib = NS::TransferPtr(impl_->device->newLibrary(url, &err));
    if (!lib) {
        throw_from_ns_error("Failed to load metallib", err);
    }
    impl_->library = std::move(lib);
}

Pipeline Device::make_pipeline(const std::string& function_name) const {
    if (!impl_->library) {
        throw GpuError("Device::make_pipeline called before load_library().");
    }
    auto func = NS::TransferPtr(impl_->library->newFunction(
        NS::String::string(function_name.c_str(), NS::UTF8StringEncoding)));
    if (!func) {
        throw GpuError("Kernel function not found in metallib: " + function_name);
    }
    NS::Error* err = nullptr;
    auto pso = NS::TransferPtr(impl_->device->newComputePipelineState(func.get(), &err));
    if (!pso) {
        throw_from_ns_error(("PSO creation failed for " + function_name).c_str(), err);
    }
    Pipeline p;
    p.impl_->pso  = std::move(pso);
    p.impl_->name = function_name;
    return p;
}

// S48: function-constant-specialized pipeline. Builds an
// MTLFunctionConstantValues, calls `newFunction(name, constants, &err)`
// to specialize, then `newComputePipelineState(specialized, &err)`.
//
// Each call creates an independent PSO. The Metal compiler runs at
// PSO-creation time and inlines the constant values into the kernel,
// enabling DCE, loop unrolling, and register-allocation gains.
Pipeline Device::make_pipeline(const std::string& function_name,
                               const FunctionConstants& constants) const {
    if (!impl_->library) {
        throw GpuError("Device::make_pipeline called before load_library().");
    }

    auto fcv = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    if (!fcv) {
        throw GpuError("Failed to allocate MTLFunctionConstantValues.");
    }
    for (const auto& [slot, value] : constants.ints()) {
        // Metal expects a void pointer to the constant value, plus its
        // data type. `int` in MSL maps to MTLDataTypeInt.
        const int v = value;
        fcv->setConstantValue(&v, MTL::DataTypeInt, NS::UInteger(slot));
    }

    NS::Error* err = nullptr;
    auto func = NS::TransferPtr(impl_->library->newFunction(
        NS::String::string(function_name.c_str(), NS::UTF8StringEncoding),
        fcv.get(), &err));
    if (!func) {
        throw_from_ns_error(
            ("Specialized newFunction failed for " + function_name).c_str(), err);
    }
    err = nullptr;
    auto pso = NS::TransferPtr(
        impl_->device->newComputePipelineState(func.get(), &err));
    if (!pso) {
        throw_from_ns_error(
            ("Specialized PSO creation failed for " + function_name).c_str(), err);
    }
    Pipeline p;
    p.impl_->pso  = std::move(pso);
    p.impl_->name = function_name;
    return p;
}

MTL::Device*       Device::raw_device()        const noexcept { return impl_->device.get();  }
MTL::CommandQueue* Device::raw_command_queue() const noexcept { return impl_->queue.get();   }
MTL::Library*      Device::raw_library()       const noexcept { return impl_->library.get(); }

}  // namespace av::gpu
