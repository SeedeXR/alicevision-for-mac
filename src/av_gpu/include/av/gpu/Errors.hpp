#pragma once

#include <stdexcept>
#include <string>

namespace NS { class Error; }

namespace av::gpu {

// Domain error type for Metal-side failures. Carries the original
// NSError localizedDescription when one is available.
class GpuError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Convert an NSError* into a GpuError and throw, prefixed by `what`.
// Pass a null Error* to throw a bare GpuError(what).
[[noreturn]] void throw_from_ns_error(const char* what, const NS::Error* err);

}  // namespace av::gpu
