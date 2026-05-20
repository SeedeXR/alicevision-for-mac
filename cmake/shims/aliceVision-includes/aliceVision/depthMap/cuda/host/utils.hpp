#pragma once

// utils.hpp — Apple Silicon type-shim. Replaces upstream's
// `aliceVision/depthMap/cuda/host/utils.hpp` (which pulls in
// `<cuda_runtime.h>` for `cudaError_t`, `cudaGetErrorString`,
// `cudaMemGetInfo`, etc.).
//
// Surface mirrors upstream:
//
//   * int  listCudaDevices()                — 1 on Apple Silicon.
//   * int  getCudaDeviceId()                — 0.
//   * void setCudaDeviceId(int id)          — no-op (logs if id≠0).
//   * bool testCudaDeviceId(int id)         — id == 0.
//   * void logDeviceMemoryInfo()            — UMA host memory via Mach.
//   * void getDeviceMemoryInfo(double& a, double& u, double& t)
//                                           — UMA host memory via Mach.
//
// CUDA error macros (`CHECK_CUDA_RETURN_ERROR`, …, `THROW_ON_CUDA_ERROR`)
// are kept as no-ops — upstream `.cpp` files that include this
// header use them to wrap CUDA calls; on macOS we have no CUDA so
// the wrapped expression result is discarded.

#include <cstdio>

namespace aliceVision {
namespace depthMap {

int  listCudaDevices();
int  getCudaDeviceId();
void setCudaDeviceId(int cudaDeviceId);
bool testCudaDeviceId(int cudaDeviceId);
void logDeviceMemoryInfo();
void getDeviceMemoryInfo(double& availableMB, double& usedMB, double& totalMB);

}  // namespace depthMap
}  // namespace aliceVision

// ============================================================
// CUDA error-checking macros — kept as no-ops on macOS.
// ============================================================
//
// Upstream code uses these to wrap CUDA API calls; on macOS the
// argument is typically already a no-op (e.g. `cudaSuccess`-like
// integer or void-returning sham) and we just discard it. The
// `do { (void)(err); } while(0)` form preserves macro-as-statement
// semantics (the trailing semicolon at the call site is fine).
//
// These match upstream's spelling 1:1 so upstream `.cpp` files that
// don't get rewritten compile unmodified.

#ifndef CHECK_CUDA_RETURN_ERROR
#define CHECK_CUDA_RETURN_ERROR(err) do { (void)(err); } while (0)
#endif
#ifndef CHECK_CUDA_RETURN_ERROR_NOEXCEPT
#define CHECK_CUDA_RETURN_ERROR_NOEXCEPT(err) do { (void)(err); } while (0)
#endif
#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR() do { } while (0)
#endif
#ifndef THROW_ON_CUDA_ERROR
#define THROW_ON_CUDA_ERROR(rcode, message) do { (void)(rcode); } while (0)
#endif
