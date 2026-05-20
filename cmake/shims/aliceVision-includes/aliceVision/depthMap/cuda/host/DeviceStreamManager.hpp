#pragma once

// DeviceStreamManager.hpp — Apple Silicon type-shim. Replaces
// upstream's `aliceVision/depthMap/cuda/host/DeviceStreamManager.hpp`
// (which pulls in `cuda_runtime.h` for `cudaStream_t` /
// `cudaStreamCreate`).
//
// Public surface mirrors upstream exactly:
//
//   * ctor(nbStreams) / dtor
//   * `int  getNbStreams() const`
//   * `cudaStream_t getStream(int streamIndex)`
//   * `void waitStream(int streamIndex)`
//
// `cudaStream_t` is supplied by `memory.hpp` (sibling shim) as
// `void*`. We don't honor stream semantics in upstream callers —
// every dispatch in our forwarders finishes synchronously on the
// default queue. The pointer we hand back is the address of an
// `av::gpu::Queue` inside our internal manager, so a future
// rewire that wants to thread stream IDs through to MSL
// command-buffers can recover the queue with a
// `reinterpret_cast`.
//
// (We intentionally do NOT replicate upstream's `<cuda_runtime.h>`
// include — the `cudaStream_t` typedef arrives via the
// `memory.hpp` shim that upstream code transitively pulls in.)

#include "memory.hpp"   // brings in `cudaStream_t` (= void*)

#include <cstdint>
#include <memory>

namespace aliceVision {
namespace depthMap {

class DeviceStreamManager {
public:
    explicit DeviceStreamManager(int nbStreams);
    ~DeviceStreamManager();

    DeviceStreamManager(const DeviceStreamManager&)            = delete;
    DeviceStreamManager& operator=(const DeviceStreamManager&) = delete;

    int getNbStreams() const { return _nbStreams; }

    // Returns the opaque stream handle at index
    // `streamIndex % nbStreams`. Treated as opaque by upstream code.
    cudaStream_t getStream(int streamIndex);

    // Blocks until all work submitted to the stream at
    // `streamIndex % nbStreams` has completed.
    void waitStream(int streamIndex);

private:
    int _nbStreams;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace depthMap
}  // namespace aliceVision
