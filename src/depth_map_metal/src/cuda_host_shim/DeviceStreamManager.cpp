// DeviceStreamManager.cpp — Apple Silicon implementation for the
// upstream `aliceVision::depthMap::DeviceStreamManager` shim.
//
// The shim is a thin wrapper around our
// `av::depth_map::DeviceStreamManager`. `getStream(i)` returns the
// address of the corresponding `av::gpu::Queue` cast to `void*`
// (= `cudaStream_t` in the memory.hpp shim). Upstream treats stream
// handles as opaque, so the round-trip pointer identity is enough
// — and a future caller that wants to thread streams into MSL
// command buffers can `reinterpret_cast<av::gpu::Queue*>(stream)`.

#include "aliceVision/depthMap/cuda/host/DeviceStreamManager.hpp"

#include "av/depth_map/DeviceStreamManager.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Queue.hpp"

#include "aliceVision/depthMap/cuda/host/memory.hpp"  // require_adapter_device

#include <memory>

namespace aliceVision {
namespace depthMap {

struct DeviceStreamManager::Impl {
    av::depth_map::DeviceStreamManager native;

    explicit Impl(int nb_streams)
        : native(require_adapter_device(), nb_streams) {}
};

DeviceStreamManager::DeviceStreamManager(int nbStreams)
    : _nbStreams(nbStreams),
      impl_(std::make_unique<Impl>(nbStreams))
{}

DeviceStreamManager::~DeviceStreamManager() = default;

cudaStream_t DeviceStreamManager::getStream(int streamIndex) {
    // Return the queue pointer as an opaque handle. Upstream
    // callers only ever compare for non-null and pass it through.
    return static_cast<cudaStream_t>(
        static_cast<void*>(&impl_->native.get_stream(streamIndex)));
}

void DeviceStreamManager::waitStream(int streamIndex) {
    impl_->native.wait_stream(streamIndex);
}

}  // namespace depthMap
}  // namespace aliceVision
