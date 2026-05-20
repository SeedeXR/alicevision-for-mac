#include "av/depth_map/DeviceStreamManager.hpp"

#include "av/gpu/Device.hpp"
#include "av/gpu/Queue.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace av::depth_map {

struct DeviceStreamManager::Impl {
    std::vector<av::gpu::Queue> queues;
};

DeviceStreamManager::DeviceStreamManager(av::gpu::Device& dev, int nb_streams)
    : impl_(std::make_unique<Impl>())
{
    if (nb_streams <= 0) {
        throw std::invalid_argument(
            "DeviceStreamManager: nb_streams must be positive");
    }
    impl_->queues.reserve(static_cast<std::size_t>(nb_streams));
    for (int i = 0; i < nb_streams; ++i) {
        impl_->queues.emplace_back(dev);
        const std::string label = "av.stream." + std::to_string(i);
        impl_->queues.back().set_label(label.c_str());
    }
}

DeviceStreamManager::DeviceStreamManager(DeviceStreamManager&&) noexcept            = default;
DeviceStreamManager& DeviceStreamManager::operator=(DeviceStreamManager&&) noexcept = default;
DeviceStreamManager::~DeviceStreamManager()                                          = default;

int DeviceStreamManager::nb_streams() const noexcept {
    return static_cast<int>(impl_->queues.size());
}

av::gpu::Queue& DeviceStreamManager::get_stream(int stream_index) {
    const int n = nb_streams();
    // Match upstream's modular indexing — clamp to [0, n) by
    // taking `i % n`, but handle negative indices the C++ way
    // (`%` of a negative is implementation-defined sign of result)
    // by adding n before the second modulo.
    const int idx = ((stream_index % n) + n) % n;
    return impl_->queues[static_cast<std::size_t>(idx)];
}

void DeviceStreamManager::wait_stream(int stream_index) {
    get_stream(stream_index).wait_until_completed();
}

void DeviceStreamManager::wait_all() {
    for (auto& q : impl_->queues) q.wait_until_completed();
}

}  // namespace av::depth_map
