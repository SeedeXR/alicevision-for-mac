#include "av/depth_map/DeviceCache.hpp"

#include "av/gpu/Device.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace av::depth_map {

struct DeviceCache::Impl {
    av::gpu::Device& device;

    // LRU slot trackers. Each tracker maps a key → slot index.
    LRUCache<int>        mipmap_cache;
    LRUCache<CameraPair> camera_cache;

    // Slot storage. Indexed by the slot returned from the
    // matching LRUCache.
    std::vector<std::unique_ptr<DeviceMipmapImage>>  mipmaps;
    std::vector<DeviceCameraParams>                  camera_params;

    // Mirror of (camera_id, downscale) → slot, kept in sync with
    // `camera_cache` for the key lookup in
    // request_camera_params.
    std::vector<CameraPair> camera_params_keys;

    Impl(av::gpu::Device& d, int max_mip, int max_cam)
        : device(d),
          mipmap_cache(max_mip),
          camera_cache(max_cam),
          mipmaps(static_cast<std::size_t>(max_mip)),
          camera_params(static_cast<std::size_t>(max_cam)),
          camera_params_keys(static_cast<std::size_t>(max_cam),
                              CameraPair(-1))
    {}
};

DeviceCache::DeviceCache(av::gpu::Device& dev,
                         int max_mipmap_images,
                         int max_camera_params)
    : impl_(std::make_unique<Impl>(dev, max_mipmap_images, max_camera_params))
{
    if (max_mipmap_images <= 0 || max_camera_params <= 0) {
        throw std::invalid_argument(
            "DeviceCache: max sizes must be positive");
    }
}

DeviceCache::DeviceCache(DeviceCache&&) noexcept            = default;
DeviceCache& DeviceCache::operator=(DeviceCache&&) noexcept = default;
DeviceCache::~DeviceCache()                                  = default;

void DeviceCache::add_mipmap_image(int                    camera_id,
                                   std::uint32_t          min_downscale,
                                   std::uint32_t          max_downscale,
                                   std::span<const float> rgba_image,
                                   std::uint32_t          width,
                                   std::uint32_t          height)
{
    if (camera_id < 0) {
        throw std::invalid_argument(
            "DeviceCache::add_mipmap_image: camera_id must be non-negative");
    }

    int slot = -1;
    int old_val = -1;
    const bool inserted_or_evicted =
        impl_->mipmap_cache.insert(camera_id, slot, old_val);

    if (inserted_or_evicted) {
        // Either new slot or LRU eviction — (re)build the mipmap.
        auto fresh = std::make_unique<DeviceMipmapImage>(impl_->device);
        fresh->fill(rgba_image, width, height,
                    min_downscale, max_downscale);
        impl_->mipmaps[static_cast<std::size_t>(slot)] = std::move(fresh);
    }
    // else: already cached → LRU recency was refreshed; no
    // rebuild needed (assumes caller hasn't changed the image).
}

void DeviceCache::add_camera_params(int                       camera_id,
                                    int                       downscale,
                                    const DeviceCameraParams& params)
{
    if (camera_id < 0 || downscale <= 0) {
        throw std::invalid_argument(
            "DeviceCache::add_camera_params: bad camera_id/downscale");
    }

    const CameraPair key(camera_id, downscale);
    int slot = -1;
    CameraPair old_key(-1);
    const bool inserted_or_evicted =
        impl_->camera_cache.insert(key, slot, old_key);
    (void)inserted_or_evicted;   // either way, we update the slot
    // with the caller-supplied params. If the slot already held
    // this same key, refreshing is harmless.

    impl_->camera_params[static_cast<std::size_t>(slot)]      = params;
    impl_->camera_params_keys[static_cast<std::size_t>(slot)] = key;
}

const DeviceMipmapImage&
DeviceCache::request_mipmap_image(int camera_id) const
{
    const int slot = impl_->mipmap_cache.get_index(camera_id);
    if (slot < 0) {
        throw std::out_of_range(
            "DeviceCache::request_mipmap_image: camera_id " +
            std::to_string(camera_id) + " not cached");
    }
    auto& ptr = impl_->mipmaps[static_cast<std::size_t>(slot)];
    if (!ptr) {
        throw std::logic_error(
            "DeviceCache::request_mipmap_image: slot exists but mipmap is null");
    }
    return *ptr;
}

const DeviceCameraParams&
DeviceCache::request_camera_params(int camera_id, int downscale) const
{
    const CameraPair key(camera_id, downscale);
    const int slot = impl_->camera_cache.get_index(key);
    if (slot < 0) {
        throw std::out_of_range(
            "DeviceCache::request_camera_params: (camera_id=" +
            std::to_string(camera_id) +
            ", downscale=" + std::to_string(downscale) + ") not cached");
    }
    return impl_->camera_params[static_cast<std::size_t>(slot)];
}

void DeviceCache::clear() {
    impl_->mipmap_cache.clear();
    impl_->camera_cache.clear();
    for (auto& m : impl_->mipmaps) m.reset();
    for (auto& k : impl_->camera_params_keys) k = CameraPair(-1);
}

int DeviceCache::mipmap_count() const noexcept {
    return impl_->mipmap_cache.size();
}
int DeviceCache::camera_param_count() const noexcept {
    return impl_->camera_cache.size();
}

}  // namespace av::depth_map
