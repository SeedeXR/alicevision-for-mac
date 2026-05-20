#pragma once

// DeviceCache — GPU-resident cache of mipmap images and camera
// parameters, keyed by camera id. Mirrors upstream's
// `aliceVision::depthMap::DeviceCache`.
//
// Differences from upstream:
//   * Not a singleton. Callers own the instance. Upstream's
//     singleton wraps a `_cachePerDevice` map keyed by CUDA
//     device id — Metal selects a single MTLDevice at app
//     startup (Apple Silicon machines have one); we don't need
//     the per-device indirection.
//   * Drops the dependency on `MultiViewParams` / `ImagesCache`
//     / `image::Image<RGBAfColor>`. Callers supply raw RGBA
//     float data + a `DeviceCameraParams`. The eventual rewire
//     of upstream's `Sgm.cpp` / `Refine.cpp` will adapt their
//     calls to this simpler API.
//   * `request_camera_params` returns a const reference to the
//     stored params struct. Upstream returns an integer "id"
//     into a CUDA constant-memory array; Metal passes params
//     directly via `set_bytes` per dispatch, so no ID indirection.
//
// Cache semantics (mirrors upstream):
//   * Two independent LRU pools: mipmap-images and camera-params.
//   * Mipmap pool keyed by `camera_id`.
//   * Camera-params pool keyed by `(camera_id, downscale)`.
//   * Capacity set at construction; eviction is LRU.

#include "av/depth_map/DeviceMipmapImage.hpp"
#include "av/depth_map/LRUCache.hpp"
#include "av/depth_map/PatchOps.hpp"   // DeviceCameraParams

#include <cstdint>
#include <memory>
#include <span>

namespace av::gpu {
    class Device;
}

namespace av::depth_map {

class DeviceCache {
public:
    DeviceCache(av::gpu::Device& dev,
                int max_mipmap_images,
                int max_camera_params);

    DeviceCache(const DeviceCache&)            = delete;
    DeviceCache& operator=(const DeviceCache&) = delete;
    DeviceCache(DeviceCache&&) noexcept;
    DeviceCache& operator=(DeviceCache&&) noexcept;
    ~DeviceCache();

    // Add (or refresh LRU position of) a mipmap image for
    // `camera_id`. If the entry is new, this builds the mipmap
    // pyramid from the supplied RGBA image. If the cache is full,
    // the least-recently-used entry is evicted and the new one
    // takes its slot.
    void add_mipmap_image(int                   camera_id,
                          std::uint32_t         min_downscale,
                          std::uint32_t         max_downscale,
                          std::span<const float> rgba_image,
                          std::uint32_t         width,
                          std::uint32_t         height);

    // Add (or refresh) camera params for `(camera_id, downscale)`.
    void add_camera_params(int camera_id, int downscale,
                           const DeviceCameraParams& params);

    // Retrieve a previously-added mipmap. Throws if not cached.
    const DeviceMipmapImage& request_mipmap_image(int camera_id) const;

    // Retrieve previously-added camera params. Throws if not
    // cached.
    const DeviceCameraParams&
    request_camera_params(int camera_id, int downscale) const;

    // Drop all entries.
    void clear();

    int mipmap_count() const noexcept;
    int camera_param_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
