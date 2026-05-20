#pragma once

// DeviceCache.hpp — Apple Silicon type-shim. Replaces upstream's
// `aliceVision/depthMap/cuda/host/DeviceCache.hpp` (which pulls
// in `cuda_runtime.h` via the `device/DeviceCameraParams.hpp`
// include and uses CUDA constant-memory semantics).
//
// Public surface is byte-for-byte identical with upstream's
// `aliceVision::depthMap::DeviceCache`:
//
//   * Singleton via `getInstance()`.
//   * `build(maxMipmapImages, maxCameraParams)` / `clear()`.
//   * `addMipmapImage(camId, minDownscale, maxDownscale,
//                     imageCache, mp)`.
//   * `addCameraParams(camId, downscale, mp)`.
//   * `requestMipmapImage(camId, mp) → const DeviceMipmapImage&`
//     (returns the shim's `DeviceMipmapImage`, which is wired to
//     our `av::depth_map::DeviceMipmapImage`).
//   * `requestCameraParamsId(camId, downscale, mp) → const int`
//     (returns an integer ID that `av::depth_map::upstream_adapter::
//     get_camera_param(id)` can resolve to a `DeviceCameraParams&`).
//
// Internals: backed by our `av::depth_map::DeviceCache` for LRU
// semantics, plus the process-global camera-params translation
// table in `upstream_adapter.cpp`. The translation table is the
// bridge that closes the loop: upstream's `cuda_*` host code
// receives integer IDs from `requestCameraParamsId`, and the
// `cuda_*` forwarders look those IDs back up in the table to find
// the matching `DeviceCameraParams` to feed to MSL kernels.

#include "DeviceMipmapImage.hpp"  // shim's DeviceMipmapImage

// Pull in `DeviceCameraParams` + the
// `ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS` macro that
// upstream's host code uses for per-batch sizing. The shim version
// in cuda/device/ drops the CUDA `__constant__` array declaration.
#include <aliceVision/depthMap/cuda/device/DeviceCameraParams.hpp>

#include <aliceVision/mvsUtils/MultiViewParams.hpp>
#include <aliceVision/mvsUtils/ImagesCache.hpp>
#include <aliceVision/image/Image.hpp>
#include <aliceVision/image/pixelTypes.hpp>

#include <map>
#include <memory>

namespace aliceVision {
namespace depthMap {

class DeviceCache {
public:
    static DeviceCache& getInstance() {
        static DeviceCache instance;
        return instance;
    }

    // Singleton — no copy / no move.
    DeviceCache(const DeviceCache&)            = delete;
    DeviceCache& operator=(const DeviceCache&) = delete;

    // Drop all cached entries on the current device.
    void clear();

    // (Re)build the cache for the current device.
    void build(int maxMipmapImages, int maxCameraParams);

    // Insert (or refresh LRU position of) a mipmap image keyed
    // by `camId`. The image is read out of `imageCache` via its
    // sync getter and converted to RGBA-float-255 before being
    // pushed into our `av::depth_map::DeviceCache`.
    void addMipmapImage(int camId,
                        int minDownscale,
                        int maxDownscale,
                        mvsUtils::ImagesCache<image::Image<image::RGBAfColor>>& imageCache,
                        const mvsUtils::MultiViewParams& mp);

    // Insert (or refresh) camera params for `(camId, downscale)`.
    // The slot index becomes the integer "id" upstream uses to
    // refer to these params from kernels. We also register the
    // params struct in the global adapter translation table so
    // the `cuda_*` forwarders can resolve `id → DeviceCameraParams&`.
    void addCameraParams(int camId, int downscale,
                         const mvsUtils::MultiViewParams& mp);

    // Look up the mipmap image previously inserted for `camId`.
    // Throws if not cached. The returned reference is the shim's
    // `DeviceMipmapImage`, which has `av_texture()` / `av_impl()`
    // bridges used by the `cuda_*` forwarders.
    const DeviceMipmapImage& requestMipmapImage(
        int camId, const mvsUtils::MultiViewParams& mp);

    // Look up the camera-params ID previously assigned to
    // `(camId, downscale)`. Throws if not cached. The returned ID
    // is meaningful only via the adapter translation table.
    const int requestCameraParamsId(
        int camId, int downscale, const mvsUtils::MultiViewParams& mp);

private:
    DeviceCache();
    ~DeviceCache();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace depthMap
}  // namespace aliceVision
