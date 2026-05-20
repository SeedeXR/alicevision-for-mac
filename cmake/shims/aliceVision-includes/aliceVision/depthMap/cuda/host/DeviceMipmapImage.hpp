#pragma once

// DeviceMipmapImage.hpp — Apple Silicon type-shim. Replaces
// upstream's `aliceVision/depthMap/cuda/host/DeviceMipmapImage.hpp`
// (which depends on `cudaTextureObject_t` / `cudaMipmappedArray_t`,
// neither of which exist on Apple Silicon).
//
// The shim's `aliceVision::depthMap::DeviceMipmapImage` is a thin
// wrapper around a pointer to our `av::depth_map::DeviceMipmapImage`
// (declared in `<av/depth_map/DeviceMipmapImage.hpp>`). It exposes
// the upstream-facing surface used by upstream's host `.cpp`
// orchestration code (Sgm.cpp / Refine.cpp / etc.):
//
//   * `getLevel(unsigned int downscale)`           — float
//   * `getDimensions(unsigned int downscale)`      — CudaSize<2>
//   * `getMinDownscale() / getMaxDownscale()`
//   * `fill(...)` — accepts the upstream `CudaHostMemoryHeap<CudaRGBA, 2>&`
//                   signature; converts and forwards to our `fill()` which
//                   takes a `std::span<const float>`.
//
// Plus the adapter-only extension:
//
//   * `av_texture()` — returns the `av::gpu::Texture&` that backs
//     the mipmap. This is the bridge the `cuda_*` forwarders use
//     when calling our `av::depth_map::Volume / DepthSimMap` methods.
//
// Stub-only: `getTextureObject()` returns `nullptr`. No host `.cpp`
// in scope calls it (verified by grep across the 5 host files).
//
// Construction: a shim DeviceMipmapImage holds an
// `av::depth_map::DeviceMipmapImage` instance internally — the
// adapter forwarders never construct one directly; tests construct
// one and pass it via the shim's helper ctor.

#include "av/depth_map/DeviceMipmapImage.hpp"
#include "av/gpu/Texture.hpp"
#include "memory.hpp"   // CudaSize, CudaHostMemoryHeap, CudaRGBA (sibling shim)

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace aliceVision {
namespace depthMap {

// Upstream uses `cudaTextureObject_t` (= `unsigned long long`).
// Define a stand-in so signatures parse. Never honored — see
// notes above; host `.cpp` files don't call `getTextureObject()`.
#if !defined(__CUDACC__) && !defined(AV_ADAPTER_CUDA_TEXTURE_TYPES_DEFINED)
#define AV_ADAPTER_CUDA_TEXTURE_TYPES_DEFINED 1
using cudaTextureObject_t   = unsigned long long;
using cudaMipmappedArray_t  = void*;
#endif

class DeviceMipmapImage {
public:
    DeviceMipmapImage() = default;

    // Adapter-only construction: build a shim around an existing
    // `av::depth_map::DeviceMipmapImage` (lifetime-borrowed). The
    // shim does NOT take ownership; the caller must keep the inner
    // object alive at least as long as the shim.
    explicit DeviceMipmapImage(av::depth_map::DeviceMipmapImage& inner) noexcept
        : _inner(&inner) {}

    DeviceMipmapImage(const DeviceMipmapImage&)            = delete;
    DeviceMipmapImage& operator=(const DeviceMipmapImage&) = delete;
    DeviceMipmapImage(DeviceMipmapImage&&) noexcept            = default;
    DeviceMipmapImage& operator=(DeviceMipmapImage&&) noexcept = default;
    ~DeviceMipmapImage() = default;

    // Adapter setter — installs a non-owning pointer to our impl.
    // Useful when the shim is default-constructed (e.g., as a
    // singleton in DeviceCache).
    void set_av_impl(av::depth_map::DeviceMipmapImage& inner) noexcept {
        _inner = &inner;
    }

    // ------- Upstream surface (used by host .cpp orchestration) -------

    // Ingest an RGBA host image (`CudaHostMemoryHeap<CudaRGBA, 2>`,
    // row-packed). Forwards to our `av::depth_map::DeviceMipmapImage::fill`
    // by reinterpreting the host buffer as a flat float span.
    void fill(const CudaHostMemoryHeap<CudaRGBA, 2>& in_img_hmh,
              int minDownscale, int maxDownscale)
    {
        if (!_inner) {
            throw std::runtime_error(
                "aliceVision::depthMap::DeviceMipmapImage::fill: "
                "no av::depth_map::DeviceMipmapImage attached "
                "(call set_av_impl first).");
        }
        const auto& dim = in_img_hmh.getSize();
        const std::uint32_t w = static_cast<std::uint32_t>(dim.x());
        const std::uint32_t h = static_cast<std::uint32_t>(dim.y());
        // CudaRGBA is float4 in the default config (4 floats per pixel).
        const float* p = reinterpret_cast<const float*>(in_img_hmh.getBuffer());
        _inner->fill({ p, std::size_t(w) * std::size_t(h) * 4u },
                     w, h,
                     static_cast<std::uint32_t>(minDownscale),
                     static_cast<std::uint32_t>(maxDownscale));
    }

    // `level()` to pass when sampling for a given downscale.
    float getLevel(unsigned int downscale) const {
        return require_inner().get_level(static_cast<std::uint32_t>(downscale));
    }

    // `(width, height)` of the level matching `downscale`.
    CudaSize<2> getDimensions(unsigned int downscale) const {
        auto [w, h] = require_inner().get_dimensions(
            static_cast<std::uint32_t>(downscale));
        return CudaSize<2>(static_cast<std::size_t>(w),
                           static_cast<std::size_t>(h));
    }

    unsigned int getMinDownscale() const {
        return require_inner().min_downscale();
    }

    unsigned int getMaxDownscale() const {
        return require_inner().max_downscale();
    }

    // STUB: upstream host code never reads this (verified). Return
    // zero so existing test fixtures don't crash.
    cudaTextureObject_t getTextureObject() const { return 0; }

    // ------- Adapter-only extension -------

    // Returns the `av::gpu::Texture&` that backs this mipmap. This
    // is the bridge that the `cuda_*` forwarders use to feed our
    // `av::depth_map::Volume::*` / `DepthSimMap::*` methods.
    const av::gpu::Texture& av_texture() const {
        return require_inner().texture();
    }

    // Access to the underlying `av::depth_map::DeviceMipmapImage&`
    // for advanced cases (e.g., when forwarders need to call
    // `get_level()` directly).
    av::depth_map::DeviceMipmapImage&       av_impl()       { return require_inner(); }
    const av::depth_map::DeviceMipmapImage& av_impl() const { return require_inner(); }

private:
    av::depth_map::DeviceMipmapImage* _inner = nullptr;

    av::depth_map::DeviceMipmapImage& require_inner() const {
        if (!_inner) {
            throw std::runtime_error(
                "aliceVision::depthMap::DeviceMipmapImage: "
                "no av::depth_map::DeviceMipmapImage attached.");
        }
        return *_inner;
    }
};

}  // namespace depthMap
}  // namespace aliceVision
