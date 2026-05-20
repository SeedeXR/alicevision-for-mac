#pragma once

// MipmappedArray — host driver for the upstream-port mipmap
// construction kernel
// (`av_create_mipmapped_array_level` in mipmap_array.metal).
//
// Mirrors upstream's `cuda_createMipmappedArrayFromImage` from
// `depthMap/cuda/imageProcessing/deviceMipmappedArray.cu`. Each
// downscale step samples a 5×5 stencil from the previous level
// using bilinear filtering and separable Gaussian weights at scale
// index 1 (radius=2, delta=1.0). This matches upstream verbatim
// and is intentionally NOT a textbook 2× box-average, so the
// output differs from Metal's built-in `generateMipmaps:` blit.
//
// Usage: `build_mip_cascade(dst, gauss, dst_width, dst_height,
// dst_levels)`. The destination texture's level 0 must already
// hold the seed image (e.g. the Lab-converted level-0 from
// DeviceMipmapImage::fill).

#include <cstdint>
#include <memory>

namespace av::gpu {
    class Device;
    class Texture;
}

namespace av::depth_map {

class GaussianTable;

class MipmappedArray {
public:
    explicit MipmappedArray(av::gpu::Device& dev, GaussianTable& table);

    MipmappedArray(const MipmappedArray&)            = delete;
    MipmappedArray& operator=(const MipmappedArray&) = delete;
    MipmappedArray(MipmappedArray&&) noexcept;
    MipmappedArray& operator=(MipmappedArray&&) noexcept;
    ~MipmappedArray();

    // Build mip levels 1..levels-1 of `dst` using the upstream
    // weighted-Gaussian downscale. `dst.mip_levels()` must equal
    // `levels` (so the caller has already sized the texture).
    // Level 0 must already be populated.
    //
    // Source/format: dst must be RGBA32Float. Level dims follow
    // standard halving: level L has dims (max(1, w0>>L),
    // max(1, h0>>L)).
    void build_mip_cascade(av::gpu::Texture& dst);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
