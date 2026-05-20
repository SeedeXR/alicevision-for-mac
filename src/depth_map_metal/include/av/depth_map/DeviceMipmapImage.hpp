#pragma once

// DeviceMipmapImage — host class that maintains an image
// pyramid in GPU memory. Mirrors upstream's
// `aliceVision::depthMap::DeviceMipmapImage` (in
// `depthMap/cuda/host/DeviceMipmapImage.hpp`).
//
// Internally wraps an `av::gpu::Texture` with `min_downscale ..
// max_downscale` mipmap levels (where the levels are spaced by
// powers of two). `fill()` ingests a host-side RGBA float image,
// optionally downscales it via Gaussian blur (when min_downscale
// > 1), converts RGB → CIELAB in-place, and generates the
// mipmap pyramid via Metal's built-in 2× downsampler.
//
// Apple Silicon note: upstream's CUDA `cuda_createMipmappedArray
// FromImage` (in `deviceMipmappedArray.cu`, Phase 6 remainder)
// uses a custom MSL kernel for the mip cascade. We use
// `MTLBlitCommandEncoder generateMipmapsForTexture:` which is
// the standard 2× downsampler. Numerical difference vs upstream
// will be small (each level is a 2× average of the previous, on
// either side); a custom port can replace `generate_mipmaps()`
// later if exact upstream parity is needed.
//
// Lab scale: our port's `xyz2lab` (see `color.h`) multiplies the
// classical [0, 100] / [-128, 128] Lab output by 2.55 — matching
// upstream's CUDA convention so the values fit a uchar/[0, 255]
// scale. Consumers of this texture should expect L in
// `[0, ~255]` and a/b in `[~-256, ~256]`, NOT classical Lab.

#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace av::gpu {
    class Device;
    class Texture;
}

namespace av::depth_map {

class GaussianTable;
class GaussianFilter;
class ImageColorConversion;
class MipmappedArray;

class DeviceMipmapImage {
public:
    explicit DeviceMipmapImage(av::gpu::Device& dev);

    DeviceMipmapImage(const DeviceMipmapImage&)            = delete;
    DeviceMipmapImage& operator=(const DeviceMipmapImage&) = delete;
    DeviceMipmapImage(DeviceMipmapImage&&) noexcept;
    DeviceMipmapImage& operator=(DeviceMipmapImage&&) noexcept;
    ~DeviceMipmapImage();

    // Toggle the mip-cascade generator used by `fill()`:
    //   * `false` (default): Metal's built-in `generateMipmaps:`
    //     (a standard 2× box average per level). Fast and matches
    //     no specific reference but is a textbook downsample.
    //   * `true`: the ported upstream kernel
    //     `av_create_mipmapped_array_level` (5×5 Gaussian-weighted
    //     bilinear stencil at upstream scale=1). Matches upstream
    //     CUDA `cuda_createMipmappedArrayFromImage` numerically.
    //
    // The setting must be applied before `fill()`. Changing it
    // after `fill()` only affects subsequent fills.
    void set_use_upstream_mipgen(bool enabled) noexcept;
    bool use_upstream_mipgen() const noexcept;

    // Ingest a host RGBA float image (row-packed, `width × height
    // × 4` floats). Then:
    //   1. If `min_downscale > 1`: downscale via Gaussian blur
    //      (radius = min_downscale).
    //   2. RGB → Lab (in-place on the downscaled level-0 texture).
    //   3. Generate the mip cascade via either Metal blit (default)
    //      or the upstream-ported MSL kernel (when
    //      `set_use_upstream_mipgen(true)` was called).
    //
    // Constraints: `min_downscale` and `max_downscale` must both
    // be positive powers of two with `min_downscale ≤
    // max_downscale`. The resulting texture has
    // `log2(max_downscale / min_downscale) + 1` mip levels.
    void fill(std::span<const float> rgba_image,
              std::uint32_t          width,
              std::uint32_t          height,
              std::uint32_t          min_downscale,
              std::uint32_t          max_downscale);

    // The underlying mipmapped texture. Valid after `fill()`.
    const av::gpu::Texture& texture() const;

    // The MSL `level()` value to pass when sampling the texture
    // for a given downscale factor. Returns `log2(downscale /
    // min_downscale)`. Throws on out-of-range.
    float get_level(std::uint32_t downscale) const;

    // The (width, height) of the mipmap level for the given
    // downscale (= ceil(original / downscale)). Throws on
    // out-of-range.
    std::pair<std::uint32_t, std::uint32_t>
    get_dimensions(std::uint32_t downscale) const;

    std::uint32_t min_downscale() const noexcept;
    std::uint32_t max_downscale() const noexcept;
    std::uint32_t width () const noexcept;   // original full-res
    std::uint32_t height() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
