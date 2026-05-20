#pragma once

// av::gpu::Texture — RAII handle on an MTL::Texture, plus a small
// set of helpers we need for the depthMap port:
//
//   * 2D textures (single layer, no array or cube).
//   * UMA-backed Shared storage so CPU upload is a memcpy.
//   * Mipmap allocation + generation via a blit encoder.
//   * Bilinear / mipmap sampling on the GPU side is configured
//     in MSL with `constexpr sampler s(...)`. CPU-side
//     MTLSamplerState is intentionally not exposed yet.
//
// Pixel formats are limited to the three we need for the early
// kernels. More can be added as ports arrive.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace MTL { class Texture; }

namespace av::gpu {

class Device;

enum class PixelFormat : std::uint8_t {
    RGBA8Unorm,    // 4 bytes/pixel
    R32Float,      // 4 bytes/pixel
    RGBA32Float,   // 16 bytes/pixel
};

class Texture {
public:
    struct Descriptor {
        std::uint32_t width      = 0;
        std::uint32_t height     = 0;
        // Number of mip levels. 0 = auto (1 + floor(log2(max(w,h)))).
        std::uint32_t mip_levels = 1;
        PixelFormat   format     = PixelFormat::RGBA8Unorm;
    };

    // Allocate. Storage is always Shared on Apple Silicon.
    Texture(const Device& dev, Descriptor desc);

    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&) noexcept;
    Texture& operator=(Texture&&) noexcept;
    ~Texture();

    std::uint32_t width()      const noexcept;
    std::uint32_t height()     const noexcept;
    std::uint32_t mip_levels() const noexcept;
    PixelFormat   format()     const noexcept;

    // Bytes per pixel for `fmt`.
    static std::size_t bytes_per_pixel(PixelFormat fmt) noexcept;

    // Upload to a specific mip level. The source is row-packed
    // (no row padding); we compute the bytesPerRow as
    // `bytes_per_pixel(format()) * mip_width`.
    void upload_level(std::span<const std::byte> src,
                      std::uint32_t mip_level = 0);

    // Convenience: upload a typed span to mip 0.
    template <class T>
    void upload(std::span<const T> src, std::uint32_t mip_level = 0) {
        upload_level({ reinterpret_cast<const std::byte*>(src.data()),
                       src.size_bytes() }, mip_level);
    }

    // Download a specific mip level to a host-side, row-packed
    // buffer. Useful for read-back after an in-place kernel writes
    // to the texture (e.g., color-conversion kernels).
    void download_level(std::span<std::byte> dst,
                        std::uint32_t mip_level = 0) const;

    template <class T>
    void download(std::span<T> dst, std::uint32_t mip_level = 0) const {
        download_level({ reinterpret_cast<std::byte*>(dst.data()),
                         dst.size_bytes() }, mip_level);
    }

    // Synchronous mipmap generation via a private blit command
    // buffer. Source levels >= 1 are written.
    void generate_mipmaps();

    void set_label(const char* label);

    MTL::Texture* raw() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::gpu
