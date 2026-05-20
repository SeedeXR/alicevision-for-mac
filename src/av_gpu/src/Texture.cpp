#include "av/gpu/Texture.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>

namespace av::gpu {

namespace {

MTL::PixelFormat to_metal(PixelFormat fmt) {
    switch (fmt) {
        case PixelFormat::RGBA8Unorm:  return MTL::PixelFormatRGBA8Unorm;
        case PixelFormat::R32Float:    return MTL::PixelFormatR32Float;
        case PixelFormat::RGBA32Float: return MTL::PixelFormatRGBA32Float;
    }
    return MTL::PixelFormatInvalid;
}

std::uint32_t auto_mip_levels(std::uint32_t w, std::uint32_t h) {
    const std::uint32_t mx = std::max<std::uint32_t>(w, h);
    if (mx == 0) return 1;
    // 1 + floor(log2(mx))
    return 1u + std::bit_width(mx) - 1u;
}

}  // namespace

std::size_t Texture::bytes_per_pixel(PixelFormat fmt) noexcept {
    switch (fmt) {
        case PixelFormat::RGBA8Unorm:  return 4;
        case PixelFormat::R32Float:    return 4;
        case PixelFormat::RGBA32Float: return 16;
    }
    return 0;
}

struct Texture::Impl {
    NS::SharedPtr<MTL::Texture> tex;
    Descriptor                  desc;
    MTL::Device*                device = nullptr;   // non-owning; held by Device
    MTL::CommandQueue*          queue  = nullptr;   // non-owning
};

Texture::Texture(const Device& dev, Descriptor desc)
    : impl_(std::make_unique<Impl>())
{
    if (desc.width == 0 || desc.height == 0) {
        throw std::invalid_argument("Texture: width and height must be > 0");
    }
    if (desc.mip_levels == 0) {
        desc.mip_levels = auto_mip_levels(desc.width, desc.height);
    }
    impl_->desc   = desc;
    impl_->device = dev.raw_device();
    impl_->queue  = dev.raw_command_queue();
    if (!impl_->device) {
        throw GpuError("Texture: Device has no underlying MTL::Device.");
    }

    auto* tdesc = MTL::TextureDescriptor::alloc()->init();
    tdesc->setTextureType(MTL::TextureType2D);
    tdesc->setPixelFormat(to_metal(desc.format));
    tdesc->setWidth(desc.width);
    tdesc->setHeight(desc.height);
    tdesc->setMipmapLevelCount(desc.mip_levels);
    tdesc->setStorageMode(MTL::StorageModeShared);
    tdesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);

    impl_->tex = NS::TransferPtr(impl_->device->newTexture(tdesc));
    tdesc->release();

    if (!impl_->tex) {
        throw GpuError("Texture: MTL::Device::newTexture failed.");
    }
}

Texture::Texture(Texture&&) noexcept            = default;
Texture& Texture::operator=(Texture&&) noexcept = default;
Texture::~Texture()                             = default;

std::uint32_t Texture::width()      const noexcept { return impl_->desc.width;      }
std::uint32_t Texture::height()     const noexcept { return impl_->desc.height;     }
std::uint32_t Texture::mip_levels() const noexcept { return impl_->desc.mip_levels; }
PixelFormat   Texture::format()     const noexcept { return impl_->desc.format;     }

void Texture::download_level(std::span<std::byte> dst, std::uint32_t mip_level) const {
    if (mip_level >= impl_->desc.mip_levels) {
        throw std::out_of_range("Texture::download_level: mip_level out of range");
    }
    const std::uint32_t mw = std::max<std::uint32_t>(1, impl_->desc.width  >> mip_level);
    const std::uint32_t mh = std::max<std::uint32_t>(1, impl_->desc.height >> mip_level);
    const std::size_t   bpp = bytes_per_pixel(impl_->desc.format);
    const std::size_t   bpr = bpp * mw;
    const std::size_t   need = bpr * mh;
    if (dst.size_bytes() != need) {
        throw std::invalid_argument(
            "Texture::download_level: dst size does not match mip-level extent");
    }

    MTL::Region region = MTL::Region::Make2D(0, 0, mw, mh);
    impl_->tex->getBytes(dst.data(),
                         bpr,
                         /*bytesPerImage*/ 0,
                         region,
                         mip_level,
                         /*slice*/ 0);
}

void Texture::upload_level(std::span<const std::byte> src, std::uint32_t mip_level) {
    if (mip_level >= impl_->desc.mip_levels) {
        throw std::out_of_range("Texture::upload_level: mip_level out of range");
    }
    const std::uint32_t mw = std::max<std::uint32_t>(1, impl_->desc.width  >> mip_level);
    const std::uint32_t mh = std::max<std::uint32_t>(1, impl_->desc.height >> mip_level);
    const std::size_t   bpp = bytes_per_pixel(impl_->desc.format);
    const std::size_t   bpr = bpp * mw;
    const std::size_t   need = bpr * mh;
    if (src.size_bytes() != need) {
        throw std::invalid_argument(
            "Texture::upload_level: src size does not match mip-level extent");
    }

    MTL::Region region = MTL::Region::Make2D(0, 0, mw, mh);
    impl_->tex->replaceRegion(region,
                              mip_level,
                              /*slice*/ 0,
                              src.data(),
                              bpr,
                              /*bytesPerImage*/ 0);
}

void Texture::generate_mipmaps() {
    if (impl_->desc.mip_levels <= 1) return;
    if (!impl_->queue) {
        throw GpuError("Texture::generate_mipmaps: command queue unavailable.");
    }
    MTL::CommandBuffer* cb = impl_->queue->commandBuffer();
    if (!cb) {
        throw GpuError("Texture::generate_mipmaps: failed to create command buffer.");
    }
    cb->setLabel(NS::String::string("texture.generate_mipmaps", NS::UTF8StringEncoding));
    MTL::BlitCommandEncoder* enc = cb->blitCommandEncoder();
    enc->generateMipmaps(impl_->tex.get());
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    auto* err = cb->error();
    if (err) {
        throw_from_ns_error("Texture::generate_mipmaps", err);
    }
}

void Texture::set_label(const char* label) {
    if (impl_->tex && label) {
        impl_->tex->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
    }
}

MTL::Texture* Texture::raw() const noexcept { return impl_->tex.get(); }

}  // namespace av::gpu
