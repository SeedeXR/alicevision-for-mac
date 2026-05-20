#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <cstring>
#include <stdexcept>

namespace av::gpu {

struct Buffer::Impl {
    NS::SharedPtr<MTL::Buffer> buf;
    Storage                    mode = Storage::Shared;
};

Buffer::Buffer(const Device& dev, std::size_t bytes, Storage mode)
    : impl_(std::make_unique<Impl>())
{
    if (bytes == 0) {
        throw std::invalid_argument("Buffer: bytes must be > 0");
    }
    auto* raw_dev = dev.raw_device();
    if (!raw_dev) {
        throw GpuError("Buffer: Device has no underlying MTL::Device.");
    }
    MTL::ResourceOptions opts = (mode == Storage::Private)
        ? MTL::ResourceStorageModePrivate
        : MTL::ResourceStorageModeShared;
    impl_->buf  = NS::TransferPtr(raw_dev->newBuffer(bytes, opts));
    impl_->mode = mode;
    if (!impl_->buf) {
        throw GpuError("Buffer: MTL::Device::newBuffer failed.");
    }
}

Buffer::Buffer(Buffer&&) noexcept            = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;
Buffer::~Buffer()                            = default;

std::size_t Buffer::size_bytes() const noexcept {
    return impl_->buf ? impl_->buf->length() : 0;
}

Storage Buffer::storage() const noexcept {
    return impl_->mode;
}

void* Buffer::data() noexcept {
    return (impl_->buf && impl_->mode == Storage::Shared)
        ? impl_->buf->contents() : nullptr;
}
const void* Buffer::data() const noexcept {
    return (impl_->buf && impl_->mode == Storage::Shared)
        ? impl_->buf->contents() : nullptr;
}

void Buffer::upload_bytes(const void* src, std::size_t bytes) {
    if (impl_->mode != Storage::Shared) {
        throw GpuError("Buffer::upload requires Storage::Shared.");
    }
    if (bytes > size_bytes()) {
        throw std::out_of_range("Buffer::upload: bytes > size_bytes()");
    }
    std::memcpy(data(), src, bytes);
}

void Buffer::set_label(const char* label) {
    if (impl_->buf && label) {
        impl_->buf->setLabel(NS::String::string(label, NS::UTF8StringEncoding));
    }
}

MTL::Buffer* Buffer::raw() const noexcept { return impl_->buf.get(); }

}  // namespace av::gpu
