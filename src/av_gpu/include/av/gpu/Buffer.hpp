#pragma once

#include <cstddef>
#include <cstring>
#include <span>
#include <memory>
#include <type_traits>

namespace MTL { class Buffer; }

namespace av::gpu {

class Device;

// Storage mode for buffers. On Apple Silicon prefer Shared.
enum class Storage : std::uint8_t {
    Shared,   // CPU+GPU coherent; default on Apple Silicon UMA.
    Private,  // GPU-only; reading from CPU requires a staging buffer.
};

// RAII handle on an MTL::Buffer. The Shared variant exposes a
// pointer to the contents — same pointer the GPU sees (UMA).
class Buffer {
public:
    // Allocate an uninitialized buffer. Throws GpuError on failure.
    Buffer(const Device& dev, std::size_t bytes,
           Storage mode = Storage::Shared);

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;
    ~Buffer();

    std::size_t size_bytes() const noexcept;
    Storage     storage()    const noexcept;

    // Pointer to contents for Storage::Shared. Returns nullptr for
    // Storage::Private.
    void*       data()       noexcept;
    const void* data() const noexcept;

    // Typed views for Shared buffers.
    template <class T>
    std::span<T> as_span() noexcept {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Buffer::as_span<T> requires trivially-copyable T");
        return { static_cast<T*>(data()), size_bytes() / sizeof(T) };
    }
    template <class T>
    std::span<const T> as_span() const noexcept {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Buffer::as_span<T> requires trivially-copyable T");
        return { static_cast<const T*>(data()), size_bytes() / sizeof(T) };
    }

    // Copy from a host span into a Shared buffer (memcpy).
    template <class T>
    void upload(std::span<const T> src) {
        // Throws if mode != Shared or bytes mismatch.
        upload_bytes(src.data(), src.size_bytes());
    }

    void set_label(const char* label);

    MTL::Buffer* raw() const noexcept;

private:
    void upload_bytes(const void* src, std::size_t bytes);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::gpu
