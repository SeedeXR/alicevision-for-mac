#pragma once

// memory.hpp — Apple Silicon type-shim. Replaces upstream's
// `aliceVision/depthMap/cuda/host/memory.hpp` (~967 LOC,
// CUDA-heavy) with a minimal subset backed by our
// `av::gpu::Buffer` on Apple's unified-memory architecture.
//
// This file is placed at `cmake/shims/aliceVision-includes/...`
// and added to the upstream-depthMap target's include path
// `BEFORE` upstream's `src/` so an `#include
// <aliceVision/depthMap/cuda/host/memory.hpp>` from upstream's
// host code resolves to this shim instead.
//
// Audited surface (grep across `upstream/src/aliceVision/depthMap/*.cpp,hpp`):
//
//   CudaSize<2/3>:        ctor(s0, s1[, s2]), [i], x(), y(), z(),
//                         operator+, operator-, operator==/!=
//   CudaDeviceMemoryPitched<T, N>:
//       ctor(size), allocate(size), getSize(), getPitch(),
//       getBytesPadded(), getBytesUnpadded(), getBuffer(),
//       copyFrom(other, stream=0)
//   CudaHostMemoryHeap<T, N>:
//       same surface; backing is std::vector<T>.
//
// Anything else upstream defines but our host code doesn't call
// is omitted to keep the shim auditable. If a future depthMap
// version starts calling a new method, we add it here.
//
// CUDA primitive types not on Apple: define POD substitutes for
// `float2`, `float3`, `float4`, `uchar4`, `__half`, `cudaStream_t`.
// These match CUDA's binary layout (Metal's `packed_float3` is a
// distinct concept used only inside kernels; here we mirror the
// host-side CUDA types).

#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"

#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

// -------- CUDA primitive type stand-ins --------
// These are 1-to-1 with CUDA's host-side vector types
// (`<vector_types.h>`). Trivially-constructible POD aggregates.
#if !defined(__CUDACC__) && !defined(AV_ADAPTER_CUDA_TYPES_DEFINED)
#define AV_ADAPTER_CUDA_TYPES_DEFINED 1

struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct uchar4 { unsigned char x, y, z, w; };

// CUDA host-side factory functions. `make_floatN`/`make_uchar4` are
// declared in `<vector_functions.h>`; upstream's host code uses them
// for terse aggregate construction.
inline float2 make_float2(float x, float y)                { return {x, y};       }
inline float3 make_float3(float x, float y, float z)       { return {x, y, z};    }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline uchar4 make_uchar4(unsigned char x, unsigned char y,
                          unsigned char z, unsigned char w) { return {x, y, z, w}; }

// `cudaStream_t` is a pointer in CUDA; we never honor stream
// semantics from upstream callers (everything dispatches on
// our default queue). Define as `void*` so `cuda_*` signatures
// remain compatible.
using cudaStream_t = void*;

// `cudaError_t` + `cudaDeviceSynchronize` stubs. Upstream's
// `DepthMapEstimator.cpp` calls `cudaDeviceSynchronize()` in two
// places to drain all pending kernel work before a save. On Apple
// Silicon, every dispatch waits via `commit_and_wait` already, so
// there's nothing to drain — make this a no-op returning success.
typedef int cudaError_t;
inline cudaError_t cudaDeviceSynchronize() { return 0; }

#endif  // AV_ADAPTER_CUDA_TYPES_DEFINED

namespace aliceVision {
namespace depthMap {

// Match upstream's `ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR/HALF`
// macro toggles: default branch is float4 (the production
// upstream config on Linux uses this).
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR
using CudaColorBaseType = unsigned char;
using CudaRGBA          = uchar4;
#else
using CudaColorBaseType = float;
using CudaRGBA          = float4;
#endif

// ============================================================
// CudaSize<N> — N-dim size with operator[] / x() / y() / z()
// ============================================================

// Note on operator+/- and arithmetic operators on CudaSize: the
// upstream header defines them but the host code audit
// (grep across `upstream/src/aliceVision/depthMap/*.{cpp,hpp}`)
// shows no callers — so this shim omits them to keep the surface
// small. If a future upstream version starts using them, port
// the upstream form into a free-standing `operator-` etc. so
// the return type is `CudaSize<N>` (not `CudaSizeBase<N>`).

template<unsigned Dim>
class CudaSizeBase {
public:
    CudaSizeBase() {
        for (unsigned i = 0; i < Dim; ++i) size[i] = 0;
    }
    inline std::size_t  operator[](std::size_t i) const { return size[i]; }
    inline std::size_t& operator[](std::size_t i)       { return size[i]; }

protected:
    std::size_t size[Dim];
};

template<unsigned Dim> bool
operator==(const CudaSizeBase<Dim>& a, const CudaSizeBase<Dim>& b) {
    for (unsigned i = 0; i < Dim; ++i) if (a[i] != b[i]) return false;
    return true;
}
template<unsigned Dim> bool
operator!=(const CudaSizeBase<Dim>& a, const CudaSizeBase<Dim>& b) {
    return !(a == b);
}

template<unsigned Dim>
class CudaSize : public CudaSizeBase<Dim> { public: CudaSize() {} };

template<>
class CudaSize<1> : public CudaSizeBase<1> {
public:
    CudaSize() {}
    explicit CudaSize(std::size_t s0) { size[0] = s0; }
    inline std::size_t x() const { return size[0]; }
};

template<>
class CudaSize<2> : public CudaSizeBase<2> {
public:
    CudaSize() {}
    CudaSize(std::size_t s0, std::size_t s1) { size[0] = s0; size[1] = s1; }
    inline std::size_t x() const { return size[0]; }
    inline std::size_t y() const { return size[1]; }
};

template<>
class CudaSize<3> : public CudaSizeBase<3> {
public:
    CudaSize() {}
    CudaSize(std::size_t s0, std::size_t s1, std::size_t s2) {
        size[0] = s0; size[1] = s1; size[2] = s2;
    }
    inline std::size_t x() const { return size[0]; }
    inline std::size_t y() const { return size[1]; }
    inline std::size_t z() const { return size[2]; }
};

// ============================================================
// Shared allocator — picks up the process-global Device once
// `set_adapter_device()` has been called by the host code.
// On Apple UMA, "device" memory is `av::gpu::Buffer` (Shared
// storage). "Host" memory is std::vector.
// ============================================================

namespace av_shim_detail {

// Pointer to the Device used to back CudaDeviceMemoryPitched.
// Owned externally (e.g., by `DepthMapEstimator`'s startup
// code). The shim calls `*get_device()` on each allocate.
inline av::gpu::Device*& current_device() {
    static av::gpu::Device* p = nullptr;
    return p;
}

}  // namespace av_shim_detail

// Adapter glue. Call this once at startup (before any
// `CudaDeviceMemoryPitched` constructor runs) to wire in the
// process-global Metal device.
inline void set_adapter_device(av::gpu::Device& dev) {
    av_shim_detail::current_device() = &dev;
}

inline av::gpu::Device& require_adapter_device() {
    auto* p = av_shim_detail::current_device();
    if (!p) {
        // Auto-init from system default Metal device. Stored in a
        // function-local static so its lifetime spans the process.
        // Upstream CLI binaries don't call `set_adapter_device()`
        // explicitly — they expect the device to "just exist". Also
        // load `default.metallib` from next to the executable, so the
        // kernels are available for `Device::make_pipeline()`.
        static av::gpu::Device s_default = []() {
            auto d = av::gpu::Device::default_device();
            d.load_library({});
            return d;
        }();
        set_adapter_device(s_default);
        return s_default;
    }
    if (false) {
        throw std::runtime_error(
            "aliceVision::depthMap::CudaDeviceMemoryPitched: "
            "set_adapter_device(av::gpu::Device&) must be called "
            "before any device allocation.");
    }
    return *p;
}

// ============================================================
// CudaDeviceMemoryPitched<T, N>
// Backed by av::gpu::Buffer (Shared storage).
// On UMA: pitch == size[0] * sizeof(T); no padding.
// ============================================================

template<class T, unsigned Dim>
class CudaDeviceMemoryPitched {
public:
    CudaDeviceMemoryPitched() = default;

    explicit CudaDeviceMemoryPitched(const CudaSize<Dim>& s) { allocate(s); }

    CudaDeviceMemoryPitched(const CudaDeviceMemoryPitched&)            = delete;
    CudaDeviceMemoryPitched& operator=(const CudaDeviceMemoryPitched&) = delete;
    CudaDeviceMemoryPitched(CudaDeviceMemoryPitched&&) noexcept            = default;
    CudaDeviceMemoryPitched& operator=(CudaDeviceMemoryPitched&&) noexcept = default;

    void allocate(const CudaSize<Dim>& s) {
        _size  = s;
        _pitch = s[0] * sizeof(T);
        std::size_t prod = 1;
        for (unsigned i = 0; i < Dim; ++i) prod *= s[i];
        const std::size_t bytes = prod * sizeof(T);
        _buf = std::make_unique<av::gpu::Buffer>(
            require_adapter_device(), bytes);
    }

    inline const CudaSize<Dim>& getSize() const         { return _size;  }
    inline std::size_t           getPitch() const       { return _pitch; }
    inline std::size_t           getBytesPadded() const {
        std::size_t prod = _pitch;
        for (unsigned i = 1; i < Dim; ++i) prod *= _size[i];
        return prod;
    }
    inline std::size_t getBytesUnpadded() const { return getBytesPadded(); }
    inline std::size_t getUnitsTotal() const {
        std::size_t prod = 1;
        for (unsigned i = 0; i < Dim; ++i) prod *= _size[i];
        return prod;
    }
    // Bytes for an n-dim slice starting at dim 0; "dim" itself is included.
    inline std::size_t getBytesPaddedUpToDim(int dim) const {
        std::size_t prod = _pitch;
        for (int i = 1; i <= dim; ++i) prod *= _size[i];
        return prod;
    }

    // Indexed access. operator()(x) is linear; operator()(x,y) walks
    // pitch-bytes per row. Both match upstream's host-side API.
    inline T& operator()(std::size_t x) {
        return reinterpret_cast<T*>(data())[x];
    }
    inline const T& operator()(std::size_t x) const {
        return reinterpret_cast<const T*>(data())[x];
    }
    inline T& operator()(std::size_t x, std::size_t y) {
        auto* p = reinterpret_cast<unsigned char*>(data()) + y * _pitch;
        return reinterpret_cast<T*>(p)[x];
    }
    inline const T& operator()(std::size_t x, std::size_t y) const {
        auto* p = reinterpret_cast<const unsigned char*>(data()) + y * _pitch;
        return reinterpret_cast<const T*>(p)[x];
    }
    inline unsigned char* getBytePtr() {
        return reinterpret_cast<unsigned char*>(data());
    }
    inline const unsigned char* getBytePtr() const {
        return reinterpret_cast<const unsigned char*>(data());
    }

    // Returns T*; backed by the av::gpu::Buffer's shared-storage
    // host pointer. Same address visible to both CPU and GPU on
    // Apple Silicon UMA.
    inline T*       getBuffer()       { return reinterpret_cast<T*>(data()); }
    inline const T* getBuffer() const { return reinterpret_cast<const T*>(data()); }

    // Adapter-only extension: hand out the underlying Buffer&
    // for direct use with our Metal `set_buffer` calls. Not
    // present in upstream's API.
    inline av::gpu::Buffer&       gpu_buffer()       { return *_buf; }
    inline const av::gpu::Buffer& gpu_buffer() const { return *_buf; }

    // copyFrom: host→device, device→device, or device→device.
    // On UMA all three reduce to memcpy on the shared backing.
    template<class Other>
    void copyFrom(const Other& other, cudaStream_t /*stream*/ = nullptr) {
        if (!_buf) allocate(other.getSize());
        if (other.getSize() != _size) {
            throw std::runtime_error(
                "CudaDeviceMemoryPitched::copyFrom: size mismatch");
        }
        std::memcpy(data(), other.dataRaw(), getBytesPadded());
    }

    // Internal: byte pointer to the start of the backing memory.
    void*       data()       { return _buf ? _buf->data() : nullptr; }
    const void* data() const { return _buf ? _buf->data() : nullptr; }

    // For copyFrom-from-this — accessor used by other shims.
    const void* dataRaw() const { return data(); }

private:
    CudaSize<Dim> _size{};
    std::size_t   _pitch = 0;
    std::unique_ptr<av::gpu::Buffer> _buf;
};

// ============================================================
// CudaHostMemoryHeap<T, N>
// Backed by std::vector. Same surface as the device variant.
// On UMA, host vs device is purely conceptual — the data lives
// in the same memory either way.
// ============================================================

template<class T, unsigned Dim>
class CudaHostMemoryHeap {
public:
    CudaHostMemoryHeap() = default;
    explicit CudaHostMemoryHeap(const CudaSize<Dim>& s) { allocate(s); }

    CudaHostMemoryHeap(const CudaHostMemoryHeap&)            = delete;
    CudaHostMemoryHeap& operator=(const CudaHostMemoryHeap&) = delete;
    CudaHostMemoryHeap(CudaHostMemoryHeap&&) noexcept            = default;
    CudaHostMemoryHeap& operator=(CudaHostMemoryHeap&&) noexcept = default;

    void allocate(const CudaSize<Dim>& s) {
        _size  = s;
        _pitch = s[0] * sizeof(T);
        std::size_t prod = 1;
        for (unsigned i = 0; i < Dim; ++i) prod *= s[i];
        _vec.assign(prod, T{});
    }

    inline const CudaSize<Dim>& getSize() const         { return _size;  }
    inline std::size_t           getPitch() const       { return _pitch; }
    inline std::size_t           getBytesPadded() const {
        std::size_t prod = _pitch;
        for (unsigned i = 1; i < Dim; ++i) prod *= _size[i];
        return prod;
    }
    inline std::size_t getBytesUnpadded() const { return getBytesPadded(); }
    inline std::size_t getUnitsTotal() const { return _vec.size(); }
    inline std::size_t getBytesPaddedUpToDim(int dim) const {
        std::size_t prod = _pitch;
        for (int i = 1; i <= dim; ++i) prod *= _size[i];
        return prod;
    }

    inline T*       getBuffer()       { return _vec.data(); }
    inline const T* getBuffer() const { return _vec.data(); }

    // Indexed access (matches upstream's host-side API).
    inline T& operator()(std::size_t x) { return _vec[x]; }
    inline const T& operator()(std::size_t x) const { return _vec[x]; }
    inline T& operator()(std::size_t x, std::size_t y) {
        auto* p = reinterpret_cast<unsigned char*>(_vec.data()) + y * _pitch;
        return reinterpret_cast<T*>(p)[x];
    }
    inline const T& operator()(std::size_t x, std::size_t y) const {
        auto* p = reinterpret_cast<const unsigned char*>(_vec.data()) + y * _pitch;
        return reinterpret_cast<const T*>(p)[x];
    }
    inline unsigned char* getBytePtr() {
        return reinterpret_cast<unsigned char*>(_vec.data());
    }
    inline const unsigned char* getBytePtr() const {
        return reinterpret_cast<const unsigned char*>(_vec.data());
    }

    template<class Other>
    void copyFrom(const Other& other, cudaStream_t /*stream*/ = nullptr) {
        if (_vec.empty()) allocate(other.getSize());
        if (other.getSize() != _size) {
            throw std::runtime_error(
                "CudaHostMemoryHeap::copyFrom: size mismatch");
        }
        std::memcpy(_vec.data(), other.dataRaw(), getBytesPadded());
    }

    const void* dataRaw() const { return _vec.data(); }

private:
    CudaSize<Dim>  _size{};
    std::size_t    _pitch = 0;
    std::vector<T> _vec;
};

}  // namespace depthMap
}  // namespace aliceVision
