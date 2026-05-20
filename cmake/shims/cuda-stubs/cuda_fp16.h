#pragma once

// cuda_fp16.h — Apple Silicon stub. Upstream's
// `depthMap/cuda/planeSweeping/similarity.hpp` does
// `#include <cuda_fp16.h>` to pick up `__half`, the CUDA 16-bit
// floating-point host/device type. On macOS we have no CUDA
// toolchain, so we provide a binary-compatible stand-in:
//
//   * Apple Clang on arm64 supports the `__fp16` storage-only
//     extension type. It is 16-bit, IEEE 754 binary16-formatted,
//     and convertible to/from `float` for arithmetic.
//   * That matches CUDA's `__half` host-side behavior (host-side
//     `__half` is also storage-only without explicit intrinsics).
//
// The upstream depthMap host code uses `__half` only as a TSimRefine
// type tag inside `CudaDeviceMemoryPitched<__half, 3>` allocations
// and the occasional float→half conversion. Arithmetic happens
// inside our Metal kernels (MSL `half`), not on host.

#if defined(__clang__) && defined(__aarch64__)
// Use C++23 `_Float16` (a true arithmetic type), NOT `__fp16` (ARM
// storage-only). Our adapter at `src/depth_map_metal/include/av/depth_map/
// upstream_adapter.hpp:53` exports symbols mangled with `_Float16`; using
// `__fp16` here would mangle to "half" (Dh) and break linkage.
typedef _Float16 __half;
#else
// Fallback: a 16-bit POD with no operators. CudaDeviceMemoryPitched
// only needs `sizeof(__half) == 2`. Anyone who tries arithmetic on
// host-side `__half` outside arm64 Clang will get a compiler error,
// which is the intended outcome (host arithmetic shouldn't happen).
struct __half { unsigned short __bits; };
#endif

// CUDA's `__float2half` / `__half2float` host-side helpers. Used by
// upstream's `cuda/host/DeviceCache.cpp` (in the uchar→half image
// conversion path; we don't ship that variant) and a small number
// of depthMap host call sites. Provide inline implementations.
static inline __half __float2half(float f) {
#if defined(__clang__) && defined(__aarch64__)
    return (__half)f;
#else
    // IEEE 754 binary32 → binary16 with round-to-nearest-even.
    unsigned int x;
    __builtin_memcpy(&x, &f, 4);
    unsigned int sign = (x >> 16) & 0x8000;
    int          exp  = (int)((x >> 23) & 0xff) - 127 + 15;
    unsigned int mant = x & 0x7fffff;
    unsigned short h;
    if (exp <= 0) {
        h = (unsigned short)sign;
    } else if (exp >= 31) {
        h = (unsigned short)(sign | 0x7c00 | (mant ? 0x200 : 0));
    } else {
        unsigned int round = (mant >> 12) & 1;
        h = (unsigned short)(sign | ((unsigned int)exp << 10) | (mant >> 13));
        h = (unsigned short)(h + round);
    }
    __half r;
    r.__bits = h;
    return r;
#endif
}

static inline float __half2float(__half h) {
#if defined(__clang__) && defined(__aarch64__)
    return (float)h;
#else
    unsigned short hb = h.__bits;
    unsigned int   sign = ((unsigned int)hb & 0x8000) << 16;
    unsigned int   exp  = (hb >> 10) & 0x1f;
    unsigned int   mant = hb & 0x3ff;
    unsigned int   x;
    if (exp == 0) {
        x = sign;
    } else if (exp == 31) {
        x = sign | 0x7f800000 | (mant << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    __builtin_memcpy(&f, &x, 4);
    return f;
#endif
}
