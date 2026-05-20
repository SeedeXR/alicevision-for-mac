// DevicePatchPattern.h — Metal port of
// depthMap/cuda/device/DevicePatchPattern.hpp (+ .cu).
//
// Upstream layout (CUDA, file DevicePatchPattern.hpp):
//
//   struct DevicePatchPatternSubpart {
//       float2 coordinates[24];   // 192 B, align 8
//       int    nbCoordinates;     // 4 B
//       float  level;             // 4 B
//       float  downscale;         // 4 B
//       float  weight;            // 4 B
//       bool   isCircle;          // 1 B + 3 B tail padding
//       int    wsh;               // 4 B
//   };  // total = 216 B in the CUDA layout
//
//   struct DevicePatchPattern {
//       DevicePatchPatternSubpart subparts[4];
//       int nbSubparts;
//   };
//
// Translation notes for MSL:
//   * MSL's `bool` is 1 byte but interactions with host struct
//     padding are easy to get wrong across compilers / address
//     spaces. We use `int isCircle` here (and on the host) so the
//     layout is mechanically identical and there is no ambiguity.
//     The upstream `bool` was treated as a 0/non-zero predicate in
//     the kernel; an `int` 0/1 selector is bit-for-bit equivalent
//     under that contract.
//   * The CUDA `__constant__` symbol `constantPatchPattern_d` has
//     no direct analogue in MSL. Instead the kernel takes a
//     `constant DevicePatchPattern&` argument; the host binds the
//     struct via `set_bytes` (it is < 4 KB).

#pragma once

#include <metal_stdlib>
using namespace metal;

namespace av_depthmap {

// Match upstream's preprocessor constants.
constant constexpr int kPatchMaxSubparts          = 4;
constant constexpr int kPatchMaxCoordsPerSubpart  = 24;

// Subpart of a custom patch pattern. One similarity score per
// subpart; the kernel folds them by `weight`.
struct DevicePatchPatternSubpart
{
    float2 coordinates[kPatchMaxCoordsPerSubpart];  //< subpart coordinate list
    int    nbCoordinates;                            //< subpart number of coordinate
    float  level;                                    //< subpart related mipmap level (>=0)
    float  downscale;                                //< subpart related mipmap downscale (>=1)
    float  weight;                                   //< subpart related similarity weight in (0, 1)
    int    isCircle;                                 //< subpart is a circle (0/1; matches upstream bool)
    int    wsh;                                      //< subpart half-width (full and circle)
};

// Custom patch pattern (up to kPatchMaxSubparts subparts).
struct DevicePatchPattern
{
    DevicePatchPatternSubpart subparts[kPatchMaxSubparts];  //< pattern subparts
    int                       nbSubparts;                   //< number of subparts (>0)
};

}  // namespace av_depthmap
