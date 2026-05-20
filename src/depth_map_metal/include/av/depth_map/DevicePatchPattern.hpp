#pragma once

// DevicePatchPattern — host mirror of the MSL `DevicePatchPattern`
// struct in src/shaders/depth_map/DevicePatchPattern.h.
//
// Layout MUST be byte-identical to the MSL side so the whole struct
// can be uploaded with `set_bytes` into a `constant` buffer binding
// (it is < 4 KB, so set_bytes is the right call here).
//
// Upstream parity note: the CUDA original declared `bool isCircle`,
// which on CUDA host/device works because both ends are nvcc-built.
// MSL's `bool` (1 B) vs C++ host `bool` (1 B, with implementation-
// defined padding) is layout-fragile across compilers, so we use
// `int isCircle` on both sides instead. Semantics (0 = false, non-
// zero = true) are unchanged.

#include <cstdint>

namespace av::depth_map {

// Match the upstream preprocessor constants.
inline constexpr int kPatchMaxSubparts         = 4;
inline constexpr int kPatchMaxCoordsPerSubpart = 24;

struct DevicePatchPatternSubpart {
    float coordinates[kPatchMaxCoordsPerSubpart][2]; // float2[24] — 192 B
    std::int32_t nbCoordinates;                       // 4 B
    float        level;                               // 4 B
    float        downscale;                           // 4 B
    float        weight;                              // 4 B
    std::int32_t isCircle;                            // 4 B (was `bool` upstream)
    std::int32_t wsh;                                 // 4 B
};
static_assert(sizeof(DevicePatchPatternSubpart) ==
                  (kPatchMaxCoordsPerSubpart * 2 + 6) * sizeof(float),
              "DevicePatchPatternSubpart must be tightly packed to mirror MSL.");

struct DevicePatchPattern {
    DevicePatchPatternSubpart subparts[kPatchMaxSubparts];
    std::int32_t              nbSubparts;
};
static_assert(sizeof(DevicePatchPattern) ==
                  kPatchMaxSubparts * sizeof(DevicePatchPatternSubpart)
                      + sizeof(std::int32_t),
              "DevicePatchPattern must be tightly packed to mirror MSL.");

}  // namespace av::depth_map
