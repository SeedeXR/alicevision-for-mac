#pragma once

// DeviceCameraParams.hpp — Apple Silicon shim. Replaces upstream's
// `cuda/device/DeviceCameraParams.hpp` which declares an
// `extern __constant__ DeviceCameraParams constantCameraParametersArray_d[]`.
// Apple Clang has no `__constant__`. The struct itself is plain POD and
// usable as-is; we drop the constant-array declaration and keep the
// `ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS` macro that upstream's
// `DepthMapEstimator.cpp` uses for batching arithmetic.

// memory.hpp's CUDA-type stand-ins provide `float3`.
#include <aliceVision/depthMap/cuda/host/memory.hpp>

namespace aliceVision {
namespace depthMap {

struct DeviceCameraParams
{
    float P[12];
    float iP[9];
    float R[9];
    float iR[9];
    float K[9];
    float iK[9];
    float3 C;
    float3 XVect;
    float3 YVect;
    float3 ZVect;
};

// Match upstream's value verbatim. On Apple Silicon there is no
// CUDA constant-memory ceiling, but the macro is also used by
// `DepthMapEstimator` to limit per-batch camera-param counts —
// keep the same cap so batching behaviour is identical.
#define ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS 100

}  // namespace depthMap
}  // namespace aliceVision
