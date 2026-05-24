// CoreMLMoGeRunner — CoreML wrapper around the user's MoGe-2 .mlpackage
// (DINOv2 ViT-B/14, exported at 504×672 fixed input resolution).
//
// Model contract (verified at ai-models/moge2_504x672_t1728.mlpackage):
//   inputs:
//     image          : MultiArray [1, 3, 504, 672] float32, values in [0,1]
//   outputs:
//     points         : MultiArray [1, 504, 672, 3] float32 — per-pixel XYZ
//                      in MoGe's relative reference frame (Z = forward).
//     normal         : MultiArray [1, 504, 672, 3] float32 — per-pixel
//                      surface normal in the same frame.
//     mask           : MultiArray [1, 504, 672]    float32 — validity
//                      (1.0 valid, 0.0 invalid).
//     metric_scale   : MultiArray [1]              float32 — scalar
//                      multiplier to convert relative depth to meters.
//
// Runs on MLComputeUnits.all so the system picks ANE / GPU / CPU. User's
// measurement (2026-05-24): ~228 ms on ANE vs ~384 ms on CPU on M-series
// (the model partially runs on ANE).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace av {
namespace moge {

// Native MoGe output dimensions (model-fixed).
inline constexpr int kModelWidth = 672;
inline constexpr int kModelHeight = 504;

struct MoGeResult
{
    int width = 0;   // = kModelWidth
    int height = 0;  // = kModelHeight

    // depthMeters[y * width + x] — Z component of points * metric_scale.
    std::vector<float> depthMeters;

    // normalXYZ[(y * width + x) * 3 + c] — surface normal as a unit vec.
    std::vector<float> normalXYZ;

    // mask[y * width + x] — 0 (invalid) or 1 (valid).
    std::vector<uint8_t> mask;

    float metricScale = 1.f;

    // Original image dimensions (informational; the maps above are in
    // the model's native resolution, NOT resampled to origW/origH).
    int origImageWidth = 0;
    int origImageHeight = 0;
};

class CoreMLMoGeRunner
{
public:
    // Load the CoreML .mlpackage at `mlpackagePath`. Uses
    // MLComputeUnits.all (ANE/GPU/CPU). Throws std::runtime_error on
    // load failure.
    explicit CoreMLMoGeRunner(const std::string& mlpackagePath);
    ~CoreMLMoGeRunner();

    CoreMLMoGeRunner(const CoreMLMoGeRunner&) = delete;
    CoreMLMoGeRunner& operator=(const CoreMLMoGeRunner&) = delete;

    // Run MoGe inference on the image at `imagePath`. Caller is
    // responsible for resampling the returned maps to their target
    // resolution (the model emits at 504×672 always; AliceVision's
    // DepthMapTracksInjecting consumes depth at whatever resolution
    // the file ships at).
    MoGeResult predict(const std::string& imagePath);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}  // namespace moge
}  // namespace av
