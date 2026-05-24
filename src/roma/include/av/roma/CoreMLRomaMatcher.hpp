// CoreMLRomaMatcher — CoreML wrapper around TinyRoMa
// (ai-models/tiny_roma_v1_480x640.mlpackage).
//
// Production rec from ai-models/README.md (also verified on M-series):
// LOAD WITH MLComputeUnitsCPUAndGPU. Do NOT use .all — this model
// contains two `grid_sample` ops that force CPU↔ANE handoffs and the
// resulting plan is ~4× slower than CPU. GPU gives ~12 ms / pair at
// 480×640 (the other compute_units options ranged 21–84 ms).
//
// Model contract (verified at ai-models/tiny_roma_v1_480x640.mlpackage):
//   inputs:
//     im_A           : MultiArray [1, 3, 480, 640] float32, RGB in [0,1]
//     im_B           : MultiArray [1, 3, 480, 640] float32, RGB in [0,1]
//   outputs:
//     coarse_flow    : MultiArray [1, 2, 60,  80]  float32 — A→B flow at stride 8
//     coarse_certainty: MultiArray [1, 1, 60,  80] float32 — coarse logits
//     fine_flow      : MultiArray [1, 2, 120, 160] float32 — A→B flow at stride 4
//     fine_certainty : MultiArray [1, 1, 120, 160] float32 — fine logits
//
// Flow values are normalized to [-1, 1] over im_B. Convert to pixel
// coordinates with: x_B_pixels = (flow_x + 1) * W / 2.
// Certainty is unnormalized logits (apply sigmoid client-side if you
// want a probability-like value).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace av {
namespace roma {

// Native input dimensions (model-fixed by the conversion).
inline constexpr int kInputWidth = 640;
inline constexpr int kInputHeight = 480;

// Output dimensions.
inline constexpr int kCoarseWidth = 80;   // stride 8
inline constexpr int kCoarseHeight = 60;
inline constexpr int kFineWidth = 160;    // stride 4
inline constexpr int kFineHeight = 120;

struct RomaMatch
{
    // Original image dimensions (informational; flows are normalized to
    // [-1, 1] over im_B regardless of original resolution).
    int origImageAWidth = 0;
    int origImageAHeight = 0;
    int origImageBWidth = 0;
    int origImageBHeight = 0;

    // Coarse outputs, channel-first layout: [c, y, x].
    // flow has 2 channels (flow_x, flow_y); certainty has 1.
    std::vector<float> coarseFlow;       // size = 2 * kCoarseHeight * kCoarseWidth
    std::vector<float> coarseCertainty;  // size = kCoarseHeight * kCoarseWidth

    // Fine outputs, same layout.
    std::vector<float> fineFlow;         // size = 2 * kFineHeight * kFineWidth
    std::vector<float> fineCertainty;    // size = kFineHeight * kFineWidth
};

class CoreMLRomaMatcher
{
public:
    // Load the CoreML .mlpackage at `mlpackagePath`. Uses
    // MLComputeUnitsCPUAndGPU (NOT .all — ANE is a regression for this
    // model). Throws std::runtime_error on load failure.
    explicit CoreMLRomaMatcher(const std::string& mlpackagePath);
    ~CoreMLRomaMatcher();

    CoreMLRomaMatcher(const CoreMLRomaMatcher&) = delete;
    CoreMLRomaMatcher& operator=(const CoreMLRomaMatcher&) = delete;

    // Run TinyRoMa on a pair of images (paths). Both are read with
    // OpenCV, resized to 640×480 (model-native; aspect not preserved —
    // matches the conversion script which baked in a plain resize).
    RomaMatch match(const std::string& imageAPath, const std::string& imageBPath);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

// Convenience: sigmoid for converting certainty logits to probabilities.
inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

}  // namespace roma
}  // namespace av
