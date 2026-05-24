// CoreMLSphereDetector — CoreML drop-in replacement for the upstream
// aliceVision_sphereDetection ONNX Runtime sphere detector.
//
// Loads a Vision-style YOLOv8 .mlpackage (with NMS baked into the model
// graph) and runs detection on a single image, returning bounding boxes
// in (x1, y1, x2, y2) IMAGE-PIXEL coordinates + per-detection scores,
// matching what upstream's predict() / sphereDetection() expects.
//
// The header is plain C++ so callers compile without Objective-C++.
// Implementation lives in CoreMLSphereDetector.mm.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace av {
namespace sphere {

struct Detection
{
    float x1 = 0.f;
    float y1 = 0.f;
    float x2 = 0.f;
    float y2 = 0.f;
    float score = 0.f;
};

struct DetectionResult
{
    std::vector<Detection> detections;
    int imageWidth = 0;
    int imageHeight = 0;
};

class CoreMLSphereDetector
{
public:
    // Load the CoreML .mlpackage at `mlpackagePath`. Uses
    // MLComputeUnits.all so the system picks ANE / GPU / CPU.
    // Throws std::runtime_error on load failure.
    explicit CoreMLSphereDetector(const std::string& mlpackagePath);
    ~CoreMLSphereDetector();

    CoreMLSphereDetector(const CoreMLSphereDetector&) = delete;
    CoreMLSphereDetector& operator=(const CoreMLSphereDetector&) = delete;

    // Run detection on the image at `imagePath`. Filters out detections
    // with score below `minScore`. NMS happens inside the model.
    // Throws std::runtime_error on read / inference failure.
    DetectionResult predict(const std::string& imagePath, float minScore);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

}  // namespace sphere
}  // namespace av
