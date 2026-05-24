// Drop-in replacement for upstream's
// upstream/src/aliceVision/sphereDetection/sphereDetection.hpp.
//
// Same namespace + same function signatures EXCEPT the ONNXRuntime
// session is swapped for the av::sphere::CoreMLSphereDetector wrapper.
// modelExplore() is removed (it was an ONNX I/O dump and has no
// meaningful CoreML equivalent — Vision YOLOv8 metadata is exposed via
// MLModel.modelDescription if needed at runtime).
//
// This header MUST appear before upstream's on the include search path
// so the binary + library see this version.

#pragma once

#include <av/sphere/CoreMLSphereDetector.hpp>

// OpenCV (kept — still used for Prediction::size)
#include <opencv2/opencv.hpp>

// Boost Property Tree
#include <boost/property_tree/ptree.hpp>

// SFMData
#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <filesystem>

namespace aliceVision {
namespace sphereDetection {

namespace fs = std::filesystem;
namespace bpt = boost::property_tree;

struct Prediction
{
    std::vector<std::vector<float>> bboxes;
    std::vector<float> scores;
    cv::Size size;
};

void fillShapeTree(bpt::ptree& fileTree, const bpt::ptree& spheresTree);

void sphereDetection(const sfmData::SfMData& sfmData,
                     av::sphere::CoreMLSphereDetector& detector,
                     fs::path outputPath,
                     const float minScore);

bool writeManualSphereJSON(const sfmData::SfMData& sfmData,
                           const std::vector<std::string>& x,
                           const std::vector<std::string>& y,
                           const std::vector<std::string>& radius,
                           fs::path outputPath,
                           bool fillMissingSpheres);

bool writeManualSphereJSON(const sfmData::SfMData& sfmData,
                           const std::string& sphereFile,
                           const std::string& outputPath,
                           bool fillMissingSpheres);

}  // namespace sphereDetection
}  // namespace aliceVision
