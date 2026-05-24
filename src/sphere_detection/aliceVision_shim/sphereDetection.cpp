// Drop-in replacement for upstream's
// upstream/src/aliceVision/sphereDetection/sphereDetection.cpp.
//
// Routes inference through av::sphere::CoreMLSphereDetector instead of
// Ort::Session; otherwise preserves the JSON writers and the
// auto-detect outer loop verbatim.

#include <aliceVision/sphereDetection/sphereDetection.hpp>

#include <aliceVision/utils/convert.hpp>
#include <aliceVision/image/Image.hpp>
#include <aliceVision/image/io.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <boost/property_tree/json_parser.hpp>
#include <boost/algorithm/string.hpp>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

namespace aliceVision {
namespace sphereDetection {

void fillShapeTree(bpt::ptree& fileTree, const bpt::ptree& spheresTree)
{
    bpt::ptree shapesTree;
    {
        bpt::ptree shapeTree;
        shapeTree.put("name", "Manual Sphere Detection");
        shapeTree.put("type", "Circle");

        bpt::ptree shapeProperties;
        shapeProperties.put("color", "green");
        shapeTree.add_child("properties", shapeProperties);

        shapeTree.add_child("observations", spheresTree);
        shapesTree.push_back(std::make_pair("", shapeTree));
    }
    fileTree.add_child("shapes", shapesTree);
}

void sphereDetection(const sfmData::SfMData& sfmData,
                     av::sphere::CoreMLSphereDetector& detector,
                     fs::path outputPath,
                     const float minScore)
{
    bpt::ptree spheresTree;

    for (auto& viewID : sfmData.getViews())
    {
        ALICEVISION_LOG_DEBUG("View Id: " << viewID);

        const std::string sphereName = std::to_string(viewID.second->getViewId());
        const fs::path imagePath = fs::path(
            sfmData.getView(viewID.second->getViewId()).getImage().getImagePath());

        if (boost::algorithm::icontains(imagePath.stem().string(), "ambient"))
            continue;

        av::sphere::DetectionResult pred;
        try
        {
            pred = detector.predict(imagePath.string(), minScore);
        }
        catch (const std::exception& e)
        {
            ALICEVISION_LOG_WARNING("CoreML inference failed for '"
                                   << imagePath << "': " << e.what());
            continue;
        }

        if (!pred.detections.empty())
        {
            // Highest-score detection (predict() sorts).
            const auto& det = pred.detections.front();
            const float r = std::min(det.y2 - det.y1, det.x2 - det.x1) / 2.f;
            const float x = det.x1 + r;
            const float y = det.y1 + r;

            bpt::ptree sphereNode;
            sphereNode.put("center.x", x);
            sphereNode.put("center.y", y);
            sphereNode.put("radius", r);
            sphereNode.put("score", det.score);
            sphereNode.put("type", "matte");

            spheresTree.add_child(sphereName, sphereNode);
        }
        else
        {
            ALICEVISION_LOG_WARNING("No sphere detected for '" << imagePath << "'.");
        }
    }

    bpt::ptree fileTree;
    fillShapeTree(fileTree, spheresTree);
    bpt::write_json(outputPath.string(), fileTree);
}

bool writeManualSphereJSON(const sfmData::SfMData& sfmData,
                           const std::vector<std::string>& x,
                           const std::vector<std::string>& y,
                           const std::vector<std::string>& radius,
                           fs::path outputPath,
                           bool fillMissingSpheres)
{
    auto xValues = aliceVision::utils::dictStringToStringMap(x);
    auto yValues = aliceVision::utils::dictStringToStringMap(y);
    auto radiusValues = aliceVision::utils::dictStringToStringMap(radius);

    bpt::ptree spheresTree;

    for (auto& viewID : sfmData.getViews())
    {
        ALICEVISION_LOG_DEBUG("View ID: " << viewID);
        const std::string sphereName = std::to_string(viewID.second->getViewId());

        std::vector<float> sphereParams;
        auto pos = xValues.find(sphereName);
        if (pos == xValues.end())
        {
            ALICEVISION_LOG_INFO("Sphere shape for view ID " << sphereName << " not found.");

            if (fillMissingSpheres && !xValues.empty())
            {
                ALICEVISION_LOG_INFO("Using sphere position from view ID "
                                     << xValues.rbegin()->first << ".");
                sphereParams = {std::stof(xValues.rbegin()->second),
                                std::stof(yValues.rbegin()->second),
                                std::stof(radiusValues.rbegin()->second)};
            }
        }
        else
        {
            ALICEVISION_LOG_DEBUG("Sphere shape for view ID " << sphereName << " found.");
            sphereParams = {std::stof(xValues.at(sphereName)),
                            std::stof(yValues.at(sphereName)),
                            std::stof(radiusValues.at(sphereName))};
        }

        if (!sphereParams.empty())
        {
            bpt::ptree sphereNode;
            sphereNode.put("center.x", sphereParams[0]);
            sphereNode.put("center.y", sphereParams[1]);
            sphereNode.put("radius", sphereParams[2]);
            sphereNode.put("type", "matte");
            spheresTree.add_child(sphereName, sphereNode);
        }
    }

    bpt::ptree fileTree;
    fillShapeTree(fileTree, spheresTree);
    bpt::write_json(outputPath.string(), fileTree);
    return true;
}

bool writeManualSphereJSON(const sfmData::SfMData& sfmData,
                           const std::string& sphereFile,
                           const std::string& outputPath,
                           bool fillMissingSpheres)
{
    if (!fillMissingSpheres)
    {
        fs::copy_file(sphereFile, outputPath);
        return true;
    }

    bpt::ptree fileTree;
    bpt::read_json(sphereFile, fileTree);

    bpt::ptree spheresTree;
    const auto shapesTreeOpt = fileTree.get_child_optional("shapes");
    if (shapesTreeOpt && !shapesTreeOpt->empty())
    {
        const auto& firstShapeTree = shapesTreeOpt->begin()->second;
        spheresTree = firstShapeTree.get_child("observations");
    }
    else
    {
        ALICEVISION_THROW_ERROR("Cannot find sphere detection data in '" << sphereFile << "'.");
    }

    std::string lastSphereViewID = spheresTree.rbegin()->first;
    std::vector<float> sphereParams = {
        spheresTree.rbegin()->second.get("center.x", 0.0f),
        spheresTree.rbegin()->second.get("center.y", 0.0f),
        spheresTree.rbegin()->second.get("radius", 0.0f)};

    ALICEVISION_LOG_INFO("Got last known sphere position: " << lastSphereViewID);

    for (auto& viewID : sfmData.getViews())
    {
        ALICEVISION_LOG_DEBUG("View ID: " << viewID);
        const std::string sphereName = std::to_string(viewID.second->getViewId());

        auto sphereExists = (spheresTree.get_child_optional(sphereName)).is_initialized();
        if (!sphereExists)
        {
            ALICEVISION_LOG_INFO("Sphere exists");
            bpt::ptree sphereNode;
            sphereNode.put("center.x", sphereParams[0]);
            sphereNode.put("center.y", sphereParams[1]);
            sphereNode.put("radius", sphereParams[2]);
            sphereNode.put("type", "matte");
            spheresTree.add_child(sphereName, sphereNode);
        }
    }

    fileTree.clear();
    fillShapeTree(fileTree, spheresTree);
    bpt::write_json(outputPath, fileTree);
    return true;
}

}  // namespace sphereDetection
}  // namespace aliceVision
