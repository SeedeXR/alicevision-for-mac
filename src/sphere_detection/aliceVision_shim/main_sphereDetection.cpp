// Drop-in replacement for upstream's
// upstream/src/software/pipeline/main_sphereDetection.cpp.
//
// Same CLI surface as upstream; the only change is the auto-detect
// branch now creates an av::sphere::CoreMLSphereDetector instead of
// an Ort::Session.

#include <aliceVision/cmdline/cmdline.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/system/Logger.hpp>

#include <aliceVision/sphereDetection/sphereDetection.hpp>

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#define ALICEVISION_SOFTWARE_VERSION_MAJOR 2
#define ALICEVISION_SOFTWARE_VERSION_MINOR 1

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace aliceVision;

int aliceVision_main(int argc, char** argv)
{
    system::Timer timer;

    std::string inputSfMDataPath;
    std::string inputModelPath;
    std::string outputPath;
    float inputMinScore;

    bool autoDetect;
    std::vector<std::string> x, y, radius;
    std::string sphereFile;
    bool fillMissingSpheres;

    // clang-format off
    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&inputSfMDataPath)->required(),
         "SfMData input path.")
        ("modelPath,m", po::value<std::string>(&inputModelPath)->required(),
         "CoreML .mlpackage (or compiled .mlmodelc) input path.")
        ("autoDetect,a", po::value<bool>(&autoDetect)->required(),
         "True if the sphere is to be automatically detected, false otherwise.")
        ("output,o", po::value<std::string>(&outputPath)->required(),
         "Output path.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("minScore,s", po::value<float>(&inputMinScore)->default_value(0.0),
         "Minimum detection score.")
        ("x,x", po::value<std::vector<std::string>>(&x)->multitoken(),
         "Sphere's center X (pixels).")
        ("y,y", po::value<std::vector<std::string>>(&y)->multitoken(),
         "Sphere's center Y (pixels).")
        ("radius,r", po::value<std::vector<std::string>>(&radius)->multitoken(),
         "Sphere's radius (pixels).")
        ("sphereFile,f", po::value<std::string>(&sphereFile)->default_value(""),
         "File containing the positions for the spheres in all the images.")
        ("fillMissingSpheres,m", po::value<bool>(&fillMissingSpheres)->default_value(true),
         "True if a sphere position is to be written as detected although it was not "
         "provided. In that case, the position of the last known sphere will be used.");
    // clang-format on

    CmdLine cmdline("AliceVision sphereDetection");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);

    if (!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    sfmData::SfMData sfmData;
    if (!sfmDataIO::load(sfmData, inputSfMDataPath,
                         sfmDataIO::ESfMData(sfmDataIO::VIEWS | sfmDataIO::INTRINSICS)))
    {
        ALICEVISION_LOG_ERROR("The input file '" + inputSfMDataPath + "' cannot be read");
        return EXIT_FAILURE;
    }

    fs::path fsOutputPath(outputPath);

    if (autoDetect)
    {
        try
        {
            av::sphere::CoreMLSphereDetector detector(inputModelPath);
            sphereDetection::sphereDetection(sfmData, detector, fsOutputPath, inputMinScore);
        }
        catch (const std::exception& e)
        {
            ALICEVISION_LOG_ERROR("CoreML sphere detector init failed: " << e.what());
            return EXIT_FAILURE;
        }
    }
    else
    {
        if (sphereFile.empty())
        {
            sphereDetection::writeManualSphereJSON(sfmData, x, y, radius,
                                                   fsOutputPath, fillMissingSpheres);
        }
        else
        {
            if (!sphereDetection::writeManualSphereJSON(sfmData, sphereFile,
                                                        outputPath, fillMissingSpheres))
            {
                return EXIT_FAILURE;
            }
        }
    }

    ALICEVISION_LOG_INFO("Task done in (s): " + std::to_string(timer.elapsed()));
    return EXIT_SUCCESS;
}
