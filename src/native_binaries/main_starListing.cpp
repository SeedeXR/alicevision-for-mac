// alicevision-for-mac — Phase 14.7
// aliceVision_starListing — Mac-port native implementation.
//
// Upstream's 2026.1.0 release does NOT ship this binary in source; the
// reference pipeline performs the "star" image-pair listing through an
// external Python step. The Meshroom descriptor at
// `meshroom-mac/nodes/aliceVision/StarListing.py` declares the CLI
// surface; we mirror it here so the cameraTrackingRoma template can be
// driven by the same `aliceVision_starListing {allParams}` command line
// every other node uses.
//
// Algorithm (simple "next-K neighbours" star, deterministic):
//   1. Load SfMData (--input).
//   2. Sort view IDs ascending (so the topology is deterministic across
//      runs).
//   3. For each view at index i and every k in 1..radiusKeyFrames:
//        if i+k < N: emit "<viewId[i]> <viewId[i+k]>".
//   4. Copy the input SfMData to the pass-through output so downstream
//      RomaMatcher / MatchMasking can read it as
//      {StarListing_1.inputSfMData}.
//
// CLI args (from StarListing.py):
//   --input <SfMData>                  REQUIRED  full SfMData
//   --keySfMData <SfMData>             optional  keyframes-only SfMData
//                                                (accepted for descriptor
//                                                 parity; not used by
//                                                 this implementation —
//                                                 we star over *all*
//                                                 views, which is a
//                                                 strict superset of
//                                                 the upstream behaviour
//                                                 and keeps the matcher
//                                                 input format identical)
//   --radiusKeyFrames <int>            default 5
//   --output <SfMData>                 REQUIRED  pass-through SfMData
//   --outputPairsList <path>           REQUIRED  image-pairs .txt
//   --verboseLevel <level>             optional

#include <aliceVision/cmdline/cmdline.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/Timer.hpp>

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace aliceVision;

#define ALICEVISION_SOFTWARE_VERSION_MAJOR 1
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

int aliceVision_main(int argc, char** argv)
{
    system::Timer timer;

    std::string inputSfMDataPath;
    std::string keySfMDataPath;
    std::string outputSfMDataPath;
    std::string outputPairsListPath;
    int radiusKeyFrames = 5;

    // clang-format off
    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&inputSfMDataPath)->required(),
         "Input SfMData (full set of views).")
        ("output,o", po::value<std::string>(&outputSfMDataPath)->required(),
         "Output SfMData (pass-through copy of --input).")
        ("outputPairsList", po::value<std::string>(&outputPairsListPath)->required(),
         "Output image-pairs list file (text, one '<viewIdA> <viewIdB>' per line).");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("keySfMData", po::value<std::string>(&keySfMDataPath)->default_value(""),
         "Keyframe-only SfMData. Accepted for descriptor parity but not "
         "required by the Mac-port implementation (we star over all views).")
        ("radiusKeyFrames", po::value<int>(&radiusKeyFrames)->default_value(5),
         "For each view, pair it with the next N views in viewId order.");
    // clang-format on

    CmdLine cmdline("AliceVision starListing (Mac-port native impl.)");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);

    if (!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    if (radiusKeyFrames < 1)
    {
        ALICEVISION_LOG_WARNING("starListing: radiusKeyFrames=" << radiusKeyFrames
                                << " is invalid; clamping to 1.");
        radiusKeyFrames = 1;
    }

    // -------- Load SfMData --------
    sfmData::SfMData sfmData;
    if (!sfmDataIO::load(sfmData, inputSfMDataPath,
                         sfmDataIO::ESfMData(sfmDataIO::VIEWS | sfmDataIO::INTRINSICS)))
    {
        ALICEVISION_LOG_ERROR("starListing: cannot read input SfMData '"
                              << inputSfMDataPath << "'.");
        return EXIT_FAILURE;
    }

    // -------- Sort viewIds --------
    std::vector<IndexT> viewIds;
    viewIds.reserve(sfmData.getViews().size());
    for (const auto& kv : sfmData.getViews())
    {
        viewIds.push_back(kv.first);
    }
    std::sort(viewIds.begin(), viewIds.end());

    const std::size_t N = viewIds.size();
    ALICEVISION_LOG_INFO("starListing: loaded " << N << " views; radiusKeyFrames="
                         << radiusKeyFrames << ".");

    if (N < 2)
    {
        ALICEVISION_LOG_WARNING("starListing: fewer than 2 views — "
                                "emitting empty pairs list.");
    }

    // -------- Write image pairs --------
    {
        fs::path outDir = fs::path(outputPairsListPath).parent_path();
        if (!outDir.empty())
        {
            std::error_code ec;
            fs::create_directories(outDir, ec);  // ok if exists
        }
    }

    std::ofstream pairsOut(outputPairsListPath);
    if (!pairsOut.is_open())
    {
        ALICEVISION_LOG_ERROR("starListing: cannot write pairs file '"
                              << outputPairsListPath << "'.");
        return EXIT_FAILURE;
    }

    std::size_t numPairs = 0;
    for (std::size_t i = 0; i + 1 < N; ++i)
    {
        const std::size_t maxK = std::min<std::size_t>(
            static_cast<std::size_t>(radiusKeyFrames), N - 1 - i);
        for (std::size_t k = 1; k <= maxK; ++k)
        {
            pairsOut << viewIds[i] << ' ' << viewIds[i + k] << '\n';
            ++numPairs;
        }
    }
    pairsOut.close();

    ALICEVISION_LOG_INFO("starListing: wrote " << numPairs << " image pair(s) to '"
                         << outputPairsListPath << "'.");

    // -------- Pass-through SfMData --------
    {
        fs::path outDir = fs::path(outputSfMDataPath).parent_path();
        if (!outDir.empty())
        {
            std::error_code ec;
            fs::create_directories(outDir, ec);
        }
    }
    if (!sfmDataIO::save(sfmData, outputSfMDataPath, sfmDataIO::ESfMData::ALL))
    {
        ALICEVISION_LOG_ERROR("starListing: cannot write pass-through SfMData '"
                              << outputSfMDataPath << "'.");
        return EXIT_FAILURE;
    }
    ALICEVISION_LOG_INFO("starListing: wrote pass-through SfMData to '"
                         << outputSfMDataPath << "'.");

    ALICEVISION_LOG_INFO("starListing: done in (s): " + std::to_string(timer.elapsed()));
    return EXIT_SUCCESS;
}
