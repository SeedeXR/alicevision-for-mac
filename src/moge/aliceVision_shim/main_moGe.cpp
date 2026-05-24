// alicevision-for-mac — Phase 14.8
// aliceVision_moGe — Mac-port native CoreML binary.
//
// Background:
//   The 2026.1.0 cameraTrackingDepth.mg template introduces a
//   monocular-geometry depth prior arm built around the MoGe neural
//   network. Upstream does NOT ship a `aliceVision_moGe` C++ binary —
//   the reference runs MoGe externally via a Python pipeline.
//
//   This binary replaces the Phase 14.7 honest stub with a real
//   CoreML-backed implementation. The user converted MoGe-2 (DINOv2
//   ViT-B/14) to ai-models/moge2_504x672_t1728.mlpackage and verified
//   that it runs partially on ANE (~228 ms vs ~384 ms CPU). The
//   wrapper uses MLComputeUnits.all so the system schedules the parts
//   it can on ANE and falls back to GPU/CPU for the rest.
//
// Output convention (matches what DepthMapTracksInjecting consumes):
//   <output>/<viewId>_depth.exr     — single-channel float depth in meters
//   <output>/<viewId>_normals.exr   — RGB float per-pixel surface normal
//                                     (optional, only when --outputNormals=true)
//
// MoGe operates on 504×672 native — that's the resolution emitted by
// the EXR files. DepthMapTracksInjecting reads the file's intrinsic
// dimensions, so emitting at 504×672 is fine (the downstream node
// resamples to its consumer's needs).

#include <av/moge/CoreMLMoGeRunner.hpp>

#include <aliceVision/cmdline/cmdline.hpp>
#include <aliceVision/system/main.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/system/Timer.hpp>

#include <aliceVision/sfmData/SfMData.hpp>
#include <aliceVision/sfmDataIO/sfmDataIO.hpp>

#include <aliceVision/image/Image.hpp>
#include <aliceVision/image/io.hpp>
#include <aliceVision/image/pixelTypes.hpp>

#include <boost/program_options.hpp>

#include <filesystem>
#include <string>

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace aliceVision;

#define ALICEVISION_SOFTWARE_VERSION_MAJOR 3
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

namespace {

constexpr const char* kDefaultModelEnv = "ALICEVISION_MOGE_MLPACKAGE";

// Default search paths for the MoGe .mlpackage. The descriptor doesn't
// pass --modelPath (the way sphereDetection does) — the binary picks it
// up from env or the standard repo location.
std::string discoverModelPath()
{
    if (const char* env = std::getenv(kDefaultModelEnv)) {
        if (*env != '\0' && fs::exists(env)) {
            return env;
        }
    }
    // Standard repo location.
    if (const char* root = std::getenv("ALICEVISION_ROOT")) {
        fs::path candidate = fs::path(root) / "ai-models" / "moge2_504x672_t1728.mlpackage";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
    // Best-effort relative-to-executable path. The bundle installer
    // co-locates ai-models/ with the binaries.
    fs::path here = fs::current_path();
    while (!here.empty() && here != here.root_path()) {
        fs::path candidate = here / "ai-models" / "moge2_504x672_t1728.mlpackage";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        here = here.parent_path();
    }
    return {};
}

}  // namespace

int aliceVision_main(int argc, char** argv)
{
    system::Timer timer;

    std::string inputSfMDataPath;
    std::string outputFolder;
    std::string foVEstimationMode = "Metadata";
    float fixedFoV = 60.0f;
    bool outputNormals = false;
    std::string modelPathOpt;

    // clang-format off
    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&inputSfMDataPath)->required(),
         "Input SfMData (or folder of images).")
        ("output,o", po::value<std::string>(&outputFolder)->required(),
         "Output folder for per-view depth (and optional normals) maps.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("foVEstimationMode",
         po::value<std::string>(&foVEstimationMode)->default_value("Metadata"),
         "How to infer FoV. One of 'Metadata', 'Estimate', 'Fixed'. "
         "Accepted for descriptor parity; the CoreML MoGe model "
         "estimates its own focal length internally.")
        ("fixedFoV",
         po::value<float>(&fixedFoV)->default_value(60.0f),
         "FoV in degrees when foVEstimationMode=='Fixed'. Ignored by "
         "the CoreML path.")
        ("outputNormals",
         po::value<bool>(&outputNormals)->default_value(false),
         "If true, also emit per-pixel normal maps next to the depth maps.")
        ("modelPath",
         po::value<std::string>(&modelPathOpt)->default_value(""),
         "Optional override of the MoGe .mlpackage path. Defaults to "
         "$ALICEVISION_MOGE_MLPACKAGE, then "
         "$ALICEVISION_ROOT/ai-models/moge2_504x672_t1728.mlpackage, "
         "then walk-up from CWD.");
    // clang-format on

    CmdLine cmdline("AliceVision moGe (Mac-port CoreML)");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);

    if (!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // Resolve model path.
    std::string modelPath = modelPathOpt.empty() ? discoverModelPath() : modelPathOpt;
    if (modelPath.empty() || !fs::exists(modelPath))
    {
        ALICEVISION_LOG_ERROR(
            "moGe: cannot find the MoGe .mlpackage. Set "
            "ALICEVISION_MOGE_MLPACKAGE, --modelPath, or place "
            "ai-models/moge2_504x672_t1728.mlpackage under "
            "ALICEVISION_ROOT.");
        return EXIT_FAILURE;
    }

    ALICEVISION_LOG_INFO("moGe: loading CoreML model from '" << modelPath << "'.");

    std::unique_ptr<av::moge::CoreMLMoGeRunner> runner;
    try
    {
        runner = std::make_unique<av::moge::CoreMLMoGeRunner>(modelPath);
    }
    catch (const std::exception& e)
    {
        ALICEVISION_LOG_ERROR("moGe: CoreML model load failed: " << e.what());
        return EXIT_FAILURE;
    }

    // -------- Load SfMData --------
    sfmData::SfMData sfmData;
    if (!sfmDataIO::load(sfmData, inputSfMDataPath,
                         sfmDataIO::ESfMData(sfmDataIO::VIEWS | sfmDataIO::INTRINSICS)))
    {
        ALICEVISION_LOG_ERROR("moGe: cannot read input SfMData '"
                              << inputSfMDataPath << "'.");
        return EXIT_FAILURE;
    }

    // -------- Ensure output folder --------
    {
        std::error_code ec;
        fs::create_directories(outputFolder, ec);
        if (ec)
        {
            ALICEVISION_LOG_ERROR("moGe: cannot create output folder '"
                                  << outputFolder << "': " << ec.message());
            return EXIT_FAILURE;
        }
    }

    image::ImageWriteOptions writeOpts;
    writeOpts.toColorSpace(image::EImageColorSpace::NO_CONVERSION);
    writeOpts.fromColorSpace(image::EImageColorSpace::NO_CONVERSION);

    // -------- Per-view inference --------
    std::size_t numWritten = 0;
    std::size_t numFailed = 0;
    for (const auto& kv : sfmData.getViews())
    {
        const IndexT viewId = kv.first;
        const fs::path imagePath = fs::path(kv.second->getImage().getImagePath());
        if (!fs::exists(imagePath))
        {
            ALICEVISION_LOG_WARNING("moGe: image missing for view " << viewId
                                    << " at '" << imagePath << "', skipping.");
            ++numFailed;
            continue;
        }

        av::moge::MoGeResult res;
        try
        {
            res = runner->predict(imagePath.string());
        }
        catch (const std::exception& e)
        {
            ALICEVISION_LOG_WARNING("moGe: inference failed for view " << viewId
                                    << " (" << imagePath.filename() << "): " << e.what());
            ++numFailed;
            continue;
        }

        // -------- Pack depth into an aliceVision Image<float> --------
        image::Image<float> depthImg(res.width, res.height, false);
        for (int y = 0; y < res.height; ++y)
        {
            for (int x = 0; x < res.width; ++x)
            {
                const size_t pix = static_cast<size_t>(y) * res.width + x;
                // Use 0.0 as the "invalid depth" sentinel for masked pixels —
                // matches DepthMapTracksInjecting's convention (0 = no data).
                depthImg(y, x) = res.mask[pix] ? res.depthMeters[pix] : 0.0f;
            }
        }

        const fs::path depthPath =
            fs::path(outputFolder) / (std::to_string(viewId) + "_depth.exr");
        try
        {
            image::writeImage(depthPath.string(), depthImg, writeOpts);
            ++numWritten;
        }
        catch (const std::exception& e)
        {
            ALICEVISION_LOG_WARNING("moGe: failed to write depth for view " << viewId
                                    << " (" << depthPath << "): " << e.what());
            ++numFailed;
            continue;
        }

        if (outputNormals)
        {
            image::Image<image::RGBfColor> normalImg(res.width, res.height, false);
            for (int y = 0; y < res.height; ++y)
            {
                for (int x = 0; x < res.width; ++x)
                {
                    const size_t pix = static_cast<size_t>(y) * res.width + x;
                    if (res.mask[pix])
                    {
                        normalImg(y, x) = image::RGBfColor(
                            res.normalXYZ[pix * 3 + 0],
                            res.normalXYZ[pix * 3 + 1],
                            res.normalXYZ[pix * 3 + 2]);
                    }
                    else
                    {
                        normalImg(y, x) = image::RGBfColor(0.f, 0.f, 0.f);
                    }
                }
            }
            const fs::path normalPath =
                fs::path(outputFolder) / (std::to_string(viewId) + "_normals.exr");
            try
            {
                image::writeImage(normalPath.string(), normalImg, writeOpts);
            }
            catch (const std::exception& e)
            {
                ALICEVISION_LOG_WARNING("moGe: failed to write normals for view " << viewId
                                        << " (" << normalPath << "): " << e.what());
            }
        }
    }

    ALICEVISION_LOG_INFO("moGe: wrote " << numWritten << " depth map(s) (and "
                         << (outputNormals ? numWritten : 0) << " normal map(s)) to '"
                         << outputFolder << "'. " << numFailed << " view(s) failed.");
    ALICEVISION_LOG_INFO("moGe: done in (s): " + std::to_string(timer.elapsed()));
    return EXIT_SUCCESS;
}
