// alicevision-for-mac — Phase 14.9
// aliceVision_matchMasking — Mac-port REAL Roma-backed implementation.
//
// Replaces the Phase 14.7 honest pass-through with actual TinyRoMa
// dense matching + optional segmentation-mask filtering of certainty
// volumes. Uses the user's ai-models/tiny_roma_v1_480x640.mlpackage
// loaded via av::roma::CoreMLRomaMatcher.
//
// Pipeline shape:
//   1. Load input SfMData + image-pairs list.
//   2. For each pair (viewIdA, viewIdB):
//        a. Run TinyRoMa on the two images → (coarse_flow, coarse_cert,
//                                              fine_flow, fine_cert).
//        b. If masksFolder is provided, mask out certainty values that
//           fall outside either view's segmentation mask.
//        c. Write coarse and fine flow + certainty EXRs to
//           outputWarpFolder / outputCertaintyFolder.
//   3. Pass-through SfMData + pairs list.
//
// Output convention (per pair):
//   outputWarpFolder/<viewIdA>_<viewIdB>_coarse_flow.exr     RGB float
//                                                            (R=flow_x, G=flow_y, B=0)
//   outputWarpFolder/<viewIdA>_<viewIdB>_fine_flow.exr       RGB float
//   outputCertaintyFolder/<viewIdA>_<viewIdB>_coarse_certainty.exr  single-channel
//   outputCertaintyFolder/<viewIdA>_<viewIdB>_fine_certainty.exr    single-channel
//
// Certainty volumes are sigmoid-normalized to [0, 1] before writing
// (model emits unnormalized logits — see ai-models/README.md).

#include <av/roma/CoreMLRomaMatcher.hpp>

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

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
namespace po = boost::program_options;

using namespace aliceVision;

#define ALICEVISION_SOFTWARE_VERSION_MAJOR 2
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

namespace {

constexpr const char* kDefaultModelEnv = "ALICEVISION_ROMA_MLPACKAGE";

// Mirrors the moGe / sphereDetection auto-discovery pattern.
std::string discoverModelPath()
{
    if (const char* env = std::getenv(kDefaultModelEnv)) {
        if (*env != '\0' && fs::exists(env)) {
            return env;
        }
    }
    if (const char* root = std::getenv("ALICEVISION_ROOT")) {
        fs::path candidate = fs::path(root) / "ai-models" / "tiny_roma_v1_480x640.mlpackage";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
    fs::path here = fs::current_path();
    while (!here.empty() && here != here.root_path()) {
        fs::path candidate = here / "ai-models" / "tiny_roma_v1_480x640.mlpackage";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        here = here.parent_path();
    }
    return {};
}

// Parse a pairs-list file. Upstream's convention is one pair per line
// as `<viewIdA> <viewIdB>`. Empty lines and `#`-prefixed comments are
// ignored.
std::vector<std::pair<IndexT, IndexT>> parsePairsList(const fs::path& path)
{
    std::vector<std::pair<IndexT, IndexT>> pairs;
    std::ifstream in(path);
    if (!in.is_open()) return pairs;
    std::string line;
    while (std::getline(in, line)) {
        // Strip trailing CR (Windows line endings) for safety.
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        IndexT a = 0, b = 0;
        if (is >> a >> b) {
            pairs.emplace_back(a, b);
        }
    }
    return pairs;
}

// Try to load a per-view binary mask (single-channel uint8 PNG) from
// masksFolder. Returns an empty Image when the file isn't present.
image::Image<unsigned char> loadMaskOrEmpty(const fs::path& masksFolder, IndexT viewId)
{
    image::Image<unsigned char> mask;
    if (masksFolder.empty()) return mask;
    // Try a couple of common naming conventions.
    for (const char* ext : {".png", ".PNG", ".exr"}) {
        const fs::path candidate = masksFolder / (std::to_string(viewId) + ext);
        std::error_code ec;
        if (fs::exists(candidate, ec)) {
            try {
                image::readImage(candidate.string(), mask, image::EImageColorSpace::NO_CONVERSION);
                return mask;
            } catch (const std::exception&) {
                // fall through
            }
        }
    }
    return mask;
}

// Bilinear-sample a value from a per-pixel mask. Coordinates are
// normalized [0,1] over the mask's dimensions.
inline bool sampleMaskValid(const image::Image<unsigned char>& mask,
                            float nx, float ny)
{
    if (mask.width() == 0 || mask.height() == 0) return true;  // no mask = valid
    int x = static_cast<int>(nx * mask.width());
    int y = static_cast<int>(ny * mask.height());
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= mask.width()) x = mask.width() - 1;
    if (y >= mask.height()) y = mask.height() - 1;
    return mask(y, x) > 0;
}

// Compose RGB EXR for flow (B channel left = 0) and single-channel for
// certainty. Caller owns the lifetime; we just hand back the image.
image::Image<image::RGBfColor> packFlowImage(const std::vector<float>& flow,
                                              int H, int W)
{
    // flow is CHW with C=2: flow[0*H*W + ...] = flow_x, flow[1*H*W + ...] = flow_y.
    image::Image<image::RGBfColor> img(W, H, false);
    const float* fx = flow.data();
    const float* fy = flow.data() + H * W;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            img(y, x) = image::RGBfColor(fx[y * W + x], fy[y * W + x], 0.f);
        }
    }
    return img;
}

// Pack per-pixel sigmoid'd certainty + optional per-view mask AND into
// a single-channel Image<float>.
image::Image<float> packCertaintyImage(const std::vector<float>& certLogits,
                                        int H, int W,
                                        const image::Image<unsigned char>& maskA,
                                        const image::Image<unsigned char>& maskB,
                                        const std::vector<float>& fineFlowForMaskWarp,
                                        int flowH, int flowW)
{
    image::Image<float> img(W, H, false);
    const bool hasMasks = (maskA.width() > 0) || (maskB.width() > 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float cert = av::roma::sigmoid(certLogits[y * W + x]);
            if (hasMasks) {
                // im_A pixel at this output cell → normalized coord.
                const float nax = (x + 0.5f) / static_cast<float>(W);
                const float nay = (y + 0.5f) / static_cast<float>(H);
                bool okA = sampleMaskValid(maskA, nax, nay);
                bool okB = true;
                if (maskB.width() > 0 && !fineFlowForMaskWarp.empty()) {
                    // Warp into im_B via the flow at this resolution
                    // (or upscale-from-fine for the coarse certainty).
                    // For simplicity, sample the fine-flow grid at the
                    // corresponding normalized coord.
                    int fy = std::min(flowH - 1, std::max(0,
                        static_cast<int>(nay * flowH)));
                    int fx = std::min(flowW - 1, std::max(0,
                        static_cast<int>(nax * flowW)));
                    const float* fxArr = fineFlowForMaskWarp.data();
                    const float* fyArr = fineFlowForMaskWarp.data() + flowH * flowW;
                    const float bx = (fxArr[fy * flowW + fx] + 1.f) * 0.5f;
                    const float by = (fyArr[fy * flowW + fx] + 1.f) * 0.5f;
                    okB = sampleMaskValid(maskB, bx, by);
                }
                if (!okA || !okB) cert = 0.f;
            }
            img(y, x) = cert;
        }
    }
    return img;
}

}  // namespace

int aliceVision_main(int argc, char** argv)
{
    system::Timer timer;

    std::string inputSfMDataPath;
    std::string inputPairsListPath;
    std::string warpFolderIn;
    std::string certaintyFolderIn;
    std::string masksFolderIn;
    std::string modelPathOpt;

    std::string outputSfMDataPath;
    std::string outputPairsListPath;
    std::string outputWarpFolder;
    std::string outputCertaintyFolder;

    // clang-format off
    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&inputSfMDataPath)->required(),
         "Input SfMData.")
        ("output,o", po::value<std::string>(&outputSfMDataPath)->required(),
         "Output SfMData (pass-through).")
        ("outputCertaintyFolder",
         po::value<std::string>(&outputCertaintyFolder)->required(),
         "Output folder for filtered certainty volumes.");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("imagePairsList",
         po::value<std::string>(&inputPairsListPath)->default_value(""),
         "Input image-pairs list (one '<viewIdA> <viewIdB>' per line). "
         "If unset, no matches are produced.")
        ("outputPairsList",
         po::value<std::string>(&outputPairsListPath)->default_value(""),
         "Output image-pairs list (pass-through).")
        ("warpFolder",
         po::value<std::string>(&warpFolderIn)->default_value(""),
         "Input warp folder. Accepted for descriptor parity; ignored — "
         "matchMasking produces its own warps from the Roma model.")
        ("outputWarpFolder",
         po::value<std::string>(&outputWarpFolder)->default_value(""),
         "Output folder for per-pair flow EXRs.")
        ("certaintyFolder",
         po::value<std::string>(&certaintyFolderIn)->default_value(""),
         "Input certainty folder. Accepted for descriptor parity; ignored.")
        ("masksFolder",
         po::value<std::string>(&masksFolderIn)->default_value(""),
         "Input mask folder (<viewId>.png per view). When provided, "
         "certainty is zeroed where either view's mask is invalid.")
        ("modelPath",
         po::value<std::string>(&modelPathOpt)->default_value(""),
         "Optional override of the Roma .mlpackage path. Defaults to "
         "$ALICEVISION_ROMA_MLPACKAGE, then "
         "$ALICEVISION_ROOT/ai-models/tiny_roma_v1_480x640.mlpackage.");
    // clang-format on

    CmdLine cmdline("AliceVision matchMasking (Mac-port CoreML Roma)");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);

    if (!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    const std::string modelPath = modelPathOpt.empty() ? discoverModelPath() : modelPathOpt;
    if (modelPath.empty() || !fs::exists(modelPath))
    {
        ALICEVISION_LOG_ERROR(
            "matchMasking: cannot find the Roma .mlpackage. Set "
            "ALICEVISION_ROMA_MLPACKAGE, --modelPath, or place "
            "ai-models/tiny_roma_v1_480x640.mlpackage under "
            "ALICEVISION_ROOT.");
        return EXIT_FAILURE;
    }

    ALICEVISION_LOG_INFO("matchMasking: loading TinyRoMa model from '"
                         << modelPath << "'.");

    std::unique_ptr<av::roma::CoreMLRomaMatcher> matcher;
    try {
        matcher = std::make_unique<av::roma::CoreMLRomaMatcher>(modelPath);
    } catch (const std::exception& e) {
        ALICEVISION_LOG_ERROR("matchMasking: Roma model load failed: " << e.what());
        return EXIT_FAILURE;
    }

    // -------- Load + write SfMData --------
    sfmData::SfMData sfmData;
    if (!sfmDataIO::load(sfmData, inputSfMDataPath,
                         sfmDataIO::ESfMData(sfmDataIO::VIEWS | sfmDataIO::INTRINSICS)))
    {
        ALICEVISION_LOG_ERROR("matchMasking: cannot read input SfMData '"
                              << inputSfMDataPath << "'.");
        return EXIT_FAILURE;
    }

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
        ALICEVISION_LOG_ERROR("matchMasking: cannot write SfMData '"
                              << outputSfMDataPath << "'.");
        return EXIT_FAILURE;
    }

    // -------- Build a viewId → image-path map --------
    std::unordered_map<IndexT, std::string> viewIdToPath;
    viewIdToPath.reserve(sfmData.getViews().size());
    for (const auto& kv : sfmData.getViews()) {
        viewIdToPath.emplace(kv.first, kv.second->getImage().getImagePath());
    }

    // -------- Output dirs --------
    {
        std::error_code ec;
        if (!outputWarpFolder.empty()) fs::create_directories(outputWarpFolder, ec);
        fs::create_directories(outputCertaintyFolder, ec);
    }

    // -------- Pairs list --------
    std::vector<std::pair<IndexT, IndexT>> pairs;
    if (!inputPairsListPath.empty() && fs::exists(inputPairsListPath)) {
        pairs = parsePairsList(inputPairsListPath);
    } else {
        ALICEVISION_LOG_WARNING(
            "matchMasking: no imagePairsList provided — no Roma matches "
            "will be produced. Run aliceVision_starListing first.");
    }

    // Pass through the pairs list (or write empty placeholder).
    if (!outputPairsListPath.empty())
    {
        std::error_code ec;
        fs::path outDir = fs::path(outputPairsListPath).parent_path();
        if (!outDir.empty()) fs::create_directories(outDir, ec);
        if (!inputPairsListPath.empty() && fs::exists(inputPairsListPath, ec)) {
            fs::copy_file(inputPairsListPath, outputPairsListPath,
                          fs::copy_options::overwrite_existing, ec);
        } else {
            std::ofstream(outputPairsListPath).close();
        }
    }

    image::ImageWriteOptions writeOpts;
    writeOpts.toColorSpace(image::EImageColorSpace::NO_CONVERSION);
    writeOpts.fromColorSpace(image::EImageColorSpace::NO_CONVERSION);

    std::size_t numMatched = 0;
    std::size_t numFailed = 0;

    for (const auto& [vA, vB] : pairs)
    {
        const auto itA = viewIdToPath.find(vA);
        const auto itB = viewIdToPath.find(vB);
        if (itA == viewIdToPath.end() || itB == viewIdToPath.end()) {
            ALICEVISION_LOG_WARNING("matchMasking: pair ("
                                    << vA << "," << vB << ") refers to "
                                    "unknown view(s); skipping.");
            ++numFailed;
            continue;
        }
        if (!fs::exists(itA->second) || !fs::exists(itB->second)) {
            ALICEVISION_LOG_WARNING("matchMasking: image missing for pair ("
                                    << vA << "," << vB << "); skipping.");
            ++numFailed;
            continue;
        }

        av::roma::RomaMatch m;
        try {
            m = matcher->match(itA->second, itB->second);
        } catch (const std::exception& e) {
            ALICEVISION_LOG_WARNING("matchMasking: Roma failed for pair ("
                                    << vA << "," << vB << "): " << e.what());
            ++numFailed;
            continue;
        }

        // Optional masks (loaded once per pair).
        image::Image<unsigned char> maskA = loadMaskOrEmpty(masksFolderIn, vA);
        image::Image<unsigned char> maskB = loadMaskOrEmpty(masksFolderIn, vB);

        const std::string pairStem = std::to_string(vA) + "_" + std::to_string(vB);

        // -------- Flow EXRs --------
        if (!outputWarpFolder.empty()) {
            auto coarseFlowImg = packFlowImage(
                m.coarseFlow, av::roma::kCoarseHeight, av::roma::kCoarseWidth);
            auto fineFlowImg = packFlowImage(
                m.fineFlow, av::roma::kFineHeight, av::roma::kFineWidth);
            try {
                image::writeImage(
                    (fs::path(outputWarpFolder) / (pairStem + "_coarse_flow.exr")).string(),
                    coarseFlowImg, writeOpts);
                image::writeImage(
                    (fs::path(outputWarpFolder) / (pairStem + "_fine_flow.exr")).string(),
                    fineFlowImg, writeOpts);
            } catch (const std::exception& e) {
                ALICEVISION_LOG_WARNING("matchMasking: failed to write flow for pair "
                                        << pairStem << ": " << e.what());
                ++numFailed;
                continue;
            }
        }

        // -------- Certainty EXRs (sigmoid + optional mask) --------
        auto coarseCertImg = packCertaintyImage(
            m.coarseCertainty, av::roma::kCoarseHeight, av::roma::kCoarseWidth,
            maskA, maskB, m.fineFlow, av::roma::kFineHeight, av::roma::kFineWidth);
        auto fineCertImg = packCertaintyImage(
            m.fineCertainty, av::roma::kFineHeight, av::roma::kFineWidth,
            maskA, maskB, m.fineFlow, av::roma::kFineHeight, av::roma::kFineWidth);
        try {
            image::writeImage(
                (fs::path(outputCertaintyFolder) / (pairStem + "_coarse_certainty.exr")).string(),
                coarseCertImg, writeOpts);
            image::writeImage(
                (fs::path(outputCertaintyFolder) / (pairStem + "_fine_certainty.exr")).string(),
                fineCertImg, writeOpts);
            ++numMatched;
        } catch (const std::exception& e) {
            ALICEVISION_LOG_WARNING("matchMasking: failed to write certainty for pair "
                                    << pairStem << ": " << e.what());
            ++numFailed;
        }
    }

    ALICEVISION_LOG_INFO("matchMasking: " << numMatched << " pair(s) matched, "
                         << numFailed << " failed. Outputs in '"
                         << outputCertaintyFolder << "' (+ '"
                         << outputWarpFolder << "' if requested).");
    ALICEVISION_LOG_INFO("matchMasking: done in (s): "
                         + std::to_string(timer.elapsed()));
    return EXIT_SUCCESS;
}
