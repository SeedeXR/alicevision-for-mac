#pragma once

// upstream_adapter_types.hpp — private internal definitions shared
// between `upstream_adapter.cpp` and its tests.
//
// Provides the minimal POD definitions for the types that
// `upstream_adapter.hpp` only forward-declares:
//
//   * `aliceVision::Range`     — matches upstream's actual location.
//   * `aliceVision::ROI`        — matches upstream's actual location.
//   * `aliceVision::depthMap::SgmParams` — full struct definition,
//     mirroring `upstream/.../depthMap/SgmParams.hpp`.
//   * `aliceVision::depthMap::RefineParams` — full struct definition,
//     mirroring `upstream/.../depthMap/RefineParams.hpp`.
//
// History:
//   * S34: defined Range/ROI in different namespaces (mvsData::ROI,
//     depthMap::Range) — incorrect, the adapter header used wrong
//     forward-decls.
//   * S35: reconciled to upstream's actual `aliceVision::Range` /
//     `aliceVision::ROI` locations. Adapter header + this header
//     + the .cpp all now agree with upstream.
//
// This header is consumed only by the adapter implementation and
// its tests. It is not installed.

#include <cstdint>

namespace aliceVision {

// Mirrors upstream `aliceVision::Range`. Field surface verified
// from `upstream/.../mvsData/ROI.hpp:35`.
struct Range {
    unsigned int begin = 0;
    unsigned int end   = 0;
};

// Mirrors upstream `aliceVision::ROI` field surface (width/height
// derived from x/y subranges). Verified from
// `upstream/.../mvsData/ROI.hpp:76`.
struct ROI {
    Range x{};
    Range y{};
    unsigned int width()  const { return x.end - x.begin; }
    unsigned int height() const { return y.end - y.begin; }
};

}  // namespace aliceVision

namespace aliceVision {
namespace depthMap {

// Mirror of upstream's `SgmParams` (POD aggregate). Field set
// verified against `upstream/src/aliceVision/depthMap/SgmParams.hpp`.
struct SgmParams {
    int    scale                = 2;
    int    stepXY               = 2;
    int    stepZ                = -1;
    int    wsh                  = 4;
    int    maxDepths            = 1500;
    int    maxTCamsPerTile      = 4;
    double seedsRangeInflate    = 0.2;
    double depthThicknessInflate = 0.0;
    double maxSimilarity        = 1.0;
    double gammaC               = 5.5;
    double gammaP               = 8.0;
    double p1                   = 10.0;
    // Signed convention: < 0 → fixed P2 = |p2Weighting|;
    //                    >= 0 → adaptive P2 (sigmoid-driven).
    double p2Weighting          = 100.0;
    bool   useSfmSeeds          = true;
    bool   depthListPerTile     = false;
    bool   useConsistentScale   = false;
    bool   useCustomPatchPattern = false;
};

// Mirror of upstream's `RefineParams`.
struct RefineParams {
    int    scale                  = 1;
    int    stepXY                 = 1;
    int    wsh                    = 3;
    int    halfNbDepths           = 15;
    int    nbSubsamples           = 10;
    int    maxTCamsPerTile        = 4;
    int    optimizationNbIterations = 100;
    double sigma                  = 15.0;
    double gammaC                 = 15.5;
    double gammaP                 = 8.0;
    bool   interpolateMiddleDepth = false;
    bool   useConsistentScale     = false;
    bool   useCustomPatchPattern  = false;
    bool   useRefineFuse          = true;
    bool   useColorOptimization   = true;
    bool   useSgmNormalMap        = false;
};

}  // namespace depthMap
}  // namespace aliceVision
