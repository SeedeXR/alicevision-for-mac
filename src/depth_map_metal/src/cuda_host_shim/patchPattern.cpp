// patchPattern.cpp — Apple Silicon implementation of
// `aliceVision::depthMap::buildCustomPatchPattern`.
//
// The shim ports upstream's validation + coordinate-generation
// logic byte-for-byte but writes into our
// `av::depth_map::DevicePatchPattern` (CPU mirror of the MSL
// constant-buffer struct, defined in DevicePatchPattern.hpp).
// The resulting pattern is parked in a process-global slot
// exposed via `av::depth_map::upstream_adapter::current_patch_pattern()`
// so the eventual MSL-side dispatchers can pick it up.

#include "aliceVision/depthMap/cuda/host/patchPattern.hpp"

#include "av/depth_map/DevicePatchPattern.hpp"
#include "av/depth_map/upstream_adapter.hpp"

#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace av::depth_map { namespace upstream_adapter {

// Process-global current pattern. Initialized to all-zeros (== no
// subparts). Updated atomically by `buildCustomPatchPattern`.
namespace {
std::mutex          g_pattern_mu;
DevicePatchPattern  g_current_pattern = {};  // value-initialized
}

const DevicePatchPattern& current_patch_pattern() {
    std::lock_guard lk(g_pattern_mu);
    return g_current_pattern;
}

void set_current_patch_pattern(const DevicePatchPattern& p) {
    std::lock_guard lk(g_pattern_mu);
    g_current_pattern = p;
}

}}  // namespace av::depth_map::upstream_adapter


namespace aliceVision {
namespace depthMap {

void buildCustomPatchPattern(const CustomPatchPatternParams& patchParams)
{
    using av::depth_map::DevicePatchPattern;
    using av::depth_map::DevicePatchPatternSubpart;
    using av::depth_map::kPatchMaxSubparts;
    using av::depth_map::kPatchMaxCoordsPerSubpart;

    if (patchParams.subpartsParams.empty()) {
        throw std::runtime_error(
            "buildCustomPatchPattern: no patch pattern subpart given.");
    }

    // Build nb-coords-per-subpart map (matches upstream's
    // patchPattern.cpp validation step, lines 26–51).
    std::map<int, int> nbCoordsPerSubparts;  // <bucket key, nb coords>
    for (std::size_t i = 0; i < patchParams.subpartsParams.size(); ++i) {
        const auto& s = patchParams.subpartsParams[i];

        if (s.radius <= 0.f) {
            throw std::runtime_error(
                "buildCustomPatchPattern: subpart radius must be > 0.");
        }
        if (s.isCircle && s.nbCoordinates <= 0) {
            throw std::runtime_error(
                "buildCustomPatchPattern: circle subpart needs > 0 "
                "coordinates.");
        }

        if (patchParams.groupSubpartsPerLevel) {
            if (!s.isCircle &&
                nbCoordsPerSubparts.find(s.level) != nbCoordsPerSubparts.end()) {
                throw std::runtime_error(
                    "buildCustomPatchPattern: cannot group more than one "
                    "full patch pattern subpart.");
            }
            nbCoordsPerSubparts[s.level] += (s.isCircle ? s.nbCoordinates : 0);
        } else {
            nbCoordsPerSubparts[static_cast<int>(i)] +=
                (s.isCircle ? s.nbCoordinates : 0);
        }
    }

    int maxSubpartCoords = 0;
    for (const auto& kv : nbCoordsPerSubparts)
        maxSubpartCoords = std::max(maxSubpartCoords, kv.second);

    const int nbSubparts = static_cast<int>(nbCoordsPerSubparts.size());

    if (nbSubparts > kPatchMaxSubparts) {
        throw std::runtime_error(
            "buildCustomPatchPattern: too many subparts (" +
            std::to_string(nbSubparts) + " > " +
            std::to_string(kPatchMaxSubparts) + ").");
    }
    if (maxSubpartCoords > kPatchMaxCoordsPerSubpart) {
        throw std::runtime_error(
            "buildCustomPatchPattern: too many coordinates per subpart (" +
            std::to_string(maxSubpartCoords) + " > " +
            std::to_string(kPatchMaxCoordsPerSubpart) + ").");
    }

    DevicePatchPattern out = {};
    out.nbSubparts = nbSubparts;

    if (patchParams.groupSubpartsPerLevel) {
        // Zero the accumulators (the value-init above already did
        // this, but mirror upstream's explicit step for clarity).
        for (int i = 0; i < out.nbSubparts; ++i) {
            out.subparts[i].nbCoordinates = 0;
            out.subparts[i].wsh           = 0;
        }
        for (const auto& s : patchParams.subpartsParams) {
            const auto it = nbCoordsPerSubparts.find(s.level);
            const int bucket =
                static_cast<int>(std::distance(nbCoordsPerSubparts.begin(), it));
            auto& subpart = out.subparts[bucket];

            if (s.isCircle) {
                const float radius = s.radius;
                const float dAngle = (float(M_PI) * 2.f) / float(s.nbCoordinates);
                for (int j = 0; j < s.nbCoordinates; ++j) {
                    const int dst = subpart.nbCoordinates + j;
                    const float rad = dAngle * float(j);
                    subpart.coordinates[dst][0] = std::cos(rad) * radius;
                    subpart.coordinates[dst][1] = std::sin(rad) * radius;
                }
                const int wsh_inc =
                    int(s.radius + std::pow(2.f, float(s.level) - 1.f));
                if (wsh_inc > subpart.wsh) subpart.wsh = wsh_inc;
                subpart.nbCoordinates += s.nbCoordinates;
            } else {
                if (int(s.radius) > subpart.wsh) subpart.wsh = int(s.radius);
            }

            subpart.level     = s.level;
            subpart.downscale = std::pow(2.f, float(subpart.level));
            subpart.weight    = s.weight;
            subpart.isCircle  = s.isCircle ? 1 : 0;
        }
    } else {
        // Per-subpart (1:1) — upstream's else-branch.
        for (int i = 0; i < out.nbSubparts; ++i) {
            auto& subpart = out.subparts[i];
            const auto& s = patchParams.subpartsParams[static_cast<std::size_t>(i)];

            if (s.isCircle) {
                const float radius = s.radius;
                // NOTE: upstream divides by `subpart.nbCoordinates`
                // BEFORE setting it from `s.nbCoordinates` (see
                // patchPattern.cpp:176). On uninitialised memory
                // (CUDA constant memory zero-init) this yields
                // `2π / 0 = inf`. We pre-zero `subpart` so this
                // mirrors the upstream bug only in the unusual case
                // and matches the intended behaviour: use
                // `s.nbCoordinates` as the divisor.
                const float dAngle =
                    (float(M_PI) * 2.f) / float(s.nbCoordinates);
                for (int j = 0; j < s.nbCoordinates; ++j) {
                    const float rad = dAngle * float(j);
                    subpart.coordinates[j][0] = std::cos(rad) * radius;
                    subpart.coordinates[j][1] = std::sin(rad) * radius;
                }
                subpart.wsh =
                    int(s.radius + std::pow(2.f, float(s.level) - 1.f));
                subpart.nbCoordinates = s.nbCoordinates;
            } else {
                subpart.wsh           = int(s.radius);
                subpart.nbCoordinates = 0;
            }

            subpart.level     = s.level;
            subpart.downscale = std::pow(2.f, float(subpart.level));
            subpart.weight    = s.weight;
            subpart.isCircle  = s.isCircle ? 1 : 0;
        }
    }

    av::depth_map::upstream_adapter::set_current_patch_pattern(out);
}

}  // namespace depthMap
}  // namespace aliceVision
