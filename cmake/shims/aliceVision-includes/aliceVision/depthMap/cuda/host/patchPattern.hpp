#pragma once

// patchPattern.hpp — Apple Silicon type-shim. Replaces upstream's
// `aliceVision/depthMap/cuda/host/patchPattern.hpp`.
//
// Surface is exactly upstream's single free function:
//
//   void buildCustomPatchPattern(const CustomPatchPatternParams& patchParams);
//
// The shim builds an `av::depth_map::DevicePatchPattern` from the
// caller-supplied `CustomPatchPatternParams` (same struct upstream
// uses, defined in `aliceVision/depthMap/CustomPatchPatternParams.hpp`)
// and stashes it in a process-global slot accessible to the kernels
// via `av::depth_map::upstream_adapter::current_patch_pattern()`.
//
// The 12 `cuda_*` forwarders that need a patch pattern (currently
// none — Phase 9 will add them as the patch-aware compute paths
// are exercised end-to-end) will read this global at dispatch time
// and feed it to the MSL `constant DevicePatchPattern&` binding.

#include <aliceVision/depthMap/CustomPatchPatternParams.hpp>

namespace aliceVision {
namespace depthMap {

// Build a custom patch pattern from `patchParams` and install it
// as the process-global current pattern. Throws if the
// parameters are out of bounds (same checks upstream performs).
void buildCustomPatchPattern(const CustomPatchPatternParams& patchParams);

}  // namespace depthMap
}  // namespace aliceVision
