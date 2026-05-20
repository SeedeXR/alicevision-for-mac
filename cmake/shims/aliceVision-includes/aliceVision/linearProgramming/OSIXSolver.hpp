// S39 shim for upstream's OSIXSolver.hpp.
//
// Upstream's header `#include "OsiClpSolverInterface.hpp"` from Coin-OR Clp,
// which is not packaged on Homebrew. Our `Coin::Clp/CoinUtils/Osi` targets
// in the root CMakeLists are empty INTERFACE imported targets — they don't
// expose any include path.
//
// `OSI_CISolverWrapper = OSIXSolver<OsiClpSolverInterface>` is referenced by:
//   * lInfinityCV/resection_kernel.cpp (patched at CMake time to throw).
//   * Indirect includes through `linearProgramming/linearProgramming.hpp`
//     (transitively pulled into many sfm/colorHarmonization/HalfPlane TUs
//     but never *used* in the SfM-only pipeline path).
//
// Strategy: forward-declare `OsiClpSolverInterface` (and the optional Mosek
// one) so the `OSIXSolver<>` template + `OSI_CISolverWrapper` typedef
// remain well-formed but unusable until instantiated. The template body
// dereferences `si->setLogLevel(...)` etc., which is only instantiated when
// a TU uses the typedef. Because no remaining TU we compile actually
// invokes the wrapper after the resection_kernel patch, the template body
// never gets instantiated and the missing methods don't matter.

#pragma once

#include <aliceVision/config.hpp>
#include <aliceVision/linearProgramming/ISolver.hpp>

#include <vector>

namespace aliceVision {
namespace linearProgramming {

// Forward declarations only — no methods invoked unless an OSIXSolver<>
// is actually instantiated.
class OsiClpSolverInterface;
#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_MOSEK)
class OsiMskSolverInterface;
#endif

/// OSI_X wrapper for the ISolver (S39 shim).
///
/// All methods are defined inline as no-op stubs:
///   * `setup` returns true (no constraints stored)
///   * `solve` returns true (always declares the LP "feasible")
///   * `getSolution` returns an all-zeros vector
///
/// The only known caller in the SfM pipeline path is
/// `geometry::halfPlane::isNotEmpty()` (used by `FrustumFilter.cpp`) which
/// checks whether a set of frustum half-planes intersect. Returning true
/// is a safe over-approximation — extra image pairs are kept for
/// matching, no pairs are wrongly culled.
template<typename SOLVERINTERFACE>
class OSIXSolver : public ISolver
{
  public:
    OSIXSolver(int nbParams) : ISolver(nbParams) {}
    ~OSIXSolver() = default;

    bool setup(const LPConstraints& /*constraints*/) { return true; }
    bool setup(const LPConstraintsSparse& /*constraints*/) { return true; }
    bool solve() { return true; }
    bool getSolution(std::vector<double>& estimatedParams)
    {
        // `nbParams` is the protected base member set by ISolver(int).
        estimatedParams.assign(static_cast<std::size_t>(ISolver::_nbParams), 0.0);
        return true;
    }
};

typedef OSIXSolver<OsiClpSolverInterface> OSI_CISolverWrapper;
#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_MOSEK)
typedef OSIXSolver<OsiMskSolverInterface> OSI_MOSEK_SolverWrapper;
#endif

}  // namespace linearProgramming
}  // namespace aliceVision
