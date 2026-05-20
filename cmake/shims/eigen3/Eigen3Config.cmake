# Eigen3 shim — points at Homebrew Eigen 5.x install but reports
# itself to find_package() as a 3.x-compatible Eigen, satisfying
# upstream AliceVision's `find_package(Eigen3 3.3 REQUIRED)` while
# still using Homebrew's modern Eigen.
#
# Why this exists: Eigen 5's stock Eigen3ConfigVersion.cmake refuses
# to satisfy a single-version request of 3.3 (it only matches the
# requested major.minor exactly), so upstream's find_package call
# fails despite Eigen being installed. Modifying upstream is off
# limits in this out-of-tree overlay port, so we shim instead.

# Re-export the Eigen3::Eigen target from Homebrew's config.
include("/opt/homebrew/share/eigen3/cmake/Eigen3Config.cmake")
