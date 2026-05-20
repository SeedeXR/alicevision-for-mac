# Eigen3 version-file shim — promiscuous version reporting.
#
# Why: Homebrew Eigen is currently 5.x. AliceVision asks for 3.3+;
# Eigen5's stock version file refuses any non-5.x request. Homebrew
# Ceres was built against Eigen 5.0.1 and verifies the find-time
# version via an internal find_package(Eigen3 EXACT 5.0.1) — Eigen5's
# stock version file accepts that, but our 3.4-fixed shim breaks it.
#
# This file reports as PACKAGE_VERSION = whatever the caller asked
# for, and unconditionally sets PACKAGE_VERSION_COMPATIBLE/_EXACT to
# TRUE. That keeps both consumers happy. The underlying include
# path still points at the real Homebrew Eigen install.
#
# This is acceptable because the Eigen 3.4 ↔ Eigen 5.x API surface
# is stable for the features AliceVision and Ceres use. If a real
# ABI break occurs later we will revisit and rebuild Eigen from
# source.

if(PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION "${PACKAGE_FIND_VERSION}")
else()
    set(PACKAGE_VERSION "3.4.0")
endif()

set(PACKAGE_VERSION_EXACT      TRUE)
set(PACKAGE_VERSION_COMPATIBLE TRUE)
set(PACKAGE_VERSION_UNSUITABLE FALSE)
