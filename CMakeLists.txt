cmake_minimum_required(VERSION 3.30)

# ============================================================
# alicevision-for-mac — Apple Silicon port of AliceVision
# ============================================================
# This is an out-of-tree overlay: we add Metal-backed kernels,
# a unified CLI, and macOS-native build glue. The upstream
# AliceVision repository is consumed through upstream/AliceVision
# (a symlink to the read-only reference clone) and is NOT
# configured automatically — see options below.
# ============================================================

project(alicevision_for_mac
    VERSION 0.1.0
    DESCRIPTION "Apple Silicon-native AliceVision port"
    LANGUAGES C CXX OBJC OBJCXX
)

if(NOT APPLE)
    message(FATAL_ERROR "alicevision-for-mac targets Apple platforms only.")
endif()

# ---------- Architectures / deployment target ----------

if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
endif()
if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    message(WARNING "x86_64 macOS is out of scope for milestone 1 — proceed at your own risk.")
endif()
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "" FORCE)
endif()

# ---------- C++ standard ----------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------- Build type default ----------

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

# ---------- Project options ----------

# AV_USE_METAL: gate for the Metal GPU backend. Default ON on Apple (the only
# supported platform here). Currently this option is **reserved-for-future**:
# every Metal-backed subdirectory (src/av_gpu, src/depth_map_metal,
# src/shaders) is added unconditionally because the project has no CPU-only
# fallback — flipping this to OFF would produce a non-functional build.
# It is wired into the `ALICEVISION_HAVE_DEPTHMAP` gate (Phase 9, S31) so
# downstream consumers can ask "is there *any* GPU depthMap backend?" via a
# single define regardless of which side (CUDA on Linux/Win, Metal on Apple)
# is active. When a CPU-only path lands we'll start using AV_USE_METAL=OFF
# to actually skip the Metal subdirs.
option(AV_USE_METAL              "Enable Metal backend"                       ON)
option(AV_USE_CUDA               "Enable CUDA (disabled on Apple)"            OFF)
option(AV_BUILD_TESTS            "Build unit + integration tests"             ON)
option(AV_BUILD_HELLO_METAL      "Build the Metal hello-kernel smoke test"    ON)
option(AV_BUILD_UPSTREAM         "Configure upstream/AliceVision (separate)"  OFF)
option(AV_BUILD_UPSTREAM_DEPTHMAP "Build the upstream depthMap-only dep tree" OFF)
option(AV_USE_HOMEBREW_DEPS      "Resolve dependencies via Homebrew prefix"   ON)
# S43 Phase 14: enable adapter-forwarder timing accumulator. Default OFF
# (zero runtime cost, no symbol change). Turn ON to dump a per-function
# summary table on process exit. See `src/depth_map_metal/src/adapter_profile.cpp`.
option(AV_PROFILE_ADAPTER        "Profile cuda_* adapter forwarders (S43)"    OFF)

if(AV_USE_CUDA)
    message(FATAL_ERROR "CUDA is not available on macOS. Set -DAV_USE_CUDA=OFF.")
endif()
if(NOT AV_USE_METAL)
    message(FATAL_ERROR
        "AV_USE_METAL=OFF is not yet supported — there is no CPU-only "
        "fallback for the depthMap / av_gpu modules. Leave it ON (the "
        "default) until a CPU path is wired in.")
endif()

# ---------- Homebrew prefix discovery ----------

if(AV_USE_HOMEBREW_DEPS)
    execute_process(
        COMMAND brew --prefix
        OUTPUT_VARIABLE HOMEBREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _brew_rc
    )
    if(_brew_rc EQUAL 0 AND HOMEBREW_PREFIX)
        list(PREPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")
        message(STATUS "Homebrew prefix: ${HOMEBREW_PREFIX}")
    else()
        message(WARNING "AV_USE_HOMEBREW_DEPS=ON but `brew --prefix` failed.")
    endif()
endif()

# ---------- RPATH (so installed binaries find dylibs) ----------

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "@executable_path/../lib;@loader_path/../lib")

# ---------- Helpers ----------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# GNUInstallDirs sets CMAKE_INSTALL_{BIN,LIB,DATA,INCLUDE,etc.}DIR
# defaults. Upstream's per-module CMakeLists call e.g.
# `install(... DESTINATION ${CMAKE_INSTALL_DATADIR}/aliceVision)`; without
# GNUInstallDirs the variable is empty → leading "/" makes it absolute →
# `install` errors with "file cannot create directory: /aliceVision".
include(GNUInstallDirs)

include(Warnings)
include(Metal)

# ---------- Eigen (header-only; for CPU reference in tests) ----------

# Resolve Eigen3 via the same shim we use for upstream AliceVision —
# the shim's promiscuous version reporting bridges Homebrew Eigen 5.x
# to anyone asking for 3.3+.
if(NOT Eigen3_DIR)
    set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/cmake/shims/eigen3" CACHE PATH "")
endif()
find_package(Eigen3 3.3 CONFIG)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found (via shim) — CPU-reference tests enabled.")
else()
    message(WARNING "Eigen3 not found; CPU-reference tests will be skipped.")
endif()

# ---------- Subdirectories ----------

add_subdirectory(third_party)
add_subdirectory(src/av_gpu)
add_subdirectory(src/depth_map_metal)
add_subdirectory(src/shaders)

# ---------- Upstream (depthMap-only) ----------
#
# Path C: pull in just the upstream modules that depthMap's host-side
# code depends on. Gated behind AV_BUILD_UPSTREAM_DEPTHMAP because
# (a) the dependency surface is large and incremental work, and
# (b) most contributors don't need the upstream tree to validate the
# Metal kernels in isolation.
if(AV_BUILD_UPSTREAM_DEPTHMAP)
    include(UpstreamShim)

    # Variables expected by the per-module CMakeLists we'll add.
    set(ALICEVISION_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/upstream/src")
    set(generatedDir            "${CMAKE_BINARY_DIR}/upstream-generated")
    file(MAKE_DIRECTORY "${generatedDir}")

    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(ALICEVISION_OPENMP_CXX_TARGETS OpenMP::OpenMP_CXX)
    else()
        set(ALICEVISION_OPENMP_CXX_TARGETS "")
    endif()
    set(ALICEVISION_NVTX_LIBRARY "")
    # Upstream supports two Eigen alignment modes:
    #   1. ALICEVISION_EIGEN_REQUIRE_ALIGNMENT=1 (Eigen default alignment)
    #   2. EIGEN_MAX_ALIGN_BYTES=0 (disable alignment — safer ABI across boundaries)
    # Mode 2 matches upstream's "AV_EIGEN_REQUIRE_ALIGNMENT" OFF branch
    # at upstream/src/CMakeLists.txt:421. We pick mode 2 to mirror the
    # default upstream Linux/macOS build.
    set(AV_EIGEN_DEFINITIONS -DEIGEN_MAX_ALIGN_BYTES=0 -DEIGEN_MAX_STATIC_ALIGN_BYTES=0)

    # Generate aliceVision/config.hpp into generatedDir. Upstream's
    # root CMakeLists builds this from src/cmake/config.hpp.in by
    # substituting feature flags; on Apple we know the answers
    # statically: no CUDA, no MOSEK, no OpenCV/CCTAG/APRILTAG/POPSIFT
    # (we don't need them for depthMap), Alembic + USD + ONNX/ONNX_GPU
    # off by default (turn on later if needed).
    set(ALICEVISION_HAVE_OPENMP    0)
    set(ALICEVISION_HAVE_SSE       0)
    set(ALICEVISION_HAVE_MOSEK     0)
    set(ALICEVISION_HAVE_OPENCV    0)
    set(ALICEVISION_HAVE_OCVSIFT   0)
    set(ALICEVISION_HAVE_ALEMBIC   0)
    set(ALICEVISION_HAVE_USD       0)
    set(ALICEVISION_HAVE_CCTAG     0)
    set(ALICEVISION_HAVE_APRILTAG  0)
    set(ALICEVISION_HAVE_POPSIFT   0)
    set(ALICEVISION_HAVE_CUDA      0)
    set(ALICEVISION_HAVE_ONNX      0)
    set(ALICEVISION_HAVE_ONNX_GPU  0)
    if(OpenMP_CXX_FOUND)
        set(ALICEVISION_HAVE_OPENMP 1)
    endif()

    # ALICEVISION_HAVE_DEPTHMAP — Phase 9 gate (S31).
    # Capability flag for "is there *any* GPU depthMap backend wired in?".
    # True when CUDA is on (Linux/Windows upstream) OR Metal is on AND the
    # host-side depthMap module is part of this build. On Apple, AV_USE_CUDA
    # is always OFF; AV_USE_METAL is always ON; AV_BUILD_UPSTREAM_DEPTHMAP
    # gates whether we even reach this block — so on Apple this evaluates
    # to 1 here, and to 0 in builds that skip the upstream tree.
    set(ALICEVISION_HAVE_DEPTHMAP 0)
    if(AV_USE_CUDA OR (AV_USE_METAL AND AV_BUILD_UPSTREAM_DEPTHMAP))
        set(ALICEVISION_HAVE_DEPTHMAP 1)
    endif()

    # Also expose AV_HAVE_DEPTHMAP as a plain -D flag for any target below
    # that wants a simple object-like macro (no config.hpp include needed).
    # The function-like config.hpp form `ALICEVISION_HAVE_DEPTHMAP()` (paired
    # with `ALICEVISION_IS_DEFINED(...)`) remains the canonical, upstream-
    # compatible spelling — don't redefine that name on the command line, or
    # clang flags it as `-Wmacro-redefined`.
    add_compile_definitions(AV_HAVE_DEPTHMAP=${ALICEVISION_HAVE_DEPTHMAP})

    # Use OUR config template instead of the upstream one — upstream's
    # cmake/config.hpp.in is read-only, and we need to add the new
    # ALICEVISION_HAVE_DEPTHMAP define above. See cmake/upstream-config.hpp.in
    # for the diff vs upstream.
    configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/upstream-config.hpp.in"
        "${generatedDir}/aliceVision/config.hpp"
        @ONLY
    )

    # NOTE (Phase 9, item 3): the third Phase-9 bullet — "compile the
    # depthMap library against the Metal backend on Apple" — is
    # deliberately deferred to Session 32. It depends on the host-side
    # depthMap orchestration rewire (`Sgm.cpp`, `Refine.cpp`,
    # `DepthMapEstimator.cpp`, `NormalMapEstimator.cpp`,
    # `computeOnMultiGPUs.cpp`) tracked under Phase 8 in memory/todo.md,
    # which in turn needs an adapter shim from upstream's CUDA-shaped
    # APIs to our Metal `av::depth_map::*` host classes. Until that lands
    # we only build the depthMap *dependencies* (system, numeric, image,
    # …, gpu); the `depthMap` subdirectory itself stays out of the
    # add_subdirectory list below.

    # Boost: required by `system` (log, thread, system, date_time)
    # and almost every other upstream module. Modern Boost (≥1.86)
    # made `Boost.System` header-only and Homebrew no longer installs
    # the separately-packaged CMake config for it — so we don't
    # request `system` from find_package, and instead provide a stub
    # `Boost::system` interface target (header-only) below.
    find_package(Boost CONFIG REQUIRED COMPONENTS
        log log_setup thread date_time
    )

    if(NOT TARGET Boost::system)
        # Header-only stub matching the modern Boost.System layout.
        # Upstream's link lines reference this target; we forward to
        # the umbrella Boost::headers target for header access.
        add_library(Boost::system INTERFACE IMPORTED)
        if(TARGET Boost::headers)
            set_target_properties(Boost::system PROPERTIES
                INTERFACE_LINK_LIBRARIES Boost::headers)
        endif()
    endif()

    # Image-pipeline dependencies (image module + its consumers).
    find_package(OpenImageIO CONFIG REQUIRED)
    find_package(OpenEXR     CONFIG REQUIRED)
    find_package(Imath       CONFIG REQUIRED)

    # Coin-OR LP backend: not available on Homebrew, and the
    # depthMap pipeline doesn't actually solve any LP problem.
    # The upstream linearProgramming module is INTERFACE-only
    # (header-only forwarder) so stubbing Coin::* as empty
    # interface targets lets it compile. If anything we depend on
    # eventually *calls* a Coin-OR function, we'll see a clear
    # link error and know we need to either port that call or
    # build the Coin-OR libs from source (todo.md Path A).
    foreach(_t IN ITEMS Coin::Clp Coin::CoinUtils Coin::Osi)
        if(NOT TARGET ${_t})
            add_library(${_t} INTERFACE IMPORTED)
        endif()
    endforeach()

    # mvs* + multiview deps.
    find_package(ZLIB                REQUIRED)
    find_package(Ceres CONFIG        REQUIRED)

    # LEMON: upstream's `multiview/rotationAveraging/l1.cpp` uses
    # LEMON for graph algorithms — but it has an
    # `ALICEVISION_ROTATION_AVERAGING_WITH_BOOST` `#ifdef` that
    # falls back to Boost.Graph for the same routines. Homebrew's
    # `lemon` package is the parser generator, NOT the
    # COIN-OR LEMON graph library. We use the Boost.Graph fallback
    # for `l1.cpp` but ALSO vendor LEMON v1.3.1 under
    # `third_party/lemon/` because `aliceVision_track` (S39) uses
    # `lemon::ListDigraph` + `lemon::UnionFindEnum` with no fallback
    # path. The vendored target is `av_lemon` (= alias `lemon::lemon`).
    add_compile_definitions(ALICEVISION_ROTATION_AVERAGING_WITH_BOOST)
    set(LEMON_LIBRARY av_lemon)

    # LIDAR / E57: `dataio/E57Reader.cpp` is built only when
    # `ALICEVISION_BUILD_LIDAR` is not "OFF" (note the inverted check).
    # The E57 SDK (libE57Format) is not packaged on Homebrew, so we
    # explicitly disable LIDAR support.
    set(ALICEVISION_BUILD_LIDAR OFF)

    set(CERES_LIBRARIES Ceres::ceres)

    add_subdirectory(upstream/src/aliceVision/system)
    add_subdirectory(upstream/src/aliceVision/numeric)
    add_subdirectory(upstream/src/aliceVision/image)
    add_subdirectory(upstream/src/aliceVision/stl)
    # ----------------------------------------------------------------
    # S39 — Patch lInfinityCV/resection_kernel.cpp
    # ----------------------------------------------------------------
    # `resection_kernel.cpp` is the only translation unit in
    # `aliceVision_lInftyComputerVision` (everything else is headers).
    # It instantiates `OSI_CISolverWrapper` (= `OSIXSolver<OsiClpSolverInterface>`),
    # which #includes "OsiClpSolverInterface.hpp" from Coin-OR Clp. We
    # stub `Coin::Clp/CoinUtils/Osi` as empty INTERFACE imported targets
    # (linearProgramming is INTERFACE-only on our side), so the header
    # isn't on the include path.
    #
    # `l1SixPointResectionSolver` is referenced only by the `lInftyCV_*`
    # tests, none of which we build. We replace the function body with a
    # `throw` to preserve the symbol and surface a clear runtime error if
    # SfM ever lands on this code path.
    set(_LINF_SRC     "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/linearProgramming/lInfinityCV/resection_kernel.cpp")
    set(_LINF_PATCHED "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/linearProgramming/lInfinityCV/resection_kernel.cpp")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/linearProgramming/lInfinityCV")
    file(READ "${_LINF_SRC}" _linf_content)
    string(REPLACE
        "#include <aliceVision/linearProgramming/OSIXSolver.hpp>"
        "#include <stdexcept> /* S39: OSIXSolver stripped — Coin-OR Clp absent */"
        _linf_content "${_linf_content}")
    string(REPLACE
        "    OSI_CISolverWrapper wrapperLpSolve(vec_solution.size());\n    Resection_L1_ConstraintBuilder cstBuilder(pt2D, XPoints);\n    if ((BisectionLP<Resection_L1_ConstraintBuilder, LPConstraintsSparse>(wrapperLpSolve, cstBuilder, &vec_solution, 1.0, 0.0)))"
        "    throw std::runtime_error(\"l1SixPointResectionSolver requires Coin-OR Clp (not built on this platform)\");\n    if (false)"
        _linf_content "${_linf_content}")
    file(WRITE "${_LINF_PATCHED}" "${_linf_content}")

    add_subdirectory(upstream/src/aliceVision/linearProgramming)

    # Swap resection_kernel.cpp for the patched copy + extend includes so
    # the patched file finds sibling headers under the original dir.
    get_target_property(_linf_srcs aliceVision_lInftyComputerVision SOURCES)
    list(TRANSFORM _linf_srcs REPLACE
        "resection_kernel\\.cpp$" "${_LINF_PATCHED}")
    set_target_properties(aliceVision_lInftyComputerVision PROPERTIES
        SOURCES "${_linf_srcs}")
    target_include_directories(aliceVision_lInftyComputerVision PRIVATE
        "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/linearProgramming/lInfinityCV")
    add_subdirectory(upstream/src/aliceVision/geometry)
    add_subdirectory(upstream/src/aliceVision/camera)
    add_subdirectory(upstream/src/aliceVision/robustEstimation)

    # Patch multiview/rotationAveraging/l1.cpp before adding the
    # subdirectory. Upstream's `const Eigen::Matrix x;` pattern
    # (followed by `(double*)x.data()` cast-mutation) is rejected
    # by clang 21 even in C++17 mode — the rule that a const-
    # qualified class object requires a user-provided default ctor
    # is in [dcl.init] since C++11 but enforcement tightened. Three
    # offending sites, two unique strings. We can't edit
    # `upstream/` (read-only reference), so we copy the file to
    # `${CMAKE_BINARY_DIR}/upstream-patched/` with the const
    # dropped, then swap it into the multiview target.
    set(_L1_SRC     "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/multiview/rotationAveraging/l1.cpp")
    set(_L1_PATCHED "${CMAKE_BINARY_DIR}/upstream-patched/multiview/rotationAveraging/l1.cpp")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/upstream-patched/multiview/rotationAveraging")
    file(READ "${_L1_SRC}" _l1_content)
    string(REPLACE "const aliceVision::Vec3 erij;"
                   "aliceVision::Vec3 erij;" _l1_content "${_l1_content}")
    string(REPLACE "const Mat3 eRi;"
                   "Mat3 eRi;" _l1_content "${_l1_content}")
    file(WRITE "${_L1_PATCHED}" "${_l1_content}")

    add_subdirectory(upstream/src/aliceVision/multiview)

    # Swap the original l1.cpp source for our patched copy. Also
    # add the original rotationAveraging/ dir to the multiview
    # target's PRIVATE includes — the patched file lives elsewhere
    # but `#include "l1.hpp"` still expects the sibling header.
    get_target_property(_mv_srcs aliceVision_multiview SOURCES)
    list(TRANSFORM _mv_srcs REPLACE
        "rotationAveraging/l1\\.cpp" "${_L1_PATCHED}")
    set_target_properties(aliceVision_multiview PROPERTIES
        SOURCES "${_mv_srcs}")
    target_include_directories(aliceVision_multiview PRIVATE
        "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/multiview/rotationAveraging")
    add_subdirectory(upstream/src/aliceVision/mvsData)
    add_subdirectory(upstream/src/aliceVision/mvsUtils)
    # S38 W3 — CMake-time patch of gpu/gpu.cpp.
    # `gpuSupportCUDA()` returns false when ALICEVISION_HAVE_CUDA=0, which
    # blocks `aliceVision_depthMapEstimation` from running. On Apple
    # Silicon Metal builds we DO have a usable GPU; report success when
    # AV_HAVE_DEPTHMAP is set. Patches `bool gpuSupportCUDA(...)` to
    # short-circuit `return true;` after the ALICEVISION_HAVE_CUDA guard
    # bails to the no-CUDA path.
    set(_GPU_SRC     "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/gpu/gpu.cpp")
    set(_GPU_PATCHED "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/gpu/gpu.cpp")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/gpu")
    file(READ "${_GPU_SRC}" _gpu_content)
    string(REPLACE
        "information = \"AliceVision built without CUDA support.\";"
        "information = \"AliceVision built with Metal backend (Apple Silicon).\";"
        _gpu_content "${_gpu_content}")
    # The full-function-body replacement of `gpuSupportCUDA` lives outside
    # of the original `#if` so it works whether or not HAVE_CUDA is set.
    string(REPLACE
        "bool gpuSupportCUDA(int minComputeCapabilityMajor, int minComputeCapabilityMinor, int minTotalDeviceMemory)\n{"
        "bool gpuSupportCUDA(int minComputeCapabilityMajor, int minComputeCapabilityMinor, int minTotalDeviceMemory)\n{\n#if AV_HAVE_DEPTHMAP\n    ALICEVISION_LOG_INFO(\"Apple Silicon Metal-backed GPU detected (Path: aliceVision_depthMap).\");\n    return true;\n#endif"
        _gpu_content "${_gpu_content}")
    file(WRITE "${_GPU_PATCHED}" "${_gpu_content}")

    add_subdirectory(upstream/src/aliceVision/gpu)

    # Swap gpu.cpp for our patched copy + extend includes so the
    # patched file (which lives outside upstream/) can find sibling
    # gpu.hpp via "gpu.hpp" relative include.
    get_target_property(_gpu_srcs aliceVision_gpu SOURCES)
    list(TRANSFORM _gpu_srcs REPLACE "gpu\\.cpp$" "${_GPU_PATCHED}")
    set_target_properties(aliceVision_gpu PROPERTIES SOURCES "${_gpu_srcs}")
    target_include_directories(aliceVision_gpu PRIVATE
        ${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/gpu)

    # ---------- S36: sensorDB + feature cascade ----------
    #
    # `aliceVision_sensorDB` only pulls in Boost::system (stub) + Boost::boost
    # (umbrella header target). Its sources include
    # `aliceVision/utils/filesIO.hpp` which is header-only, so we don't need
    # to add the `utils` subdirectory — the upstream include root already
    # resolves the path.
    add_subdirectory(upstream/src/aliceVision/sensorDB)

    # `aliceVision_feature` links `aliceVision_image`, `aliceVision_numeric`,
    # `aliceVision_system`, `aliceVision_gpu`, plus the vendored `vlsift`
    # static library at `upstream/src/nonFree/sift/`. CCTag/AprilTag/PopSIFT/
    # OpenCV image-describers are gated on `ALICEVISION_HAVE_*` flags that
    # we keep at 0 — those sources stay out of the build.
    #
    # Add the nonFree/sift dir first so the `vlsift` target exists when
    # `aliceVision_feature` tries to link it.
    add_subdirectory(upstream/src/nonFree/sift)
    add_subdirectory(upstream/src/aliceVision/feature)

    # S37 — sfmData + sfmDataIO. Last cascade step before upstream
    # depthMap host code can be linked.
    add_subdirectory(upstream/src/aliceVision/sfmData)

    # sfmDataIO needs assimp (Homebrew has it) + Boost.regex.
    find_package(assimp CONFIG REQUIRED)
    if(NOT TARGET Boost::regex)
        find_package(Boost CONFIG REQUIRED COMPONENTS regex)
    endif()
    add_subdirectory(upstream/src/aliceVision/sfmDataIO)

    # S38 — cmdline module (Boost.program_options wrapper).
    # Single source. Needed by pipeline binaries.
    if(NOT TARGET Boost::program_options)
        find_package(Boost CONFIG REQUIRED COMPONENTS program_options)
    endif()
    add_subdirectory(upstream/src/aliceVision/cmdline)

    # ----------------------------------------------------------------
    # S39 — Phase 2 SfM cascade: module deps
    # ----------------------------------------------------------------
    # New modules required by cameraInit / featureExtraction /
    # imageMatching / featureMatching / incrementalSfM /
    # prepareDenseScene pipeline binaries.

    # External dependencies introduced by S39 modules:
    #   * expat   — XML parser used by lensCorrectionProfile/lcp.cpp.
    #   * flann   — kdtree/knn library used by matching's
    #               ArrayMatcher_kdtreeFlann + matchingImageCollection.
    #               Upstream pairs flann with lz4 (FLANN_LIBRARIES var).
    #   * Boost::timer — perf timers in matchingImageCollection and the
    #                    featureExtraction/prepareDenseScene binaries.
    #   * Boost::json  — JSON I/O used by track/dataio/matchingImageCollection.
    # Homebrew installs expat's CMake config under
    # `<prefix>/lib/cmake/expat-<version>/` — the unversioned `lib/cmake/`
    # tree is what CMake usually scans, so we hint the version directory
    # explicitly. Match upstream's lowercase name; the alias target
    # `expat::expat` works either way.
    if(NOT expat_DIR)
        file(GLOB _expat_cfg LIST_DIRECTORIES TRUE
             "${HOMEBREW_PREFIX}/opt/expat/lib/cmake/expat-*")
        if(_expat_cfg)
            list(GET _expat_cfg 0 expat_DIR)
        endif()
    endif()
    find_package(expat CONFIG REQUIRED)
    # Homebrew's lz4 keg does not ship a CMake config (only pkg-config).
    # Hand-roll an IMPORTED target `lz4::lz4` matching upstream's
    # CMake target name so `set(FLANN_LIBRARIES ... lz4::lz4)` below works.
    if(NOT TARGET lz4::lz4)
        find_path(LZ4_INCLUDE_DIR lz4.h HINTS ${HOMEBREW_PREFIX}/opt/lz4/include)
        find_library(LZ4_LIBRARY  lz4   HINTS ${HOMEBREW_PREFIX}/opt/lz4/lib)
        if(NOT LZ4_INCLUDE_DIR OR NOT LZ4_LIBRARY)
            message(FATAL_ERROR "lz4 not found via Homebrew opt/lz4 prefix")
        endif()
        add_library(lz4::lz4 SHARED IMPORTED)
        set_target_properties(lz4::lz4 PROPERTIES
            IMPORTED_LOCATION             "${LZ4_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}"
        )
    endif()
    find_package(flann CONFIG REQUIRED)
    if(TARGET lz4::lz4)
        set(FLANN_LIBRARIES flann::flann_cpp lz4::lz4)
    elseif(TARGET LZ4::lz4_static)
        set(FLANN_LIBRARIES flann::flann_cpp LZ4::lz4_static)
    elseif(TARGET LZ4::lz4_shared)
        set(FLANN_LIBRARIES flann::flann_cpp LZ4::lz4_shared)
    else()
        message(FATAL_ERROR "FLANN cannot be linked: no lz4 target")
    endif()
    if(NOT TARGET Boost::timer)
        find_package(Boost CONFIG REQUIRED COMPONENTS timer)
    endif()
    if(NOT TARGET Boost::json)
        find_package(Boost CONFIG REQUIRED COMPONENTS json)
    endif()

    # S41 — extra Boost components + external deps for meshing pipeline.
    #   * Boost::graph — used by fuseCut for max-flow graph traversal
    #     (Tetrahedralization + GraphFiller).
    #   * Boost::container — used by fuseCut PointCloud / Mesher
    #     (flat_set, small_vector for tetrahedron neighbour lookup).
    #   * nanoflann — kd-tree library for fuseCut PointCloud nearest-
    #     neighbour queries during depth map fusion.
    #   * OpenMesh (OpenMeshCore) — half-edge mesh library used by
    #     aliceVision_mesh (MeshClean / MeshAnalyze).
    #   * Geogram — CDT (constrained Delaunay triangulation) backend
    #     for fuseCut, plus restricted Voronoi for meshing.
    if(NOT TARGET Boost::graph)
        find_package(Boost CONFIG REQUIRED COMPONENTS graph)
    endif()
    if(NOT TARGET Boost::container)
        find_package(Boost CONFIG REQUIRED COMPONENTS container)
    endif()
    find_package(nanoflann CONFIG REQUIRED)
    find_package(OpenMesh CONFIG REQUIRED)
    # Geogram has no CMake config on Homebrew, only a MODULE finder
    # at /opt/homebrew/share/cmake/Modules/FindGeogram.cmake. Use the
    # MODULE mode (no CONFIG keyword).
    find_package(Geogram REQUIRED)

    # Module add order (respect PUBLIC_LINKS chain):
    #   lensCorrectionProfile → (feature already done) used by cameraInit
    #   voctree → needs feature, sfmData, system; used by imageMatching lib
    #   matching → needs camera, feature, numeric, stl, system + FLANN
    #              (adds kvld via add_subdirectory; ignored here, no caller uses it)
    #   matchingImageCollection → matching + multiview + robustEstimation + sfmData
    #   imageMatching (module) → voctree + image
    #   featureEngine → feature + image + sfmData + system
    #   graph (INTERFACE) → system   (header-only deps for sfm)
    #   track → feature + matching + stl + Boost::json
    #   dataio → camera + image + sfmData + sfmDataIO + system + Boost::json
    #   sfm → all of the above plus multiview/geometry/camera/lInftyComputerVision
    #
    # Note: aliceVision_lInftyComputerVision is already built via the
    # linearProgramming/CMakeLists.txt add_subdirectory(lInfinityCV) call.
    add_subdirectory(upstream/src/aliceVision/lensCorrectionProfile)
    add_subdirectory(upstream/src/aliceVision/voctree)
    add_subdirectory(upstream/src/aliceVision/matching)
    add_subdirectory(upstream/src/aliceVision/matchingImageCollection)
    add_subdirectory(upstream/src/aliceVision/imageMatching)
    add_subdirectory(upstream/src/aliceVision/featureEngine)
    add_subdirectory(upstream/src/aliceVision/graph)
    add_subdirectory(upstream/src/aliceVision/track)
    add_subdirectory(upstream/src/aliceVision/dataio)
    add_subdirectory(upstream/src/aliceVision/sfm)

    # S41 — mesh / fuseCut / sfmMvsUtils for meshing+texturing pipeline
    # Link chain:
    #   mesh        → mvsData + mvsUtils + Geogram + assimp + OpenMeshCore
    #   fuseCut     → mesh + sfmData + nanoflann + Boost::graph/container/json
    #   sfmMvsUtils → mesh + mvsData + sfmData
    add_subdirectory(upstream/src/aliceVision/mesh)
    add_subdirectory(upstream/src/aliceVision/fuseCut)
    add_subdirectory(upstream/src/aliceVision/sfmMvsUtils)

    # Upstream's sfm/CMakeLists only PRIVATE-links `${LEMON_LIBRARY}` into
    # `aliceVision_sfm`, but `aliceVision_sfm_bundle` also includes
    # `LocalBundleAdjustmentGraph.hpp` transitively (via
    # `BundleAdjustmentCeres.hpp`), which pulls in `<lemon/list_graph.h>`.
    # Forward the vendored LEMON to sfm_bundle so its TUs can find the
    # headers.
    target_link_libraries(aliceVision_sfm_bundle PUBLIC av_lemon)

    # ----------------------------------------------------------------
    # S38 — CMake-time patches for depthMap host headers
    # ----------------------------------------------------------------
    # `Sgm.hpp` and `Refine.hpp` declare `~Sgm() = default;` /
    # `~Refine() = default;` explicitly. Under libc++ (Apple Clang)
    # that user-declared destructor suppresses the implicit move
    # constructor; combined with non-copyable `CudaDeviceMemoryPitched`
    # members the class becomes neither MoveConstructible nor
    # CopyConstructible, breaking `std::vector<Sgm/Refine>::reserve`
    # at instantiation. libstdc++ is lazier and accepts this. We add
    # an explicit defaulted move ctor; reference members make move
    # assignment ill-formed (and unused).
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/depthMap")
    foreach(_hdr Sgm Refine)
        set(_orig "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/depthMap/${_hdr}.hpp")
        set(_pat  "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/depthMap/${_hdr}.hpp")
        file(READ "${_orig}" _src)
        string(REPLACE "~${_hdr}() = default;"
                       "~${_hdr}() = default; ${_hdr}(${_hdr}&&) noexcept = default;"
                       _src "${_src}")
        file(WRITE "${_pat}" "${_src}")
    endforeach()

    # ----------------------------------------------------------------
    # S38 — Upstream depthMap host library
    # ----------------------------------------------------------------
    # Custom library target instead of `add_subdirectory(upstream/.../
    # depthMap)`: upstream's CMakeLists declares USE_CUDA and the .cu
    # sources, which our shim's `alicevision_add_library` rejects. We
    # source the 9 host .cpp's directly, pull cuda_* function symbols
    # from `av_depth_map_metal` + `av_cuda_host_shim`, and stub
    # CUDA primitive types via `cmake/shims/cuda-stubs/cuda_fp16.h`
    # plus the existing memory.hpp / DeviceMipmapImage.hpp shims.
    set(_av_dm_dir upstream/src/aliceVision/depthMap)
    add_library(aliceVision_depthMap STATIC
        ${_av_dm_dir}/computeOnMultiGPUs.cpp
        ${_av_dm_dir}/CustomPatchPatternParams.cpp
        ${_av_dm_dir}/DepthMapEstimator.cpp
        ${_av_dm_dir}/depthMapUtils.cpp
        ${_av_dm_dir}/NormalMapEstimator.cpp
        ${_av_dm_dir}/Refine.cpp
        ${_av_dm_dir}/Sgm.cpp
        ${_av_dm_dir}/SgmDepthList.cpp
        ${_av_dm_dir}/volumeIO.cpp
    )
    target_include_directories(aliceVision_depthMap
        PUBLIC
            # Shim path FIRST so cuda/host/{memory,DeviceMipmapImage,
            # DeviceCache,DeviceStreamManager,patchPattern,utils}.hpp
            # resolve to our overrides, not upstream's.
            ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
            # cuda_fp16.h stub.
            ${CMAKE_SOURCE_DIR}/cmake/shims/cuda-stubs
            # Patched depthMap host headers (Sgm.hpp, Refine.hpp with
            # explicit move ctors). Must come BEFORE the unpatched
            # upstream/src path.
            ${CMAKE_BINARY_DIR}/upstream-patched
            # generated config.hpp.
            ${CMAKE_BINARY_DIR}/upstream-generated
            # upstream's own headers (declarations: similarity.hpp,
            # deviceSimilarityVolume.hpp, etc. — definitions come
            # from av_depth_map_metal via the cuda_* adapter).
            ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_depthMap PUBLIC
        ${AV_EIGEN_DEFINITIONS}
        ALICEVISION_ROTATION_AVERAGING_WITH_BOOST
    )
    target_link_libraries(aliceVision_depthMap
        PUBLIC
            aliceVision_mvsData
            aliceVision_mvsUtils
            aliceVision_system
            aliceVision_sfmData
            aliceVision_sfmDataIO
            assimp::assimp
        PRIVATE
            aliceVision_gpu
            av_depth_map_metal
            av_cuda_host_shim
    )

    # ----------------------------------------------------------------
    # S38 — Pipeline binary: aliceVision_depthMapEstimation
    # ----------------------------------------------------------------
    # First end-user CLI tool. Validates the linked depthMap library by
    # invoking it through upstream's actual main file. Matches the binary
    # name Meshroom expects (Phase 11 unblock).
    add_executable(aliceVision_depthMapEstimation
        upstream/src/software/pipeline/main_depthMapEstimation.cpp
    )
    target_include_directories(aliceVision_depthMapEstimation PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_SOURCE_DIR}/cmake/shims/cuda-stubs
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_depthMapEstimation PRIVATE
        ${AV_EIGEN_DEFINITIONS}
    )
    target_link_libraries(aliceVision_depthMapEstimation PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_mvsUtils
        aliceVision_depthMap
        aliceVision_gpu
    )
    # Stage default.metallib next to the binary so Device::load_library({})
    # finds it via "next-to-executable" lookup.
    av_install_metallib(
        FROM       av_shaders
        EXECUTABLE aliceVision_depthMapEstimation
    )

    # ----------------------------------------------------------------
    # S38 W3 — Pipeline binary: aliceVision_importMiddlebury
    # ----------------------------------------------------------------
    # Single-step ingest of Middlebury MVS scenes (templeRing,
    # templeSparseRing, dinoRing, etc.) into an SfMData JSON that the
    # rest of the pipeline can consume. Avoids the full cameraInit +
    # featureExtraction + SfM cascade because Middlebury ships
    # calibrated cameras + extrinsics in its .par files.
    add_executable(aliceVision_importMiddlebury
        upstream/src/software/utils/main_importMiddlebury.cpp
    )
    target_include_directories(aliceVision_importMiddlebury PRIVATE
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_importMiddlebury PRIVATE
        ${AV_EIGEN_DEFINITIONS}
    )
    target_link_libraries(aliceVision_importMiddlebury PRIVATE
        aliceVision_numeric
        aliceVision_geometry
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_system
        aliceVision_cmdline
    )

    # ----------------------------------------------------------------
    # S39 — Phase 2 SfM cascade: pipeline binaries
    # ----------------------------------------------------------------
    # Six binaries, run in sequence:
    #   cameraInit → featureExtraction → imageMatching → featureMatching
    #   → incrementalSfM → prepareDenseScene → (depthMapEstimation already built)
    #
    # Each binary follows the depthMapEstimation pattern:
    #   * PRIVATE includes: shim path FIRST, then patched, then generated,
    #     then upstream/src.
    #   * AV_EIGEN_DEFINITIONS for alignment-disabled Eigen.
    #   * PRIVATE link lines mirror upstream's
    #     software/pipeline/CMakeLists.txt.

    # (1) cameraInit — EXIF + sensor DB → SfMData with cameras.
    add_executable(aliceVision_cameraInit
        upstream/src/software/pipeline/main_cameraInit.cpp
    )
    target_include_directories(aliceVision_cameraInit PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_cameraInit PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_cameraInit PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_sensorDB
        aliceVision_image
        aliceVision_feature
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_lensCorrectionProfile
        Boost::program_options
        Boost::boost
    )

    # (2) featureExtraction — SIFT/AKAZE features per view.
    add_executable(aliceVision_featureExtraction
        upstream/src/software/pipeline/main_featureExtraction.cpp
    )
    target_include_directories(aliceVision_featureExtraction PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_featureExtraction PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_featureExtraction PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_gpu
        aliceVision_image
        aliceVision_feature
        aliceVision_featureEngine
        aliceVision_multiview
        aliceVision_sfmData
        aliceVision_sfmDataIO
        vlsift
        Boost::program_options
        Boost::boost
        Boost::timer
    )

    # (3) imageMatching — image pair candidate selection.
    # NOTE: target name collision avoidance — upstream's library is also
    # called `aliceVision_imageMatching` (built earlier). CMake forbids two
    # targets sharing a name in the same project, so we name the executable
    # target `aliceVision_imageMatching_bin` and override its on-disk
    # filename to `aliceVision_imageMatching` to match Meshroom's
    # expectation.
    add_executable(aliceVision_imageMatching_bin
        upstream/src/software/pipeline/main_imageMatching.cpp
    )
    set_target_properties(aliceVision_imageMatching_bin PROPERTIES
        OUTPUT_NAME aliceVision_imageMatching)
    target_include_directories(aliceVision_imageMatching_bin PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_imageMatching_bin PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_imageMatching_bin PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_imageMatching
        aliceVision_matchingImageCollection
        aliceVision_sfm
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_voctree
        Boost::program_options
    )

    # (4) featureMatching — actual descriptor matching.
    add_executable(aliceVision_featureMatching
        upstream/src/software/pipeline/main_featureMatching.cpp
    )
    target_include_directories(aliceVision_featureMatching PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_featureMatching PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_featureMatching PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_feature
        aliceVision_multiview
        aliceVision_matchingImageCollection
        aliceVision_sfm
        aliceVision_sfmData
        aliceVision_sfmDataIO
        Boost::program_options
    )

    # (5) incrementalSfM — SfM reconstruction (cameras + landmarks).
    add_executable(aliceVision_incrementalSfM
        upstream/src/software/pipeline/main_incrementalSfM.cpp
    )
    target_include_directories(aliceVision_incrementalSfM PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_incrementalSfM PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_incrementalSfM PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_image
        aliceVision_feature
        aliceVision_sfm
        aliceVision_sfmData
        aliceVision_sfmDataIO
        Boost::program_options
    )

    # (6) prepareDenseScene — undistortion + scene prep for MVS.
    add_executable(aliceVision_prepareDenseScene
        upstream/src/software/pipeline/main_prepareDenseScene.cpp
    )
    target_include_directories(aliceVision_prepareDenseScene PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_prepareDenseScene PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_prepareDenseScene PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_image
        aliceVision_feature
        aliceVision_sfmData
        aliceVision_sfmDataIO
        Boost::program_options
        Boost::boost
        Boost::timer
    )

    # ----------------------------------------------------------------
    # S41 — Meshing pipeline binaries (4)
    # ----------------------------------------------------------------
    # End-to-end output: textured 3D mesh. Pipeline order:
    #   depthMapFiltering → meshing → meshFiltering → texturing
    # All link upstream/src/software/pipeline/main_*.cpp against the
    # mesh/fuseCut/sfmMvsUtils libs added above.

    # (7) depthMapFiltering — post-MVS depth map refinement.
    add_executable(aliceVision_depthMapFiltering
        upstream/src/software/pipeline/main_depthMapFiltering.cpp
    )
    target_include_directories(aliceVision_depthMapFiltering PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_SOURCE_DIR}/cmake/shims/cuda-stubs
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_depthMapFiltering PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_depthMapFiltering PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_mvsData
        aliceVision_mvsUtils
        aliceVision_fuseCut
        aliceVision_depthMap
        aliceVision_sfmData
        aliceVision_sfmDataIO
        Boost::program_options
    )

    # (8) meshing — depth maps → 3D mesh (fusion + Delaunay).
    add_executable(aliceVision_meshing
        upstream/src/software/pipeline/main_meshing.cpp
    )
    target_include_directories(aliceVision_meshing PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_meshing PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_meshing PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_mvsData
        aliceVision_mvsUtils
        aliceVision_mesh
        aliceVision_fuseCut
        aliceVision_sfmData
        aliceVision_sfmDataIO
        Boost::program_options
    )

    # (9) meshFiltering — mesh cleanup (decimation, smoothing).
    add_executable(aliceVision_meshFiltering
        upstream/src/software/pipeline/main_meshFiltering.cpp
    )
    target_include_directories(aliceVision_meshFiltering PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_meshFiltering PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_meshFiltering PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_mvsUtils
        aliceVision_mesh
        Boost::program_options
    )

    # (10) texturing — UV unwrap + texture from source images.
    add_executable(aliceVision_texturing
        upstream/src/software/pipeline/main_texturing.cpp
    )
    target_include_directories(aliceVision_texturing PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_texturing PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_texturing PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_mvsData
        aliceVision_sfmMvsUtils
        aliceVision_mvsUtils
        aliceVision_mesh
        aliceVision_sfmData
        aliceVision_sfmDataIO
        Boost::program_options
    )

    # ----------------------------------------------------------------
    # Phase 12 — Install rules (Homebrew packaging)
    # ----------------------------------------------------------------
    # Layout (matches the Homebrew formula expectation):
    #   <prefix>/bin/aliceVision_*           (12 pipeline binaries)
    #   <prefix>/bin/default.metallib        (Metal shader archive,
    #                                        sibling of the binaries so
    #                                        Device::load_library({})
    #                                        finds it via @executable_path)
    #   <prefix>/share/aliceVision/
    #       config.ocio
    #       luts/...
    #       cameraSensors.db
    #
    # Notes:
    #   * aliceVision_imageMatching_bin's on-disk filename is overridden
    #     to `aliceVision_imageMatching` via OUTPUT_NAME (see S39); the
    #     install rule below picks that up automatically through
    #     install(TARGETS).
    #   * default.metallib is generated by the `av_shaders` custom target
    #     and staged next to executables at build time via
    #     av_install_metallib(). We re-install one canonical copy here
    #     so cmake --install picks it up regardless of which binary
    #     happens to be the post-build trigger.
    #   * Codesigning runs as a CMake install-time post-step (see below).
    set(AV_PIPELINE_BINARIES
        aliceVision_cameraInit
        aliceVision_featureExtraction
        aliceVision_imageMatching_bin       # OUTPUT_NAME = aliceVision_imageMatching
        aliceVision_featureMatching
        aliceVision_incrementalSfM
        aliceVision_prepareDenseScene
        aliceVision_depthMapEstimation
        aliceVision_depthMapFiltering
        aliceVision_meshing
        aliceVision_meshFiltering
        aliceVision_texturing
        aliceVision_importMiddlebury
    )

    install(TARGETS ${AV_PIPELINE_BINARIES}
        RUNTIME DESTINATION bin
    )

    # default.metallib is produced into ${CMAKE_BINARY_DIR}/default.metallib
    # by src/shaders/CMakeLists.txt (av_shaders custom target).
    get_target_property(_av_metallib_path av_shaders AV_METALLIB_PATH)
    if(_av_metallib_path)
        install(FILES "${_av_metallib_path}"
            DESTINATION bin
            RENAME default.metallib
        )
    else()
        message(WARNING
            "av_shaders has no AV_METALLIB_PATH — default.metallib will not "
            "be installed. Check src/shaders/CMakeLists.txt.")
    endif()

    # Runtime data: OCIO config + LUTs + sensor DB. Upstream is read-only
    # so we install directly from upstream/src/aliceVision/.
    install(DIRECTORY
        "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/image/share/aliceVision/"
        DESTINATION share/aliceVision
    )
    install(FILES
        "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/sensorDB/cameraSensors.db"
        DESTINATION share/aliceVision
    )

    # S48 audit: ship LICENSE in the install layout. Required for
    # downstream packaging (Homebrew formula, distribution tarballs).
    if(EXISTS "${CMAKE_SOURCE_DIR}/LICENSE")
        install(FILES "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION share/aliceVision)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/upstream/COPYING.md")
        install(FILES "${CMAKE_SOURCE_DIR}/upstream/COPYING.md"
            DESTINATION share/aliceVision RENAME LICENSE)
    endif()

    # S48 audit: dSYM bundles for debugging. CMake builds with
    # CMAKE_BUILD_TYPE=RelWithDebInfo by default (see project options
    # block) — clang -g emits inline DWARF rather than a separate
    # .dSYM. We use `dsymutil` at install time to extract DWARF into
    # side-by-side `<binary>.dSYM/` bundles in the install dir. dSYMs
    # are large; bundle them into `lib/dSYM/` so the bin/ layout stays
    # clean for end users who don't want debug symbols.
    set(_av_dsym_dir "${CMAKE_INSTALL_PREFIX}/lib/dSYM")
    foreach(_av_bin IN LISTS AV_PIPELINE_BINARIES)
        get_target_property(_av_outname ${_av_bin} OUTPUT_NAME)
        if(NOT _av_outname)
            set(_av_outname "${_av_bin}")
        endif()
        install(CODE "
            set(_bin \"\${CMAKE_INSTALL_PREFIX}/bin/${_av_outname}\")
            set(_dsym_out \"\${CMAKE_INSTALL_PREFIX}/lib/dSYM/${_av_outname}.dSYM\")
            file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/lib/dSYM\")
            message(STATUS \"Extracting dSYM: \${_bin} → \${_dsym_out}\")
            execute_process(
                COMMAND dsymutil -o \"\${_dsym_out}\" \"\${_bin}\"
                RESULT_VARIABLE _dsym_rc
            )
            if(NOT _dsym_rc EQUAL 0)
                message(WARNING \"dsymutil failed for \${_bin} (rc=\${_dsym_rc}); \"
                                \"install completed but debug symbols are not bundled.\")
            endif()
        ")
    endforeach()

    # ----------------------------------------------------------------
    # Phase 12 — Ad-hoc codesigning at install time
    # ----------------------------------------------------------------
    # macOS Gatekeeper + library validation requires every binary to be
    # signed. For dev distribution an ad-hoc signature ("-" identity)
    # is sufficient; Developer-ID + notarization is a follow-up task.
    #
    # We emit one install(CODE) block per binary so the codesign runs
    # AFTER install(TARGETS) has placed the file at its final path and
    # CMake has fixed up the RPATHs. Running codesign earlier would
    # invalidate the signature when RPATH rewriting happens.
    foreach(_av_bin IN LISTS AV_PIPELINE_BINARIES)
        # OUTPUT_NAME may differ from the target name (see imageMatching_bin).
        get_target_property(_av_outname ${_av_bin} OUTPUT_NAME)
        if(NOT _av_outname)
            set(_av_outname "${_av_bin}")
        endif()
        install(CODE "
            set(_bin \"\${CMAKE_INSTALL_PREFIX}/bin/${_av_outname}\")
            message(STATUS \"Codesigning (ad-hoc): \${_bin}\")
            execute_process(
                COMMAND codesign --sign - --deep --force \"\${_bin}\"
                RESULT_VARIABLE _cs_rc
            )
            if(NOT _cs_rc EQUAL 0)
                message(WARNING \"codesign failed for \${_bin} (rc=\${_cs_rc}); \"
                                \"the install completed but the binary may not run \"
                                \"under hardened-runtime enforcement.\")
            endif()
        ")
    endforeach()

    # ----------------------------------------------------------------
    # Phase 12 — `package` custom target
    # ----------------------------------------------------------------
    # One-shot release packaging: stages a clean install tree under
    # build/release/alicevision-for-mac-<VERSION>/ and tars it as
    # build/release/alicevision-for-mac-<VERSION>-arm64.tar.gz.
    #
    # Usage (from the build/ directory):
    #   ninja package           # Ninja generator
    #   cmake --build . --target package
    #
    # The tarball ships binaries + default.metallib + share/ only.
    # Homebrew runtime deps (boost/eigen/openimageio/etc.) are NOT
    # vendored — consumers are expected to `brew install` them per
    # the Formula. The binaries inside are ad-hoc codesigned (see
    # install(CODE ...) block above); a Developer-ID + notarized
    # tarball is a follow-up task.
    set(AV_RELEASE_NAME "alicevision-for-mac-${PROJECT_VERSION}")
    set(AV_RELEASE_STAGE "${CMAKE_BINARY_DIR}/release/${AV_RELEASE_NAME}")
    set(AV_RELEASE_TARBALL "${CMAKE_BINARY_DIR}/release/${AV_RELEASE_NAME}-arm64.tar.gz")
    set(AV_RELEASE_DSYM_TARBALL "${CMAKE_BINARY_DIR}/release/${AV_RELEASE_NAME}-arm64-dSYM.tar.gz")
    # Pre-create the release/ dir at configure time so WORKING_DIRECTORY
    # can resolve to it from the first run.
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/release")
    add_custom_target(package
        # 1. Wipe any previous stage so we never tar a stale tree.
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${AV_RELEASE_STAGE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/release"
        # 2. Run the full install pipeline (TARGETS + metallib + share/
        #    + codesign install(CODE) blocks + dSYM extraction) into
        #    the staging dir.
        COMMAND ${CMAKE_COMMAND}
                --install "${CMAKE_BINARY_DIR}"
                --prefix "${AV_RELEASE_STAGE}"
        # 3. Move the dSYM bundles OUT of the main install tree before
        #    tarring — dSYMs are ~12× larger than binaries (S48 audit:
        #    184 MB with dSYM vs 15 MB without). Ship them as a SEPARATE
        #    tarball so end users get the small binary tarball and
        #    developers/crash-investigators can grab the dSYM tarball
        #    when they need symbolicated backtraces.
        COMMAND ${CMAKE_COMMAND} -E rename
                "${AV_RELEASE_STAGE}/lib/dSYM"
                "${CMAKE_BINARY_DIR}/release/${AV_RELEASE_NAME}-dSYM"
        # 4. Tar the binary tree (small). -E tar uses libarchive
        #    (built into CMake) so this works the same on BSD and GNU.
        COMMAND ${CMAKE_COMMAND} -E tar czf
                "${AV_RELEASE_TARBALL}"
                "${AV_RELEASE_NAME}"
        # 5. Tar the dSYM tree (large, separate).
        COMMAND ${CMAKE_COMMAND} -E tar czf
                "${AV_RELEASE_DSYM_TARBALL}"
                "${AV_RELEASE_NAME}-dSYM"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/release"
        COMMENT "Building release tarball: ${AV_RELEASE_TARBALL} + dSYM tarball: ${AV_RELEASE_DSYM_TARBALL}"
        USES_TERMINAL
        VERBATIM
    )
endif()

if(AV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------- Summary ----------

message(STATUS "")
message(STATUS "================================================================")
message(STATUS "alicevision-for-mac configuration summary")
message(STATUS "================================================================")
message(STATUS "  Build type             : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard           : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Architectures          : ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "  Deployment target      : ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "  AV_USE_METAL           : ${AV_USE_METAL}")
message(STATUS "  AV_USE_CUDA            : ${AV_USE_CUDA}")
message(STATUS "  AV_BUILD_TESTS         : ${AV_BUILD_TESTS}")
message(STATUS "  AV_BUILD_HELLO_METAL   : ${AV_BUILD_HELLO_METAL}")
message(STATUS "  AV_BUILD_UPSTREAM      : ${AV_BUILD_UPSTREAM}")
message(STATUS "  AV_BUILD_UPSTREAM_DEPTHMAP : ${AV_BUILD_UPSTREAM_DEPTHMAP}")
if(AV_BUILD_UPSTREAM_DEPTHMAP)
    message(STATUS "  ALICEVISION_HAVE_DEPTHMAP : ${ALICEVISION_HAVE_DEPTHMAP}")
endif()
message(STATUS "  AV_USE_HOMEBREW_DEPS   : ${AV_USE_HOMEBREW_DEPS}")
message(STATUS "  Install prefix         : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Metal SDK              : ${CMAKE_OSX_SYSROOT}")
message(STATUS "  metal-cpp include      : ${METAL_CPP_INCLUDE_DIR}")
message(STATUS "================================================================")
message(STATUS "")
