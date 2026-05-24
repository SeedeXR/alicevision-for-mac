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
option(AV_BUILD_PYALICEVISION    "Build C++ pyalicevision SWIG bindings (Phase 13)" ON)

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

    # Phase 14.6 — Alembic. Unlocks exportAlembic + exportAnimatedCamera
    # binaries (and therefore 8 Meshroom templates that depend on Alembic
    # cameras / point clouds). Homebrew ships an `AlembicConfig.cmake` at
    # /opt/homebrew/lib/cmake/Alembic which provides the Alembic::Alembic
    # imported target. Setting ALICEVISION_HAVE_ALEMBIC=1 BEFORE the
    # add_subdirectory(upstream/src/aliceVision/sfmDataIO) call below
    # causes the sublib's own `if (ALICEVISION_HAVE_ALEMBIC) ... endif()`
    # blocks to fold AlembicExporter.cpp + AlembicImporter.cpp +
    # ExternalAlembicImporter.cpp into aliceVision_sfmDataIO and to link
    # Alembic::Alembic PRIVATE — no source-list patching required.
    # The same flag is propagated into aliceVision/config.hpp by the
    # configure_file() call a few lines below, so the
    # `#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_ALEMBIC)` gates in
    # AlembicExporter.cpp evaluate true at compile time.
    find_package(Alembic CONFIG REQUIRED)
    if(TARGET Alembic::Alembic)
        set(ALICEVISION_HAVE_ALEMBIC 1)
    endif()

    # Phase 13 — flip upstream's SWIG-binding switch when AV_BUILD_PYALICEVISION
    # is on. Each sublib (sfmData/hdr/sfmDataIO/camera/etc.) has a
    # `if(ALICEVISION_BUILD_SWIG_BINDING) alicevision_swig_add_library(...)`
    # branch that we want to take. SWIG + Python3 must be found before
    # we enter any of those sublib add_subdirectory() calls below.
    if(AV_BUILD_PYALICEVISION)
        find_package(SWIG REQUIRED)
        include(UseSWIG)
        find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
        set(ALICEVISION_BUILD_SWIG_BINDING ON)
        cmake_policy(SET CMP0078 NEW)
        cmake_policy(SET CMP0086 NEW)
    else()
        set(ALICEVISION_BUILD_SWIG_BINDING OFF)
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
    # OpenCV — required by photometricStereo + lightingEstimation sublibs
    # (Phase 14.4). Sublibs reference ${OpenCV_LIBS} directly so the
    # find_package must run BEFORE add_subdirectory(...) calls below.
    find_package(OpenCV CONFIG REQUIRED)

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

    # Phase 14 — patch `relativePoses.hpp` to add `inline` to the two
    # `tag_invoke` free functions for `ReconstructedPair`. Upstream
    # defines them in the header WITHOUT `inline`, so every
    # translation unit that includes it (transitively via
    # `PairsScoring.hpp`) emits its own definition → duplicate-symbol
    # link errors when `main_sfmBootstrapping.cpp` links against
    # `libaliceVision_sfm.a` (which already contains the same
    # definitions via `PairsScoring.cpp`). Following the
    # `upstream-patched/` pattern already used for resection_kernel.cpp
    # and l1.cpp, we materialize a patched copy + the include path
    # `${CMAKE_BINARY_DIR}/upstream-patched/` (already on every binary's
    # include list, see CMakeLists L638 + L759 etc.) picks it up first.
    set(_RP_SRC     "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/sfm/pipeline/relativePoses.hpp")
    set(_RP_PATCHED "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/sfm/pipeline/relativePoses.hpp")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/sfm/pipeline")
    file(READ "${_RP_SRC}" _rp_content)
    string(REPLACE
        "void tag_invoke(const boost::json::value_from_tag&"
        "inline void tag_invoke(const boost::json::value_from_tag&"
        _rp_content "${_rp_content}")
    string(REPLACE
        "ReconstructedPair tag_invoke(boost::json::value_to_tag<ReconstructedPair>"
        "inline ReconstructedPair tag_invoke(boost::json::value_to_tag<ReconstructedPair>"
        _rp_content "${_rp_content}")
    file(WRITE "${_RP_PATCHED}" "${_rp_content}")

    add_subdirectory(upstream/src/aliceVision/sfm)

    # Force the sfm sublib (which compiles PairsScoring.cpp etc.) to see
    # the patched relativePoses.hpp BEFORE the unpatched original.
    # `BEFORE PRIVATE` prepends the include path so the patched copy
    # wins resolution. Without this, every TU in libaliceVision_sfm.a
    # compiles tag_invoke as non-inline → duplicate symbols when a main
    # binary that ALSO includes the header tries to link against the lib.
    target_include_directories(aliceVision_sfm BEFORE PRIVATE
        "${CMAKE_BINARY_DIR}/upstream-patched")
    target_include_directories(aliceVision_sfm_bundle BEFORE PRIVATE
        "${CMAKE_BINARY_DIR}/upstream-patched")

    # Phase 14.2 — HDR sublib (unlocks LdrToHdr* binaries + the
    # hdrFusion/panoramaHdr/panoramaFisheyeHdr templates' compute path).
    # The sublib uses our existing alicevision_add_library shim;
    # dependencies (system, image, sfmData, ceres) are all built.
    add_subdirectory(upstream/src/aliceVision/hdr)

    # Phase 14.3 — panorama sublib (unlocks panoramaWarping/Seams/
    # Compositing/etc. binaries which drive panoramaHdr,
    # panoramaFisheyeHdr templates). PUBLIC_LINKS numeric;
    # PRIVATE_LINKS system + image + camera — all built.
    add_subdirectory(upstream/src/aliceVision/panorama)

    # Phase 14.4a — photometric stereo sublibs (unlocks 2 of 3
    # photometricStereo binaries; sphereDetection deferred — it
    # upstream-requires ONNX Runtime, we'll do a CoreML port in 14.4b).
    # Both sublibs PUBLIC_LINKS ${OpenCV_LIBS} — find_package(OpenCV)
    # ran above.
    add_subdirectory(upstream/src/aliceVision/photometricStereo)
    add_subdirectory(upstream/src/aliceVision/lightingEstimation)

    # Phase 14.4b — CoreML sphereDetection sublib (av_sphere_detection).
    # Drop-in replacement for upstream's ONNX Runtime-based
    # aliceVision_sphereDetection. Provides BOTH
    # av::sphere::CoreMLSphereDetector (wrapper) AND
    # aliceVision::sphereDetection::* (upstream namespace, same API
    # except Ort::Session swapped for CoreMLSphereDetector). Links
    # CoreML + Foundation + CoreVideo frameworks. NO ONNX Runtime dep.
    add_subdirectory(src/sphere_detection)

    # Phase 14.8 — CoreML MoGe (monocular geometry) sublib (av_moge).
    # Wraps the user's ai-models/moge2_504x672_t1728.mlpackage (DINOv2
    # ViT-B/14). Used by aliceVision_moGe binary; replaces the Phase
    # 14.7 honest stub with real per-view depth + normals from the
    # neural network. Links CoreML + Foundation + OpenCV.
    add_subdirectory(src/moge)

    # Phase 14.9 — CoreML TinyRoMa dense-matcher sublib (av_roma).
    # Wraps ai-models/tiny_roma_v1_480x640.mlpackage. Used by
    # aliceVision_matchMasking, replacing the Phase 14.7 pass-through.
    # CRITICAL: the wrapper loads with MLComputeUnitsCPUAndGPU — NOT
    # `.all` — because ANE is a 4× regression for this model (see
    # ai-models/README.md). Links CoreML + Foundation + OpenCV.
    add_subdirectory(src/roma)

    # Phase 14.5 — calibration / keyframe / imageProcessing sublibs.
    # These are small, dep-clean sublibs that unlock a large number of
    # remaining utility/pipeline binaries (checkerboard/distortion/
    # colorChecker/keyframe/imageProcessing-exe). No ONNX/CUDA.
    #   calibration     → image, system, dataio, Boost, Ceres
    #   keyframe        → OpenCV (sequence-keyframe selection)
    #   imageProcessing → numeric, image, sfmData, camera, lensCorrectionProfile, OpenCV
    add_subdirectory(upstream/src/aliceVision/calibration)
    add_subdirectory(upstream/src/aliceVision/keyframe)
    add_subdirectory(upstream/src/aliceVision/imageProcessing)

    # Phase 14.5 — patch `image/conversionOpenCV.hpp` to drop the
    # `#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_OPENCV)` guard.
    # The two main_colorChecker*.cpp binaries depend on the inline
    # `imageRGBAToCvMatBGR` / `cvMatBGRToImageRGBA` helpers in that
    # header, but flipping ALICEVISION_HAVE_OPENCV=1 globally would
    # also pull OpenCV-SIFT, OpenCV-feed, OpenCV-SVG paths into
    # feature/dataio/matching sublibs — we don't want those. Patch
    # just this one header so the helpers are unconditionally
    # available; the cv:: types come from the explicit OpenCV_LIBS
    # link on the colorChecker binaries.
    set(_CVCONV_SRC     "${CMAKE_SOURCE_DIR}/upstream/src/aliceVision/image/conversionOpenCV.hpp")
    set(_CVCONV_PATCHED "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/image/conversionOpenCV.hpp")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/upstream-patched/aliceVision/image")
    file(READ "${_CVCONV_SRC}" _cvconv_content)
    string(REPLACE
        "#if ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_OPENCV)"
        "#if 1  // patched: always-on for colorChecker binaries"
        _cvconv_content "${_cvconv_content}")
    string(REPLACE
        "#endif  // ALICEVISION_HAVE_OPENCV"
        "#endif  // patched"
        _cvconv_content "${_cvconv_content}")
    file(WRITE "${_CVCONV_PATCHED}" "${_cvconv_content}")

    # The upstream CMakeLists for keyframe + imageProcessing gate OpenCV
    # linking behind `if (ALICEVISION_HAVE_OPENCV)` which we keep set to 0
    # (because flipping it would also enable OpenCV-SIFT, OpenCV-feed,
    # and OpenCV-SVG paths in feature/dataio/matching that we don't want
    # in this port). But the sublib *sources* + headers reference
    # cv::optflow / cv::ximgproc unconditionally — so attach OpenCV
    # include dirs + libs to these specific targets only.
    target_include_directories(aliceVision_keyframe PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(aliceVision_keyframe PUBLIC ${OpenCV_LIBS})
    target_include_directories(aliceVision_imageProcessing PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(aliceVision_imageProcessing PUBLIC ${OpenCV_LIBS})
    target_include_directories(aliceVision_calibration PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(aliceVision_calibration PUBLIC ${OpenCV_LIBS})

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

    # ============================================================
    # Phase 14.1 — additional pipeline binaries (2026-05-23)
    # ============================================================
    # 6 binaries that use only sublibs we already build. Adding them
    # unlocks modern-SfM templates (sfmBootstrapping + sfmExpanding +
    # relativePoseEstimating + sfmTriangulation + tracksBuilding) plus
    # mesh decimation. Each block mirrors the canonical pattern
    # established by aliceVision_meshing above.

    # (13) sfmBootstrapping — initialize SfM from feature tracks.
    add_executable(aliceVision_sfmBootstrapping
        upstream/src/software/pipeline/main_sfmBootstrapping.cpp
    )
    target_include_directories(aliceVision_sfmBootstrapping PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_sfmBootstrapping PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_sfmBootstrapping PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_sfm
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_feature
        aliceVision_track
        aliceVision_dataio
        # aliceVision_mesh transitively brings Geogram include paths,
        # required because sfmData/MeshPointFetcher.hpp includes
        # mesh/MeshIntersection.hpp which #include <geogram/mesh/mesh.h>.
        aliceVision_mesh
        Boost::program_options
    )

    # (14) sfmExpanding — incremental expansion (modern SfM core).
    add_executable(aliceVision_sfmExpanding
        upstream/src/software/pipeline/main_sfmExpanding.cpp
    )
    target_include_directories(aliceVision_sfmExpanding PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_sfmExpanding PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_sfmExpanding PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_sfm
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_track
        # See sfmBootstrapping comment: sfmData/MeshPointFetcher.hpp
        # transitively pulls geogram via aliceVision_mesh.
        aliceVision_mesh
        Boost::program_options
    )

    # (15) relativePoseEstimating — relative-pose pipeline step.
    add_executable(aliceVision_relativePoseEstimating
        upstream/src/software/pipeline/main_relativePoseEstimating.cpp
    )
    target_include_directories(aliceVision_relativePoseEstimating PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_relativePoseEstimating PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_relativePoseEstimating PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_sfm
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_track
        aliceVision_feature
        aliceVision_multiview
        aliceVision_camera
        aliceVision_geometry
        aliceVision_robustEstimation
        aliceVision_matchingImageCollection
        Boost::program_options
    )

    # (16) sfmTriangulation — triangulate landmarks from tracks.
    add_executable(aliceVision_sfmTriangulation
        upstream/src/software/pipeline/main_sfmTriangulation.cpp
    )
    target_include_directories(aliceVision_sfmTriangulation PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_sfmTriangulation PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_sfmTriangulation PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_sfm
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_feature
        aliceVision_track
        Boost::program_options
    )

    # (17) tracksBuilding — build feature tracks across views.
    add_executable(aliceVision_tracksBuilding
        upstream/src/software/pipeline/main_tracksBuilding.cpp
    )
    target_include_directories(aliceVision_tracksBuilding PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_tracksBuilding PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_tracksBuilding PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_sfm
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_feature
        aliceVision_track
        Boost::program_options
    )

    # ============================================================
    # Phase 14.2 — HDR pipeline binaries (LdrToHdr*)
    # ============================================================
    # Three binaries that drive the hdrFusion / panoramaHdr /
    # panoramaFisheyeHdr templates. Depends on alicevision_hdr sublib
    # (added above).

    # (19) LdrToHdrSampling — pick HDR-fusion-ready brackets per view.
    add_executable(aliceVision_LdrToHdrSampling
        upstream/src/software/pipeline/main_LdrToHdrSampling.cpp
    )
    target_include_directories(aliceVision_LdrToHdrSampling PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_LdrToHdrSampling PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_LdrToHdrSampling PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_image
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_hdr
        Boost::program_options
    )

    # (20) LdrToHdrCalibration — calibrate camera response from samples.
    add_executable(aliceVision_LdrToHdrCalibration
        upstream/src/software/pipeline/main_LdrToHdrCalibration.cpp
    )
    target_include_directories(aliceVision_LdrToHdrCalibration PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_LdrToHdrCalibration PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_LdrToHdrCalibration PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_image
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_hdr
        Boost::program_options
    )

    # (21) LdrToHdrMerge — merge bracket groups into HDR EXRs.
    add_executable(aliceVision_LdrToHdrMerge
        upstream/src/software/pipeline/main_LdrToHdrMerge.cpp
    )
    target_include_directories(aliceVision_LdrToHdrMerge PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_LdrToHdrMerge PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_LdrToHdrMerge PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_image
        aliceVision_sfmData
        aliceVision_sfmDataIO
        aliceVision_hdr
        Boost::program_options
    )

    # ============================================================
    # Phase 14.3 — panorama pipeline binaries (10 binaries)
    # ============================================================
    # Use a macro to cut boilerplate. Each binary needs the standard
    # set of includes + the AV_EIGEN_DEFINITIONS compile defs; only
    # link libraries vary.
    macro(av_add_pipeline_binary _name)
        add_executable(${_name}
            upstream/src/software/pipeline/main_${_name}.cpp
        )
        # The binary's main_X.cpp file lives at
        # upstream/src/software/pipeline/main_<NameWithoutPrefix>.cpp where
        # _name has the `aliceVision_` prefix. Compute the upstream stem.
        # (Handled by passing a relative-from-pipeline arg; see callers.)
    endmacro()

    # Panorama binaries — link list per file follows the include pattern:
    #   numeric+system+cmdline+image+sfmData+sfmDataIO baseline
    #   + camera (panoramaInit)
    #   + hdr (panoramaPrepareImages)
    #   + panorama (panoramaWarping/Seams/Compositing)
    #   + sfm+feature (panoramaEstimation)
    #   + sfm+track+dataio (panoramaRefining)
    #   + stl (panoramaPostProcessing)

    add_executable(aliceVision_panoramaInit upstream/src/software/pipeline/main_panoramaInit.cpp)
    target_include_directories(aliceVision_panoramaInit PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_panoramaInit PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_panoramaInit PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_camera aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options)

    add_executable(aliceVision_panoramaPrepareImages upstream/src/software/pipeline/main_panoramaPrepareImages.cpp)
    target_include_directories(aliceVision_panoramaPrepareImages PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_panoramaPrepareImages PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_panoramaPrepareImages PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_hdr
        Boost::program_options)

    add_executable(aliceVision_panoramaWarping upstream/src/software/pipeline/main_panoramaWarping.cpp)
    target_include_directories(aliceVision_panoramaWarping PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_panoramaWarping PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_panoramaWarping PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_panorama
        Boost::program_options)

    add_executable(aliceVision_panoramaEstimation upstream/src/software/pipeline/main_panoramaEstimation.cpp)
    target_include_directories(aliceVision_panoramaEstimation PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_panoramaEstimation PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_panoramaEstimation PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_sfm
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_feature
        aliceVision_mesh    # transitive geogram, like sfmBootstrapping
        Boost::program_options)

    add_executable(aliceVision_panoramaCompositing upstream/src/software/pipeline/main_panoramaCompositing.cpp)
    target_include_directories(aliceVision_panoramaCompositing PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_panoramaCompositing PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_panoramaCompositing PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_panorama
        Boost::program_options)

    add_executable(aliceVision_panoramaMerging upstream/src/software/pipeline/main_panoramaMerging.cpp)
    target_include_directories(aliceVision_panoramaMerging PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_panoramaMerging PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_panoramaMerging PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options)

    add_executable(aliceVision_panoramaSeams upstream/src/software/pipeline/main_panoramaSeams.cpp)
    target_include_directories(aliceVision_panoramaSeams PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_panoramaSeams PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_panoramaSeams PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_panorama
        Boost::program_options)

    add_executable(aliceVision_panoramaPostProcessing upstream/src/software/pipeline/main_panoramaPostProcessing.cpp)
    target_include_directories(aliceVision_panoramaPostProcessing PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_panoramaPostProcessing PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_panoramaPostProcessing PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_stl
        Boost::program_options)

    # Bonus: aliceVision_sfmTransform (utils/) — unlocks panoramaHdr +
    # panoramaFisheyeHdr templates (only remaining dep after panorama
    # sublib). Also needed by photogrammetry + cameraTracking variants.
    add_executable(aliceVision_sfmTransform upstream/src/software/utils/main_sfmTransform.cpp)
    target_include_directories(aliceVision_sfmTransform PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_sfmTransform PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_sfmTransform PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_sfm aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_track aliceVision_geometry aliceVision_mesh
        Boost::program_options)

    # ============================================================
    # Phase 14.4a — photometric stereo binaries (2 of 3)
    # ============================================================
    # aliceVision_sphereDetection deferred to 14.4b (CoreML port to
    # replace the upstream ONNX Runtime dep). These two binaries are
    # pure Eigen/OpenCV math and ship now.

    add_executable(aliceVision_lightingCalibration
        upstream/src/software/pipeline/main_lightingCalibration.cpp)
    target_include_directories(aliceVision_lightingCalibration PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_lightingCalibration PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_lightingCalibration PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_lightingEstimation
        Boost::program_options)

    # Binary target name collides with the photometricStereo static lib
    # (same convention as upstream which gets around it via separate
    # subdir scopes). We disambiguate the CMake target with `_bin`
    # suffix; OUTPUT_NAME keeps the shipped binary filename what
    # Meshroom's PhotometricStereo descriptor expects.
    add_executable(aliceVision_photometricStereo_bin
        upstream/src/software/pipeline/main_photometricStereo.cpp)
    set_target_properties(aliceVision_photometricStereo_bin PROPERTIES
        OUTPUT_NAME aliceVision_photometricStereo)
    target_include_directories(aliceVision_photometricStereo_bin PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_photometricStereo_bin PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_photometricStereo_bin PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_photometricStereo
        Boost::program_options)

    # ============================================================
    # Phase 14.4b — aliceVision_sphereDetection (CoreML)
    # ============================================================
    # Replaces upstream's ONNX Runtime-based binary. Uses the user's
    # converted YOLOv8 .mlpackage (Vision-style export with NMS baked
    # in). Sources live in src/sphere_detection/aliceVision_shim/ so
    # the existing upstream-patched/ runtime-rewrite pattern isn't
    # needed — the changes were too large for a string-replace patch.

    add_executable(aliceVision_sphereDetection
        src/sphere_detection/aliceVision_shim/main_sphereDetection.cpp)
    # The shim's aliceVision/sphereDetection/sphereDetection.hpp must
    # SHADOW the upstream header (which #includes onnxruntime). Force
    # the shim include dir before upstream/src via `BEFORE PRIVATE`.
    target_include_directories(aliceVision_sphereDetection BEFORE PRIVATE
        ${CMAKE_SOURCE_DIR}/src/sphere_detection/aliceVision_shim)
    target_include_directories(aliceVision_sphereDetection PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_sphereDetection PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_sphereDetection PRIVATE
        av_sphere_detection            # provides aliceVision::sphereDetection::*
        aliceVision_system aliceVision_cmdline
        aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options)

    # Bonus: aliceVision_sfmTransfer (utils/) — needed by
    # multi-viewPhotometricStereo template (only remaining dep besides
    # sphereDetection-CoreML). Same lib set as sfmTransform.
    add_executable(aliceVision_sfmTransfer upstream/src/software/utils/main_sfmTransfer.cpp)
    target_include_directories(aliceVision_sfmTransfer PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_sfmTransfer PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_sfmTransfer PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_sfm aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options)

    # (18) meshDecimate — OpenMesh-based mesh simplification.
    add_executable(aliceVision_meshDecimate
        upstream/src/software/pipeline/main_meshDecimate.cpp
    )
    target_include_directories(aliceVision_meshDecimate PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
    )
    target_compile_definitions(aliceVision_meshDecimate PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_meshDecimate PRIVATE
        aliceVision_system
        aliceVision_cmdline
        aliceVision_mvsUtils
        OpenMeshCore
        OpenMeshTools
        Boost::program_options
    )

    # ================================================================
    # Phase 14.5 — bulk utility / pipeline / convert / export binaries
    # ================================================================
    # 21 additional binaries on top of the existing 34. exportAlembic +
    # exportAnimatedCamera were originally skipped here (need Alembic);
    # they are now enabled in Phase 14.6 further down once Alembic is
    # turned on globally.
    # All binaries below follow the established pattern:
    #   shim/aliceVision-includes -> upstream-patched -> upstream-generated -> upstream/src
    #   AV_EIGEN_DEFINITIONS so Eigen alignment matches the static libs.
    # Link deps are taken from each binary's upstream
    # software/{utils,pipeline,convert,export}/CMakeLists.txt LINKS line
    # plus its actual #include <aliceVision/...> set.

    # -- utils ----------------------------------------------------------

    # applyCalibration — propagate calibrated intrinsics + distortion
    # back into an SfMData. Pure sfmData/sfmDataIO/dataio/camera math.
    add_executable(aliceVision_applyCalibration
        upstream/src/software/utils/main_applyCalibration.cpp)
    target_include_directories(aliceVision_applyCalibration PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_applyCalibration PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_applyCalibration PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_dataio aliceVision_camera
        Boost::program_options)

    # colorCheckerCorrection — applies a Macbeth-chart CCM to images.
    # Needs OpenCV mcc contrib module (cv::mcc::CCM) at link time +
    # the patched conversionOpenCV.hpp on the include path (BEFORE so
    # it shadows the unpatched upstream copy in upstream/src/).
    add_executable(aliceVision_colorCheckerCorrection
        upstream/src/software/utils/main_colorCheckerCorrection.cpp)
    target_include_directories(aliceVision_colorCheckerCorrection BEFORE PRIVATE
        ${CMAKE_BINARY_DIR}/upstream-patched)
    target_include_directories(aliceVision_colorCheckerCorrection PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
        ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(aliceVision_colorCheckerCorrection PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_colorCheckerCorrection PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO
        ${OpenCV_LIBS}
        Boost::program_options)

    # colorCheckerDetection — detects Macbeth charts in input views
    # (cv::mcc::CCheckerDetector). Same patched-header treatment as
    # colorCheckerCorrection.
    add_executable(aliceVision_colorCheckerDetection
        upstream/src/software/utils/main_colorCheckerDetection.cpp)
    target_include_directories(aliceVision_colorCheckerDetection BEFORE PRIVATE
        ${CMAKE_BINARY_DIR}/upstream-patched)
    target_include_directories(aliceVision_colorCheckerDetection PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src
        ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(aliceVision_colorCheckerDetection PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_colorCheckerDetection PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO
        ${OpenCV_LIBS}
        Boost::program_options)

    # depthmapTracksInjecting — injects tracks back into depthmap pass.
    add_executable(aliceVision_depthmapTracksInjecting
        upstream/src/software/utils/main_depthmapTracksInjecting.cpp)
    target_include_directories(aliceVision_depthmapTracksInjecting PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_depthmapTracksInjecting PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_depthmapTracksInjecting PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_track
        Boost::program_options)

    # exportImages — re-export views (with optional undistortion).
    add_executable(aliceVision_exportImages
        upstream/src/software/utils/main_exportImages.cpp)
    target_include_directories(aliceVision_exportImages PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_exportImages PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_exportImages PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options)

    # imageProcessing — denoise / sharpen / lcp / resize utility. Binary
    # target name collides with the static lib of the same name — use
    # the same _bin suffix trick as photometricStereo.
    add_executable(aliceVision_imageProcessing_bin
        upstream/src/software/utils/main_imageProcessing.cpp)
    set_target_properties(aliceVision_imageProcessing_bin PROPERTIES
        OUTPUT_NAME aliceVision_imageProcessing)
    target_include_directories(aliceVision_imageProcessing_bin PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_imageProcessing_bin PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_imageProcessing_bin PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sensorDB aliceVision_feature aliceVision_sfm
        aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_lensCorrectionProfile
        aliceVision_imageProcessing
        ${OpenCV_LIBS}
        Boost::program_options)

    # intrinsicsTransforming — transform intrinsics + track keypoints.
    add_executable(aliceVision_intrinsicsTransforming
        upstream/src/software/utils/main_intrinsicsTransforming.cpp)
    target_include_directories(aliceVision_intrinsicsTransforming PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_intrinsicsTransforming PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_intrinsicsTransforming PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_track
        Boost::program_options)

    # keyframeSelection — picks best frames from a video sequence.
    # Depends on the new aliceVision_keyframe sublib (OpenCV-backed).
    add_executable(aliceVision_keyframeSelection
        upstream/src/software/utils/main_keyframeSelection.cpp)
    target_include_directories(aliceVision_keyframeSelection PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_keyframeSelection PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_keyframeSelection PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_keyframe
        Boost::program_options)

    # sfmColorizing — colorize SfM landmarks from input views.
    add_executable(aliceVision_sfmColorizing
        upstream/src/software/utils/main_sfmColorizing.cpp)
    target_include_directories(aliceVision_sfmColorizing PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_sfmColorizing PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_sfmColorizing PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_sfm aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options)

    # sfmMerge — merge two SfMData files (resolving id collisions).
    add_executable(aliceVision_sfmMerge
        upstream/src/software/utils/main_sfmMerge.cpp)
    target_include_directories(aliceVision_sfmMerge PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_sfmMerge PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_sfmMerge PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_sfm aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_matching
        Boost::program_options)

    # tracksMerging — main_*.cpp #includes TracksMerger.cpp inline, so
    # the binary TU defines the same symbols as the static lib. Clang's
    # archive-extraction skips redundant objects, so this links cleanly
    # without us needing to drop TracksMerger.cpp from aliceVision_track.
    add_executable(aliceVision_tracksMerging
        upstream/src/software/utils/main_tracksMerging.cpp)
    target_include_directories(aliceVision_tracksMerging PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_tracksMerging PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_tracksMerging PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_track
        Boost::program_options)

    # -- convert --------------------------------------------------------

    # convertDistortion — convert between distortion model
    # parameterizations (uses calibration::DistortionEstimationPoint).
    add_executable(aliceVision_convertDistortion
        upstream/src/software/convert/main_convertDistortion.cpp)
    target_include_directories(aliceVision_convertDistortion PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_convertDistortion PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_convertDistortion PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_feature
        aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_calibration
        Boost::program_options)

    # convertSfMFormat — JSON <-> internal SfMData converter.
    add_executable(aliceVision_convertSfMFormat
        upstream/src/software/convert/main_convertSfMFormat.cpp)
    target_include_directories(aliceVision_convertSfMFormat PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_convertSfMFormat PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_convertSfMFormat PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_feature
        aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options)

    # -- export ---------------------------------------------------------

    # exportDistortion — render the per-intrinsic distortion field.
    add_executable(aliceVision_exportDistortion
        upstream/src/software/export/main_exportDistortion.cpp)
    target_include_directories(aliceVision_exportDistortion PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_exportDistortion PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_exportDistortion PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_camera
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_image
        Boost::program_options)

    # -- pipeline -------------------------------------------------------

    # checkerboardDetection — find chessboard corners in calibration
    # views. Needs the new aliceVision_calibration sublib.
    add_executable(aliceVision_checkerboardDetection
        upstream/src/software/pipeline/main_checkerboardDetection.cpp)
    target_include_directories(aliceVision_checkerboardDetection PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_checkerboardDetection PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_checkerboardDetection PRIVATE
        aliceVision_dataio aliceVision_image aliceVision_calibration
        aliceVision_system aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_cmdline
        Boost::program_options)

    # distortionCalibration — Ceres-based distortion refinement from
    # checkerboard detections.
    add_executable(aliceVision_distortionCalibration
        upstream/src/software/pipeline/main_distortionCalibration.cpp)
    target_include_directories(aliceVision_distortionCalibration PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_distortionCalibration PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_distortionCalibration PRIVATE
        aliceVision_dataio aliceVision_image aliceVision_calibration
        aliceVision_system aliceVision_cmdline
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_camera
        Boost::program_options)

    # geometricFilterEstimating — fundamental-matrix-based pairwise
    # match filtering (essentially a standalone version of the geometric
    # filter step inside featureMatching).
    add_executable(aliceVision_geometricFilterEstimating
        upstream/src/software/pipeline/main_geometricFilterEstimating.cpp)
    target_include_directories(aliceVision_geometricFilterEstimating PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_geometricFilterEstimating PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_geometricFilterEstimating PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_feature
        aliceVision_multiview aliceVision_matchingImageCollection
        aliceVision_sfm aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_matching aliceVision_robustEstimation
        Boost::program_options)

    # nodalSfM — pure-rotation SfM (for nodal-pan panoramas).
    add_executable(aliceVision_nodalSfM
        upstream/src/software/pipeline/main_nodalSfM.cpp)
    target_include_directories(aliceVision_nodalSfM PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_nodalSfM PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_nodalSfM PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_feature aliceVision_sfm
        aliceVision_sfmData aliceVision_sfmDataIO aliceVision_track
        aliceVision_geometry aliceVision_multiview aliceVision_robustEstimation
        Boost::program_options)

    # lidarMeshing — Delaunay tetrahedralization on LiDAR scans, then
    # graph-cut surface extraction. Uses fuseCut::InputSet + Mesher.
    add_executable(aliceVision_lidarMeshing
        upstream/src/software/pipeline/main_lidarMeshing.cpp)
    target_include_directories(aliceVision_lidarMeshing PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_lidarMeshing PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_lidarMeshing PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_mvsData aliceVision_mvsUtils
        aliceVision_mesh aliceVision_fuseCut
        aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options)

    # lidarMerging — merge per-scan meshes by Delaunay tetrahedra
    # intersection.
    add_executable(aliceVision_lidarMerging
        upstream/src/software/pipeline/main_lidarMerging.cpp)
    target_include_directories(aliceVision_lidarMerging PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_lidarMerging PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_lidarMerging PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_mvsData aliceVision_mvsUtils
        aliceVision_mesh aliceVision_fuseCut
        aliceVision_geometry aliceVision_stl
        Boost::program_options)

    # lidarDecimating — OpenMesh-based simplification for the merged
    # LiDAR mesh.
    add_executable(aliceVision_lidarDecimating
        upstream/src/software/pipeline/main_lidarDecimating.cpp)
    target_include_directories(aliceVision_lidarDecimating PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_lidarDecimating PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_lidarDecimating PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_mvsData aliceVision_mvsUtils
        aliceVision_mesh aliceVision_fuseCut
        OpenMeshCore OpenMeshTools
        Boost::program_options)

    # ================================================================
    # Phase 14.6 — Alembic export binaries
    # ================================================================
    # These two binaries unlock 8 Meshroom templates that need Alembic
    # camera / point cloud export (exportAlembic blocks 6 templates,
    # exportAnimatedCamera blocks 3 — with one overlap). Both call
    # sfmDataIO::AlembicExporter, whose .cpp is gated by
    # ALICEVISION_HAVE_ALEMBIC in upstream/src/aliceVision/sfmDataIO/
    # CMakeLists.txt. Setting the flag earlier in this file (search
    # "Phase 14.6 — Alembic") folds the Alembic TUs into the existing
    # aliceVision_sfmDataIO static lib and links Alembic::Alembic
    # PRIVATE to it, so the binaries below only need to link
    # aliceVision_sfmDataIO + the usual sfmData / system / cmdline set.
    # Link list matches upstream's software/export/CMakeLists.txt.

    add_executable(aliceVision_exportAlembic
        upstream/src/software/export/main_exportAlembic.cpp)
    target_include_directories(aliceVision_exportAlembic PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_exportAlembic PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_exportAlembic PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options Boost::timer)

    add_executable(aliceVision_exportAnimatedCamera
        upstream/src/software/export/main_exportAnimatedCamera.cpp)
    target_include_directories(aliceVision_exportAnimatedCamera PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_exportAnimatedCamera PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_exportAnimatedCamera PRIVATE
        aliceVision_system aliceVision_cmdline aliceVision_image
        aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_camera
        Boost::program_options Boost::timer)

    # ================================================================
    # Phase 14.7 — Mac-port native binaries for 2026.1.0 features that
    # upstream defers to external Python pipelines. We ship our own
    # native C++ implementations / honest stubs to round out coverage
    # for the cameraTrackingRoma and cameraTrackingDepth templates.
    # ================================================================
    # Source layout: src/native_binaries/ (NOT under upstream/, because
    # these are OUR code with no upstream provenance). The descriptors
    # live at meshroom-mac/nodes/aliceVision/{StarListing,
    # MatchMasking, MoGe}.py and were authored as loadable shims
    # ahead of these binaries landing — Phase 14.7 closes the loop.
    #
    # All three binaries reuse only existing static libs from this
    # build tree (no new sublibs / no new third-party deps):
    #   * starListing    : sfmData + sfmDataIO + cmdline + system
    #   * matchMasking   : sfmData + sfmDataIO + cmdline + system
    #   * moGe           : + image (EXR placeholder writing)
    add_executable(aliceVision_starListing
        src/native_binaries/main_starListing.cpp)
    target_include_directories(aliceVision_starListing PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_starListing PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_starListing PRIVATE
        aliceVision_system aliceVision_cmdline
        aliceVision_sfmData aliceVision_sfmDataIO
        Boost::program_options)

    # Phase 14.9 — REAL TinyRoMa-backed aliceVision_matchMasking
    # (replaces the Phase 14.7 honest pass-through). Source moved from
    # src/native_binaries/ to src/roma/aliceVision_shim/ to mirror the
    # sphere_detection / moge layouts. Links av_roma for the
    # CoreMLRomaMatcher wrapper.
    add_executable(aliceVision_matchMasking
        src/roma/aliceVision_shim/main_matchMasking.cpp)
    target_include_directories(aliceVision_matchMasking PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_matchMasking PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_matchMasking PRIVATE
        av_roma                       # provides av::roma::CoreMLRomaMatcher
        aliceVision_system aliceVision_cmdline
        aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_image
        Boost::program_options)

    # Phase 14.8 — REAL CoreML-backed aliceVision_moGe (replaces the
    # Phase 14.7 honest stub). Source moved from src/native_binaries/
    # to src/moge/aliceVision_shim/ to mirror the sphere_detection
    # layout. Links av_moge for the CoreMLMoGeRunner wrapper.
    add_executable(aliceVision_moGe
        src/moge/aliceVision_shim/main_moGe.cpp)
    target_include_directories(aliceVision_moGe PRIVATE
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        ${CMAKE_BINARY_DIR}/upstream-patched
        ${CMAKE_BINARY_DIR}/upstream-generated
        ${CMAKE_SOURCE_DIR}/upstream/src)
    target_compile_definitions(aliceVision_moGe PRIVATE ${AV_EIGEN_DEFINITIONS})
    target_link_libraries(aliceVision_moGe PRIVATE
        av_moge                       # provides av::moge::CoreMLMoGeRunner
        aliceVision_system aliceVision_cmdline
        aliceVision_sfmData aliceVision_sfmDataIO
        aliceVision_image
        Boost::program_options)

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
