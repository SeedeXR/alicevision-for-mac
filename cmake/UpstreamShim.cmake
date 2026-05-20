# UpstreamShim.cmake — minimal shim so upstream/src/aliceVision/*/CMakeLists.txt
# files can be `add_subdirectory()`-ed into our build without pulling in
# upstream's full CMake machinery.
#
# The per-module CMakeLists call:
#   alicevision_add_library(<name> SOURCES ... PUBLIC_LINKS ... PRIVATE_LINKS ...
#                                  PUBLIC_INCLUDE_DIRS ... PRIVATE_INCLUDE_DIRS ...
#                                  PUBLIC_DEFINITIONS ... PRIVATE_DEFINITIONS ...)
#   alicevision_add_test(<file> NAME <n> LINKS ...)
#   alicevision_swig_add_library(...)
#
# Our shim:
#   * Implements `alicevision_add_library` as a plain `add_library` + the
#     PUBLIC_LINKS / PRIVATE_LINKS / includes / definitions wiring. No install
#     rules, no SOVERSION dance, no Windows .rc generation, no SWIG.
#   * `alicevision_add_test` is a no-op — upstream's tests use Boost.Test
#     which we don't link. Our own validation lives under tests/ and exercises
#     just the Metal kernels.
#   * `alicevision_swig_add_library` is a no-op.
#
# Required variables set by the caller before add_subdirectory():
#   ALICEVISION_INCLUDE_DIR       = upstream/src                (upstream's public include root)
#   generatedDir                  = a writable build dir        (some modules add it as include)
#   ALICEVISION_OPENMP_CXX_TARGETS = "" or OpenMP::OpenMP_CXX   (target wired by find_package(OpenMP))
#   ALICEVISION_NVTX_LIBRARY      = ""                          (CUDA-only; empty on Apple)
#   AV_EIGEN_DEFINITIONS          = "" or whatever flags upstream usually sets
#
# Upstream targets (Eigen3::Eigen, Boost::log, etc.) must already be findable
# at the point of add_subdirectory.

function(alicevision_add_library library_name)
    set(options USE_CUDA)
    set(multipleValues
        SOURCES
        PUBLIC_LINKS PRIVATE_LINKS
        PUBLIC_INCLUDE_DIRS PRIVATE_INCLUDE_DIRS
        PUBLIC_DEFINITIONS PRIVATE_DEFINITIONS
        RESOURCES
    )
    cmake_parse_arguments(LIB "${options}" "" "${multipleValues}" ${ARGN})

    if(NOT library_name)
        message(FATAL_ERROR "alicevision_add_library: name required")
    endif()
    if(NOT LIB_SOURCES)
        message(FATAL_ERROR "alicevision_add_library(${library_name}): SOURCES required")
    endif()
    if(LIB_USE_CUDA)
        # CUDA isn't available on Apple — the upstream gate AV_USE_CUDA
        # should prevent this branch from ever firing on us, but be loud
        # if a caller forgets.
        message(FATAL_ERROR "alicevision_add_library(${library_name}): USE_CUDA is not supported (Apple Silicon).")
    endif()

    add_library(${library_name} STATIC ${LIB_SOURCES})

    target_link_libraries(${library_name}
        PUBLIC  ${LIB_PUBLIC_LINKS}
        PRIVATE ${LIB_PRIVATE_LINKS}
    )

    target_include_directories(${library_name}
        PUBLIC  # S39: shim path FIRST so headers like
                # `aliceVision/linearProgramming/OSIXSolver.hpp` resolve
                # to our overrides (drops Coin-OR Clp dep that we don't
                # have on Apple) rather than upstream's CUDA/Clp version.
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes>
                $<BUILD_INTERFACE:${ALICEVISION_INCLUDE_DIR}>
                $<BUILD_INTERFACE:${generatedDir}>
                ${LIB_PUBLIC_INCLUDE_DIRS}
        PRIVATE ${LIB_PRIVATE_INCLUDE_DIRS}
    )

    target_compile_definitions(${library_name}
        PUBLIC  ${LIB_PUBLIC_DEFINITIONS}
        PRIVATE ${LIB_PRIVATE_DEFINITIONS}
    )

    set_target_properties(${library_name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    # The upstream modules tend to declare bigger interface headers than
    # they need; let them through metal-cpp-style warnings to keep the
    # noise down on first build. Refine per-target later.
    target_compile_options(${library_name} PRIVATE
        -Wno-deprecated-declarations
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-shadow
        -Wno-sign-compare
    )
endfunction()

function(alicevision_add_test test_file)
    # No-op: upstream's tests are Boost.Test-based and out of scope for
    # the depthMap-only build. Our own validation is under
    # alicevision-for-mac/tests/.
endfunction()

function(alicevision_add_interface interface_name)
    set(multipleValues SOURCES LINKS)
    cmake_parse_arguments(IFACE "" "" "${multipleValues}" ${ARGN})
    if(NOT interface_name)
        message(FATAL_ERROR "alicevision_add_interface: name required")
    endif()
    add_library(${interface_name} INTERFACE)
    if(IFACE_LINKS)
        target_link_libraries(${interface_name} INTERFACE ${IFACE_LINKS})
    endif()
endfunction()

function(alicevision_swig_add_library)
    # No-op.
endfunction()

# alicevision_add_software(<name> SOURCE <main.cpp> FOLDER <id> LINKS ...)
#
# Upstream's macro reads `ALICEVISION_SOFTWARE_VERSION_{MAJOR,MINOR}` from
# the source and (on Windows) generates a .rc file. We just compile the
# executable and link. Install rules deferred to Phase 12.
function(alicevision_add_software software_name)
    set(options "")
    set(singleValues FOLDER)
    set(multipleValues SOURCE LINKS INCLUDE_DIRS)
    cmake_parse_arguments(SOFTWARE "${options}" "${singleValues}"
                          "${multipleValues}" ${ARGN})
    if(NOT software_name)
        message(FATAL_ERROR "alicevision_add_software: name required")
    endif()
    if(NOT SOFTWARE_SOURCE)
        message(FATAL_ERROR "alicevision_add_software(${software_name}): SOURCE required")
    endif()
    add_executable(${software_name} ${SOFTWARE_SOURCE})
    if(SOFTWARE_LINKS)
        target_link_libraries(${software_name} PRIVATE ${SOFTWARE_LINKS})
    endif()
    if(SOFTWARE_INCLUDE_DIRS)
        target_include_directories(${software_name} PRIVATE ${SOFTWARE_INCLUDE_DIRS})
    endif()
    target_compile_definitions(${software_name} PRIVATE ${AV_EIGEN_DEFINITIONS})
    # Inherit warning policy from `av::warnings` if present.
    if(TARGET av::warnings)
        target_link_libraries(${software_name} PRIVATE av::warnings)
    endif()
endfunction()
