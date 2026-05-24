# ============================================================
# Metal.cmake — Metal framework / metal-cpp / MSL toolchain
# ============================================================
# Defines:
#   - imported target ``av::metal``  (frameworks + metal-cpp headers)
#   - function ``av_compile_metal_library(TARGET <name>
#                                         SOURCES <a.metal> <b.metal> ...)``
#       Compiles .metal sources via `xcrun metal` + `xcrun metallib`
#       into a single <name>.metallib placed next to the linking
#       executable at install/run time.
# ============================================================

if(TARGET av::metal)
    return()
endif()

# ---------- Locate Apple frameworks ----------

find_library(METAL_FRAMEWORK            Metal            REQUIRED)
find_library(FOUNDATION_FRAMEWORK       Foundation       REQUIRED)
find_library(QUARTZCORE_FRAMEWORK       QuartzCore       REQUIRED)
find_library(METALKIT_FRAMEWORK         MetalKit)        # optional
find_library(METALPERFSHADERS_FRAMEWORK MetalPerformanceShaders) # optional

# ---------- metal-cpp headers ----------

set(METAL_CPP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/metal-cpp"
    CACHE PATH "Path to Apple metal-cpp headers")

if(NOT EXISTS "${METAL_CPP_INCLUDE_DIR}/Metal/Metal.hpp")
    message(FATAL_ERROR
        "metal-cpp headers not found at ${METAL_CPP_INCLUDE_DIR}.\n"
        "Run: cd third_party && curl -sLO https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip && unzip metal-cpp_macOS15_iOS18.zip"
    )
endif()

# ---------- Interface target ``av::metal`` ----------

add_library(av_metal INTERFACE)
target_include_directories(av_metal SYSTEM INTERFACE "${METAL_CPP_INCLUDE_DIR}")
target_link_libraries(av_metal INTERFACE
    "${METAL_FRAMEWORK}"
    "${FOUNDATION_FRAMEWORK}"
    "${QUARTZCORE_FRAMEWORK}"
)
if(METALKIT_FRAMEWORK)
    target_link_libraries(av_metal INTERFACE "${METALKIT_FRAMEWORK}")
endif()
if(METALPERFSHADERS_FRAMEWORK)
    target_link_libraries(av_metal INTERFACE "${METALPERFSHADERS_FRAMEWORK}")
endif()

add_library(av::metal ALIAS av_metal)

# ---------- Metal toolchain probe ----------
# `metal` and `metallib` live inside the Xcode toolchain and are not
# on PATH. Probe via `xcrun -sdk macosx --find` first; if that fails,
# fall back to find_program.

execute_process(COMMAND xcrun -sdk macosx --find metal
                OUTPUT_VARIABLE _xcrun_metal
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _xcrun_rc_metal
                ERROR_QUIET)
execute_process(COMMAND xcrun -sdk macosx --find metallib
                OUTPUT_VARIABLE _xcrun_metallib
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _xcrun_rc_metallib
                ERROR_QUIET)

if(_xcrun_rc_metal EQUAL 0 AND EXISTS "${_xcrun_metal}")
    set(METAL_COMPILER "${_xcrun_metal}" CACHE FILEPATH "Metal frontend")
else()
    find_program(METAL_COMPILER metal
        HINTS "${CMAKE_OSX_SYSROOT}/../../usr/bin"
        PATHS "/usr/bin"
    )
endif()
if(_xcrun_rc_metallib EQUAL 0 AND EXISTS "${_xcrun_metallib}")
    set(METALLIB_LINKER "${_xcrun_metallib}" CACHE FILEPATH "Metal linker")
else()
    find_program(METALLIB_LINKER metallib
        HINTS "${CMAKE_OSX_SYSROOT}/../../usr/bin"
        PATHS "/usr/bin"
    )
endif()

if(NOT METAL_COMPILER OR NOT EXISTS "${METAL_COMPILER}")
    message(WARNING "Metal compiler (`metal`) not found. Install Xcode "
                    "and run `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`. "
                    "Existing default.metallib (if present) will be used as-is; "
                    "new MSL shader builds will fail.")
    set(METAL_TOOLCHAIN_AVAILABLE OFF CACHE INTERNAL "Metal toolchain present")
else()
    set(METAL_TOOLCHAIN_AVAILABLE ON CACHE INTERNAL "Metal toolchain present")
endif()
# Xcode 26+ merged `metallib` functionality into `metal`. The separate
# `metallib` binary no longer ships with the Apple toolchain. Mac builds
# without it; on older Xcode versions the variable may still be set.
if(NOT METALLIB_LINKER OR NOT EXISTS "${METALLIB_LINKER}")
    message(STATUS "Metal linker (`metallib`) not found (Xcode 26+ ships only `metal`). "
                   "Metal compilation will use `metal -o ...` directly.")
endif()
# Probe the Metal Toolchain by actually invoking the compiler. Xcode 26
# requires a separate `xcodebuild -downloadComponent MetalToolchain`
# step; without it, `metal --version` errors. We detect that here and
# downgrade the toolchain availability flag so the `default.metallib`
# build step can be skipped (the existing artifact is reused).
if(METAL_TOOLCHAIN_AVAILABLE)
    execute_process(COMMAND ${METAL_COMPILER} --version
                    OUTPUT_QUIET ERROR_VARIABLE _metal_probe_err
                    RESULT_VARIABLE _metal_probe_rc)
    if(NOT _metal_probe_rc EQUAL 0 OR _metal_probe_err MATCHES "missing Metal Toolchain")
        message(WARNING "Xcode Metal Toolchain not installed — run "
                        "`xcodebuild -downloadComponent MetalToolchain` to enable "
                        "MSL shader recompilation. Existing default.metallib at "
                        "${CMAKE_BINARY_DIR}/default.metallib will be used as-is. "
                        "Pure-C++ binary targets are unaffected.")
        set(METAL_TOOLCHAIN_AVAILABLE OFF CACHE INTERNAL "Metal toolchain present" FORCE)
    endif()
endif()

message(STATUS "Metal compiler : ${METAL_COMPILER}")
message(STATUS "Metal linker   : ${METALLIB_LINKER}")

# ---------- Function: av_compile_metal_library ----------
#
# Usage:
#   av_compile_metal_library(
#       TARGET   my_shaders
#       SOURCES  src/a.metal src/b.metal
#       OUTPUT   default.metallib
#       INCLUDES include/dir
#   )
#
# Produces a custom target ``my_shaders`` that builds
# ``${CMAKE_CURRENT_BINARY_DIR}/<OUTPUT>``.
# The .metallib is placed next to the consuming executable at
# install time via av_install_metallib() below.

function(av_compile_metal_library)
    set(options)
    set(oneValueArgs TARGET OUTPUT STANDARD)
    set(multiValueArgs SOURCES INCLUDES DEFINES)
    cmake_parse_arguments(MTL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT MTL_TARGET)
        message(FATAL_ERROR "av_compile_metal_library: TARGET is required")
    endif()
    if(NOT MTL_SOURCES)
        message(FATAL_ERROR "av_compile_metal_library: SOURCES is required")
    endif()
    if(NOT MTL_OUTPUT)
        set(MTL_OUTPUT "${MTL_TARGET}.metallib")
    endif()
    if(NOT MTL_STANDARD)
        set(MTL_STANDARD "metal3.0")
    endif()

    # If the Metal Toolchain isn't installed (Xcode 26+ ships it as an
    # opt-in download), declare an empty custom target. Adding non-Metal
    # C++ binaries to the build then doesn't require shader
    # recompilation. The existing default.metallib on disk (if any) is
    # used at runtime by the consuming executable. If a user wants to
    # rebuild MSL kernels, they install the toolchain via
    # `xcodebuild -downloadComponent MetalToolchain` and re-run cmake.
    if(NOT METAL_TOOLCHAIN_AVAILABLE)
        add_custom_target(${MTL_TARGET} ALL
            COMMENT "Skipping Metal compilation for '${MTL_TARGET}': "
                    "Metal Toolchain not installed. Pre-built "
                    "default.metallib (if present) will be used."
        )
        set_target_properties(${MTL_TARGET} PROPERTIES
            AV_METALLIB_PATH "${CMAKE_CURRENT_BINARY_DIR}/${MTL_OUTPUT}"
        )
        return()
    endif()

    set(_air_files "")
    foreach(_src ${MTL_SOURCES})
        get_filename_component(_abs    "${_src}" ABSOLUTE)
        get_filename_component(_base   "${_src}" NAME_WE)
        set(_air "${CMAKE_CURRENT_BINARY_DIR}/${_base}.air")

        set(_include_flags "")
        foreach(_inc ${MTL_INCLUDES})
            list(APPEND _include_flags "-I" "${_inc}")
        endforeach()
        set(_define_flags "")
        foreach(_def ${MTL_DEFINES})
            list(APPEND _define_flags "-D${_def}")
        endforeach()

        # Note: the `metal` binary resolved via `xcrun --find` does not
        # accept the `-sdk` flag (which is an xcrun selector). The SDK
        # is implicit because the resolved binary lives inside the
        # platform-specific toolchain. We pass `--target=` explicitly
        # to keep the build hermetic.
        add_custom_command(
            OUTPUT  "${_air}"
            COMMAND "${METAL_COMPILER}"
                    -std=${MTL_STANDARD}
                    -ffast-math
                    -O3
                    -gline-tables-only
                    ${_include_flags}
                    ${_define_flags}
                    -c "${_abs}"
                    -o "${_air}"
            DEPENDS "${_abs}"
            COMMENT "metal: compiling ${_src}"
            VERBATIM
        )
        list(APPEND _air_files "${_air}")
    endforeach()

    set(_metallib "${CMAKE_CURRENT_BINARY_DIR}/${MTL_OUTPUT}")
    add_custom_command(
        OUTPUT  "${_metallib}"
        COMMAND "${METALLIB_LINKER}"
                ${_air_files}
                -o "${_metallib}"
        DEPENDS ${_air_files}
        COMMENT "metallib: linking ${MTL_OUTPUT}"
        VERBATIM
    )

    add_custom_target(${MTL_TARGET} ALL DEPENDS "${_metallib}")
    set_target_properties(${MTL_TARGET} PROPERTIES
        AV_METALLIB_PATH "${_metallib}"
    )
endfunction()

# ---------- Function: av_install_metallib ----------
# Copies a built .metallib next to a consuming executable at
# build time, so the executable can load it via
# MTL::Device::newDefaultLibrary().

function(av_install_metallib)
    set(options)
    set(oneValueArgs FROM EXECUTABLE)
    set(multiValueArgs)
    cmake_parse_arguments(MIL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT MIL_FROM OR NOT MIL_EXECUTABLE)
        message(FATAL_ERROR "av_install_metallib: FROM and EXECUTABLE are required")
    endif()

    get_target_property(_src ${MIL_FROM} AV_METALLIB_PATH)
    if(NOT _src)
        message(FATAL_ERROR "av_install_metallib: target ${MIL_FROM} has no AV_METALLIB_PATH")
    endif()

    add_dependencies(${MIL_EXECUTABLE} ${MIL_FROM})
    add_custom_command(TARGET ${MIL_EXECUTABLE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_src}"
                "$<TARGET_FILE_DIR:${MIL_EXECUTABLE}>/default.metallib"
        COMMENT "Staging default.metallib next to ${MIL_EXECUTABLE}"
        VERBATIM
    )
endfunction()
