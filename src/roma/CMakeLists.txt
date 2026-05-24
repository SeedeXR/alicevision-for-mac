# av_roma — CoreML wrapper around the user's TinyRoMa .mlpackage
# (ai-models/tiny_roma_v1_480x640.mlpackage).
#
# Pure-C++ public header + Objective-C++ implementation. Same shape as
# src/sphere_detection/ and src/moge/. find_package(OpenCV) is done by
# the top-level CMakeLists.txt before this add_subdirectory is reached.

add_library(av_roma STATIC
    include/av/roma/CoreMLRomaMatcher.hpp
    src/CoreMLRomaMatcher.mm
)

target_include_directories(av_roma PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(av_roma PUBLIC cxx_std_20)

set_source_files_properties(src/CoreMLRomaMatcher.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

find_library(COREML_FRAMEWORK CoreML REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

target_link_libraries(av_roma
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        ${COREML_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
)
