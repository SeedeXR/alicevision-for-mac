# av_moge — CoreML wrapper around the user's MoGe-2 .mlpackage
# (DINOv2 ViT-B/14, exported at 504×672 fixed input resolution).
#
# Pure-C++ public header + Objective-C++ implementation. Same shape as
# src/sphere_detection/. find_package(OpenCV) is done by the top-level
# CMakeLists.txt before this add_subdirectory is reached.

add_library(av_moge STATIC
    include/av/moge/CoreMLMoGeRunner.hpp
    src/CoreMLMoGeRunner.mm
)

target_include_directories(av_moge PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(av_moge PUBLIC cxx_std_20)

set_source_files_properties(src/CoreMLMoGeRunner.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

find_library(COREML_FRAMEWORK CoreML REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

target_link_libraries(av_moge
    PUBLIC
        ${OpenCV_LIBS}
    PRIVATE
        ${COREML_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
)
