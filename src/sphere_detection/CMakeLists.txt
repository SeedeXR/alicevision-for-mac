# av_sphere_detection — CoreML drop-in replacement for the upstream
# aliceVision_sphereDetection ONNX Runtime sphere detector.
#
# Two layers:
#   1. av::sphere::CoreMLSphereDetector  — pure-C++ public API +
#      Objective-C++ implementation. Wraps a Vision-style YOLOv8
#      .mlpackage. Runs on MLComputeUnits.all (ANE / GPU / CPU).
#   2. aliceVision::sphereDetection::*   — drop-in for upstream's
#      namespace. Same public API as the upstream header EXCEPT the
#      Ort::Session parameter is swapped for CoreMLSphereDetector.
#
# Together these provide everything `aliceVision_sphereDetection`
# (main_sphereDetection.cpp, also in this directory) needs.
#
# `find_package(OpenCV)` is done by the top-level CMakeLists.txt.

add_library(av_sphere_detection STATIC
    include/av/sphere/CoreMLSphereDetector.hpp
    src/CoreMLSphereDetector.mm
    aliceVision_shim/aliceVision/sphereDetection/sphereDetection.hpp
    aliceVision_shim/sphereDetection.cpp
)

target_include_directories(av_sphere_detection PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/aliceVision_shim>
    # AliceVision headers transitively included by the shim's .cpp/.hpp
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/upstream-generated>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/upstream/src>
)

target_compile_features(av_sphere_detection PUBLIC cxx_std_20)
target_compile_definitions(av_sphere_detection PUBLIC ${AV_EIGEN_DEFINITIONS})

set_source_files_properties(src/CoreMLSphereDetector.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

find_library(COREML_FRAMEWORK CoreML REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)

target_link_libraries(av_sphere_detection
    PUBLIC
        ${OpenCV_LIBS}
        aliceVision_system
        aliceVision_image
        aliceVision_sfmData
        aliceVision_sfmDataIO
    PRIVATE
        ${COREML_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREVIDEO_FRAMEWORK}
)
