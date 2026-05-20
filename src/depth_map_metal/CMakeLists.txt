# depth_map_metal — Metal port of upstream depthMap CUDA kernels.
#
# This is where the per-kernel ports live. As the surface grows we
# may split into per-kernel sub-targets, but for now one library is
# fine.

add_library(av_depth_map_metal STATIC
    include/av/depth_map/Eig33.hpp
    include/av/depth_map/MatrixOps.hpp
    include/av/depth_map/PatchOps.hpp
    include/av/depth_map/ColorOps.hpp
    include/av/depth_map/SimStatOps.hpp
    include/av/depth_map/CompNCC.hpp
    include/av/depth_map/DevicePatchPattern.hpp
    include/av/depth_map/ImageColorConversion.hpp
    include/av/depth_map/GaussianTable.hpp
    include/av/depth_map/GaussianFilter.hpp
    include/av/depth_map/MipmappedArray.hpp
    include/av/depth_map/Volume.hpp
    include/av/depth_map/DepthSimMap.hpp
    include/av/depth_map/DeviceMipmapImage.hpp
    include/av/depth_map/LRUCache.hpp
    include/av/depth_map/DeviceCache.hpp
    include/av/depth_map/DeviceStreamManager.hpp
    include/av/depth_map/upstream_adapter.hpp
    include/av/depth_map/adapter_profile.hpp
    src/Eig33.cpp
    src/MatrixOps.cpp
    src/PatchOps.cpp
    src/ColorOps.cpp
    src/SimStatOps.cpp
    src/CompNCC.cpp
    src/ImageColorConversion.cpp
    src/GaussianTable.cpp
    src/GaussianFilter.cpp
    src/MipmappedArray.cpp
    src/Volume.cpp
    src/DepthSimMap.cpp
    src/DeviceMipmapImage.cpp
    src/DeviceCache.cpp
    src/DeviceStreamManager.cpp
    src/upstream_adapter.cpp
    src/adapter_profile.cpp
)

# S43 Phase 14: optional per-forwarder timing. When OFF (default) the
# preprocessor strips all `AV_ADAPTER_PROFILE_SCOPE` invocations down to
# `do {} while(0)` and `adapter_profile.cpp` becomes an empty TU — zero
# runtime cost, identical behaviour, ctest 37/37 unchanged.
if(AV_PROFILE_ADAPTER)
    target_compile_definitions(av_depth_map_metal PUBLIC AV_PROFILE_ADAPTER=1)
    message(STATUS "AV_PROFILE_ADAPTER=ON — cuda_* forwarder timing enabled.")
endif()

target_include_directories(av_depth_map_metal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(av_depth_map_metal
    PUBLIC  av::gpu
    PRIVATE av::warnings
)

set_target_properties(av_depth_map_metal PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(av::depth_map_metal ALIAS av_depth_map_metal)

# ============================================================
# cuda_host_shim — bridge layer between upstream's host depthMap
# `.cpp` files (Sgm, Refine, DepthMapEstimator, …) and our native
# `av::depth_map::*` types.
# ============================================================
#
# Lives behind `AV_BUILD_UPSTREAM_DEPTHMAP` because it consumes
# `aliceVision_image`, `aliceVision_mvsData`, `aliceVision_mvsUtils`
# (only built when the upstream tree is configured).
#
# The four shim header pairs implemented here are:
#   * DeviceCache       — singleton mirror of upstream's
#   * DeviceStreamManager — fixed-size queue pool
#   * patchPattern      — custom-patch builder
#   * utils             — listCudaDevices / getDeviceMemoryInfo
#
# Each `.cpp` lives in `src/cuda_host_shim/`; each `.hpp` lives in
# `cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/`
# so an `#include <aliceVision/depthMap/cuda/host/X.hpp>` from
# upstream's `Sgm.cpp` etc. resolves to the shim (shim include
# path is `BEFORE` upstream's `src/`).
if(AV_BUILD_UPSTREAM_DEPTHMAP)
    add_library(av_cuda_host_shim STATIC
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/DeviceCache.hpp
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/DeviceStreamManager.hpp
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/patchPattern.hpp
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/utils.hpp
        src/cuda_host_shim/DeviceCache.cpp
        src/cuda_host_shim/DeviceStreamManager.cpp
        src/cuda_host_shim/patchPattern.cpp
        src/cuda_host_shim/utils.cpp
    )

    target_include_directories(av_cuda_host_shim PUBLIC
        # Resolve `#include <aliceVision/depthMap/cuda/host/X.hpp>`
        # to our shim layer BEFORE upstream's `src/`.
        ${CMAKE_SOURCE_DIR}/cmake/shims/aliceVision-includes
        # Upstream headers for `mvsUtils::MultiViewParams`,
        # `image::Image`, `mvsData::Matrix3x3`, …
        ${CMAKE_SOURCE_DIR}/upstream/src
        ${CMAKE_BINARY_DIR}/upstream-generated
    )

    target_link_libraries(av_cuda_host_shim
        PUBLIC
            av::depth_map_metal
            aliceVision_image
            aliceVision_mvsData
            aliceVision_mvsUtils
        PRIVATE
            av::warnings
    )

    set_target_properties(av_cuda_host_shim PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    add_library(av::cuda_host_shim ALIAS av_cuda_host_shim)
endif()
