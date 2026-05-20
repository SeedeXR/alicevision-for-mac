#pragma once

// upstream_adapter.hpp — Phase 8 adapter shim. Declares the 12
// `cuda_*` symbols that upstream's host orchestration code
// (`Sgm.cpp`, `Refine.cpp`, `DepthMapEstimator.cpp`,
// `NormalMapEstimator.cpp`) expects at link time. Each function
// forwards to its `av::depth_map::*` equivalent.
//
// Status (S33): contract captured here; impl skeleton in
// `upstream_adapter.cpp` but most forwarders throw until the
// `CudaDeviceMemoryPitched` type-shim lands (S34 prerequisite).
// See `memory/phase8_adapter_map.md` for the full per-symbol
// translation rules.
//
// The intent is that upstream's `depthMap/cuda/host/*.hpp`
// declarations resolve at link time to OUR implementations
// here, so upstream's `Sgm.cpp` etc. can be compiled
// unmodified once the type-shim layer for
// `CudaDeviceMemoryPitched<T, N>`, `CudaSize<N>`, etc. is in
// place.

// NOTE: this header deliberately uses upstream's types
// (CudaDeviceMemoryPitched, DeviceMipmapImage, SgmParams, etc.).
// It's NOT meant to be consumed by our own Metal-native code —
// our code uses `av::gpu::Buffer` + `av::depth_map::*` directly.
// Including this header requires upstream's headers to be on
// the include path, i.e., requires `AV_BUILD_UPSTREAM_DEPTHMAP=ON`.

// Forward declarations of upstream types to keep this header
// self-contained when only the adapter API is needed.
namespace aliceVision {
// Range + ROI live in upstream's `aliceVision::` namespace directly
// (defined in `mvsData/ROI.hpp`), NOT under `aliceVision::depthMap`
// or `aliceVision::mvsData`. Fixed S35 after S34 link audit.
struct Range;
struct ROI;
namespace depthMap {
    template<class T, unsigned Dim> class CudaDeviceMemoryPitched;
    class DeviceMipmapImage;
    struct SgmParams;
    struct RefineParams;
}
}

// Match upstream's pixel-type aliases used in the function
// signatures below (declared in upstream's `BufPtr.hpp` and
// `cuda/device/buffer.cuh` — see upstream's
// `DepthMapTypes.hpp` for the full list).
namespace aliceVision {
namespace depthMap {
    using TSim       = unsigned char;
    using TSimAcc    = unsigned int;
    using TSimRefine = _Float16;
}
}

// `float2`, `float3`, `cudaStream_t` — CUDA-isms not on macOS.
// The type-shim layer (S34) will provide POD struct
// definitions for `float2` / `float3` and a `cudaStream_t`
// alias (probably `void*` since we don't honor streams). For
// now, forward-declare bare names so this header parses.
#if !defined(__CUDACC__) && !defined(AV_ADAPTER_TYPES_DEFINED)
struct float2;
struct float3;
typedef void* cudaStream_t;
#endif

namespace aliceVision {
namespace depthMap {

// ============================================================
// Cost-volume primitives (`deviceSimilarityVolume.hpp` surface)
// ============================================================

// Forward to `av::depth_map::Volume::init_sim` / `init_refine`.
// One signature per `TSim` (uchar) and `TSimRefine` (half).
void cuda_volumeInitialize(CudaDeviceMemoryPitched<TSim, 3>& inout_volume_dmp,
                            TSim value,
                            cudaStream_t stream);
void cuda_volumeInitialize(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volume_dmp,
                            TSimRefine value,
                            cudaStream_t stream);

// Forward to `Volume::add_refine`. STUB-OK — declared but not
// called by the 5 host .cpp files in scope.
void cuda_volumeAdd(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volume_dmp,
                    const CudaDeviceMemoryPitched<TSimRefine, 3>& in_volume_dmp,
                    cudaStream_t stream);

// Forward to `Volume::update_uninitialized`. Note arg-order
// swap: upstream (best, secBest); ours (secBest /* inout */,
// best /* in */).
void cuda_volumeUpdateUninitializedSimilarity(
    const CudaDeviceMemoryPitched<TSim, 3>& in_volBestSim_dmp,
    CudaDeviceMemoryPitched<TSim, 3>&       inout_volSecBestSim_dmp,
    cudaStream_t                            stream);

// Forward to `Volume::compute_similarity`. Heaviest translation
// (see adapter map §2.4).
void cuda_volumeComputeSimilarity(
    CudaDeviceMemoryPitched<TSim, 3>&       out_volBestSim_dmp,
    CudaDeviceMemoryPitched<TSim, 3>&       out_volSecBestSim_dmp,
    const CudaDeviceMemoryPitched<float, 2>& in_depths_dmp,
    const int                                rcDeviceCameraParamsId,
    const int                                tcDeviceCameraParamsId,
    const DeviceMipmapImage&                 rcDeviceMipmapImage,
    const DeviceMipmapImage&                 tcDeviceMipmapImage,
    const SgmParams&                         sgmParams,
    const Range&                             depthRange,
    const ROI&                      roi,
    cudaStream_t                             stream);

// Forward to `Volume::refine_similarity`. Drops
// `in_sgmNormalMap_dmpPtr` (no equivalent — risk #2 in adapter
// map); when `useSgmNormalMap` is requested the adapter logs
// a one-time warning and proceeds without normals.
void cuda_volumeRefineSimilarity(
    CudaDeviceMemoryPitched<TSimRefine, 3>&  inout_volSim_dmp,
    const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthPixSizeMap_dmp,
    const CudaDeviceMemoryPitched<float3, 2>* in_sgmNormalMap_dmpPtr,
    const int                                 rcDeviceCameraParamsId,
    const int                                 tcDeviceCameraParamsId,
    const DeviceMipmapImage&                  rcDeviceMipmapImage,
    const DeviceMipmapImage&                  tcDeviceMipmapImage,
    const RefineParams&                       refineParams,
    const Range&                              depthRange,
    const ROI&                       roi,
    cudaStream_t                              stream);

// Forward to `Volume::optimize` (with adaptive-P2 from S31).
void cuda_volumeOptimize(
    CudaDeviceMemoryPitched<TSim, 3>&      out_volSimFiltered_dmp,
    CudaDeviceMemoryPitched<TSimAcc, 2>&    inout_volSliceAccA_dmp,
    CudaDeviceMemoryPitched<TSimAcc, 2>&    inout_volSliceAccB_dmp,
    CudaDeviceMemoryPitched<TSimAcc, 2>&    inout_volAxisAcc_dmp,
    const CudaDeviceMemoryPitched<TSim, 3>& in_volSim_dmp,
    const DeviceMipmapImage&                rcDeviceMipmapImage,
    const SgmParams&                        sgmParams,
    const int                               lastDepthIndex,
    const ROI&                     roi,
    cudaStream_t                            stream);

// Forward to `Volume::retrieve_best_depth`. Note: Sgm.cpp:312
// requests camera params at downscale 1 (not sgmParams.scale).
void cuda_volumeRetrieveBestDepth(
    CudaDeviceMemoryPitched<float2, 2>&    out_sgmDepthThicknessMap_dmp,
    CudaDeviceMemoryPitched<float2, 2>&    out_sgmDepthSimMap_dmp,
    const CudaDeviceMemoryPitched<float, 2>& in_depths_dmp,
    const CudaDeviceMemoryPitched<TSim, 3>&  in_volSim_dmp,
    const int                                rcDeviceCameraParamsId,
    const SgmParams&                         sgmParams,
    const Range&                             depthRange,
    const ROI&                      roi,
    cudaStream_t                             stream);

// Forward to `Volume::refine_best_depth`.
void cuda_volumeRefineBestDepth(
    CudaDeviceMemoryPitched<float2, 2>&        out_refineDepthSimMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>&  in_sgmDepthPixSizeMap_dmp,
    const CudaDeviceMemoryPitched<TSimRefine, 3>& in_volSim_dmp,
    const RefineParams&                        refineParams,
    const ROI&                        roi,
    cudaStream_t                               stream);

// ============================================================
// Depth/sim map primitives (`deviceDepthSimilarityMap.hpp`)
// ============================================================

void cuda_depthSimMapCopyDepthOnly(
    CudaDeviceMemoryPitched<float2, 2>&       out_depthSimMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>& in_depthSimMap_dmp,
    float                                     defaultSim,
    cudaStream_t                              stream);

void cuda_normalMapUpscale(
    CudaDeviceMemoryPitched<float3, 2>&       out_upscaledMap_dmp,
    const CudaDeviceMemoryPitched<float3, 2>& in_map_dmp,
    const ROI&                       roi,
    cudaStream_t                              stream);

void cuda_depthThicknessSmoothThickness(
    CudaDeviceMemoryPitched<float2, 2>& inout_depthThicknessMap_dmp,
    const SgmParams&                    sgmParams,
    const RefineParams&                 refineParams,
    const ROI&                 roi,
    cudaStream_t                        stream);

void cuda_computeSgmUpscaledDepthPixSizeMap(
    CudaDeviceMemoryPitched<float2, 2>&       out_upscaledDepthPixSizeMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthThicknessMap_dmp,
    const int                                 rcDeviceCameraParamsId,
    const DeviceMipmapImage&                  rcDeviceMipmapImage,
    const RefineParams&                       refineParams,
    const ROI&                       roi,
    cudaStream_t                              stream);

void cuda_depthSimMapComputeNormal(
    CudaDeviceMemoryPitched<float3, 2>&       out_normalMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>& in_depthSimMap_dmp,
    const int                                 rcDeviceCameraParamsId,
    const int                                 stepXY,
    const ROI&                       roi,
    cudaStream_t                              stream);

void cuda_depthSimMapOptimizeGradientDescent(
    CudaDeviceMemoryPitched<float2, 2>&       out_optimizeDepthSimMap_dmp,
    CudaDeviceMemoryPitched<float, 2>&         inout_imgVariance_dmp,
    CudaDeviceMemoryPitched<float, 2>&         inout_tmpOptDepthMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>&  in_sgmDepthPixSizeMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>&  in_refineDepthSimMap_dmp,
    const int                                  rcDeviceCameraParamsId,
    const DeviceMipmapImage&                   rcDeviceMipmapImage,
    const RefineParams&                        refineParams,
    const ROI&                        roi,
    cudaStream_t                               stream);

// ============================================================
// Adapter-side state — the camera-params indirection table.
// ============================================================
//
// Upstream addresses camera params by integer ID (an index into
// a CUDA `__constant__` memory array). Our API takes
// `const av::depth_map::DeviceCameraParams&` by reference. The
// adapter holds a translation table; callers must `set()` each
// ID before any kernel that consumes it runs.
//
// These functions are NOT part of the upstream API; they're
// glue used by `DepthMapEstimator`-equivalent code paths.

}}  // namespace aliceVision::depthMap

namespace av::depth_map {
    struct DeviceCameraParams;
    struct DevicePatchPattern;
    class DeviceCache;
}

namespace av::depth_map { namespace upstream_adapter {

// Register / clear / query a camera-params slot identified by
// upstream's `int` id. Idempotent. Backed by a process-global
// table (matches upstream's singleton-style usage).
void set_camera_param(int id, const DeviceCameraParams& params);
const DeviceCameraParams& get_camera_param(int id);   // throws if unset
void clear_camera_params();

// Reset the global DeviceCache used by the adapter forwarders.
// Mainly for tests. The cache is owned by the adapter
// translation unit.
av::depth_map::DeviceCache& adapter_device_cache();

// Current custom patch pattern. Written by the
// `buildCustomPatchPattern` shim (cuda_host_shim/patchPattern.cpp);
// read by the patch-aware `cuda_*` forwarders at dispatch time.
// Defaults to all-zeros (nbSubparts == 0) before any caller has
// built a pattern.
const DevicePatchPattern& current_patch_pattern();
void set_current_patch_pattern(const DevicePatchPattern& p);

}}  // namespace av::depth_map::upstream_adapter
