// upstream_adapter.cpp — Phase 8 adapter implementations.
//
// STATUS (S35): the 12 `cuda_*` forwarders are now real. Each one
// extracts the buffer / dimensions / camera-params / texture from
// the upstream-typed arguments (via the `CudaDeviceMemoryPitched`
// shim and the `aliceVision::depthMap::DeviceMipmapImage` shim) and
// forwards to the corresponding `av::depth_map::*` method.
//
// Pattern (see memory/phase8_adapter_map.md):
//   1. Extract dims via `dmp.getSize()` (CudaSize<2|3>).
//   2. Extract `gpu_buffer()` (adapter-only ext on the shim).
//   3. Look up `DeviceCameraParams` via `get_camera_param(id)`.
//   4. For mipmaps: `mipmap.av_texture()` → `av::gpu::Texture&`;
//      `mipmap.av_impl().get_level(...)` / `get_dimensions(...)`
//      → mipmap level and per-level (w, h).
//   5. Drop `cudaStream_t` (we don't honor streams; everything
//      dispatches on the default queue and finishes synchronously).
//   6. Re-use static `Volume` / `DepthSimMap` singletons that bind
//      to the process-global device set via `set_adapter_device`.

#include "av/depth_map/upstream_adapter.hpp"
#include "av/depth_map/DeviceCache.hpp"
#include "av/depth_map/PatchOps.hpp"           // DeviceCameraParams
#include "av/depth_map/Volume.hpp"
#include "av/depth_map/DepthSimMap.hpp"
#include "av/depth_map/adapter_profile.hpp"    // S43 perf profiling (no-op by default)

#include "av/gpu/Device.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Texture.hpp"

// Bring in the CudaDeviceMemoryPitched / CudaHostMemoryHeap type-
// shim (lives next to memory.hpp's existing copy) AND the
// DeviceMipmapImage shim (new in S35, sits next to memory.hpp).
#include "../../../cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/memory.hpp"
#include "../../../cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/DeviceMipmapImage.hpp"

// Local matching definitions for Range / ROI / SgmParams /
// RefineParams in the namespaces the adapter header forward-
// declares.
#include "upstream_adapter_types.hpp"

#include <cstdio>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

// ============================================================
// Camera-params translation table.
// ============================================================
namespace av::depth_map { namespace upstream_adapter {

namespace {
std::mutex g_mu;
std::unordered_map<int, DeviceCameraParams> g_camera_params;
}  // namespace

void set_camera_param(int id, const DeviceCameraParams& params) {
    std::lock_guard lk(g_mu);
    g_camera_params[id] = params;
}

const DeviceCameraParams& get_camera_param(int id) {
    std::lock_guard lk(g_mu);
    auto it = g_camera_params.find(id);
    if (it == g_camera_params.end()) {
        throw std::out_of_range(
            "upstream_adapter::get_camera_param: id " +
            std::to_string(id) + " not registered. Call "
            "set_camera_param(id, params) first.");
    }
    return it->second;
}

void clear_camera_params() {
    std::lock_guard lk(g_mu);
    g_camera_params.clear();
}

// Adapter-owned DeviceCache (lazy, capacity = 8 mipmaps + 32 cams).
DeviceCache& adapter_device_cache() {
    static DeviceCache instance{
        aliceVision::depthMap::require_adapter_device(),
        /*max_mipmap=*/8,
        /*max_camera=*/32};
    return instance;
}

}}  // namespace av::depth_map::upstream_adapter


// ============================================================
// Shared adapter helpers (Volume / DepthSimMap singletons).
// ============================================================
namespace {

av::depth_map::Volume& adapter_volume() {
    static av::depth_map::Volume v(aliceVision::depthMap::require_adapter_device());
    return v;
}

av::depth_map::DepthSimMap& adapter_depth_sim_map() {
    static av::depth_map::DepthSimMap d(
        aliceVision::depthMap::require_adapter_device());
    return d;
}

// Scratch textures for `cuda_depthSimMapOptimizeGradientDescent`.
// Held as thread_local unique_ptr so the same texture object
// survives across calls; reallocated only when the ROI grows.
struct OptScratch {
    std::unique_ptr<av::gpu::Texture> variance;
    std::unique_ptr<av::gpu::Texture> tmp_depth;
    std::uint32_t w = 0, h = 0;
};

OptScratch& opt_scratch() {
    static thread_local OptScratch s;
    return s;
}

void ensure_opt_scratch(std::uint32_t w, std::uint32_t h) {
    auto& s = opt_scratch();
    if (s.w == w && s.h == h && s.variance && s.tmp_depth) return;
    av::gpu::Texture::Descriptor d;
    d.width      = w;
    d.height     = h;
    d.mip_levels = 1;
    d.format     = av::gpu::PixelFormat::R32Float;
    auto& dev = aliceVision::depthMap::require_adapter_device();
    s.variance  = std::make_unique<av::gpu::Texture>(dev, d);
    s.tmp_depth = std::make_unique<av::gpu::Texture>(dev, d);
    s.w = w;
    s.h = h;
}

}  // namespace


// ============================================================
// 12 cuda_* forwarders.
// ============================================================
namespace aliceVision { namespace depthMap {

// ───────── 2.1 cuda_volumeInitialize (TSim = uchar) ─────────
void cuda_volumeInitialize(CudaDeviceMemoryPitched<TSim, 3>& inout_volume_dmp,
                           TSim                              value,
                           cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_volumeInitialize<TSim>");
    const auto& sz = inout_volume_dmp.getSize();
    const av::depth_map::VolumeDims dims{
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        static_cast<std::uint32_t>(sz[2])};
    adapter_volume().init_sim(inout_volume_dmp.gpu_buffer(),
                              dims,
                              static_cast<std::uint8_t>(value));
}

// ───────── 2.1 cuda_volumeInitialize (TSimRefine = half) ─────────
void cuda_volumeInitialize(CudaDeviceMemoryPitched<TSimRefine, 3>& inout_volume_dmp,
                           TSimRefine                              value,
                           cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_volumeInitialize<TSimRefine>");
    const auto& sz = inout_volume_dmp.getSize();
    const av::depth_map::VolumeDims dims{
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        static_cast<std::uint32_t>(sz[2])};
    adapter_volume().init_refine(inout_volume_dmp.gpu_buffer(),
                                 dims,
                                 static_cast<float>(value));
}

// ───────── 2.2 cuda_volumeAdd (STUB-OK; not called by host code) ─────
void cuda_volumeAdd(CudaDeviceMemoryPitched<TSimRefine, 3>&       inout_volume_dmp,
                    const CudaDeviceMemoryPitched<TSimRefine, 3>& in_volume_dmp,
                    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_volumeAdd");
    const auto& sz = inout_volume_dmp.getSize();
    const av::depth_map::VolumeDims dims{
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        static_cast<std::uint32_t>(sz[2])};
    adapter_volume().add_refine(inout_volume_dmp.gpu_buffer(),
                                in_volume_dmp.gpu_buffer(),
                                dims);
}

// ───────── 2.3 cuda_volumeUpdateUninitializedSimilarity (arg swap) ─────
void cuda_volumeUpdateUninitializedSimilarity(
    const CudaDeviceMemoryPitched<TSim, 3>& in_volBestSim_dmp,
    CudaDeviceMemoryPitched<TSim, 3>&       inout_volSecBestSim_dmp,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_volumeUpdateUninitializedSimilarity");
    // Note arg-order swap: upstream (best, secBest); ours (secBest, best).
    const auto& sz = inout_volSecBestSim_dmp.getSize();
    const av::depth_map::VolumeDims dims{
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        static_cast<std::uint32_t>(sz[2])};
    adapter_volume().update_uninitialized(inout_volSecBestSim_dmp.gpu_buffer(),
                                          in_volBestSim_dmp.gpu_buffer(),
                                          dims);
}

// ───────── 2.4 cuda_volumeComputeSimilarity ─────────
void cuda_volumeComputeSimilarity(
    CudaDeviceMemoryPitched<TSim, 3>&        out_volBestSim_dmp,
    CudaDeviceMemoryPitched<TSim, 3>&        out_volSecBestSim_dmp,
    const CudaDeviceMemoryPitched<float, 2>& in_depths_dmp,
    const int                                rcDeviceCameraParamsId,
    const int                                tcDeviceCameraParamsId,
    const DeviceMipmapImage&                 rcDeviceMipmapImage,
    const DeviceMipmapImage&                 tcDeviceMipmapImage,
    const SgmParams&                         sgmParams,
    const Range&                             depthRange,
    const ROI&                      roi,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_volumeComputeSimilarity");
    using namespace av::depth_map;

    const auto& rc_cam = upstream_adapter::get_camera_param(rcDeviceCameraParamsId);
    const auto& tc_cam = upstream_adapter::get_camera_param(tcDeviceCameraParamsId);

    auto& rc_inner = rcDeviceMipmapImage.av_impl();
    auto& tc_inner = tcDeviceMipmapImage.av_impl();

    const auto [rc_lw, rc_lh] = rc_inner.get_dimensions(static_cast<std::uint32_t>(sgmParams.scale));
    const auto [tc_lw, tc_lh] = tc_inner.get_dimensions(static_cast<std::uint32_t>(sgmParams.scale));

    const auto& sz = out_volBestSim_dmp.getSize();

    Volume::ComputeSimilarityParams p;
    p.dims = VolumeDims{
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        static_cast<std::uint32_t>(sz[2])};
    p.rc_sgm_level_width  = rc_lw;
    p.rc_sgm_level_height = rc_lh;
    p.tc_sgm_level_width  = tc_lw;
    p.tc_sgm_level_height = tc_lh;
    p.rc_mipmap_level     = rc_inner.get_level(static_cast<std::uint32_t>(sgmParams.scale));
    p.step_xy             = static_cast<std::int32_t>(sgmParams.stepXY);
    p.wsh                 = static_cast<std::int32_t>(sgmParams.wsh);
    p.inv_gamma_c         = 1.0f / static_cast<float>(sgmParams.gammaC);
    p.inv_gamma_p         = 1.0f / static_cast<float>(sgmParams.gammaP);
    p.use_consistent_scale = sgmParams.useConsistentScale ? 1u : 0u;
    p.depth_range_begin    = depthRange.begin;
    p.depth_range_end      = depthRange.end;
    p.roi_x_begin          = roi.x.begin;
    p.roi_y_begin          = roi.y.begin;
    p.roi_width            = roi.width();
    p.roi_height           = roi.height();
    (void)in_depths_dmp.getSize();   // dims sanity-check is left to the kernel.

    adapter_volume().compute_similarity(
        out_volBestSim_dmp.gpu_buffer(),
        out_volSecBestSim_dmp.gpu_buffer(),
        in_depths_dmp.gpu_buffer(),
        rcDeviceMipmapImage.av_texture(),
        tcDeviceMipmapImage.av_texture(),
        rc_cam,
        tc_cam,
        p);
}

// ───────── 2.5 cuda_volumeRefineSimilarity (drop sgmNormalMap) ─────────
void cuda_volumeRefineSimilarity(
    CudaDeviceMemoryPitched<TSimRefine, 3>&     inout_volSim_dmp,
    const CudaDeviceMemoryPitched<float2, 2>&    in_sgmDepthPixSizeMap_dmp,
    const CudaDeviceMemoryPitched<float3, 2>*    in_sgmNormalMap_dmpPtr,
    const int                                    rcDeviceCameraParamsId,
    const int                                    tcDeviceCameraParamsId,
    const DeviceMipmapImage&                     rcDeviceMipmapImage,
    const DeviceMipmapImage&                     tcDeviceMipmapImage,
    const RefineParams&                          refineParams,
    const Range&                                 depthRange,
    const ROI&                          roi,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_volumeRefineSimilarity");
    // One-time warning if the caller supplied a SGM normal map.
    if (in_sgmNormalMap_dmpPtr != nullptr) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                "cuda_volumeRefineSimilarity: in_sgmNormalMap_dmpPtr "
                "ignored (no equivalent in av::depth_map; see "
                "memory/handover_session.md S22)\n");
            warned = true;
        }
    }

    using namespace av::depth_map;
    const auto& rc_cam = upstream_adapter::get_camera_param(rcDeviceCameraParamsId);
    const auto& tc_cam = upstream_adapter::get_camera_param(tcDeviceCameraParamsId);

    auto& rc_inner = rcDeviceMipmapImage.av_impl();
    auto& tc_inner = tcDeviceMipmapImage.av_impl();

    const auto [rc_lw, rc_lh] = rc_inner.get_dimensions(static_cast<std::uint32_t>(refineParams.scale));
    const auto [tc_lw, tc_lh] = tc_inner.get_dimensions(static_cast<std::uint32_t>(refineParams.scale));

    const auto& sz = inout_volSim_dmp.getSize();

    Volume::RefineSimilarityParams p;
    p.dims = VolumeDims{
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        static_cast<std::uint32_t>(sz[2])};
    p.rc_refine_level_width  = rc_lw;
    p.rc_refine_level_height = rc_lh;
    p.tc_refine_level_width  = tc_lw;
    p.tc_refine_level_height = tc_lh;
    p.rc_mipmap_level        = rc_inner.get_level(static_cast<std::uint32_t>(refineParams.scale));
    p.step_xy                = static_cast<std::int32_t>(refineParams.stepXY);
    p.wsh                    = static_cast<std::int32_t>(refineParams.wsh);
    p.inv_gamma_c            = 1.0f / static_cast<float>(refineParams.gammaC);
    p.inv_gamma_p            = 1.0f / static_cast<float>(refineParams.gammaP);
    p.use_consistent_scale   = refineParams.useConsistentScale ? 1u : 0u;
    p.depth_range_begin      = depthRange.begin;
    p.depth_range_end        = depthRange.end;
    p.roi_x_begin            = roi.x.begin;
    p.roi_y_begin            = roi.y.begin;
    p.roi_width              = roi.width();
    p.roi_height             = roi.height();
    (void)in_sgmDepthPixSizeMap_dmp.getSize();

    adapter_volume().refine_similarity(
        inout_volSim_dmp.gpu_buffer(),
        in_sgmDepthPixSizeMap_dmp.gpu_buffer(),
        rcDeviceMipmapImage.av_texture(),
        tcDeviceMipmapImage.av_texture(),
        rc_cam,
        tc_cam,
        p);
}

// ───────── 2.6 cuda_volumeOptimize (P2 sign convention) ─────────
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
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_volumeOptimize");
    using namespace av::depth_map;

    const auto& sz = in_volSim_dmp.getSize();
    auto& rc_inner = rcDeviceMipmapImage.av_impl();
    const auto [rc_lw, rc_lh] = rc_inner.get_dimensions(static_cast<std::uint32_t>(sgmParams.scale));

    Volume::OptimizeParams p;
    p.dims = VolumeDims{
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        static_cast<std::uint32_t>(sz[2])};
    p.last_depth_index = static_cast<std::uint32_t>(lastDepthIndex);
    p.p1               = static_cast<float>(sgmParams.p1);

    // Upstream P2 sign convention: see deviceSimilarityVolumeKernels.cuh:698:
    //   _P2 < 0  → fixed P2 = abs(_P2)         (skip deltaC)
    //   _P2 >= 0 → adaptive P2 via sigmoid(80, 255, 80, _P2, deltaC)
    const float p2_signed = static_cast<float>(sgmParams.p2Weighting);
    if (p2_signed < 0.0f) {
        p.adaptive_p2 = false;
        p.p2_abs      = -p2_signed;
        p.p2_sig_mid  = -p2_signed;   // unused but keep deterministic
    } else {
        p.adaptive_p2 = true;
        p.p2_sig_mid  = p2_signed;    // sigMid (signed, NOT abs'd; matches upstream)
        p.p2_abs      = p2_signed;    // unused in adaptive path
    }

    p.step_xy        = static_cast<std::int32_t>(sgmParams.stepXY);
    p.roi_x_begin    = static_cast<std::int32_t>(roi.x.begin);
    p.roi_y_begin    = static_cast<std::int32_t>(roi.y.begin);
    p.rc_level_width  = rc_lw;
    p.rc_level_height = rc_lh;
    p.rc_mipmap_level = rc_inner.get_level(static_cast<std::uint32_t>(sgmParams.scale));

    const av::gpu::Texture* rc_tex_ptr =
        p.adaptive_p2 ? &rcDeviceMipmapImage.av_texture() : nullptr;

    adapter_volume().optimize(out_volSimFiltered_dmp.gpu_buffer(),
                              inout_volSliceAccA_dmp.gpu_buffer(),
                              inout_volSliceAccB_dmp.gpu_buffer(),
                              inout_volAxisAcc_dmp.gpu_buffer(),
                              in_volSim_dmp.gpu_buffer(),
                              p,
                              rc_tex_ptr);
}

// ───────── 2.7 cuda_volumeRetrieveBestDepth ─────────
void cuda_volumeRetrieveBestDepth(
    CudaDeviceMemoryPitched<float2, 2>&      out_sgmDepthThicknessMap_dmp,
    CudaDeviceMemoryPitched<float2, 2>&      out_sgmDepthSimMap_dmp,
    const CudaDeviceMemoryPitched<float, 2>& in_depths_dmp,
    const CudaDeviceMemoryPitched<TSim, 3>&  in_volSim_dmp,
    const int                                rcDeviceCameraParamsId,
    const SgmParams&                         sgmParams,
    const Range&                             depthRange,
    const ROI&                      roi,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_volumeRetrieveBestDepth");
    using namespace av::depth_map;
    // Per the adapter map: Sgm.cpp:312 requests camera params at
    // downscale 1, so `get_camera_param(rcDeviceCameraParamsId)`
    // already points to the right cam.
    const auto& rc_cam = upstream_adapter::get_camera_param(rcDeviceCameraParamsId);

    const auto& sz = in_volSim_dmp.getSize();

    Volume::RetrieveBestDepthParams p;
    p.dims = VolumeDims{
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        static_cast<std::uint32_t>(sz[2])};
    p.depth_range_begin = depthRange.begin;
    p.depth_range_end   = depthRange.end;
    p.roi_x_begin       = roi.x.begin;
    p.roi_y_begin       = roi.y.begin;
    p.scale_step        = static_cast<std::int32_t>(sgmParams.scale * sgmParams.stepXY);
    // Upstream's `deviceSimilarityVolume.cu:440` computes
    //   `const float thicknessMultFactor = 1.f + sgmParams.depthThicknessInflate;`
    // i.e. the user-facing field is an INFLATION ADDED to the default of
    // 1.0. Default `depthThicknessInflate=0` should yield mult=1.0
    // (raw inter-plane gap). Without the `1.f +`, default mult=0 zeroes
    // out the thickness — propagates through SgmUpscale → pixSize=0 →
    // division by zero in color-opt → NaN cascade → final depth EXR
    // collapses to 0 / sentinel values.
    p.thickness_mult_factor = 1.f + static_cast<float>(sgmParams.depthThicknessInflate);
    // Upstream scales maxSimilarity by 254 before passing to the kernel
    // (deviceSimilarityVolume.cu:441) — converts from [0, 1] application
    // semantics to the [0, 254] uchar range that bestSim lives in inside
    // the kernel. Without this scaling, default `sgmParams.maxSimilarity
    // = 1.0` causes the kernel to reject almost every valid voxel
    // (bestSim values in [0, 254] are almost always > 1.0).
    p.max_similarity        = static_cast<float>(sgmParams.maxSimilarity) * 254.f;

    // Upstream's Sgm.cpp:58 conditionally allocates `_depthSimMap_dmp`
    // (only when SgmParams.computeDepthSimMap is true); when it's not
    // allocated, the buffer is still passed here. CUDA tolerates writes
    // to a null device pointer (silent no-op on most kernels). Our Metal
    // adapter dereferences a null unique_ptr in `gpu_buffer()` → UB. So
    // we lazy-allocate the secondary output to match the thickness map's
    // dimensions when needed. The kernel result for this map is discarded
    // by callers in the !computeDepthSimMap path.
    if (out_sgmDepthSimMap_dmp.getBytesPadded() == 0) {
        out_sgmDepthSimMap_dmp.allocate(out_sgmDepthThicknessMap_dmp.getSize());
    }
    adapter_volume().retrieve_best_depth(
        out_sgmDepthThicknessMap_dmp.gpu_buffer(),
        out_sgmDepthSimMap_dmp.gpu_buffer(),
        in_depths_dmp.gpu_buffer(),
        in_volSim_dmp.gpu_buffer(),
        rc_cam,
        p);
}

// ───────── 2.8 cuda_volumeRefineBestDepth ─────────
void cuda_volumeRefineBestDepth(
    CudaDeviceMemoryPitched<float2, 2>&            out_refineDepthSimMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>&      in_sgmDepthPixSizeMap_dmp,
    const CudaDeviceMemoryPitched<TSimRefine, 3>&  in_volSim_dmp,
    const RefineParams&                            refineParams,
    const ROI&                            roi,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_volumeRefineBestDepth");
    using namespace av::depth_map;

    const auto& sz = in_volSim_dmp.getSize();

    Volume::RefineBestDepthParams p;
    p.dims = VolumeDims{
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        static_cast<std::uint32_t>(sz[2])};
    p.samples_per_pix_size = static_cast<std::int32_t>(refineParams.nbSubsamples);
    // Upstream's `deviceSimilarityVolume.cu:478` computes
    //   halfNbSamples = refineParams.nbSubsamples * refineParams.halfNbDepths
    // which is the number of sub-sample slots (in front and behind mid depth)
    // across the full Z sweep. The kernel uses this to bound its Z loop;
    // `halfNbDepths` is the COARSE depth-plane count and is passed
    // separately. Mirror the exact product here.
    p.half_nb_samples = static_cast<std::int32_t>(refineParams.halfNbDepths *
                                                  refineParams.nbSubsamples);
    p.half_nb_depths  = static_cast<std::int32_t>(refineParams.halfNbDepths);
    const float sigma = static_cast<float>(refineParams.sigma);
    p.two_times_sigma_pow_two = 2.0f * sigma * sigma;
    p.roi_width  = roi.width();
    p.roi_height = roi.height();
    (void)in_sgmDepthPixSizeMap_dmp.getSize();

    adapter_volume().refine_best_depth(
        out_refineDepthSimMap_dmp.gpu_buffer(),
        in_sgmDepthPixSizeMap_dmp.gpu_buffer(),
        in_volSim_dmp.gpu_buffer(),
        p);
}

// ───────── 2.9 cuda_depthSimMapCopyDepthOnly ─────────
void cuda_depthSimMapCopyDepthOnly(
    CudaDeviceMemoryPitched<float2, 2>&       out_depthSimMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>& in_depthSimMap_dmp,
    float                                     defaultSim,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_depthSimMapCopyDepthOnly");
    const auto& sz = out_depthSimMap_dmp.getSize();
    adapter_depth_sim_map().copy_depth_only(
        out_depthSimMap_dmp.gpu_buffer(),
        in_depthSimMap_dmp.gpu_buffer(),
        static_cast<std::uint32_t>(sz[0]),
        static_cast<std::uint32_t>(sz[1]),
        defaultSim);
}

// ───────── 2.10 cuda_normalMapUpscale ─────────
void cuda_normalMapUpscale(
    CudaDeviceMemoryPitched<float3, 2>&       out_upscaledMap_dmp,
    const CudaDeviceMemoryPitched<float3, 2>& in_map_dmp,
    const ROI&                       roi,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_normalMapUpscale");
    const auto& in_sz = in_map_dmp.getSize();
    adapter_depth_sim_map().normal_map_upscale(
        out_upscaledMap_dmp.gpu_buffer(),
        in_map_dmp.gpu_buffer(),
        roi.width(),
        roi.height(),
        static_cast<std::uint32_t>(in_sz[0]),
        static_cast<std::uint32_t>(in_sz[1]));
}

// ───────── 2.11 cuda_depthThicknessSmoothThickness ─────────
void cuda_depthThicknessSmoothThickness(
    CudaDeviceMemoryPitched<float2, 2>& inout_depthThicknessMap_dmp,
    const SgmParams&                    sgmParams,
    const RefineParams&                 refineParams,
    const ROI&                 roi,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_depthThicknessSmoothThickness");
    // sgmScaleStep    = sgmParams.scale    * sgmParams.stepXY
    // refineScaleStep = refineParams.scale * refineParams.stepXY
    // max_nb_refine_samples = max(sgmScaleStep / refineScaleStep, 2)
    // min_thickness_inflate = refineParams.halfNbDepths / max_nb_refine_samples
    // max_thickness_inflate = refineParams.halfNbDepths / 2
    const float sgm_scale_step    = static_cast<float>(sgmParams.scale    * sgmParams.stepXY);
    const float refine_scale_step = static_cast<float>(refineParams.scale * refineParams.stepXY);
    const float ratio             = (refine_scale_step > 0.0f)
                                  ? (sgm_scale_step / refine_scale_step) : 2.0f;
    const float max_nb_refine_samples = (ratio < 2.0f) ? 2.0f : ratio;
    const float min_thickness_inflate =
        static_cast<float>(refineParams.halfNbDepths) / max_nb_refine_samples;
    const float max_thickness_inflate =
        static_cast<float>(refineParams.halfNbDepths) * 0.5f;

    adapter_depth_sim_map().smooth_thickness(
        inout_depthThicknessMap_dmp.gpu_buffer(),
        roi.width(), roi.height(),
        min_thickness_inflate, max_thickness_inflate);
}

// ───────── 2.12 cuda_computeSgmUpscaledDepthPixSizeMap ─────────
void cuda_computeSgmUpscaledDepthPixSizeMap(
    CudaDeviceMemoryPitched<float2, 2>&       out_upscaledDepthPixSizeMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthThicknessMap_dmp,
    const int                                 /*rcDeviceCameraParamsId*/,
    const DeviceMipmapImage&                  rcDeviceMipmapImage,
    const RefineParams&                       refineParams,
    const ROI&                       roi,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_computeSgmUpscaledDepthPixSizeMap");
    using namespace av::depth_map;

    auto& rc_inner = rcDeviceMipmapImage.av_impl();
    const auto [rc_lw, rc_lh] = rc_inner.get_dimensions(static_cast<std::uint32_t>(refineParams.scale));

    const auto& out_sz = out_upscaledDepthPixSizeMap_dmp.getSize();
    const auto& in_sz  = in_sgmDepthThicknessMap_dmp.getSize();

    DepthSimMap::ComputeUpscaledDepthPixSizeMapParams p;
    p.out_width       = static_cast<std::uint32_t>(out_sz[0]);
    p.out_height      = static_cast<std::uint32_t>(out_sz[1]);
    p.in_width        = static_cast<std::uint32_t>(in_sz[0]);
    p.in_height       = static_cast<std::uint32_t>(in_sz[1]);
    p.roi_x_begin     = roi.x.begin;
    p.roi_y_begin     = roi.y.begin;
    p.rc_level_width  = rc_lw;
    p.rc_level_height = rc_lh;
    p.rc_mipmap_level = rc_inner.get_level(static_cast<std::uint32_t>(refineParams.scale));
    p.step_xy         = static_cast<std::int32_t>(refineParams.stepXY);
    p.half_nb_depths  = static_cast<std::int32_t>(refineParams.halfNbDepths);
    p.bilinear        = refineParams.interpolateMiddleDepth;

    adapter_depth_sim_map().compute_sgm_upscaled_depth_pix_size_map(
        out_upscaledDepthPixSizeMap_dmp.gpu_buffer(),
        in_sgmDepthThicknessMap_dmp.gpu_buffer(),
        rcDeviceMipmapImage.av_texture(),
        p);
}

// ───────── 2.13 cuda_depthSimMapComputeNormal ─────────
void cuda_depthSimMapComputeNormal(
    CudaDeviceMemoryPitched<float3, 2>&       out_normalMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>& in_depthSimMap_dmp,
    const int                                 rcDeviceCameraParamsId,
    const int                                 stepXY,
    const ROI&                       roi,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_depthSimMapComputeNormal");
    using namespace av::depth_map;
    const auto& rc_cam = upstream_adapter::get_camera_param(rcDeviceCameraParamsId);

    DepthSimMap::ComputeNormalParams p;
    p.width        = roi.width();
    p.height       = roi.height();
    p.roi_x_begin  = roi.x.begin;
    p.roi_y_begin  = roi.y.begin;
    p.step_xy      = static_cast<std::int32_t>(stepXY);
    p.wsh          = 3;   // upstream hardcodes wsh=3 template specialization
    (void)in_depthSimMap_dmp.getSize();

    adapter_depth_sim_map().compute_normal(
        out_normalMap_dmp.gpu_buffer(),
        in_depthSimMap_dmp.gpu_buffer(),
        rc_cam,
        p);
}

// ───────── 2.14 cuda_depthSimMapOptimizeGradientDescent ─────────
void cuda_depthSimMapOptimizeGradientDescent(
    CudaDeviceMemoryPitched<float2, 2>&        out_optimizeDepthSimMap_dmp,
    CudaDeviceMemoryPitched<float, 2>&         /*inout_imgVariance_dmp*/,
    CudaDeviceMemoryPitched<float, 2>&         /*inout_tmpOptDepthMap_dmp*/,
    const CudaDeviceMemoryPitched<float2, 2>&  in_sgmDepthPixSizeMap_dmp,
    const CudaDeviceMemoryPitched<float2, 2>&  in_refineDepthSimMap_dmp,
    const int                                  rcDeviceCameraParamsId,
    const DeviceMipmapImage&                   rcDeviceMipmapImage,
    const RefineParams&                        refineParams,
    const ROI&                        roi,
    cudaStream_t /*stream*/)
{
    AV_ADAPTER_PROFILE_SCOPE("cuda_depthSimMapOptimizeGradientDescent");
    using namespace av::depth_map;
    const auto& rc_cam = upstream_adapter::get_camera_param(rcDeviceCameraParamsId);

    auto& rc_inner = rcDeviceMipmapImage.av_impl();
    const auto [rc_lw, rc_lh] = rc_inner.get_dimensions(static_cast<std::uint32_t>(refineParams.scale));

    const std::uint32_t w = roi.width();
    const std::uint32_t h = roi.height();

    // Upstream's variance + tmp-depth scratch buffers are
    // `CudaDeviceMemoryPitched<float, 2>` (pitched 2D float
    // buffers). Our method needs `av::gpu::Texture&` for both.
    // We allocate persistent scratch textures, reallocating only
    // when the ROI grows.
    ensure_opt_scratch(w, h);
    auto& s = opt_scratch();

    DepthSimMap::OptimizeGradientDescentParams p;
    p.width           = w;
    p.height          = h;
    p.roi_x_begin     = roi.x.begin;
    p.roi_y_begin     = roi.y.begin;
    p.rc_level_width  = rc_lw;
    p.rc_level_height = rc_lh;
    p.rc_mipmap_level = rc_inner.get_level(static_cast<std::uint32_t>(refineParams.scale));
    p.step_xy         = static_cast<std::int32_t>(refineParams.stepXY);
    p.nb_iterations   = static_cast<std::int32_t>(refineParams.optimizationNbIterations);

    adapter_depth_sim_map().optimize_depth_sim_map(
        out_optimizeDepthSimMap_dmp.gpu_buffer(),
        in_sgmDepthPixSizeMap_dmp.gpu_buffer(),
        in_refineDepthSimMap_dmp.gpu_buffer(),
        rcDeviceMipmapImage.av_texture(),
        *s.variance,
        *s.tmp_depth,
        rc_cam,
        p);
}

}}  // namespace aliceVision::depthMap
