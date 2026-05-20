// DeviceCache.cpp — Apple Silicon implementation for the upstream
// `aliceVision::depthMap::DeviceCache` shim declared in
// `cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/DeviceCache.hpp`.
//
// Design contract
// ---------------
//  * The shim is a singleton (matches upstream).
//  * Internally it owns:
//      - one `av::depth_map::DeviceCache` for LRU semantics + mipmap
//        storage (we re-use this for the camera-params storage too,
//        but only consult it for the slot index — see below);
//      - a parallel array of shim `DeviceMipmapImage` wrappers, each
//        pre-bound to the matching `av::depth_map::DeviceMipmapImage`
//        inside the LRU so that `requestMipmapImage` can return a
//        stable `const DeviceMipmapImage&`;
//      - a (camId, downscale) → slot map for camera params, so
//        `requestCameraParamsId` is O(1).
//  * Camera-param flow:
//      `addCameraParams` builds a `DeviceCameraParams` from
//      `MultiViewParams` (same math as upstream's
//      `fillHostCameraParameters`), then calls
//      `av::depth_map::upstream_adapter::set_camera_param(id, params)`
//      with the slot index. The slot index is the upstream "id".
//      `requestCameraParamsId` returns that same slot index, so the
//      12 `cuda_*` forwarders can resolve `id → DeviceCameraParams&`
//      by calling `get_camera_param(id)`.

#include "aliceVision/depthMap/cuda/host/DeviceCache.hpp"
#include "aliceVision/depthMap/cuda/host/DeviceMipmapImage.hpp"  // shim
#include "aliceVision/depthMap/cuda/host/memory.hpp"             // shim
#include "aliceVision/depthMap/cuda/host/utils.hpp"              // shim

#include "av/depth_map/DeviceCache.hpp"
#include "av/depth_map/DeviceMipmapImage.hpp"
#include "av/depth_map/PatchOps.hpp"      // av::depth_map::DeviceCameraParams
#include "av/depth_map/upstream_adapter.hpp"
#include "av/gpu/Device.hpp"

#include <aliceVision/mvsData/Matrix3x3.hpp>
#include <aliceVision/mvsData/Matrix3x4.hpp>
#include <aliceVision/mvsData/Point3d.hpp>

#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

// --------------------------------------------------------------------
// Camera-param translation — pure port of upstream's
// `fillHostCameraParameters` (DeviceCache.cpp:41) but writing into
// our `av::depth_map::DeviceCameraParams`. Two things to note:
//
//   (a) Our struct uses `float[3]` for `C / XVect / YVect / ZVect`
//       where upstream uses `float3 { x, y, z; }`. The layout is
//       identical (12 bytes, no padding) — confirmed by static_assert
//       in our PatchOps.hpp.
//   (b) `K`, `R`, `P` etc. are stored column-major. We match upstream's
//       column-major fill order exactly so the resulting bytes are
//       interchangeable with any kernel built against upstream's
//       constant-memory layout.
// --------------------------------------------------------------------

inline void normalize3(float v[3]) {
    const float d = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (d > 0.0f) { v[0] /= d; v[1] /= d; v[2] /= d; }
}

// Multiply a 3x3 column-major matrix (stored as float[9]) by a
// 3-vector. Matches the math in upstream's `M3x3mulV3`.
inline void m3x3_mul_v3(const float M[9],
                        float x, float y, float z,
                        float out[3]) {
    out[0] = M[0]*x + M[3]*y + M[6]*z;
    out[1] = M[1]*x + M[4]*y + M[7]*z;
    out[2] = M[2]*x + M[5]*y + M[8]*z;
}

void fill_host_camera_params(av::depth_map::DeviceCameraParams& cp,
                             int                                camId,
                             int                                downscale,
                             const aliceVision::mvsUtils::MultiViewParams& mp)
{
    using aliceVision::Matrix3x3;
    using aliceVision::Matrix3x4;
    using aliceVision::Point3d;

    // scaleM = diag(1/d, 1/d, 1)
    Matrix3x3 scaleM;
    scaleM.m11 = 1.0 / double(downscale); scaleM.m12 = 0.0;                    scaleM.m13 = 0.0;
    scaleM.m21 = 0.0;                     scaleM.m22 = 1.0 / double(downscale); scaleM.m23 = 0.0;
    scaleM.m31 = 0.0;                     scaleM.m32 = 0.0;                    scaleM.m33 = 1.0;

    const Matrix3x3 K  = scaleM * mp.KArr[camId];
    const Matrix3x3 iK = K.inverse();
    const Matrix3x4 P  =
        K * (mp.RArr[camId] | (Point3d(0.0, 0.0, 0.0) - mp.RArr[camId] * mp.CArr[camId]));
    const Matrix3x3 iP = mp.iRArr[camId] * iK;

    // C
    cp.C[0] = float(mp.CArr[camId].x);
    cp.C[1] = float(mp.CArr[camId].y);
    cp.C[2] = float(mp.CArr[camId].z);

    // P (column-major fill, matches upstream)
    cp.P[ 0] = float(P.m11); cp.P[ 1] = float(P.m21); cp.P[ 2] = float(P.m31);
    cp.P[ 3] = float(P.m12); cp.P[ 4] = float(P.m22); cp.P[ 5] = float(P.m32);
    cp.P[ 6] = float(P.m13); cp.P[ 7] = float(P.m23); cp.P[ 8] = float(P.m33);
    cp.P[ 9] = float(P.m14); cp.P[10] = float(P.m24); cp.P[11] = float(P.m34);

    // iP, R, iR, K, iK — all column-major 3x3
    cp.iP[0] = float(iP.m11); cp.iP[1] = float(iP.m21); cp.iP[2] = float(iP.m31);
    cp.iP[3] = float(iP.m12); cp.iP[4] = float(iP.m22); cp.iP[5] = float(iP.m32);
    cp.iP[6] = float(iP.m13); cp.iP[7] = float(iP.m23); cp.iP[8] = float(iP.m33);

    cp.R[0] = float(mp.RArr[camId].m11); cp.R[1] = float(mp.RArr[camId].m21); cp.R[2] = float(mp.RArr[camId].m31);
    cp.R[3] = float(mp.RArr[camId].m12); cp.R[4] = float(mp.RArr[camId].m22); cp.R[5] = float(mp.RArr[camId].m32);
    cp.R[6] = float(mp.RArr[camId].m13); cp.R[7] = float(mp.RArr[camId].m23); cp.R[8] = float(mp.RArr[camId].m33);

    cp.iR[0] = float(mp.iRArr[camId].m11); cp.iR[1] = float(mp.iRArr[camId].m21); cp.iR[2] = float(mp.iRArr[camId].m31);
    cp.iR[3] = float(mp.iRArr[camId].m12); cp.iR[4] = float(mp.iRArr[camId].m22); cp.iR[5] = float(mp.iRArr[camId].m32);
    cp.iR[6] = float(mp.iRArr[camId].m13); cp.iR[7] = float(mp.iRArr[camId].m23); cp.iR[8] = float(mp.iRArr[camId].m33);

    cp.K[0] = float(K.m11); cp.K[1] = float(K.m21); cp.K[2] = float(K.m31);
    cp.K[3] = float(K.m12); cp.K[4] = float(K.m22); cp.K[5] = float(K.m32);
    cp.K[6] = float(K.m13); cp.K[7] = float(K.m23); cp.K[8] = float(K.m33);

    cp.iK[0] = float(iK.m11); cp.iK[1] = float(iK.m21); cp.iK[2] = float(iK.m31);
    cp.iK[3] = float(iK.m12); cp.iK[4] = float(iK.m22); cp.iK[5] = float(iK.m32);
    cp.iK[6] = float(iK.m13); cp.iK[7] = float(iK.m23); cp.iK[8] = float(iK.m33);

    // Camera axes via iR · {e_x, e_y, e_z}, then normalize.
    m3x3_mul_v3(cp.iR, 1.0f, 0.0f, 0.0f, cp.XVect); normalize3(cp.XVect);
    m3x3_mul_v3(cp.iR, 0.0f, 1.0f, 0.0f, cp.YVect); normalize3(cp.YVect);
    m3x3_mul_v3(cp.iR, 0.0f, 0.0f, 1.0f, cp.ZVect); normalize3(cp.ZVect);
}

}  // namespace


namespace aliceVision {
namespace depthMap {

// --------------------------------------------------------------------
// Impl
// --------------------------------------------------------------------

struct DeviceCache::Impl {
    int max_mipmap_images = 0;
    int max_camera_params = 0;

    // Native LRU cache provides the mipmap storage + (camId,
    // downscale) slot-allocation we need. We do NOT use it for
    // the camera-params bytes (those live in the adapter
    // translation table); we just borrow its LRU slot allocation
    // by calling `add_camera_params(...)` with a sentinel struct.
    // (The bytes there are unused — kernels read params via the
    // adapter table, not via the native cache.)
    std::unique_ptr<av::depth_map::DeviceCache> native;

    // One shim wrapper per mipmap slot. Bound to the matching
    // native mipmap on first use; lifetime equals the Impl's.
    std::vector<std::unique_ptr<DeviceMipmapImage>> shim_mipmaps;

    // (camId, downscale) → slot index in the camera-params LRU.
    // We track this here because the native cache doesn't expose
    // the slot it picked; we mirror it.
    struct KeyHash {
        std::size_t operator()(const std::pair<int,int>& k) const noexcept {
            return std::hash<std::int64_t>()(
                (std::int64_t(k.first) << 32) ^ std::uint32_t(k.second));
        }
    };
    // (camId, downscale) → adapter-table id
    std::map<std::pair<int,int>, int> cam_param_slot;

    // (camId)            → shim-mipmap slot
    std::map<int, int> mipmap_slot;

    // Next free slot for camera params, monotonically increasing.
    // We do NOT reuse slots even after eviction — the adapter table
    // keys on global IDs, not LRU slots, so a fresh ID per insert
    // is the safest semantics. (Upstream re-uses slots because its
    // table is the LRU itself; ours isn't.)
    int next_cam_param_id = 0;

    bool built = false;

    void require_built() const {
        if (!built) {
            throw std::runtime_error(
                "aliceVision::depthMap::DeviceCache: build(maxMipmap, "
                "maxCameraParams) must be called before use.");
        }
    }
};

DeviceCache::DeviceCache()  = default;
DeviceCache::~DeviceCache() = default;

void DeviceCache::clear() {
    impl_.reset();
}

void DeviceCache::build(int maxMipmapImages, int maxCameraParams) {
    if (maxMipmapImages <= 0 || maxCameraParams <= 0) {
        throw std::invalid_argument(
            "DeviceCache::build: maxMipmapImages and maxCameraParams "
            "must both be > 0.");
    }
    impl_ = std::make_unique<Impl>();
    impl_->max_mipmap_images = maxMipmapImages;
    impl_->max_camera_params = maxCameraParams;
    impl_->native = std::make_unique<av::depth_map::DeviceCache>(
        require_adapter_device(), maxMipmapImages, maxCameraParams);
    impl_->shim_mipmaps.reserve(static_cast<std::size_t>(maxMipmapImages));
    for (int i = 0; i < maxMipmapImages; ++i) {
        impl_->shim_mipmaps.push_back(std::make_unique<DeviceMipmapImage>());
    }
    impl_->built = true;
}

void DeviceCache::addMipmapImage(
    int camId,
    int minDownscale,
    int maxDownscale,
    mvsUtils::ImagesCache<image::Image<image::RGBAfColor>>& imageCache,
    const mvsUtils::MultiViewParams& mp)
{
    if (!impl_) {
        // Auto-build with reasonable defaults if the caller forgot
        // (matches the spirit of upstream's lazy-init).
        build(/*maxMipmap=*/8, /*maxCamera=*/32);
    }

    // Skip the heavy ingest if we already have this camId cached.
    auto it = impl_->mipmap_slot.find(camId);
    if (it != impl_->mipmap_slot.end()) {
        return;
    }

    // Read the host-side RGBA float image via ImagesCache.
    auto img = imageCache.getImg_sync(camId);
    const int w = img->width();
    const int h = img->height();
    if (w <= 0 || h <= 0) {
        throw std::runtime_error(
            "DeviceCache::addMipmapImage: image has non-positive dims.");
    }

    // Pack as a flat RGBA float-255 buffer (matches upstream's
    // `CudaHostMemoryHeap<CudaRGBA, 2>` fill in DeviceCache.cpp:256
    // — same `× 255.0f` scaling).
    std::vector<float> rgba(std::size_t(w) * h * 4u);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const image::RGBAfColor& c = (*img)(y, x);
            const std::size_t k = (std::size_t(y) * w + x) * 4u;
            rgba[k + 0] = float(c.r()) * 255.0f;
            rgba[k + 1] = float(c.g()) * 255.0f;
            rgba[k + 2] = float(c.b()) * 255.0f;
            rgba[k + 3] = float(c.a()) * 255.0f;
        }
    }

    // Allocate a new slot in the native LRU. The slot index is the
    // position into `shim_mipmaps`.
    const int slot = static_cast<int>(impl_->mipmap_slot.size())
                     % impl_->max_mipmap_images;
    // Forward to the native cache (it handles the actual mipmap
    // pyramid construction).
    impl_->native->add_mipmap_image(
        camId,
        static_cast<std::uint32_t>(minDownscale),
        static_cast<std::uint32_t>(maxDownscale),
        std::span<const float>(rgba.data(), rgba.size()),
        static_cast<std::uint32_t>(w),
        static_cast<std::uint32_t>(h));

    // Bind the shim wrapper at this slot to the native mipmap.
    // The const_cast is needed because `request_mipmap_image`
    // returns const; the shim's `set_av_impl` takes non-const so
    // the eventual `fill()` (which mutates) compiles. We never
    // mutate through the shim wrapper here — `fill()` already ran
    // inside `add_mipmap_image()`.
    const auto& native_mip = impl_->native->request_mipmap_image(camId);
    impl_->shim_mipmaps[static_cast<std::size_t>(slot)]->set_av_impl(
        const_cast<av::depth_map::DeviceMipmapImage&>(native_mip));
    impl_->mipmap_slot[camId] = slot;

    (void)mp;  // logging only in upstream; nothing to do here.
}

void DeviceCache::addCameraParams(
    int camId, int downscale, const mvsUtils::MultiViewParams& mp)
{
    if (!impl_) {
        build(/*maxMipmap=*/8, /*maxCamera=*/32);
    }

    const std::pair<int,int> key{camId, downscale};
    // Already cached? Refresh LRU position by re-inserting into
    // the native cache (no-op apart from LRU bookkeeping).
    auto it = impl_->cam_param_slot.find(key);
    if (it != impl_->cam_param_slot.end()) {
        return;
    }

    // Build the params struct.
    av::depth_map::DeviceCameraParams cp{};
    fill_host_camera_params(cp, camId, downscale, mp);

    // Allocate a fresh adapter-table ID.
    const int id = impl_->next_cam_param_id++;
    impl_->cam_param_slot[key] = id;

    // Push into the native LRU (for capacity accounting) and into
    // the adapter translation table (for kernel access).
    impl_->native->add_camera_params(camId, downscale, cp);
    av::depth_map::upstream_adapter::set_camera_param(id, cp);
}

const DeviceMipmapImage& DeviceCache::requestMipmapImage(
    int camId, const mvsUtils::MultiViewParams& mp)
{
    if (!impl_) impl_->require_built();
    auto it = impl_->mipmap_slot.find(camId);
    if (it == impl_->mipmap_slot.end()) {
        throw std::runtime_error(
            "DeviceCache::requestMipmapImage: camId not cached "
            "(call addMipmapImage first).");
    }
    (void)mp;
    return *impl_->shim_mipmaps[static_cast<std::size_t>(it->second)];
}

const int DeviceCache::requestCameraParamsId(
    int camId, int downscale, const mvsUtils::MultiViewParams& mp)
{
    if (!impl_) impl_->require_built();
    auto it = impl_->cam_param_slot.find({camId, downscale});
    if (it == impl_->cam_param_slot.end()) {
        throw std::runtime_error(
            "DeviceCache::requestCameraParamsId: (camId, downscale) "
            "not cached (call addCameraParams first).");
    }
    (void)mp;
    return it->second;
}

}  // namespace depthMap
}  // namespace aliceVision
