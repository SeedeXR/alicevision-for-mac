#pragma once

// GaussianFilter — host driver for upstream
// imageProcessing/deviceGaussianFilter kernels:
//
//   downscale_with_gaussian_blur(in_tex, out_buf, downscale, radius)
//       reads an RGBA float texture, applies a non-separable 2D
//       Gaussian, and writes the downscaled RGBA result.
//
//   median_filter_3(in_tex, out_buf)
//       reads a single-channel R32Float texture, computes a 7×7
//       median per pixel (radius hardcoded to 3), writes to a flat
//       float buffer. Border pixels are left untouched (upstream
//       contract).
//
//   gaussian_blur_volume_z(inout_volume, dims, gauss_radius)
//       in-place 1D Gaussian along the Z axis of a packed float
//       volume (cost-volume smoothing for SGM).
//
//   gaussian_blur_volume_xyz(inout_volume, dims, gauss_radius)
//       in-place 3D Gaussian over a packed float volume.

#include "av/depth_map/Volume.hpp"   // for VolumeDims

#include <cstddef>
#include <cstdint>
#include <memory>

namespace av::gpu {
    class Device;
    class Buffer;
    class Texture;
}

namespace av::depth_map {

class GaussianTable;

class GaussianFilter {
public:
    explicit GaussianFilter(av::gpu::Device& dev, GaussianTable& table);

    GaussianFilter(const GaussianFilter&)            = delete;
    GaussianFilter& operator=(const GaussianFilter&) = delete;
    GaussianFilter(GaussianFilter&&) noexcept;
    GaussianFilter& operator=(GaussianFilter&&) noexcept;
    ~GaussianFilter();

    // Downscale an RGBA texture by `downscale` using a
    // (2·gaussRadius+1)² Gaussian. The output buffer must hold
    // `(in_width / downscale) * (in_height / downscale)` float4s.
    void downscale_with_gaussian_blur(const av::gpu::Texture& in_tex,
                                      av::gpu::Buffer&        out_buf,
                                      std::uint32_t           downscaled_w,
                                      std::uint32_t           downscaled_h,
                                      std::int32_t            downscale,
                                      std::int32_t            gauss_radius);

    // 7×7 median filter (radius=3) on a single-channel R32Float
    // texture. The output buffer holds `width × height` floats;
    // border pixels (within `radius` of any edge) are not written.
    void median_filter_3(const av::gpu::Texture& in_tex,
                         av::gpu::Buffer&        out_buf,
                         std::uint32_t           width,
                         std::uint32_t           height);

    // In-place 1D Gaussian blur along the Z axis of a packed float
    // cost volume. `inout_volume` must hold `dims.voxel_count()`
    // float elements. Matches upstream's `cuda_gaussianBlurVolumeZ`
    // (deviceGaussianFilter.cu:289), including the upstream quirk
    // that the boundary test `(iz < volDimZ) && (iz > 0)` excludes
    // the iz==0 plane from the convolution support.
    void gaussian_blur_volume_z(av::gpu::Buffer& inout_volume,
                                VolumeDims       dims,
                                std::int32_t     gauss_radius);

    // In-place 3D Gaussian blur over a packed float cost volume.
    // Matches upstream's `cuda_gaussianBlurVolumeXYZ`
    // (deviceGaussianFilter.cu:314). Border voxels use a clipped
    // kernel (sum is renormalized).
    void gaussian_blur_volume_xyz(av::gpu::Buffer& inout_volume,
                                  VolumeDims       dims,
                                  std::int32_t     gauss_radius);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
