#pragma once

// GaussianTable — host-built table of Gaussian weights matching
// upstream's `cuda_createConstantGaussianArray`. Owns two GPU
// buffers:
//   * weights[]  — flat Gaussian samples for scales 0..MaxScales-1
//   * offsets[]  — start index of each scale inside `weights`
//
// Scale `s` has radius r = s+1 and size 2r+1, so the layout is:
//   scale 0 (r=1):  3 floats  at offset 0
//   scale 1 (r=2):  5 floats  at offset 3
//   scale 2 (r=3):  7 floats  at offset 8
//   ...
//   scale 9 (r=10): 21 floats at offset 99
// Total: 120 floats. (Upstream caps at MAX_CONSTANT_GAUSS_MEM_SIZE=128.)
//
// The kernel-side helper `av_getGauss(weights, offsets, scale, idx)`
// reads from the same layout.

#include <cstddef>
#include <cstdint>
#include <memory>

namespace av::gpu { class Device; class Buffer; }

namespace av::depth_map {

class GaussianTable {
public:
    static constexpr int kMaxScales  = 10;
    static constexpr int kMaxMemSize = 128;

    // Build and upload the table.
    explicit GaussianTable(av::gpu::Device& dev,
                           int scales = kMaxScales);

    GaussianTable(const GaussianTable&)            = delete;
    GaussianTable& operator=(const GaussianTable&) = delete;
    GaussianTable(GaussianTable&&) noexcept;
    GaussianTable& operator=(GaussianTable&&) noexcept;
    ~GaussianTable();

    av::gpu::Buffer& weights() noexcept;
    av::gpu::Buffer& offsets() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
