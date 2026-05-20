#pragma once

// MatrixOps — host driver for the matrix-helper validation kernel.
//
// Not a "useful kernel" in the depthMap pipeline — its sole job is
// to dispatch matrix.h helpers under controlled inputs so we can
// verify their numerics against an Eigen FP64 reference.

#include <cstddef>
#include <memory>
#include <span>

namespace av::gpu { class Device; }

namespace av::depth_map {

class MatrixOps {
public:
    // Inputs / outputs per validation case.
    static constexpr std::size_t kInPerCase  = 48;
    static constexpr std::size_t kOutPerCase = 35;

    explicit MatrixOps(av::gpu::Device& dev);

    MatrixOps(const MatrixOps&)            = delete;
    MatrixOps& operator=(const MatrixOps&) = delete;
    MatrixOps(MatrixOps&&) noexcept;
    MatrixOps& operator=(MatrixOps&&) noexcept;
    ~MatrixOps();

    // Run `count = inputs.size() / kInPerCase` test cases.
    // outputs must have size = count * kOutPerCase.
    void validate(std::span<const float> inputs,
                  std::span<float>        outputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
