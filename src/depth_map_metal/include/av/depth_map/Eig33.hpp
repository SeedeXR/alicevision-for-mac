#pragma once

// Eig33 — Metal port of depthMap/cuda/device/eig33.cuh's
// symmetric 3×3 eigendecomposition. One matrix per thread; the
// host owns a Metal pipeline and dispatches in batches.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace av::gpu { class Device; }

namespace av::depth_map {

class Eig33 {
public:
    explicit Eig33(av::gpu::Device& dev);

    Eig33(const Eig33&)            = delete;
    Eig33& operator=(const Eig33&) = delete;
    Eig33(Eig33&&) noexcept;
    Eig33& operator=(Eig33&&) noexcept;
    ~Eig33();

    // Decompose `count` symmetric 3×3 matrices.
    //
    // matrices_in : row-major flattened, length = count * 9 floats.
    //               Only the upper triangle is consulted; the
    //               kernel assumes the lower mirrors it.
    // values_out  : eigenvalues ascending,   length = count * 3.
    // vectors_out : eigenvectors as rows in a row-major 3×3,
    //                                       length = count * 9.
    //
    // Blocking; commits and waits. For long-batch async use we'll
    // add a non-blocking overload when a real use case appears.
    void decompose(std::span<const float> matrices_in,
                   std::span<float>        values_out,
                   std::span<float>        vectors_out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
