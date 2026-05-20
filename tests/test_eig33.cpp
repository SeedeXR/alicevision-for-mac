// test_eig33.cpp — numerical validation of the Metal port of
// depthMap/cuda/device/eig33.cuh against Eigen's CPU reference.
//
// Strategy:
//   1. Generate N random symmetric 3×3 matrices with controlled
//      conditioning (random orthogonal eigenvectors + sampled
//      eigenvalues).
//   2. Run our Metal eig33_decompose on all N in one dispatch.
//   3. Run Eigen::SelfAdjointEigenSolver on the same N (FP64).
//   4. Compare:
//        - eigenvalues: max relative error < 1e-4
//        - eigenvectors: |cos(θ)| > 1 - 1e-3 for the corresponding
//          pair (allowing sign flip; eigenvectors are defined up to
//          sign).
//
// Tolerances reflect the FP32-vs-FP64 precision change in the port;
// see eig33.metal for the precision-impact discussion. We expect
// well-conditioned matrices to pass these easily.
//
// Exit codes:
//    0 all matrices pass tolerances
//    1 any matrix fails
//    2 setup error

#include "av/depth_map/Eig33.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kNumMatrices    = 4096;
constexpr float       kEigValueTolRel = 1e-4f;
constexpr float       kEigVecCosTol   = 1e-3f;

// Build N symmetric 3×3 matrices: Q diag(d) Q^T with random Q.
// Eigenvalues drawn from a moderate range to keep conditioning sane.
void make_symmetric_matrices(std::size_t n,
                             std::vector<float>& out_row_major,
                             std::vector<Eigen::Matrix3d>& out_eigen,
                             std::uint64_t seed)
{
    out_row_major.resize(n * 9);
    out_eigen.resize(n);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> eigval_dist(0.5, 5.0);
    std::normal_distribution<double>       normal(0.0, 1.0);

    for (std::size_t k = 0; k < n; ++k) {
        // Random orthogonal Q via QR of a Gaussian matrix.
        Eigen::Matrix3d M;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                M(i, j) = normal(rng);
        Eigen::HouseholderQR<Eigen::Matrix3d> qr(M);
        Eigen::Matrix3d Q = qr.householderQ();

        // Diagonal of eigenvalues.
        Eigen::Vector3d d;
        d << eigval_dist(rng), eigval_dist(rng), eigval_dist(rng);

        Eigen::Matrix3d A = Q * d.asDiagonal() * Q.transpose();
        // Symmetrize to suppress numerical drift.
        A = 0.5 * (A + A.transpose());

        out_eigen[k] = A;
        // Pack row-major into the float buffer (precision loss
        // is part of what we're measuring).
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                out_row_major[k * 9 + i * 3 + j] = static_cast<float>(A(i, j));
    }
}

struct EigVecMatch {
    int   index;       // column in V_ref corresponding to gpu row r
    float abs_cos;     // |cos(angle)| of the matched pair
};

// Find which Eigen eigenvector best matches the GPU row r.
// Both eigenvalue lists are ascending after sorting; we still match
// by cosine in case of degenerate eigenvalues (where row vs column
// assignment is ambiguous up to a basis rotation in the eigenspace).
EigVecMatch best_match(const Eigen::Vector3f& gpu_vec,
                       const Eigen::Matrix3f& ref_vectors)
{
    EigVecMatch best{ -1, -1.0f };
    for (int j = 0; j < 3; ++j) {
        float c = std::abs(gpu_vec.dot(ref_vectors.col(j)));
        if (c > best.abs_cos) {
            best.abs_cos = c;
            best.index   = j;
        }
    }
    return best;
}

}  // namespace

int main() try
{
    // -------- setup --------
    auto dev = av::gpu::Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    av::depth_map::Eig33 solver(dev);

    // -------- inputs --------
    std::vector<float>           matrices_in;
    std::vector<Eigen::Matrix3d> matrices_ref;
    make_symmetric_matrices(kNumMatrices, matrices_in, matrices_ref, 0xc0ffee);

    std::vector<float> values_gpu (kNumMatrices * 3);
    std::vector<float> vectors_gpu(kNumMatrices * 9);

    // -------- GPU dispatch --------
    solver.decompose(matrices_in, values_gpu, vectors_gpu);

    // -------- compare against Eigen FP64 reference --------
    int    bad_eigval = 0;
    int    bad_eigvec = 0;
    double worst_eigval_rel = 0.0;
    double worst_eigvec_dot = 1.0;  // we look for max(1 - |cos|)

    for (std::size_t k = 0; k < kNumMatrices; ++k) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(matrices_ref[k]);
        Eigen::Vector3d d_ref = es.eigenvalues();   // ascending
        Eigen::Matrix3d V_ref = es.eigenvectors();  // columns

        // Convert reference to float for comparison.
        Eigen::Vector3f d_ref_f = d_ref.cast<float>();
        Eigen::Matrix3f V_ref_f = V_ref.cast<float>();

        // GPU values (already ascending).
        Eigen::Vector3f d_gpu;
        d_gpu << values_gpu[k * 3 + 0],
                 values_gpu[k * 3 + 1],
                 values_gpu[k * 3 + 2];

        // GPU vectors are written row-major; row i is the i'th
        // eigenvector returned by the kernel. The kernel's V[i][j]
        // is element (row i, col j) of an orthogonal matrix whose
        // *columns* are eigenvectors — so to fetch the j'th
        // eigenvector we read column j across rows. Our row-major
        // write means vectors_gpu[i*3 + j] = V[i][j], so the j'th
        // eigenvector is { vectors_gpu[0*3+j], vectors_gpu[1*3+j],
        // vectors_gpu[2*3+j] }.

        // Compare each eigenvalue:
        for (int i = 0; i < 3; ++i) {
            const float scale = std::max(std::abs(d_ref_f(i)), 1e-6f);
            const float rel   = std::abs(d_gpu(i) - d_ref_f(i)) / scale;
            if (rel > kEigValueTolRel) {
                if (bad_eigval < 4) {
                    std::fprintf(stderr,
                        "k=%zu i=%d eigval mismatch: gpu=%g ref=%g rel=%g\n",
                        k, i,
                        static_cast<double>(d_gpu(i)),
                        static_cast<double>(d_ref_f(i)),
                        static_cast<double>(rel));
                }
                ++bad_eigval;
            }
            worst_eigval_rel = std::max(worst_eigval_rel, static_cast<double>(rel));
        }

        // Compare each eigenvector — match GPU column j (=eigenvec j)
        // to the closest reference eigenvector (cosine-wise).
        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3f v_gpu;
            v_gpu << vectors_gpu[k * 9 + 0 * 3 + j],
                     vectors_gpu[k * 9 + 1 * 3 + j],
                     vectors_gpu[k * 9 + 2 * 3 + j];

            // The GPU output is not strictly unit-norm at FP32; we
            // normalize before angle comparison, matching what the
            // downstream cuda_stat3d consumer does.
            const float n = v_gpu.norm();
            if (n > 0.0f) v_gpu /= n;

            EigVecMatch m = best_match(v_gpu, V_ref_f);
            const float gap = 1.0f - m.abs_cos;
            if (gap > kEigVecCosTol) {
                if (bad_eigvec < 4) {
                    std::fprintf(stderr,
                        "k=%zu j=%d eigvec mismatch: best |cos|=%g (matched col %d)\n",
                        k, j,
                        static_cast<double>(m.abs_cos), m.index);
                }
                ++bad_eigvec;
            }
            worst_eigvec_dot = std::min(worst_eigvec_dot,
                                        static_cast<double>(m.abs_cos));
        }
    }

    std::printf("[info] matrices         : %zu\n", kNumMatrices);
    std::printf("[info] worst eigval rel : %.3g (budget %.3g)\n",
                worst_eigval_rel, static_cast<double>(kEigValueTolRel));
    std::printf("[info] worst |cos|      : %.6f (budget %.6f)\n",
                worst_eigvec_dot, 1.0 - static_cast<double>(kEigVecCosTol));

    if (bad_eigval || bad_eigvec) {
        std::fprintf(stderr,
            "FAIL: %d eigval mismatches, %d eigvec mismatches\n",
            bad_eigval, bad_eigvec);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
catch (const av::gpu::GpuError& e) {
    std::fprintf(stderr, "GpuError: %s\n", e.what());
    return 2;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "exception: %s\n", e.what());
    return 2;
}
