// eig33.metal — kernel entry point only.
//
// All helpers live in `eig33.h` so other MSL TUs (e.g., the PCA
// plane fit in `depth_sim_map.metal`) can call `eig33_decompose`
// directly.
//
// Precision: FP32 only (Apple Silicon GPUs have no native FP64).
// See `eig33.h` and the S3 / S22 handover entries for impact
// analysis.

#include "eig33.h"

kernel void av_eig33_decompose(
    device const float* matrices_in   [[buffer(0)]],   // [count * 9]
    device       float* values_out    [[buffer(1)]],   // [count * 3]
    device       float* vectors_out   [[buffer(2)]],   // [count * 9]
    constant     uint&  count         [[buffer(3)]],
    uint                gid           [[thread_position_in_grid]])
{
    if (gid >= count) return;

    const device float* in    = matrices_in + gid * 9;
    device       float* val   = values_out  + gid * 3;
    device       float* vec   = vectors_out + gid * 9;

    float A[3][3];
    A[0][0] = in[0];  A[0][1] = in[1];  A[0][2] = in[2];
    A[1][0] = in[3];  A[1][1] = in[4];  A[1][2] = in[5];
    A[2][0] = in[6];  A[2][1] = in[7];  A[2][2] = in[8];

    float V[3][3], d[3];
    eig33_decompose(A, V, d);

    val[0] = d[0];  val[1] = d[1];  val[2] = d[2];
    vec[0] = V[0][0];  vec[1] = V[0][1];  vec[2] = V[0][2];
    vec[3] = V[1][0];  vec[4] = V[1][1];  vec[5] = V[1][2];
    vec[6] = V[2][0];  vec[7] = V[2][1];  vec[8] = V[2][2];
}
