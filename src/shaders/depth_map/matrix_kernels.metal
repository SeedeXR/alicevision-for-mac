// matrix_kernels.metal — validation kernel for matrix.h helpers.
//
// The kernel exercises the most load-bearing helpers from matrix.h
// — matrix-vector / matrix-matrix multiplies, perspective projection,
// outer product, and the 3D line-line nearest-points routine — on
// a flat input span. One test case per thread.
//
// Inputs per case (48 floats, packed in this order):
//      A        9 floats     column-major 3x3
//      B        9 floats     column-major 3x3
//      P        12 floats    column-major 3x4 camera matrix
//      v3       3 floats     test vector
//      u3       3 floats     test vector (for outer product)
//      p1..p4   12 floats    four 3D points for lineLineIntersect
//
// Outputs per case (35 floats):
//      Av       3 floats     A * v3
//      AB       9 floats     A * B
//      Pv       3 floats     P * (v3, 1) — affine, before perspective
//      proj     2 floats     project3DPoint(P, v3)
//      outer    9 floats     v3 ⊗ u3
//      mid      3 floats     midpoint from lineLineIntersect
//      lli1     3 floats     foot of perpendicular on line p1-p2
//      lli2     3 floats     foot of perpendicular on line p3-p4
//
// (lineLineIntersect's `k`, `l` scalars are intentionally not in
// the output — they're a side product validated indirectly via the
// foot-of-perpendicular positions.)

#include <metal_stdlib>
#include "matrix.h"
using namespace metal;

constant constexpr uint kInPerCase  = 48;
constant constexpr uint kOutPerCase = 35;

kernel void av_matrix_validate(
    device const float* in_buf  [[buffer(0)]],   // [count * kInPerCase]
    device       float* out_buf [[buffer(1)]],   // [count * kOutPerCase]
    constant     uint&  count   [[buffer(2)]],
    uint                gid     [[thread_position_in_grid]])
{
    if (gid >= count) return;

    const device float* in  = in_buf  + gid * kInPerCase;
    device       float* out = out_buf + gid * kOutPerCase;

    // Pull inputs into per-thread storage (matrix.h overloads accept
    // `thread const float*`).
    thread float A[9];
    thread float B[9];
    thread float P[12];
    for (int i = 0; i < 9;  ++i) A[i] = in[i];
    for (int i = 0; i < 9;  ++i) B[i] = in[9 + i];
    for (int i = 0; i < 12; ++i) P[i] = in[18 + i];

    const float3 v3 = float3(in[30], in[31], in[32]);
    const float3 u3 = float3(in[33], in[34], in[35]);
    const float3 p1 = float3(in[36], in[37], in[38]);
    const float3 p2 = float3(in[39], in[40], in[41]);
    const float3 p3 = float3(in[42], in[43], in[44]);
    const float3 p4 = float3(in[45], in[46], in[47]);

    // 1. M3x3 * v
    const float3 Av = M3x3mulV3(A, v3);
    out[0] = Av.x; out[1] = Av.y; out[2] = Av.z;

    // 2. M3x3 * M3x3
    thread float AB[9];
    M3x3mulM3x3(AB, A, B);
    for (int i = 0; i < 9; ++i) out[3 + i] = AB[i];

    // 3. M3x4 * (v, 1) — affine result before perspective divide.
    const float3 Pv = M3x4mulV3(P, v3);
    out[12] = Pv.x; out[13] = Pv.y; out[14] = Pv.z;

    // 4. project3DPoint (perspective divide).
    const float2 proj = project3DPoint(P, v3);
    out[15] = proj.x; out[16] = proj.y;

    // 5. Outer product v3 ⊗ u3.
    thread float outer[9];
    outerMultiply(outer, v3, u3);
    for (int i = 0; i < 9; ++i) out[17 + i] = outer[i];

    // 6. lineLineIntersect — foot points and midpoint.
    float  k, l;
    float3 lli1, lli2;
    const float3 mid = lineLineIntersect(&k, &l, &lli1, &lli2,
                                         p1, p2, p3, p4);
    out[26] = mid.x;  out[27] = mid.y;  out[28] = mid.z;
    out[29] = lli1.x; out[30] = lli1.y; out[31] = lli1.z;
    out[32] = lli2.x; out[33] = lli2.y; out[34] = lli2.z;
}
