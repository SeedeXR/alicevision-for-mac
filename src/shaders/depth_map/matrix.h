// matrix.h — Metal port of depthMap/cuda/device/matrix.cuh
//
// CUDA → MSL translation notes:
//
//  * dot, cross, normalize, length(=size), distance(=dist)
//    are built-in MSL functions on the vector types. The original
//    file's hand-written versions are dropped; callers use the MSL
//    built-ins directly. Equivalences:
//        size(v)        → length(v)
//        dist(a,b)      → distance(a,b)
//        normalize(v&)  → v = normalize(v)   // MSL returns by value
//
//  * Matrix layout is COLUMN-MAJOR, packed into a flat float array:
//        M3x3[col * 3 + row]
//    This matches Eigen's default storage, so cross-validating
//    against Eigen is direct (no transpose needed when copying
//    in/out of buffers).
//
//  * `__fdividef(1.0f, x)` in CUDA was an explicit fast-reciprocal
//    intrinsic. MSL with `-ffast-math` already turns `1.0f / x`
//    into the fast path, and the conservative `precise::divide`
//    builtin is also available. We use `1.0f / x` throughout
//    because `-ffast-math` is set in cmake/Metal.cmake.
//
//  * `acos`'s double-precision use in `angleBetwABandAC` is
//    demoted to `float` (no FP64 on Apple GPUs). The clamping that
//    the CUDA code did via `isinf(a) ? 0.0 : a` for out-of-domain
//    inputs is preserved.
//
//  * Constants:
//        CUDART_PI_F  → M_PI_F  (or float(M_PI))
//        CUDART_PI    → M_PI    (=double-precision; unavailable in
//                                MSL — use M_PI_F)
//
// The header is intended to be `#include`d into other MSL sources
// that need matrix/geometry helpers. It has no kernel definitions;
// kernels exercising these helpers live in `matrix_kernels.metal`.

#pragma once

#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------
// uchar4 ↔ float4 conversions
// ----------------------------------------------------------------

inline uchar4 float4_to_uchar4(float4 a)
{
    return uchar4(uchar(a.x), uchar(a.y), uchar(a.z), uchar(a.w));
}

inline float4 uchar4_to_float4(uchar4 a)
{
    return float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

// ----------------------------------------------------------------
// Column-major 3x3 / 3x4 matrix-vector / matrix-matrix helpers
//
// "M3x3[k]" maps to entry (k % 3, k / 3) in (row, col) form.
// ----------------------------------------------------------------

// M (3x3) * v (3) → vec3
inline float3 M3x3mulV3(constant const float* M3x3, float3 v)
{
    return float3(M3x3[0] * v.x + M3x3[3] * v.y + M3x3[6] * v.z,
                  M3x3[1] * v.x + M3x3[4] * v.y + M3x3[7] * v.z,
                  M3x3[2] * v.x + M3x3[5] * v.y + M3x3[8] * v.z);
}

inline float3 M3x3mulV3(device const float* M3x3, float3 v)
{
    return float3(M3x3[0] * v.x + M3x3[3] * v.y + M3x3[6] * v.z,
                  M3x3[1] * v.x + M3x3[4] * v.y + M3x3[7] * v.z,
                  M3x3[2] * v.x + M3x3[5] * v.y + M3x3[8] * v.z);
}

inline float3 M3x3mulV3(thread const float* M3x3, float3 v)
{
    return float3(M3x3[0] * v.x + M3x3[3] * v.y + M3x3[6] * v.z,
                  M3x3[1] * v.x + M3x3[4] * v.y + M3x3[7] * v.z,
                  M3x3[2] * v.x + M3x3[5] * v.y + M3x3[8] * v.z);
}

// M (3x3) * (v.x, v.y, 1) — homogeneous 2D pt → 3D
inline float3 M3x3mulV2(constant const float* M3x3, float2 v)
{
    return float3(M3x3[0] * v.x + M3x3[3] * v.y + M3x3[6],
                  M3x3[1] * v.x + M3x3[4] * v.y + M3x3[7],
                  M3x3[2] * v.x + M3x3[5] * v.y + M3x3[8]);
}

inline float3 M3x3mulV2(thread const float* M3x3, float2 v)
{
    return float3(M3x3[0] * v.x + M3x3[3] * v.y + M3x3[6],
                  M3x3[1] * v.x + M3x3[4] * v.y + M3x3[7],
                  M3x3[2] * v.x + M3x3[5] * v.y + M3x3[8]);
}

// M (3x4) * (v.x, v.y, v.z, 1)
inline float3 M3x4mulV3(constant const float* M3x4, float3 v)
{
    return float3(M3x4[0] * v.x + M3x4[3] * v.y + M3x4[6] * v.z + M3x4[9],
                  M3x4[1] * v.x + M3x4[4] * v.y + M3x4[7] * v.z + M3x4[10],
                  M3x4[2] * v.x + M3x4[5] * v.y + M3x4[8] * v.z + M3x4[11]);
}

inline float3 M3x4mulV3(thread const float* M3x4, float3 v)
{
    return float3(M3x4[0] * v.x + M3x4[3] * v.y + M3x4[6] * v.z + M3x4[9],
                  M3x4[1] * v.x + M3x4[4] * v.y + M3x4[7] * v.z + M3x4[10],
                  M3x4[2] * v.x + M3x4[5] * v.y + M3x4[8] * v.z + M3x4[11]);
}

// Projective transform of a 2D point through a 3x3 homography.
inline float2 V2M3x3mulV2(thread const float* M3x3, float2 v)
{
    const float d = M3x3[2] * v.x + M3x3[5] * v.y + M3x3[8];
    return float2((M3x3[0] * v.x + M3x3[3] * v.y + M3x3[6]) / d,
                  (M3x3[1] * v.x + M3x3[4] * v.y + M3x3[7]) / d);
}

// Perspective projection of a 3D point by a 3x4 camera matrix.
inline float2 project3DPoint(constant const float* M3x4, float3 v)
{
    const float3 p = M3x4mulV3(M3x4, v);
    const float  pzInv = 1.0f / p.z;
    return float2(p.x * pzInv, p.y * pzInv);
}

inline float2 project3DPoint(thread const float* M3x4, float3 v)
{
    const float3 p = M3x4mulV3(M3x4, v);
    const float  pzInv = 1.0f / p.z;
    return float2(p.x * pzInv, p.y * pzInv);
}

// O = A * B   (all 3x3 column-major flat)
inline void M3x3mulM3x3(thread float* O, thread const float* A, thread const float* B)
{
    O[0] = A[0] * B[0] + A[3] * B[1] + A[6] * B[2];
    O[3] = A[0] * B[3] + A[3] * B[4] + A[6] * B[5];
    O[6] = A[0] * B[6] + A[3] * B[7] + A[6] * B[8];

    O[1] = A[1] * B[0] + A[4] * B[1] + A[7] * B[2];
    O[4] = A[1] * B[3] + A[4] * B[4] + A[7] * B[5];
    O[7] = A[1] * B[6] + A[4] * B[7] + A[7] * B[8];

    O[2] = A[2] * B[0] + A[5] * B[1] + A[8] * B[2];
    O[5] = A[2] * B[3] + A[5] * B[4] + A[8] * B[5];
    O[8] = A[2] * B[6] + A[5] * B[7] + A[8] * B[8];
}

inline void M3x3minusM3x3(thread float* O, thread const float* A, thread const float* B)
{
    for (int i = 0; i < 9; ++i) O[i] = A[i] - B[i];
}

inline void M3x3transpose(thread float* O, thread const float* A)
{
    O[0] = A[0];  O[1] = A[3];  O[2] = A[6];
    O[3] = A[1];  O[4] = A[4];  O[5] = A[7];
    O[6] = A[2];  O[7] = A[5];  O[8] = A[8];
}

// Outer product a ⊗ b → 3x3 column-major.
inline void outerMultiply(thread float* O, float3 a, float3 b)
{
    O[0] = a.x * b.x;
    O[3] = a.x * b.y;
    O[6] = a.x * b.z;
    O[1] = a.y * b.x;
    O[4] = a.y * b.y;
    O[7] = a.y * b.z;
    O[2] = a.z * b.x;
    O[5] = a.z * b.y;
    O[8] = a.z * b.z;
}

// ----------------------------------------------------------------
// Geometric primitives
// ----------------------------------------------------------------

inline float3 linePlaneIntersect(float3 linePoint,
                                 float3 lineVect,
                                 float3 planePoint,
                                 float3 planeNormal)
{
    const float k = (dot(planePoint, planeNormal) - dot(planeNormal, linePoint))
                  /  dot(planeNormal, lineVect);
    return linePoint + lineVect * k;
}

inline float3 closestPointOnPlaneToPoint(float3 point,
                                         float3 planePoint,
                                         float3 planeNormalNormalized)
{
    return point - planeNormalNormalized * dot(planeNormalNormalized, point - planePoint);
}

inline float3 closestPointToLine3D(float3 point,
                                   float3 linePoint,
                                   float3 lineVectNormalized)
{
    return linePoint + lineVectNormalized * dot(lineVectNormalized, point - linePoint);
}

inline float pointLineDistance3D(float3 point,
                                 float3 linePoint,
                                 float3 lineVectNormalized)
{
    return length(cross(lineVectNormalized, linePoint - point));
}

// Angle (degrees) between two NON-normalized vectors.
// CUDA original normalized in-place by reference; here we take by
// value and normalize locally — same semantics for the result.
inline float angleBetwV1andV2(float3 iV1, float3 iV2)
{
    float3 V1 = normalize(iV1);
    float3 V2 = normalize(iV2);
    return fabs(acos(dot(V1, V2)) / (M_PI_F / 180.0f));
}

// Angle BAC (degrees) at vertex A.
// The CUDA original used `double` for the acos; here we stay in
// float (FP64 unavailable on Apple GPUs) and preserve the
// isinf-guard.
inline float angleBetwABandAC(float3 A, float3 B, float3 C)
{
    float3 V1 = normalize(B - A);
    float3 V2 = normalize(C - A);
    const float x = clamp(dot(V1, V2), -1.0f, 1.0f);  // hardened domain
    float a = acos(x);
    if (isinf(a)) a = 0.0f;
    return fabs(a) / (M_PI_F / 180.0f);
}

// Shortest line segment between two 3D lines p1p2 and p3p4.
// Outputs:
//   k, l  : line parameters for the two foot-of-perpendicular pts
//   lli1  : foot of perpendicular on line 1
//   lli2  : foot of perpendicular on line 2
// Return: midpoint of the foot-of-perpendiculars (the "average"
// intersection used by upstream's triangulation).
inline float3 lineLineIntersect(thread float* k,
                                thread float* l,
                                thread float3* lli1,
                                thread float3* lli2,
                                float3 p1, float3 p2,
                                float3 p3, float3 p4)
{
    const float3 p13 = p1 - p3;
    const float3 p43 = p4 - p3;
    const float3 p21 = p2 - p1;

    const float d1343 = dot(p13, p43);
    const float d4321 = dot(p43, p21);
    const float d1321 = dot(p13, p21);
    const float d4343 = dot(p43, p43);
    const float d2121 = dot(p21, p21);

    const float denom = d2121 * d4343 - d4321 * d4321;
    const float numer = d1343 * d4321 - d1321 * d4343;

    const float mua = numer / denom;
    const float mub = (d1343 + d4321 * mua) / d4343;

    const float3 pa = p1 + p21 * mua;
    const float3 pb = p3 + p43 * mub;

    *k    = mua;
    *l    = mub;
    *lli1 = pa;
    *lli2 = pb;
    return (pa + pb) * 0.5f;
}

// ----------------------------------------------------------------
// Sigmoid filters (unchanged from CUDA)
// ----------------------------------------------------------------

inline float sigmoid(float zeroVal, float endVal,
                     float sigwidth, float sigMid, float xval)
{
    return zeroVal + (endVal - zeroVal)
                   * (1.0f / (1.0f + exp(10.0f * ((xval - sigMid) / sigwidth))));
}

inline float sigmoid2(float zeroVal, float endVal,
                      float sigwidth, float sigMid, float xval)
{
    return zeroVal + (endVal - zeroVal)
                   * (1.0f / (1.0f + exp(10.0f * ((sigMid - xval) / sigwidth))));
}
