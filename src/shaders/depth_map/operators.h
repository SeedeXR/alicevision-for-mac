// operators.h — Metal port note for depthMap/cuda/device/operators.cuh
//
// The CUDA original (~80 LOC) provided component-wise arithmetic
// operators on `float2`, `float3`, `float4`, and `int2`. CUDA's
// vector types only carry the `.x .y .z .w` members and do not
// overload arithmetic operators — every depthMap kernel that wanted
// `a + b` on a `float3` had to include this header.
//
// In MSL these operators are *built into the language* for the
// equivalent vector types (`float2`, `float3`, `float4`, `int2`,
// and friends). All thirteen overloads from operators.cuh exist
// natively — including scalar-on-left forms like `float * float4`.
//
// Therefore this header has no operator definitions. It exists so
// the depthMap MSL sources can include it where the CUDA originals
// included <cuda_runtime.h> and the depthMap operators header,
// matching the include topology of the upstream tree for diff
// readability.
//
// Reference for the MSL vector arithmetic surface:
//   Metal Shading Language Specification, "Operators" section
//   (Section 6.3 in the v3.x spec). Component-wise +, -, *, /,
//   unary -, ++ / -- on integer vectors, and mixed scalar/vector
//   forms are all defined.

#pragma once

// Intentionally empty. See file comment.
