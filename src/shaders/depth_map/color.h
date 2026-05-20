// color.h — Metal port of depthMap/cuda/device/color.cuh
//
// CUDA → MSL translation map:
//
//   norm3df(x,y,z)   → length(float3(x,y,z))   (or sqrt(x²+y²+z²))
//   __powf(x, e)     → pow(x, e)               (-ffast-math approximates CUDA intrinsic)
//   __expf(x)        → exp(x)                  (-ffast-math)
//   __fsqrt_rn(x)    → sqrt(x)
//   __fdividef(a,b)  → a / b                   (-ffast-math)
//   cbrtf(x)         → pow(x, 1.0/3.0)         (MSL has no cbrt; the
//                                                input domain in xyz2lab
//                                                is always non-negative
//                                                so a copysign wrap is
//                                                not needed.)
//
// All conversions are float-precision (Apple GPUs have no FP64).
// Drift vs an FP64 CPU reference is bounded by FP32 ULP across the
// algorithm length; expected ≤ 1e-4 absolute for chained operations
// like sRGB → XYZ → Lab.

#pragma once

#include <metal_stdlib>
using namespace metal;

namespace av_depthmap {

// ----------------------------------------------------------------
// Alpha thresholds — match upstream macros.
// ----------------------------------------------------------------

constant constexpr float kRcMinAlpha = 255.0f * 0.9f;
constant constexpr float kTcMinAlpha = 255.0f * 0.4f;

// ----------------------------------------------------------------
// Euclidean distance helpers.
// ----------------------------------------------------------------

inline float euclideanDist3(float3 x1, float3 x2)
{
    return length(x1 - x2);
}

inline float euclideanDist3(float4 x1, float4 x2)
{
    return length(x1.xyz - x2.xyz);
}

// ----------------------------------------------------------------
// sRGB (0..1) → linear RGB (0..1).
// Per-channel piecewise sRGB EOTF.
// ----------------------------------------------------------------

inline float3 srgb2rgb(float3 c)
{
    return float3(
        c.x <= 0.04045f ? c.x / 12.92f : pow((c.x + 0.055f) / 1.055f, 2.4f),
        c.y <= 0.04045f ? c.y / 12.92f : pow((c.y + 0.055f) / 1.055f, 2.4f),
        c.z <= 0.04045f ? c.z / 12.92f : pow((c.z + 0.055f) / 1.055f, 2.4f));
}

// ----------------------------------------------------------------
// Linear RGB (0..1) → XYZ (0..1) using sRGB primaries (D65).
// ----------------------------------------------------------------

inline float3 rgb2xyz(float3 c)
{
    return float3(0.4124564f * c.x + 0.3575761f * c.y + 0.1804375f * c.z,
                  0.2126729f * c.x + 0.7151522f * c.y + 0.0721750f * c.z,
                  0.0193339f * c.x + 0.1191920f * c.y + 0.9503041f * c.z);
}

// ----------------------------------------------------------------
// Linear RGB (0..1) → HSL (0..1).
// ----------------------------------------------------------------

inline float3 rgb2hsl(float3 c)
{
    const float cmin = min(c.x, min(c.y, c.z));
    const float cmax = max(c.x, max(c.y, c.z));

    float h = 0.0f;
    if (cmin == cmax) {
        // h = 0
    } else if (cmax == c.x) {
        h = ((c.y - c.z) / (cmax - cmin) + 6.0f) / 6.0f;
        if (h >= 1.0f) h -= 1.0f;
    } else if (cmax == c.y) {
        h = ((c.z - c.x) / (cmax - cmin) + 2.0f) / 6.0f;
    } else {
        h = ((c.x - c.y) / (cmax - cmin) + 4.0f) / 6.0f;
    }

    const float l = 0.5f * (cmin + cmax);

    float s = 0.0f;
    if (cmin == cmax) {
        // s = 0
    } else if (l <= 0.5f) {
        s = (cmax - cmin) / (2.0f * l);
    } else {
        s = (cmax - cmin) / (2.0f - 2.0f * l);
    }

    return float3(h, s, l);
}

// ----------------------------------------------------------------
// XYZ (0..1) → CIELAB, then scaled by 2.55 so values fit roughly in
// [0..255]. (Note: a* and b* can still go negative or > 255 for
// saturated colors; upstream comments call this out as a TODO to
// move to float textures.)
// ----------------------------------------------------------------

inline float3 xyz2lab(float3 c)
{
    // D65 white point: (0.95047, 1.0, 1.08883).
    float3 r = float3(c.x / 0.95047f, c.y, c.z / 1.08883f);

    constexpr float kappa  = 24389.0f / 27.0f;   // ≈ 903.296
    constexpr float thresh = 216.0f / 24389.0f;  // ≈ 0.008856

    constexpr float kThird = 1.0f / 3.0f;
    float3 f;
    f.x = (r.x > thresh) ? pow(r.x, kThird) : (kappa * r.x + 16.0f) / 116.0f;
    f.y = (r.y > thresh) ? pow(r.y, kThird) : (kappa * r.y + 16.0f) / 116.0f;
    f.z = (r.z > thresh) ? pow(r.z, kThird) : (kappa * r.z + 16.0f) / 116.0f;

    float3 out = float3(116.0f * f.y - 16.0f,
                        500.0f * (f.x - f.y),
                        200.0f * (f.y - f.z));

    return out * 2.55f;
}

// ----------------------------------------------------------------
// RGB (uchar4) → gray (float) using the BT.601 / NTSC luma weights.
// ----------------------------------------------------------------

inline float rgb2gray(uchar4 c)
{
    return 0.2989f * float(c.x) + 0.5870f * float(c.y) + 0.1140f * float(c.z);
}

// ----------------------------------------------------------------
// Adaptive support-weight (Yoon & Kweon) — 6-arg variant.
// Weight combines color similarity (Lab Euclidean distance) and
// spatial proximity to the patch center.
//
// Returns: exp(-(deltaC*invGammaC + deltaP*invGammaP))
// where deltaC = ||c1.xyz - c2.xyz||,
//       deltaP = sqrt(dx² + dy²).
// ----------------------------------------------------------------

inline float CostYKfromLab(int dx, int dy,
                           float4 c1, float4 c2,
                           float invGammaC, float invGammaP)
{
    float deltaC = euclideanDist3(c1, c2);
    deltaC *= invGammaC;

    float deltaP = sqrt(float(dx * dx + dy * dy));
    deltaP *= invGammaP;

    return exp(-(deltaC + deltaP));
}

// 3-arg variant: only the color term (used in patch-pattern paths
// where dx/dy aren't natural).
inline float CostYKfromLab(float4 c1, float4 c2, float invGammaC)
{
    const float deltaC = euclideanDist3(c1, c2);
    return exp(-(deltaC * invGammaC));
}

}  // namespace av_depthmap
