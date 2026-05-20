// SimStat.h — Metal port of depthMap/cuda/device/SimStat.cuh
//
// Weighted moments accumulator for normalized cross-correlation
// (NCC) over a patch. Each thread holds its own `simStat`,
// calls `update(...)` once per sample, then `computeWSim()` to
// produce a similarity in (-1, 0) (lower = better match), or 1.0f
// if the variance is non-finite.
//
// Translation notes from CUDA:
//   * `__fdividef(a, b)` → `a / b` (`-ffast-math` engages the
//     hardware fast-divide).
//   * `__device__` qualifiers drop; everything is `inline` /
//     member methods on the struct in MSL.
//   * The CUDA original mixed `__device__ float2&` and bare
//     `float, float` overloads of `update`. We keep both shapes;
//     callers select on signature.

#pragma once

#include <metal_stdlib>
using namespace metal;

namespace av_depthmap {

struct simStat {
    float xsum;
    float ysum;
    float xxsum;
    float yysum;
    float xysum;
    float count;
    float wsum;

    // Default-construct to zeros (MSL doesn't run member-init in all
    // contexts; explicit init via `init_zero()` keeps callers in
    // control).
    void init_zero() thread {
        xsum = ysum = xxsum = yysum = xysum = count = wsum = 0.0f;
    }

    // -------- update overloads --------

    void update(float2 g) thread {
        count += 1.0f;
        xsum  += g.x;
        ysum  += g.y;
        xxsum += g.x * g.x;
        yysum += g.y * g.y;
        xysum += g.x * g.y;
    }

    void update(float2 g, float w) thread {
        wsum  += w;
        count += 1.0f;
        xsum  += w * g.x;
        ysum  += w * g.y;
        xxsum += w * g.x * g.x;
        yysum += w * g.y * g.y;
        xysum += w * g.x * g.y;
    }

    void update(float gx, float gy) thread {
        count += 1.0f;
        xsum  += gx;
        ysum  += gy;
        xxsum += gx * gx;
        yysum += gy * gy;
        xysum += gx * gy;
    }

    void update(float gx, float gy, float w) thread {
        wsum  += w;
        count += 1.0f;
        xsum  += w * gx;
        ysum  += w * gy;
        xxsum += w * gx * gx;
        yysum += w * gy * gy;
        xysum += w * gx * gy;
    }

    void update(float3 c1, float3 c2) thread {
        update(float2(c1.x, c2.x));
        update(float2(c1.y, c2.y));
        update(float2(c1.z, c2.z));
    }

    // -------- variance / covariance accessors --------

    float getVarianceX() const thread {
        return (xxsum / count - (xsum * xsum) / (count * count));
    }
    float getVarianceY() const thread {
        return (yysum / count - (ysum * ysum) / (count * count));
    }
    float getVarianceXY() const thread {
        return (xysum / count - (xsum * ysum) / (count * count));
    }

    // Weighted variants — Welford-style with wsum as the normalizer.
    float getVarianceXW() const thread {
        return (xxsum - xsum * xsum / wsum) / wsum;
    }
    float getVarianceYW() const thread {
        return (yysum - ysum * ysum / wsum) / wsum;
    }
    float getVarianceXYW() const thread {
        return (xysum - xsum * ysum / wsum) / wsum;
    }

    // -------- similarity --------

    // Unweighted-NCC variant used by upstream's compNCCbyH path
    // (currently commented-out in the CUDA source). Preserved for
    // future port parity.
    float computeSimSub(thread simStat& ss) const thread {
        float avx = ss.xsum / ss.count;
        float avy = ss.ysum / ss.count;
        float dx  = xsum - 2.0f * avx * xsum + count * avx;
        float dy  = ysum - 2.0f * avy * ysum + count * avy;
        float d   = dx * dy;
        float u   = xysum - avx * ysum - avy * xsum + count * avx * avy;

        float sim = 1.0f;
        if (fabs(d) > 0.0f) {
            sim = u / sqrt(d);
            sim = 0.0f - sim;
        }
        return sim;
    }

    // Weighted NCC. Returns:
    //   - sim ∈ (-1, 0) for valid correlations (negative = best
    //     match; matches upstream's "-rawSim" convention).
    //   - 1.0f if the variance product is non-finite (degenerate).
    float computeWSim() const thread {
        const float rawSim = getVarianceXYW()
                           / sqrt(getVarianceXW() * getVarianceYW());
        return isfinite(rawSim) ? -rawSim : 1.0f;
    }
};

}  // namespace av_depthmap
