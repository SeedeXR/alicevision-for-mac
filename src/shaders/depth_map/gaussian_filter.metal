// gaussian_filter.metal — Metal port of two kernels from
// depthMap/cuda/imageProcessing/deviceGaussianFilter.cu:
//   * downscaleWithGaussianBlur  (texture → buffer, 2D Gaussian + decimation)
//   * medianFilter3              (texture → buffer, 7×7 median)
//
// The two volume Gaussian kernels (gaussianBlurVolumeZ /
// gaussianBlurVolumeXYZ) are SGM cost-volume helpers and are
// deferred until the planeSweeping port pulls them in.
//
// CUDA → MSL translation map:
//
//   tex2D_float4(tex, fx, fy)         → tex.sample(smp, float2(fx/w, fy/h))
//                                       (we pre-normalize the coords on the
//                                       host? No — easier to use coord::pixel
//                                       directly in MSL, which we do.)
//   __constant__ float[MEM_SIZE]      → device pointer arg passed via buffer(N)
//   __constant__ int[SCALES]          → constant pointer arg passed via buffer(N)
//   getGauss(scale, idx) lookup       → ported as a tiny inline helper
//                                       reading from the bound buffers.
//
// We use `coord::pixel` on the texture sampler to keep the upstream's
// pixel-space arithmetic (`x * downscale + j + 0.5`) intact. MSL
// translates that to the same hardware sampling units.

#include <metal_stdlib>
using namespace metal;

// Look up a Gaussian weight from the host-precomputed LUT. The
// layout matches upstream: `offsets[scale]` is the starting index
// of scale `scale` inside the flat `weights` array; weights for a
// scale of radius r = scale+1 occupy the next (2r+1) floats.
inline float av_getGauss(constant const float* weights,
                         constant const int*   offsets,
                         int scale, int idx)
{
    return weights[offsets[scale] + idx];
}

// ----------------------------------------------------------------
// downscaleWithGaussianBlur — 2D non-separable Gaussian + decimation.
//
// Output: a downscaled `width × height` RGBA buffer.
// Input:  a high-res RGBA texture with bilinear filtering.
// Each output pixel reads (2·gaussRadius+1)² texture samples,
// applies separable Gaussian weights, normalizes by the (sum of)
// weights actually used, and writes the result.
//
// The upstream comment says the kernel is intentionally
// non-separable for simplicity, not performance. We preserve that
// to keep numerical agreement with the CUDA reference.
// ----------------------------------------------------------------

struct DownscaleParams {
    uint  downscaledWidth;
    uint  downscaledHeight;
    int   downscale;
    int   gaussRadius;
    uint  inputWidth;     // for pixel→normalized conversion
    uint  inputHeight;
};

kernel void av_downscale_with_gaussian_blur(
    texture2d<float, access::sample> in_tex     [[texture(0)]],
    device   float4*                 out_buf    [[buffer(0)]],
    constant const float*            gauss_w    [[buffer(1)]],
    constant const int*              gauss_off  [[buffer(2)]],
    constant DownscaleParams&        p          [[buffer(3)]],
    uint2                            gid        [[thread_position_in_grid]])
{
    constexpr sampler smp(coord::normalized,
                          address::clamp_to_edge,
                          filter::linear);

    if (gid.x >= p.downscaledWidth || gid.y >= p.downscaledHeight)
        return;

    const float s = float(p.downscale) * 0.5f;
    const float invW = 1.0f / float(p.inputWidth);
    const float invH = 1.0f / float(p.inputHeight);

    float4 acc        = float4(0.0f);
    float  sumFactor  = 0.0f;
    const int scale   = p.downscale - 1;

    for (int i = -p.gaussRadius; i <= p.gaussRadius; ++i) {
        for (int j = -p.gaussRadius; j <= p.gaussRadius; ++j) {
            const float fx = float(int(gid.x) * p.downscale + j) + s;
            const float fy = float(int(gid.y) * p.downscale + i) + s;

            const float4 curPix = in_tex.sample(smp,
                                                float2(fx * invW, fy * invH));
            const float  factor = av_getGauss(gauss_w, gauss_off, scale, i + p.gaussRadius)
                                * av_getGauss(gauss_w, gauss_off, scale, j + p.gaussRadius);

            acc       += curPix * factor;
            sumFactor += factor;
        }
    }

    out_buf[gid.y * p.downscaledWidth + gid.x] = acc / sumFactor;
}

// ----------------------------------------------------------------
// medianFilter3 — 7×7 median on a single-channel float texture.
//
// Upstream uses an O(n²) compare-swap "sort then take the middle"
// approach with a 49-element register buffer (radius hardcoded to
// 3). Preserved verbatim for parity with the CUDA original.
// ----------------------------------------------------------------

struct MedianParams {
    uint width;
    uint height;
    // `scale` was used by upstream's call site but ignored inside
    // the kernel; not a real parameter.
};

kernel void av_median_filter_3(
    texture2d<float, access::sample> in_tex  [[texture(0)]],
    device   float*                  out_buf [[buffer(0)]],
    constant MedianParams&           p       [[buffer(1)]],
    uint2                            gid     [[thread_position_in_grid]])
{
    // Upstream skips the border region; we keep that policy.
    constexpr int radius        = 3;
    constexpr int filterWidth   = radius * 2 + 1;       // 7
    constexpr int filterPixels  = filterWidth * filterWidth; // 49

    if (gid.x < uint(radius) || gid.y < uint(radius) ||
        gid.x >= p.width  - uint(radius) ||
        gid.y >= p.height - uint(radius))
        return;

    // Texture sampling: nearest-neighbor read at integer coords. We
    // use the sampler with filter::nearest to match upstream's
    // tex2D<float>(tex, ix, iy) semantics (which is point-sampled at
    // texel centers — index + 0.5 in coord::pixel).
    constexpr sampler smp(coord::pixel,
                          address::clamp_to_edge,
                          filter::nearest);

    float buf[filterPixels];
    for (int yi = 0; yi < filterWidth; ++yi) {
        for (int xi = 0; xi < filterWidth; ++xi) {
            const float fx = float(int(gid.x) + xi - radius) + 0.5f;
            const float fy = float(int(gid.y) + yi - radius) + 0.5f;
            buf[yi * filterWidth + xi] = in_tex.sample(smp,
                                                       float2(fx, fy)).x;
        }
    }

    // Selection-sort descending: after the outer loop, buf[k] is the
    // (k+1)th largest. The element at the median index (24 for a 49-
    // length array) is the median. Cost is O(n²) ≈ 49² = 2401 ops;
    // fine at the small radius we run.
    for (int k = 0; k < filterPixels; ++k) {
        for (int l = 0; l < filterPixels; ++l) {
            if (buf[k] < buf[l]) {
                const float t = buf[k];
                buf[k] = buf[l];
                buf[l] = t;
            }
        }
    }

    out_buf[gid.y * p.width + gid.x] = buf[radius * filterWidth + radius];
}

// ----------------------------------------------------------------
// gaussianBlurVolumeZ — separable 1D Gaussian along the Z axis of
// a packed 3D float volume.
//
// Upstream (deviceGaussianFilter.cu:82). For each (vx, vy, vz):
//   * For rz in [-r, +r]: read voxel (vx, vy, vz+rz) when in-bounds
//     according to upstream's `(iz < volDimZ) && (iz > 0)` test.
//   * The `iz > 0` (rather than `iz >= 0`) is an upstream quirk —
//     vz=0 is skipped. Preserved for bit-for-bit parity.
//   * Accumulate value * weight and normalize by sum-of-weights.
//
// Layout: packed, no padding, linear index = z*(X*Y) + y*X + x.
// Gaussian scale: gaussScale = gaussRadius - 1, matching upstream.
// ----------------------------------------------------------------

struct VolumeBlurParams {
    uint volDimX;
    uint volDimY;
    uint volDimZ;
    int  gaussRadius;
};

inline uint av_vol_idx(uint vx, uint vy, uint vz, uint X, uint Y)
{
    return vz * (X * Y) + vy * X + vx;
}

kernel void av_gaussian_blur_volume_z(
    device   float*               out_volume [[buffer(0)]],
    device const float*           in_volume  [[buffer(1)]],
    constant const float*         gauss_w    [[buffer(2)]],
    constant const int*           gauss_off  [[buffer(3)]],
    constant VolumeBlurParams&    p          [[buffer(4)]],
    uint3                         gid        [[thread_position_in_grid]])
{
    const uint vx = gid.x;
    const uint vy = gid.y;
    const uint vz = gid.z;

    if (vx >= p.volDimX || vy >= p.volDimY || vz >= p.volDimZ) return;

    const int gaussScale = p.gaussRadius - 1;

    float sum       = 0.0f;
    float sumFactor = 0.0f;

    for (int rz = -p.gaussRadius; rz <= p.gaussRadius; ++rz) {
        const int iz = int(vz) + rz;
        // Match upstream's `(iz < volDimZ) && (iz > 0)` — note the
        // strict `> 0` (not `>= 0`) which skips iz==0.
        if (iz < int(p.volDimZ) && iz > 0) {
            const float value  = in_volume[av_vol_idx(vx, vy, uint(iz),
                                                      p.volDimX, p.volDimY)];
            const float factor = av_getGauss(gauss_w, gauss_off,
                                             gaussScale, rz + p.gaussRadius);
            sum       += value * factor;
            sumFactor += factor;
        }
    }

    out_volume[av_vol_idx(vx, vy, vz, p.volDimX, p.volDimY)] = sum / sumFactor;
}

// ----------------------------------------------------------------
// gaussianBlurVolumeXYZ — 3D separable Gaussian (computed as a full
// 3-nested loop, matching upstream's intentionally non-separable
// implementation). For each (vx, vy, vz):
//   * Clamp the kernel range to the in-bounds region per axis.
//   * Accumulate value * (w_x * w_y * w_z) and normalize.
//
// Upstream (deviceGaussianFilter.cu:113).
// ----------------------------------------------------------------

kernel void av_gaussian_blur_volume_xyz(
    device   float*               out_volume [[buffer(0)]],
    device const float*           in_volume  [[buffer(1)]],
    constant const float*         gauss_w    [[buffer(2)]],
    constant const int*           gauss_off  [[buffer(3)]],
    constant VolumeBlurParams&    p          [[buffer(4)]],
    uint3                         gid        [[thread_position_in_grid]])
{
    const uint vx = gid.x;
    const uint vy = gid.y;
    const uint vz = gid.z;

    if (vx >= p.volDimX || vy >= p.volDimY || vz >= p.volDimZ) return;

    const int gaussScale = p.gaussRadius - 1;

    const int xMinRadius = max(-p.gaussRadius, -int(vx));
    const int yMinRadius = max(-p.gaussRadius, -int(vy));
    const int zMinRadius = max(-p.gaussRadius, -int(vz));

    const int xMaxRadius = min( p.gaussRadius, int(p.volDimX) - int(vx) - 1);
    const int yMaxRadius = min( p.gaussRadius, int(p.volDimY) - int(vy) - 1);
    const int zMaxRadius = min( p.gaussRadius, int(p.volDimZ) - int(vz) - 1);

    float sum       = 0.0f;
    float sumFactor = 0.0f;

    for (int rx = xMinRadius; rx <= xMaxRadius; ++rx) {
        const int ix = int(vx) + rx;
        const float fx = av_getGauss(gauss_w, gauss_off,
                                     gaussScale, rx + p.gaussRadius);
        for (int ry = yMinRadius; ry <= yMaxRadius; ++ry) {
            const int iy = int(vy) + ry;
            const float fy = av_getGauss(gauss_w, gauss_off,
                                         gaussScale, ry + p.gaussRadius);
            for (int rz = zMinRadius; rz <= zMaxRadius; ++rz) {
                const int iz = int(vz) + rz;
                const float fz = av_getGauss(gauss_w, gauss_off,
                                             gaussScale, rz + p.gaussRadius);
                const float value = in_volume[av_vol_idx(uint(ix), uint(iy), uint(iz),
                                                         p.volDimX, p.volDimY)];
                const float factor = fx * fy * fz;
                sum       += value * factor;
                sumFactor += factor;
            }
        }
    }

    out_volume[av_vol_idx(vx, vy, vz, p.volDimX, p.volDimY)] = sum / sumFactor;
}
