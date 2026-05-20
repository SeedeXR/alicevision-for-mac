// test_texture_smoke.cpp — end-to-end validation of
// av::gpu::Texture, including:
//   * Allocation with auto-mip-level count.
//   * CPU upload of a known pattern via UMA (Shared storage).
//   * Mipmap generation through Texture::generate_mipmaps().
//   * Sampling at sub-pixel + at non-zero mip levels from a
//     Metal kernel using a constexpr sampler.
//
// Pattern: 64x64 RGBA8Unorm ramp where pixel (i, j) has
//          ((i * 4) & 0xff, (j * 4) & 0xff, 0, 255).
//
// Probes:
//   P1. (u=0.5, v=0.5, level=0)  -> bilinear midpoint
//   P2. (u=0.0, v=0.0, level=0)  -> sample corner texel
//   P3. (u=0.5, v=0.5, level=5)  -> mip 5 = 2x2; average of all
//   P4. (u=1/128, v=1/128, level=0) -> half-pixel from corner
//
// All comparisons in normalized [0..1] color space because the
// sampler returns RGBA8 mapped to float.

#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Texture.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr std::uint32_t W = 64;
constexpr std::uint32_t H = 64;

struct TexProbe { float u, v, level; };

// Build the level-0 pattern on the CPU.
std::vector<std::uint8_t> make_pattern()
{
    std::vector<std::uint8_t> px(W * H * 4);
    for (std::uint32_t j = 0; j < H; ++j) {
        for (std::uint32_t i = 0; i < W; ++i) {
            const std::size_t k = (j * W + i) * 4;
            px[k + 0] = static_cast<std::uint8_t>((i * 4) & 0xff);
            px[k + 1] = static_cast<std::uint8_t>((j * 4) & 0xff);
            px[k + 2] = 0;
            px[k + 3] = 255;
        }
    }
    return px;
}

// Expected float [0..1] from a u8 byte through RGBA8Unorm
// conversion (the sampler returns byte/255.0).
constexpr float u8f(int x) { return static_cast<float>(x) / 255.0f; }

bool nearly(float a, float b, float tol)
{
    return std::abs(a - b) <= tol;
}

}  // namespace

int main() try
{
    using namespace av::gpu;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    // Allocate texture (auto mip levels = 7 for 64x64).
    Texture tex(dev, Texture::Descriptor{
        .width      = W,
        .height     = H,
        .mip_levels = 0,                          // auto
        .format     = PixelFormat::RGBA8Unorm,
    });
    tex.set_label("texture_smoke.input");
    std::printf("[info] mip_levels   : %u\n", tex.mip_levels());

    // Upload pattern to mip 0.
    {
        const auto px = make_pattern();
        tex.upload_level({ reinterpret_cast<const std::byte*>(px.data()),
                           px.size() * sizeof(std::uint8_t) }, 0);
    }

    // Generate mips.
    tex.generate_mipmaps();

    // Build probes.
    const std::array<TexProbe, 4> probes = {{
        { 0.5f,        0.5f,        0.0f },
        { 0.0f,        0.0f,        0.0f },
        { 0.5f,        0.5f,        5.0f },
        { 1.0f / 128,  1.0f / 128,  0.0f },
    }};
    const std::uint32_t count = static_cast<std::uint32_t>(probes.size());

    // Allocate Metal buffers and stage data.
    Buffer probe_buf(dev, sizeof(probes));
    Buffer out_buf  (dev, probes.size() * 4 * sizeof(float));
    probe_buf.set_label("texture_smoke.probes");
    out_buf  .set_label("texture_smoke.out");
    probe_buf.upload(std::span<const TexProbe>(probes));

    auto pipe = dev.make_pipeline("av_texture_sample");

    CommandBuffer cb(dev);
    cb.set_label("texture_smoke.cb")
      .set_pipeline (pipe)
      .set_texture  (0, tex)
      .set_buffer   (0, probe_buf)
      .set_buffer   (1, out_buf)
      .set_bytes    (2, &count, sizeof(count))
      .dispatch_1d  (pipe, count);
    cb.commit_and_wait();

    auto out = out_buf.as_span<const float>();
    auto rgba_at = [&](std::size_t i) {
        return std::array<float, 4>{ out[i * 4 + 0], out[i * 4 + 1],
                                     out[i * 4 + 2], out[i * 4 + 3] };
    };

    int bad = 0;

    // ---------------- P1: bilinear midpoint at (32, 32) -----------
    //
    // Sampler `coord::normalized` + `address::clamp_to_edge` maps
    // u=0.5 to texel coord 32.0 (centered on the boundary between
    // texels 31 and 32). With filter::linear the result is the
    // average of texels (31, 32), (32, 32), (31, 31), (32, 31).
    {
        const auto c = rgba_at(0);
        // Expected red: avg of (4*31), (4*32) = (124, 128) → 126
        // Expected green: same pattern → 126
        const float exp_r = (u8f(124) + u8f(128)) * 0.5f;
        const float exp_g = (u8f(124) + u8f(128)) * 0.5f;
        if (!nearly(c[0], exp_r, 1.5f / 255.f) ||
            !nearly(c[1], exp_g, 1.5f / 255.f) ||
            !nearly(c[2], 0.0f, 1e-3f) ||
            !nearly(c[3], 1.0f, 1e-3f)) {
            std::fprintf(stderr,
                "P1 midpoint: got (%.4f, %.4f, %.4f, %.4f) "
                "want ≈ (%.4f, %.4f, 0, 1)\n",
                static_cast<double>(c[0]), static_cast<double>(c[1]),
                static_cast<double>(c[2]), static_cast<double>(c[3]),
                static_cast<double>(exp_r), static_cast<double>(exp_g));
            ++bad;
        } else {
            std::printf("[ok]  P1 mid       (%.4f, %.4f, %.4f, %.4f)\n",
                static_cast<double>(c[0]), static_cast<double>(c[1]),
                static_cast<double>(c[2]), static_cast<double>(c[3]));
        }
    }

    // ---------------- P2: corner (0, 0) ----------------------------
    //
    // u=0 lands exactly on the left edge: with clamp_to_edge the
    // sampled value at texel 0 is the texel itself: (0, 0, 0, 255).
    {
        const auto c = rgba_at(1);
        if (!nearly(c[0], 0.0f, 1e-3f) ||
            !nearly(c[1], 0.0f, 1e-3f) ||
            !nearly(c[3], 1.0f, 1e-3f)) {
            std::fprintf(stderr,
                "P2 corner: got (%.4f, %.4f, %.4f, %.4f)\n",
                static_cast<double>(c[0]), static_cast<double>(c[1]),
                static_cast<double>(c[2]), static_cast<double>(c[3]));
            ++bad;
        } else {
            std::printf("[ok]  P2 corner    (%.4f, %.4f, %.4f, %.4f)\n",
                static_cast<double>(c[0]), static_cast<double>(c[1]),
                static_cast<double>(c[2]), static_cast<double>(c[3]));
        }
    }

    // ---------------- P3: mip level 5 midpoint --------------------
    //
    // 64x64 → mip 5 = 2x2 texture. Sampling at u=v=0.5 on level 5
    // is bilinear-interpolating the 4 texels of the 2x2. Each of
    // those texels is the box-filter average of a 32x32 region of
    // the original. Hence the expected R is the global average of
    // (i*4) over i in [0,64): mean of {0,4,8,...,252} = 126 → /255.
    // Same for G (since pattern is symmetric in j).
    //
    // generateMipmaps on RGBA8Unorm rounds at each level; small
    // integer drift accumulates over 5 reductions, so we budget
    // a few ULPs in u8 space (4/255).
    {
        const auto c = rgba_at(2);
        constexpr float exp_avg = 126.0f / 255.0f;
        const float tol = 4.0f / 255.0f;
        if (!nearly(c[0], exp_avg, tol) ||
            !nearly(c[1], exp_avg, tol) ||
            !nearly(c[2], 0.0f, 1e-3f) ||
            !nearly(c[3], 1.0f, 1e-3f)) {
            std::fprintf(stderr,
                "P3 mip5 mid: got (%.4f, %.4f, %.4f, %.4f) "
                "want ≈ (%.4f, %.4f, 0, 1)\n",
                static_cast<double>(c[0]), static_cast<double>(c[1]),
                static_cast<double>(c[2]), static_cast<double>(c[3]),
                static_cast<double>(exp_avg), static_cast<double>(exp_avg));
            ++bad;
        } else {
            std::printf("[ok]  P3 mip5      (%.4f, %.4f, %.4f, %.4f)\n",
                static_cast<double>(c[0]), static_cast<double>(c[1]),
                static_cast<double>(c[2]), static_cast<double>(c[3]));
        }
    }

    // ---------------- P4: u=v=1/128, near top-left ----------------
    //
    // With 64-wide texture, normalized u=1/128 maps to texel coord
    // 0.5 — i.e. perfectly aligned with the center of texel 0. The
    // bilinear filter returns texel(0,0) exactly = (0,0,0,1).
    {
        const auto c = rgba_at(3);
        if (!nearly(c[0], 0.0f, 1e-3f) ||
            !nearly(c[1], 0.0f, 1e-3f) ||
            !nearly(c[3], 1.0f, 1e-3f)) {
            std::fprintf(stderr,
                "P4 near-corner: got (%.4f, %.4f, %.4f, %.4f)\n",
                static_cast<double>(c[0]), static_cast<double>(c[1]),
                static_cast<double>(c[2]), static_cast<double>(c[3]));
            ++bad;
        } else {
            std::printf("[ok]  P4 near0,0   (%.4f, %.4f, %.4f, %.4f)\n",
                static_cast<double>(c[0]), static_cast<double>(c[1]),
                static_cast<double>(c[2]), static_cast<double>(c[3]));
        }
    }

    if (bad) {
        std::fprintf(stderr, "FAIL: %d probe(s) out of tolerance\n", bad);
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
