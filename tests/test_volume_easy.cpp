// test_volume_easy.cpp — validation of the three "easy" cost-
// volume kernels ported in Session 10:
//
//   volume_init_uchar          — set all voxels to a value (TSim = uchar)
//   volume_init_half           — set all voxels to a value (TSimRefine = half)
//   volume_add_half            — in-place a += b on FP16 volumes
//   volume_update_uninitialized — where 2nd-best == 255, copy 1st-best
//
// These don't touch textures or camera params, so the test is a
// pure buffer-in / buffer-out validation. The CPU reference is
// trivial: we just check every voxel.
//
// FP16 note: the GPU does `a + b` in float then demotes; we
// reproduce the same pattern host-side with a small float-to-half
// helper to keep the comparison bit-faithful where possible.

#include "av/depth_map/Volume.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr std::uint32_t W = 64;
constexpr std::uint32_t H = 32;
constexpr std::uint32_t D = 16;

constexpr av::depth_map::VolumeDims kDims{ W, H, D };

// IEEE binary16 helpers (matching Volume.cpp's float_to_half_bits).
std::uint16_t f2h_bits(float f) {
    const std::uint32_t fi = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (fi >> 16) & 0x8000u;
    std::int32_t        exp  = static_cast<std::int32_t>((fi >> 23) & 0xffu) - 127 + 15;
    std::uint32_t       mant =  fi & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant |= 0x800000u;
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
        const std::uint32_t round_bit = 1u << (shift - 1);
        std::uint32_t       half_mant = mant >> shift;
        if ((mant & ((1u << shift) - 1u)) > round_bit ||
            ((mant & ((1u << shift) - 1u)) == round_bit && (half_mant & 1u))) {
            half_mant += 1u;
        }
        return static_cast<std::uint16_t>(sign | half_mant);
    }
    if (exp >= 31) {
        if (((fi >> 23) & 0xffu) == 0xffu && mant != 0)
            return static_cast<std::uint16_t>(sign | 0x7c00u | (mant >> 13) | 1u);
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    const std::uint32_t round_bit = 1u << 12;
    std::uint32_t       half_mant = mant >> 13;
    if ((mant & 0x1fffu) > round_bit ||
        ((mant & 0x1fffu) == round_bit && (half_mant & 1u))) {
        half_mant += 1u;
    }
    if (half_mant == 0x400u) {
        half_mant = 0;
        exp += 1;
        if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(exp) << 10) | half_mant);
}

float h2f(std::uint16_t h) {
    const std::uint32_t sign  = (h & 0x8000u) << 16;
    const std::uint32_t exp16 = (h >> 10) & 0x1fu;
    const std::uint32_t mant  =  h & 0x3ffu;

    if (exp16 == 0) {
        if (mant == 0) return std::bit_cast<float>(sign);
        // Subnormal: normalize.
        int e = -1;
        std::uint32_t m = mant;
        do { m <<= 1; ++e; } while ((m & 0x400u) == 0);
        const std::uint32_t fi = sign
            | (static_cast<std::uint32_t>(127 - 15 - e) << 23)
            | ((m & 0x3ffu) << 13);
        return std::bit_cast<float>(fi);
    }
    if (exp16 == 31) {
        const std::uint32_t fi = sign | 0x7f800000u | (mant << 13);
        return std::bit_cast<float>(fi);
    }
    const std::uint32_t fi = sign
        | ((exp16 + 127 - 15) << 23)
        | (mant << 13);
    return std::bit_cast<float>(fi);
}

}  // namespace

int main() try
{
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device       : %s\n", dev.name().c_str());

    Volume vol(dev);

    const std::size_t voxel_count = kDims.voxel_count();
    int bad = 0;

    // ============== volume_init (uchar) =========================
    {
        Buffer buf(dev, voxel_count * sizeof(std::uint8_t));
        buf.set_label("init_uchar.buf");
        // Pre-fill with a contrasting value so the kernel actually has to write.
        std::vector<std::uint8_t> pre(voxel_count, 0xaa);
        buf.upload(std::span<const std::uint8_t>(pre));

        constexpr std::uint8_t kInitVal = 0x42;
        vol.init_sim(buf, kDims, kInitVal);

        const auto* gpu = static_cast<const std::uint8_t*>(buf.data());
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < voxel_count; ++i)
            if (gpu[i] != kInitVal) ++mismatches;
        if (mismatches) {
            std::fprintf(stderr,
                "init_sim: %zu voxels not initialized to 0x%02x\n",
                mismatches, kInitVal);
            ++bad;
        } else {
            std::printf("[ok]  init_sim     %u×%u×%u voxels all = 0x%02x\n",
                        kDims.x, kDims.y, kDims.z, kInitVal);
        }
    }

    // ============== volume_init (half) ==========================
    {
        Buffer buf(dev, voxel_count * sizeof(std::uint16_t));
        buf.set_label("init_half.buf");
        std::vector<std::uint16_t> pre(voxel_count, 0xbeef);
        buf.upload(std::span<const std::uint16_t>(pre));

        constexpr float kInitFloat = 0.25f;
        const std::uint16_t kInitBits = f2h_bits(kInitFloat);
        vol.init_refine(buf, kDims, kInitFloat);

        const auto* gpu = static_cast<const std::uint16_t*>(buf.data());
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < voxel_count; ++i)
            if (gpu[i] != kInitBits) ++mismatches;
        if (mismatches) {
            std::fprintf(stderr,
                "init_refine: %zu voxels not initialized to half=%.4f (bits=0x%04x)\n",
                mismatches,
                static_cast<double>(kInitFloat),
                kInitBits);
            ++bad;
        } else {
            std::printf("[ok]  init_refine  %u×%u×%u voxels all = %.4f (half bits 0x%04x)\n",
                        kDims.x, kDims.y, kDims.z,
                        static_cast<double>(kInitFloat), kInitBits);
        }
    }

    // ============== volume_add (half) ===========================
    {
        Buffer inout(dev, voxel_count * sizeof(std::uint16_t));
        Buffer in   (dev, voxel_count * sizeof(std::uint16_t));
        inout.set_label("add_refine.inout");
        in.   set_label("add_refine.in");

        std::mt19937_64 rng(0x600ca1);
        std::uniform_real_distribution<float> U(-5.0f, 5.0f);
        std::vector<std::uint16_t> a(voxel_count), b(voxel_count);
        for (std::size_t i = 0; i < voxel_count; ++i) {
            a[i] = f2h_bits(U(rng));
            b[i] = f2h_bits(U(rng));
        }
        inout.upload(std::span<const std::uint16_t>(a));
        in   .upload(std::span<const std::uint16_t>(b));

        vol.add_refine(inout, in, kDims);

        // Reference: promote both halves to float, add, demote.
        const auto* gpu = static_cast<const std::uint16_t*>(inout.data());
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < voxel_count; ++i) {
            const std::uint16_t expected = f2h_bits(h2f(a[i]) + h2f(b[i]));
            if (gpu[i] != expected) ++mismatches;
        }
        if (mismatches) {
            std::fprintf(stderr,
                "add_refine: %zu / %zu voxels disagree (half promote-add-demote)\n",
                mismatches, voxel_count);
            ++bad;
        } else {
            std::printf("[ok]  add_refine   %zu voxels bit-exact vs CPU reference\n",
                        voxel_count);
        }
    }

    // ============== volume_update_uninitialized =================
    {
        Buffer v2nd(dev, voxel_count * sizeof(std::uint8_t));
        Buffer v1st(dev, voxel_count * sizeof(std::uint8_t));
        v2nd.set_label("update.v2nd");
        v1st.set_label("update.v1st");

        std::mt19937_64 rng(0xfee1);
        std::uniform_int_distribution<int> U255(0, 255);
        std::vector<std::uint8_t> a(voxel_count), b(voxel_count), pre(voxel_count);
        for (std::size_t i = 0; i < voxel_count; ++i) {
            a[i] = static_cast<std::uint8_t>(U255(rng));
            // 30% of 2nd-best slots are uninitialized (== 255).
            pre[i] = (U255(rng) < 76) ? std::uint8_t(255)
                                      : static_cast<std::uint8_t>(U255(rng));
            b[i] = pre[i];
        }
        v1st.upload(std::span<const std::uint8_t>(a));
        v2nd.upload(std::span<const std::uint8_t>(b));

        vol.update_uninitialized(v2nd, v1st, kDims);

        const auto* gpu = static_cast<const std::uint8_t*>(v2nd.data());
        std::size_t fixed = 0, kept = 0, mismatches = 0;
        for (std::size_t i = 0; i < voxel_count; ++i) {
            const std::uint8_t expected = (pre[i] == 255) ? a[i] : pre[i];
            if (gpu[i] != expected) ++mismatches;
            if (pre[i] == 255) ++fixed; else ++kept;
        }
        if (mismatches) {
            std::fprintf(stderr,
                "update_uninitialized: %zu voxels disagree\n",
                mismatches);
            ++bad;
        } else {
            std::printf(
                "[ok]  update       %zu sentinel voxels overwritten, %zu kept\n",
                fixed, kept);
        }
    }

    if (bad) {
        std::fprintf(stderr, "FAIL: %d kernel(s) failed\n", bad);
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
