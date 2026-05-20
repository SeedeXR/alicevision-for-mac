#include "av/depth_map/MipmappedArray.hpp"
#include "av/depth_map/GaussianTable.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Texture.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace av::depth_map {

namespace {

// Mirror MSL `MipmapLevelParams`.
struct MipmapLevelParams {
    std::uint32_t width;
    std::uint32_t height;
    std::int32_t  radius;
};

constexpr std::int32_t kUpstreamRadius = 2;  // upstream TRadius=2

}  // namespace

struct MipmappedArray::Impl {
    av::gpu::Device&  device;
    av::gpu::Pipeline pipeline;
    GaussianTable&    table;

    Impl(av::gpu::Device& d, av::gpu::Pipeline p, GaussianTable& t) noexcept
        : device(d), pipeline(std::move(p)), table(t) {}
};

MipmappedArray::MipmappedArray(av::gpu::Device& dev, GaussianTable& table)
    : impl_(std::make_unique<Impl>(
          dev,
          dev.make_pipeline("av_create_mipmapped_array_level"),
          table))
{}

MipmappedArray::MipmappedArray(MipmappedArray&&) noexcept            = default;
MipmappedArray& MipmappedArray::operator=(MipmappedArray&&) noexcept = default;
MipmappedArray::~MipmappedArray()                                    = default;

void MipmappedArray::build_mip_cascade(av::gpu::Texture& dst)
{
    using namespace av::gpu;

    if (dst.format() != PixelFormat::RGBA32Float)
        throw std::invalid_argument(
            "MipmappedArray::build_mip_cascade: dst must be RGBA32Float");

    const std::uint32_t levels = dst.mip_levels();
    if (levels <= 1) return;

    const std::uint32_t w0 = dst.width();
    const std::uint32_t h0 = dst.height();

    // Pull level 0 into a single-mip working texture; subsequent
    // iterations read this and write into a flat float4 buffer at
    // the destination level dims. After the kernel, we upload the
    // buffer to the destination's mip level AND copy it into the
    // working texture for the next iteration. Two single-mip
    // textures (prev / cur) form a ping-pong pair.
    //
    // Why this indirection: our Texture API doesn't expose mip-level
    // texture views, so we can't bind level N-1 of `dst` directly as
    // a sampleable texture (Metal mip sampling requires LOD bias /
    // explicit-LOD form, which our MSL kernel doesn't use — matching
    // upstream's plain `tex2D` reads). Using a fresh single-mip
    // texture per step keeps the bilinear sampling semantics exact.
    const std::size_t lv0_bytes =
        std::size_t(w0) * std::size_t(h0)
        * Texture::bytes_per_pixel(PixelFormat::RGBA32Float);
    std::vector<std::byte> staging(lv0_bytes);
    dst.download_level(std::span<std::byte>(staging), 0);

    Texture prev_tex(impl_->device,
        Texture::Descriptor{ w0, h0, /*mip_levels=*/1u,
                             PixelFormat::RGBA32Float });
    prev_tex.set_label("mipmap.prev_level");
    prev_tex.upload_level(std::span<const std::byte>(staging), 0);

    for (std::uint32_t L = 1; L < levels; ++L) {
        const std::uint32_t cur_w = std::max<std::uint32_t>(1, w0 >> L);
        const std::uint32_t cur_h = std::max<std::uint32_t>(1, h0 >> L);
        const std::size_t cur_bytes =
            std::size_t(cur_w) * std::size_t(cur_h)
            * Texture::bytes_per_pixel(PixelFormat::RGBA32Float);

        Buffer out_buf(impl_->device, cur_bytes);
        out_buf.set_label("mipmap.level_out");

        MipmapLevelParams params{ cur_w, cur_h, kUpstreamRadius };

        CommandBuffer cb(impl_->device);
        cb.set_label  ("mipmap.create_level")
          .set_pipeline(impl_->pipeline)
          .set_texture (0, prev_tex)
          .set_buffer  (0, out_buf)
          .set_buffer  (1, impl_->table.weights())
          .set_buffer  (2, impl_->table.offsets())
          .set_bytes   (3, &params, sizeof(params))
          .dispatch({ cur_w, cur_h, 1u },
                    { 16u, 16u, 1u });
        cb.commit_and_wait();

        // Copy this level into the destination's mip.
        dst.upload_level(
            std::span<const std::byte>(
                static_cast<const std::byte*>(out_buf.data()),
                out_buf.size_bytes()),
            L);

        // Roll prev_tex forward (rebuild a single-mip texture sized
        // to `cur`, seed it with the buffer we just wrote).
        if (L + 1 < levels) {
            Texture next_prev(impl_->device,
                Texture::Descriptor{ cur_w, cur_h, /*mip_levels=*/1u,
                                     PixelFormat::RGBA32Float });
            next_prev.set_label("mipmap.prev_level");
            next_prev.upload_level(
                std::span<const std::byte>(
                    static_cast<const std::byte*>(out_buf.data()),
                    out_buf.size_bytes()),
                0);
            prev_tex = std::move(next_prev);
        }
    }
}

}  // namespace av::depth_map
