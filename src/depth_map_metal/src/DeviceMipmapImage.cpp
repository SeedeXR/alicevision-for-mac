#include "av/depth_map/DeviceMipmapImage.hpp"

#include "av/depth_map/GaussianFilter.hpp"
#include "av/depth_map/GaussianTable.hpp"
#include "av/depth_map/ImageColorConversion.hpp"
#include "av/depth_map/MipmappedArray.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Texture.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace av::depth_map {

namespace {

bool is_pow2(std::uint32_t v) noexcept {
    return v != 0 && (v & (v - 1)) == 0;
}

std::uint32_t log2_u32(std::uint32_t v) noexcept {
    // v is assumed to be a positive power of two.
    std::uint32_t r = 0;
    while ((1u << r) < v) ++r;
    return r;
}

}  // namespace

struct DeviceMipmapImage::Impl {
    av::gpu::Device&                      device;
    GaussianTable                         gauss_table;
    GaussianFilter                        gauss_filter;
    ImageColorConversion                  rgb2lab;
    MipmappedArray                        mip_builder;
    std::unique_ptr<av::gpu::Texture>     tex;

    // metadata
    std::uint32_t orig_width   = 0;
    std::uint32_t orig_height  = 0;
    std::uint32_t min_dnscale  = 0;
    std::uint32_t max_dnscale  = 0;
    std::uint32_t levels       = 0;
    bool          use_upstream = false;

    explicit Impl(av::gpu::Device& d)
        : device(d),
          gauss_table(d),
          gauss_filter(d, gauss_table),
          rgb2lab(d),
          mip_builder(d, gauss_table) {}
};

DeviceMipmapImage::DeviceMipmapImage(av::gpu::Device& dev)
    : impl_(std::make_unique<Impl>(dev))
{}

DeviceMipmapImage::DeviceMipmapImage(DeviceMipmapImage&&) noexcept            = default;
DeviceMipmapImage& DeviceMipmapImage::operator=(DeviceMipmapImage&&) noexcept = default;
DeviceMipmapImage::~DeviceMipmapImage()                                       = default;

void DeviceMipmapImage::fill(std::span<const float> rgba_image,
                             std::uint32_t          width,
                             std::uint32_t          height,
                             std::uint32_t          min_downscale,
                             std::uint32_t          max_downscale)
{
    using namespace av::gpu;

    if (width == 0 || height == 0)
        throw std::invalid_argument("DeviceMipmapImage::fill: zero dims");
    if (!is_pow2(min_downscale) || !is_pow2(max_downscale))
        throw std::invalid_argument(
            "DeviceMipmapImage::fill: downscale factors must be positive powers of two");
    if (min_downscale > max_downscale)
        throw std::invalid_argument(
            "DeviceMipmapImage::fill: min_downscale > max_downscale");
    const std::size_t need_floats =
        std::size_t(width) * std::size_t(height) * 4;
    if (rgba_image.size() < need_floats)
        throw std::invalid_argument(
            "DeviceMipmapImage::fill: input image too small");

    impl_->orig_width  = width;
    impl_->orig_height = height;
    impl_->min_dnscale = min_downscale;
    impl_->max_dnscale = max_downscale;
    impl_->levels      = log2_u32(max_downscale / min_downscale) + 1;

    const std::uint32_t lv0_w =
        (width  + min_downscale - 1) / min_downscale;
    const std::uint32_t lv0_h =
        (height + min_downscale - 1) / min_downscale;

    // Allocate (or reallocate) the destination texture with the
    // correct level-0 dims + mip levels.
    impl_->tex = std::make_unique<Texture>(
        impl_->device,
        Texture::Descriptor{ lv0_w, lv0_h, impl_->levels,
                             PixelFormat::RGBA32Float });

    // Working texture: single-mip RGBA32Float at level-0 dims.
    // `rgb2lab` runs here. The result is then copied into level 0
    // of the destination mipmapped texture.
    //
    // Why the indirection: MSL's `texture2d<T, access::read_write>`
    // on a multi-mip-level texture requires the explicit-LOD form
    // (`read(coord, level)` / `write(value, coord, level)`) for
    // defined behavior. Our `av_rgb2lab` kernel uses the implicit
    // form `read(gid)`, which works fine on single-mip textures
    // but produces wrong values on multi-mip ones (observed: L
    // values 1.5×–2× the expected range, because the texture
    // sample doesn't bind to level 0 the way you'd expect).
    Texture working_tex(impl_->device,
        Texture::Descriptor{ lv0_w, lv0_h, /*mip_levels=*/1u,
                             PixelFormat::RGBA32Float });

    if (min_downscale > 1) {
        // Upload to a temporary full-res texture for the downscale.
        Texture full_tex(impl_->device,
            Texture::Descriptor{ width, height, 1u,
                                 PixelFormat::RGBA32Float });
        full_tex.upload(rgba_image);

        Buffer ds_buf(impl_->device,
            std::size_t(lv0_w) * std::size_t(lv0_h)
                * 4 * sizeof(float));
        impl_->gauss_filter.downscale_with_gaussian_blur(
            full_tex, ds_buf,
            lv0_w, lv0_h,
            std::int32_t(min_downscale),
            std::int32_t(min_downscale)  /* gauss radius == downscale */);

        working_tex.upload_level(
            std::span<const std::byte>(
                static_cast<const std::byte*>(ds_buf.data()),
                ds_buf.size_bytes()),
            /*mip_level=*/0);
    } else {
        working_tex.upload(rgba_image);
    }

    // RGB → CIELAB on the single-mip working texture. Note: our
    // port (matching upstream) scales the Lab output by 2.55 so
    // that L lands roughly in [0, 255] (instead of the classical
    // [0, 100]). See `xyz2lab` in `color.h`.
    impl_->rgb2lab.rgb2lab(working_tex);

    // Copy the converted level-0 image into the destination
    // mipmapped texture's level 0. UMA makes the host-side
    // memcpy effectively free.
    const std::size_t lv0_bytes =
        std::size_t(lv0_w) * std::size_t(lv0_h) * 4 * sizeof(float);
    std::vector<std::byte> staging(lv0_bytes);
    working_tex.download_level(std::span<std::byte>(staging), 0);
    impl_->tex->upload_level(std::span<const std::byte>(staging), 0);

    // Generate the mip cascade. Default path: Metal's built-in
    // blit-encoder 2× downsampler. Opt-in upstream path: the ported
    // `av_create_mipmapped_array_level` kernel (5×5 Gaussian-weighted
    // bilinear stencil) for bit-for-bit-ish parity with upstream.
    if (impl_->levels > 1) {
        if (impl_->use_upstream) {
            impl_->mip_builder.build_mip_cascade(*impl_->tex);
        } else {
            impl_->tex->generate_mipmaps();
        }
    }
}

void DeviceMipmapImage::set_use_upstream_mipgen(bool enabled) noexcept {
    impl_->use_upstream = enabled;
}

bool DeviceMipmapImage::use_upstream_mipgen() const noexcept {
    return impl_->use_upstream;
}

const av::gpu::Texture& DeviceMipmapImage::texture() const {
    if (!impl_->tex)
        throw std::logic_error("DeviceMipmapImage::texture: fill() not called");
    return *impl_->tex;
}

float DeviceMipmapImage::get_level(std::uint32_t downscale) const {
    if (downscale < impl_->min_dnscale || downscale > impl_->max_dnscale ||
        !is_pow2(downscale))
        throw std::out_of_range(
            "DeviceMipmapImage::get_level: downscale outside [min, max] or not power-of-two");
    return std::log2(float(downscale) / float(impl_->min_dnscale));
}

std::pair<std::uint32_t, std::uint32_t>
DeviceMipmapImage::get_dimensions(std::uint32_t downscale) const {
    if (downscale < impl_->min_dnscale || downscale > impl_->max_dnscale ||
        !is_pow2(downscale))
        throw std::out_of_range(
            "DeviceMipmapImage::get_dimensions: downscale outside [min, max] or not power-of-two");
    const std::uint32_t w =
        (impl_->orig_width  + downscale - 1) / downscale;
    const std::uint32_t h =
        (impl_->orig_height + downscale - 1) / downscale;
    return { w, h };
}

std::uint32_t DeviceMipmapImage::min_downscale() const noexcept { return impl_->min_dnscale; }
std::uint32_t DeviceMipmapImage::max_downscale() const noexcept { return impl_->max_dnscale; }
std::uint32_t DeviceMipmapImage::width () const noexcept { return impl_->orig_width;  }
std::uint32_t DeviceMipmapImage::height() const noexcept { return impl_->orig_height; }

}  // namespace av::depth_map
