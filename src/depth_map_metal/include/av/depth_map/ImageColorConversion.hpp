#pragma once

// ImageColorConversion — host driver for the in-place per-pixel
// RGB→CIELAB kernel (port of upstream's `cuda_rgb2lab`).
//
// Input/output is a single RGBA32Float texture; the kernel reads
// each pixel and writes the Lab result back in place. Alpha is
// preserved.

#include <cstddef>
#include <memory>

namespace av::gpu { class Device; class Texture; }

namespace av::depth_map {

class ImageColorConversion {
public:
    explicit ImageColorConversion(av::gpu::Device& dev);

    ImageColorConversion(const ImageColorConversion&)            = delete;
    ImageColorConversion& operator=(const ImageColorConversion&) = delete;
    ImageColorConversion(ImageColorConversion&&) noexcept;
    ImageColorConversion& operator=(ImageColorConversion&&) noexcept;
    ~ImageColorConversion();

    // Run the in-place rgb2lab kernel over the full extent of `tex`.
    // `tex` must be RGBA32Float and was created with
    // `MTLTextureUsageShaderRead | _Write`. Blocking; commits and
    // waits.
    void rgb2lab(av::gpu::Texture& tex);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
