#pragma once

// ColorOps — host driver for the color.h validation kernel.

#include <cstddef>
#include <memory>
#include <span>

namespace av::gpu { class Device; }

namespace av::depth_map {

class ColorOps {
public:
    static constexpr std::size_t kInPerCase  = 12;
    static constexpr std::size_t kOutPerCase = 15;

    explicit ColorOps(av::gpu::Device& dev);

    ColorOps(const ColorOps&)            = delete;
    ColorOps& operator=(const ColorOps&) = delete;
    ColorOps(ColorOps&&) noexcept;
    ColorOps& operator=(ColorOps&&) noexcept;
    ~ColorOps();

    void validate(std::span<const float> inputs,
                  std::span<float>        outputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
