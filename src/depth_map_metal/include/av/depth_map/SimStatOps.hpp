#pragma once

#include <cstddef>
#include <memory>
#include <span>

namespace av::gpu { class Device; }

namespace av::depth_map {

class SimStatOps {
public:
    static constexpr std::size_t kSamples     = 64;
    static constexpr std::size_t kInPerCase   = kSamples * 3;
    static constexpr std::size_t kOutPerCase  = 4;

    explicit SimStatOps(av::gpu::Device& dev);

    SimStatOps(const SimStatOps&)            = delete;
    SimStatOps& operator=(const SimStatOps&) = delete;
    SimStatOps(SimStatOps&&) noexcept;
    SimStatOps& operator=(SimStatOps&&) noexcept;
    ~SimStatOps();

    void validate(std::span<const float> inputs,
                  std::span<float>        outputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace av::depth_map
