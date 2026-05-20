#include "av/depth_map/GaussianTable.hpp"

#include "av/gpu/Buffer.hpp"
#include "av/gpu/Device.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace av::depth_map {

namespace {

// Mirrors upstream's exp(-x²/2) layout (delta = 1.0).
std::pair<std::vector<int>, std::vector<float>>
build_table(int scales)
{
    if (scales <= 0 || scales > GaussianTable::kMaxScales)
        throw std::invalid_argument("GaussianTable: scales out of range");

    std::vector<int> offsets(GaussianTable::kMaxScales, 0);
    int sumSizes = 0;
    for (int s = 0; s < GaussianTable::kMaxScales; ++s) {
        offsets[s] = sumSizes;
        const int radius = s + 1;
        sumSizes += 2 * radius + 1;
    }
    if (sumSizes > GaussianTable::kMaxMemSize)
        throw std::runtime_error("GaussianTable: too little memory budget");

    std::vector<float> weights(sumSizes, 0.0f);
    for (int s = 0; s < scales; ++s) {
        const int   radius = s + 1;
        const int   size   = 2 * radius + 1;
        const float delta  = 1.0f;
        const float two_d2 = 2.0f * delta * delta;
        for (int idx = 0; idx < size; ++idx) {
            const int   x = idx - radius;
            weights[offsets[s] + idx] = std::exp(-static_cast<float>(x * x) / two_d2);
        }
    }
    return { std::move(offsets), std::move(weights) };
}

}  // namespace

struct GaussianTable::Impl {
    av::gpu::Buffer weights;
    av::gpu::Buffer offsets;

    Impl(av::gpu::Device& dev,
         std::span<const float> w_data,
         std::span<const int>   o_data)
        : weights(dev, w_data.size_bytes()),
          offsets(dev, o_data.size_bytes())
    {
        weights.set_label("gaussian.weights");
        offsets.set_label("gaussian.offsets");
        weights.upload(w_data);
        offsets.upload(o_data);
    }
};

GaussianTable::GaussianTable(av::gpu::Device& dev, int scales)
{
    auto [offsets, weights] = build_table(scales);
    impl_ = std::make_unique<Impl>(
        dev,
        std::span<const float>(weights.data(), weights.size()),
        std::span<const int>  (offsets.data(), offsets.size()));
}

GaussianTable::GaussianTable(GaussianTable&&) noexcept            = default;
GaussianTable& GaussianTable::operator=(GaussianTable&&) noexcept = default;
GaussianTable::~GaussianTable()                                   = default;

av::gpu::Buffer& GaussianTable::weights() noexcept { return impl_->weights; }
av::gpu::Buffer& GaussianTable::offsets() noexcept { return impl_->offsets; }

}  // namespace av::depth_map
