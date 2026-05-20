// test_metal_hello.cpp — end-to-end smoke test for the av::gpu skeleton.
//
// Validates:
//   1. Default Metal device acquisition.
//   2. Loading the build-time-produced default.metallib next to the
//      running executable.
//   3. Building a compute pipeline.
//   4. Allocating Shared MTL buffers and direct-UMA-writing from CPU.
//   5. Dispatching a 1D kernel.
//   6. Reading results back through the same Shared pointer.
//   7. SIMD-group reduction (simd_sum).
//
// Exit codes:
//   0  all checks passed
//   1  numerical mismatch
//   2  setup / API error (rethrows as GpuError; non-zero exit)

#include "av/gpu/Device.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Errors.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr std::uint32_t N = 1u << 16;  // 65 536 floats
constexpr float         A = 2.5f;

int test_saxpy(av::gpu::Device& dev) {
    using namespace av::gpu;

    Buffer y(dev, N * sizeof(float));
    Buffer x(dev, N * sizeof(float));
    y.set_label("hello.saxpy.y");
    x.set_label("hello.saxpy.x");

    auto y_span = y.as_span<float>();
    auto x_span = x.as_span<float>();
    for (std::uint32_t i = 0; i < N; ++i) {
        x_span[i] = static_cast<float>(i);
        y_span[i] = 1.0f;
    }

    struct SaxpyParams { std::uint32_t count; float a; };
    SaxpyParams params{ N, A };

    auto pipe = dev.make_pipeline("av_hello_saxpy");

    CommandBuffer cb(dev);
    cb.set_label("hello.saxpy.cb")
      .set_pipeline(pipe)
      .set_buffer(0, y)
      .set_buffer(1, x)
      .set_bytes (2, &params, sizeof(params))
      .dispatch_1d(pipe, N);
    cb.commit_and_wait();

    // Validate: y[i] = A * i + 1
    int failures = 0;
    for (std::uint32_t i = 0; i < N; ++i) {
        const float expected = A * static_cast<float>(i) + 1.0f;
        if (std::abs(y_span[i] - expected) > 1e-3f * std::abs(expected) + 1e-5f) {
            if (failures < 5) {
                std::fprintf(stderr,
                    "saxpy mismatch @%u : got %g, want %g\n",
                    i,
                    static_cast<double>(y_span[i]),
                    static_cast<double>(expected));
            }
            ++failures;
        }
    }
    if (failures) {
        std::fprintf(stderr, "saxpy: %d / %u mismatches\n", failures, N);
        return 1;
    }
    std::printf("[ok]  saxpy        N=%u, a=%g\n", N, static_cast<double>(A));
    return 0;
}

int test_simdsum(av::gpu::Device& dev) {
    using namespace av::gpu;

    constexpr std::uint32_t SIMD_WIDTH = 32;
    constexpr std::uint32_t COUNT      = SIMD_WIDTH * 1024;

    Buffer x(dev, COUNT * sizeof(float));
    Buffer partials(dev, (COUNT / SIMD_WIDTH) * sizeof(float));

    auto x_span = x.as_span<float>();
    for (std::uint32_t i = 0; i < COUNT; ++i) {
        x_span[i] = 1.0f;     // expect partial = 32 * 1.0 = 32; total = COUNT
    }

    auto pipe = dev.make_pipeline("av_hello_simdsum");

    // Dispatch with threadgroup == SIMD width. With this layout
    // each threadgroup is exactly one SIMD group, so simd_sum
    // produces the correct per-group sum.
    if (pipe.thread_execution_width() != SIMD_WIDTH) {
        std::fprintf(stderr,
            "warn: thread_execution_width=%zu (expected %u)\n",
            pipe.thread_execution_width(), SIMD_WIDTH);
    }

    CommandBuffer cb(dev);
    std::uint32_t count_param = COUNT;
    cb.set_label("hello.simdsum.cb")
      .set_pipeline(pipe)
      .set_buffer(0, x)
      .set_buffer(1, partials)
      .set_bytes (2, &count_param, sizeof(count_param))
      .dispatch({ COUNT, 1, 1 }, { SIMD_WIDTH, 1, 1 });
    cb.commit_and_wait();

    auto p_span = partials.as_span<const float>();
    double total = 0.0;
    for (auto v : p_span) total += static_cast<double>(v);

    const double expected = static_cast<double>(COUNT);
    if (std::abs(total - expected) > 1e-3) {
        std::fprintf(stderr, "simdsum total mismatch: got %g, want %g\n",
                     total, expected);
        return 1;
    }
    std::printf("[ok]  simdsum      count=%u, sum=%g\n", COUNT, total);
    return 0;
}

}  // namespace

int main() try {
    auto dev = av::gpu::Device::default_device();
    std::printf("[info] device       : %s\n", dev.name().c_str());
    std::printf("[info] unified mem  : %s\n",
                dev.has_unified_memory() ? "yes" : "no");
    std::printf("[info] working set  : %.1f GiB\n",
                static_cast<double>(dev.recommended_working_set())
                    / (1024.0 * 1024.0 * 1024.0));

    dev.load_library({});   // load default.metallib next to the executable

    int rc = 0;
    rc |= test_saxpy   (dev);
    rc |= test_simdsum (dev);
    if (rc != 0) {
        std::fprintf(stderr, "FAIL\n");
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
