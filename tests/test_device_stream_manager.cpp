// test_device_stream_manager.cpp — validation of
// `DeviceStreamManager` and the new `av::gpu::Queue`.
//
// Coverage:
//   * nb_streams() returns the configured pool size.
//   * get_stream(i) for i in [0, N) returns N distinct queues
//     (no two return the same `raw_command_queue()` pointer).
//   * Modular access: get_stream(N + k) returns the same queue
//     as get_stream(k). Same for negative indices.
//   * Functional check: dispatch a small kernel on two different
//     streams in parallel; wait_stream blocks until that one's
//     work is done. Validate the kernel result is correct.
//   * wait_all() drains all queues.

#include "av/depth_map/DeviceStreamManager.hpp"
#include "av/gpu/Buffer.hpp"
#include "av/gpu/CommandBuffer.hpp"
#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"
#include "av/gpu/Pipeline.hpp"
#include "av/gpu/Queue.hpp"

#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

int main() try {
    using namespace av::gpu;
    using namespace av::depth_map;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    constexpr int kN = 4;
    DeviceStreamManager mgr(dev, kN);

    int failed = 0;

    // ---- nb_streams + distinctness ----
    if (mgr.nb_streams() != kN) {
        std::fprintf(stderr, "FAIL: nb_streams = %d (expected %d)\n",
                     mgr.nb_streams(), kN);
        ++failed;
    }
    std::set<void*> seen;
    for (int i = 0; i < kN; ++i) {
        void* p = static_cast<void*>(mgr.get_stream(i).raw_command_queue());
        if (!seen.insert(p).second) {
            std::fprintf(stderr, "FAIL: queue %d duplicates a previous queue\n", i);
            ++failed;
        }
    }
    std::printf("[mgr ] %d distinct queues\n", kN);

    // ---- modular indexing ----
    for (int extra : { 0, 1, kN, kN + 2, 2 * kN + 3 }) {
        const int expected_slot = ((extra % kN) + kN) % kN;
        auto* got =      mgr.get_stream(extra).raw_command_queue();
        auto* expected = mgr.get_stream(expected_slot).raw_command_queue();
        if (got != expected) {
            std::fprintf(stderr,
                "FAIL: get_stream(%d) != get_stream(%d)\n",
                extra, expected_slot);
            ++failed;
        }
    }
    for (int neg : { -1, -kN, -(2 * kN + 3) }) {
        const int expected_slot = ((neg % kN) + kN) % kN;
        auto* got =      mgr.get_stream(neg).raw_command_queue();
        auto* expected = mgr.get_stream(expected_slot).raw_command_queue();
        if (got != expected) {
            std::fprintf(stderr,
                "FAIL: get_stream(%d) != get_stream(%d)\n",
                neg, expected_slot);
            ++failed;
        }
    }
    std::printf("[mod ] modular indexing ok (positive + negative)\n");

    // ---- dispatch a kernel on two distinct streams ----
    // Use the SAXPY hello kernel (av_hello_saxpy): y[i] = a*x[i] + y[i].
    // Each stream owns its own y buffer (no aliasing); they share x.
    auto saxpy = dev.make_pipeline("av_hello_saxpy");

    constexpr std::uint32_t N_elem = 4096;
    Buffer x_buf (dev, N_elem * sizeof(float));
    Buffer y_buf0(dev, N_elem * sizeof(float));
    Buffer y_buf1(dev, N_elem * sizeof(float));

    {
        auto x  = x_buf .as_span<float>();
        auto y0 = y_buf0.as_span<float>();
        auto y1 = y_buf1.as_span<float>();
        for (std::uint32_t i = 0; i < N_elem; ++i) {
            x[i]  = float(i);
            y0[i] = float(i) * 0.5f;
            y1[i] = float(i) * 0.5f;
        }
    }

    struct SaxpyParams { std::uint32_t count; float a; };
    SaxpyParams p0{ N_elem, 2.0f };
    SaxpyParams p1{ N_elem, 3.0f };

    {
        CommandBuffer cb(mgr.get_stream(0));
        cb.set_label("saxpy.q0")
          .set_pipeline(saxpy)
          .set_buffer  (0, y_buf0)
          .set_buffer  (1, x_buf)
          .set_bytes   (2, &p0, sizeof(p0))
          .dispatch_1d(saxpy, N_elem);
        cb.commit_async();
    }
    {
        CommandBuffer cb(mgr.get_stream(1));
        cb.set_label("saxpy.q1")
          .set_pipeline(saxpy)
          .set_buffer  (0, y_buf1)
          .set_buffer  (1, x_buf)
          .set_bytes   (2, &p1, sizeof(p1))
          .dispatch_1d(saxpy, N_elem);
        cb.commit_async();
    }

    // Drain queue 0 and verify its result.
    mgr.wait_stream(0);
    {
        const auto* y = static_cast<const float*>(y_buf0.data());
        int bad = 0;
        for (std::uint32_t i = 0; i < N_elem; ++i) {
            const float expected = 2.0f * float(i) + float(i) * 0.5f;
            if (y[i] != expected) {
                if (bad < 3) std::fprintf(stderr,
                    "  q0 i=%u y=%g expected=%g\n", i, double(y[i]), double(expected));
                ++bad;
            }
        }
        if (bad) {
            std::fprintf(stderr, "FAIL: queue 0 result wrong (%d/%u)\n", bad, N_elem);
            failed += 1;
        }
    }

    // Drain queue 1 and verify.
    mgr.wait_stream(1);
    {
        const auto* y = static_cast<const float*>(y_buf1.data());
        int bad = 0;
        for (std::uint32_t i = 0; i < N_elem; ++i) {
            const float expected = 3.0f * float(i) + float(i) * 0.5f;
            if (y[i] != expected) {
                if (bad < 3) std::fprintf(stderr,
                    "  q1 i=%u y=%g expected=%g\n", i, double(y[i]), double(expected));
                ++bad;
            }
        }
        if (bad) {
            std::fprintf(stderr, "FAIL: queue 1 result wrong (%d/%u)\n", bad, N_elem);
            failed += 1;
        }
    }
    std::printf("[disp] two parallel SAXPYs, both correct\n");

    // wait_all should drain queues 2 and 3 too (which have no
    // work) without complaint.
    mgr.wait_all();
    std::printf("[all ] wait_all ok\n");

    if (failed) {
        std::fprintf(stderr, "FAIL: %d issues\n", failed);
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
