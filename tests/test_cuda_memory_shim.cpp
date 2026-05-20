// test_cuda_memory_shim.cpp — smoke test for the
// CudaDeviceMemoryPitched / CudaHostMemoryHeap / CudaSize<N>
// shim that backs upstream's memory.hpp on Apple Silicon.
//
// We don't go through upstream's path here; we include the
// shim directly. Verifies that the audited surface
// (allocate / getSize / getPitch / getBytesPadded /
// getBytesUnpadded / getBuffer / copyFrom) does what we
// expect on Apple's UMA backing.

#include "av/gpu/Device.hpp"
#include "av/gpu/Errors.hpp"

// Pull in the shim directly. (In production builds, the
// upstream-depthMap include path puts this dir BEFORE
// upstream's src/, so `#include <aliceVision/depthMap/cuda/host/memory.hpp>`
// from upstream code resolves here. For this test we include
// by path so we don't have to set up that include order.)
#include "../cmake/shims/aliceVision-includes/aliceVision/depthMap/cuda/host/memory.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

int main() try {
    using namespace av::gpu;
    using namespace aliceVision::depthMap;

    auto dev = Device::default_device();
    dev.load_library({});
    std::printf("[info] device : %s\n", dev.name().c_str());

    set_adapter_device(dev);

    int failed = 0;

    // ---- CudaSize<N> sanity ----
    {
        CudaSize<2> s2(13, 17);
        if (s2.x() != 13 || s2.y() != 17 || s2[0] != 13 || s2[1] != 17) {
            std::fprintf(stderr, "FAIL: CudaSize<2>(13, 17)\n"); ++failed;
        }
        CudaSize<3> s3(5, 7, 9);
        if (s3.x() != 5 || s3.y() != 7 || s3.z() != 9) {
            std::fprintf(stderr, "FAIL: CudaSize<3>\n"); ++failed;
        }
        CudaSize<2> s2b(13, 17);
        if (s2 != s2b) {
            std::fprintf(stderr, "FAIL: CudaSize<2> operator!= (false negative)\n");
            ++failed;
        }
        CudaSize<2> s2c(0, 0);
        if (s2 == s2c) {
            std::fprintf(stderr, "FAIL: CudaSize<2> operator== (false positive)\n");
            ++failed;
        }
    }
    std::printf("[size] ok\n");

    // ---- CudaDeviceMemoryPitched + CudaHostMemoryHeap roundtrip ----
    {
        const CudaSize<2> dim(64, 48);
        CudaHostMemoryHeap<float2, 2>          host(dim);
        CudaDeviceMemoryPitched<float2, 2>     dev_mem(dim);

        if (host.getSize() != dim || dev_mem.getSize() != dim) {
            std::fprintf(stderr, "FAIL: getSize() roundtrip\n"); ++failed;
        }
        if (host.getPitch() != 64 * sizeof(float2)) {
            std::fprintf(stderr,
                "FAIL: host getPitch %zu (expected %zu)\n",
                host.getPitch(), 64 * sizeof(float2));
            ++failed;
        }
        if (host.getBytesPadded() != 64 * 48 * sizeof(float2)) {
            std::fprintf(stderr, "FAIL: host getBytesPadded\n"); ++failed;
        }
        if (host.getBytesUnpadded() != host.getBytesPadded()) {
            std::fprintf(stderr,
                "FAIL: padded != unpadded (UMA expectation)\n");
            ++failed;
        }

        // Populate the host buffer with a unique-per-pixel fingerprint.
        auto* h = host.getBuffer();
        for (std::size_t k = 0; k < dim.x() * dim.y(); ++k) {
            h[k].x = float(k);
            h[k].y = float(k) * 0.5f;
        }

        // Device.copyFrom(host) — should be a fast UMA memcpy.
        dev_mem.copyFrom(host);

        // Read back via the device's getBuffer().
        auto* d = dev_mem.getBuffer();
        int bad = 0;
        for (std::size_t k = 0; k < dim.x() * dim.y(); ++k) {
            if (d[k].x != float(k) || d[k].y != float(k) * 0.5f) ++bad;
        }
        if (bad) {
            std::fprintf(stderr, "FAIL: host→device roundtrip (%d bad)\n", bad);
            ++failed;
        }

        // Now go device → host_new
        CudaHostMemoryHeap<float2, 2> host_back(dim);
        host_back.copyFrom(dev_mem);
        auto* hb = host_back.getBuffer();
        bad = 0;
        for (std::size_t k = 0; k < dim.x() * dim.y(); ++k) {
            if (hb[k].x != float(k) || hb[k].y != float(k) * 0.5f) ++bad;
        }
        if (bad) {
            std::fprintf(stderr, "FAIL: device→host (%d bad)\n", bad);
            ++failed;
        }

        // Device → device.
        CudaDeviceMemoryPitched<float2, 2> dev_copy(dim);
        dev_copy.copyFrom(dev_mem);
        auto* dc = dev_copy.getBuffer();
        bad = 0;
        for (std::size_t k = 0; k < dim.x() * dim.y(); ++k) {
            if (dc[k].x != float(k) || dc[k].y != float(k) * 0.5f) ++bad;
        }
        if (bad) {
            std::fprintf(stderr, "FAIL: device→device (%d bad)\n", bad);
            ++failed;
        }
    }
    std::printf("[mem ] roundtrips ok\n");

    // ---- 3D dim sanity ----
    {
        const CudaSize<3> vd(16, 12, 8);
        CudaDeviceMemoryPitched<unsigned char, 3> vol(vd);
        if (vol.getUnitsTotal() != 16 * 12 * 8) {
            std::fprintf(stderr, "FAIL: 3D getUnitsTotal\n"); ++failed;
        }
        if (vol.getBytesPadded() != 16 * 12 * 8) {
            std::fprintf(stderr, "FAIL: 3D getBytesPadded\n"); ++failed;
        }
    }
    std::printf("[3d  ] ok\n");

    // ---- gpu_buffer() adapter extension ----
    {
        const CudaSize<2> dim(32, 24);
        CudaDeviceMemoryPitched<float, 2> dev_mem(dim);
        av::gpu::Buffer& buf = dev_mem.gpu_buffer();
        if (buf.size_bytes() < dim.x() * dim.y() * sizeof(float)) {
            std::fprintf(stderr, "FAIL: gpu_buffer().size_bytes() too small\n");
            ++failed;
        }
    }
    std::printf("[ext ] gpu_buffer() ok\n");

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
