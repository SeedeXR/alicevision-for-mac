// utils.cpp — Apple Silicon implementation of upstream's
// `aliceVision/depthMap/cuda/host/utils.hpp` free functions.
//
// We expose a single-device "CUDA topology" matching the Metal
// runtime: one device (the default `MTLCreateSystemDefaultDevice`)
// is always selected at id 0. UMA on Apple Silicon means "device
// memory" and "system memory" are the same physical RAM, so the
// device-memory functions report system memory instead.
//
// Memory APIs used:
//   * `sysctlbyname("hw.memsize", ...)` for total physical RAM —
//     matches what System Info / Activity Monitor reports.
//   * `host_statistics64(HOST_VM_INFO64)` for free + inactive +
//     wired page counts → "available" memory. Inactive pages can
//     be reclaimed without paging, so they count as available;
//     this matches Activity Monitor's "Memory Pressure" math.
//
// All values are converted to MB to keep upstream's contract.
//
// UMA gotcha: on Intel + discrete GPU Macs there *is* a separate
// VRAM pool, queryable via Metal's `MTLDevice.recommendedMaxWorkingSetSize`.
// We deliberately report unified system RAM instead — the
// AliceVision depthMap pipeline expects the depthMap target box
// (max volume size) to fit in *something*, and on Apple Silicon
// the only "something" is system RAM. If you ever port to Intel
// Mac + dGPU, swap this for `recommendedMaxWorkingSetSize`.

#include "aliceVision/depthMap/cuda/host/utils.hpp"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace aliceVision {
namespace depthMap {

int listCudaDevices() {
    // On Apple Silicon we always have exactly one Metal device
    // (the integrated GPU). Log a brief description.
    std::fprintf(stderr,
        "[av] Detected GPU devices: 1 (Apple Silicon integrated GPU "
        "via Metal). Reported as CUDA device id 0.\n");
    return 1;
}

int getCudaDeviceId() {
    // We only ever have one device.
    return 0;
}

void setCudaDeviceId(int cudaDeviceId) {
    if (cudaDeviceId != 0) {
        std::fprintf(stderr,
            "[av] setCudaDeviceId(%d): ignored — only device 0 "
            "(the integrated GPU) exists on this Mac.\n",
            cudaDeviceId);
    }
    // Otherwise: no-op (already the active device).
}

bool testCudaDeviceId(int cudaDeviceId) {
    return cudaDeviceId == 0;
}

namespace {

// Total physical RAM in bytes via sysctl. Safer than
// `NSProcessInfo.physicalMemory` because it returns the raw
// hardware total, not a per-process cap.
std::uint64_t total_ram_bytes() {
    std::uint64_t mem = 0;
    std::size_t   sz  = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &sz, nullptr, 0) != 0) {
        return 0;
    }
    return mem;
}

// Available memory (in bytes) via host_statistics64. We count
// free + inactive + speculative pages — inactive pages can be
// reclaimed without paging, so they're effectively available.
// Mirrors Activity Monitor's reporting.
std::uint64_t available_ram_bytes() {
    mach_port_t host = mach_host_self();
    vm_size_t   page_size = 0;
    if (host_page_size(host, &page_size) != KERN_SUCCESS) {
        return 0;
    }
    vm_statistics64_data_t vmstat{};
    mach_msg_type_number_t  count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vmstat),
                          &count) != KERN_SUCCESS) {
        return 0;
    }
    const std::uint64_t available_pages =
        std::uint64_t(vmstat.free_count) +
        std::uint64_t(vmstat.inactive_count) +
        std::uint64_t(vmstat.speculative_count);
    return available_pages * std::uint64_t(page_size);
}

}  // namespace

void logDeviceMemoryInfo() {
    const std::uint64_t total = total_ram_bytes();
    const std::uint64_t avail = available_ram_bytes();
    const std::uint64_t used  = (total > avail) ? (total - avail) : 0;

    const double oneMB = 1024.0 * 1024.0;
    std::fprintf(stderr,
        "[av] Device memory (UMA, device id 0):\n"
        "       - used     : %.1f MB\n"
        "       - available: %.1f MB\n"
        "       - total    : %.1f MB\n",
        double(used)  / oneMB,
        double(avail) / oneMB,
        double(total) / oneMB);
}

void getDeviceMemoryInfo(double& availableMB, double& usedMB, double& totalMB) {
    const std::uint64_t total = total_ram_bytes();
    const std::uint64_t avail = available_ram_bytes();
    const std::uint64_t used  = (total > avail) ? (total - avail) : 0;

    const double oneMB = 1024.0 * 1024.0;
    totalMB     = double(total) / oneMB;
    availableMB = double(avail) / oneMB;
    usedMB      = double(used)  / oneMB;
}

}  // namespace depthMap
}  // namespace aliceVision
