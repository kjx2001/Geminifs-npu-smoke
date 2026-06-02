/**
 * memory_smoke.cu -- exercise HostDeviceMemorySubsystem end-to-end
 * over one OR multiple NVMe controllers.
 *
 * Single-process: brings up libnvm via LocalNvmeDirectRegistry (which
 * is what real R5+ callers will do).
 *
 * Single-NVMe invocation:
 *   sudo ./memory_smoke --gpu 0 --cap 32 0000:08:00.0
 *
 * Multi-NVMe invocation (target_devices spans every device, exercising
 * the per-(region, Device*) DMA mapping table):
 *   sudo ./memory_smoke --gpu 0 --cap 32 \
 *        0000:4b:00.0 0000:57:00.0 0000:63:00.0
 *
 * The smoke validates:
 *   - cuda driver prime + cudaSetDevice
 *   - registry brings up N controllers
 *   - HostDeviceMemorySubsystem ctor is parameter-less
 *   - register_tensor() drives DMA mapping per spec.target_devices
 *     for EVERY device in the registry
 *   - per-(region, Device*) DMA tracking + idempotent re-call
 *   - set_descriptor_format() acceptance
 *   - descriptor_slice() returns false (R7 stub)
 *   - free / unregister cleanly tears down all (region, device) maps
 *
 * Test contract:
 *   - sole owner of every NVMe (NVMeService daemon MUST NOT be running)
 *   - CUDA-visible GPU at --gpu N
 *
 * NOT destructive: no LBA writes; only DMA mapping.
 */

#include "host_device_memory_subsystem.h"
#include "cuda_helpers.cuh"

#include "../../device_manager/include/local_nvme_direct_registry.h"
#include "../../runtime/include/device.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace {

int g_step = 0;

#define STEP_OK(fmt, ...) do { \
    ++g_step; \
    std::fprintf(stderr, "[ OK ] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__); \
} while (0)

#define STEP_FAIL(fmt, ...) do { \
    ++g_step; \
    std::fprintf(stderr, "[FAIL] step=%-2d  " fmt "\n", g_step, ##__VA_ARGS__); \
    std::_Exit(2); \
} while (0)

void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [--gpu N] [--cap N] <PCI_BDF> [<PCI_BDF>...]\n"
        "  e.g. (single):  %s --gpu 0 --cap 32 0000:08:00.0\n"
        "  e.g. (multi):   %s --gpu 0 --cap 32 0000:4b:00.0 0000:57:00.0\n"
        "Exercises tutti::HostDeviceMemorySubsystem against one or\n"
        "multiple libnvm controllers via LocalNvmeDirectRegistry.\n"
        "Daemon MUST NOT be running.  Non-destructive (no LBA writes).\n",
        prog, prog, prog);
}

void prime_cuda(int cuda_dev) {
    cudaError_t cerr = cudaFree(0);
    if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
        STEP_FAIL("cuda driver prime (cudaFree(0)) failed: %s",
                  cudaGetErrorString(cerr));
    }
    (void)cudaGetLastError();

    CUDA_OK(cudaSetDevice(cuda_dev));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);
}

int run(int cuda_dev, const std::vector<std::string>& pci_addrs, uint32_t cap) {
    prime_cuda(cuda_dev);

    // [1] Bring up every requested controller via the direct registry.
    std::vector<tutti::LocalNvmeDirectConfig> cfgs;
    cfgs.reserve(pci_addrs.size());
    for (const auto& bdf : pci_addrs) {
        cfgs.push_back({bdf, cap, /*display_name=*/{}});
    }

    tutti::LocalNvmeDirectRegistry reg(std::move(cfgs));
    if (!reg.Open()) STEP_FAIL("LocalNvmeDirectRegistry::Open() (n=%zu)",
                                pci_addrs.size());
    if (reg.device_count() != pci_addrs.size())
        STEP_FAIL("device_count=%zu != requested=%zu",
                  reg.device_count(), pci_addrs.size());

    std::vector<const tutti::Device*> devices;
    devices.reserve(pci_addrs.size());
    for (size_t i = 0; i < pci_addrs.size(); ++i) {
        devices.push_back(reg.device_at(i));
    }
    STEP_OK("registry up: n=%zu", devices.size());

    // [2] Build the memory subsystem (no ctrl in ctor).
    tutti::HostDeviceMemorySubsystem mem;
    mem.set_descriptor_format(tutti::DescriptorFormat::PRP);
    if (mem.descriptor_format() != tutti::DescriptorFormat::PRP)
        STEP_FAIL("descriptor_format() != PRP after set");
    STEP_OK("HostDeviceMemorySubsystem instantiated, format=PRP");

    // [3] allocate HOST
    auto* r_host = mem.allocate_host(64 * 1024, tutti::MemoryKind::HOST);
    if (r_host == nullptr) STEP_FAIL("allocate_host(HOST)");
    STEP_OK("allocate_host(HOST) id=%lu host=%p size=%lu",
            (unsigned long)r_host->region_id, r_host->host_ptr,
            (unsigned long)r_host->size);

    // [4] allocate PINNED_HOST
    auto* r_pin = mem.allocate_host(4096, tutti::MemoryKind::PINNED_HOST);
    if (r_pin == nullptr) STEP_FAIL("allocate_host(PINNED_HOST)");
    STEP_OK("allocate_host(PINNED_HOST) id=%lu host=%p",
            (unsigned long)r_pin->region_id, r_pin->host_ptr);

    // [5] allocate DEVICE
    auto* r_dev = mem.allocate_device(1 << 20, tutti::MemoryKind::DEVICE,
                                       cuda_dev);
    if (r_dev == nullptr) STEP_FAIL("allocate_device(DEVICE)");
    STEP_OK("allocate_device(DEVICE) id=%lu dev=%p size=1MiB on cuda=%d",
            (unsigned long)r_dev->region_id, r_dev->device_ptr,
            r_dev->cuda_device);

    // [6] register_tensor on the device region; spec.target_devices
    //     contains EVERY device, so memory should DMA-map this single
    //     buffer to all N controllers.
    {
        tutti::TensorRegistrationSpec spec{};
        spec.ptr             = r_dev->device_ptr;
        spec.size            = r_dev->size;
        spec.target_devices  = devices;     // <-- all of them
        auto* same = mem.register_tensor(spec);
        if (same != r_dev) STEP_FAIL("register_tensor: returned region != r_dev "
                                      "(got=%p, want=%p)", (void*)same, (void*)r_dev);
    }
    // Per-device verification: every device must have its own mapping.
    for (size_t i = 0; i < devices.size(); ++i) {
        std::size_t n=0, ps=0; uint64_t first=0;
        if (!mem.query_nvme_mapping(r_dev, devices[i], &n, &ps, &first))
            STEP_FAIL("query_nvme_mapping(device region, dev[%zu]) -- not mapped", i);
        if (n == 0 || ps == 0 || first == 0)
            STEP_FAIL("device mapping empty on dev[%zu]: n=%zu ps=%zu first=0x%lx",
                      i, n, ps, (unsigned long)first);
        STEP_OK("register_tensor(device) on dev[%zu] (id=%d) "
                "ioaddrs=%zu page=%zu first=0x%lx",
                i, devices[i]->device_id, n, ps, (unsigned long)first);
    }

    // [7] register_tensor idempotency: calling again with the same set
    //     should be a no-op for already-mapped (region, device) pairs.
    {
        tutti::TensorRegistrationSpec spec{};
        spec.ptr             = r_dev->device_ptr;
        spec.size            = r_dev->size;
        spec.target_devices  = devices;
        if (mem.register_tensor(spec) != r_dev)
            STEP_FAIL("register_tensor idempotent re-call");
    }
    STEP_OK("register_tensor(device) idempotent re-call ok across %zu devices",
            devices.size());

    // [8] register_external(APP_MANAGED) on a caller-cudaMalloc'd buffer,
    //     then register_tensor against ALL devices to drive DMA mapping.
    void* app_buf = nullptr;
    CUDA_OK(cudaMalloc(&app_buf, 256 * 1024));
    tutti::ExternalMemorySpec espec{};
    espec.source = tutti::ExternalMemorySource::APP_MANAGED;
    auto* r_ext = mem.register_external(/*host=*/nullptr, app_buf,
                                          256 * 1024, espec);
    if (r_ext == nullptr) STEP_FAIL("register_external(APP_MANAGED)");
    {
        tutti::TensorRegistrationSpec spec{};
        spec.ptr             = app_buf;
        spec.size            = 256 * 1024;
        spec.target_devices  = devices;
        if (mem.register_tensor(spec) != r_ext)
            STEP_FAIL("register_tensor(external) returned wrong region");
    }
    for (size_t i = 0; i < devices.size(); ++i) {
        std::size_t n=0, ps=0; uint64_t first=0;
        if (!mem.query_nvme_mapping(r_ext, devices[i], &n, &ps, &first))
            STEP_FAIL("query_nvme_mapping(external, dev[%zu]) -- not mapped", i);
    }
    STEP_OK("register_external(APP_MANAGED) id=%lu dev=%p mapped to %zu devices",
            (unsigned long)r_ext->region_id, app_buf, devices.size());

    // [9] descriptor_slice is a v0.1 stub.
    {
        tutti::AddressDescriptor d{};
        std::size_t cnt = 1;
        bool ok = mem.descriptor_slice(r_dev, devices[0], 0, 4096, &d, &cnt);
        if (ok) STEP_FAIL("descriptor_slice: expected stub-false, got true");
    }
    STEP_OK("descriptor_slice() returns Unimplemented (R7 stub)");

    // [10] lookup
    {
        tutti::MemoryLookupKey k{};
        k.by  = tutti::MemoryLookupKey::By::HOST_PTR;
        k.ptr = r_host->host_ptr;
        if (mem.lookup(k) != r_host) STEP_FAIL("lookup HOST_PTR");
        k.by  = tutti::MemoryLookupKey::By::DEVICE_PTR;
        k.ptr = r_dev->device_ptr;
        if (mem.lookup(k) != r_dev) STEP_FAIL("lookup DEVICE_PTR");
        k.by        = tutti::MemoryLookupKey::By::REGION_ID;
        k.region_id = r_ext->region_id;
        if (mem.lookup(k) != r_ext) STEP_FAIL("lookup REGION_ID");
    }
    STEP_OK("lookup HOST_PTR / DEVICE_PTR / REGION_ID ok (regions=%zu)",
            mem.region_count());

    // [11] release
    mem.unregister(r_ext);
    cudaFree(app_buf);
    mem.free(r_dev);
    mem.free(r_pin);
    mem.free(r_host);
    if (mem.region_count() != 0)
        STEP_FAIL("region_count != 0 after release: %zu", mem.region_count());
    STEP_OK("free / unregister cleared the table (drops %zu*N DMA maps)",
            (size_t)2);   // 2 regions had per-device maps (r_dev + r_ext)

    // [12] tear down the registry (closes ctrls via nvm_ctrl_free).
    reg.Close();
    STEP_OK("registry closed (chrdev_remove + unbind, n=%zu)", devices.size());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    int      cuda_dev = 0;
    uint32_t cap      = 32;
    int      argi     = 1;

    while (argi < argc && argv[argi][0] == '-' && argv[argi][1] == '-') {
        const char* a = argv[argi];
        if (std::strcmp(a, "--gpu") == 0 && argi + 1 < argc) {
            cuda_dev = std::atoi(argv[++argi]);
            ++argi;
        } else if (std::strcmp(a, "--cap") == 0 && argi + 1 < argc) {
            cap = (uint32_t)std::atoi(argv[++argi]);
            ++argi;
        } else { usage(argv[0]); return 1; }
    }
    if (argi >= argc) { usage(argv[0]); return 1; }

    std::vector<std::string> pci_addrs;
    while (argi < argc) {
        pci_addrs.emplace_back(argv[argi++]);
    }

    int rc = run(cuda_dev, pci_addrs, cap);
    if (rc == 0) {
        std::fprintf(stderr,
            "\n=== memory_smoke (n=%zu): all %d steps passed ===\n",
            pci_addrs.size(), g_step);
    }
    return rc;
}
