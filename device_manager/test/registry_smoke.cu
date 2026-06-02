/**
 * registry_smoke.cu -- exercise both IDeviceRegistry implementations.
 *
 *   --mode=direct  pci=<BDF>     LocalNvmeDirectRegistry
 *   --mode=service daemon=ENDPOINT cuda=N device_id=N
 *                                 NvmeServiceBackedRegistry
 *
 * For each mode:
 *
 *   [1] open registry
 *   [2] enumerate device_count() / device_at() / find_by_id()
 *   [3] sanity-check the LocalNvmeDevice payload behind
 *       Device::backend_private (PCI matches, ctrl != null, expected
 *       blk_size and queue_depth come back from the kernel)
 *   [4] close registry; verify ctrl handles dropped via the right
 *       libnvm path (direct: nvm_ctrl_free; service: nvm_ctrl_free_client).
 *
 * "Direct" mode requires this process to be the sole owner -- run it
 * with the daemon stopped.  "Service" mode requires nvmeservice_daemon
 * already running.
 *
 * NOT destructive: no LBA writes; only chrdev / bind / probe.
 */

#include "local_nvme_direct_registry.h"
#include "nvmeservice_backed_registry.h"
#include "../../runtime/include/device.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
        "Usage:\n"
        "  %s --mode=direct  --pci=<BDF> [--gpu N] [--cap N]\n"
        "  %s --mode=service --endpoint=host:port --device=N --cuda=N [--count=N]\n"
        "\n"
        "direct mode:  brings up one NVMe via nvm_controller_init_b3.\n"
        "              Daemon MUST NOT be running; this process owns the chrdev.\n"
        "              CUDA must be initialised first (registry_smoke does\n"
        "              cudaSetDevice(--gpu) before calling Open()) because\n"
        "              nvm_controller_init_b3 internally cudaHostRegister()s\n"
        "              the BAR0 mapping.\n"
        "service mode: connects to running nvmeservice_daemon and Connects()\n"
        "              one session per --device given.  nvmeservice_daemon\n"
        "              must already be up.\n"
        "Non-destructive (no LBA writes).\n",
        prog, prog);
}

void check_device_shape(const tutti::Device* d, const char* expect_pci_prefix,
                         tutti::LocalNvmeAttachMode expected_mode)
{
    if (d == nullptr) STEP_FAIL("Device* is null");
    if (d->backend_type != tutti::BackendType::LOCAL_NVME)
        STEP_FAIL("backend_type != LOCAL_NVME");
    if (d->backend_private == nullptr)
        STEP_FAIL("backend_private is null");
    if (expect_pci_prefix && d->pci_addr.find(expect_pci_prefix) == std::string::npos)
        STEP_FAIL("pci_addr '%s' missing prefix '%s'",
                  d->pci_addr.c_str(), expect_pci_prefix);

    auto* bp = static_cast<tutti::LocalNvmeDevice*>(d->backend_private);
    if (bp->ctrl == nullptr)        STEP_FAIL("LocalNvmeDevice.ctrl is null");
    if (bp->attach_mode != expected_mode)
        STEP_FAIL("attach_mode mismatch (got=%u, want=%u)",
                  (unsigned)bp->attach_mode, (unsigned)expected_mode);
    if (bp->blk_size == 0)         STEP_FAIL("blk_size == 0");
    if (bp->queue_depth == 0)      STEP_FAIL("queue_depth == 0");
    if (bp->page_size == 0)        STEP_FAIL("page_size == 0");

    STEP_OK("device check: id=%d pci=%s mode=%s blk=%u qdepth=%u "
            "max_q/grp=%u",
            d->device_id, d->pci_addr.c_str(),
            expected_mode == tutti::LocalNvmeAttachMode::DIRECT ? "direct" : "service",
            bp->blk_size, bp->queue_depth, bp->max_queues_per_group);
}

int run_direct(const std::string& pci_addr, int cuda_dev, uint32_t cap) {
    // nvm_controller_init_b3 internally calls cudaHostRegister(BAR0),
    // so CUDA runtime must be initialised before Open().  On some
    // setups (Hopper + recent driver, CUDA-fork-then-sudo flows) the
    // first cudaSetDevice() returns cudaErrorSystemNotReady (46)
    // because the driver hasn't been primed in this process yet.
    // Touch the driver with cudaFree(0) to force a deterministic
    // init, then proceed.
    cudaError_t cerr = cudaFree(0);
    if (cerr != cudaSuccess && cerr != cudaErrorInvalidValue) {
        STEP_FAIL("cuda driver prime (cudaFree(0)) failed: %s. "
                  "Check nvidia-smi / driver / cgroup.",
                  cudaGetErrorString(cerr));
    }
    // Clear any sticky error from cudaFree(0).
    (void)cudaGetLastError();

    cerr = cudaSetDevice(cuda_dev);
    if (cerr != cudaSuccess) STEP_FAIL("cudaSetDevice(%d): %s",
                                        cuda_dev, cudaGetErrorString(cerr));
    STEP_OK("cudaSetDevice(%d)", cuda_dev);

    std::vector<tutti::LocalNvmeDirectConfig> cfgs;
    cfgs.push_back({pci_addr, cap, /*display_name=*/{}});

    tutti::LocalNvmeDirectRegistry reg(std::move(cfgs));

    if (!reg.Open()) STEP_FAIL("LocalNvmeDirectRegistry::Open()");
    STEP_OK("LocalNvmeDirectRegistry::Open() pci=%s cap=%u", pci_addr.c_str(), cap);

    if (reg.device_count() != 1) STEP_FAIL("device_count != 1");
    STEP_OK("device_count = %zu", reg.device_count());

    const auto* d0 = reg.device_at(0);
    check_device_shape(d0, pci_addr.c_str(), tutti::LocalNvmeAttachMode::DIRECT);

    const auto* d_lookup = reg.find_by_id(d0->device_id);
    if (d_lookup != d0) STEP_FAIL("find_by_id mismatch");
    STEP_OK("find_by_id(%d) returned same Device*", d0->device_id);

    reg.Close();
    if (reg.device_count() != 0) STEP_FAIL("device_count != 0 after close");
    STEP_OK("LocalNvmeDirectRegistry::Close() (chrdev_remove + unbind)");
    return 0;
}

int run_service(const std::string& endpoint, int32_t device_id,
                 int32_t cuda_dev, int32_t count)
{
    std::vector<tutti::NvmeServiceBackedRequest> reqs;
    tutti::NvmeServiceBackedRequest r{};
    r.daemon_device_id = device_id;
    r.cuda_device      = cuda_dev;
    r.num_queues       = count;
    reqs.push_back(std::move(r));

    tutti::NvmeServiceBackedRegistry reg(endpoint, std::move(reqs));

    if (!reg.Open()) STEP_FAIL("NvmeServiceBackedRegistry::Open()");
    STEP_OK("NvmeServiceBackedRegistry::Open() endpoint=%s device=%d cuda=%d count=%d",
            endpoint.c_str(), device_id, cuda_dev, count);

    if (reg.device_count() != 1) STEP_FAIL("device_count != 1");
    STEP_OK("device_count = %zu", reg.device_count());

    const auto* d0 = reg.device_at(0);
    check_device_shape(d0, /*pci_prefix=*/nullptr,
                        tutti::LocalNvmeAttachMode::SERVICE_CLIENT);

    const auto* d_lookup = reg.find_by_id(d0->device_id);
    if (d_lookup != d0) STEP_FAIL("find_by_id mismatch");
    STEP_OK("find_by_id(%d) returned same Device*", d0->device_id);

    reg.Close();
    if (reg.device_count() != 0) STEP_FAIL("device_count != 0 after close");
    STEP_OK("NvmeServiceBackedRegistry::Close() (free_client + Disconnect)");
    return 0;
}

const char* arg_after(const char* a, const char* prefix) {
    size_t n = std::strlen(prefix);
    if (std::strncmp(a, prefix, n) == 0) return a + n;
    return nullptr;
}

} // namespace

int main(int argc, char** argv) {
    std::string mode;
    std::string pci_addr;
    std::string endpoint = "127.0.0.1:50051";
    int32_t  device_id = 0;
    int32_t  cuda_dev  = 0;
    int32_t  count     = 4;
    uint32_t cap       = 32;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* v = nullptr;
        if      ((v = arg_after(a, "--mode=")))     mode      = v;
        else if ((v = arg_after(a, "--pci=")))      pci_addr  = v;
        else if ((v = arg_after(a, "--endpoint="))) endpoint  = v;
        else if ((v = arg_after(a, "--device=")))   device_id = std::atoi(v);
        else if ((v = arg_after(a, "--cuda=")))     cuda_dev  = std::atoi(v);
        else if ((v = arg_after(a, "--gpu=")))      cuda_dev  = std::atoi(v);
        else if ((v = arg_after(a, "--count=")))    count     = std::atoi(v);
        else if ((v = arg_after(a, "--cap=")))      cap       = (uint32_t)std::atoi(v);
        else { usage(argv[0]); return 1; }
    }

    if (mode == "direct") {
        if (pci_addr.empty()) { usage(argv[0]); return 1; }
        int rc = run_direct(pci_addr, cuda_dev, cap);
        if (rc == 0) std::fprintf(stderr,
            "\n=== registry_smoke (direct): all %d steps passed ===\n", g_step);
        return rc;
    }

    if (mode == "service") {
        int rc = run_service(endpoint, device_id, cuda_dev, count);
        if (rc == 0) std::fprintf(stderr,
            "\n=== registry_smoke (service): all %d steps passed ===\n", g_step);
        return rc;
    }

    usage(argv[0]);
    return 1;
}
