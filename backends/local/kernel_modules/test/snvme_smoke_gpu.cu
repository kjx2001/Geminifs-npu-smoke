/*
 * snvme_smoke_gpu.cu -- SNVMe GPU-path sanity test.
 *
 * Same idea as snvme_smoke.c (sibling file), but additionally exercises the
 * paths that depend on the proprietary NVIDIA driver:
 *
 *     NVM_MAP_DEVICE_QUEUE_MEMORY   <- nvfs_nvidia_p2p_get_pages, queue ring
 *     NVM_MAP_DEVICE_MEMORY         <- nvfs_nvidia_p2p_get_pages, data buffer
 *     NVM_UNMAP_DEVICE_QUEUE_MEMORY
 *     NVM_UNMAP_DEVICE_MEMORY
 *
 * It also (in --bind mode) drives s_nvme_probe() with GPU-resident SQ/CQ
 * rings so the in-kernel branch
 *
 *     if (ctrl->ioq_num == ctrl->ioq_map_num && ctrl->use_sreg) ...
 *
 * is genuinely traversed using nvidia_p2p IO addresses, not host pages.
 *
 * Two test modes, mirroring snvme_smoke.c:
 *   default ("UAPI smoke")  -- exercise every UAPI entry that does not
 *                              trigger a probe. Safe.
 *   --bind  ("full bring-up") -- additionally bind, NVM_GET_DEV_INFO,
 *                              pread() the resulting block device,
 *                              unbind. DESTRUCTIVE.
 *
 * Pre-conditions:
 *   - snvme-core.ko + snvme.ko loaded.
 *   - The proprietary NVIDIA driver is loaded AND nvfs_nvidia_p2p_init()
 *     succeeded at module-load time (the snvme module logs a fatal error
 *     and refuses to load otherwise -- see pci.c:nvme_init).
 *   - At least one GPU visible to CUDA.
 *   - You ran as root (PCI bind requires CAP_SYS_ADMIN).
 *
 * Build:        make           (the parent Makefile compiles this with nvcc)
 * Invoke:       sudo ./snvme_smoke_gpu [--bind] [--gpu N] <PCI_BDF>
 *
 * Exit codes:
 *   0  -- all steps passed; SNVMe GPU paths are healthy on this kernel.
 *   1  -- usage error.
 *   2  -- a smoke step failed; see stderr for which one.
 */

#include <cuda_runtime.h>

#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" {
#include "ioctl.h"
}

/* GPU page size used by the kernel module
 * (snvme/map.c:GPU_PAGE_SHIFT=16). Buffers passed to NVM_MAP_DEVICE_*
 * MUST be aligned to and sized in multiples of this. */
static constexpr size_t GPU_PAGE_SIZE = 1ULL << 16;   /* 64 KiB */

/* ------------------------------------------------------------------ */
/* Logging helpers                                                    */
/* ------------------------------------------------------------------ */

static int g_step = 0;

static void step_ok(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[ OK ] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void step_warn(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[WARN] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

static void __attribute__((noreturn)) step_fail(int err, const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[FAIL] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, " errno=%d (%s)\n", err, err ? strerror(err) : "n/a");
    exit(2);
}

#define CUDA_OK(call, what)                                                  \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            step_fail(0, "%s -> %s", (what), cudaGetErrorString(_e));        \
        }                                                                    \
    } while (0)

/* ------------------------------------------------------------------ */
/* MMIO helper, BDF parser, ioctl wrapper -- copy of snvme_smoke.c     */
/* ------------------------------------------------------------------ */

#define NVME_REG_CAP    0x0000

static uint64_t mmio_read64(volatile void* base, size_t off) {
    volatile uint64_t* p = (volatile uint64_t*)((volatile char*)base + off);
    return *p;
}

static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

static int do_ioctl(int fd, unsigned long req, void* arg, const char* what) {
    int r = ioctl(fd, req, arg);
    if (r < 0) {
        int e = errno;
        fprintf(stderr, "ioctl(%s) failed: %s\n", what, strerror(e));
        errno = e;
    }
    return r;
}

/* ------------------------------------------------------------------ */
/* Round size up to GPU_PAGE_SIZE and round pointer down               */
/* ------------------------------------------------------------------ */

static inline size_t round_up_gpu(size_t n) {
    return (n + GPU_PAGE_SIZE - 1) & ~(GPU_PAGE_SIZE - 1);
}

/* cudaMalloc returns 256-byte-aligned pointers in practice. The kernel side
 * (nvfs_nvidia_p2p_get_pages) requires GPU-page (64 KiB) alignment, so we
 * over-allocate and align the *user-facing* pointer ourselves. We keep the
 * raw allocation around so we can free it. */
struct GpuBlock {
    void*  raw;          /* cudaMalloc'ed, kept for cudaFree */
    void*  aligned;      /* GPU_PAGE_SIZE-aligned slice handed to the kernel */
    size_t aligned_size; /* multiple of GPU_PAGE_SIZE */
    size_t n_pages;      /* aligned_size / GPU_PAGE_SIZE */
};

static GpuBlock alloc_gpu_block(size_t bytes, const char* label) {
    GpuBlock b{};
    b.aligned_size = round_up_gpu(bytes);
    b.n_pages = b.aligned_size / GPU_PAGE_SIZE;

    /* Over-alloc by one GPU page so we can align inside. */
    size_t alloc = b.aligned_size + GPU_PAGE_SIZE;
    CUDA_OK(cudaMalloc(&b.raw, alloc), "cudaMalloc(GpuBlock)");
    uintptr_t base = reinterpret_cast<uintptr_t>(b.raw);
    uintptr_t aligned = (base + GPU_PAGE_SIZE - 1) & ~(GPU_PAGE_SIZE - 1);
    b.aligned = reinterpret_cast<void*>(aligned);

    /* Zero the slice we hand to the kernel so the queue rings start clean. */
    CUDA_OK(cudaMemset(b.aligned, 0, b.aligned_size), "cudaMemset(GpuBlock)");

    fprintf(stderr, "      %s: raw=%p aligned=%p size=%zu (%zu GPU pages)\n",
            label, b.raw, b.aligned, b.aligned_size, b.n_pages);
    return b;
}

static void free_gpu_block(GpuBlock& b) {
    if (b.raw) {
        cudaFree(b.raw);
        b.raw = nullptr;
    }
}

/* ------------------------------------------------------------------ */
/* CLI                                                                */
/* ------------------------------------------------------------------ */

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--bind] [--gpu N] <PCI_BDF>\n"
        "  e.g.: %s 0000:50:00.0                # UAPI + GPU map paths (safe)\n"
        "        %s --gpu 1 0000:50:00.0        # use cuda device 1\n"
        "        %s --bind 0000:50:00.0         # full bring-up (destructive)\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    int do_bind = 0;
    int cuda_device = 0;
    const char* bdf_str = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bind") == 0) {
            do_bind = 1;
        } else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            cuda_device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (bdf_str == nullptr) {
            bdf_str = argv[i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (bdf_str == nullptr) {
        usage(argv[0]);
        return 1;
    }

    struct pci_device_addr orig_bdf;
    if (parse_bdf(bdf_str, &orig_bdf) != 0) {
        fprintf(stderr, "Bad BDF: '%s' (expected DDDD:BB:DD.F)\n", bdf_str);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* [G0] Pick the CUDA device. We do this first so a missing/disabled  */
    /*      GPU fails fast before we touch SNVMe state.                    */
    /* ------------------------------------------------------------------ */
    int n_devs = 0;
    CUDA_OK(cudaGetDeviceCount(&n_devs), "cudaGetDeviceCount");
    if (cuda_device < 0 || cuda_device >= n_devs)
        step_fail(0, "cuda device %d out of range (have %d)", cuda_device, n_devs);
    CUDA_OK(cudaSetDevice(cuda_device), "cudaSetDevice");
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, cuda_device), "cudaGetDeviceProperties");
    step_ok("cuda device=%d name='%s' pci=%04x:%02x:%02x.0",
            cuda_device, prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    /* ------------------------------------------------------------------ */
    /* [1] /dev/snvm_control                                              */
    /* ------------------------------------------------------------------ */
    int fd_ctl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctl < 0)
        step_fail(errno, "open(/dev/snvm_control)");
    step_ok("open(/dev/snvm_control) fd=%d", fd_ctl);

    /* ------------------------------------------------------------------ */
    /* [2] SNVM_CHRDEV_CREATE                                             */
    /* ------------------------------------------------------------------ */
    struct pci_device_addr addr = orig_bdf;
    if (do_ioctl(fd_ctl, SNVM_CHRDEV_CREATE, &addr, "SNVM_CHRDEV_CREATE") < 0)
        step_fail(errno, "SNVM_CHRDEV_CREATE %s", bdf_str);
    int minor_n = addr.domain;
    step_ok("SNVM_CHRDEV_CREATE minor=%d", minor_n);

    /* ------------------------------------------------------------------ */
    /* [3] /dev/ssnvme<N>                                                 */
    /* ------------------------------------------------------------------ */
    char dev_path[64];
    snprintf(dev_path, sizeof(dev_path), "/dev/ssnvme%d", minor_n);
    int fd_dev = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_dev < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ------------------------------------------------------------------ */
    /* [4] mmap BAR0 (no MAP_LOCKED -- BAR0 is device memory, not pageable) */
    /* ------------------------------------------------------------------ */
    const size_t bar0_size = 8192;
    void* bar0 = mmap(nullptr, bar0_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_dev, 0);
    if (bar0 == MAP_FAILED)
        step_fail(errno, "mmap(BAR0, %zu)", bar0_size);
    step_ok("mmap(BAR0, %zu) -> %p", bar0_size, bar0);

    /* ------------------------------------------------------------------ */
    /* [5] CAP sanity (see snvme_smoke.c for the all-zeros/all-ones logic) */
    /* ------------------------------------------------------------------ */
    uint64_t cap = mmio_read64(bar0, NVME_REG_CAP);
    if (cap == 0)
        step_fail(EIO, "BAR0 CAP reads as all-zeros "
                       "(BAR not mapped or pci_resource_start==0)");
    if (cap == (uint64_t)-1)
        step_warn("BAR0 CAP=0xFFF..FF -- controller powered down; UAPI-smoke continues");
    else
        step_ok("BAR0 CAP=0x%016" PRIx64, cap);

    /* ------------------------------------------------------------------ */
    /* [6] NVM_SET_IOQ_NUM(2)                                              */
    /*                                                                    */
    /* Field semantics (NB: the names are misleading on this ioctl):      */
    /*   request.ioq_idx -> total queue count                              */
    /*   request.is_cq   -> on_host flag                                   */
    /*                       0 = queues live on a CUDA device              */
    /*                       1 = queues live in host memory                */
    /* See snvme_smoke.c [6] for the trap; this binary is the GPU path,   */
    /* so on_host=0 (= request.is_cq = 0) is correct here.                 */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.ioq_idx = 2;
        req.is_cq   = 0;     /* on_host=0 -> device_queue_list */
        if (do_ioctl(fd_dev, NVM_SET_IOQ_NUM, &req, "NVM_SET_IOQ_NUM") < 0)
            step_fail(errno, "NVM_SET_IOQ_NUM nr=2");
    }
    step_ok("NVM_SET_IOQ_NUM nr=2 on_host=0");

    /* ------------------------------------------------------------------ */
    /* [7] cudaMalloc + NVM_MAP_DEVICE_QUEUE_MEMORY (SQ ring)              */
    /*                                                                    */
    /* GPU queue rings are mapped through nvidia_p2p_get_pages. Both the  */
    /* base address AND the size must be GPU_PAGE_SIZE-aligned, so we     */
    /* use alloc_gpu_block() to over-allocate and align inside.            */
    /* ------------------------------------------------------------------ */
    GpuBlock sq = alloc_gpu_block(GPU_PAGE_SIZE, "SQ ring");
    uint64_t sq_ioaddrs[16] = {0};   /* room for up to 16 GPU pages */
    if (sq.n_pages > 16)
        step_fail(0, "SQ ring needs %zu GPU pages, bump sq_ioaddrs[]", sq.n_pages);
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)sq.aligned;
        req.n_pages     = sq.n_pages;
        req.ioaddrs     = sq_ioaddrs;
        req.ioq_idx     = 0;        /* user queue #0 (0-based, see snvme_smoke.c [7]) */
        req.is_cq       = 0;        /* SQ */
        if (do_ioctl(fd_dev, NVM_MAP_DEVICE_QUEUE_MEMORY, &req,
                     "NVM_MAP_DEVICE_QUEUE_MEMORY(SQ)") < 0)
            step_fail(errno,
                "NVM_MAP_DEVICE_QUEUE_MEMORY SQ -- nvidia_p2p_get_pages failed; "
                "is the NVIDIA driver loaded and the GPU's BAR1 P2P-capable?");
    }
    step_ok("NVM_MAP_DEVICE_QUEUE_MEMORY(SQ) gpu_va=%p ioaddr[0]=0x%016" PRIx64,
            sq.aligned, sq_ioaddrs[0]);

    /* ------------------------------------------------------------------ */
    /* [8] NVM_MAP_DEVICE_QUEUE_MEMORY (CQ ring)                           */
    /* ------------------------------------------------------------------ */
    GpuBlock cq = alloc_gpu_block(GPU_PAGE_SIZE, "CQ ring");
    uint64_t cq_ioaddrs[16] = {0};
    if (cq.n_pages > 16)
        step_fail(0, "CQ ring needs %zu GPU pages, bump cq_ioaddrs[]", cq.n_pages);
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)cq.aligned;
        req.n_pages     = cq.n_pages;
        req.ioaddrs     = cq_ioaddrs;
        req.ioq_idx     = 0;        /* user queue #0 (matches SQ above) */
        req.is_cq       = 1;
        if (do_ioctl(fd_dev, NVM_MAP_DEVICE_QUEUE_MEMORY, &req,
                     "NVM_MAP_DEVICE_QUEUE_MEMORY(CQ)") < 0)
            step_fail(errno, "NVM_MAP_DEVICE_QUEUE_MEMORY CQ");
    }
    step_ok("NVM_MAP_DEVICE_QUEUE_MEMORY(CQ) gpu_va=%p ioaddr[0]=0x%016" PRIx64,
            cq.aligned, cq_ioaddrs[0]);

    /* ------------------------------------------------------------------ */
    /* [9] Bonus: NVM_MAP_DEVICE_MEMORY (data buffer, ioq_idx<0)           */
    /*                                                                    */
    /* This goes through map_device_memory() (not _ioqueue_memory()): it  */
    /* does NOT increment ioq_map_num and is not part of the SQ/CQ pool.  */
    /* It's the path libnvm uses for PRP / PRP-list buffers backing       */
    /* user-side IO requests. Smoke-test it here.                          */
    /* ------------------------------------------------------------------ */
    GpuBlock prp = alloc_gpu_block(GPU_PAGE_SIZE, "data/PRP buffer");
    uint64_t prp_ioaddrs[16] = {0};
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)prp.aligned;
        req.n_pages     = prp.n_pages;
        req.ioaddrs     = prp_ioaddrs;
        req.ioq_idx     = -1;       /* not a queue ring */
        req.is_cq       = 0;
        if (do_ioctl(fd_dev, NVM_MAP_DEVICE_MEMORY, &req,
                     "NVM_MAP_DEVICE_MEMORY(data)") < 0)
            step_fail(errno, "NVM_MAP_DEVICE_MEMORY data");
    }
    step_ok("NVM_MAP_DEVICE_MEMORY(data) gpu_va=%p ioaddr[0]=0x%016" PRIx64,
            prp.aligned, prp_ioaddrs[0]);

    /* ------------------------------------------------------------------ */
    /* [10] NVM_SET_SHARE_REG -- arms the use_sreg gate                    */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.ioq_idx = 1;
        if (do_ioctl(fd_dev, NVM_SET_SHARE_REG, &req, "NVM_SET_SHARE_REG") < 0)
            step_fail(errno, "NVM_SET_SHARE_REG(1)");
    }
    step_ok("NVM_SET_SHARE_REG(1)");

    /* ================================================================== */
    /*  --bind path                                                       */
    /* ================================================================== */
    if (do_bind) {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_BIND, &bdf, "SNVM_DEVICE_BIND") < 0)
            step_fail(errno,
                "SNVM_DEVICE_BIND %s -- the in-tree nvme driver may still own this device, "
                "or use_sreg/ioq_map_num invariants are off (see PORTING.md §5)",
                bdf_str);

        /* Poll NVM_GET_DEV_INFO until async probe + namespace scan finish. */
        struct nvm_ioctl_dev info;
        int ok = 0;
        for (int i = 0; i < 100; i++) {   /* up to ~10 s */
            memset(&info, 0, sizeof(info));
            if (ioctl(fd_dev, NVM_GET_DEV_INFO, &info) == 0 &&
                info.disk_name[0] != '\0') {
                ok = 1;
                break;
            }
            usleep(100 * 1000);
        }
        if (!ok)
            step_fail(errno, "NVM_GET_DEV_INFO did not succeed within 10s after bind");
        step_ok("SNVM_DEVICE_BIND %s (probe done with GPU queues)", bdf_str);

        char disk_name[DISK_NAME_LEN + 1] = {0};
        memcpy(disk_name, info.disk_name, DISK_NAME_LEN);
        step_ok("NVM_GET_DEV_INFO disk='%s' nr_user_q=%u block_size=%zu max_data_size=%zu",
                disk_name, info.nr_user_q, info.block_size, info.max_data_size);

        char blk_path[DISK_NAME_LEN + 8];
        snprintf(blk_path, sizeof(blk_path), "/dev/%s", disk_name);
        int fd_blk = open(blk_path, O_RDONLY);
        if (fd_blk < 0)
            step_fail(errno, "open(%s)", blk_path);
        char buf[512];
        ssize_t got = pread(fd_blk, buf, sizeof(buf), 0);
        if (got != (ssize_t)sizeof(buf))
            step_fail(errno, "pread(%s, 512) returned %zd", blk_path, got);
        close(fd_blk);
        step_ok("pread(%s, 512) ok", blk_path);

        bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_UNBIND, &bdf, "SNVM_DEVICE_UNBIND") < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND %s", bdf_str);
        step_ok("SNVM_DEVICE_UNBIND %s", bdf_str);
    }

    /* ------------------------------------------------------------------ */
    /* [F1] Cleanup: unmap rings + data buffer, clear ioq state.           */
    /*                                                                    */
    /* Order matters only for ioq_map_num bookkeeping: unmapping the two  */
    /* queue rings must happen via NVM_UNMAP_DEVICE_QUEUE_MEMORY (not the */
    /* plain DEVICE variant), because the kernel keeps them on a separate */
    /* device_queue_list (pci.c case NVM_UNMAP_DEVICE_QUEUE_MEMORY).      */
    /* ------------------------------------------------------------------ */
    {
        uint64_t v;
        v = (uint64_t)(uintptr_t)sq.aligned;
        if (do_ioctl(fd_dev, NVM_UNMAP_DEVICE_QUEUE_MEMORY, &v,
                     "NVM_UNMAP_DEVICE_QUEUE_MEMORY(SQ)") < 0)
            step_fail(errno, "NVM_UNMAP_DEVICE_QUEUE_MEMORY SQ");
        v = (uint64_t)(uintptr_t)cq.aligned;
        if (do_ioctl(fd_dev, NVM_UNMAP_DEVICE_QUEUE_MEMORY, &v,
                     "NVM_UNMAP_DEVICE_QUEUE_MEMORY(CQ)") < 0)
            step_fail(errno, "NVM_UNMAP_DEVICE_QUEUE_MEMORY CQ");
        v = (uint64_t)(uintptr_t)prp.aligned;
        if (do_ioctl(fd_dev, NVM_UNMAP_DEVICE_MEMORY, &v,
                     "NVM_UNMAP_DEVICE_MEMORY(data)") < 0)
            step_fail(errno, "NVM_UNMAP_DEVICE_MEMORY data");
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_dev, NVM_CLEAR_IOQ_NUM, &req,
                     "NVM_CLEAR_IOQ_NUM") < 0)
            step_fail(errno, "NVM_CLEAR_IOQ_NUM");
    }
    step_ok("NVM_UNMAP_DEVICE_QUEUE_MEMORY x2 + NVM_UNMAP_DEVICE_MEMORY + NVM_CLEAR_IOQ_NUM");

    /* ------------------------------------------------------------------ */
    /* [F2] cudaFree + munmap(BAR0) + close(/dev/ssnvme<N>)                */
    /* ------------------------------------------------------------------ */
    free_gpu_block(sq);
    free_gpu_block(cq);
    free_gpu_block(prp);
    if (munmap(bar0, bar0_size) < 0)
        step_fail(errno, "munmap(BAR0)");
    if (close(fd_dev) < 0)
        step_fail(errno, "close(%s)", dev_path);
    step_ok("cudaFree x3 + munmap(BAR0) + close(%s)", dev_path);

    /* ------------------------------------------------------------------ */
    /* [F3] SNVM_CHRDEV_REMOVE                                              */
    /* ------------------------------------------------------------------ */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf,
                     "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE %s", bdf_str);
    }
    step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);

    close(fd_ctl);
    fprintf(stderr, "\nAll %d steps passed. SNVMe GPU paths are healthy%s.\n",
            g_step, do_bind ? " (full bring-up)" : " (UAPI-smoke)");
    return 0;
}
