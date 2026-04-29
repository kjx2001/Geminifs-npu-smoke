#include "shared_ctrl.h"
#include "ctrl.h"
#include "queue.h"
#include "regs.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Controller: shared-resource constructor
// ---------------------------------------------------------------------------

Controller::Controller(const SharedControllerSpec& spec)
    : ctrl(nullptr),
      n_sqs(0), n_cqs(0), n_qps(0), n_user_qps(0),
      deviceId(static_cast<uint32_t>(spec.cuda_device)),
      h_qps(nullptr), d_qps(nullptr),
      page_size(spec.page_size),
      blk_size(spec.blk_size),
      blk_size_log(spec.blk_size_log),
      dev_path(nullptr),
      dev_mount_path(spec.mount_path),
      d_ctrl_ptr(nullptr),
      is_shared(true)
{
    // POD sub-structs: zero-init then fill what the kernels/runtime actually read.
    std::memset(&info, 0, sizeof(info));
    std::memset(&ns,   0, sizeof(ns));
    std::memset(&disk, 0, sizeof(disk));

    disk.ns_id      = spec.namespace_id;
    disk.page_size  = spec.page_size;
    disk.block_size = spec.blk_size;
}

// ---------------------------------------------------------------------------
// QueuePair: shared-resource constructor
// ---------------------------------------------------------------------------

QueuePair::QueuePair(const SharedQueueSpec& qspec,
                      uint32_t cudaDevice,
                      uint32_t dstrd,
                      uint32_t nvmNamespace_,
                      uint32_t page_size_,
                      uint32_t block_size_,
                      uint32_t block_size_log_,
                      void*    bar0_gpu_va)
{
    pageSize           = page_size_;
    block_size         = block_size_;
    block_size_minus_1 = block_size_ - 1;
    block_size_log     = block_size_log_;
    nvmNamespace       = nvmNamespace_;
    qp_id              = static_cast<uint16_t>(qspec.queue_id);
    is_shared          = true;

    // 1. Import SQ memory via CUDA IPC.
    cudaError_t err = cudaIpcOpenMemHandle(
        &shared_sq_ptr,
        const_cast<cudaIpcMemHandle_t&>(qspec.sq_handle),
        cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaIpcOpenMemHandle(SQ) failed: ") + cudaGetErrorString(err));
    }

    // 2. Import CQ memory.
    err = cudaIpcOpenMemHandle(
        &shared_cq_ptr,
        const_cast<cudaIpcMemHandle_t&>(qspec.cq_handle),
        cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        cudaIpcCloseMemHandle(shared_sq_ptr);
        shared_sq_ptr = nullptr;
        throw std::runtime_error(
            std::string("cudaIpcOpenMemHandle(CQ) failed: ") + cudaGetErrorString(err));
    }

    // 3. Import PRP memory if the daemon provided one.
    if (qspec.has_prp) {
        err = cudaIpcOpenMemHandle(
            &shared_prp_ptr,
            const_cast<cudaIpcMemHandle_t&>(qspec.prp_handle),
            cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            // Non-fatal: treat as "client allocates own PRP later".
            std::fprintf(stderr,
                "warning: cudaIpcOpenMemHandle(PRP) failed for queue %d: %s\n",
                qspec.queue_id, cudaGetErrorString(err));
            shared_prp_ptr = nullptr;
        }
    }

    // 4. Set nvm_queue_t sizes + derived log/mask fields.
    sq.qs          = qspec.sq_entries;
    sq.qs_minus_1  = qspec.sq_entries - 1;
    sq.qs_log2     = static_cast<uint32_t>(std::log2(qspec.sq_entries));
    sq.es          = sizeof(nvm_cmd_t);

    cq.qs          = qspec.cq_entries;
    cq.qs_minus_1  = qspec.cq_entries - 1;
    cq.qs_log2     = static_cast<uint32_t>(std::log2(qspec.cq_entries));
    cq.es          = sizeof(nvm_cpl_t);

    // 5. Initialise nvm_queue_t fields that nvm_queue_clear would normally set.
    //    Atomics are already zero-initialised by memset in operator new or by
    //    default in-place -- set them explicitly for safety in case this
    //    object was heap-allocated.
    sq.no        = qspec.queue_id;
    sq.head      = 0;
    sq.tail      = 0;
    sq.last      = 0;
    sq.phase     = 1;
    sq.local     = 0;
    sq.head_lock = 0;
    sq.tail_lock = 0;
    sq.in_ticket = 0;
    sq.cid_ticket= 0;
    sq.vaddr     = reinterpret_cast<volatile void*>(shared_sq_ptr);
    sq.ioaddr    = qspec.sq_ioaddr;

    cq.no        = qspec.queue_id;
    cq.head      = 0;
    cq.tail      = 0;
    cq.last      = 0;
    cq.phase     = 1;
    cq.local     = 0;
    cq.head_lock = 0;
    cq.tail_lock = 0;
    cq.in_ticket = 0;
    cq.cid_ticket= 0;
    cq.vaddr     = reinterpret_cast<volatile void*>(shared_cq_ptr);
    cq.ioaddr    = qspec.cq_ioaddr;

    // 6. Doorbell pointers: computed from this process's own BAR0 GPU VA,
    //    NOT the daemon's (those were valid only inside the daemon).
    sq.db = SQ_DBL(bar0_gpu_va, qspec.queue_id, dstrd);
    cq.db = CQ_DBL(bar0_gpu_va, qspec.queue_id, dstrd);

    // 7. Allocate this process's own tickets / marks / cid / pos_locks.
    //    These coordinate GPU threads within this process; they are NOT
    //    shared with the daemon.
    init_gpu_specific_struct(cudaDevice);
}

// ---------------------------------------------------------------------------
// build_shared_controller
// ---------------------------------------------------------------------------

namespace {

struct SharedResources {
    int      bar0_fd    = -1;
    void*    bar0_mmap  = nullptr;
    uint64_t bar0_size  = 0;
    void*    bar0_gpu_va = nullptr;
};

void cleanup_shared_resources(SharedResources* res) {
    if (res == nullptr) return;
    if (res->bar0_mmap != nullptr) {
        cudaHostUnregister(res->bar0_mmap);
        munmap(res->bar0_mmap, res->bar0_size);
        res->bar0_mmap = nullptr;
    }
    if (res->bar0_fd >= 0) {
        close(res->bar0_fd);
        res->bar0_fd = -1;
    }
}

} // namespace

std::shared_ptr<Controller>
build_shared_controller(const SharedControllerSpec& spec) {
    if (spec.cuda_device < 0) {
        throw std::runtime_error("build_shared_controller: invalid cuda_device");
    }
    if (spec.queues.empty()) {
        throw std::runtime_error("build_shared_controller: spec has no queues");
    }

    cudaError_t cerr = cudaSetDevice(spec.cuda_device);
    if (cerr != cudaSuccess) {
        throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                                  cudaGetErrorString(cerr));
    }

    auto res = std::make_shared<SharedResources>();
    res->bar0_size = spec.bar0_size;

    // 1. Open SNVMe device file for BAR0 mmap (multi-process safe).
    res->bar0_fd = ::open(spec.snvme_dev_path.c_str(), O_RDWR);
    if (res->bar0_fd < 0) {
        throw std::runtime_error("open(" + spec.snvme_dev_path + ") failed: " +
                                 std::strerror(errno));
    }

    // 2. mmap BAR0 into this process.
    res->bar0_mmap = ::mmap(nullptr, spec.bar0_size,
                             PROT_READ | PROT_WRITE, MAP_SHARED,
                             res->bar0_fd, 0);
    if (res->bar0_mmap == MAP_FAILED) {
        res->bar0_mmap = nullptr;
        close(res->bar0_fd);
        res->bar0_fd = -1;
        throw std::runtime_error("mmap(BAR0) failed: " + std::string(std::strerror(errno)));
    }

    // 3. Register BAR0 with CUDA as IO memory, obtain this-process GPU VA.
    cerr = cudaHostRegister(res->bar0_mmap, spec.bar0_size, cudaHostRegisterIoMemory);
    if (cerr != cudaSuccess) {
        cleanup_shared_resources(res.get());
        throw std::runtime_error(std::string("cudaHostRegister(BAR0) failed: ") +
                                  cudaGetErrorString(cerr));
    }

    cerr = cudaHostGetDevicePointer(&res->bar0_gpu_va, res->bar0_mmap, 0);
    if (cerr != cudaSuccess) {
        cleanup_shared_resources(res.get());
        throw std::runtime_error(std::string("cudaHostGetDevicePointer(BAR0) failed: ") +
                                  cudaGetErrorString(cerr));
    }

    // 4. Construct Controller in shared mode (pointers null, fields from spec).
    Controller* ctrl = new Controller(spec);

    // 5. Populate queue pool.
    const size_t nq = spec.queues.size();
    ctrl->n_qps      = static_cast<uint16_t>(nq);
    ctrl->n_sqs      = static_cast<uint16_t>(nq);
    ctrl->n_cqs      = static_cast<uint16_t>(nq);
    ctrl->n_user_qps = static_cast<uint16_t>(nq);

    ctrl->h_qps = static_cast<QueuePair**>(malloc(sizeof(QueuePair*) * nq));
    if (ctrl->h_qps == nullptr) {
        delete ctrl;
        cleanup_shared_resources(res.get());
        throw std::runtime_error("malloc(h_qps) failed");
    }
    for (size_t i = 0; i < nq; ++i) ctrl->h_qps[i] = nullptr;

    cerr = cudaMalloc(reinterpret_cast<void**>(&ctrl->d_qps), sizeof(QueuePair) * nq);
    if (cerr != cudaSuccess) {
        free(ctrl->h_qps);
        ctrl->h_qps = nullptr;
        delete ctrl;
        cleanup_shared_resources(res.get());
        throw std::runtime_error(std::string("cudaMalloc(d_qps) failed: ") +
                                  cudaGetErrorString(cerr));
    }

    for (size_t i = 0; i < nq; ++i) {
        try {
            ctrl->h_qps[i] = new QueuePair(spec.queues[i],
                                            static_cast<uint32_t>(spec.cuda_device),
                                            spec.dstrd,
                                            spec.namespace_id,
                                            spec.page_size,
                                            spec.blk_size,
                                            spec.blk_size_log,
                                            res->bar0_gpu_va);
        } catch (...) {
            delete ctrl;  // ~Controller handles partial h_qps via is_shared path
            cleanup_shared_resources(res.get());
            throw;
        }

        cerr = cudaMemcpy(ctrl->d_qps + i, ctrl->h_qps[i],
                           sizeof(QueuePair), cudaMemcpyHostToDevice);
        if (cerr != cudaSuccess) {
            delete ctrl;
            cleanup_shared_resources(res.get());
            throw std::runtime_error(std::string("cudaMemcpy(d_qps[i]) failed: ") +
                                      cudaGetErrorString(cerr));
        }
    }

    // 6. Make Controller itself available on device (d_ctrl_ptr).
    ctrl->d_ctrl_buff = createBuffer(sizeof(Controller), spec.cuda_device);
    ctrl->d_ctrl_ptr  = ctrl->d_ctrl_buff.get();
    cerr = cudaMemcpy(ctrl->d_ctrl_ptr, ctrl, sizeof(Controller), cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        delete ctrl;
        cleanup_shared_resources(res.get());
        throw std::runtime_error(std::string("cudaMemcpy(d_ctrl_ptr) failed: ") +
                                  cudaGetErrorString(cerr));
    }

    // 7. Wrap in shared_ptr with a deleter that releases BAR0 resources
    //    *after* Controller's destructor (which closes IPC imports via the
    //    QueuePair destructors).
    return std::shared_ptr<Controller>(ctrl, [res](Controller* p) {
        delete p;                       // ~Controller -> ~QueuePair -> cudaIpcCloseMemHandle
        cleanup_shared_resources(res.get());
    });
}
