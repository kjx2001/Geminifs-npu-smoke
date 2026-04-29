#ifndef __BENCHMARK_CTRL_H__
#define __BENCHMARK_CTRL_H__

// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <cstdint>
#include "buffer.h"
#include "ioctl.h"
#include "nvm_types.h"
#include "nvm_ctrl.h"
#include "nvm_error.h"
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda/atomic>
#include "file.h"
#include "queue.h"

// Forward declarations
struct SharedControllerSpec;   // defined in shared_ctrl.h

// Per-queue memory target. Each IO queue's SQ/CQ ring memory either
// lives on a specific GPU (on_host=false, cuda_device=<id>) or in host
// memory (on_host=true). The host-memory path is reserved for future
// CPU_SUBMIT support: init_queues currently rejects on_host=true with
// ENOTSUP. cuda_device values can be arbitrary, non-contiguous CUDA
// device indices (e.g. {0, 2}); they are NOT treated as positional
// indices into anything.
struct QueueMemTarget {
    bool     on_host     = false;
    uint32_t cuda_device = 0;
};

#define MAX_QUEUES 1024
#define NVM_CTRL_IOQ_MINNUM    64

struct Controller
{
    cuda::atomic<uint64_t, cuda::thread_scope_device> access_counter;
    nvm_ctrl_t*             ctrl;
    struct nvm_ctrl_info    info;
    struct nvm_ns_info      ns;
    struct disk             disk;
    uint16_t                n_sqs;
    uint16_t                n_cqs;
    uint16_t                n_qps;
    uint16_t                n_user_qps;

    uint32_t                deviceId;
    QueuePair**             h_qps;
    QueuePair*              d_qps;


    cuda::atomic<uint64_t, cuda::thread_scope_device> queue_counter;

    uint32_t page_size;
    uint32_t blk_size;
    uint32_t blk_size_log;
    char* dev_path;
    // char* dev_mount_path;
    std::string dev_mount_path;
    void* d_ctrl_ptr;
    BufferPtr d_ctrl_buff;

    // True when this Controller was assembled from shared resources
    // (imported via NVMeService / build_shared_controller). Shared mode
    // skips standalone-only cleanup in the destructor: no Host_file_system_exit
    // (daemon owns the mount), no nvm_ctrl_free (daemon owns nvm_ctrl_t).
    bool                    is_shared = false;

    Controller(const char* snvme_control_path,
                const char* pci_addr,
                std::string mount_path,
                uint32_t ns_id,
                uint32_t cudaDevice,
                uint64_t queueDepth,
                uint64_t numQueues);

    // Per-queue-target constructor. queue_targets[i] specifies where
    // queue i's SQ/CQ memory lives. The "primary" cuda device used for
    // the controller-wide d_qps and d_ctrl_buff allocations is the
    // first GPU target found in queue_targets. queue_targets must
    // contain at least one !on_host entry until host-mem queues are
    // implemented.
    Controller(const char* snvme_control_path,
                const char* pci_addr,
                std::string mount_path,
                uint32_t ns_id,
                const std::vector<QueueMemTarget>& queue_targets,
                uint64_t queueDepth);

    // Shared-resource constructor. Defined in libnvm/src/shared_ctrl.cu.
    // Callers should use build_shared_controller() rather than invoking this
    // directly -- the free function also sets up BAR0 mmap and the IPC
    // imports referenced by each QueuePair.
    explicit Controller(const struct SharedControllerSpec& spec);

    void print_reset_stats(void);
    int init_queues(uint32_t ns_id,  uint32_t cudaDevice,
                     uint64_t numQueues, uint64_t queueDepth);

    // Per-queue-target init_queues. Returns ENOTSUP if any
    // queue_targets[i].on_host is true (CPU-resident queues are
    // reserved for a later round). The legacy single-cudaDevice
    // init_queues delegates to this with a uniform GPU vector.
    int init_queues(uint32_t ns_id,
                     const std::vector<QueueMemTarget>& queue_targets,
                     uint64_t queueDepth);

    ~Controller();
};



using error = std::runtime_error;
using std::string;


inline void Controller::print_reset_stats(void) {
    cuda_err_chk(cudaMemcpy(&access_counter, d_ctrl_ptr, sizeof(cuda::atomic<uint64_t, cuda::thread_scope_device>), cudaMemcpyDeviceToHost));
    std::cout << "------------------------------------" << std::endl;
    std::cout << std::dec << "#SSDAccesses:\t" << access_counter << std::endl;

    cuda_err_chk(cudaMemset(d_ctrl_ptr, 0, sizeof(cuda::atomic<uint64_t, cuda::thread_scope_device>)));
}





inline Controller::Controller(const char* snvme_control_path,
                                const char* pci_addr,
                                std::string mount_path,
                                uint32_t ns_id,
                                uint32_t cudaDevice,
                                uint64_t queueDepth,
                                uint64_t numQueues)
    : Controller(snvme_control_path, pci_addr, std::move(mount_path),
                 ns_id,
                 std::vector<QueueMemTarget>(numQueues,
                     QueueMemTarget{/*on_host=*/false, cudaDevice}),
                 queueDepth)
{
    // Delegating constructor: builds a uniform GPU target vector and
    // forwards to the per-queue-target constructor below.
}

inline Controller::Controller(const char* snvme_control_path,
                                const char* pci_addr,
                                std::string mount_path,
                                uint32_t ns_id,
                                const std::vector<QueueMemTarget>& queue_targets,
                                uint64_t queueDepth)
    : ctrl(nullptr), deviceId(0)
{
    if (queue_targets.empty()) {
        nvm_throw_error("Controller: queue_targets is empty", EINVAL);
    }
    // Pick the first GPU-backed entry as the "primary" device used for
    // d_qps / d_ctrl_buff. CPU entries are reserved for a later round.
    bool primary_set = false;
    for (const auto& t : queue_targets) {
        if (!t.on_host) { deviceId = t.cuda_device; primary_set = true; break; }
    }
    if (!primary_set) {
        nvm_throw_error("Controller: no GPU-backed entry in queue_targets "
                        "(host-only queue pools are not yet supported)", ENOTSUP);
    }

    int status;

    status = nvm_controller_init(&ctrl, snvme_control_path, pci_addr);
    if (status != 0){
        nvm_throw_error("Failed to nvm_controller_init", status);
    }

    status = init_queues(ns_id, queue_targets, queueDepth);
    if (status != 0){
        nvm_throw_error("Failed to init_queues", status);
    }

    page_size = ctrl->page_size;
    blk_size = disk.block_size;
    blk_size_log = std::log2(blk_size);

    dev_path = (char *)malloc(40 * sizeof(char));
    snprintf(dev_path, 40, "/dev/%s", disk.disk_name);

    dev_mount_path = mount_path;

    d_ctrl_buff = createBuffer(sizeof(Controller), deviceId);
    d_ctrl_ptr = d_ctrl_buff.get();
    cuda_err_chk(cudaMemcpy(d_ctrl_ptr, this, sizeof(Controller), cudaMemcpyHostToDevice));

    Host_file_system_int(dev_path, dev_mount_path.c_str());
}


inline int Controller::init_queues(uint32_t ns_id,  uint32_t cudaDevice,
                                    uint64_t numQueues, uint64_t queueDepth){
    // Legacy single-cudaDevice path: delegate to the per-queue-target
    // overload with a uniform vector.
    return init_queues(ns_id,
                       std::vector<QueueMemTarget>(numQueues,
                           QueueMemTarget{/*on_host=*/false, cudaDevice}),
                       queueDepth);
}

inline int Controller::init_queues(uint32_t ns_id,
                                    const std::vector<QueueMemTarget>& queue_targets,
                                    uint64_t queueDepth){

    uint16_t max_queue = 75;
    void* devicePtr = nullptr;
    size_t i;
    int status;

    // Reject host-resident queues for now -- API accepts them so the
    // shape is stable, but the implementation only handles GPU memory.
    for (size_t k = 0; k < queue_targets.size(); ++k) {
        if (queue_targets[k].on_host) {
            printf("init_queues: host-resident queue %zu not yet supported\n", k);
            return ENOTSUP;
        }
    }

    const uint64_t numQueues = static_cast<uint64_t>(queue_targets.size());
    this->n_qps = std::min(max_queue, (uint16_t)numQueues);
    this->n_sqs = n_qps;
    this->n_cqs = n_qps;

    this->ctrl->cq_num = this->n_cqs;
    this->ctrl->sq_num = this->n_sqs;

    status = nvm_queue_set(ctrl, this->n_sqs + this->n_cqs);
    if (status != 0){
        printf("Failed to nvm_queue_set : %s\n", nvm_strerror(status));
        return EFAULT;
    }

    this->h_qps = (QueuePair**) malloc(sizeof(QueuePair) * n_qps);
    cuda_err_chk(cudaMalloc((void**)&this->d_qps, sizeof(QueuePair) * this->n_qps));

    for (i = 0; i < n_qps; i++) {
        // Per-queue cuda device: SQ/CQ ring memory and intra-queue
        // tickets/marks live on this GPU. May differ across queues.
        const uint32_t per_queue_dev = queue_targets[i].cuda_device;
        h_qps[i] = new QueuePair(ctrl, per_queue_dev, i, queueDepth);
    }

    status = nvm_device_init(ctrl);
    if (status != 0){
        printf("Failed to nvm_device_init : %s\n", nvm_strerror(status));
        return EFAULT;
    }

    this->disk.ns_id = ns_id;
    this->disk.page_size = ctrl->page_size;
    status = init_userioq_device(ctrl, this->h_qps, &this->disk);
    if (status != 0){
        printf("Failed to init userioq : %s\n", nvm_strerror(status));
    }

    this->n_qps = ctrl->nr_user_q;

    for (i = 0; i < this->n_qps; i++) {
        // Doorbell GPU VAs are obtained via cudaHostGetDevicePointer on
        // the BAR0 host mapping. The returned VA is the *current device*'s
        // view of that host memory, so set the device matching this queue
        // before each lookup.
        const uint32_t per_queue_dev = queue_targets[i].cuda_device;
        cuda_err_chk(cudaSetDevice(per_queue_dev));

        devicePtr = nullptr;
        cudaError_t err = cudaHostGetDevicePointer(&devicePtr, (void*) h_qps[i]->cq.db, 0);
        if (err != cudaSuccess) {
            printf("Failed to get device pointer %s\n", cudaGetErrorString(err));
            return -1;
        }
        h_qps[i]->cq.db = (volatile uint32_t*) devicePtr;

        // Get a valid device pointer for SQ doorbell
        err = cudaHostGetDevicePointer(&devicePtr, (void*) h_qps[i]->sq.db, 0);
        if (err != cudaSuccess) {
            printf("Failed to get device pointer %s\n", cudaGetErrorString(err));
            return -1;
        }
        h_qps[i]->sq.db = (volatile uint32_t*) devicePtr;

        // d_qps lives on the primary GPU; cudaMemcpy from any host source
        // to that primary works regardless of per_queue_dev.
        cuda_err_chk(cudaMemcpy(d_qps + i, h_qps[i], sizeof(QueuePair), cudaMemcpyHostToDevice));
    }
    return 0;
}

inline Controller::~Controller()
{
    if (d_qps != nullptr) {
        cudaFree(d_qps);
        d_qps = nullptr;
    }
    if (h_qps != nullptr) {
        for (size_t i = 0; i < n_qps; i++) {
            delete h_qps[i];
        }
        free(h_qps);
        h_qps = nullptr;
    }

    if (is_shared) {
        // Shared mode: the mount, nvm_ctrl_t, BAR0 mmap and IPC imports are
        // managed by the external deleter in build_shared_controller().
        // Nothing else to do here.
        return;
    }

    int ret = Host_file_system_exit(dev_path);
    if(ret < 0)
        exit(-1);
    printf("Controller release\n");
    nvm_ctrl_free(ctrl);
}


#endif
