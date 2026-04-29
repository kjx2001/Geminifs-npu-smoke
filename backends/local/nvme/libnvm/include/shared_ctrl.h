#ifndef __SHARED_CTRL_H__
#define __SHARED_CTRL_H__

/**
 * shared_ctrl.h -- libnvm API for building a Controller from shared resources.
 *
 * Used by NVMeService clients. The daemon process calls the regular
 * Controller(pci_addr, ...) constructor to do full init. A client process
 * receives IPC handles via RPC and passes them through a SharedControllerSpec
 * to build_shared_controller(), which:
 *
 *   1. Opens the SNVMe device file and mmaps BAR0 (multi-process safe --
 *      SNVMe has no exclusive-open semantics)
 *   2. cudaHostRegister's BAR0 + cudaHostGetDevicePointer to obtain this
 *      process's own doorbell GPU VA (the daemon's VA is not valid here)
 *   3. For each queue: cudaIpcOpenMemHandle for SQ / CQ / optional PRP memory
 *   4. For each queue: cudaMalloc's fresh tickets / marks / cid / pos_locks
 *      (intra-process GPU thread coordination -- not shared with daemon)
 *   5. Builds a fully-populated Controller + QueuePair array, returned as
 *      shared_ptr with a custom deleter that handles all shared-mode cleanup
 *      (cudaIpcCloseMemHandle, cudaHostUnregister, munmap, etc.)
 *
 * The returned Controller is interchangeable with one built by the regular
 * constructor: kernels see the same h_qps/d_qps layout.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// Forward declaration -- avoid pulling ctrl.h into every consumer
struct Controller;

/**
 * Per-queue shared memory descriptor from the daemon's AllocResponse.
 *
 * SQ / CQ pointers are always valid. PRP is optional: when has_prp is false,
 * build_shared_controller() will cudaMalloc a fresh PRP pool in this process
 * instead of importing the daemon's.
 */
struct SharedQueueSpec {
    int32_t             queue_id       = -1;
    cudaIpcMemHandle_t  sq_handle      {};
    cudaIpcMemHandle_t  cq_handle      {};
    cudaIpcMemHandle_t  prp_handle     {};
    bool                has_prp        = false;
    uint32_t            sq_entries     = 0;
    uint32_t            cq_entries     = 0;
    uint64_t            sq_ioaddr      = 0;     // for debugging only
    uint64_t            cq_ioaddr      = 0;
};

/**
 * Device + queue-range description needed to reconstruct a Controller locally.
 */
struct SharedControllerSpec {
    std::string                   snvme_dev_path;   // e.g. "/dev/snvm_nvme0n1"
    uint64_t                      bar0_size    = 0;
    uint32_t                      dstrd        = 0; // doorbell stride (encoded)
    uint32_t                      page_size    = 0;
    uint32_t                      blk_size     = 0;
    uint32_t                      blk_size_log = 0;
    uint32_t                      namespace_id = 0;
    int32_t                       cuda_device  = -1;
    std::string                   mount_path;       // for Controller::dev_mount_path
    uint32_t                      queue_depth  = 0;
    std::vector<SharedQueueSpec>  queues;
};

/**
 * Build a Controller whose queue resources are imported from the daemon.
 *
 * Throws std::runtime_error on failure. The returned shared_ptr owns all
 * local-process resources (imported IPC mappings, local GPU buffers,
 * BAR0 mmap); they are released when the last reference drops.
 *
 * The caller is expected to have already set the active CUDA device to
 * spec.cuda_device (or this function can do it -- it does, defensively).
 */
std::shared_ptr<Controller>
build_shared_controller(const SharedControllerSpec& spec);

#endif // __SHARED_CTRL_H__
