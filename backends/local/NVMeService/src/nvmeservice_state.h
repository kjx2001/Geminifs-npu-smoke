#ifndef __NVMESERVICE_STATE_H__
#define __NVMESERVICE_STATE_H__

/**
 * nvmeservice_state.h -- daemon-side service state.
 *
 * Responsibilities:
 *   - Own one libnvm Controller per configured NVMe device (admin queues,
 *     full queue pool creation)
 *   - Precompute cudaIpcMemHandle_t for each queue's SQ / CQ / PRP memory
 *   - Accept AllocateQueues requests, carve out a contiguous queue range
 *     for the caller, and return the IPC handles
 *   - Track per-allocation leases: PID, /proc/<pid>/stat starttime,
 *     last_heartbeat
 *   - Run a background reaper that reclaims queue ranges whose lease has
 *     expired AND whose client PID is dead (starttime-checked to defeat
 *     PID reuse)
 *
 * Thread safety: all public operations are serialised by an internal mutex.
 * No protobuf dependency -- the server layer translates to/from proto.
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "nvmeservice_config.h"

// Forward declarations to avoid pulling libnvm headers into this file.
struct Controller;

namespace nvmeservice {

// -----------------------------------------------------------------
// Value types (mirror of proto messages, but proto-free)
// -----------------------------------------------------------------

struct QueueGroupSnapshot {
    int32_t cuda_device     = -1;
    int32_t queue_start_idx = 0;     // absolute queue index where this group begins
    int32_t queue_count     = 0;
    int32_t available       = 0;     // unallocated queues in this group
};

struct DeviceSnapshot {
    int32_t     device_id        = -1;
    std::string pci_addr;
    std::string snvme_dev_path;
    int32_t     cuda_device      = -1;   // legacy: equals groups[0].cuda_device
    uint32_t    namespace_id     = 0;
    uint32_t    page_size        = 0;
    uint32_t    blk_size         = 0;
    uint32_t    blk_size_log     = 0;
    uint32_t    queue_depth      = 0;
    uint32_t    dstrd            = 0;
    uint64_t    bar0_size        = 0;
    int32_t     total_queues     = 0;
    int32_t     available_queues = 0;        // sum across all groups
    std::vector<QueueGroupSnapshot> groups;  // per-GPU partition view
};

struct QueueShared {
    int32_t             queue_id     = -1;
    cudaIpcMemHandle_t  ipc_sq   {};
    cudaIpcMemHandle_t  ipc_cq   {};
    cudaIpcMemHandle_t  ipc_prp  {};     // zeroed when client should allocate its own PRP
    bool                has_prp  = false;
    uint32_t            sq_entries   = 0;
    uint32_t            cq_entries   = 0;
    uint64_t            sq_ioaddr    = 0;
    uint64_t            cq_ioaddr    = 0;
};

struct AllocationGrant {
    std::string               allocation_id;
    int32_t                   device_id              = -1;
    std::string               pci_addr;
    std::string               snvme_dev_path;
    // GPU-view filesystem path for this allocation: the symlink under
    // the consuming GPU's mount_path that points at the per-GPU
    // subdirectory on the NVMe. Empty if symlink install failed.
    std::string               mount_path;
    uint64_t                  bar0_size              = 0;
    uint32_t                  dstrd                  = 0;
    int32_t                   queue_start_idx        = 0;
    int32_t                   queue_count            = 0;
    std::vector<QueueShared>  queue_shared;
    uint32_t                  namespace_id           = 0;
    uint32_t                  page_size              = 0;
    uint32_t                  blk_size               = 0;
    uint32_t                  blk_size_log           = 0;
    uint32_t                  queue_depth            = 0;
    uint32_t                  heartbeat_interval_sec = 0;
    uint32_t                  lease_timeout_sec      = 0;
};

// -----------------------------------------------------------------
// Internal allocation record
// -----------------------------------------------------------------

struct Allocation {
    std::string                              allocation_id;
    int32_t                                  device_id            = -1;
    int32_t                                  cuda_device          = -1;  // GPU this allocation is bound to (= group's cuda_device)
    int32_t                                  queue_start_idx      = 0;   // absolute queue index
    int32_t                                  queue_count          = 0;
    uint32_t                                  client_pid          = 0;
    uint64_t                                  client_pid_starttime = 0;  // /proc/<pid>/stat field 22
    std::chrono::steady_clock::time_point    last_heartbeat;
};

// -----------------------------------------------------------------
// Per-device state
// -----------------------------------------------------------------

// One queue partition inside a device: a contiguous range of queues
// physically allocated on a single GPU. queue_allocated is the per-queue
// busy bitmap, indexed locally (0..count-1) within this group.
struct DeviceQueueGroup {
    int32_t           cuda_device     = -1;
    int32_t           queue_start_idx = 0;     // absolute queue index (offset into queue_handles)
    int32_t           count           = 0;
    std::vector<bool> queue_allocated;          // size == count
    // GPU-view symlink path that the daemon installed for this group's
    // GPU, e.g. "/mnt/gpu0/snvm_nvme0n1" -> "/mnt/nvme0/GPU0". Empty
    // if symlink installation failed; allocate() copies it into the
    // grant so the client can see the right mount path.
    std::string       gpu_view_path;
};

struct DeviceState {
    int32_t                     device_id        = -1;
    std::string                 pci_addr;
    std::string                 snvme_dev_path;      // /dev/snvm_*
    std::string                 mount_path;          // real NVMe mount, e.g. "/mnt/nvme0"
    uint64_t                    bar0_size        = 0;
    uint32_t                    dstrd            = 0;
    uint32_t                    namespace_id     = 0;
    uint32_t                    page_size        = 0;
    uint32_t                    blk_size         = 0;
    uint32_t                    blk_size_log     = 0;
    uint32_t                    queue_depth      = 0;
    int32_t                     total_queues     = 0;

    std::shared_ptr<Controller> controller;          // libnvm handle

    std::vector<QueueShared>      queue_handles;     // size == total_queues
    std::vector<DeviceQueueGroup> groups;             // per-GPU partitions

    // Symlinks created at init_device time, removed in dtor.
    // Each entry is the absolute symlink path under a GpuEntry.mount_path,
    // e.g. "/mnt/gpu0/snvm_nvme0n1". The corresponding subdirectory on
    // the NVMe (e.g. "/mnt/nvme0/GPU0") is also tracked for best-effort
    // rmdir on shutdown.
    std::vector<std::string>    created_symlinks;
    std::vector<std::string>    created_nvme_subdirs;
};

// -----------------------------------------------------------------
// Public service state
// -----------------------------------------------------------------

class ServiceState {
public:
    /**
     * Initialise all devices from config: open Controller, populate
     * queue_handles. Throws std::runtime_error on failure.
     */
    explicit ServiceState(const ServiceConfig& cfg);
    ~ServiceState();

    ServiceState(const ServiceState&)            = delete;
    ServiceState& operator=(const ServiceState&) = delete;

    // Start the background reaper thread (lease expiry + PID liveness).
    void start_reaper();
    void stop_reaper();

    // --- Query ---

    std::vector<DeviceSnapshot> list_devices() const;

    // --- Allocation lifecycle ---

    struct AllocResult {
        bool            success = false;
        std::string     error;
        AllocationGrant grant;
    };

    /**
     * Allocate a contiguous queue range on the specified device.
     * num_queues == 0 means "use daemon default". The count is clamped to
     * the configured max and to what's actually available.
     */
    AllocResult allocate(int32_t device_id,
                          int32_t cuda_device,
                          int32_t num_queues,
                          uint32_t client_pid);

    bool release(const std::string& allocation_id,
                 uint32_t client_pid,
                 std::string* error);

    /**
     * Update last_heartbeat for allocation_id.
     * Returns false (with error set) if the allocation is unknown.
     */
    bool update_heartbeat(const std::string& allocation_id, std::string* error);

    /**
     * Check whether an allocation exists (e.g. for heartbeat stream setup).
     */
    bool has_allocation(const std::string& allocation_id) const;

private:
    // --- Init helpers ---
    // Build one DeviceState from an NvmeEntry. The full gpus vector is
    // needed to translate each queue_groups[].gpu_id into its
    // GpuEntry.mount_path for symlink setup.
    void init_device(const std::vector<GpuEntry>& gpus,
                     const NvmeEntry& nvme,
                     int32_t device_id);
    void init_queue_handles(DeviceState& dev);

    // Setup / teardown of GPU-view symlinks (best effort on teardown).
    void install_gpu_symlinks(DeviceState& dev,
                              const std::vector<GpuEntry>& gpus,
                              const NvmeEntry& nvme);
    void remove_gpu_symlinks(DeviceState& dev);

    // --- Queue range reservation (group-local) ---
    // reserve_range searches `group` for a contiguous run of `count`
    // free queues. On success returns true and writes the *absolute*
    // queue start index (group.queue_start_idx + local offset) plus the
    // granted count.
    bool reserve_range(DeviceQueueGroup& group, int32_t count,
                       int32_t* out_start, int32_t* out_count);
    // release_range takes an absolute queue start and a count; the
    // owning group is determined from `start` falling inside one of
    // dev.groups.
    void release_range(DeviceState& dev, int32_t start, int32_t count);

    // --- Reaper ---
    void reaper_loop();
    bool is_pid_dead(uint32_t pid, uint64_t recorded_starttime) const;

    // --- Helpers ---
    static std::string generate_allocation_id();
    static std::optional<uint64_t> read_pid_starttime(uint32_t pid);

    // --- Data members ---
    ServiceConfig                                   cfg_;
    std::vector<DeviceState>                        devices_;
    std::unordered_map<std::string, Allocation>    allocations_;
    mutable std::mutex                              state_mtx_;

    std::thread                                     reaper_thread_;
    std::atomic<bool>                               reaper_running_{false};
};

} // namespace nvmeservice

#endif // __NVMESERVICE_STATE_H__
