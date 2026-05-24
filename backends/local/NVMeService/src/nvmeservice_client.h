#ifndef __NVMESERVICE_CLIENT_H__
#define __NVMESERVICE_CLIENT_H__

/**
 * nvmeservice_client.h -- thin gRPC client + local Controller rebuild.
 *
 * A process that wants to use NVMe queues handed out by the NVMeService daemon:
 *
 *   NvmeServiceClient client("127.0.0.1:50051");
 *   auto alloc = client.allocate(device_id=0, num_queues=32);
 *   // alloc->controller is a libnvm Controller; use like any other
 *   BlockDeviceManager dm(alloc->controller, "/mnt/gpu0");
 *   ...
 *   // alloc goes out of scope -> dtor releases on server + frees local resources
 *
 * Heartbeat is maintained by an internal thread as long as any Allocation
 * is live. No configuration needed: interval/timeout come from the daemon's
 * AllocResponse.
 */

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <grpcpp/grpcpp.h>
#include "nvmeservice.grpc.pb.h"

// Forward declaration -- include libnvm's ctrl.h in the .cpp
struct Controller;

namespace nvmeservice {

/**
 * Queue partition advertised by the daemon for a single device.
 * Each entry maps a contiguous queue range to one cuda_device.
 */
struct ClientQueueGroup {
    int32_t cuda_device     = -1;
    int32_t queue_start_idx = 0;
    int32_t queue_count     = 0;
    int32_t available       = 0;
};

/**
 * DeviceInfo mirror for callers that want to enumerate devices.
 */
struct ClientDeviceInfo {
    int32_t     device_id    = -1;
    std::string pci_addr;
    std::string snvme_dev_path;
    // Legacy single cuda_device field: equals queue_groups[0].cuda_device.
    // Modern callers should iterate `queue_groups` and pass an explicit
    // cuda_device to allocate().
    int32_t     cuda_device  = -1;
    uint32_t    namespace_id = 0;
    uint32_t    page_size    = 0;
    uint32_t    blk_size     = 0;
    uint32_t    blk_size_log = 0;
    uint32_t    queue_depth  = 0;
    int32_t     total_queues     = 0;
    int32_t     available_queues = 0;
    std::vector<ClientQueueGroup> queue_groups;
};

class NvmeServiceClient {
public:
    /**
     * An active queue allocation. Owns the local Controller and all
     * associated resources (IPC imports, BAR0 mmap, local GPU buffers).
     * The destructor sends a ReleaseQueues RPC and cleans up local state.
     */
    struct Allocation {
        std::string                 allocation_id;
        int32_t                     device_id;
        int32_t                     queue_start_idx;
        int32_t                     queue_count;
        std::shared_ptr<Controller> controller;      // libnvm Controller

        // GPU-view filesystem path the daemon installed for this
        // allocation: a symlink under the consuming GPU's mount_path
        // that resolves to the per-GPU subdirectory on the NVMe (e.g.
        // "/mnt/gpu0/snvm_nvme0n1" -> "/mnt/nvme0/GPU0"). Hand this to
        // BlockDeviceManager (or any FileManager-style consumer) so
        // file paths stay GPU-isolated. Empty if symlink installation
        // failed at daemon init time -- callers can fall back to
        // `controller->dev_mount_path` (libnvm carries the same
        // string from the AllocResponse).
        std::string                 mount_path;

        // These are filled by the client but exposed for debugging only.
        uint32_t                    heartbeat_interval_sec;
        uint32_t                    lease_timeout_sec;

        // Client-side-only -- needed for ReleaseQueues RPC.
        uint32_t                    client_pid;

        Allocation() = default;
        ~Allocation();

        Allocation(const Allocation&)            = delete;
        Allocation& operator=(const Allocation&) = delete;

    private:
        friend class NvmeServiceClient;
        NvmeServiceClient* owner = nullptr;
    };

    explicit NvmeServiceClient(const std::string& endpoint);
    ~NvmeServiceClient();

    NvmeServiceClient(const NvmeServiceClient&)            = delete;
    NvmeServiceClient& operator=(const NvmeServiceClient&) = delete;

    /**
     * Enumerate devices exposed by the daemon.
     */
    std::vector<ClientDeviceInfo> list_devices();

    /**
     * Request a contiguous queue range on device_id.
     *
     * num_queues == 0 -> use daemon default (capped by config).
     * Returns nullptr on failure; error details go to stderr.
     *
     * The heartbeat thread is started on first successful allocate() and
     * stopped when the last Allocation is destroyed.
     *
     * The 2-arg overload picks the first queue_group's cuda_device,
     * preserving single-GPU caller behaviour. Multi-GPU clients should
     * use the 3-arg overload to target a specific GPU.
     */
    std::unique_ptr<Allocation> allocate(int32_t device_id, int32_t num_queues);
    std::unique_ptr<Allocation> allocate(int32_t device_id,
                                          int32_t cuda_device,
                                          int32_t num_queues);

private:
    friend struct Allocation;

    // Called by Allocation dtor.
    void release_allocation(Allocation* alloc);

    // Heartbeat thread management.
    void ensure_heartbeat_started();
    void stop_heartbeat();
    void heartbeat_loop();

    std::string                                          endpoint_;
    std::shared_ptr<grpc::Channel>                       channel_;
    std::unique_ptr<NvmeService::Stub>                   stub_;

    // Tracking live allocations for heartbeat.
    struct LiveAlloc {
        std::string allocation_id;
        uint32_t    heartbeat_interval_sec = 10;
    };
    std::mutex                                           live_mtx_;
    std::unordered_map<std::string, LiveAlloc>           live_allocs_;

    std::thread                                          hb_thread_;
    std::atomic<bool>                                    hb_running_{false};
};

} // namespace nvmeservice

#endif // __NVMESERVICE_CLIENT_H__
