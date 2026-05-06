#ifndef __TUTTI_RUNTIME_LEASE_H__
#define __TUTTI_RUNTIME_LEASE_H__

/**
 * lease.h -- time-bounded resource grants the runtime hands out.
 *
 * Layer: Core Runtime (Roadmap.md §3 object model).
 *
 * Role:
 *   - A Lease is a generic "you may use this resource until it expires"
 *     token. The first concrete user is the NVMeService client model
 *     (queue ranges leased per client; reaper reclaims dead ones), but
 *     the abstraction also covers future leases:
 *        * RDMA QP lease (multi-process NIC sharing)
 *        * Mount-point lease (filesystem ownership in shared backends)
 *        * Bandwidth quota lease (QoS scheduling)
 *   - The runtime keeps a heartbeat thread per ILeaseManager; clients
 *     refresh leases by RPC (or in-process function call). Expired
 *     leases are reclaimed proactively.
 *
 * Why a separate header (vs reusing NVMeService::Allocation):
 *   - NVMeService::Allocation is the local_nvme implementation detail;
 *     ILeaseManager is the runtime-facing contract. Different layers,
 *     deliberately decoupled so future backends can plug in their own
 *     lease-issuing logic without rewriting upper layers.
 *
 * Lifetime:
 *   - Lease handles are owned by the requester. The runtime tracks
 *     them by lease_id. Releasing a Lease early via release_lease()
 *     is fine; otherwise the heartbeat keeps it alive.
 *   - The "what was leased" payload is opaque (`payload`); each
 *     resource type defines its own struct stored at that pointer.
 */

#include <cstdint>
#include <chrono>
#include <string>

namespace tutti {

/**
 * Tag identifying which resource shape `payload` carries. New kinds
 * append; numeric values stable.
 */
enum class LeaseKind : uint32_t {
    NVME_QUEUE_RANGE = 0,   // payload = pointer to a nvme-queue-range descriptor
    RDMA_QP          = 1,   // future
    BANDWIDTH_QUOTA  = 2,   // future
};

/**
 * One outstanding lease. Heartbeat-managed.
 *
 * `last_heartbeat` is updated by ILeaseManager::heartbeat() and
 * inspected by the reaper. `client_pid` + `client_pid_starttime` form
 * the (PID, /proc/<pid>/stat starttime) tuple the reaper uses to
 * defeat PID reuse, mirroring the NVMeService design.
 */
struct Lease {
    std::string lease_id;          // opaque, server-issued (typically a hex token)
    LeaseKind   kind;
    int32_t     device_id;         // Device this lease draws from
    uint32_t    client_pid;        // 0 if lease is in-process / no PID identity
    uint64_t    client_pid_starttime;  // /proc/<pid>/stat field 22; 0 if N/A
    std::chrono::steady_clock::time_point last_heartbeat;

    // Lease-kind-specific payload. ILeaseManager-owned for the life
    // of the lease. Cast based on `kind`.
    void* payload;

    // Lease parameters chosen at issue time.
    uint32_t heartbeat_interval_sec;
    uint32_t timeout_sec;
};

/**
 * The lease lifecycle interface. One ILeaseManager per resource pool
 * (NVMeService::ServiceState already implements this contract in
 * spirit; future backends can implement it directly).
 */
class ILeaseManager {
public:
    virtual ~ILeaseManager() = default;

    /// Refresh the lease's heartbeat. Returns false (with error set
    /// if non-null) if the lease is unknown or already revoked.
    virtual bool heartbeat(const std::string& lease_id, std::string* error) = 0;

    /// Release a lease early. Returns false if the lease is unknown
    /// or `client_pid` doesn't match the recorded one (sanity check).
    virtual bool release_lease(const std::string& lease_id,
                                uint32_t           client_pid,
                                std::string*       error) = 0;

    /// Whether `lease_id` is currently outstanding.
    virtual bool has_lease(const std::string& lease_id) const = 0;
};

} // namespace tutti

#endif // __TUTTI_RUNTIME_LEASE_H__
