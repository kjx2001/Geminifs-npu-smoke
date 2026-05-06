#ifndef __NVMESERVICE_CONFIG_H__
#define __NVMESERVICE_CONFIG_H__

/**
 * nvmeservice_config.h -- sys_config.yaml parser for NVMeService daemon.
 *
 * YAML schema:
 *   grpc:
 *     endpoint: "127.0.0.1:50051"
 *
 *   gpus:
 *     - id: 0
 *       mount_path: "/mnt/gpu0"      # GPU view directory; daemon
 *                                    # creates per-NVMe symlinks here
 *
 *   nvmes:
 *     - pci_addr: "0000:50:00.0"
 *       mount_path: "/mnt/nvme0"     # real NVMe block-device mount
 *       namespace_id: 1
 *       queue_depth: 1024
 *       total_queues: 128            # pool size
 *
 *       # Required: split this NVMe pool across one or more GPUs.
 *       # Each entry binds a contiguous range of queues to one GPU.
 *       # Sum of count must be <= total_queues; any leftover queues
 *       # stay idle (not bound to any GPU). gpu_id == -1 reserves the
 *       # range for host/CPU memory (API + YAML placeholder; libnvm
 *       # currently rejects with ENOTSUP at init).
 *       #
 *       # CONTRACT: queue_group order determines absolute queue numbers.
 *       # The first group occupies queues [0, count_0); the second
 *       # occupies [count_0, count_0 + count_1); and so on. Reordering
 *       # queue_groups in YAML therefore reshuffles which queue ID
 *       # belongs to which GPU -- treat the order as part of the
 *       # configuration's ABI for any client that pins queue IDs.
 *       queue_groups:
 *         - { gpu_id: 0, count: 64 }
 *         - { gpu_id: 1, count: 64 }
 *
 *   queue_pool:
 *     default_per_client: 32
 *     max_per_client: 128
 *
 *   lease:
 *     heartbeat_interval_sec: 10
 *     timeout_sec: 30
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nvmeservice {

struct GrpcConfig {
    std::string endpoint = "127.0.0.1:50051";
};

struct GpuEntry {
    int         id = -1;
    std::string mount_path;
};

// One queue partition inside a single NVMe: a contiguous range of
// `count` IO queues whose SQ/CQ memory is allocated on `gpu_id`.
struct QueueGroup {
    int gpu_id = -1;
    int count  = 0;
};

struct NvmeEntry {
    std::string             pci_addr;
    std::string             mount_path;          // real NVMe mount, e.g. "/mnt/nvme0"
    uint32_t                namespace_id = 1;
    uint64_t                queue_depth  = 1024;
    uint64_t                total_queues = 128;  // size of the queue pool on this device

    // Per-GPU queue partition. Required: at least one group.
    // Each group binds `count` consecutive queues to one GPU.
    std::vector<QueueGroup> queue_groups;
};

struct QueuePoolConfig {
    int default_per_client = 32;
    int max_per_client     = 128;
};

struct LeaseConfig {
    uint32_t heartbeat_interval_sec = 10;
    uint32_t timeout_sec            = 30;
};

struct ServiceConfig {
    GrpcConfig              grpc;
    std::vector<GpuEntry>   gpus;
    std::vector<NvmeEntry>  nvmes;
    QueuePoolConfig         queue_pool;
    LeaseConfig             lease;
};

/**
 * Parse a YAML file into ServiceConfig. Returns std::nullopt on parse error;
 * the error message is written to *error when non-null.
 */
std::optional<ServiceConfig> parse_config_file(const std::string& path,
                                                std::string* error = nullptr);

/**
 * Validate that cross-references are consistent (every nvme.gpu_id has a
 * matching gpus[].id, no duplicate pci_addr, pool sizes sane, etc.).
 *
 * Returns true if valid; writes a human-readable message to *error on failure.
 */
bool validate_config(const ServiceConfig& cfg, std::string* error = nullptr);

} // namespace nvmeservice

#endif // __NVMESERVICE_CONFIG_H__
