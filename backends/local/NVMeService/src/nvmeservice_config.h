#ifndef __NVMESERVICE_CONFIG_H__
#define __NVMESERVICE_CONFIG_H__

/**
 * nvmeservice_config.h -- sys_config.yaml parser for NVMeService daemon.
 *
 * Unit conventions
 * ----------------
 * Throughout the daemon we keep ONE unit boundary, and only one:
 *
 *   * The YAML file talks to humans in QueuePair units (1 pair = 1 SQ +
 *     1 CQ). queue_groups[].count and total_queues are both QPs.
 *   * NvmeEntry::queue_setup is the snvme NVM_SET_IOQ_NUM payload (see
 *     backends/local/nvme/libnvm/include/ioctl.h). Its groups[].count
 *     is in kernel ABI units (SQ + CQ entries = 2 * QP). The YAML
 *     parser does the * 2 translation once at parse time, so every
 *     downstream consumer of NvmeEntry already speaks kernel units.
 *
 * The daemon never re-translates units after parse_config_file
 * returns: init_device forwards NvmeEntry::queue_setup straight into
 * Controller's explicit-setup ctor, and the per-GPU partition for
 * symlinks / DeviceQueueGroup is read off setup.groups[] directly.
 *
 * YAML schema (user view, QP units):
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
 *       total_queues: 128            # pool size, in QueuePairs
 *
 *       # Required: split this NVMe pool across one or more GPUs.
 *       # Each entry binds a contiguous range of `count` QueuePairs
 *       # to one GPU. Sum of count must satisfy the local invariant:
 *       #   sum(queue_groups[].count) + queue_setup.kernel_ioq_cap
 *       #       <= total_queues       (all QueuePair units)
 *       # gpu_id == -1 reserves the range for host/CPU memory (API +
 *       # YAML placeholder; libnvm currently rejects with ENOTSUP at
 *       # init).
 *       #
 *       # CONTRACT: queue_group order determines absolute queue numbers.
 *       # The first group occupies queues [0, count_0); the second
 *       # occupies [count_0, count_0 + count_1); and so on. Reordering
 *       # queue_groups in YAML therefore reshuffles which queue ID
 *       # belongs to which GPU -- treat the order as part of the
 *       # configuration's ABI for any client that pins queue IDs.
 *       queue_groups:
 *         - { gpu_id: 0, count: 64 } # 64 QueuePairs on GPU 0
 *         - { gpu_id: 1, count: 64 }
 *
 *       # OPTIONAL: snvme queue-budget tuning. Maps 1:1 onto the
 *       # kernel's struct nvm_ioctl_setup (see
 *       # backends/local/nvme/libnvm/include/ioctl.h). When omitted
 *       # the daemon submits a default setup equivalent to
 *       # {kernel_ioq_cap: 0, on_host: false, nr_write: 0, nr_poll: 0},
 *       # which lets the kernel pick num_possible_cpus() for the
 *       # kernel-side IOQ count and uses queue_groups above as the
 *       # per-owner partition.
 *       queue_setup:
 *         kernel_ioq_cap: 32         # 0 = kernel default (num_possible_cpus())
 *         on_host: false             # false = GPU memory, true = host RAM (reserved)
 *         nr_write: 0                # 0 = leave at module default
 *         nr_poll: 0                 # 0 = leave at module default
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

// We keep struct nvm_ioctl_setup as the single source of truth for the
// queue-setup payload. Pulling the kernel-facing UAPI header into the
// config TU is fine because every consumer of NvmeEntry already needs
// libnvm anyway (the daemon is the only consumer).
#include "ioctl.h"

namespace nvmeservice {

struct GrpcConfig {
    std::string endpoint = "127.0.0.1:50051";
};

struct GpuEntry {
    int         id = -1;
    std::string mount_path;
};

struct NvmeEntry {
    std::string             pci_addr;
    std::string             mount_path;          // real NVMe mount, e.g. "/mnt/nvme0"
    uint32_t                namespace_id = 1;
    uint64_t                queue_depth  = 1024;
    // Maximum number of QueuePairs the daemon will claim on this NVMe
    // (user budget, in QueuePair units). The local invariant the
    // parser enforces is:
    //
    //     sum(queue_setup.groups[].count)/2 + queue_setup.cap_kernel_ioq
    //         <= total_queues
    //
    // The hardware ceiling (controller-reported max queues) is checked
    // by the kernel at NVM_SET_IOQ_NUM time. total_queues stays in
    // QueuePair units because operators reason about queues that way.
    uint64_t                total_queues = 128;

    // Authoritative snvme queue-budget payload, in kernel ABI units:
    //   * groups[].count is SQ + CQ entries (= 2 * QueuePair count)
    //   * cap_kernel_ioq is in QueuePair units (matches snvme spec
    //     "I/O queue count" semantics; pci.c::nvme_max_io_queues)
    //
    // The YAML parser performs the QP -> SQ+CQ translation once at
    // parse time, so init_device can forward this struct to libnvm's
    // Controller ctor verbatim (no further rewriting).
    //
    // groups[].owner_id stores the GPU CUDA device index from
    // queue_groups[].gpu_id (gpu_id < 0 placeholders are dropped at
    // parse time -- see parse_queue_groups). nr_groups counts how many
    // entries in groups[] are valid.
    //
    // Default-constructed (zeroed) means "no groups yet"; the parser
    // always populates at least one group, so an NvmeEntry returned
    // from parse_config_file with nr_groups == 0 indicates a YAML
    // queue_groups list that contained only CPU placeholders -- in
    // which case validate_config will have already rejected the file.
    struct nvm_ioctl_setup  queue_setup{};

    // Convenience: stash the original gpu_id values from YAML so we
    // can keep host (gpu_id < 0) placeholders for symlink/error-path
    // bookkeeping after the * 2 translation lands them in
    // queue_setup.groups[].owner_id (which is uint32_t and cannot
    // express a negative sentinel). Vector size mirrors the YAML
    // queue_groups list, INCLUDING gpu_id < 0 entries that are not
    // copied into queue_setup.groups[].
    //
    // Each entry is { gpu_id, qp_count } in QueuePair units. Used by:
    //   - validate_config (cross-check gpu_id against gpus[])
    //   - install_gpu_symlinks (skip CPU placeholders, mkdir per GPU)
    //
    // init_device does NOT use this for queue allocation -- it walks
    // queue_setup.groups[] directly.
    struct YamlQueueGroup {
        int      gpu_id = -1;
        uint32_t count  = 0;     // QueuePair units (as written in YAML)
    };
    std::vector<YamlQueueGroup> yaml_queue_groups;
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
 *
 * On success NvmeEntry::queue_setup is fully populated in kernel ABI
 * units (groups[].count == 2 * QP), ready for libnvm.
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
