#include "nvmeservice_config.h"

#include <yaml-cpp/yaml.h>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace nvmeservice {

namespace {

template <typename T>
T get_or(const YAML::Node& n, const std::string& key, T def) {
    if (!n || !n[key]) return def;
    return n[key].as<T>();
}

void parse_grpc(const YAML::Node& root, GrpcConfig& out) {
    if (!root["grpc"]) return;
    const auto& g = root["grpc"];
    out.endpoint = get_or<std::string>(g, "endpoint", out.endpoint);
}

void parse_gpus(const YAML::Node& root, std::vector<GpuEntry>& out) {
    if (!root["gpus"]) return;
    for (const auto& node : root["gpus"]) {
        GpuEntry e;
        e.id         = get_or<int>        (node, "id",         -1);
        e.mount_path = get_or<std::string>(node, "mount_path", "");
        out.push_back(std::move(e));
    }
}

// Translate the YAML queue_groups list (QueuePair units, gpu_id may be
// negative for CPU placeholders) into the kernel-facing
// nvm_ioctl_setup.groups[] (SQ+CQ entry units, owner_id = gpu_id).
//
// Side effects:
//   * Fills out.yaml_queue_groups verbatim (preserves CPU placeholders
//     and the original QP unit so symlink installation / validation
//     can keep using the YAML view).
//   * Populates out.queue_setup.groups[] for non-CPU entries only,
//     setting nr_groups accordingly. CPU placeholders are dropped from
//     groups[] because the kernel cannot express them today (libnvm
//     rejects on_host=true at init), but the validator below still
//     produces a friendly error if every YAML entry was a placeholder.
//   * Caps the kernel-side groups[] copy at NVM_MAX_QUEUE_GROUPS;
//     validate_config catches the overflow with a clearer message.
void parse_queue_groups(const YAML::Node& nvme_node, NvmeEntry& out) {
    if (!nvme_node["queue_groups"]) return;

    uint32_t kernel_idx = 0;
    for (const auto& g : nvme_node["queue_groups"]) {
        NvmeEntry::YamlQueueGroup yg;
        yg.gpu_id = get_or<int>     (g, "gpu_id", -1);
        // Negative count would underflow when we cast to uint32_t; let
        // validate_config emit the user-facing error rather than wrap
        // here.
        const int raw_count = get_or<int>(g, "count", 0);
        yg.count = (raw_count > 0) ? static_cast<uint32_t>(raw_count) : 0u;
        out.yaml_queue_groups.push_back(yg);

        if (yg.gpu_id < 0) continue;                          // CPU placeholder
        if (kernel_idx >= NVM_MAX_QUEUE_GROUPS) continue;     // validator flags

        out.queue_setup.groups[kernel_idx].owner_id  =
            static_cast<uint32_t>(yg.gpu_id);
        // QueuePair count -> SQ + CQ entry count (kernel ABI). 2 * QP.
        out.queue_setup.groups[kernel_idx].count     = yg.count * 2u;
        out.queue_setup.groups[kernel_idx].numa_node = -1;    // doc-only hint
        out.queue_setup.groups[kernel_idx].reserved  = 0;
        ++kernel_idx;
    }
    out.queue_setup.nr_groups = kernel_idx;
}

// Optional block. Missing block leaves cap_kernel_ioq / nr_write /
// nr_poll at zero (= "kernel default"). on_host maps onto the
// NVM_QUEUE_SETUP_F_ON_HOST flag bit.
void parse_queue_setup(const YAML::Node& nvme_node, NvmeEntry& out) {
    if (!nvme_node["queue_setup"]) return;
    const auto& s = nvme_node["queue_setup"];

    out.queue_setup.cap_kernel_ioq =
        get_or<uint32_t>(s, "kernel_ioq_cap", out.queue_setup.cap_kernel_ioq);
    out.queue_setup.nr_write       =
        get_or<uint32_t>(s, "nr_write",       out.queue_setup.nr_write);
    out.queue_setup.nr_poll        =
        get_or<uint32_t>(s, "nr_poll",        out.queue_setup.nr_poll);

    const bool on_host = get_or<bool>(s, "on_host", false);
    if (on_host) {
        out.queue_setup.flags |= NVM_QUEUE_SETUP_F_ON_HOST;
    } else {
        out.queue_setup.flags &= ~NVM_QUEUE_SETUP_F_ON_HOST;
    }
}

void parse_nvmes(const YAML::Node& root, std::vector<NvmeEntry>& out) {
    if (!root["nvmes"]) return;
    for (const auto& node : root["nvmes"]) {
        NvmeEntry e;
        // Zero-init the kernel payload so any field we don't touch
        // (reserved[], ioq_num until libnvm fills it, groups beyond
        // nr_groups) is well-defined on the wire.
        std::memset(&e.queue_setup, 0, sizeof(e.queue_setup));

        e.pci_addr     = get_or<std::string>(node, "pci_addr",     "");
        e.mount_path   = get_or<std::string>(node, "mount_path",   "");
        e.namespace_id = get_or<uint32_t>   (node, "namespace_id", 1u);
        e.queue_depth  = get_or<uint64_t>   (node, "queue_depth",  1024ull);
        e.total_queues = get_or<uint64_t>   (node, "total_queues", 128ull);
        parse_queue_groups(node, e);
        parse_queue_setup (node, e);
        out.push_back(std::move(e));
    }
}

void parse_pool(const YAML::Node& root, QueuePoolConfig& out) {
    if (!root["queue_pool"]) return;
    const auto& q = root["queue_pool"];
    out.default_per_client = get_or<int>(q, "default_per_client", out.default_per_client);
    out.max_per_client     = get_or<int>(q, "max_per_client",     out.max_per_client);
}

void parse_lease(const YAML::Node& root, LeaseConfig& out) {
    if (!root["lease"]) return;
    const auto& l = root["lease"];
    out.heartbeat_interval_sec = get_or<uint32_t>(l, "heartbeat_interval_sec", out.heartbeat_interval_sec);
    out.timeout_sec            = get_or<uint32_t>(l, "timeout_sec",            out.timeout_sec);
}

} // namespace

std::optional<ServiceConfig> parse_config_file(const std::string& path,
                                                std::string* error) {
    try {
        YAML::Node root = YAML::LoadFile(path);
        ServiceConfig cfg;

        parse_grpc (root, cfg.grpc);
        parse_gpus (root, cfg.gpus);
        parse_nvmes(root, cfg.nvmes);
        parse_pool (root, cfg.queue_pool);
        parse_lease(root, cfg.lease);

        if (std::string verr; !validate_config(cfg, &verr)) {
            if (error) *error = "validation failed: " + verr;
            return std::nullopt;
        }
        return cfg;
    } catch (const YAML::Exception& e) {
        if (error) *error = std::string("YAML parse error: ") + e.what();
        return std::nullopt;
    } catch (const std::exception& e) {
        if (error) *error = std::string("parse error: ") + e.what();
        return std::nullopt;
    }
}

bool validate_config(const ServiceConfig& cfg, std::string* error) {
    auto emit = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };

    if (cfg.grpc.endpoint.empty()) {
        return emit("grpc.endpoint is empty");
    }
    if (cfg.gpus.empty()) {
        return emit("gpus list is empty");
    }
    if (cfg.nvmes.empty()) {
        return emit("nvmes list is empty");
    }

    std::set<int> gpu_ids;
    for (const auto& g : cfg.gpus) {
        if (g.id < 0) {
            return emit("gpus[].id must be >= 0");
        }
        if (!gpu_ids.insert(g.id).second) {
            std::ostringstream ss;
            ss << "duplicate gpus[].id: " << g.id;
            return emit(ss.str());
        }
        if (g.mount_path.empty()) {
            std::ostringstream ss;
            ss << "gpus[id=" << g.id << "].mount_path is empty";
            return emit(ss.str());
        }
    }

    std::set<std::string> pci_seen;
    std::set<std::string> mount_seen;
    for (const auto& n : cfg.nvmes) {
        if (n.pci_addr.empty()) {
            return emit("nvmes[].pci_addr is empty");
        }
        if (!pci_seen.insert(n.pci_addr).second) {
            return emit("duplicate nvmes[].pci_addr: " + n.pci_addr);
        }
        if (n.mount_path.empty()) {
            return emit("nvmes[pci=" + n.pci_addr + "].mount_path is empty");
        }
        if (!mount_seen.insert(n.mount_path).second) {
            return emit("duplicate nvmes[].mount_path: " + n.mount_path);
        }
        if (n.total_queues == 0) {
            std::ostringstream ss;
            ss << "nvmes[pci=" << n.pci_addr << "].total_queues must be > 0";
            return emit(ss.str());
        }

        // queue_groups: required and non-empty (in the YAML view --
        // CPU placeholders count toward this check). Each entry's
        // gpu_id must be -1 (placeholder) or reference a known GPU,
        // count must be > 0, and no duplicate gpu_id.
        if (n.yaml_queue_groups.empty()) {
            std::ostringstream ss;
            ss << "nvmes[pci=" << n.pci_addr
               << "] has no queue_groups";
            return emit(ss.str());
        }
        if (n.yaml_queue_groups.size() > static_cast<size_t>(NVM_MAX_QUEUE_GROUPS)) {
            std::ostringstream ss;
            ss << "nvmes[pci=" << n.pci_addr
               << "].queue_groups has " << n.yaml_queue_groups.size()
               << " entries, exceeds NVM_MAX_QUEUE_GROUPS="
               << NVM_MAX_QUEUE_GROUPS;
            return emit(ss.str());
        }

        std::set<int>  group_gpu_ids;
        uint64_t       group_count_sum = 0;   // QueuePair units
        bool           has_kernel_visible = false;
        for (const auto& g : n.yaml_queue_groups) {
            if (g.count == 0) {
                std::ostringstream ss;
                ss << "nvmes[pci=" << n.pci_addr
                   << "].queue_groups[gpu_id=" << g.gpu_id
                   << "].count must be > 0";
                return emit(ss.str());
            }
            // gpu_id < 0 is the host/CPU placeholder (API + YAML
            // reserved for future CPU-resident queues; libnvm rejects
            // with ENOTSUP at init time). Skip the gpus[] cross-check
            // for it.
            if (g.gpu_id >= 0) {
                if (gpu_ids.find(g.gpu_id) == gpu_ids.end()) {
                    std::ostringstream ss;
                    ss << "nvmes[pci=" << n.pci_addr
                       << "].queue_groups[].gpu_id=" << g.gpu_id
                       << " has no matching entry in gpus[]";
                    return emit(ss.str());
                }
                has_kernel_visible = true;
            }
            if (!group_gpu_ids.insert(g.gpu_id).second) {
                std::ostringstream ss;
                ss << "nvmes[pci=" << n.pci_addr
                   << "].queue_groups has duplicate gpu_id=" << g.gpu_id;
                return emit(ss.str());
            }
            group_count_sum += g.count;
        }
        // libnvm needs at least one GPU-resident group to bring the
        // controller up; an all-CPU placeholder list is rejected
        // upstream at Controller ctor with a less helpful errno, so
        // we surface it here.
        if (!has_kernel_visible) {
            std::ostringstream ss;
            ss << "nvmes[pci=" << n.pci_addr
               << "].queue_groups has no GPU entry (every gpu_id<0); "
                  "libnvm cannot bring up the controller without at "
                  "least one GPU-resident group";
            return emit(ss.str());
        }

        // queue_setup: only checks that can be made without talking to
        // the controller. The hardware-imposed bound
        // `cap_kernel_ioq + sum(groups) <= controller_total_queues`
        // is enforced by the snvme kernel module at NVM_SET_IOQ_NUM
        // time -- the daemon surfaces that as an init failure with
        // the kernel's errno + dmesg context.
        //
        // on_host=true is reserved for the future CPU_SUBMIT path;
        // libnvm rejects it at init today, so we surface a friendlier
        // message here instead of letting Controller construction throw.
        if (n.queue_setup.flags & NVM_QUEUE_SETUP_F_ON_HOST) {
            std::ostringstream ss;
            ss << "nvmes[pci=" << n.pci_addr
               << "].queue_setup.on_host=true is reserved for the future "
                  "CPU_SUBMIT path; libnvm currently only supports "
                  "GPU-resident user queues";
            return emit(ss.str());
        }

        // Local queue-budget invariant: the user-side share
        // (sum(yaml queue_groups[].count)) plus the kernel-side cap
        // (cap_kernel_ioq) must fit inside total_queues, i.e. the
        // controller's hardware IOQ budget the operator is willing
        // to commit on this NVMe. All three values are in QueuePair
        // units. The kernel-side bound is enforced by snvme at
        // NVM_SET_IOQ_NUM time against the controller's actual MQES.
        const uint64_t kernel_cap = n.queue_setup.cap_kernel_ioq;
        if (group_count_sum + kernel_cap > n.total_queues) {
            std::ostringstream ss;
            ss << "nvmes[pci=" << n.pci_addr
               << "]: sum(queue_groups[].count)=" << group_count_sum
               << " + queue_setup.kernel_ioq_cap=" << kernel_cap
               << " exceeds total_queues=" << n.total_queues
               << " (all in QueuePair units)";
            return emit(ss.str());
        }
    }

    if (cfg.queue_pool.default_per_client <= 0 ||
        cfg.queue_pool.max_per_client     <= 0 ||
        cfg.queue_pool.default_per_client >  cfg.queue_pool.max_per_client) {
        return emit("queue_pool: default_per_client must be in (0, max_per_client]");
    }

    if (cfg.lease.heartbeat_interval_sec == 0 ||
        cfg.lease.timeout_sec            == 0 ||
        cfg.lease.heartbeat_interval_sec >= cfg.lease.timeout_sec) {
        return emit("lease: heartbeat_interval_sec must be in (0, timeout_sec)");
    }

    return true;
}

} // namespace nvmeservice
