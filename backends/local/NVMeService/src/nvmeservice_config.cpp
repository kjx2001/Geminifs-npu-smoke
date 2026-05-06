#include "nvmeservice_config.h"

#include <yaml-cpp/yaml.h>
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

void parse_queue_groups(const YAML::Node& nvme_node,
                        std::vector<QueueGroup>& out) {
    if (!nvme_node["queue_groups"]) return;
    for (const auto& g : nvme_node["queue_groups"]) {
        QueueGroup qg;
        qg.gpu_id = get_or<int>(g, "gpu_id", -1);
        qg.count  = get_or<int>(g, "count",  0);
        out.push_back(qg);
    }
}

void parse_nvmes(const YAML::Node& root, std::vector<NvmeEntry>& out) {
    if (!root["nvmes"]) return;
    for (const auto& node : root["nvmes"]) {
        NvmeEntry e;
        e.pci_addr     = get_or<std::string>(node, "pci_addr",     "");
        e.mount_path   = get_or<std::string>(node, "mount_path",   "");
        e.namespace_id = get_or<uint32_t>   (node, "namespace_id", 1u);
        e.queue_depth  = get_or<uint64_t>   (node, "queue_depth",  1024ull);
        e.total_queues = get_or<uint64_t>   (node, "total_queues", 128ull);
        parse_queue_groups(node, e.queue_groups);
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

        // queue_groups: required and non-empty. Counts must NOT exceed
        // total_queues (under-using the pool is allowed -- unbound
        // queues simply stay idle). Every gpu_id must reference a
        // known GPU, and no duplicate gpu_id within this nvme.
        if (n.queue_groups.empty()) {
            std::ostringstream ss;
            ss << "nvmes[pci=" << n.pci_addr
               << "] has no queue_groups";
            return emit(ss.str());
        }
        std::set<int> group_gpu_ids;
        uint64_t group_count_sum = 0;
        for (const auto& g : n.queue_groups) {
            if (g.count <= 0) {
                std::ostringstream ss;
                ss << "nvmes[pci=" << n.pci_addr
                   << "].queue_groups[gpu_id=" << g.gpu_id
                   << "].count must be > 0 (got " << g.count << ")";
                return emit(ss.str());
            }
            // gpu_id < 0 is the host/CPU placeholder (API + YAML reserved
            // for future CPU-resident queues; libnvm rejects with ENOTSUP
            // at init time). Skip the gpus[] cross-check for it.
            if (g.gpu_id >= 0 && gpu_ids.find(g.gpu_id) == gpu_ids.end()) {
                std::ostringstream ss;
                ss << "nvmes[pci=" << n.pci_addr
                   << "].queue_groups[].gpu_id=" << g.gpu_id
                   << " has no matching entry in gpus[]";
                return emit(ss.str());
            }
            if (!group_gpu_ids.insert(g.gpu_id).second) {
                std::ostringstream ss;
                ss << "nvmes[pci=" << n.pci_addr
                   << "].queue_groups has duplicate gpu_id=" << g.gpu_id;
                return emit(ss.str());
            }
            group_count_sum += static_cast<uint64_t>(g.count);
        }
        if (group_count_sum > n.total_queues) {
            std::ostringstream ss;
            ss << "nvmes[pci=" << n.pci_addr
               << "]: sum of queue_groups[].count (" << group_count_sum
               << ") exceeds total_queues (" << n.total_queues << ")";
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
