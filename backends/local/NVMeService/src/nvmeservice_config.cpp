#include "nvmeservice_config.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unordered_map>

#include <yaml-cpp/yaml.h>

namespace nvmeservice {

namespace {

std::string requireString(const YAML::Node& node, const char* key, std::string* error_message) {
    if (!node || !node[key]) {
        if (error_message) {
            *error_message = std::string("Missing key: ") + key;
        }
        return {};
    }
    return node[key].as<std::string>();
}

bool requireUint32(const YAML::Node& node, const char* key, uint32_t& out, std::string* error_message) {
    if (!node || !node[key]) {
        if (error_message) {
            *error_message = std::string("Missing key: ") + key;
        }
        return false;
    }
    out = node[key].as<uint32_t>();
    return true;
}

bool optionalUint32(const YAML::Node& node, const char* key, uint32_t& out) {
    if (!node || !node[key]) {
        return false;
    }
    out = node[key].as<uint32_t>();
    return true;
}

bool parseGrpc(const YAML::Node& root, ParsedConfigAll& out, std::string* error_message) {
    if (!root["grpc"]) {
        if (error_message) {
            *error_message = "Missing grpc section";
        }
        return false;
    }

    const auto grpc = root["grpc"];
    out.grpc_endpoint = grpc["endpoint"] ? grpc["endpoint"].as<std::string>() : kDefaultGrpcEndpoint;
    out.max_queues_per_process = grpc["max_queues_per_process"]
        ? grpc["max_queues_per_process"].as<uint32_t>()
        : kDefaultMaxQueuesPerProcess;
    return true;
}

bool parseGpus(const YAML::Node& root, std::unordered_map<int, std::string>& gpu_mounts, std::string* error_message) {
    if (!root["gpus"] || !root["gpus"].IsSequence()) {
        if (error_message) {
            *error_message = "Missing gpus list";
        }
        return false;
    }

    for (const auto& gpu : root["gpus"]) {
        if (!gpu["id"] || !gpu["mount_path"]) {
            if (error_message) {
                *error_message = "Each gpu entry requires id and mount_path";
            }
            return false;
        }
        int id = gpu["id"].as<int>();
        std::string mount_path = gpu["mount_path"].as<std::string>();
        if (mount_path.empty()) {
            if (error_message) {
                *error_message = "gpu.mount_path cannot be empty";
            }
            return false;
        }
        gpu_mounts[id] = mount_path;
    }

    if (gpu_mounts.empty()) {
        if (error_message) {
            *error_message = "No GPU sections found";
        }
        return false;
    }

    return true;
}

bool parseNvmes(const YAML::Node& root,
               const std::unordered_map<int, std::string>& gpu_mounts,
               std::vector<CtrlConfig>& out_ctrls,
               std::string* error_message) {
    if (!root["nvmes"] || !root["nvmes"].IsSequence()) {
        if (error_message) {
            *error_message = "Missing nvmes list";
        }
        return false;
    }

    for (const auto& nvme : root["nvmes"]) {
        CtrlConfig cfg{};
        std::string mount_path = requireString(nvme, "mount_path", error_message);
        std::string pci_addr = requireString(nvme, "pci_addr", error_message);
        uint32_t ns_id = 0;
        uint32_t queue_depth = 0;
        uint32_t num_queues = 0;
        uint32_t cuda_device = 0;
        uint32_t max_io_kb = 0;

        if (mount_path.empty() || pci_addr.empty()) return false;
        if (!requireUint32(nvme, "ns_id", ns_id, error_message)) return false;
        if (!requireUint32(nvme, "queueDepth", queue_depth, error_message)) return false;
        if (!requireUint32(nvme, "numQueues", num_queues, error_message)) return false;
        if (!optionalUint32(nvme, "gpu_id", cuda_device)) {
            if (!requireUint32(nvme, "cudaDevice", cuda_device, error_message)) return false;
        }
        if (!requireUint32(nvme, "maxIOsize", max_io_kb, error_message)) return false;

        auto gpu_it = gpu_mounts.find(static_cast<int>(cuda_device));
        if (gpu_it == gpu_mounts.end()) {
            if (error_message) {
                *error_message = "No GPU section found for gpu_id=" + std::to_string(cuda_device);
            }
            return false;
        }

        std::filesystem::path leaf_path(mount_path);
        std::filesystem::path base_path(gpu_it->second);
        std::filesystem::path full_path = leaf_path.is_absolute() ? leaf_path : (base_path / leaf_path);

        std::snprintf(cfg.mount_path, sizeof(cfg.mount_path), "%s", full_path.string().c_str());
        std::snprintf(cfg.pci_addr, sizeof(cfg.pci_addr), "%s", pci_addr.c_str());
        cfg.ns_id = ns_id;
        cfg.queue_depth = queue_depth;
        cfg.num_queues = num_queues;
        cfg.cuda_device = cuda_device;
        cfg.max_io_kb = max_io_kb;

        out_ctrls.push_back(cfg);
    }

    if (out_ctrls.empty()) {
        if (error_message) {
            *error_message = "No NVMe controllers found";
        }
        return false;
    }

    return true;
}

} // namespace

bool parseSysConfig(const std::string& path, ParsedConfigAll& out, std::string* error_message) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const std::exception& ex) {
        if (error_message) {
            *error_message = std::string("Failed to load YAML: ") + ex.what();
        }
        return false;
    }

    if (!parseGrpc(root, out, error_message)) {
        return false;
    }

    std::unordered_map<int, std::string> gpu_mounts;
    if (!parseGpus(root, gpu_mounts, error_message)) {
        return false;
    }

    out.ctrls.clear();
    if (!parseNvmes(root, gpu_mounts, out.ctrls, error_message)) {
        return false;
    }

    if (gpu_mounts.size() == 1) {
        out.multi_mount = false;
        out.mount_base_path = gpu_mounts.begin()->second;
    } else {
        out.multi_mount = true;
        out.mount_base_path.clear();
    }

    return true;
}

} // namespace nvmeservice
