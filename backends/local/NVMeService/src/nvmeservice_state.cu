#include "nvmeservice_state.h"

#include <filesystem>
#include <iostream>

#include "ctrl.h"

namespace nvmeservice {

namespace {

std::string resolve_mount_path(const std::string& base, const char* leaf) {
    if (base.empty()) {
        return std::string(leaf);
    }
    std::filesystem::path base_path(base);
    std::filesystem::path leaf_path(leaf);
    if (leaf_path.is_absolute()) {
        return leaf_path.string();
    }
    return (base_path / leaf_path).string();
}

} // namespace

bool ServiceState::applyAdminInitAll(const std::vector<CtrlConfig>& ctrls, const std::string& mount_base_path) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    controllers.clear();
    this->mount_base_path = mount_base_path;

    if (ctrls.empty()) {
        std::cerr << "NVMeService: No NVMe controllers parsed from config" << std::endl;
        return false;
    }

    controllers.reserve(ctrls.size());
    for (const auto& cfg : ctrls) {
        ControllerEntry entry{};
        entry.config = cfg;

        std::string mount_path = resolve_mount_path(this->mount_base_path, cfg.mount_path);

        try {
            entry.controller = std::make_shared<Controller>(
                "/dev/snvm_control",
                cfg.pci_addr,
                mount_path,
                cfg.ns_id,
                cfg.cuda_device,
                cfg.queue_depth,
                cfg.num_queues);
        } catch (...) {
            controllers.clear();
            return false;
        }

        if (!entry.controller) {
            controllers.clear();
            return false;
        }

        entry.free_qids.clear();
        entry.free_qids.reserve(entry.controller->n_qps);
        for (uint32_t qid = 0; qid < entry.controller->n_qps; ++qid) {
            entry.free_qids.push_back(qid);
        }

        controllers.push_back(std::move(entry));
    }

    return true;
}

bool ServiceState::allocQueues(uint32_t controller_index, uint32_t count, int32_t pid, std::vector<uint32_t>& out_qids) {
    (void)pid;
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (controller_index >= controllers.size()) {
        return false;
    }

    auto& entry = controllers[controller_index];
    if (count == 0 || entry.free_qids.empty()) {
        return false;
    }

    uint32_t granted = std::min<uint32_t>(count, static_cast<uint32_t>(entry.free_qids.size()));
    out_qids.clear();
    out_qids.reserve(granted);
    for (uint32_t i = 0; i < granted; ++i) {
        out_qids.push_back(entry.free_qids.back());
        entry.free_qids.pop_back();
    }

    return !out_qids.empty();
}

bool ServiceState::releaseQueues(uint32_t controller_index, int32_t pid, const std::vector<uint32_t>& qids) {
    (void)pid;
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (controller_index >= controllers.size()) {
        return false;
    }

    auto& entry = controllers[controller_index];
    for (uint32_t qid : qids) {
        entry.free_qids.push_back(qid);
    }

    return true;
}

} // namespace nvmeservice
