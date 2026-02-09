#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "nvmeservice_protocol.h"

struct Controller;

namespace nvmeservice {

struct ControllerEntry {
    CtrlConfig config{};
    std::shared_ptr<Controller> controller;
    std::vector<uint32_t> free_qids;
};

class ServiceState {
public:
    std::string mount_base_path;
    uint32_t max_queues_per_process = 0;

    bool applyAdminInitAll(const std::vector<CtrlConfig>& ctrls, const std::string& mount_base_path);
    bool allocQueues(uint32_t controller_index, uint32_t count, int32_t pid, std::vector<uint32_t>& out_qids);
    bool releaseQueues(uint32_t controller_index, int32_t pid, const std::vector<uint32_t>& qids);

    std::vector<ControllerEntry> controllers;

private:
    std::mutex state_mutex_;
};

} // namespace nvmeservice
