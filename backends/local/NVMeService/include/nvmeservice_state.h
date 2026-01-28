#ifndef NVMESERVICE_STATE_H
#define NVMESERVICE_STATE_H

#include "nvmeservice_protocol.h"
#include "nvmeservice_nvme_controller.h"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvmeservice {

struct ControllerState {
    CtrlConfig config{};
    std::deque<uint32_t> free_queues;
    std::unordered_map<int32_t, uint32_t> per_process_alloc;
};

struct ServiceState {
    std::string mount_base_path;
    std::vector<ControllerState> controllers;
    std::vector<NVMeControllerPtr> runtime_controllers;
    uint32_t max_queues_per_process = 32;

    void reset();
    bool applyAdminInit(const AdminInitReq& req, const CtrlConfig* ctrls, uint32_t count);
    bool allocateQueues(uint32_t controller_index, int32_t pid, uint32_t count, std::vector<uint32_t>& out);
    bool releaseQueues(uint32_t controller_index, int32_t pid, const uint32_t* queue_ids, uint32_t count);
};

} // namespace nvmeservice

#endif // NVMESERVICE_STATE_H
