#include "nvmeservice_state.h"

namespace nvmeservice {

void ServiceState::reset() {
    mount_base_path.clear();
    controllers.clear();
    runtime_controllers.clear();
    max_queues_per_process = 32;
}

bool ServiceState::applyAdminInit(const AdminInitReq& req, const CtrlConfig* ctrls, uint32_t count) {
    reset();
    mount_base_path = req.mount_base_path;
    controllers.resize(count);
    runtime_controllers.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        controllers[i].config = ctrls[i];
        controllers[i].free_queues.clear();
        controllers[i].per_process_alloc.clear();
        for (uint32_t q = 0; q < ctrls[i].num_queues; ++q) {
            controllers[i].free_queues.push_back(q);
        }

        nvme_ctrl_param params{};
        params.mount_path = ctrls[i].mount_path;
        params.pci_addr = ctrls[i].pci_addr;
        params.cudaDevice = static_cast<int>(ctrls[i].cuda_device);
        params.ns_id = ctrls[i].ns_id;
        params.queueDepth = ctrls[i].queue_depth;
        params.numQueues = ctrls[i].num_queues;
        params.maxIOsize = ctrls[i].max_io_kb;

        try {
            runtime_controllers.push_back(std::make_shared<NVMeController>(params));
        } catch (...) {
            reset();
            return false;
        }
    }
    return true;
}

bool ServiceState::allocateQueues(uint32_t controller_index, int32_t pid, uint32_t count, std::vector<uint32_t>& out) {
    if (controller_index >= controllers.size()) return false;
    if (count == 0) return false;

    auto& ctrl = controllers[controller_index];
    auto& proc_count = ctrl.per_process_alloc[pid];
    if (proc_count + count > max_queues_per_process) return false;
    if (ctrl.free_queues.size() < count) return false;

    out.clear();
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        out.push_back(ctrl.free_queues.front());
        ctrl.free_queues.pop_front();
    }
    proc_count += count;
    return true;
}

bool ServiceState::releaseQueues(uint32_t controller_index, int32_t pid, const uint32_t* queue_ids, uint32_t count) {
    if (controller_index >= controllers.size()) return false;
    auto& ctrl = controllers[controller_index];

    auto it = ctrl.per_process_alloc.find(pid);
    if (it == ctrl.per_process_alloc.end() || it->second < count) return false;

    for (uint32_t i = 0; i < count; ++i) {
        ctrl.free_queues.push_back(queue_ids[i]);
    }
    it->second -= count;
    if (it->second == 0) {
        ctrl.per_process_alloc.erase(it);
    }
    return true;
}

} // namespace nvmeservice
