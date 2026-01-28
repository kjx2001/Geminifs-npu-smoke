#ifndef NVMESERVICE_NVME_CONTROLLER_H
#define NVMESERVICE_NVME_CONTROLLER_H

#include <cstdint>
#include <memory>
#include <string>

#include <ctrl.h>

namespace nvmeservice {

using ControllerPtr = std::shared_ptr<Controller>;

struct nvme_ctrl_param {
    std::string mount_path;
    std::string pci_addr;
    int cudaDevice;
    uint32_t ns_id;
    uint64_t queueDepth;
    uint64_t numQueues;
    uint64_t maxIOsize;
};

class NVMeController {
public:
    explicit NVMeController(const nvme_ctrl_param& params);
    ~NVMeController();

    NVMeController(const NVMeController&) = delete;
    NVMeController& operator=(const NVMeController&) = delete;

    bool is_initialized() const { return is_initialized_; }

private:
    ControllerPtr controller_;
    std::string mount_path_;
    uint64_t maxIOsize_ = 0;
    bool is_initialized_ = false;
};

using NVMeControllerPtr = std::shared_ptr<NVMeController>;

} // namespace nvmeservice

#endif // NVMESERVICE_NVME_CONTROLLER_H
