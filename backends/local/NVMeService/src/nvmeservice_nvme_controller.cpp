#include "nvmeservice_nvme_controller.h"

#include <filesystem>
#include <stdexcept>

namespace nvmeservice {

// Static path for SNVMe control device
static constexpr const char* kSnvmeControlPath = "/dev/snvm_control";

NVMeController::NVMeController(const nvme_ctrl_param& params)
    : mount_path_(params.mount_path) {
    // Set maximum I/O size (KB -> bytes)
    maxIOsize_ = params.maxIOsize * 1024;

    // Validate max IO size
    if (params.maxIOsize > 1024) {
        throw std::runtime_error("maxIOsize exceeds 1024 KB");
    }
    if (maxIOsize_ % 4096 != 0) {
        throw std::runtime_error("maxIOsize must be 4K aligned");
    }

    // Create mount directory if it doesn't exist
    std::filesystem::create_directories(mount_path_);

    // Initialize controller (this will create queues and mount filesystem via libnvm)
    controller_ = std::make_shared<Controller>(
        kSnvmeControlPath,
        params.pci_addr.c_str(),
        mount_path_.c_str(),
        params.ns_id,
        params.cudaDevice,
        params.queueDepth,
        params.numQueues);

    if (!controller_) {
        throw std::runtime_error("Failed to create Controller");
    }

    is_initialized_ = true;
}

NVMeController::~NVMeController() = default;

} // namespace nvmeservice
