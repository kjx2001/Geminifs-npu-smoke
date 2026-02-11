#pragma once

#include <cstdint>

namespace nvmeservice {

struct CtrlConfig {
    char mount_path[256];
    char pci_addr[32];
    uint32_t ns_id;
    uint32_t queue_depth;
    uint32_t num_queues;
    uint32_t cuda_device;
    uint32_t max_io_kb;
};

} // namespace nvmeservice
