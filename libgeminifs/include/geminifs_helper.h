#ifndef GEMINIFS_HELPER_H
#define GEMINIFS_HELPER_H

#include <vector>
#include <string>
#include "geminifs.h"




// PCI BDF地址结构体
struct PCI_BDF {
    uint16_t domain = 0;  // 域（可选）
    uint8_t bus;          // 总线号
    uint8_t device;       // 设备号
    uint8_t function;     // 功能号

    PCI_BDF(const std::string& str) {
        size_t colon1 = str.find(':');
        size_t colon2 = str.find(':', colon1+1);
        size_t dot = str.find('.');
        
        // 处理带域的情况（如0000:50:00.0）
        if (colon2 != std::string::npos) {
            domain = static_cast<uint16_t>(std::stoul(str.substr(0, colon1), 0, 16));
            bus = static_cast<uint8_t>(std::stoul(str.substr(colon1+1, colon2-colon1-1), 0, 16));
            device = static_cast<uint8_t>(std::stoul(str.substr(colon2+1, dot-colon2-1), 0, 16));
        } else { // 常规BDF（如50:00.0）
            bus = static_cast<uint8_t>(std::stoul(str.substr(0, colon1), 0, 16));
            device = static_cast<uint8_t>(std::stoul(str.substr(colon1+1, dot-colon1-1), 0, 16));
        }
        function = static_cast<uint8_t>(std::stoul(str.substr(dot+1), 0, 16));
    }
};

struct SystemConfig {
    std::string root_path;
    unsigned cluster_gpus;
    std::vector<unsigned> gpu_ids;
    std::vector<PCI_BDF> gpu_pci_addresses;
    unsigned cluster_disks;
    std::vector<PCI_BDF> nvme_pci_addresses;
};

struct system_overview {
    std::vector<geminifs_ctrl_params> overview;
    std::vector<PCI_BDF> remote_disks; //  disk not included in same pcie switch with GPU
    SystemConfig cfg;
};

std::vector<std::string> split(const std::string& s, char delimiter);
system_overview parseSystemOverview(const std::string& filepath);
void printSystemOverview(const system_overview& sys);
int calculate_pci_distance(const PCI_BDF& bdf1, const PCI_BDF& bdf2);
#endif