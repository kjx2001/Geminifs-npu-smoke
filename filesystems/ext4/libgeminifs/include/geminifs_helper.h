#ifndef GEMINIFS_HELPER_H
#define GEMINIFS_HELPER_H

#include <vector>
#include <string>
#include <sys/resource.h>
#include "geminifs.h"

// Configuration parsing structures
struct GPUConfig {
    std::string mount_path;
    int cudaDevice;
};

struct NVMeConfig {
    std::string mount_path;
    std::string pci_addr;
    uint32_t ns_id;
    uint64_t queueDepth;
    uint64_t numQueues;
    int cudaDevice;
    uint64_t maxIOsize;
};

struct SystemConfigGroup {
    GPUConfig gpu;
    std::vector<NVMeConfig> nvmes;
};

struct ParsedSystemConfig {
    std::vector<SystemConfigGroup> groups;
    bool valid;
    std::string error_message;
};


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

// File descriptor limit management functions
bool increase_fd_limit(rlim_t desired_limit);
void auto_configure_fd_limits(int num_files_to_open = 1000);

// Memory alignment utility functions
bool is_aligned(uint64_t value, size_t alignment = 65536ul);  // GPU_PAGE_SIZE = 65536ul
bool is_ptr_aligned(const void* ptr, size_t alignment = 65536ul);

// CUDA device pointer utility functions
bool is_device_pointer(const void* ptr, const char* error_msg = nullptr);
cudaError_t cudaMallocAligned(void** alignedPtr, void** rawPtr, size_t size, size_t alignment=4096);

// User confirmation for destructive operations
bool confirm_dangerous_operation(const std::string& operation_description);

ParsedSystemConfig parse_system_config(const std::string& config_file_path);
std::vector<nvme_ctrl_param> convert_to_nvme_ctrl_params(const ParsedSystemConfig& config);
std::vector<nvme_ctrl_param> convert_to_nvme_ctrl_params_group(const SystemConfigGroup & group_config);
#define geminifs_info(fmt, ...) \
    printf("[INFO][%s:%d] %s: " fmt "", __FILE__, __LINE__, __func__, ##__VA_ARGS__);

#define geminifs_error(fmt, ...) \
    printf("[ERROR][%s:%d] %s: " fmt "", __FILE__, __LINE__, __func__, ##__VA_ARGS__);

#define geminifs_warn(fmt, ...) \
    printf("[WARN][%s:%d] %s: " fmt "", __FILE__, __LINE__, __func__, ##__VA_ARGS__);


#ifdef DEBUG
#define geminifs_debug(fmt, ...) \
    printf("[DEBUG][%s:%d] %s: " fmt "", __FILE__, __LINE__, __func__, ##__VA_ARGS__);
#else
#define geminifs_debug(fmt, ...) \

#endif

// File system utility functions
std::string build_file_path(const std::string& directory, const std::string& filename);
bool create_directories(const std::string& path);

#endif