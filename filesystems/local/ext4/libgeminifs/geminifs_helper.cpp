#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <iostream>
#include <cuda_runtime.h>
#include "json.h"
#include "geminifs_helper.h"

#include <fstream>  
#include <map>
#include <set>
#include <sstream>
#include <algorithm>
#include "ioctl.h"
#include "gemini_fiemap.h"

// Include for PyTorch TORCH_CHECK macro
#ifdef TORCH_CHECK
// PyTorch is available
#include <torch/torch.h>
#else
// Fallback definition for TORCH_CHECK when PyTorch is not available
#define TORCH_CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Error: " << message << std::endl; \
            std::abort(); \
        } \
    } while(0)
#endif

using json = nlohmann::json;

// 计算两个PCI设备的距离
static inline int ioctl_get_pci_distance(const char *snvme_control_path, struct pci_device_addr_pair * pci_addr_pair) {
    int snvme_c_fd;
    int err;
    snvme_c_fd = open(snvme_control_path, O_RDWR | O_NONBLOCK);
    if (snvme_c_fd < 0){
        throw std::runtime_error("Failed to open control descriptor");
        return EFAULT;
    }
    err = ioctl(snvme_c_fd, SNVM_CACULATE_PCIDISTANCE, pci_addr_pair);
    close(snvme_c_fd);
    return err;
}

static std::string pci_bdf_to_string(const PCI_BDF& bdf) {
    char buffer[14];
    snprintf(buffer, sizeof(buffer), "%04x:%02x:%02x.%x", 
             bdf.domain, bdf.bus, bdf.device, bdf.function);
    return std::string(buffer, sizeof(buffer));
}

// 计算两个PCI设备的距离
int calculate_pci_distance(const PCI_BDF& bdf1, const PCI_BDF& bdf2) {

    struct pci_device_addr_pair pci_addr_pair;
    int distance = -1;
    pci_addr_pair.pairs[0] = {
        .domain = bdf1.domain,
        .bus = bdf1.bus,
        .slot = bdf1.device,
        .func = bdf1.function
    };

    pci_addr_pair.pairs[1] = {
        .domain = bdf2.domain,
        .bus = bdf2.bus,
        .slot = bdf2.device,
        .func = bdf2.function
    };

    distance = ioctl_get_pci_distance("/dev/snvm_control", &pci_addr_pair);

    return distance;
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

static inline SystemConfig parse_json(const std::string& filepath) {
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + filepath);
    }
    json j;
    file >> j;
    SystemConfig cfg;
    // 基础参数
    cfg.root_path = j["root_path"].get<std::string>();
    cfg.cluster_gpus = j["cluster_gpus"].get<unsigned>();
    cfg.cluster_disks = j["cluster_disks"].get<unsigned>();
    // GPU编号解析
    const std::string gpu_str = j["gpus_num"].get<std::string>();
    for (const auto& id_str : split(gpu_str, ',')) {
        if (id_str.find("cuda") == 0) {
            cfg.gpu_ids.push_back(static_cast<unsigned>(
                std::stoul(id_str.substr(4))));
        } else {
            throw std::runtime_error("Invalid GPU format: " + id_str);
        }
    }
    // GPU PCI地址解析
    for (auto id : cfg.gpu_ids) {
        char pciBusId[256];
        if (cudaDeviceGetPCIBusId(pciBusId, 256, id) != cudaSuccess) {
            std::cerr << "Failed to get PCI Bus ID" << std::endl;
            return cfg;
        }
        // 解析PCI地址
        std::string pci_str(pciBusId);
        size_t colon1 = pci_str.find(':');
        size_t colon2 = pci_str.find(':', colon1 + 1);
        // size_t dot = pci_str.find('.');
        
        if (colon2 != std::string::npos) {
            cfg.gpu_pci_addresses.emplace_back(pci_str);
        } else {
            throw std::runtime_error("Invalid PCI format: " + pci_str);
        }
    }
    // PCI地址解析
    const std::string pci_str = j["disks_pci_addr"].get<std::string>();
    for (const auto& pci : split(pci_str, ',')) {
        cfg.nvme_pci_addresses.emplace_back(pci);
    }
    return cfg;
}

system_overview parseSystemOverview(const std::string& filepath)
{
    system_overview sys;
    SystemConfig cfg = parse_json(filepath);
    sys.cfg = cfg;
    // 打印验证
    std::cout << "根路径: " << cfg.root_path << "\n"
              << "集群GPU数量: " << cfg.cluster_gpus << "\n"
              << "集群磁盘数量: " << cfg.cluster_disks << "\n"
              << "GPU编号: ";
    for (auto id : cfg.gpu_ids) std::cout << id << " ";
    
    std::cout << "\n NVMe PCI地址:\n";
    for (const auto& pci : cfg.nvme_pci_addresses) {
        printf("%04x:%02x:%02x.%x\n", 
               pci.domain, pci.bus, pci.device, pci.function);
    }
    std::cout << "\n GPU PCI地址:\n";
    for (const auto& pci : cfg.gpu_pci_addresses) {
        printf("%04x:%02x:%02x.%x\n", 
               pci.domain, pci.bus, pci.device, pci.function);
    }
    
    // 用于跟踪已分配的 NVMe 设备
    std::vector<bool> nvme_assigned(cfg.nvme_pci_addresses.size(), false);

    for (size_t i = 0; i < cfg.cluster_gpus; i++)
    {
        geminifs_ctrl_params params;
        params.cudaDevice = cfg.gpu_ids[i];
        params.mount_path = cfg.root_path;
        params.snvme_control_path = "/dev/snvm_control";
        params.ns_id = 1;
        params.queueDepth = 1024;
        params.numQueues = 64;
        
        const PCI_BDF& gpu_bdf = cfg.gpu_pci_addresses[i];
        // 查找与当前 GPU 在同一 PCIe switch 下的 NVMe 设备
        for (size_t j = 0; j < cfg.nvme_pci_addresses.size(); j++) {
            if (!nvme_assigned[j]) {
                const PCI_BDF& nvme_bdf = cfg.nvme_pci_addresses[j];

                // 计算 PCI 距离
                int distance = calculate_pci_distance(gpu_bdf, nvme_bdf);
                
                if (distance == 4) {
                    params.pci_addr.push_back(pci_bdf_to_string(nvme_bdf));
                    nvme_assigned[j] = true;
                }
            }
        } 
        sys.overview.push_back(params);
    }
    // 将剩余未分配的 NVMe 设备加入到 remote_disks
    for (size_t j = 0; j < cfg.nvme_pci_addresses.size(); j++) {
        if (!nvme_assigned[j]) {
            sys.remote_disks.push_back(cfg.nvme_pci_addresses[j]);
        }
    }
    return sys;
}

void printSystemOverview(const system_overview& sys) {
    std::cout << "系统概览:" << std::endl;

    // 打印 overview 中的 geminifs_ctrl_params
    for (const auto& ctrl : sys.overview) {
        std::cout << "GPU ID: " << ctrl.cudaDevice << std::endl;
        std::cout << "PCI 地址: ";
        for (const auto& addr : ctrl.pci_addr) {
            std::cout << addr << " ";
        }
        std::cout << std::endl;
    }

    // 打印 remote_disks 中的 PCI BDF
    std::cout << "未分配的 NVMe PCI 地址 (remote_disks):" << std::endl;
    std::cout << "remote ssd size is " << sys.remote_disks.size() << std::endl;
    for (const auto& bdf : sys.remote_disks) {
        std::cout << pci_bdf_to_string(bdf) << std::endl;
    }
}

// 动态调整文件描述符限制
bool increase_fd_limit(rlim_t desired_limit) {
    struct rlimit rlim;
    
    // 获取当前限制
    if (getrlimit(RLIMIT_NOFILE, &rlim) != 0) {
        std::cerr << "Failed to get current rlimit: " << strerror(errno) << std::endl;
        return false;
    }
    
    std::cout << "Current limits: soft=" << rlim.rlim_cur 
              << ", hard=" << rlim.rlim_max << std::endl;
    
    // 如果当前软限制已经满足需求，直接返回成功
    if (rlim.rlim_cur >= desired_limit) {
        std::cout << "Current soft limit (" << rlim.rlim_cur 
                  << ") already meets or exceeds desired limit (" << desired_limit << ")" << std::endl;
        return true;
    }
    
    // 首先尝试将软限制提高到硬限制
    if (rlim.rlim_cur < rlim.rlim_max) {
        rlim_t new_soft = std::min(desired_limit, rlim.rlim_max);
        rlim.rlim_cur = new_soft;
        
        std::cout << "Attempting to increase soft limit to " << new_soft << "..." << std::endl;
        
        if (setrlimit(RLIMIT_NOFILE, &rlim) == 0) {
            std::cout << "Successfully increased soft limit to " << new_soft << std::endl;
            if (new_soft >= desired_limit) {
                return true;
            }
        } else {
            std::cerr << "Failed to increase soft limit: " << strerror(errno) << std::endl;
        }
    }
    
    // 如果仍然不够，尝试同时提高硬限制（需要特权）
    if (rlim.rlim_max < desired_limit) {
        std::cout << "Attempting to increase hard limit to " << desired_limit << "..." << std::endl;
        
        rlim.rlim_cur = desired_limit;
        rlim.rlim_max = desired_limit;
        
        if (setrlimit(RLIMIT_NOFILE, &rlim) == 0) {
            std::cout << "Successfully increased both soft and hard limits to " << desired_limit << std::endl;
            return true;
        } else {
            std::cerr << "Failed to increase hard limit (may need root privileges): " << strerror(errno) << std::endl;
            
            // 回退到只设置软限制到硬限制
            if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
                rlim.rlim_cur = rlim.rlim_max;
                if (setrlimit(RLIMIT_NOFILE, &rlim) == 0) {
                    std::cout << "Fallback: Set soft limit to hard limit (" << rlim.rlim_max << ")" << std::endl;
                    return rlim.rlim_max >= desired_limit;
                }
            }
        }
    }
    
    return false;
}

// 显示当前文件描述符限制
void show_fd_limits() {
    struct rlimit rlim;
    if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
        std::cout << "File descriptor limits: soft=" << rlim.rlim_cur 
                  << ", hard=" << rlim.rlim_max << std::endl;
    }
}

// 自动配置文件描述符限制
void auto_configure_fd_limits(int num_files_to_open) {
    // std::cout << "\n=== Configuring File Descriptor Limits ===" << std::endl;
    
    // 显示当前限制
    // show_fd_limits();
    
    // 根据要打开的文件数量计算需要的限制
    // num_files_to_open + 一些余量用于系统文件描述符 (stdin, stdout, stderr, 日志文件等)
    rlim_t required_limit = static_cast<rlim_t>(num_files_to_open) + 100;
    
    // std::cout << "Planning to open " << num_files_to_open << " files" << std::endl;
    // std::cout << "Required limit for operation: " << required_limit << " file descriptors" << std::endl;
    
    // 尝试设置更高的限制 (推荐值)
    rlim_t recommended_limit = std::max(required_limit, (rlim_t)65536);
    
    // std::cout << "Attempting to set recommended limit: " << recommended_limit << std::endl;
    
    if (increase_fd_limit(recommended_limit)) {
        // std::cout << "✓ File descriptor limit configured successfully!" << std::endl;
    } else {
        // std::cout << "⚠ Warning: Could not achieve recommended limit." << std::endl;
        // std::cout << "   Operation may fail if trying to open too many files simultaneously." << std::endl;
        // std::cout << "   Consider running with elevated privileges or adjusting system limits." << std::endl;
        // std::cout << "   Quick fix: sudo ulimit -n 65536 && your_program" << std::endl;
    }
    
    // 显示最终限制
    std::cout << "Final limits:" << std::endl;
    show_fd_limits();
    std::cout << std::endl;
}

// Memory alignment utility functions
bool is_aligned(uint64_t value, size_t alignment) {
    return (value & (alignment - 1)) == 0;
}

bool is_ptr_aligned(const void* ptr, size_t alignment) {
    return is_aligned(reinterpret_cast<uint64_t>(ptr), alignment);
}


// Utility function to trim whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Parse system configuration file
ParsedSystemConfig parse_system_config(const std::string& config_file_path) {
    ParsedSystemConfig result;
    result.valid = false;
    
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        result.error_message = "Cannot open config file: " + config_file_path;
        return result;
    }
    
    std::vector<GPUConfig> gpus;
    std::vector<NVMeConfig> nvmes;
    std::string line;
    std::string current_section;
    
    GPUConfig current_gpu;
    NVMeConfig current_nvme;
    bool in_gpu_section = false;
    bool in_nvme_section = false;
    
    while (std::getline(file, line)) {
        line = trim(line);
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line.substr(0, 2) == "//") {
            continue;
        }
        
        // Check for section headers
        if (line[0] == '<' && line.back() == '>') {
            // Save previous section data
            if (in_gpu_section) {
                gpus.push_back(current_gpu);
                current_gpu = GPUConfig();
            }
            if (in_nvme_section) {
                nvmes.push_back(current_nvme);
                current_nvme = NVMeConfig();
            }
            
            current_section = line.substr(1, line.length() - 2);
            in_gpu_section = current_section.find("GPU") == 0;
            in_nvme_section = current_section.find("NVMe") == 0;
            continue;
        }
        
        // Parse key-value pairs
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));
        
        // Remove quotes if present
        if (value.length() >= 2 && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.length() - 2);
        }
        
        // Parse GPU section
        if (in_gpu_section) {
            if (key == "mount_path") {
                current_gpu.mount_path = value;
            } else if (key == "cudaDevice") {
                current_gpu.cudaDevice = std::stoi(value);
            }
        }
        // Parse NVMe section
        else if (in_nvme_section) {
            if (key == "mount_path") {
                current_nvme.mount_path = value;
            } else if (key == "pci_addr") {
                current_nvme.pci_addr = value;
            } else if (key == "ns_id") {
                current_nvme.ns_id = std::stoul(value);
            } else if (key == "queueDepth") {
                current_nvme.queueDepth = std::stoull(value);
            } else if (key == "numQueues") {
                current_nvme.numQueues = std::stoull(value);
            } else if (key == "cudaDevice") {
                current_nvme.cudaDevice = std::stoi(value);
            } else if (key == "maxIOsize") {
                current_nvme.maxIOsize = std::stoull(value);
            }
        }
    }
    
    // Save last section data
    if (in_gpu_section) {
        gpus.push_back(current_gpu);
    }
    if (in_nvme_section) {
        nvmes.push_back(current_nvme);
    }
    
    file.close();
    
    // Validation: Check if we have at least one GPU and one NVMe
    if (gpus.empty()) {
        result.error_message = "No GPU configuration found in config file";
        return result;
    }
    
    if (nvmes.empty()) {
        result.error_message = "No NVMe configuration found in config file";
        return result;
    }
    
    // Group NVMes by cudaDevice and match with GPUs
    std::map<int, std::vector<NVMeConfig>> nvme_groups;
    for (const auto& nvme : nvmes) {
        nvme_groups[nvme.cudaDevice].push_back(nvme);
    }
    
    // Create system groups
    for (const auto& gpu : gpus) {
        SystemConfigGroup group;
        group.gpu = gpu;
        
        // Find NVMes belonging to this GPU
        auto it = nvme_groups.find(gpu.cudaDevice);
        if (it != nvme_groups.end()) {
            group.nvmes = it->second;
        }
        
        // Validate that NVMes in this group have unique pci_addr and mount_path
        std::set<std::string> pci_addrs;
        std::set<std::string> mount_paths;
        
        for (const auto& nvme : group.nvmes) {
            if (pci_addrs.find(nvme.pci_addr) != pci_addrs.end()) {
                result.error_message = "Duplicate pci_addr found in GPU group " + 
                                       std::to_string(gpu.cudaDevice) + ": " + nvme.pci_addr;
                return result;
            }
            pci_addrs.insert(nvme.pci_addr);
            
            if (mount_paths.find(nvme.mount_path) != mount_paths.end()) {
                result.error_message = "Duplicate mount_path found in GPU group " + 
                                       std::to_string(gpu.cudaDevice) + ": " + nvme.mount_path;
                return result;
            }
            mount_paths.insert(nvme.mount_path);
        }
        
        if (!group.nvmes.empty()) {
            result.groups.push_back(group);
        }
    }
    
    // Final validation: ensure we have at least one valid group
    if (result.groups.empty()) {
        result.error_message = "No valid GPU-NVMe groups found. Each GPU must have at least one associated NVMe device with matching cudaDevice";
        return result;
    }
    
    result.valid = true;
    return result;
}

// Convert parsed config to nvme_ctrl_param structures
std::vector<nvme_ctrl_param> convert_to_nvme_ctrl_params(const ParsedSystemConfig& config) {
    std::vector<nvme_ctrl_param> params;
    
    if (!config.valid) {
        return params;
    }
    
    for (const auto& group : config.groups) {
        for (const auto& nvme : group.nvmes) {
            nvme_ctrl_param param;
            param.mount_path = nvme.mount_path;
            param.pci_addr = nvme.pci_addr;
            param.cudaDevice = nvme.cudaDevice;
            param.ns_id = nvme.ns_id;
            param.queueDepth = nvme.queueDepth;
            param.numQueues = nvme.numQueues;
            param.maxIOsize = nvme.maxIOsize;
            params.push_back(param);
        }
    }
    
    return params;
}

// Convert parsed config to nvme_ctrl_param structures
std::vector<nvme_ctrl_param> convert_to_nvme_ctrl_params_group(const SystemConfigGroup & group_config) {
    std::vector<nvme_ctrl_param> params;
   
    for (const auto& nvme : group_config.nvmes) {
        nvme_ctrl_param param;
        param.mount_path = nvme.mount_path;
        param.pci_addr = nvme.pci_addr;
        param.cudaDevice = nvme.cudaDevice;
        param.ns_id = nvme.ns_id;
        param.queueDepth = nvme.queueDepth;
        param.numQueues = nvme.numQueues;
        param.maxIOsize = nvme.maxIOsize;
        params.push_back(param);
    }
    return params;
}

/**
 * Check if a pointer is a CUDA device pointer
 * @param ptr The pointer to check
 * @param error_msg Optional error message. If provided and pointer is not a device pointer, TORCH_CHECK will be called
 * @return Returns true if the pointer is a device pointer, false otherwise
 */
bool is_device_pointer(const void* ptr, const char* error_msg) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    
    bool is_device = (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice);
    
    if (error_msg != nullptr && !is_device) {
        TORCH_CHECK(false, error_msg);
    }
    
    return is_device;
}


cudaError_t cudaMallocAligned(void** alignedPtr, void** rawPtr, size_t size, size_t alignment) {
    // 1. 分配一块比需求稍大的内存
    // 多分配 (alignment - 1) 的空间
    cudaError_t err = cudaMalloc(rawPtr, size + alignment - 1);
    if (err != cudaSuccess) {
        *alignedPtr = nullptr;
        *rawPtr = nullptr;
        return err;
    }

    // 2. 在分配的内存块中，计算出对齐后的地址
    // 将指针转换为整数类型，方便进行位运算
    uintptr_t ptr_val = reinterpret_cast<uintptr_t>(*rawPtr);
    
    // 使用位运算向上舍入到最近的对齐边界
    // 这是一个标准算法: (ptr + alignment - 1) & ~(alignment - 1)
    uintptr_t aligned_ptr_val = (ptr_val + alignment - 1) & ~(alignment - 1);
    
    // 将整数地址转回指针
    *alignedPtr = reinterpret_cast<void*>(aligned_ptr_val);

    return cudaSuccess;
}

// 用户确认危险操作
bool confirm_dangerous_operation(const std::string& operation_description) {
    std::cout << "\n⚠️  WARNING: Dangerous Operation ⚠️" << std::endl;
    std::cout << "You are about to perform a potentially destructive operation:" << std::endl;
    std::cout << operation_description << std::endl;
    std::cout << "\nThis action may cause data loss or system changes that cannot be undone." << std::endl;
    std::cout << "Are you sure you want to continue? (y/N): ";
    std::cout.flush();
    
    std::string input;
    std::getline(std::cin, input);
    
    // 转换为小写并去除空格
    std::transform(input.begin(), input.end(), input.begin(), ::tolower);
    input = trim(input);
    
    if (input == "y" || input == "yes") {
        std::cout << "✓ Operation confirmed by user" << std::endl;
        return true;
    } else {
        std::cout << "✗ Operation cancelled by user" << std::endl;
        return false;
    }
}

// File system utility functions (avoiding std::filesystem for compatibility)
std::string build_file_path(const std::string& directory, const std::string& filename) {
    std::string path = directory;
    if (!path.empty() && path.back() != '/') {
        path += "/";
    }
    path += filename;
    return path;
}

bool create_directories(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    
    // Check if directory already exists
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    
    // Find parent directory
    size_t last_slash = path.find_last_of('/');
    if (last_slash != std::string::npos && last_slash > 0) {
        std::string parent = path.substr(0, last_slash);
        if (!create_directories(parent)) {
            return false;
        }
    }
    
    // Create this directory
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
    }
    
    return true;
}
