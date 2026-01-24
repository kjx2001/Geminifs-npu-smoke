#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <ctime>
#include <string>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <dirent.h>
#include <sys/resource.h>
#include <errno.h>
#include <algorithm>


#include "buffer.h"
#include "geminifs.h"
#include "linux/ioctl.h"
#include "geminifs.cuh"
#include "gemini_fiemap.h"

#include "geminifs_helper.h"
#include "nvm_error.h"
#include "utils.cuh"
#include "backtrace.h"


// 定义常量
const int NUM_FILES = 100000;
const size_t FILE_SIZE = 8ULL * 1024 * 1024 * 1024; // 1GB

// 获取当前进程打开的文件描述符数量
int get_open_fd_count() {
    char proc_fd_path[256];
    snprintf(proc_fd_path, sizeof(proc_fd_path), "/proc/%d/fd", getpid());
    
    int fd_count = 0;
    DIR* fd_dir = opendir(proc_fd_path);
    if (fd_dir) {
        struct dirent* entry;
        while ((entry = readdir(fd_dir)) != NULL) {
            if (entry->d_name[0] >= '0' && entry->d_name[0] <= '9') {
                fd_count++;
            }
        }
        closedir(fd_dir);
    }
    return fd_count;
}

// 显示文件描述符限制信息 (本地版本)
void show_fd_limits() {
    struct rlimit rlim;
    if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
        std::cout << "File descriptor limits: soft=" << rlim.rlim_cur 
                  << ", hard=" << rlim.rlim_max << std::endl;
    }
}

int main(int argc, char** argv) {
    setup_backtrace();
    // 在程序开始时自动配置文件描述符限制
    auto_configure_fd_limits(NUM_FILES);
    ParsedSystemConfig config = parse_system_config("/home/zwh/Geminifs/sys_config.ini");
    std::vector<nvme_ctrl_param> nvme_params;
    if (config.valid) {
        // 转换为nvme_ctrl_param格式
        nvme_params = convert_to_nvme_ctrl_params(config);
        
        // 遍历所有配置组
        for (const auto& group : config.groups) {
            std::cout << "GPU " << group.gpu.cudaDevice << " mount: " << group.gpu.mount_path << std::endl;
            for (const auto& nvme : group.nvmes) {
                std::cout << "  NVMe: " << nvme.pci_addr << " -> " << nvme.mount_path << std::endl;
            }
        }
    } else {
        std::cerr << "Config parsing failed: " << config.error_message << std::endl;
    }

    try {
        SystemConfigGroup group = config.groups.at(0);

        GPUControllerPtr gpu_controller_ = geminifs_create_gpu_controller(group.gpu.cudaDevice, group.gpu.mount_path);

        // 检查是否有NVMe参数，并且第一个参数的pci_addr不为空
        if (!nvme_params.empty() && !nvme_params.at(0).pci_addr.empty()) {
            bool success = geminifs_add_nvme_to_gpu(group.gpu.cudaDevice, nvme_params.at(0));
            if (success) {
                std::cout << "Successfully added NVMe controller to GPU " << group.gpu.cudaDevice << std::endl;
            } else {
                std::cerr << "Failed to add NVMe controller to GPU " << group.gpu.cudaDevice << std::endl;
            }
        } else {
            std::cerr << "No valid NVMe parameters found" << std::endl;
        }
        
        auto gpu_controller = geminifs_get_gpu_controller(group.gpu.cudaDevice);
        if (!gpu_controller) {
            geminifs_error("geminifs_add_nvme_to_gpu: No GPU controller found for device %d\n", group.gpu.cudaDevice);
            return false;
        }
        
        NVMeControllerPtr nvme_controller =  gpu_controller->getNVMeController(0);
        
        std::cout << "NVMeController initialized successfully!" << std::endl;
        std::cout << "Mount path: " << nvme_controller->mount_path << std::endl;

        // Use the controller and file manager
        if (nvme_controller->controller) {
            std::cout << "Controller is ready for use" << std::endl;
        }

        if (nvme_controller->file_manager) {
            std::cout << "File manager is ready for file operations" << std::endl;
            
            // // Example: Create a file record
            // NVMeFileDesc file_desc;
            // if (nvme_controller->file_manager->createFile("test_file.dat", file_desc)) {
            //     std::cout << "Created file record with slot index: " << file_desc.slot_index << std::endl;
            // }
        }

       if (nvme_controller->is_initialized()) {
            std::cout << "✓ NVMeController is properly initialized" << std::endl;
        } else {
            std::cerr << "✗ NVMeController failed to initialize" << std::endl;
            return 1;
        }
        
        // Test device file operations
        std::cout << "\n=== Testing File Size ===" << std::endl;
        const std::string device_test_filename = "tfs";
        void* device_fd = nvme_controller->g_open(device_test_filename, FILE_SIZE, O_DEVICE);
        if (device_fd != nullptr) {
            std::cout << "✓ Successfully created device file: " << device_test_filename << std::endl;
            std::cout << "Device file descriptor: " << device_fd << std::endl;
            // The device file will be automatically cleaned up by the destructor
        } else {
            std::cerr << "✗ Failed to create device file: " << device_test_filename << std::endl;
        }
        
        // Wait for user to type "end" to terminate the program
        std::cout << "\nTest completed. Type 'end' to terminate and release resources:" << std::endl;
        std::string user_input;
        while (true) {
            std::cout << "> ";
            std::getline(std::cin, user_input);
            
            if (user_input == "end") {
                std::cout << "Terminating program and releasing resources..." << std::endl;
                nvme_controller->device_file_delete_all_files_managed();
                break;
            } else if (user_input == "stats") {
                // 重新显示统计信息
                std::cout << "file size is:" << FILE_SIZE << std::endl;
            } else {
                std::cout << "Commands: 'end' to terminate, 'stats' to show statistics again." << std::endl;
            }
        }
        nvme_controller.reset(); // Explicitly reset the controller to release resources
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    
    std::cout << "Program terminated successfully." << std::endl;
    return 0;
}