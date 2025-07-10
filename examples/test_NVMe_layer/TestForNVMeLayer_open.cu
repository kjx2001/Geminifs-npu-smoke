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
#include "geminifs_helper.h"
#include "nvm_error.h"
#include "utils.cuh"
#include "backtrace.h"

// 定义常量
const int NUM_FILES = 100000;
const size_t FILE_SIZE = 16 * 1024 * 1024; // 16MB

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
    ParsedSystemConfig config = parse_system_config("/home/hzx/Geminifs/sys_config.ini");
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
        
        // 打开1万个文件并统计时间
        std::cout << "\n=== Starting file opening test ===" << std::endl;
        std::cout << "Number of files: " << NUM_FILES << std::endl;
        std::cout << "File size: " << FILE_SIZE << " bytes (" << FILE_SIZE / (1024*1024) << " MB)" << std::endl;
        
        // 显示当前文件描述符限制
        show_fd_limits();
        std::cout << "Initial open FDs: " << get_open_fd_count() << std::endl;
        
        std::vector<void*> file_handles;
        file_handles.reserve(NUM_FILES);
        
        // 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 打开文件
        int successful_opens = 0;
        int failed_opens = 0;
        
        for (int i = 1; i <= NUM_FILES; i++) {
            std::string filename = std::to_string(i);
            
            void* file_handle = nvme_controller->g_open(filename, FILE_SIZE, O_HOST);
            if (file_handle != nullptr) {
                file_handles.push_back(file_handle);
                successful_opens++;
            } else {
                failed_opens++;
                std::cout << "Failed to open file: " << filename << std::endl;
                
                // 如果连续失败太多，打印更详细的诊断信息
                if (failed_opens == 1) {
                    std::cout << "First failure detected. Current open FDs: " << get_open_fd_count() << std::endl;
                    show_fd_limits();
                }
            }
            
            // 每100个文件输出一次进度和FD使用情况
            if (i % 100 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                int current_fds = get_open_fd_count();
                std::cout << "Opened " << i << "/" << NUM_FILES << " files. "
                         << "Elapsed: " << elapsed.count() << " ms. "
                         << "Success: " << successful_opens << ", Failed: " << failed_opens 
                         << ", FDs: " << current_fds << std::endl;
                
                // 如果有失败，退出循环以避免大量重复错误
                if (failed_opens >= 10) {
                    std::cout << "Too many failures, stopping test early." << std::endl;
                    break;
                }
            }
        }
        
        // 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 输出统计结果
        std::cout << "\n=== File Opening Statistics ===" << std::endl;
        std::cout << "Total files attempted: " << NUM_FILES << std::endl;
        std::cout << "Successfully opened: " << successful_opens << std::endl;
        std::cout << "Failed to open: " << failed_opens << std::endl;
        std::cout << "Success rate: " << std::fixed << std::setprecision(2) 
                  << (double)successful_opens / NUM_FILES * 100 << "%" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
        
        if (successful_opens > 0) {
            double avg_time_per_file = (double)total_duration.count() / successful_opens;
            std::cout << "Average time per file: " << std::fixed << std::setprecision(3) 
                      << avg_time_per_file << " ms" << std::endl;
            std::cout << "Files per second: " << std::fixed << std::setprecision(1) 
                      << successful_opens / (total_duration.count() / 1000.0) << std::endl;
        }
        
        std::cout << "Total data opened: " << std::fixed << std::setprecision(2) 
                  << (double)(successful_opens * FILE_SIZE) / (1024 * 1024 * 1024) << " GB" << std::endl;
        
        // Test device file operations
        std::cout << "\n=== Testing Device File Operations ===" << std::endl;
        const std::string device_test_filename = "device_test_file";
        
        void* device_fd = nvme_controller->g_open(device_test_filename, FILE_SIZE, O_DEVICE);
        if (device_fd != nullptr) {
            std::cout << "✓ Successfully created device file: " << device_test_filename << std::endl;
            std::cout << "Device file descriptor: " << device_fd << std::endl;
            
            // The device file will be automatically cleaned up by the destructor
        } else {
            std::cerr << "✗ Failed to create device file: " << device_test_filename << std::endl;
        }
        
        // Test opening existing file as device file
        if (successful_opens > 0) {
            std::cout << "\n=== Testing Device File Opening ===" << std::endl;
            std::string existing_filename = "1"; // First file created in the batch
            
            void* existing_device_fd = nvme_controller->g_open(existing_filename, FILE_SIZE, O_DEVICE);
            if (existing_device_fd != nullptr) {
                std::cout << "✓ Successfully opened existing file as device file: " << existing_filename << std::endl;
                std::cout << "Device file descriptor: " << existing_device_fd << std::endl;
            } else {
                std::cerr << "✗ Failed to open existing file as device file: " << existing_filename << std::endl;
            }
        }
        
        // Wait for user to type "end" to terminate the program
        std::cout << "\nTest completed. Type 'end' to terminate and release resources:" << std::endl;
        std::string user_input;
        while (true) {
            std::cout << "> ";
            std::getline(std::cin, user_input);
            
            if (user_input == "end") {
                std::cout << "Terminating program and releasing resources..." << std::endl;
                break;
            } else if (user_input == "stats") {
                // 重新显示统计信息
                std::cout << "\n=== File Opening Statistics ===" << std::endl;
                std::cout << "Total files attempted: " << NUM_FILES << std::endl;
                std::cout << "Successfully opened: " << successful_opens << std::endl;
                std::cout << "Failed to open: " << failed_opens << std::endl;
                std::cout << "Success rate: " << std::fixed << std::setprecision(2) 
                          << (double)successful_opens / NUM_FILES * 100 << "%" << std::endl;
                std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
                if (successful_opens > 0) {
                    double avg_time_per_file = (double)total_duration.count() / successful_opens;
                    std::cout << "Average time per file: " << std::fixed << std::setprecision(3) 
                              << avg_time_per_file << " ms" << std::endl;
                    std::cout << "Files per second: " << std::fixed << std::setprecision(1) 
                              << successful_opens / (total_duration.count() / 1000.0) << std::endl;
                }
                std::cout << "Total data opened: " << std::fixed << std::setprecision(2) 
                          << (double)(successful_opens * FILE_SIZE) / (1024 * 1024 * 1024) << " GB" << std::endl;
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