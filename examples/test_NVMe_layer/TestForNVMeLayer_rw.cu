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
#include <torch/all.h>
#include <ATen/ATen.h> 

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

    // 在程序开始时自动配置文件描述符限制
    auto_configure_fd_limits(NUM_FILES);
    ParsedSystemConfig config = parse_system_config("/home/qs/CompanionFS/Geminifs/sys_config.ini");
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
    
                std::vector<void*> file_handles;
        file_handles.reserve(NUM_FILES);
        
        void* file_handle = nvme_controller->g_open("1", FILE_SIZE, O_HOST);

        // // 记录开始时间
        // auto start_time = std::chrono::high_resolution_clock::now();
        
        // // 打开文件
        // int successful_opens = 0;
        // int failed_opens = 0;
        
        // for (int i = 1; i <= NUM_FILES; i++) {
        //     std::string filename = std::to_string(i);
            
        //     void* file_handle = nvme_controller->g_open(filename, FILE_SIZE, O_HOST);
        //     if (file_handle != nullptr) {
        //         file_handles.push_back(file_handle);
        //         successful_opens++;
        //     } else {
        //         failed_opens++;
        //         std::cout << "Failed to open file: " << filename << std::endl;
                
        //         // 如果连续失败太多，打印更详细的诊断信息
        //         if (failed_opens == 1) {
        //             std::cout << "First failure detected. Current open FDs: " << get_open_fd_count() << std::endl;
        //             show_fd_limits();
        //         }
        //     }
            
        //     // 每100个文件输出一次进度和FD使用情况
        //     if (i % 100 == 0) {
        //         auto current_time = std::chrono::high_resolution_clock::now();
        //         auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
        //         int current_fds = get_open_fd_count();
        //         std::cout << "Opened " << i << "/" << NUM_FILES << " files. "
        //                  << "Elapsed: " << elapsed.count() << " ms. "
        //                  << "Success: " << successful_opens << ", Failed: " << failed_opens 
        //                  << ", FDs: " << current_fds << std::endl;
                
        //         // 如果有失败，退出循环以避免大量重复错误
        //         if (failed_opens >= 10) {
        //             std::cout << "Too many failures, stopping test early." << std::endl;
        //             break;
        //         }
        //     }
        // }
        
        // // 记录结束时间
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // // 输出统计结果
        // std::cout << "\n=== File Opening Statistics ===" << std::endl;
        // std::cout << "Total files attempted: " << NUM_FILES << std::endl;
        // std::cout << "Successfully opened: " << successful_opens << std::endl;
        // std::cout << "Failed to open: " << failed_opens << std::endl;
        // std::cout << "Success rate: " << std::fixed << std::setprecision(2) 
        //           << (double)successful_opens / NUM_FILES * 100 << "%" << std::endl;
        // std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
        
        // if (successful_opens > 0) {
        //     double avg_time_per_file = (double)total_duration.count() / successful_opens;
        //     std::cout << "Average time per file: " << std::fixed << std::setprecision(3) 
        //               << avg_time_per_file << " ms" << std::endl;
        //     std::cout << "Files per second: " << std::fixed << std::setprecision(1) 
        //               << successful_opens / (total_duration.count() / 1000.0) << std::endl;
        // }
        
        // std::cout << "Total data opened: " << std::fixed << std::setprecision(2) 
        //           << (double)(successful_opens * FILE_SIZE) / (1024 * 1024 * 1024) << " GB" << std::endl;
        
        // // Test device file operations
        // std::cout << "\n=== Testing Device File Operations ===" << std::endl;
        // const std::string device_test_filename = "device_test_file";
        
        // void* device_fd = nvme_controller->g_open(device_test_filename, FILE_SIZE, O_DEVICE);
        // if (device_fd != nullptr) {
        //     std::cout << "✓ Successfully created device file: " << device_test_filename << std::endl;
        //     std::cout << "Device file descriptor: " << device_fd << std::endl;
            
        //     // The device file will be automatically cleaned up by the destructor
        // } else {
        //     std::cerr << "✗ Failed to create device file: " << device_test_filename << std::endl;
        // }

        // // Allocate 100 tensors of FILE_SIZE on GPU and register them
        // std::cout << "\nAllocating 100 tensors of size " << FILE_SIZE << " bytes on GPU..." << std::endl;
        // std::vector<torch::Tensor> tensors;
        // tensors.reserve(100);
        
        // // Calculate tensor dimensions for FILE_SIZE bytes with half precision (2 bytes per element)
        // size_t num_elements = FILE_SIZE / 2; // 2 bytes per half precision element
        // for (int i = 0; i < 100; i++){
        //     tensors.push_back(torch::rand(
        //         {16, 1024, 1024, 2}, // 512kb
        //         torch::TensorOptions()
        //             .dtype(torch::kFloat16)
        //             .device(torch::kCUDA, group.gpu.cudaDevice)
        //             .pinned_memory(false)
        //         ));
        // }


        // for (int i = 0; i < 100; ++i) {
        //     // Register tensor with GPU controller
        //     bool success = gpu_controller->registerTensorMemory(tensors.at(i));
            
        //     if (!success) {
        //         std::cerr << "Failed to register tensor " << i << " with GPU controller" << std::endl;
        //     }
        // }

        std::cout << "\nTest completed. Type 'end' to terminate and release resources:" << std::endl;
        std::cout << "Available commands: 'end', 'delete'" << std::endl;
        std::string user_input;
        while (true) {
            std::cout << "> ";
            std::getline(std::cin, user_input);
            
            if (user_input == "end") {
                std::cout << "Terminating program and releasing resources..." << std::endl;
                break;
            } else if (user_input == "delete") {
                // Handle delete command
                int device_id = 0;
                size_t controller_index = 0;
                
                std::cout << "Please enter the GPU device ID (default: " << group.gpu.cudaDevice << "): ";
                std::string input_line;
                std::getline(std::cin, input_line);
                
                if (!input_line.empty()) {
                    try {
                        device_id = std::stoi(input_line);
                    } catch (const std::exception& e) {
                        std::cerr << "Error: Invalid device ID. Using default: " << group.gpu.cudaDevice << std::endl;
                        device_id = group.gpu.cudaDevice;
                    }
                } else {
                    device_id = group.gpu.cudaDevice;
                }
                
                std::cout << "Please enter the controller index (default: 0): ";
                std::getline(std::cin, input_line);
                
                if (!input_line.empty()) {
                    try {
                        controller_index = std::stoull(input_line);
                    } catch (const std::exception& e) {
                        std::cerr << "Error: Invalid controller index. Using default: 0" << std::endl;
                        controller_index = 0;
                    }
                } else {
                    controller_index = 0;
                }
                
                std::cout << "Attempting to delete all files from GPU device " << device_id 
                            << ", controller index " << controller_index << "..." << std::endl;
                
                // Confirm the operation
                std::cout << "Are you sure you want to delete ALL files? This action cannot be undone. (yes/no): ";
                std::string confirmation;
                std::getline(std::cin, confirmation);
                
                if (confirmation == "yes" || confirmation == "y" || confirmation == "YES") {
                    bool delete_success = geminifs_nvme_delete_all_files(device_id, controller_index);
                    
                    if (delete_success) {
                        std::cout << "✓ Successfully deleted all files from GPU device " << device_id 
                                    << ", controller index " << controller_index << std::endl;
                    } else {
                        std::cerr << "✗ Failed to delete all files from GPU device " << device_id 
                                    << ", controller index " << controller_index << std::endl;
                    }
                } else {
                    std::cout << "Delete operation cancelled." << std::endl;
                }
            } else {
                std::cout << "Commands: 'end' to terminate, 'delete' to delete all files." << std::endl;
            }
        }
    }catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}