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
const size_t FILE_SIZE = 16 * 1024 * 1024 ; // 16MB

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

    cudaStream_t nvme_stream;
    cudaError_t stream_err = cudaStreamCreate(&nvme_stream);
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
    


        void* device_fd = nvme_controller->g_open("1", FILE_SIZE, O_DEVICE);
        if(!is_device_pointer(device_fd,"device fd must be a device pointer")||device_fd==NULL)
        {
            geminifs_error("g_open: fail device_fd is illeagel\n");
        }



        // // 记录开始时间
        // auto start_time = std::chrono::high_resolution_clock::now();

        /*first create*/
    //     std::vector<torch::Tensor> key_caches;
    //             // Calculate tensor dimensions for FILE_SIZE bytes with half precision (2 bytes per element)
    //     size_t num_elements = FILE_SIZE / 2; // 2 bytes per half precision element
    //     for (int i = 0; i < 100; i++){
    //         key_caches.push_back(torch::rand(
    //             {16, 1024, 1024, 2}, // 512kb
    //             torch::TensorOptions()
    //                 .dtype(torch::kFloat16)
    //                 .device(torch::kCUDA, group.gpu.cudaDevice)
    //                 .pinned_memory(false)
    //             ));
    //     }
    //    for (int i = 0; i < 100; ++i) {
    //         // Register tensor with GPU controller
    //         bool success = gpu_controller->registerTensorMemory(key_caches.at(i));
            
    //         if (!success) {
    //             std::cerr << "Failed to register tensor " << i << " with GPU controller" << std::endl;
    //         }
    //     }

        auto key_cache = torch::rand({16, 1024, 1024, 2}, // 512kb
                torch::TensorOptions()
                    .dtype(torch::kFloat16)
                    .device(torch::kCUDA, group.gpu.cudaDevice)
                    .pinned_memory(false)
                );
        // Register tensor with GPU controller
        bool success = gpu_controller->registerTensorMemory(key_cache, 1024*1024* 2); // 2MB granularity
        if(!success) {
            std::cerr << "Failed to register tensor with GPU controller" << std::endl;
        }
        else
        {
            std::cout << "Successfully registered tensor with GPU controller" << std::endl;
            struct geminifs_dma* dma_context = gpu_controller->getDMAContext(key_cache.data_ptr());
            assert(dma_context != nullptr && dma_context->dma_ptr != nullptr);
            nvme_controller_g_read_kernel<<<1,1,0,nvme_stream>>>(device_fd,dma_context->prp_mappings.at(0).prp1,
                                          dma_context->prp_mappings.at(0).prp2, 0, dma_context->slice_sizes.at(0));

            if(success)
            {
                std::cout << "✓ Successfully registered tensor with GPU controller" << std::endl;
            } else{
                std::cerr << "✗ Failed to register tensor with GPU controller" << std::endl;
            }
        }



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