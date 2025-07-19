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

        auto key_cache = torch::rand({4, 1024, 1024, 2}, // 512kb
                torch::TensorOptions()
                    .dtype(torch::kFloat16)
                    .device(torch::kCUDA, group.gpu.cudaDevice)
                    .pinned_memory(false)
                );
        
        auto key_cache2 = torch::rand({4, 1024, 1024, 2}, // 512kb
                torch::TensorOptions()
                    .dtype(torch::kFloat16)
                    .device(torch::kCUDA, group.gpu.cudaDevice)
                    .pinned_memory(false)
                );
        // Register tensor with GPU controller
        bool success = gpu_controller->registerTensorMemory(key_cache, 1024*1024* 2); // 2MB granularity
        success = gpu_controller->registerTensorMemory(key_cache2, 1024*1024* 2); // 2MB granularity
        if(!success) {
            std::cerr << "Failed to register tensor with GPU controller" << std::endl;
        }
        else
        {
            std::cout << "Successfully registered tensor with GPU controller" << std::endl;
            struct geminifs_dma* dma_context = gpu_controller->getDMAContext(key_cache.data_ptr());
            assert(dma_context != nullptr && dma_context->dma_ptr != nullptr);
            
            struct geminifs_dma* dma_context2 = gpu_controller->getDMAContext(key_cache2.data_ptr());
            assert(dma_context2 != nullptr && dma_context2->dma_ptr != nullptr);
            
            // Record start time for bandwidth calculation
            auto bandwidth_start = std::chrono::high_resolution_clock::now();
            nvme_controller_g_write_kernel<<<1,1,0,nvme_stream>>>(device_fd,dma_context->prp_mappings.at(0).prp1,
                                          dma_context->prp_mappings.at(0).prp2, 0, dma_context->slice_sizes.at(0));


            nvme_controller_g_read_kernel<<<1,1,0,nvme_stream>>>(device_fd,dma_context2->prp_mappings.at(0).prp1,
                                          dma_context2->prp_mappings.at(0).prp2, 0, dma_context2->slice_sizes.at(0));
            
            // Synchronize stream to ensure kernel completion
            cudaStreamSynchronize(nvme_stream);
            
            // Record end time and calculate bandwidth
            auto bandwidth_end = std::chrono::high_resolution_clock::now();
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(bandwidth_end - bandwidth_start);
            double duration_sec = duration_us.count() / 1000000.0;
            double data_mb = dma_context->slice_sizes.at(0) / (1024.0 * 1024.0);
            double bandwidth_mbps = data_mb / duration_sec;
            
            geminifs_info("NVMe Read Bandwidth Statistics:\n");
            geminifs_info("  Data size: %.2f MB\n", data_mb);
            geminifs_info("  Duration: %.3f ms\n", duration_us.count() / 1000.0);
            geminifs_info("  Bandwidth: %.2f MB/s\n", bandwidth_mbps);

            // 检查两个tensor前1MB数据是否一致
            std::cout << "\nVerifying data consistency between key_cache and key_cache2..." << std::endl;
            
            // 计算前1MB的元素数量 (half precision = 2 bytes per element)
            size_t verify_bytes = 1024 * 1024; // 1MB
            size_t verify_elements = verify_bytes / 2; // 2 bytes per half precision element
            
            // 确保不超过tensor的实际大小
            size_t tensor_elements = key_cache.numel();
            verify_elements = std::min(verify_elements, tensor_elements);
            verify_bytes = verify_elements * 2;
            
            std::cout << "Comparing first " << verify_bytes << " bytes (" << verify_elements << " elements)..." << std::endl;
            
            // 将数据从GPU拷贝到CPU进行比较
            torch::Tensor key_cache_cpu = key_cache.flatten().slice(0, 0, verify_elements).to(torch::kCPU);
            torch::Tensor key_cache2_cpu = key_cache2.flatten().slice(0, 0, verify_elements).to(torch::kCPU);
            
            // 比较数据
            torch::Tensor diff = torch::abs(key_cache_cpu - key_cache2_cpu);
            torch::Tensor max_diff = torch::max(diff);
            torch::Tensor mean_diff = torch::mean(diff);
            
            // 检查是否完全一致
            bool data_identical = torch::allclose(key_cache_cpu, key_cache2_cpu, 1e-6, 1e-6);
            
            if (data_identical) {
                std::cout << "✓ Data verification PASSED: key_cache and key_cache2 are identical (first 1MB)" << std::endl;
            } else {
                std::cout << "✗ Data verification FAILED: key_cache and key_cache2 differ" << std::endl;
                std::cout << "  Max difference: " << max_diff.item<float>() << std::endl;
                std::cout << "  Mean difference: " << mean_diff.item<float>() << std::endl;
                
                // 统计不同元素的数量
                torch::Tensor non_zero_diff = (diff > 1e-6);
                int64_t diff_count = torch::sum(non_zero_diff).item<int64_t>();
                double diff_percentage = (double)diff_count / verify_elements * 100.0;
                
                std::cout << "  Different elements: " << diff_count << " / " << verify_elements 
                          << " (" << std::fixed << std::setprecision(2) << diff_percentage << "%)" << std::endl;
                
                // 显示前几个不同的值进行调试
                if (diff_count > 0) {
                    std::cout << "  First few differences:" << std::endl;
                    auto key_cache_data = key_cache_cpu.accessor<torch::Half, 1>();
                    auto key_cache2_data = key_cache2_cpu.accessor<torch::Half, 1>();
                    
                    int shown_diffs = 0;
                    for (int64_t i = 0; i < verify_elements && shown_diffs < 5; i++) {
                        if (std::abs(static_cast<float>(key_cache_data[i]) - static_cast<float>(key_cache2_data[i])) > 1e-6) {
                            std::cout << "    Index " << i << ": " 
                                      << static_cast<float>(key_cache_data[i]) << " vs " 
                                      << static_cast<float>(key_cache2_data[i]) << std::endl;
                            shown_diffs++;
                        }
                    }
                }
            }
            
            // 显示数据统计信息
            torch::Tensor key_cache_stats = key_cache_cpu.slice(0, 0, std::min((int64_t)10, (int64_t)verify_elements));
            torch::Tensor key_cache2_stats = key_cache2_cpu.slice(0, 0, std::min((int64_t)10, (int64_t)verify_elements));
            
            std::cout << "\nData samples (first 10 elements):" << std::endl;
            std::cout << "key_cache:  ";
            for (int i = 0; i < key_cache_stats.size(0); i++) {
                std::cout << std::fixed << std::setprecision(4) << key_cache_stats[i].item<float>() << " ";
            }
            std::cout << std::endl;
            
            std::cout << "key_cache2: ";
            for (int i = 0; i < key_cache2_stats.size(0); i++) {
                std::cout << std::fixed << std::setprecision(4) << key_cache2_stats[i].item<float>() << " ";
            }
            std::cout << std::endl;

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