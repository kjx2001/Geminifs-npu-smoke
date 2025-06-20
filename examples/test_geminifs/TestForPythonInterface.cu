#include "geminifs.cuh"
#include "torch/types.h"
#include "utils.cuh"
#include <iostream>
#include <ostream>
#include <torch/all.h>
#include <ATen/ATen.h> 

#define MB 1024 * 1024ll
int main(int argc, char **argv) {
    
    int64_t nr_file = 500;
    int64_t file_size = 32 * MB; // 64 * 4 = 256kb
    int device_id = 0;
    std::string mount_path = "/mnt/tardis";
    std::string pci_addrs = "0000:50:00.0,0000:51:00.0";
    // std::string pci_addrs = "0000:c1:00.0,0000:cb:00.0,0000:cc:00.0,0000:ce:00.0";

    // std::string pci_addrs = "0000:50:00.0,0000:51:00.0";

    std::cout << "init geminifs" << std::endl;
    if (!geminifs_init_fds_wrapper_cuda_test(nr_file, file_size, device_id, mount_path, pci_addrs)){
        std::cout << "init error" << std::endl;
    } else {
        std::cout << "init success" << std::endl;
    }

    int64_t num_tokens = 16;
    int64_t num_layers = 32;
    int64_t num_kv_heads = 32;
    int64_t head_size = 128;
    int64_t block_size = 16;
    int64_t max_num_block = nr_file;

    int64_t nr_layers = 32;

    int64_t shape_size = num_tokens * 8 * 128;

    std::vector<torch::Tensor> key_caches;
    std::vector<torch::Tensor> value_caches;

    for (int i = 0; i < num_layers; i++){
        key_caches.push_back(torch::rand(
            {max_num_block, num_tokens, num_kv_heads, head_size}, // 512kb
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false)
        ));
    }

    for (int i = 0; i < num_layers; i++){
        value_caches.push_back(torch::rand(
            {max_num_block, num_tokens, num_kv_heads, head_size}, // 512kb
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false)
        ));
    }
    
    for (int i = 0; i < num_layers; i++){
        if (!geminifs_create_dma(key_caches[i])){
            std::cout << "create dma error" << std::endl;
            return -1;
        }
        if (!geminifs_create_dma(value_caches[i])){
            std::cout << "create dma error" << std::endl;
            return -1;
        }
    }

    std::vector<torch::Tensor> key_cache_reads;
    std::vector<torch::Tensor> value_cache_reads;
                

    for (int i = 0; i < num_layers; i++){
        key_cache_reads.push_back(torch::zeros(
            {max_num_block, num_tokens, num_kv_heads, head_size}, // 512kb
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false)
        ));
        value_cache_reads.push_back(torch::zeros(
            {max_num_block, num_tokens, num_kv_heads, head_size}, // 512kb
        torch::TensorOptions()
            .dtype(torch::kFloat16)           
            .device(torch::kCUDA, device_id)
            .pinned_memory(false)
        ));
    }

    for (int i = 0; i < num_layers; i++){
        if (!geminifs_create_dma(key_cache_reads[i])){
            std::cout << "create dma error" << std::endl;
            return -1;
        }
        if (!geminifs_create_dma(value_cache_reads[i])){
            std::cout << "create dma error" << std::endl;
            return -1;
        }
    }

    torch::Tensor cached_file_ids = torch::zeros(
        {nr_file}, 
        torch::TensorOptions()
            .dtype(torch::kUInt64)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false)
    );
    for (int i = 0; i < nr_file; i++){
        cached_file_ids[i] = i;
    }

    torch::Tensor inner_block_ids = torch::zeros(
        {nr_file}, 
        torch::TensorOptions()
            .dtype(torch::kUInt64)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false)
    );
    for (int i = 0; i < nr_file; i++){
        inner_block_ids[i] = i;
    }

    if (!geminifs_device_xfer_wrapper_test2(cached_file_ids, inner_block_ids, key_caches, value_caches, 0, 32, FILE_XFER_WRITE)){
        std::cout << "xfer error" << std::endl;
        return -1;
    }

    cuda_check_error(cudaDeviceSynchronize());


    auto start = std::chrono::high_resolution_clock::now();
    if (!geminifs_device_xfer_wrapper_test2(cached_file_ids, inner_block_ids, key_cache_reads, value_cache_reads, 0, 32,FILE_XFER_READ)){
        std::cout << "xfer error" << std::endl;
        return -1;
    }

    cuda_check_error(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    auto xfer_size = nr_layers * max_num_block * shape_size * 2.0 / (1024 * 1024 * 1024);
    std::cout << "total xfer time: " << duration.count() << " ms" << std::endl;
    std::cout << "total xfer bandwidth: " << xfer_size / (duration.count() / 1000) << " GB/s" << std::endl;

    // for (int i = 0; i < num_layers; i++){
    //     if (torch::allclose(key_caches[i], key_cache_reads[i]) && torch::allclose(value_caches[i], value_cache_reads[i])) {
    //         std::cout << "xfer success" << std::endl;
    //     } else {
    //         std::cout << "xfer error" << std::endl;
    //         return -1;
    //     }
    // }

    // torch::Tensor cached_file_ids = torch::zeros(
    //     {nr_file}, 
    //     torch::TensorOptions()
    //         .dtype(torch::kUInt64)
    //         .device(torch::kCUDA, device_id)
    //         .pinned_memory(false)
    // );
    // for (int i = 0; i < nr_file; i++){
    //     cached_file_ids[i] = i;
    // }

    // torch::Tensor inner_block_ids = torch::zeros(
    //     {nr_file}, 
    //     torch::TensorOptions()
    //         .dtype(torch::kUInt64)
    //         .device(torch::kCUDA, device_id)
    //         .pinned_memory(false)
    // );
    // for (int i = 0; i < nr_file; i++){
    //     inner_block_ids[i] = i;
    // }


    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < nr_layers; i++) {
    //     if (!geminifs_device_xfer_wrapper_test(cached_file_ids, inner_block_ids, key_cache_vec[i], value_cache_vec[i], i, FILE_XFER_READ)){
    //         std::cout << "xfer error" << std::endl;
    //         return -1;
    //     }
    //     cuda_check_error(cudaDeviceSynchronize());
    // }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // auto xfer_size = nr_layers * max_num_block * shape_size * 2.0 / (1024 * 1024 * 1024);
    // auto xfer_time = duration.count();
    // std::cout << "total xfer size: " << xfer_size << " GB" << std::endl;
    // std::cout << "total xfer time: " << xfer_time << " ms" << std::endl;
    // std::cout << "xfer bandwidth: " << xfer_size / (xfer_time / 1000) << " GB/s" << std::endl;

    return 0; 
}