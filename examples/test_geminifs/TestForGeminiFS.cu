#include "geminifs.cuh"
#include "gpu_controller.cuh"
#include "geminifs_helper.h"
#include "gpu_controller.cuh"
#include <iostream>
#include <vector>
using namespace std;

constexpr size_t GPU_file_size = 32ull * 1024 * 1024; // 8MB
constexpr int GPU_file_nums = 100; // 大概需要 13TB的存储空间，每个盘小于 7TB
const int device_id = 1;
int main(int argc, char** argv) {
    // 设置当前进程在GPU 0上运行
    cudaError_t err = cudaSetDevice(1);
    if (err != cudaSuccess) {
        printf("Failed to set CUDA device 0: %s\n", cudaGetErrorString(err));
        return 1;
    }
    vector<size_t> GPU_file_shape = {2, 32, 524288}; // 32MB

    GeminiFS geminifs("/home/qs/CompanionFS/Geminifs/sys_config.ini", GPU_file_nums, GPU_file_shape, 0);
    GPUFileId file_id;
    bool success = geminifs.geminifs_gpu_open_file(device_id, file_id);
    assert(success);
    printf("finish open file, file id %u\n", file_id);
    auto key_cache = torch::rand({4, 1024, 1024, 2}, // 512kb
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false)
        );


    auto key_cache2 = torch::rand({4, 1024, 1024, 2}, // 512kb
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false)
        );
    geminifs.geminifs_register_tensor_with_gpu(key_cache);
    printf("finish register tensor keycache\n");
    geminifs.geminifs_register_tensor_with_gpu(key_cache2);
    printf("finish register tensor keycache2\n");
    geminifs.geminifs_GPU_write_kernel(key_cache, file_id, geminifs.geminifs_get_gpu_controller(device_id));
    printf("finish write kernel\n");
    geminifs.geminifs_GPU_read_kernel(key_cache2, file_id, geminifs.geminifs_get_gpu_controller(device_id));
    printf("finish read kernel\n");
    if (!torch::all(key_cache == key_cache2).item<bool>()) {
        // 将GPU上的Tensor转移到CPU进行打印
        std::cout << "Tensors are not equal!" << std::endl;
        
        auto key_cache_flattened = key_cache.flatten().slice(0, 0, 20);
        auto key_cache2_flattened = key_cache2.flatten().slice(0, 0, 20);

        // 打印前100个元素
        std::cout << "First 20 elements of key_cache: " << key_cache_flattened << std::endl;
        std::cout << "First 20 elements of key_cache2: " << key_cache2_flattened << std::endl;
            
        // 触发断言失败
        TORCH_CHECK(false, "Tensors are not equal!");
    }
    return 0;
}