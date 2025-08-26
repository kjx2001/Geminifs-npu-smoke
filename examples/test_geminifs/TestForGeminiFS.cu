#include "geminifs.cuh"
#include "gpu_controller.cuh"
#include "geminifs_helper.h"
#include "gpu_controller.cuh"
#include <iostream>
#include <vector>
using namespace std;

constexpr size_t GPU_file_size = 32ull * 1024 * 1024; // 8MB
constexpr int GPU_file_nums = 10000; // 大概需要 13TB的存储空间，每个盘小于 7TB
int main(int argc, char** argv) {
    // 设置当前进程在GPU 0上运行
    cudaError_t err = cudaSetDevice(1);
    if (err != cudaSuccess) {
        printf("Failed to set CUDA device 0: %s\n", cudaGetErrorString(err));
        return 1;
    }
    vector<size_t> GPU_file_shape = {2, 32, 524288}; // 32MB

    GeminiFS geminifs("/home/qs/CompanionFS/Geminifs/sys_config.ini", GPU_file_nums, GPU_file_shape,0);
    // geminifs.geminifs_gpu_open_file(0, 1, file_size, O_DEVICE);
    // printf("finish open file\n");
    // auto key_cache = torch::rand({4, 1024, 1024, 2}, // 512kb
    //     torch::TensorOptions()
    //         .dtype(torch::kFloat16)
    //         .device(torch::kCUDA, 0)
    //         .pinned_memory(false)
    //     );


    // auto key_cache2 = torch::rand({4, 1024, 1024, 2}, // 512kb
    //     torch::TensorOptions()
    //         .dtype(torch::kFloat16)
    //         .device(torch::kCUDA, 0)
    //         .pinned_memory(false)
    //     );
    // geminifs.geminifs_register_tensor_with_gpu(key_cache);
    // printf("finish register tensor keycache\n");
    // geminifs.geminifs_register_tensor_with_gpu(key_cache2);
    // printf("finish register tensor keycache2\n");
    // geminifs.geminifs_GPU_write_kernel(key_cache, 1, geminifs.geminifs_get_gpu_controller(0));
    // printf("finish write kernel\n");
    // geminifs.geminifs_GPU_read_kernel(key_cache2, 1, geminifs.geminifs_get_gpu_controller(0));
    // printf("finish read kernel\n");
    // torch::all(key_cache == key_cache2);
    return 0;
}