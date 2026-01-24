#include "geminifs.cuh"
#include "gpu_controller.cuh"
#include "geminifs_helper.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

constexpr int GPU_file_nums = 100;
constexpr int device_id = 1;

constexpr int num_layers = 32;
// KV cache per layer shape: [block_size, num_kv_head, head_dim]
constexpr int block_size   = 256;
constexpr int num_kv_head  = 8;
constexpr int head_dim     = 128;
constexpr int num_kvs     = 512;
torch::Tensor create_kv_layer() {
    return torch::rand(
        {num_kvs, block_size, num_kv_head, head_dim},
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false));
}



int main(int argc, char **argv)
{
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << device_id
                  << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Adjust GPU file shape to hold one layer KV (you may tune)
    // Here: bytes = block_size * num_kv_head * head_dim * sizeof(fp16) * 2 (K+V)
    // We just keep original shape placeholder (modify according to real layout inside GeminiFS)
    vector<size_t> GPU_file_shape = {2, num_layers, block_size * num_kv_head * head_dim}; 
    GeminiFS geminifs("/home/qs//CompanionFS/Geminifs/sys_config.ini",
                      GPU_file_nums, GPU_file_shape, 0);



    // One layer per tensor (key/value separate)
    auto k0 = create_kv_layer();
    cout << "here is ok! 1\n"<< endl;
    size_t granularity = block_size * num_kv_head * head_dim;
    try {
        geminifs.geminifs_register_tensor_with_gpu(k0, granularity);
    } catch (const std::exception &e) {
        std::cerr << "geminifs_register_tensor_with_gpu failed: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "geminifs_register_tensor_with_gpu failed with an unknown error" << std::endl;
    }


    return 0;
}
