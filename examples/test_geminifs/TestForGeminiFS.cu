#include "geminifs.cuh"
#include "gpu_controller.cuh"
#include "geminifs_helper.h"
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

constexpr int GPU_file_nums = 100;
constexpr int device_id = 0;

constexpr int num_layers = 32;
// KV cache per layer shape: [block_size, num_kv_head, head_dim]
constexpr int block_size   = 256;
constexpr int num_kv_head  = 8;
constexpr int head_dim     = 128;

torch::Tensor create_kv_layer() {
    return torch::rand(
        {block_size, num_kv_head, head_dim},
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false));
}

void check_kv_equal(const torch::Tensor& t1,
                    const torch::Tensor& t2,
                    const std::string& tag) {
    TORCH_CHECK(t1.sizes() == t2.sizes(), "Shape mismatch in ", tag);
    if (!torch::all(t1 == t2).item<bool>()) {
        auto t1_flat = t1.flatten().slice(0, 0, 20).cpu();
        auto t2_flat = t2.flatten().slice(0, 0, 20).cpu();
        std::cout << "[Mismatch] " << tag << "\n"
                  << "First 20 A: " << t1_flat << "\n"
                  << "First 20 B: " << t2_flat << std::endl;
        TORCH_CHECK(false, "KV layer not equal: ", tag);
    }
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
    GeminiFS geminifs("/home/yjq/Geminifs/sys_config.ini",
                      GPU_file_nums, GPU_file_shape, 1);

    GPUFileId file_id, file_id2;
    assert(geminifs.geminifs_gpu_open_file(device_id, file_id));
    assert(geminifs.geminifs_gpu_open_file(device_id, file_id2));
    printf("Opened files: %u %u\n", file_id, file_id2);

    // One layer per tensor (key/value separate)
    auto k0 = create_kv_layer();
    auto v0 = create_kv_layer();
    auto k0_r = create_kv_layer(); // read-back buffers
    auto v0_r = create_kv_layer();

    auto k1 = create_kv_layer();
    auto v1 = create_kv_layer();
    auto k1_r = create_kv_layer();
    auto v1_r = create_kv_layer();

    // Register all tensors
    for (auto* t : {&k0, &v0, &k0_r, &v0_r, &k1, &v1, &k1_r, &v1_r})
        geminifs.geminifs_register_tensor_with_gpu(*t);

    // Single layer write/read (per layer interface)
    geminifs.geminifs_GPU_write_kernel(k0, v0, file_id,
        geminifs.geminifs_get_gpu_controller(device_id));
    geminifs.geminifs_GPU_read_kernel(k0_r, v0_r, file_id,
        geminifs.geminifs_get_gpu_controller(device_id));

    std::cout << "Single layer read/write done." << std::endl;
    check_kv_equal(k0, k0_r, "layer0.key");
    check_kv_equal(v0, v0_r, "layer0.value");

    // Batched (each element already one layer)
    vector<torch::Tensor> key_layers_write  = {k0, k1};
    vector<torch::Tensor> value_layers_write= {v0, v1};
    vector<GPUFileId>     file_ids          = {file_id, file_id2};
    int           layer_ids         = 1;

    vector<torch::Tensor> key_layers_read  = {k0_r, k1_r};
    vector<torch::Tensor> value_layers_read= {v0_r, v1_r};

    geminifs.geminifs_batched_write(
        key_layers_write, value_layers_write,
        file_ids, layer_ids,
        geminifs.geminifs_get_gpu_controller(device_id));

    geminifs.geminifs_batched_read(
        key_layers_read, value_layers_read,
        file_ids, layer_ids,
        geminifs.geminifs_get_gpu_controller(device_id));

    // Per layer equal check
    for (size_t i = 0; i < key_layers_write.size(); ++i) {
        check_kv_equal(key_layers_write[i], key_layers_read[i],
                       "batched.layer" + std::to_string(i) + ".key");
        check_kv_equal(value_layers_write[i], value_layers_read[i],
                       "batched.layer" + std::to_string(i) + ".value");
    }

    // Alternate interface variant (if required by API)
    geminifs.geminifs_batched_write(
        key_layers_write, value_layers_write, file_ids, layer_ids,
        geminifs.geminifs_get_gpu_controller(device_id));

    geminifs.geminifs_batched_read(
        key_layers_write, value_layers_write, file_ids, layer_ids,
        geminifs.geminifs_get_gpu_controller(device_id));

    for (size_t i = 0; i < key_layers_write.size(); ++i) {
        check_kv_equal(key_layers_write[i], key_layers_read[i],
                       "batched2.layer" + std::to_string(i) + ".key");
        check_kv_equal(value_layers_write[i], value_layers_read[i],
                       "batched2.layer" + std::to_string(i) + ".value");
    }

    return 0;
}
