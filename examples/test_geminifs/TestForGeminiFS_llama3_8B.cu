#include "geminifs.cuh"
#include "gpu_controller.cuh"
#include "geminifs_helper.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <algorithm>

using namespace std;

constexpr int GPU_file_nums = 10000;
constexpr int device_id = 1;

int main(int argc, char **argv)
{
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << device_id
                  << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

        // --- Llama3 8B Model & vLLM Config ---
    // Model parameters
    const int num_layers = 32;
    const int num_kv_heads = 8;
    const int head_dim = 128;
    // vLLM-style memory management parameters
    const int block_size = 256; // Number of tokens per block
    const int num_blocks = 2048; // Total blocks to fill 32GB for K and 32GB for V cache

    int res;
    // Adjust GPU file shape to hold one layer KV (you may tune)
    // Here: bytes = block_size * num_kv_head * head_dim * sizeof(fp16) * 2 (K+V)
    // We just keep original shape placeholder (modify according to real layout inside GeminiFS)
    vector<size_t> GPU_file_shape = {2, num_layers, block_size * num_kv_heads * head_dim * 2}; 
    GeminiFS geminifs("/home/qs/CompanionFS/Geminifs/sys_config.ini",
                      GPU_file_nums, GPU_file_shape, 1);

    std::vector<GPUFileId> file_ids(GPU_file_nums);
    std::cout << "open files .. " << std::endl;
    for (int i = 0; i < num_blocks; ++i) {
        res = geminifs.geminifs_gpu_open_file(device_id, file_ids[i]);
        if (res == false) {
            std::cerr << "Failed to open GPU file for block " << i << std::endl;
            return 1;
        }
    }

    std::cout << "open files ok " << std::endl;


    // --- Tensor Allocations ---
    // Simulating vLLM's paged KV cache memory pool.
    // Shape: [num_blocks, num_kv_heads, head_dim, block_size]
    std::cout << "Allocating K and V tensors to simulate vLLM's paged cache..." << std::endl;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, device_id);
    torch::Tensor K_vec = torch::randn({num_blocks*num_layers, block_size, num_kv_heads, head_dim}, options);
    torch::Tensor V_vec = torch::randn({num_blocks*num_layers, block_size, num_kv_heads, head_dim}, options);
    std::cout << "Allocation successful." << std::endl;

    TORCH_CHECK(K_vec.is_contiguous());
    TORCH_CHECK(V_vec.is_contiguous());

    // Register the entire tensor memory pool with GeminiFS
    size_t reg_granularity = static_cast<size_t>(block_size) * num_kv_heads * head_dim * K_vec.element_size(); // reg the KV cache per layer

    geminifs.geminifs_register_tensor_with_gpu(K_vec, reg_granularity);
    geminifs.geminifs_register_tensor_with_gpu(V_vec, reg_granularity);

    // --- Simulate Layer-wise Cache Slicing ---
    const int total_blocks = num_blocks * num_layers;

    int transfer_files = 2048;   // number of blocks/files per batched transfer
    int transfer_layers = 16;    // number of consecutive layers to sweep

    TORCH_CHECK(transfer_files <= num_blocks,
                "transfer_files exceeds available num_blocks");
    TORCH_CHECK(transfer_layers <= num_layers,
                "transfer_layers exceeds available num_layers");

    std::vector<std::vector<torch::Tensor>> k_tensors_write_layers(transfer_layers);
    std::vector<std::vector<torch::Tensor>> v_tensors_write_layers(transfer_layers);


    for (int layer_idx = 0; layer_idx < transfer_layers; ++layer_idx) {
        k_tensors_write_layers[layer_idx].reserve(transfer_files);
        v_tensors_write_layers[layer_idx].reserve(transfer_files);


        for (int block_idx = 0; block_idx < transfer_files; ++block_idx) {
            const int64_t flat_index = static_cast<int64_t>(layer_idx) * num_blocks + block_idx;
            TORCH_CHECK(flat_index < total_blocks,
                        "Requested block index exceeds allocated tensors");
            auto k_slice = K_vec.slice(0, flat_index, flat_index + 1);
            auto v_slice = V_vec.slice(0, flat_index, flat_index + 1);
            k_tensors_write_layers[layer_idx].push_back(k_slice);
            v_tensors_write_layers[layer_idx].push_back(v_slice);
        }
    }

    std::vector<GPUFileId> batch_file_ids(file_ids.begin(), file_ids.begin() + transfer_files);
    auto gpu_controller = geminifs.geminifs_get_gpu_controller(device_id);

    std::vector<double> write_durations_ms;
    write_durations_ms.reserve(transfer_layers);

    auto write_start = std::chrono::high_resolution_clock::now();
    for (int layer_idx = 0; layer_idx < transfer_layers; ++layer_idx) {
        geminifs.geminifs_batched_read(
            k_tensors_write_layers[layer_idx],
            v_tensors_write_layers[layer_idx],
            batch_file_ids,
            layer_idx,
            gpu_controller,
            stream);
    }
    auto pre_sync = std::chrono::high_resolution_clock::now();
    cudaStreamSynchronize(stream);
    auto write_end = std::chrono::high_resolution_clock::now();

    double launch_duration_ms = std::chrono::duration<double, std::milli>(pre_sync - write_start).count();
    double sync_duration_ms = std::chrono::duration<double, std::milli>(write_end - pre_sync).count();
    double total_duration_ms = std::chrono::duration<double, std::milli>(write_end - write_start).count();
    write_durations_ms.push_back(total_duration_ms);

    const double total_bytes = static_cast<double>(transfer_layers) *
                               static_cast<double>(transfer_files) *
                               static_cast<double>(reg_granularity) * 2.0;
    const double total_seconds = total_duration_ms / 1000.0;
    const double bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
    const double bandwidth_gib_s = (total_seconds > 0.0) ? (total_bytes / bytes_per_gib / total_seconds) : 0.0;

    std::cout << "Host launch duration before cudaStreamSynchronize (ms): "
              << launch_duration_ms << std::endl;
    std::cout << "cudaStreamSynchronize duration (ms): "
              << sync_duration_ms << std::endl;
    std::cout << "Total batched write duration (ms): "
              << total_duration_ms << std::endl;
    std::cout << "Overall transfer bandwidth (GiB/s): "
              << bandwidth_gib_s << std::endl;

    std::cout << "Batched write durations (ms):";
    for (double t : write_durations_ms) {
        std::cout << ' ' << t;
    }
    std::cout << std::endl;

    // --- Cleanup ---
    cudaStreamDestroy(stream);
    // Unregistering is important, but the current GeminiFS class doesn't seem to have
    // a specific unregister method. Assuming cleanup is handled in the destructor.
    std::cout << "Test finished successfully." << std::endl;
    return 0;
}