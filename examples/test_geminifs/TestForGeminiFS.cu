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



torch::Tensor create_kv_layer(int block_size,
                              int num_kv_head,
                              int head_dim,
                              int device_id) {
    return torch::rand(
        {block_size, num_kv_head, head_dim},
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false));
}

double check_kv_equal(const torch::Tensor& t1,
                      const torch::Tensor& t2,
                      const std::string& tag) {
    TORCH_CHECK(t1.sizes() == t2.sizes(), "Shape mismatch in ", tag);
    TORCH_CHECK(t1.dtype() == t2.dtype(), "Dtype mismatch in ", tag);
    TORCH_CHECK(t1.device().type() == t2.device().type(),
                "Device type mismatch in ", tag);

    auto t1_contig = t1.contiguous();
    auto t2_contig = t2.contiguous();

    if (torch::equal(t1_contig, t2_contig)) {
        return 0.0;
    }

    TORCH_CHECK(t1_contig.numel() == t2_contig.numel(),
                "Tensor element count mismatch in ", tag);

    auto diff_mask = torch::ne(t1_contig, t2_contig);
    const auto mismatch_elems = diff_mask.count_nonzero().item<int64_t>();
    if (mismatch_elems == 0) {
        return 0.0;
    }

    const auto total_elems = diff_mask.numel();
    const double ratio = static_cast<double>(mismatch_elems) /
                         static_cast<double>(total_elems);

    auto flat_mask = diff_mask.flatten();
    auto mismatch_indices = torch::nonzero(flat_mask);
    if (mismatch_indices.numel() > 0) {
        const int64_t flat_index = mismatch_indices[0].item<int64_t>();
        auto t1_flat = t1_contig.flatten();
        auto t2_flat = t2_contig.flatten();
        double ref_val = t1_flat[flat_index].item<double>();
        double cand_val = t2_flat[flat_index].item<double>();

        std::cerr << "First mismatch at flat index " << flat_index
                  << " (tag: " << tag << ")\n"
                  << "Reference value: " << ref_val << "\n"
                  << "Candidate value: " << cand_val << std::endl;
    }

    return ratio;
}

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
    size_t reg_granularity = block_size * num_kv_heads *head_dim * 2 ;// reg the KV cache per layer

    geminifs.geminifs_register_tensor_with_gpu(K_vec, reg_granularity);
    geminifs.geminifs_register_tensor_with_gpu(V_vec, reg_granularity);

    // --- Simulate Layer-wise Cache Slicing ---
    // In vLLM, blocks are dynamically allocated. For this test, we'll just create
    // views into the main tensor to represent the cache for different layers.
    // Here, we split the K and V tensors based on the granularity.
    std::vector<torch::Tensor> k_caches_per_layer;
    std::vector<torch::Tensor> v_caches_per_layer;
    const int total_blocks = block_size * num_layers;
    k_caches_per_layer.reserve(total_blocks);
    v_caches_per_layer.reserve(total_blocks);

    for (int i = 0; i < total_blocks; ++i) {
        // Each slice is one "block" from the perspective of the cache.
        k_caches_per_layer.push_back(K_vec.slice(0, i, i + 1));
        v_caches_per_layer.push_back(V_vec.slice(0, i, i + 1));
    }
    std::cout << "Created " << k_caches_per_layer.size() << " block-wise views into the cache tensors." << std::endl;

    // --- Batch Read/Write Test ---
    // We will test writing and reading the caches for the first 4 layers
    // 
    int transfer_layers = 256;
    std::vector<torch::Tensor> k_tensors_write;
    std::vector<torch::Tensor> v_tensors_write;
    k_tensors_write.reserve(transfer_layers);
    v_tensors_write.reserve(transfer_layers);

    for(int i = 0; i < transfer_layers; ++i) {
        k_tensors_write.push_back(k_caches_per_layer[i]);
        v_tensors_write.push_back(v_caches_per_layer[i]);
    }

    // Prepare file IDs for the batch operation
    std::vector<GPUFileId> batch_file_ids(file_ids.begin(), file_ids.begin() + transfer_layers);

    // Perform batched write
    std::cout << "Performing batched write for the first " << transfer_layers << " layers..." << std::endl;
    geminifs.geminifs_batched_write(
        k_tensors_write, v_tensors_write,
        batch_file_ids, 0, // layer_idx = 0 (not used in this context)
        geminifs.geminifs_get_gpu_controller(device_id), stream);
    cudaStreamSynchronize(stream);  
    std::cout << "Batched write completed." << std::endl;

    // Create new tensors to read data back into
    std::vector<torch::Tensor> k_tensors_read;
    std::vector<torch::Tensor> v_tensors_read;
    k_tensors_read.reserve(transfer_layers);
    v_tensors_read.reserve(transfer_layers);
    
    for (int i = 0; i < transfer_layers; ++i) {
        k_tensors_read.push_back(k_caches_per_layer[i + transfer_layers]);
        v_tensors_read.push_back(v_caches_per_layer[i + transfer_layers]);
    }

    // Perform batched read
    std::cout << "Performing batched read for the first " << transfer_layers << " layers..." << std::endl;
    geminifs.geminifs_batched_read(
        k_tensors_read, v_tensors_read,
        batch_file_ids, 0, // layer_idx = 0
        geminifs.geminifs_get_gpu_controller(device_id), stream);
    cudaStreamSynchronize(stream);
    std::cout << "Batched read completed." << std::endl;

    // --- Verification ---
    std::cout << "Verifying data integrity..." << std::endl;
    bool all_match = true;
    double max_ratio = 0.0;

    for (int i = 0; i < transfer_layers; i++) {
        const std::string k_tag = "batched.k_layer" + std::to_string(i);
        const std::string v_tag = "batched.v_layer" + std::to_string(i);

        double k_ratio = check_kv_equal(k_tensors_write[i], k_tensors_read[i], k_tag);
        double v_ratio = check_kv_equal(v_tensors_write[i], v_tensors_read[i], v_tag);

        if (k_ratio > 0.0) {
            all_match = false;
            max_ratio = std::max(max_ratio, k_ratio);
            std::cerr << k_tag << " mismatch ratio: " << k_ratio << std::endl;
        }
        if (v_ratio > 0.0) {
            all_match = false;
            max_ratio = std::max(max_ratio, v_ratio);
            std::cerr << v_tag << " mismatch ratio: " << v_ratio << std::endl;
        }
    }

    if (!all_match) {
        std::cerr << "Verification failed. Max mismatch ratio: " << max_ratio << std::endl;
        return 1;
    }

    std::cout << "Verification successful for all " << transfer_layers << " layers." << std::endl;

    // --- Cleanup ---
    cudaStreamDestroy(stream);
    // Unregistering is important, but the current GeminiFS class doesn't seem to have
    // a specific unregister method. Assuming cleanup is handled in the destructor.
    std::cout << "Test finished successfully." << std::endl;
    return 0;



    /* The original test logic is preserved below but commented out as it's not compatible.
    // One layer per tensor (key/value separate)
    auto k0 = create_kv_layer();
    auto v0 = create_kv_layer();
    auto k0_r = create_kv_layer(); // read-back buffers
    auto v0_r = create_kv_layer();

    // auto k1 = create_kv_layer();
    // auto v1 = create_kv_layer();
    // auto k1_r = create_kv_layer();
    // auto v1_r = create_kv_layer();

    // Register all tensors
    for (auto* t : {&k0, &v0, &k0_r, &v0_r, &k1, &v1, &k1_r, &v1_r})
        geminifs.geminifs_register_tensor_with_gpu(*t);
    

    auto gpu_controller =  geminifs.geminifs_get_gpu_controller(device_id);
    
    // Single layer write/read (per layer interface)
    geminifs.geminifs_GPU_write_kernel(k0, v0, file_id,
        geminifs.geminifs_get_gpu_controller(device_id), 0);
    geminifs.geminifs_GPU_read_kernel(k0_r, v0_r, file_id,
        geminifs.geminifs_get_gpu_controller(device_id), 0);

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
        geminifs.geminifs_get_gpu_controller(device_id), 0);

    geminifs.geminifs_batched_read(
        key_layers_read, value_layers_read,
        file_ids, layer_ids,
        geminifs.geminifs_get_gpu_controller(device_id), 0);

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
        geminifs.geminifs_get_gpu_controller(device_id), 0);

    geminifs.geminifs_batched_read(
        key_layers_write, value_layers_write, file_ids, layer_ids,
        geminifs.geminifs_get_gpu_controller(device_id), 0);

    for (size_t i = 0; i < key_layers_write.size(); ++i) {
        check_kv_equal(key_layers_write[i], key_layers_read[i],
                       "batched2.layer" + std::to_string(i) + ".key");
        check_kv_equal(value_layers_write[i], value_layers_read[i],
                       "batched2.layer" + std::to_string(i) + ".value");
    }
    */
}