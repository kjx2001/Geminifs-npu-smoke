#pragma once

#include <cstdint>
#include <torch/all.h>

bool geminifs_init_fds_wrapper_cuda(const torch::Tensor& file_meta, 
                                    const std::string& mount_path, 
                                    const std::string& pcie_addr);

// bool geminifs_get_dma_wrapper_cuda(const torch::Tensor& tensor);

bool batch_read_direct(const torch::Tensor& cached_file_ids,   //shape = [num_cached_files,]
                                        const torch::Tensor& inner_block_ids,  // shape = [num_cached_files,]
                                        const torch::Tensor& key_cache,               // shape = [max_num_block, block_size, num_heads, head_size]
                                        const torch::Tensor& value_cache,             // shape = [max_num_block, block_size, num_heads, head_size]
                                        int64_t start_layer_idx);

bool batch_write_direct(const torch::Tensor& cached_file_ids,   //shape = [num_cached_files,]
                                        const torch::Tensor& inner_block_ids,  // shape = [num_cached_files,]
                                        const torch::Tensor& key_cache,               // shape = [max_num_block, block_size, num_heads, head_size]
                                        const torch::Tensor& value_cache,             // shape = [max_num_block, block_size, num_heads, head_size]
                                        int64_t start_layer_idx);

bool geminifs_device_mutiple_layer_read_wrapper_cuda(
                                        const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
                                        const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
                                        const std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
                                        const std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
                                        int64_t start_layer_idx, int64_t num_layers);

bool geminifs_device_mutiple_layer_write_wrapper_cuda(
                                        const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
                                        const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
                                        const std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
                                        const std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
                                        int64_t start_layer_idx, int64_t num_layers);

bool init_gpu_hash_wrapper(const torch::Tensor& metadata);

torch::Tensor batch_search_wrapper(const torch::Tensor &hash_keys);
torch::Tensor batch_insert_wrapper(const torch::Tensor &hash_keys);


bool batch_put_wrapper(torch::Tensor &hash_keys, 
                    torch::Tensor &keys,
                    torch::Tensor &values
);

bool batch_get_wrapper(const torch::Tensor &keys, const torch::Tensor &values);