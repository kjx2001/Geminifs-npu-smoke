#ifndef __GEMINIFS_CUH__
#define __GEMINIFS_CUH__

// #include "geminifs.h"
// #include "ctrl.h"
// #include "buffer.h"
#include "file.cuh"
#include <cstdint>
#include <torch/all.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <memory>
#include <concepts>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <filesystem>
#include "nvme_file.h"
#include "nvme_controller.h"
#include "geminifs_mem.h"
#include "gpu_controller.h"

using ControllerPtr = std::shared_ptr<Controller>;



__host__ GPUFilePool* geminifs_init_fds(size_t nr_files, size_t file_size);
__host__ struct geminifs_metadata* geminifs_get_metadata(int device_id);
__host__ struct geminifs_dma* geminifs_get_dma(const torch::Tensor &tensor);
__host__ bool geminifs_create_dma(const torch::Tensor& tensor);


__host__ std::vector<ControllerPtr> 
host_open_ctrls(struct geminifs_ctrl_params *ctrl_params);

__host__ std::vector<ControllerPtr> 
geminifs_nvme_host_open_ctrls(struct geminifs_ctrl_params *ctrl_params);


__host__ void 
geminifs_nvme_host_close_ctrls(std::vector<ControllerPtr> &ctrls); 


// geminifs_batch_create(int nr_device, int nr_files, size_t block_size, size_t file_size, int  cudaDevice);

__host__ GPUFile* 
geminifs_file_batch_create(std::vector<ControllerPtr> &ctrls, GPUPoolId pool_id, int nr_files, 
                        size_t block_size, size_t file_size, int cudaDevice);

__host__ DmaPtr createDmaTest(int idx, size_t block_size, int device);

__host__
DmaPtr getdeviceDmaTest(int idx, void* buffer, size_t size, int cudaDevice);




/*------------------------------xfer function-----------------------------------*/
__host__ void 
geminifs_device_xfer(GPUFile *file, uint64_t *ioaddr, 
                        size_t file_offset, size_t nbytes,
                        enum FileXferType type);
                        
__host__ void
geminifs_device_batch_xfer(GPUFile *files, uint64_t **ioaddr, 
                            size_t file_offset, size_t nbytes,
                            enum FileXferType type, size_t batch_size);

__global__ void 
__geminifs_device_batch_xfer_once(GPUFilePool *global_pool, 
                            GPUFileId file_id, uint64_t ioaddr,
                            size_t file_offset, size_t nbytes, 
                            enum FileXferType type);


__global__ void 
__geminifs_device_batch_xfer_once2(GPUFilePool *global_pool, 
                            GPUFileId file_id, cuda::std::span<uint64_t> ioaddr,
                            size_t file_offset, size_t nbytes, 
                            enum FileXferType type);




/*------------------------------xfer test-----------------------------------*/
bool geminifs_init_fds_wrapper_cuda_test(int64_t nr_files, int64_t file_size, int64_t device_id, 
                                        const std::string& mount_path, const std::string& pcie_addr);

bool geminifs_device_xfer_wrapper_test(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    const torch::Tensor &key_cache,         // shape = [max_num_block, block_size, num_heads, head_size]
    const torch::Tensor &value_cache,       // shape = [max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, enum FileXferType type);

// bool geminifs_get_dma_wrapper_cuda_test(const torch::Tensor& tensor);

bool geminifs_device_xfer_wrapper_test2(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, int64_t num_layers, enum FileXferType type);

/**
 * Create and register a GPU controller for a specific device
 */
__host__ GPUControllerPtr geminifs_create_gpu_controller(int device_id, const std::string& mount_base_path);

/**
 * Get GPU controller for a specific device
 */
__host__ GPUControllerPtr geminifs_get_gpu_controller(int device_id);

/**
 * Add an NVMe controller to a GPU controller
 */
__host__ bool geminifs_add_nvme_to_gpu(int device_id, const nvme_ctrl_param& params);

/**
 * Register tensor memory with GPU controller
 */
__host__ bool geminifs_register_tensor_with_gpu(const torch::Tensor& tensor);

/**
 * Unregister tensor memory from GPU controller
 */
__host__ bool geminifs_unregister_tensor_from_gpu(const torch::Tensor& tensor);

/**
 * Get DMA context from GPU controller
 */
__host__ struct geminifs_dma* geminifs_get_tensor_dma_from_gpu(const torch::Tensor& tensor);

/**
 * Open file using GPU controller
 */
__host__ void* geminifs_gpu_open_file(int device_id, const std::string& filename, size_t file_size, uint32_t o_flag, size_t controller_index = 0);

/**
 * Cleanup all GPU controllers
 */
__host__ void geminifs_cleanup_all_gpu_controllers();

/**
 * Delete all files managed by a specific NVMe controller
 */
__host__ bool geminifs_nvme_delete_all_files(int device_id, size_t controller_index = 0);

#endif