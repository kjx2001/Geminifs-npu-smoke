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
#include "nvme_controller.cuh"
#include "geminifs_mem.h"
#include "gpu_controller.cuh"

using ControllerPtr = std::shared_ptr<Controller>;




__host__ struct geminifs_metadata* geminifs_get_metadata(int device_id);
__host__ struct geminifs_dma* geminifs_get_dma(const torch::Tensor &tensor);
__host__ bool geminifs_create_dma(const torch::Tensor& tensor);


// geminifs_batch_create(int nr_device, int nr_files, size_t block_size, size_t file_size, int  cudaDevice);


__host__ DmaPtr createDmaTest(int idx, size_t block_size, int device);

__host__
DmaPtr getdeviceDmaTest(int idx, void* buffer, size_t size, int cudaDevice);





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