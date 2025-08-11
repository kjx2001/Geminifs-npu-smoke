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
#include "utils.cuh"
#include <list>

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
__host__ bool geminifs_register_tensor_with_gpu(const torch::Tensor& tensor, uint64_t granularity = 0);

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

class GeminiFS {
    public:
        GeminiFS(const std::string& config_file_path) {
            init(config_file_path);
        }
        ~GeminiFS() {
            cleanup();
        }

        /**
        * GPU read kernel
        */
        __host__ bool geminifs_GPU_read_kernel(torch::Tensor& tensor, GPUFileId gpu_file_id, GPUControllerPtr gpu_controller);

        /**
        * GPU write kernel
        */
        __host__ bool geminifs_GPU_write_kernel(const torch::Tensor& tensor, GPUFileId gpu_file_id, GPUControllerPtr gpu_controller);

        /**
         * Create and register a GPU controller for a specific device
         */
        __host__ GPUControllerPtr geminifs_create_gpu_controller(int device_id, const std::string& mount_base_path);

        /**
         * Get GPU controller for a specific device
         */
        __host__ GPUControllerPtr geminifs_get_gpu_controller(int device_id);

        /**
         * Register tensor memory with GPU controller
         */
        __host__ bool geminifs_register_tensor_with_gpu(const torch::Tensor& tensor, uint64_t granularity = 0);

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
        __host__ void* geminifs_gpu_open_file(int device_id, GPUFileId gpu_file_id, size_t file_size, uint32_t o_flag);

        /**
         * Cleanup all GPU controllers
         */
        __host__ void geminifs_cleanup_all_gpu_controllers();

        /**
         * Delete all files managed by a specific NVMe controller
         */
        __host__ bool geminifs_nvme_delete_all_files(int device_id, size_t controller_index = 0);
    private:
        /**
         * Add an NVMe controller to a GPU controller
         */
        __host__ bool geminifs_add_nvme_to_gpu(int device_id, const nvme_ctrl_param& params);
        
        /**
         * Initialize GeminiFS
         */
        __host__ void init(const std::string& config_file_path);

        /**
         * Cleanup GeminiFS
         */
        __host__ void cleanup();
        
        // NVMe_Link 主机端缓存（LRU，最大 5GB）
        struct CachedLinksEntry {
            std::vector<NVMe_Link> links;
            size_t bytes;
            std::list<GPUFileId>::iterator lru_it;
        };
        bool get_links_cached(GPUFileId file_id, std::vector<NVMe_Link>& out_links);
        void put_links_cache(GPUFileId file_id, const std::vector<NVMe_Link>& links);
        void touch_links_cache(GPUFileId file_id);
        void evict_links_cache_if_needed();

        static constexpr size_t LINKS_CACHE_MAX_BYTES = 5ULL * 1024ULL * 1024ULL * 1024ULL;
        size_t links_cache_bytes_ = 0;
        std::unordered_map<GPUFileId, CachedLinksEntry> links_cache_;
        std::list<GPUFileId> links_cache_lru_;

        bool is_init_ = false;
        GPUFileManager gpu_file_manager_;
        std::vector<GPUControllerPtr> gpu_controllers_;
        std::vector<nvme_ctrl_param> nvme_params_;
};

#endif