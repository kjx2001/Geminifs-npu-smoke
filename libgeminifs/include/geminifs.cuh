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

  
class GeminiFS {
    public:
        GeminiFS(const std::string& config_file_path, int GPU_file_nums, const std::vector<size_t>& GPU_file_shape, bool reset = false) {
            init(config_file_path, GPU_file_nums, GPU_file_shape, reset);
        }

        ~GeminiFS() {
            cleanup();
        }

        __host__ bool geminifs_batched_read(const std::vector<torch::Tensor>& k_caches, const std::vector<torch::Tensor>& v_caches, const std::vector<GPUFileId>& gpu_file_ids, int layer_idx, GPUControllerPtr gpu_controller, cudaStream_t stream); 
        __host__ bool geminifs_batched_write(const std::vector<torch::Tensor>& k_caches, const std::vector<torch::Tensor>& v_caches, const std::vector<GPUFileId>& gpu_file_ids, int layer_idx, GPUControllerPtr gpu_controller, cudaStream_t stream);
        /**
        * GPU read kernel
        */
        __forceinline__ __host__ bool geminifs_xfer_kernel(const torch::Tensor& tensor, GPUFileId gpu_file_id, loff_t off, 
                                                           GPUControllerPtr gpu_controller, bool is_read, cudaStream_t stream);

        __forceinline__ __host__ bool geminifs_kv_xfer_kernel(const torch::Tensor& k_cache, 
                                                              const torch::Tensor& v_cache, 
                                                              GPUFileId gpu_file_id, loff_t off, 
                                                              GPUControllerPtr gpu_controller, 
                                                              bool is_read, cudaStream_t stream);



        __host__ bool geminifs_GPU_read_kernel(torch::Tensor& tensor, GPUFileId gpu_file_id, loff_t off, GPUControllerPtr gpu_controller, cudaStream_t stream);
        /**
        * GPU write kernel
        */
        __host__ bool geminifs_GPU_write_kernel(const torch::Tensor& tensor, GPUFileId gpu_file_id, loff_t off, GPUControllerPtr gpu_controller, cudaStream_t stream);
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

        __host__ bool geminifs_register_tensors_with_gpu(const std::vector<torch::Tensor>& tensor_list, uint64_t granularity = 0);

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
        __host__ bool geminifs_gpu_open_file(int device_id, GPUFileId& gpu_file_id);

        __host__ bool geminifs_gpu_close_file(int device_id, GPUFileId gpu_file_id);

        /**
         * Cleanup all GPU controllers
         */
        __host__ void geminifs_cleanup_all_gpu_controllers();

        /**
         * Delete all files managed by a specific NVMe controller
         */
        __host__ bool geminifs_nvme_delete_all_files(int device_id, size_t controller_index = 0);

        /**
         * Get initialization parameters - file configuration
         */
        __host__ size_t get_init_num_files() const { return init_GPU_num_files_; }
        __host__ size_t get_init_file_size() const { return init_GPU_file_size_; }
        __host__ size_t get_total_init_storage_size() const { return init_GPU_num_files_ * init_GPU_file_size_; }
        __host__ bool is_initialized() const { return is_init_; }
    private:

        __forceinline__ __host__ bool geminifs_batched_xfer(const std::vector<torch::Tensor>& k_caches, 
                                                            const std::vector<torch::Tensor>& v_caches, 
                                                            const std::vector<GPUFileId>& gpu_file_ids, 
                                                            int layer_idx, GPUControllerPtr gpu_controller,
                                                            bool is_read, cudaStream_t stream);

        __forceinline__ __host__ bool get_nvme_files(GPUFileId gpu_file_id, 
                                                     NVMeFilesSpan &out_nvme_files);

        __forceinline__ __host__ bool get_prp_mappings(const torch::Tensor& tensor, 
                                                       GPUControllerPtr gpu_controller, 
                                                       PRPMappingEntrySpan &out_mappings);
        /**
         * Add an NVMe controller to a GPU controller
         */
        __host__ bool geminifs_add_nvme_to_gpu(int device_id, const nvme_ctrl_param& params);
        
        /**
         * Initialize GeminiFS
         */
        __host__ void init(const std::string& config_file_path, int GPU_file_nums ,const std::vector<size_t>& GPU_file_shape, bool reset = false);

        /**
         * Parse and setup GPU and NVMe controllers from config file
         */
        __host__ bool parse_and_setup_controllers(const std::string& config_file_path, int current_gpu_id);

        /**
         * Handle GPU file reset operations
         */
        __host__ void handle_gpu_file_reset(int current_gpu_id);

        /**
         * Cleanup GeminiFS
         */
        __host__ void cleanup();
        



        bool is_init_ = false;
        
        // 初始化参数 - 文件配置
        size_t init_GPU_num_files_ = 0;      // 需要创建的文件个数
        size_t init_GPU_file_size_ = 0;      // 每个文件的大小
        
        std::unique_ptr<GPUFileManager> gpu_file_manager_; // 用于GPU文件管理（open create delete）， 以及相应的持久化
        std::vector<GPUControllerPtr> gpu_controllers_;
        std::vector<nvme_ctrl_param> nvme_params_;
};

#endif