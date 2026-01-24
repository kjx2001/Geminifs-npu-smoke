#include <atomic>
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <sys/types.h> 
#include <sys/stat.h>
#include <sys/resource.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include <ctrl.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#include <sys/file.h>
#include <fcntl.h>

#include <torch/library.h>
#include <torch/torch.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>          // 包含 device_of 和其他张量工具
#include <thrust/device_ptr.h>
#include <cuda/std/span>


#include "buffer.h"
#include "geminifs.h"
#include "geminifs_helper.h"
#include "gpu_file_manager.cuh"
#include "prp_mapping_entry.h"
#include "torch/types.h"
#include "nvm_error.h"
#include "file.cuh"
#include "utils.cuh"
#include "geminifs.cuh"
#include "nvme_controller.cuh"
#include "gpu_controller.cuh"





struct DMAInfo{
    uint64_t *vaddr;
    uint64_t ioaddr_base;
    DmaPtr dma_ptr;
};


/**
 * Create and register a GPU controller for a specific device
 */
__host__ GPUControllerPtr GeminiFS::geminifs_create_gpu_controller(int device_id, const std::string& mount_base_path) {
    auto gpu_controller = std::make_shared<GPUController>(device_id, mount_base_path);
    
    if (!gpu_controller->isInitialized()) {
        geminifs_error("geminifs_create_gpu_controller: Failed to initialize GPU controller for device %d\n", device_id);
        return nullptr;
    }
    
    // Register with the global registry
    auto& registry = GPUControllerRegistry::getInstance();
    if (!registry.registerGPUController(device_id, gpu_controller)) {
        geminifs_error("geminifs_create_gpu_controller: Failed to register GPU controller for device %d\n", device_id);
        return nullptr;
    }
    
    geminifs_debug("geminifs_create_gpu_controller: Successfully created and registered GPU controller for device %d\n", device_id);
    return gpu_controller;
}



/**
 * Get GPU controller for a specific device
 */
__host__ GPUControllerPtr GeminiFS::geminifs_get_gpu_controller(int device_id) {
    auto& registry = GPUControllerRegistry::getInstance();
    return registry.getGPUController(device_id);
}



/**
 * Add an NVMe controller to a GPU controller
 */
__host__ bool GeminiFS::geminifs_add_nvme_to_gpu(int device_id, const nvme_ctrl_param& params) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_add_nvme_to_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    // Create modified parameters with mount path under GPU controller
    nvme_ctrl_param modified_params = params;
    std::filesystem::path gpu_mount_path = gpu_controller->getMountBasePath();
    std::filesystem::path nvme_mount_path = gpu_mount_path / ("nvme-" + params.pci_addr);
    modified_params.mount_path = nvme_mount_path.string();
    
    geminifs_debug("geminifs_add_nvme_to_gpu: Creating NVMe controller with mount path '%s' under GPU path '%s'\n", 
                   modified_params.mount_path.c_str(), gpu_mount_path.c_str());
    
    // Create NVMe controller with modified mount path
    auto nvme_controller = std::make_shared<NVMeController>(modified_params);
    if (!nvme_controller->is_initialized()) {
        geminifs_error("geminifs_add_nvme_to_gpu: Failed to initialize NVMe controller\n");
        return false;
    }
    
    // Add to GPU controller
    if (!gpu_controller->addNVMeController(nvme_controller)) {
        geminifs_error("geminifs_add_nvme_to_gpu: Failed to add NVMe controller to GPU %d\n", device_id);
        return false;
    }
    
    geminifs_debug("geminifs_add_nvme_to_gpu: Successfully added NVMe controller to GPU %d with mount path '%s'\n", 
                   device_id, modified_params.mount_path.c_str());
    return true;
}



/**
 * Register tensor memory with GPU controller
 */
__host__ bool GeminiFS::geminifs_register_tensor_with_gpu(const torch::Tensor& tensor, uint64_t granularity) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_register_tensor_with_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    // geminifs_info(" here is ok!\n");
    return gpu_controller->registerTensorMemory(tensor, granularity);
}

__host__ bool GeminiFS::geminifs_register_tensors_with_gpu(const std::vector<torch::Tensor>& tensor_list, uint64_t granularity) {
    int device_id = tensor_list[0].device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);

    if (!gpu_controller) {
        geminifs_error("geminifs_register_tensors_with_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }

    return gpu_controller->registerTesnsorList(tensor_list, granularity);
}


/**
 * Unregister tensor memory from GPU controller
 */
__host__ bool GeminiFS::geminifs_unregister_tensor_from_gpu(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_unregister_tensor_from_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    return gpu_controller->unregisterTensorMemory(tensor.data_ptr());
}

/**
 * Get DMA context from GPU controller
 */
__host__ struct geminifs_dma* GeminiFS::geminifs_get_tensor_dma_from_gpu(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_get_tensor_dma_from_gpu: No GPU controller found for device %d\n", device_id);
        return nullptr;
    }
    
    return gpu_controller->getDMAContext(tensor.data_ptr());
}

/**
 * Open file using GPU controller
 */
__host__ bool GeminiFS::geminifs_gpu_open_file(int device_id, GPUFileId& id) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_gpu_open_file: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    if (!gpu_file_manager_) {
        geminifs_error("geminifs_gpu_open_file: GPUFileManager not initialized\n");
        return false;
    }

    return gpu_file_manager_->openGPUFile(id);
}




__host__ bool GeminiFS::geminifs_gpu_close_file(int device_id, GPUFileId id) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_gpu_close_file: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    if (!gpu_file_manager_) {
        geminifs_error("geminifs_gpu_close_file: GPUFileManager not initialized\n");
        return false;
    }

    return gpu_file_manager_->closeGPUFile(id);
}   

__host__ bool GeminiFS::geminifs_batched_read(const std::vector<torch::Tensor>& k_caches, 
                                               const std::vector<torch::Tensor>& v_caches, 
                                               const std::vector<GPUFileId>& gpu_file_ids, 
                                               int layer_idx, GPUControllerPtr gpu_controller,
                                               cudaStream_t stream) {

    return geminifs_batched_xfer(k_caches, v_caches, gpu_file_ids, layer_idx, gpu_controller, true, stream);
}

__host__ bool GeminiFS::geminifs_batched_write(const std::vector<torch::Tensor>& k_caches, 
                                               const std::vector<torch::Tensor>& v_caches, 
                                               const std::vector<GPUFileId>& gpu_file_ids, 
                                               int layer_idx, GPUControllerPtr gpu_controller,
                                               cudaStream_t stream) {
    return geminifs_batched_xfer(k_caches, v_caches, gpu_file_ids, layer_idx, gpu_controller, false, stream);
}

using IoctxDataPair = std::pair<std::vector<BatchIoEntry*>, GPUFileManager*>;
void CUDART_CB release_ioctx(void *data) {
    auto release = (IoctxDataPair *)data;
    auto batches = release->first;
    auto manager = release->second;
    if (!manager) {
        geminifs_error("file manager is null");
        delete release;
        return;
    }

    for (auto &entry: batches) {
        if (entry)manager->releaseIoContexts(entry);
    }
    delete release;
}

__forceinline__ __host__ bool 
GeminiFS::geminifs_batched_xfer(const std::vector<torch::Tensor>& k_caches, 
                                const std::vector<torch::Tensor>& v_caches, 
                                const std::vector<GPUFileId>& gpu_file_ids, 
                                int layer_idx, GPUControllerPtr gpu_controller,
                                bool is_read, cudaStream_t stream) {
    if (k_caches.empty() || k_caches.size() != v_caches.size() || k_caches.size() != gpu_file_ids.size()) {
        geminifs_error("Invalid params: k=%zu v=%zu layers=%d files=%zu",
                       k_caches.size(), v_caches.size(),
                       layer_idx, gpu_file_ids.size());
        return false;
    }

    // Check for tensor and file size consistency
    uint64_t tensor_size = k_caches[0].numel() * k_caches[0].element_size();
    GPUFileDesc first_file_desc;
    if (!gpu_file_manager_->getGPUFileById(gpu_file_ids[0], first_file_desc)) {
        geminifs_error("geminifs_batched_xfer: Failed to get description for first GPU file ID %u\n", gpu_file_ids[0]);
        return false;
    }
    size_t total_file_size = first_file_desc.total_file_size;
    size_t tensor_object_size = first_file_desc.tensor_shape[2];
    // Check that the write operation is within bounds for the first file
    if (tensor_size != tensor_object_size) {
        geminifs_error("geminifs_batched_xfer: Not enough space in GPU file\n");
        return false;
    }

    for (size_t i = 1; i < k_caches.size(); ++i) {
        // Check K-tensor size consistency
        if (k_caches[i].numel() * k_caches[i].element_size() != tensor_size) {
            geminifs_error("geminifs_batched_xfer: K-cache tensor size mismatch at index %zu. Expected %lu, got %lu.\n",
                           i, tensor_size, k_caches[i].numel() * k_caches[i].element_size());
            return false;
        }
        // Check V-tensor size consistency
        if (v_caches[i].numel() * v_caches[i].element_size() != tensor_size) {
            geminifs_error("geminifs_batched_xfer: V-cache tensor size mismatch at index %zu. Expected %lu, got %lu.\n",
                           i, tensor_size, v_caches[i].numel() * v_caches[i].element_size());
            return false;
        }

        // Check GPU file size consistency
        GPUFileDesc current_file_desc;
        if (!gpu_file_manager_->getGPUFileById(gpu_file_ids[i], current_file_desc)) {
            geminifs_error("geminifs_batched_xfer: Failed to get description for GPU file ID %u at index %zu\n", gpu_file_ids[i], i);
            return false;
        }
        if (current_file_desc.total_file_size != total_file_size) {
            geminifs_error("geminifs_batched_xfer: GPU file size mismatch at index %zu. Expected %zu, got %zu.\n",
                           i, total_file_size, current_file_desc.total_file_size);
            return false;
        }
    }


    std::vector<GPUIoContext> ioctxs;
    auto fill_ctx = [&gpu_file_ids, &gpu_controller, this]
                    (const std::vector<torch::Tensor>& cache, off_t offset) -> std::vector<GPUIoContext> {
        std::vector<GPUIoContext> out;
        for (size_t i = 0; i < cache.size(); i++) {
            NVMeFilesSpan nvme_files;
            PRPMappingEntrySpan cache_mappings;

            if (!get_nvme_files(gpu_file_ids[i], nvme_files)) {
                geminifs_error("get_nvme_files failed: file %u at index %zu\n", gpu_file_ids[i], i);
                return {};
            }
            if (!get_prp_mappings(cache[i], gpu_controller, cache_mappings)) {
                geminifs_error("get_prp_mappings failed: tensor at index %zu\n", i);
                return {};
            }
            if (nvme_files.empty() || cache_mappings.empty()) {
                geminifs_error("empty nvme_files/prp_entry: file %u at index %zu\n", gpu_file_ids[i], i);
                return {};
            }

            // once allocate space
            if (i == 0) out.reserve(cache_mappings.size() * cache.size());
            
            for (int j = 0; j < cache_mappings.size(); j++) {
                GPUIoContext ctx = {
                    .nvme_files = nvme_files,
                    .prp_entry  = &cache_mappings[j],
                    .prp_idx    = j,
                    .file_offset= offset
                };
                out.emplace_back(std::move(ctx));
            }
        }
        return out;
    };

    uint64_t len = k_caches[0].numel() * k_caches[0].element_size();

    auto k_ioctxs = fill_ctx(k_caches, layer_idx * len * 2);
    auto v_ioctxs = fill_ctx(v_caches, layer_idx * len * 2 + len);
    if (k_ioctxs.empty() || v_ioctxs.empty() || k_ioctxs.size() != v_ioctxs.size()) {
        geminifs_error("geminifs_batched_xfer: Failed to fill IO contexts\n");
        return false;
    }

    // one by one combine
    ioctxs.reserve(k_ioctxs.size() + v_ioctxs.size());
    for (size_t i = 0; i < k_ioctxs.size(); i++) {
        ioctxs.emplace_back(std::move(k_ioctxs[i]));
        ioctxs.emplace_back(std::move(v_ioctxs[i]));
    }

    auto batch_loop_size = (ioctxs.size() + MAX_IOCTX_PER_BATCH - 1) / MAX_IOCTX_PER_BATCH;
    std::vector<BatchIoEntry*> batch_entries(batch_loop_size, nullptr);

    auto release = [](std::vector<BatchIoEntry*> batches, GPUFileManager* manager, cudaStream_t stream) {
        auto data = new IoctxDataPair {batches, manager};
        auto cudaErr = cudaLaunchHostFunc(stream, release_ioctx, (void *)data);
        if (cudaErr != cudaSuccess) {
            geminifs_error("Failed to launch host func");
            return false;
        }
        return true;
    };

    for (auto &entry : batch_entries) {
        entry = gpu_file_manager_->allocateIoContexts();
        if (!entry) {
            geminifs_error("geminifs_batched_xfer: Failed to allocate IO contexts from pool\n");
            // release previously allocated
            release(batch_entries, gpu_file_manager_.get(), stream);
            return false;
        }
    }
    
    auto ioctxs_per_batch = std::min(1lu * MAX_IOCTX_PER_BATCH, ioctxs.size());

    // geminifs_info("geminifs_batched_xfer: total ioctxs %zu, batch_loop_size %zu, ioctxs_per_batch %zu\n",
    //               ioctxs.size(), batch_loop_size, ioctxs_per_batch);

    for (int i = 0; i < batch_loop_size; i++) {
        auto entry = batch_entries[i];
        auto this_batch_size = std::min(ioctxs_per_batch, ioctxs.size() - ioctxs_per_batch * i);
        
        auto cudaError = cudaMemcpyAsync(entry->d_ioctxs, ioctxs.data() + i * ioctxs_per_batch,
                                        this_batch_size * sizeof(GPUIoContext), cudaMemcpyHostToDevice, stream);
        if (cudaError != cudaSuccess) {
            geminifs_error("geminifs_batched_xfer: cudaMemcpy to d_ioctxs failed: %s\n", cudaGetErrorString(cudaError));
            release(batch_entries, gpu_file_manager_.get(), stream);
            return false;
        }

        constexpr static int THREADS_PER_BLOCK = 32;
        int blocks = (this_batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        nvme_batch_xfer_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(entry, len, this_batch_size, is_read);
        // cudaError = cudaGetLastError();
        // if (cudaError != cudaSuccess) {
        //     geminifs_error("kernel launch failed: %s", cudaGetErrorString(cudaError));
        //     release();
        //     return false;
        // }
    }
    // auto err = cudaStreamSynchronize(stream);
    // if (err != cudaSuccess) {
    //     geminifs_error("geminifs_batched_xfer: nvme_batch_xfer_kernel failed: %s\n", cudaGetErrorString(err));
    //     release();
    //     return false;
    // }

    release(batch_entries, gpu_file_manager_.get(), stream);
    return true;
}

__forceinline__
__host__ bool GeminiFS::get_nvme_files(GPUFileId gpu_file_id, NVMeFilesSpan &out_nvme_files) {
    if (!gpu_file_manager_) {
        geminifs_error("get_nvme_files: GPUFileManager not initialized\n");
        return false;
    }

    if (!gpu_file_manager_->getNVMeFilesSpanById(gpu_file_id, out_nvme_files)) {
        geminifs_error("get_nvme_files: GPU File %u not opened\n", gpu_file_id);
        return false;
    }
    return true;
}

__forceinline__
__host__ bool GeminiFS::get_prp_mappings(const torch::Tensor& tensor, GPUControllerPtr gpu_controller, PRPMappingEntrySpan &out_mappings) {
    if (!gpu_controller || !gpu_controller->isInitialized()) {
         geminifs_error("get_prp_mappings: GPU controller is not initialized\n");
          return false;
    }

    if (!tensor.is_cuda()) { 
        geminifs_error("get_prp_mappings: tensor must be on CUDA device\n");
         return false;
    }

    if (!gpu_file_manager_) {
        geminifs_error("get_prp_mappings: GPUFileManager not initialized\n");
        return false;
    }

    // Do not auto-register here; only validate tensor already registered
    auto dma_ctx = gpu_controller->getDMAContext(tensor.data_ptr());
    if (!dma_ctx) {
        geminifs_error("get_prp_mappings: tensor not registered; call geminifs_register_tensor_with_gpu before IO\n");
        return false;
    }

    uint64_t tensor_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());
    // MAX COUNT ?
    auto mapping_info = gpu_controller->getMemoryMapper()->lookupMappings(tensor_ptr);
    if (!mapping_info) {
        geminifs_error("get_prp_mappings: No valid PRP mapping found for tensor pointer %p\n", tensor.data_ptr());
        return false;
    }

    out_mappings = mapping_info->mappings;
    return true;
}

// one tensor xfer
__forceinline__ __host__ bool 
GeminiFS::geminifs_xfer_kernel(const torch::Tensor& tensor, GPUFileId gpu_file_id, loff_t off, GPUControllerPtr gpu_controller, bool is_read, cudaStream_t stream) {
    NVMeFilesSpan nvme_files;
    PRPMappingEntrySpan prp_mappings;
    if (!get_nvme_files(gpu_file_id, nvme_files)) {
        return false;
    }

    if (!get_prp_mappings(tensor, gpu_controller, prp_mappings)) {
        return false;
    }

    if (nvme_files.empty() || prp_mappings.empty()) {
        geminifs_error("geminifs_xfer_kernel: No NVMe files or PRP mappings found for GPU file %u\n", gpu_file_id);
        return false;
    }

    auto len = tensor.numel() * tensor.element_size();
    const uint32_t THREADS_PER_BLOCK = 32;
    uint32_t blocks = (prp_mappings.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    nvme_xfer_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(nvme_files, prp_mappings, len, off, is_read);
    // auto err = cudaStreamSynchronize(stream);
    // if (err != cudaSuccess) {
    //     geminifs_error("GPU_read_kernel: GPU_Read_kernel_multi failed: %s\n", cudaGetErrorString(err));
    //     return false;
    // }
    return true;
}


__forceinline__ __host__ bool 
GeminiFS::geminifs_kv_xfer_kernel(const torch::Tensor& k_cache, 
                                  const torch::Tensor& v_cache, 
                                  GPUFileId gpu_file_id, loff_t off, 
                                  GPUControllerPtr gpu_controller, 
                                  bool is_read, cudaStream_t stream) {
    NVMeFilesSpan nvme_files;
    PRPMappingEntrySpan k_prp_mappings, v_prp_mappings; // for kv
       
    if (!get_nvme_files(gpu_file_id, nvme_files)) {
        return false;
    }
    
    if (!get_prp_mappings(k_cache, gpu_controller, k_prp_mappings)) {
        return false;
    }

    if (!get_prp_mappings(v_cache, gpu_controller, v_prp_mappings)) {
        return false;
    }

    if (nvme_files.empty() || k_prp_mappings.empty() || v_prp_mappings.empty()) {
        geminifs_error("geminifs_kv_xfer_kernel: No NVMe files or PRP mappings found for GPU file %u\n", gpu_file_id);
        return false;
    }

    auto len = k_cache.numel() * k_cache.element_size();

    const uint32_t THREADS_PER_BLOCK = 32;
    uint32_t blocks = (k_prp_mappings.size() + v_prp_mappings.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    nvme_kv_xfer_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(nvme_files, k_prp_mappings, v_prp_mappings, len, off, is_read);
    // auto err = cudaStreamSynchronize(stream);
    // if (err != cudaSuccess) {
    //     geminifs_error("GPU_read_kernel: GPU_Read_kernel_multi failed: %s\n", cudaGetErrorString(err));
    //     return false;
    // }
    return true;
}


__host__ bool GeminiFS::geminifs_GPU_read_kernel(torch::Tensor& tensor, GPUFileId gpu_file_id, loff_t off, GPUControllerPtr gpu_controller, cudaStream_t stream) {
    return geminifs_xfer_kernel(tensor, gpu_file_id, off, gpu_controller, true, stream);
}


__host__ bool GeminiFS::geminifs_GPU_write_kernel(const torch::Tensor& tensor, GPUFileId gpu_file_id, loff_t offset, GPUControllerPtr gpu_controller, cudaStream_t stream) {
    return geminifs_xfer_kernel(tensor, gpu_file_id, offset, gpu_controller, false, stream);
}


/**
 * Cleanup all GPU controllers
 */
__host__ void GeminiFS::geminifs_cleanup_all_gpu_controllers() {
    auto& registry = GPUControllerRegistry::getInstance();
    registry.clearAll();
    geminifs_debug("geminifs_cleanup_all_gpu_controllers: Cleaned up all GPU controllers\n");
}


/**
 * Clean all files managed by a specific NVMe controller
 */
__host__ bool GeminiFS::geminifs_nvme_delete_all_files(int device_id, size_t controller_index) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_nvme_delete_all_files: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    auto nvme_controller = gpu_controller->getNVMeController(controller_index);
    if (!nvme_controller) {
        geminifs_error("geminifs_nvme_delete_all_files: No NVMe controller found at index %zu for device %d\n", 
                       controller_index, device_id);
        return false;
    }
    
    bool success = nvme_controller->device_file_delete_all_files_managed();
    if (success) {
        geminifs_debug("geminifs_nvme_delete_all_files: Successfully cleaned all files for device %d controller %zu\n", 
                       device_id, controller_index);
    } else {
        geminifs_error("geminifs_nvme_delete_all_files: Failed to clean all files for device %d controller %zu\n", 
                       device_id, controller_index);
    }
    
    return success;
}

__host__ bool GeminiFS::parse_and_setup_controllers(const std::string& config_file_path, int current_gpu_id) {
    ParsedSystemConfig config = parse_system_config(config_file_path);
    if (config.valid) {
        // 查找当前GPU对应的config group
        bool found_group = false;
        for (const auto& group : config.groups) {
            if (group.gpu.cudaDevice == current_gpu_id) {
                found_group = true;
                nvme_params_ = convert_to_nvme_ctrl_params_group(group);
                // 检查文件总数是否超过限制
                
                if(nvme_params_.size()*init_GPU_num_files_ >NUM_FILES)
                {
                    geminifs_error("geminifs_init: The total number of files (%zu) exceeds the limit (%d). Please increase NUM_FILES in geminifs.h and recompile.\n", nvme_params_.size()*init_GPU_num_files_, NUM_FILES);
                    return false;
                }
                geminifs_debug("geminifs_init: Creating GPU controller for device %d with mount path %s\n", group.gpu.cudaDevice, group.gpu.mount_path.c_str());
                geminifs_create_gpu_controller(group.gpu.cudaDevice, group.gpu.mount_path);
                for (const auto& nvme_ctrl_param : nvme_params_) {
                    if (!nvme_ctrl_param.pci_addr.empty()) {
                        geminifs_debug("geminifs_init: Adding NVMe controller for device %d with mount path %s\n", group.gpu.cudaDevice, nvme_ctrl_param.mount_path.c_str());
                        geminifs_add_nvme_to_gpu(group.gpu.cudaDevice, nvme_ctrl_param);
                    }
                }
                break; // 找到对应的group后退出循环
            }
        }
        
        if (!found_group) {
            geminifs_error("geminifs_init: No config group found for current GPU device %d\n", current_gpu_id);
            return false;
        }
    } else {
        std::cerr << "Config parsing failed: " << config.error_message << std::endl;
        return false;
    }
    return true;
}

__host__ void GeminiFS::handle_gpu_file_reset(int current_gpu_id) {
    // 询问用户确认
    std::string operation_desc = "Reset GeminiFS for GPU device " + std::to_string(current_gpu_id) + 
                                "\n  - Clear all managed GPU files" +
                                "\n  - Clear all managed NVMe files" +
                                "\n  - Clear memory caches and links cache";
    
    if (!confirm_dangerous_operation(operation_desc)) {
        geminifs_debug("geminifs_init: Reset operation cancelled by user, continuing with normal initialization\n");
        // 用户取消了 reset，但继续正常初始化
    } else {
        auto existing_gpu_controller = geminifs_get_gpu_controller(current_gpu_id);
        if (existing_gpu_controller && gpu_file_manager_) {
            geminifs_debug("geminifs_init: Reset mode - clearing all GPU files and NVMe files for GPU device %d\n", current_gpu_id);
            
            // 第一步：删除所有GPU文件（这会自动清理对应的NVMe文件映射）
            auto gpu_file_ids = gpu_file_manager_->getAllGPUFileIds();
            geminifs_info("geminifs_init: Found %zu GPU files to delete\n", gpu_file_ids.size());
            
            for (auto gpu_file_id : gpu_file_ids) {
                geminifs_debug("geminifs_init: Deleting GPU file ID %u\n", gpu_file_id);
                bool success = gpu_file_manager_->deleteGPUFile(gpu_file_id);
                if (success) {
                    geminifs_debug("geminifs_init: Successfully deleted GPU file ID %u and its NVMe files\n", gpu_file_id);
                } else {
                    geminifs_error("geminifs_init: Failed to delete GPU file ID %u\n", gpu_file_id);
                }
            }
            
            // 第二步：清空所有NVMe控制器中的剩余文件（确保彻底清理）
            size_t nvme_count = existing_gpu_controller->getControllerCount();
            geminifs_debug("geminifs_init: Cleaning remaining files from %zu NVMe controllers\n", nvme_count);
            
            for (size_t i = 0; i < nvme_count; ++i) {
                auto nvme_controller = existing_gpu_controller->getNVMeController(i);
                if (nvme_controller) {
                    geminifs_debug("geminifs_init: Cleaning remaining files for NVMe controller %zu\n", i);
                    bool success = nvme_controller->device_file_delete_all_files_managed();
                    if (success) {
                        geminifs_info("geminifs_init: Successfully cleaned remaining files for NVMe controller %zu\n", i);
                    } else {
                        geminifs_error("geminifs_init: Failed to clean remaining files for NVMe controller %zu\n", i);
                    }
                }
            }
            
            // 第三步：强制持久化GPUFileManager的更改
            gpu_file_manager_->forcePersist();
                        
            geminifs_debug("geminifs_init: Reset completed - all GPU files, NVMe files and caches cleared\n");
        } else {
            if (!existing_gpu_controller) {
                geminifs_debug("geminifs_init: No existing GPU controller found for reset on device %d\n", current_gpu_id);
            }
            if (!gpu_file_manager_) {
                geminifs_debug("geminifs_init: GPUFileManager not initialized yet for reset on device %d\n", current_gpu_id);
            }
        }
    }
}

__host__ void GeminiFS::init(const std::string& config_file_path, int GPU_file_nums,const std::vector<size_t>& GPU_file_shape, bool reset) {
    // 检查GPU_file_shape必须是三维的
    if (GPU_file_shape.size() != 3) {
        geminifs_error("GeminiFS::init: GPU_file_shape must be 3-dimensional, got %zu dimensions\n", GPU_file_shape.size());
        return;
    }
    size_t num_files = GPU_file_nums;
    // 根据GPU_file_shape计算参数
    // GPU_file_shape[0] * GPU_file_shape[1] = GPUfile的数量 (例如: 2 * 32 = 64)
    // GPU_file_shape[2] = block size大小 (例如: 524288)
    size_t file_size = GPU_file_shape[0] * GPU_file_shape[1] * GPU_file_shape[2];
    
    geminifs_info("GeminiFS::init: GPU_file_shape=[%zu, %zu, %zu] -> num_files=%zu, file_size=%zu\n", 
                   GPU_file_shape[0], GPU_file_shape[1], GPU_file_shape[2], num_files, file_size);
    
    // 记录初始化GPU file 参数
    init_GPU_num_files_ = num_files;
    init_GPU_file_size_ = file_size;
    // size_t per_nvme_controller_files = 0;
    // size_t per_nvme_file_size = 0;
    geminifs_info("GeminiFS::init: config_file_path=%s, num_files=%zu, file_size=%zu, reset=%s\n", 
                   config_file_path.c_str(), num_files, file_size, reset ? "true" : "false");
    
    auto& registry = GPUControllerRegistry::getInstance();
    
    // 确认进程可以打开足够的文件描述符
    auto_configure_fd_limits(NUM_FILES);
    
    // 获取当前进程所在的GPU ID
    int current_gpu_id;
    cudaError_t err = cudaGetDevice(&current_gpu_id);
    if (err != cudaSuccess) {
        geminifs_error("geminifs_init: Failed to get current GPU device: %s\n", cudaGetErrorString(err));
        return;
    }

    if (is_init_ && !reset) {
        geminifs_debug("geminifs_init: GeminiFS is already initialized, skipping re-initialization\n");
        return;
    }

    if (!parse_and_setup_controllers(config_file_path, current_gpu_id)) {
        return;
    }

    // 初始化GPUFileManager，log文件放在GPU controller的mount_path下
    auto gpu_controller = geminifs_get_gpu_controller(current_gpu_id);
    if (!gpu_controller) {
        geminifs_error("GeminiFS::init: Failed to get GPU controller for device %d\n", current_gpu_id);
        return;
    }
    
    std::filesystem::path gpu_mount_path = gpu_controller->getMountBasePath();
    std::filesystem::path gpu_file_manager_log_path = gpu_mount_path / "gpu_file_manager.log";
    
    geminifs_debug("GeminiFS::init: Initializing GPUFileManager with log path: %s\n", 
                   gpu_file_manager_log_path.c_str());
    
    gpu_file_manager_ = std::make_unique<GPUFileManager>(gpu_file_manager_log_path.string(), gpu_controller);
    
    if (!gpu_file_manager_) {
        geminifs_error("GeminiFS::init: Failed to initialize GPUFileManager\n");
        return;
    }

    // 如果需要reset，清空已存在的GPU文件和对应的NVMe controller文件
    if (reset) {
        handle_gpu_file_reset(current_gpu_id);
    }


    // 检查已有的GPU文件数量，决定是否需要创建新文件
    auto existing_gpu_file_ids = gpu_file_manager_->getAllGPUFileIds();
    size_t existing_file_count = existing_gpu_file_ids.size();
    
    geminifs_info("GeminiFS::init: Found %zu existing GPU files, need %zu total files\n", 
                   existing_file_count, num_files);
    
    // 打印GPU file ID的范围信息
    if (existing_file_count > 0) {
        GPUFileId min_gpu_file_id = *std::min_element(existing_gpu_file_ids.begin(), existing_gpu_file_ids.end());
        GPUFileId max_gpu_file_id = *std::max_element(existing_gpu_file_ids.begin(), existing_gpu_file_ids.end());
        geminifs_info("GeminiFS::init: GPU file ID range: min=%u, max=%u\n", min_gpu_file_id, max_gpu_file_id);
    } else {
        geminifs_info("GeminiFS::init: No existing GPU files found\n");
    }
    
    // 验证现有文件的配置是否匹配
    bool config_mismatch = false;
    if (existing_file_count > 0) {
        // 检查第一个文件的配置作为样本
        GPUFileDesc sample_desc;
        if (gpu_file_manager_->getGPUFileById(existing_gpu_file_ids[0], sample_desc)) {
            if (sample_desc.total_file_size != file_size ||
                sample_desc.tensor_shape[0] != GPU_file_shape[0] ||
                sample_desc.tensor_shape[1] != GPU_file_shape[1] ||
                sample_desc.tensor_shape[2] != GPU_file_shape[2]) {
                
                geminifs_warn("GeminiFS::init: Existing GPU file configuration mismatch detected:\n");
                geminifs_warn("  Existing: size=%zu, shape=[%u, %u, %u]\n", 
                             sample_desc.total_file_size, 
                             sample_desc.tensor_shape[0], sample_desc.tensor_shape[1], sample_desc.tensor_shape[2]);
                geminifs_warn("  Requested: size=%zu, shape=[%zu, %zu, %zu]\n", 
                             file_size, GPU_file_shape[0], GPU_file_shape[1], GPU_file_shape[2]);
                geminifs_warn("  Consider using reset=true to clear existing files with different configuration\n");
                config_mismatch = true;
            }
        }
    }
    
    if (existing_file_count >= num_files && !config_mismatch) {
        geminifs_info("GeminiFS::init: Sufficient GPU files already exist (%zu >= %zu) with matching configuration, skipping file creation\n", 
                      existing_file_count, num_files);
    } else {
        if (config_mismatch) {
            geminifs_warn("GeminiFS::init: Will create new files despite existing files due to configuration mismatch\n");
        }
        
        size_t files_to_create = (existing_file_count < num_files) ? (num_files - existing_file_count) : num_files;
        geminifs_info("GeminiFS::init: Creating %zu additional GPU files of size %zu bytes each\n", 
                      files_to_create, file_size);
        
        std::vector<GPUFileId> new_gpu_file_ids;
        if (!gpu_file_manager_->createGPUFiles(file_size, files_to_create,
                                               {GPU_file_shape[0], 
                                                              GPU_file_shape[1], 
                                                              GPU_file_shape[2]}, 
                                               new_gpu_file_ids)) {
            geminifs_error("GeminiFS::init: Failed to create new GPU files\n");
            return;
        }
        geminifs_info("GeminiFS::init: Successfully created %zu new GPU files, total files now: %zu\n", 
                      files_to_create, existing_file_count + files_to_create);
    }

    for (uint32_t i = 0; i < num_files; ++i) {
        if (!gpu_file_manager_->initGPUFile(i)) {
            geminifs_error("GeminiFS::init: Failed to init GPU file %u\n", i);
            return;
        }

        geminifs_debug("GeminiFS::init: Successfully initialized GPU file %u\n", i);
    }




    // // 设置每个nvme controller管理的文件的大小 
    // per_nvme_controller_files = num_files;
    // per_nvme_file_size = (nvme_params_.empty()) ? 0 : file_size / nvme_params_.size();
    // // 判断per_nvme_file_size是否大于64KB 且 64KB对齐
    // if (per_nvme_file_size < __64KB__ || (per_nvme_file_size % __64KB__) != 0) {
    //     geminifs_error("geminifs_init: Each NVMe controller must manage files of at least 64KB and aligned to 64KB. Current per controller file size: %zu bytes\n", per_nvme_file_size);
    //     return;
    // }

    // // 构建GPU file 创建相应 数量的GPU 检查GPU file manager 中是否有对应数量的GPUfileID 没有则创建 并建立 link
    // for (size_t i = 0; i < per_nvme_controller_files; ++i) {
    //     GPUFileId gpu_file_id;
    //     if (!gpu_file_manager_.getGPUFileIdByIndex(i, gpu_file_id)) {
    //         // 不存在则创建
    //         if (!gpu_file_manager_.createGPUFile(per_nvme_file_size, gpu_file_id)) {
    //             geminifs_error("geminifs_init: Failed to create GPU file %zu of size %zu bytes\n", i, per_nvme_file_size);
    //             return;
    //         }
    //         geminifs_debug("geminifs_init: Created GPU file %zu with ID %lu of size %zu bytes\n", i, gpu_file_id, per_nvme_file_size);
    //     } else {
    //         geminifs_debug("geminifs_init: GPU file %zu already exists with ID %lu\n", i, gpu_file_id);
    //     }











     /***************************************************/
     //  NVMEe file 初始化以及检查
     /***************************************************/
    // // 检查并创建每个NVMe控制器中的文件
    // auto gpu_controller = geminifs_get_gpu_controller(current_gpu_id);
    // if (gpu_controller) {
    //     size_t nvme_count = gpu_controller->getControllerCount();
    //     geminifs_debug("geminifs_init: Checking files for %zu NVMe controllers\n", nvme_count);
        
    //     for (size_t i = 0; i < nvme_count; ++i) {
    //         auto nvme_controller = gpu_controller->getNVMeController(i);
    //         if (nvme_controller) {
    //             geminifs_debug("geminifs_init: Checking files for NVMe controller %zu\n", i);
                
    //             // 检查符合预期大小的有效文件数量
    //             size_t expected_file_size = per_nvme_file_size;
    //             size_t valid_file_count = nvme_controller->device_file_validate_sizes(expected_file_size);
                
    //             geminifs_info("geminifs_init: NVMe controller %zu has %zu valid files of size %zu bytes, expected %zu files\n", 
    //                           i, valid_file_count, expected_file_size, per_nvme_controller_files);
                
    //             // 如果有效文件数量不够，创建缺失的文件
    //             if (valid_file_count < per_nvme_controller_files) {
    //                 size_t files_to_create = per_nvme_controller_files - valid_file_count;
    //                 geminifs_info("geminifs_init: Need to create %zu additional files for NVMe controller %zu\n", 
    //                               files_to_create, i);
                    
    //                 // 获取现有文件总数用于命名新文件
    //                 size_t existing_file_count = nvme_controller->device_file_get_managed_file_count();
                    
    //                 for (size_t j = 0; j < files_to_create; ++j) {
    //                     std::string filename = std::to_string(existing_file_count + j) + ".KV";
                        
    //                     // Use host_file_create_only_managed to create the file without opening it
    //                     bool success = nvme_controller->host_file_create_only_managed(nvme_controller->controller->page_size, expected_file_size, filename);
                        
    //                     if (success) {
    //                         geminifs_debug("geminifs_init: Successfully created file '%s' of size %zu bytes on NVMe controller %zu\n", 
    //                                       filename.c_str(), expected_file_size, i);
    //                     } else {
    //                         geminifs_error("geminifs_init: Failed to create file '%s' on NVMe controller %zu\n", 
    //                                       filename.c_str(), i);
    //                         return;
    //                     }
    //                 }
    //                 // 重新验证文件数量
    //                 size_t final_valid_count = nvme_controller->device_file_validate_sizes(expected_file_size);
    //                 if (final_valid_count < per_nvme_controller_files) {
    //                     geminifs_error("geminifs_init: After file creation, still have insufficient valid files (%zu) for NVMe controller %zu\n", 
    //                                   final_valid_count, i);
    //                     return;
    //                 }
    //                 geminifs_info("geminifs_init: File creation successful, now have %zu valid files for NVMe controller %zu\n", 
    //                               final_valid_count, i);
    //             } else if (valid_file_count > per_nvme_controller_files) {
    //                 geminifs_warn("geminifs_init: NVMe controller %zu has more valid files (%zu) than expected (%zu)\n", 
    //                              i, valid_file_count, per_nvme_controller_files);
    //             }
                
    //             geminifs_debug("geminifs_init: NVMe controller %zu file setup completed successfully\n", i);
    //         } else {
    //             geminifs_error("geminifs_init: Failed to get NVMe controller %zu\n", i);
    //             return;
    //         }
    //     }
        
    //     geminifs_debug("geminifs_init: All NVMe controllers file setup completed successfully\n");
    // } else {
    //     geminifs_error("geminifs_init: Failed to get GPU controller for device %d\n", current_gpu_id);
    //     return;
    // }
    // }
    is_init_ = true;
    
    geminifs_debug("geminifs_init: Successfully initialized GeminiFS for GPU device %d (reset=%s)\n", 
                   current_gpu_id, reset ? "true" : "false");
}

__host__ void GeminiFS::cleanup() {
    // 释放GPUFileManager
    if (gpu_file_manager_) {
        geminifs_debug("GeminiFS::cleanup: Releasing GPUFileManager\n");
        gpu_file_manager_.reset();
    }
    
    // 清理所有GPU controllers
    geminifs_cleanup_all_gpu_controllers();
}
