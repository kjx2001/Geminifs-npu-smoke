#include "geminifs_helper.h"
#include "geminifs_mem.h"
#include "buffer.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cassert>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <sys/types.h>
#include <unordered_set>
#include <vector>
#include "gpu_controller.cuh"
#include "nvme_controller.cuh"
#include "prp_mapping_entry.h"



bool GPUMemoryMapper::initialize() {
    std::lock_guard<std::mutex> lock(mapper_mutex_);

    if (initialized_) {
        return true;
    }
    initialized_ = true;

    geminifs_debug("GPU Memory Mapper: Successfully initialized for device %d\n", device_id_);
    return true;
}

void GPUMemoryMapper::cleanup() {
    std::lock_guard<std::mutex> lock(mapper_mutex_);

    if (!initialized_) {
        return;
    }
    initialized_ = false;

    geminifs_debug("GPU Memory Mapper: Cleaned up resources for device %d\n", device_id_);
}


bool GPUMemoryMapper::addBatchMappings(uint64_t tensor_ptr, uint64_t tensor_size, 
									   const std::vector<PRPMappingEntry>& mappings, 
									   void *gpu_buffer, size_t gpu_buffer_size) {
    if (!initialized_) {
        geminifs_error("GPU Memory Mapper: Not initialized on device %d\n", device_id_);
        return false;
    }

    if (mappings.empty()){
        geminifs_warn("GPU Memory Mapper: Empty mappings provided\n");
        return true;
    }

    auto createAndCopy = [&]() -> GPUHashEntry* {
        PRPMappingEntry *d_mappings = nullptr;
		bool allocated = false;
		if (gpu_buffer) {
			if (gpu_buffer_size < mappings.size() * sizeof(PRPMappingEntry)) {
				geminifs_error("GPU Memory Mapper: gpu buffer is not enough for mapping\n");
				return nullptr;
			}
			d_mappings = static_cast<PRPMappingEntry*>(gpu_buffer);
		} else {
			if (cudaMalloc(&d_mappings, mappings.size() * sizeof(PRPMappingEntry)) != cudaSuccess) {
				geminifs_error("GPU Memory Mapper: cudaMalloc failed for mappings on device %d\n", device_id_);
				return nullptr;
			}
			allocated = true;
		}

		auto err = cudaMemcpy(d_mappings, mappings.data(), mappings.size() * sizeof(PRPMappingEntry), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            geminifs_error("GPU Memory Mapper: cudaMemcpy failed for mappings on device %d, mapping size %ld, error %d\n", device_id_, mappings.size(), err);
            if (allocated) cudaFree(d_mappings);
            return nullptr;
        }
        GPUHashEntry *entry = (GPUHashEntry*)std::malloc(sizeof(GPUHashEntry));
        *entry = {
            .dev_ptr = tensor_ptr,
            .tensor_size = tensor_size,
            .mappings = cuda::std::span<PRPMappingEntry>(d_mappings, mappings.size()),
            .next = nullptr
        };
        return entry;
    };

    std::lock_guard lock{mapper_mutex_};
    auto new_entry = createAndCopy();
    if (!new_entry) {
        return false;
    }

    // debugPrintMappings(new_entry, true);

    auto iter = host_mappings_.find(tensor_ptr);
    if (iter != host_mappings_.end()) { // 正常情况不会存在重复mapping
        iter->second->next = new_entry;
        // geminifs_info("GPU Memory Mapper: Added additional mappings for tensor 0x%lx on device %d\n", tensor_ptr, device_id_);
        return true;
    }
    host_mappings_[tensor_ptr] = new_entry;
    // geminifs_info("GPU Memory Mapper: Added new mappings for tensor 0x%lx on device %d\n", tensor_ptr, device_id_);    
    return true;
}

bool GPUMemoryMapper::removeAllMappings(uint64_t tensor_ptr) {
    if (!initialized_) {
        geminifs_error("GPU Memory Mapper: Not initialized on device %d\n", device_id_);
        return false;
    }

    std::lock_guard lock{mapper_mutex_};
    auto iter = host_mappings_.find(tensor_ptr);
    if (iter == host_mappings_.end()) {
        geminifs_warn("GPU Memory Mapper: No mappings found for tensor 0x%lx on device %d\n", tensor_ptr, device_id_);
        return true;
    }

    // 释放所有相关的GPU内存
    GPUHashEntry* entry = iter->second;
    host_mappings_.erase(iter);
    while (entry) {
        if (entry->mappings.data()) {
            cudaFree(entry->mappings.data());
        }
        GPUHashEntry* to_delete = entry;
        entry = entry->next;
        std::free(to_delete);
    }
    // geminifs_info("GPU Memory Mapper: Removed all mappings for tensor 0x%lx on device %d\n", tensor_ptr, device_id_);
    return true;
}

GPUMemoryMapper::GPUHashEntry* 
GPUMemoryMapper::lookupMappings(uint64_t tensor_ptr) const {
    if (!initialized_) {
        geminifs_error("GPU Memory Mapper: Not initialized on device %d\n", device_id_);
        return nullptr;
    }

    std::lock_guard lock{mapper_mutex_};
    auto iter = host_mappings_.find(tensor_ptr);
    if (iter == host_mappings_.end()) {
        geminifs_debug("GPU Memory Mapper: No mappings found for tensor 0x%lx on device %d\n", tensor_ptr, device_id_);
        return nullptr;
    }
    // debugPrintMappings(iter->second, true);

    // 直接返回链表头指针，调用者负责遍历
    return iter->second;
}

void GPUMemoryMapper::debugPrintMappings(const GPUHashEntry * entry, bool inDevice) const {
#ifdef DEBUG
    if (entry == nullptr) {
        geminifs_info("GPU Memory Mapper: No mappings to print on device %d\n", device_id_);
        return;
    }

    PRPMappingEntry *host_mappings = (PRPMappingEntry*)std::malloc(entry->mappings.size() * sizeof(PRPMappingEntry));
    if (!host_mappings) {
        geminifs_error("GPU Memory Mapper: malloc failed in debugPrintMappings on device %d\n", device_id_);
        return;
    }
    auto err = cudaMemcpy(host_mappings, entry->mappings.data(), entry->mappings.size() * sizeof(PRPMappingEntry), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: cudaMemcpy failed in debugPrintMappings on device %d\n", device_id_);
        std::free(host_mappings);
        return;
    }
    geminifs_info("GPU Memory Mapper: Mappings for tensor 0x%lx (size %lu) on device %d:\n", entry->dev_ptr, entry->tensor_size, device_id_);
    for (size_t i = 0; i < entry->mappings.size(); ++i) {
        const auto& mapping = host_mappings[i];
        geminifs_info("  Mapping %zu: prp1=0x%lx, prp2=0x%lx, data_length=%u\n", 
                    i, mapping.prp1, mapping.prp2, mapping.data_length);
    }
    std::free(host_mappings);
#endif
}


GPUMemoryMapper::~GPUMemoryMapper() {
    cleanup();
}


// === GPUController Implementation ===

GPUController::GPUController(int device_id, const std::string &mount_base_path)
    : device_id_(device_id), mount_base_path_(mount_base_path), is_initialized_(false)
{

    geminifs_debug("GPU Controller: Initializing for device %d with mount path '%s'\n",
                   device_id, mount_base_path.c_str());

    // Set the CUDA device context
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Controller: Failed to set CUDA device %d: %s\n",
                       device_id_, cudaGetErrorString(err));
        return;
    }

    // Initialize the controller
    if (!initialize())
    {
        geminifs_error("GPU Controller: Failed to initialize for device %d\n", device_id_);
        return;
    }

    is_initialized_.store(true);
    geminifs_debug("GPU Controller: Successfully initialized for device %d\n", device_id_);
}

GPUController::~GPUController()
{
    cleanup();
}

bool GPUController::initialize()
{
    // Create base mount directory if it doesn't exist
    std::filesystem::create_directories(mount_base_path_);

    // Initialize containers
    dma_contexts_.clear();
    nvme_controllers_.clear();

    // Initialize memory mapper
    memory_mapper_ = std::make_unique<GPUMemoryMapper>(device_id_);
    if (!memory_mapper_->initialize())
    {
        geminifs_error("GPU Controller: Failed to initialize memory mapper for device %d\n", device_id_);
        return false;
    }

    geminifs_debug("GPU Controller: Successfully initialized memory mapper for device %d\n", device_id_);

    return true;
}

void GPUController::cleanup()
{
    geminifs_debug("GPU Controller: Cleaning up device %d\n", device_id_);

    // Cleanup memory mapper first (before clearing DMA contexts that might use it)
    if (memory_mapper_)
    {
        geminifs_debug("GPU Controller: Cleaning up memory mapper\n");
        memory_mapper_->cleanup();
        memory_mapper_.reset();
    }

    // Clear all DMA contexts
    clearAllDMAContexts();

    // Reset and clear all NVMe controllers
    {
        std::lock_guard<std::mutex> lock(storage_mutex_);

        // Reset each NVMe controller before clearing
        for (auto &nvme_controller : nvme_controllers_)
        {
            if (nvme_controller)
            {
                geminifs_debug("GPU Controller: Resetting NVMe controller\n");
                nvme_controller.reset();
            }
        }

        nvme_controllers_.clear();
    }

    is_initialized_.store(false);
}

// === Memory Management Methods ===

bool GPUController::registerTesnsorList(const std::vector<torch::Tensor> &tensor_list, uint64_t granularity) {
    if (!isInitialized()) {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }

    for(const auto &tensor : tensor_list) {
        if (!validateTensor(tensor)) return false;
    }

    if (granularity > 0 && granularity % 4096 != 0) {
        geminifs_error("GPU Controller: Granularity %lu is not 4K aligned. Memory registration failed.\n", granularity);
        return false;
    }

	{
		std::lock_guard<std::mutex> lock(memory_mutex_);
		std::vector<torch::Tensor> to_register;
		for (const auto &tensor : tensor_list) {
			uint64_t tensor_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());
			// Check if tensor is already registered
			if (dma_contexts_.find(tensor_ptr) != dma_contexts_.end()) {
				geminifs_warn("GPU Controller: Tensor at %p is already registered\n", tensor.data_ptr());
				continue;
			}
			to_register.push_back(tensor);
		}

		auto dma_ctx_list = createDMAContexts(to_register, granularity);
		if (dma_ctx_list.empty() || dma_ctx_list.size() != to_register.size()) {
			geminifs_error("GPU Controller: Failed to create DMA contexts for tensor list\n");
			return false;
		}

		for (size_t i = 0; i < to_register.size(); ++i) {
			const auto &tensor = to_register[i];
			uint64_t tensor_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());
			auto tensor_size = tensor.numel() * tensor.element_size();
			if (granularity > 0) {
				// Add multiple pointers according to granularity, all pointing to the same dma_ctx
				size_t num_granules = (tensor_size + granularity - 1) / granularity; // Round up

				geminifs_debug("GPU Controller: Registering tensor at %p with size %lu using granularity %lu (%zu granules) on device %d\n",
							  tensor.data_ptr(), tensor_size, granularity, num_granules, device_id_);

				for (size_t j = 0; j < num_granules; ++j) {
					uint64_t granule_ptr = tensor_ptr + (j * granularity);
					dma_contexts_[granule_ptr] = dma_ctx_list[i];
				}
			} else {
				// Traditional single mapping for the entire tensor
				dma_contexts_[tensor_ptr] = dma_ctx_list[i];
			}
		}

		return true;
	}

	geminifs_info("GPU Controller: Successfully registered tensor list for device %d\n", device_id_);
    return true;
}



bool GPUController::registerTensorMemory(const torch::Tensor &tensor, uint64_t granularity)
{
    if (!isInitialized())
    {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }

    if (!validateTensor(tensor))
    {
        return false;
    }

    uint64_t tensor_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());
    auto tensor_size = tensor.numel() * tensor.element_size();

    // Check 4K alignment for granularity (if specified)
    if (granularity > 0 && granularity % 4096 != 0)
    {
        geminifs_error("GPU Controller: Granularity %lu is not 4K aligned. Memory registration failed.\n", granularity);
        return false;
    }
    
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);

        // Check if tensor is already registered
        if (dma_contexts_.find(tensor_ptr) != dma_contexts_.end())
        {
            geminifs_warn("GPU Controller: Tensor at %p is already registered\n", tensor.data_ptr());
            return true;
        }

        // Create DMA context
        geminifs_dma *dma_ctx = createDMAContext(tensor, granularity);
        if (dma_ctx == nullptr)
        {
            geminifs_error("GPU Controller: Failed to create DMA context for tensor at %p\n", tensor.data_ptr());
            return false;
        }

        // Add DMA context mapping based on granularity
        if (granularity > 0)
        {
            // Add multiple pointers according to granularity, all pointing to the same dma_ctx
            size_t num_granules = (tensor_size + granularity - 1) / granularity; // Round up
            
            for (size_t i = 0; i < num_granules; ++i)
            {
                uint64_t granule_ptr = tensor_ptr + (i * granularity);
                dma_contexts_[granule_ptr] = dma_ctx;
            }
        }
        else
        {
            // Traditional single mapping for the entire tensor
            dma_contexts_[tensor_ptr] = dma_ctx;
        }
    }

    geminifs_debug("GPU Controller: Successfully registered tensor at %p for device %d\n",
                   tensor.data_ptr(), device_id_);
    return true;
}

bool GPUController::unregisterTensorMemory(void *tensor_ptr)
{
    if (!isInitialized())
    {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }

    uint64_t ptr_key = reinterpret_cast<uint64_t>(tensor_ptr);

    {
        std::lock_guard<std::mutex> lock(memory_mutex_);

        auto it = dma_contexts_.find(ptr_key);
        if (it == dma_contexts_.end())
        {
            geminifs_warn("GPU Controller: Tensor at %p is not registered\n", tensor_ptr);
            return false;
        }

        // Get DMA context and remove ALL entries that point to the same dma_ctx
        geminifs_dma *dma_ctx = it->second;
        
        // Remove all granule mappings for this DMA context
        auto map_it = dma_contexts_.begin();
        while (map_it != dma_contexts_.end())
        {
            if (map_it->second == dma_ctx)
            {
                map_it = dma_contexts_.erase(map_it);
            }
            else
            {
                ++map_it;
            }
        }

        // Clean up DMA context (但不释放 CUDA 内存)
        // 注意: 不调用 cudaFree(dma_ctx->ioaddrs)，因为 CUDA 内存由应用进程管理
        delete dma_ctx;
    }

    geminifs_debug("GPU Controller: Successfully unregistered tensor at %p for device %d\n",
                   tensor_ptr, device_id_);
    return true;
}

geminifs_dma *GPUController::getDMAContext(void *tensor_ptr)
{
    uint64_t ptr_key = reinterpret_cast<uint64_t>(tensor_ptr);

    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = dma_contexts_.find(ptr_key);
    return (it != dma_contexts_.end()) ? it->second : nullptr;
}

const std::unordered_map<uint64_t, geminifs_dma *> &GPUController::getAllDMAContexts() const
{
    return dma_contexts_;
}

void GPUController::clearAllDMAContexts()
{
    std::lock_guard<std::mutex> lock(memory_mutex_);

    // Collect unique DMA contexts to avoid double deletion
    std::unordered_set<geminifs_dma *> unique_contexts;
    for (const auto &pair : dma_contexts_)
    {
        if (pair.second != nullptr)
        {
            unique_contexts.insert(pair.second);
        }
    }

    // Delete each unique DMA context only once
    for (geminifs_dma *dma_ctx : unique_contexts)
    {
        // 注意: 不调用 cudaFree，因为 CUDA 内存由应用进程管理
        // PRP 上下文会在 geminifs_dma 的析构函数中自动清理
        // delete dma_ctx;
        delete dma_ctx;
    }

    dma_contexts_.clear();
    geminifs_debug("GPU Controller: Cleared all DMA contexts for device %d\n", device_id_);
}

// === Storage Management Methods ===

bool GPUController::addNVMeController(NVMeControllerPtr nvme_controller)
{
    if (!isInitialized())
    {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }

    if (!nvme_controller)
    {
        geminifs_error("GPU Controller: Invalid NVMe controller provided\n");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(storage_mutex_);
        nvme_controllers_.push_back(nvme_controller);
    }

    geminifs_debug("GPU Controller: Added NVMe controller to device %d (total: %zu)\n",
                   device_id_, nvme_controllers_.size());
    return true;
}

bool GPUController::removeNVMeController(size_t index)
{
    if (!isInitialized())
    {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(storage_mutex_);

        if (index >= nvme_controllers_.size())
        {
            geminifs_error("GPU Controller: Invalid controller index %zu (max: %zu)\n",
                           index, nvme_controllers_.size());
            return false;
        }

        nvme_controllers_.erase(nvme_controllers_.begin() + index);
    }

    geminifs_debug("GPU Controller: Removed NVMe controller at index %zu from device %d\n",
                   index, device_id_);
    return true;
}

NVMeControllerPtr GPUController::getNVMeController(size_t index)
{
    std::lock_guard<std::mutex> lock(storage_mutex_);

    if (index >= nvme_controllers_.size())
    {
        return nullptr;
    }

    return nvme_controllers_[index];
}

const std::vector<NVMeControllerPtr> &GPUController::getAllNVMeControllers() const
{
    return nvme_controllers_;
}

size_t GPUController::getControllerCount() const
{
    std::lock_guard<std::mutex> lock(storage_mutex_);
    return nvme_controllers_.size();
}

// === Private Methods ===

bool GPUController::validateTensor(const torch::Tensor &tensor) const
{
    // Check if tensor is on the correct device
    if (!tensor.is_cuda())
    {
        geminifs_error("GPU Controller: Tensor is not on CUDA device\n");
        return false;
    }

    if (tensor.device().index() != device_id_)
    {
        geminifs_error("GPU Controller: Tensor is on device %d, expected device %d\n",
                       tensor.device().index(), device_id_);
        return false;
    }

    // Check alignment
    if (!is_ptr_aligned(tensor.data_ptr()))
    {
        geminifs_error("GPU Controller: Tensor data pointer %p is not aligned to GPU_PAGE_SIZE\n",
                       tensor.data_ptr());
        return false;
    }

    auto tensor_size = tensor.numel() * tensor.element_size();
    if (!is_aligned(tensor_size))
    {
        geminifs_error("GPU Controller: Tensor size %ld is not aligned to GPU_PAGE_SIZE\n", tensor_size);
        return false;
    }

    return true;
}

bool GPUController::performDMASlicing(geminifs_dma *dma_ctx, size_t tensor_size, uint64_t granularity)
{
    // 获取所有NVMe控制器maxIOsize的最小值（用于第二级切割）
    uint64_t min_max_io_size = UINT64_MAX;
    size_t num_nvme_controllers = nvme_controllers_.size();

    for (const auto &nvme_ctrl : nvme_controllers_)
    {
        if (nvme_ctrl && nvme_ctrl->maxIOsize > 0)
        {
            min_max_io_size = std::min(min_max_io_size, nvme_ctrl->maxIOsize);
        }
    }

    // 如果没有找到有效的maxIOsize，报错并返回
    if (min_max_io_size == UINT64_MAX)
    {
        geminifs_error("GPU Controller: No valid maxIOsize found in any NVMe controller. Memory registration failed.\n");
        return false;
    }

    // 记录切片粒度信息
    dma_ctx->slice_granularity = (granularity > 0) ? granularity : min_max_io_size;

    // 清空之前的数据
    dma_ctx->granularity_groups.clear();

    // 实现两级切割逻辑
    if (granularity > 0)
    {
        // 第一级：按照外部传入的granularity进行切割，创建granularity groups
        size_t remaining_size = tensor_size;
        size_t current_offset = 0;

        while (remaining_size > 0)
        {
            size_t granularity_size = std::min(remaining_size, (size_t)granularity);

            // 计算该granularity对应的GPU tensor数据指针
            uint64_t gpu_tensor_ptr = reinterpret_cast<uint64_t>(dma_ctx->dma_ptr->vaddr) + current_offset;

            // 创建granularity组
            GranularitySliceGroup group(gpu_tensor_ptr, current_offset, granularity_size);

            // geminifs_info("GPU Controller: First-level granularity[%zu]: offset=%zu, size=%zu bytes, gpu_ptr=0x%lx\n",
            //               dma_ctx->granularity_groups.size(), current_offset, granularity_size, gpu_tensor_ptr);

            // 第二级：对当前granularity按照maxIOsize进行切割
            if (granularity_size <= min_max_io_size)
            {
                // 当前granularity小于等于maxIOsize，不需要进一步切割
                SubSliceInfo sub_slice(0, granularity_size, current_offset);
                group.sub_slices.push_back(sub_slice);

                // geminifs_info("GPU Controller: Sub-slice[0]: local_offset=0, size=%zu, global_offset=%zu (no further cutting needed)\n",
                //               granularity_size, current_offset);
            }
            else
            {
                // 当前granularity需要按照maxIOsize进一步切割
                size_t sub_remaining = granularity_size;
                size_t sub_local_offset = 0; // granularity内的本地偏移
                size_t sub_index = 0;

                // geminifs_info("GPU Controller: Granularity size %zu > maxIOsize %llu, performing second-level cutting\n",
                //               granularity_size, min_max_io_size);

                while (sub_remaining > 0)
                {
                    size_t sub_slice_size = std::min(sub_remaining, (size_t)min_max_io_size);
                    size_t sub_global_offset = current_offset + sub_local_offset;

                    SubSliceInfo sub_slice(sub_local_offset, sub_slice_size, sub_global_offset);
                    group.sub_slices.push_back(sub_slice);

                    // geminifs_info("GPU Controller: Sub-slice[%zu]: local_offset=%zu, size=%zu, global_offset=%zu\n",
                    //               sub_index, sub_local_offset, sub_slice_size, sub_global_offset);

                    sub_local_offset += sub_slice_size;
                    sub_remaining -= sub_slice_size;
                    sub_index++;
                }
            }

            // 添加granularity组到列表
            dma_ctx->granularity_groups.push_back(std::move(group));

            current_offset += granularity_size;
            remaining_size -= granularity_size;
        }

        geminifs_debug("GPU Controller: Two-level slicing complete: %zu granularity groups "
                       "(granularity %lu -> maxIOsize %lu)\n",
                       dma_ctx->granularity_groups.size(), granularity, min_max_io_size);
    }
    else
    {
        // 没有外部粒度，只按照maxIOsize进行切割，创建单个granularity组
        uint64_t gpu_tensor_ptr = reinterpret_cast<uint64_t>(dma_ctx->dma_ptr->vaddr);
        GranularitySliceGroup group(gpu_tensor_ptr, 0, tensor_size);

        if (tensor_size <= min_max_io_size)
        {
            // 数据小于等于maxIOsize，不进行切片
            SubSliceInfo sub_slice(0, tensor_size, 0);
            group.sub_slices.push_back(sub_slice);

            geminifs_debug("GPU Controller: Tensor size %zu <= maxIOsize %lu, no slicing needed\n",
                           tensor_size, min_max_io_size);
        }
        else
        {
            // 需要按照maxIOsize进行切片
            size_t remaining_size = tensor_size;
            size_t current_offset = 0;
            size_t sub_index = 0;

            while (remaining_size > 0)
            {
                size_t slice_size = std::min(remaining_size, (size_t)min_max_io_size);

                SubSliceInfo sub_slice(current_offset, slice_size, current_offset);
                group.sub_slices.push_back(sub_slice);

                // geminifs_info("GPU Controller: Sub-slice[%zu]: local_offset=%zu, size=%zu, global_offset=%zu\n",
                            //   sub_index, current_offset, slice_size, current_offset);

                current_offset += slice_size;
                remaining_size -= slice_size;
                sub_index++;
            }

            geminifs_debug("GPU Controller: Single-level slicing: %zu slices by maxIOsize %lu\n",
                           group.sub_slices.size(), min_max_io_size);
        }

        // 添加单个granularity组
        dma_ctx->granularity_groups.push_back(std::move(group));
    }

    // // 打印granularity组的详细信息
    // for (size_t i = 0; i < dma_ctx->granularity_groups.size(); i++) {
    //     const auto& group = dma_ctx->granularity_groups[i];
    //     geminifs_info("GPU Controller: Granularity Group[%zu]: gpu_ptr=0x%lx, offset=%zu, size=%zu, sub_slices=%zu\n",
    //                   i, group.gpu_tensor_ptr, group.granularity_offset, group.granularity_size, group.sub_slices.size());

    //     for (size_t j = 0; j < group.sub_slices.size(); j++) {
    //         const auto& sub_slice = group.sub_slices[j];
    //         geminifs_info("  Sub-slice[%zu]: local_offset=%zu, size=%zu, global_offset=%zu\n",
    //                       j, sub_slice.offset, sub_slice.size, sub_slice.global_offset);
    //     }
    // }

    return true;
}

inline bool checkTensorEntries(const std::vector<torch::Tensor>& tensor_list) {
	auto tensor_size = tensor_list[0].numel() * tensor_list[0].element_size();
	for (const auto &entry : tensor_list) {
		auto entry_size = entry.numel() * entry.element_size();
		if (entry_size != tensor_size) {
			geminifs_error("GPU Controller: Inconsistent tensor sizes in the list. Expected %zu, got %zu\n",
						   tensor_size, entry_size);
			return false;
		}
	}
	return true;
}


inline uint64_t getPrpMappingSize(const geminifs_dma *dma_ctx) {
    if (!dma_ctx) {
        geminifs_error("GPU Controller: Invalid DMA context\n");
        return 0;
    }

    uint64_t nr_mappings = 0;
    for (const auto &group : dma_ctx->granularity_groups) {
        nr_mappings += group.prp_mappings.size();
    }

    return nr_mappings * sizeof(PRPMappingEntry);
}

std::vector<struct geminifs_dma*> GPUController::createDMAContexts(const std::vector<torch::Tensor>& tensor_list, uint64_t granularity) {
	if (!checkTensorEntries(tensor_list)) {
		return {};
	}

	auto tensor_size = tensor_list[0].numel() * tensor_list[0].element_size();
	if (granularity > 0 && tensor_size > granularity && tensor_size % granularity != 0) {
		geminifs_error("GPU Controller: Tensor size %zu is not a multiple of granularity %lu. Memory registration failed.\n",
					   tensor_size, granularity);
		return {};
	}

	if (nvme_controllers_.empty()) {
		geminifs_error("GPU Controller: No NVMe controllers available for DMA context creation\n");
		return {};
	}

	auto first_controller = nvme_controllers_[0];
    if (!first_controller || !first_controller->controller) {
        geminifs_error("GPU Controller: Invalid NVMe controller for DMA context creation\n");
        return {};
    }

	std::vector<geminifs_dma*> dma_ctx_list(tensor_list.size(), nullptr);
	auto releaseDMAContexts = [&ctxs = dma_ctx_list]() {
		for (auto &ctx : ctxs) {
			if (ctx != nullptr) {
				delete ctx;
			}
		}
	};

	for (size_t i = 0; i < tensor_list.size(); i++) {
		auto &dma_ctx = dma_ctx_list[i];
		dma_ctx = new geminifs_dma();
		dma_ctx->dma_ptr  = getDeviceDma(first_controller->controller->ctrl,
									  tensor_list[i].data_ptr(), tensor_size, device_id_);
		if (dma_ctx->dma_ptr == nullptr) {
			geminifs_error("GPU Controller: Failed to get DMA pointer for tensor at %p\n", tensor_list[i].data_ptr());
			// 清理已创建的DMA上下文
			releaseDMAContexts();
			return {};
		}

		if (!performDMASlicing(dma_ctx, tensor_size, granularity)) {
			geminifs_error("GPU Controller: Failed to perform DMA slicing for tensor at %p\n", tensor_list[i].data_ptr());
			// 清理已创建的DMA上下文
			releaseDMAContexts();
			return {};
		}
	}

	if (!initializePRPList(dma_ctx_list)) {
		geminifs_error("GPU Controller: Failed to initialize PRP list for DMA contexts\n");
		// 清理已创建的DMA上下文
		releaseDMAContexts();
		return {};
	}

	auto nr_mappings = getPrpMappingSize(dma_ctx_list[0]);
	if (nr_mappings == 0) {
		geminifs_error("GPU Controller: empty prp mappings\n");
		return {};
	}

	// assume all dma ctx have the same prp mappings
	void *prp_buffers = nullptr;
	auto prp_mappings_bytes = nr_mappings * sizeof(PRPMappingEntry);
	if (cudaMalloc(&prp_buffers,  prp_mappings_bytes * dma_ctx_list.size()) != cudaSuccess) {
		geminifs_error("GPU Controller: Failed to allocate GPU memory for PRP buffers\n");
		return {};
	}

	geminifs_debug("GPU Controller: Allocated %lu bytes for PRP mappings for %zu tensors at %p\n",
				  prp_mappings_bytes * dma_ctx_list.size(), dma_ctx_list.size(), prp_buffers);

	uint64_t buffer_offset = 0ul;
	for (size_t i = 0; i < tensor_list.size(); i++) {
		auto &dma_ctx = dma_ctx_list[i];
		if (!addPRPMappingsToGPU(dma_ctx, (void *)((uint64_t)prp_buffers + buffer_offset), prp_mappings_bytes)) {
			geminifs_error("GPU Controller: Failed to add PRP mappings to GPU for tensor[%ld] at %p\n", i, tensor_list[i].data_ptr());
			// 清理已创建的DMA上下文
			releaseDMAContexts();
			return {};
		}
		buffer_offset += prp_mappings_bytes;
	}

	return dma_ctx_list;
}

geminifs_dma *GPUController::createDMAContext(const torch::Tensor &tensor, uint64_t granularity)
{
    auto tensor_size = tensor.numel() * tensor.element_size();

    // 如果指定了切割粒度（非0），检查tensor大小是否为粒度的整数倍
    if (granularity > 0 && tensor_size > tensor_size)
    {
        if (tensor_size % granularity != 0)
        {
            geminifs_error("GPU Controller: Tensor size %zu is not a multiple of granularity %lu. Memory registration failed.\n",
                           tensor_size, granularity);
            return nullptr;
        }
        geminifs_debug("GPU Controller: Using external granularity %lu for tensor size %zu\n", granularity, tensor_size);
    }

    // For now, we'll assume we have at least one NVMe controller to get the ctrl pointer
    if (nvme_controllers_.empty())
    {
        geminifs_error("GPU Controller: No NVMe controllers available for DMA context creation\n");
        return nullptr;
    }

    // Use the first controller for DMA creation
    auto first_controller = nvme_controllers_[0];
    if (!first_controller || !first_controller->controller)
    {
        geminifs_error("GPU Controller: Invalid NVMe controller for DMA context creation\n");
        return nullptr;
    }

    DmaPtr dma_ptr = getDeviceDma(first_controller->controller->ctrl,
                                  tensor.data_ptr(), tensor_size, device_id_);
    if (dma_ptr == nullptr)
    {
        geminifs_error("GPU Controller: Failed to get DMA pointer for tensor\n");
        return nullptr;
    }

    geminifs_dma *dma_ctx = new geminifs_dma();
    dma_ctx->dma_ptr = dma_ptr;

    // 执行 DMA 切片
    if (!performDMASlicing(dma_ctx, tensor_size, granularity))
    {
        delete dma_ctx;
        return nullptr;
    }
    if (!initializePRPEntries(dma_ctx))
    {
        delete dma_ctx;
        return nullptr;
    }
    if (!addPRPMappingsToGPU(dma_ctx))
    {
        delete dma_ctx;
        return nullptr;
    }

    return dma_ctx;
}

static inline size_t getPrpListSize(geminifs_dma *dma_ctx) {
    size_t nr_prp_lists = 0;
    for (const auto &group : dma_ctx->granularity_groups) {
        for (size_t sub_idx = 0; sub_idx < group.sub_slices.size(); sub_idx++) {
            const auto &sub_slice = group.sub_slices[sub_idx];
            if (sub_slice.size > PRP_SIZE_DUAL_PAGE) {
                nr_prp_lists++;
            } 
        }    
    }
    return nr_prp_lists;
}

bool GPUController::initializePRPList(std::vector<geminifs_dma*> &dma_ctxs) {
    if (dma_ctxs.empty()) {
        geminifs_error("GPU Controller: DMA context is null\n");
        return false;
    }

    if (nvme_controllers_.empty()) {
        geminifs_error("GPU Controller: No NVMe controllers available\n");
        return false;
    }

    for (const auto &ctx : dma_ctxs) {
        if (ctx->granularity_groups.empty()) {
            geminifs_error("GPU Controller: No granularity groups found\n");
            return false;
        }
    }

    // use the first dma_ctx to get the total number of prp lists
    const auto prp_list_size = getPrpListSize(dma_ctxs[0]);
    if (prp_list_size == 0) {
        geminifs_debug("GPU Controller: No PRP lists needed for the provided DMA contexts\n");
        return true;
    }

    const auto memory_size = prp_list_size * PRP_SIZE_SINGLE_PAGE; // 一个type 2 的prp list需要4KB

    void *prp_list_gpu_addr = nullptr;
    if (cudaMalloc(&prp_list_gpu_addr, memory_size * dma_ctxs.size()) != cudaSuccess) {
        geminifs_error("GPU Controller: Failed to allocate GPU memory for PRP lists\n");
        return false;
    }

    void *current_prp_list_gpu_addr = prp_list_gpu_addr;
    for (const auto &dma_ctx : dma_ctxs) {
		if (!dma_ctx) {
			cudaFree(prp_list_gpu_addr);
			geminifs_error("GPU Controller: Invalid DMA context in the list\n");
			return false;
		}

        if (!doInitializePRPList(dma_ctx, current_prp_list_gpu_addr, memory_size)) {
            cudaFree(prp_list_gpu_addr);
            geminifs_error("GPU Controller: Failed to initialize PRP list for a DMA context\n");
            return false;
        }
        current_prp_list_gpu_addr = reinterpret_cast<void*>(
            reinterpret_cast<uint64_t>(current_prp_list_gpu_addr) + memory_size);
    }
    
    return true;
}

bool GPUController::doInitializePRPList(geminifs_dma *dma_ctx, void *prp_list_gpu_addr, size_t gpu_size) {
    auto first_controller = nvme_controllers_[0];
    if (!first_controller || !first_controller->controller) {
        geminifs_error("GPU Controller: Invalid NVMe controller for DMA context creation\n");
        return false;
    }

    size_t prp_list_count = getPrpListSize(dma_ctx);
    if (prp_list_count != 0 && prp_list_count != gpu_size / PRP_SIZE_SINGLE_PAGE) {
        geminifs_error("GPU Controller: Mismatch in PRP list count and allocated GPU memory size\n");
        return false;
    }

    if (prp_list_count != 0 && prp_list_gpu_addr != nullptr && gpu_size != 0) {
        dma_ctx->type2_prp_dma_ptr = getDeviceDma(first_controller->controller->ctrl, 
                                                prp_list_gpu_addr, gpu_size, device_id_);
        if (!dma_ctx->type2_prp_dma_ptr) {
            geminifs_error("GPU Controller: Failed to create DMA pointer for PRP list GPU memory\n");
            return false;
        }
    }

    size_t current_ioaddr_index = 0;   // 当前使用的ioaddrs索引
    size_t prp_list_offset = 0;         // 当前PRP列表在GPU内存中的偏移量
    for (size_t group_idx = 0; group_idx < dma_ctx->granularity_groups.size(); group_idx++) {
        auto &group = dma_ctx->granularity_groups[group_idx];
        for (size_t sub_idx = 0; sub_idx < group.sub_slices.size(); sub_idx++) {
            const auto &sub_slice = group.sub_slices[sub_idx];
            size_t slice_size = sub_slice.size;
            size_t slice_pages = (slice_size + PRP_SIZE_SINGLE_PAGE - 1) / PRP_SIZE_SINGLE_PAGE; // 切片需要的4K页数（向上取整）
            
            auto transfer_type = getPrpTransferType(slice_size);
            if (current_ioaddr_index + slice_pages > dma_ctx->dma_ptr->n_ioaddrs) {
                geminifs_error("GPU Controller: Not enough IO addrs for PRP list initialization\n");
                return false;
            }

            uint64_t prp1 = dma_ctx->dma_ptr->ioaddrs[current_ioaddr_index];
            uint64_t prp2 = 0;
            switch (transfer_type) {
				case PRP_TYPE_SINGLE_PAGE: 
                    break;
                case PRP_TYPE_DUAL_PAGE: 
                    prp2 = slice_pages > 1 ? dma_ctx->dma_ptr->ioaddrs[current_ioaddr_index + 1] : 0;
					break;
                case PRP_TYPE_LIST: {
                    // 构建PRP列表
                	if (dma_ctx->type2_prp_dma_ptr && dma_ctx->type2_prp_dma_ptr->n_ioaddrs > 0) {
						auto page_index = prp_list_offset / PRP_SIZE_SINGLE_PAGE;
						if (page_index < dma_ctx->type2_prp_dma_ptr->n_ioaddrs) {
							prp2 = dma_ctx->type2_prp_dma_ptr->ioaddrs[page_index];
						} else {
							geminifs_error("GPU Controller: PRP list offset exceeds allocated GPU memory\n");
							return false;
						}
                	} else {
						geminifs_error("GPU Controller: Invalid DMA pointer for PRP list\n");
						return false;
					}

					auto remaining_pages = slice_pages - 1;
					std::vector<uint64_t> prp_list_host(PRP_ENTRIES_PER_PAGE + 1, 0); // last entry for next PRP list pointer
					for (size_t j = 0; j < remaining_pages && j < PRP_ENTRIES_PER_PAGE; j++) {
						prp_list_host[j] = dma_ctx->dma_ptr->ioaddrs[current_ioaddr_index + 1 + j];
					}

					if (remaining_pages > PRP_ENTRIES_PER_PAGE) {
						geminifs_warn("GPU Controller: PRP list exceeds single page, chaining not implemented in this example\n");
					}

					auto err = cudaMemcpy(
						static_cast<uint8_t*>(dma_ctx->type2_prp_dma_ptr->vaddr) + prp_list_offset,
						prp_list_host.data(),
						PRP_SIZE_SINGLE_PAGE,
						cudaMemcpyHostToDevice);

					if (err != cudaSuccess) {
						geminifs_error("GPU Controller: cudaMemcpy failed during PRP list initialization: %s\n", cudaGetErrorString(err));
						return false;
					}

					prp_list_offset += PRP_SIZE_SINGLE_PAGE;
					break;
                }
                default:
                    geminifs_error("GPU Controller: Unknown transfer type %d during PRP list initialization\n", transfer_type);
                    return false;
            }

			group.prp_mappings.push_back(PRPMappingEntry(transfer_type, 
								(uint32_t)slice_size, prp1, prp2, sub_slice.offset));
			current_ioaddr_index += slice_pages;
        }
    }

    return true;
}


bool GPUController::initializePRPEntries(geminifs_dma *dma_ctx)
{
    if (!dma_ctx)
    {
        geminifs_error("GPU Controller: DMA context is null\n");
        return false;
    }

    if (nvme_controllers_.empty())
    {
        geminifs_error("GPU Controller: No NVMe controllers available\n");
        return false;
    }

    if (dma_ctx->granularity_groups.empty())
    {
        geminifs_error("GPU Controller: No granularity groups found\n");
        return false;
    }

    auto first_controller = nvme_controllers_[0];
    if (!first_controller || !first_controller->controller)
    {
        geminifs_error("GPU Controller: Invalid NVMe controller for DMA context creation\n");
        return false;
    }

    // 统计全局第三种类型(transfer_type=2)的数量
    size_t total_type2_count = 0;

    // 为每个GranularitySliceGroup构建PRP映射条目
    for (size_t group_idx = 0; group_idx < dma_ctx->granularity_groups.size(); group_idx++)
    {
        auto &group = dma_ctx->granularity_groups[group_idx];

        // geminifs_info("GPU Controller: Building PRP mappings for Granularity Group[%zu]: %zu sub-slices\n",
        //               group_idx, group.sub_slices.size());

        // 为该granularity group预留PRP映射空间
        group.prp_mappings.reserve(group.sub_slices.size());

        // 统计该group的type2数量
        size_t group_type2_count = 0;

        // 为该group的每个子切片创建PRP映射条目
        for (size_t sub_idx = 0; sub_idx < group.sub_slices.size(); sub_idx++)
        {
            const auto &sub_slice = group.sub_slices[sub_idx];
            size_t slice_size = sub_slice.size;
            uint32_t transfer_type = 0; // 默认类型0

            // 根据slice_size确定NVMe IO cmd类型
            if (slice_size <= 4096)
            {
                transfer_type = 0; // 小于等于4K
            }
            else if (slice_size <= 8192)
            {
                transfer_type = 1; // 大于4K小于等于8K
            }
            else
            {
                transfer_type = 2;   // 大于8K
                group_type2_count++; // 统计该group的第三种类型数量
                total_type2_count++; // 统计全局第三种类型数量
            }

            // 创建PRPMappingEntry (prp1和prp2先不初始化，tensor_offset使用granularity内的本地偏移)
            PRPMappingEntry entry(transfer_type, (uint32_t)slice_size, 0, 0, sub_slice.offset);
            group.prp_mappings.push_back(entry);

            geminifs_debug("GPU Controller: Group[%zu] Sub-slice[%zu]: transfer_type=%u, size=%zu\n",
                           group_idx, sub_idx, transfer_type, slice_size);
        }

        // geminifs_info("GPU Controller: Group[%zu] created %zu PRP mappings (%zu type2)\n",
                    //   group_idx, group.prp_mappings.size(), group_type2_count);
    }

    // 记录全局第三种类型的数量
    dma_ctx->type2_prp_count = total_type2_count;

    // 为第三种类型的PRP分配4KB对齐的GPU内存
    if (total_type2_count > 0)
    {
        size_t memory_size = total_type2_count * PRP_SIZE_SINGLE_PAGE; // 每个第三种类型需要4KB
        dma_ctx->type2_prp_dma_ptr = createDma(first_controller->controller->ctrl, memory_size, device_id_);
        if (!dma_ctx->type2_prp_dma_ptr)
        {
            geminifs_error("GPU Controller: Failed to create DMA pointer for type2 PRP GPU memory\n");
            return false;
        }
        // geminifs_info("GPU Controller: Allocated %zu bytes GPU memory for %zu type2 PRP entries\n",
        //               memory_size, total_type2_count);
    }
    else
    {
        geminifs_debug("GPU Controller: No type2 PRP entries found, no GPU memory allocation needed\n");
    }

    // 初始化每个granularity group的PRP条目
    size_t current_ioaddr_index = 0;    // 当前使用的ioaddrs索引
    size_t type2_gpu_memory_offset = 0; // type2_prp_gpu_memory中的偏移量

    for (size_t group_idx = 0; group_idx < dma_ctx->granularity_groups.size(); group_idx++)
    {
        auto &group = dma_ctx->granularity_groups[group_idx];

        // geminifs_info("GPU Controller: Initializing PRP entries for Group[%zu]: gpu_ptr=0x%lx\n",
                    //   group_idx, group.gpu_tensor_ptr);

        // 处理该group的每个子切片
        for (size_t sub_idx = 0; sub_idx < group.sub_slices.size(); sub_idx++)
        {
            const auto &sub_slice = group.sub_slices[sub_idx];
            size_t slice_size = sub_slice.size;
            size_t slice_pages = (slice_size + PRP_SIZE_SINGLE_PAGE - 1) / PRP_SIZE_SINGLE_PAGE; // 切片需要的4K页数（向上取整）

            PRPMappingEntry &group_entry = group.prp_mappings[sub_idx];

            // 检查ioaddrs边界
            if (current_ioaddr_index + slice_pages > dma_ctx->dma_ptr->n_ioaddrs)
            {
                geminifs_error("GPU Controller: Not enough ioaddrs for Group[%zu] Sub-slice[%zu] (need %zu, available %zu)\n",
                               group_idx, sub_idx, slice_pages, dma_ctx->dma_ptr->n_ioaddrs - current_ioaddr_index);
                return false;
            }

            // 根据NVMe命令类型设置PRP1和PRP2
            if (group_entry.transfer_type == 0)
            {
                // 类型0: <= 4K, 单页传输
                // PRP1指向数据页，PRP2不使用
                group_entry.prp1 = dma_ctx->dma_ptr->ioaddrs[current_ioaddr_index];
                group_entry.prp2 = 0;

                geminifs_debug("GPU Controller: Group[%zu] Sub-slice[%zu] Type0: PRP1=0x%lx, PRP2=0x%lx\n",
                               group_idx, sub_idx, group_entry.prp1, group_entry.prp2);
            }
            else if (group_entry.transfer_type == 1)
            {
                // 类型1: 4K < size <= 8K, 双页传输
                // PRP1指向第一个数据页，PRP2指向第二个数据页
                group_entry.prp1 = dma_ctx->dma_ptr->ioaddrs[current_ioaddr_index];
                if (slice_pages > 1)
                {
                    group_entry.prp2 = dma_ctx->dma_ptr->ioaddrs[current_ioaddr_index + 1];
                }
                else
                {
                    group_entry.prp2 = 0; // 如果实际只有一页，PRP2设为0
                }

                geminifs_debug("GPU Controller: Group[%zu] Sub-slice[%zu] Type1: PRP1=0x%lx, PRP2=0x%lx\n",
                               group_idx, sub_idx, group_entry.prp1, group_entry.prp2);
            }
            else if (group_entry.transfer_type == 2)
            {
                // 类型2: > 8K, PRP List传输
                // PRP1指向第一个数据页，PRP2指向PRP List页
                group_entry.prp1 = dma_ctx->dma_ptr->ioaddrs[current_ioaddr_index];

                // PRP2指向type2_prp_gpu_memory中对应的4K空间
                if (dma_ctx->type2_prp_dma_ptr && dma_ctx->type2_prp_dma_ptr->n_ioaddrs > 0)
                {
                    size_t type2_page_index = type2_gpu_memory_offset / PRP_SIZE_SINGLE_PAGE;
                    if (type2_page_index < dma_ctx->type2_prp_dma_ptr->n_ioaddrs)
                    {
                        group_entry.prp2 = dma_ctx->type2_prp_dma_ptr->ioaddrs[type2_page_index];
                    }
                    else
                    {
                        geminifs_error("GPU Controller: Type2 page index %zu exceeds available pages %zu\n",
                                       type2_page_index, dma_ctx->type2_prp_dma_ptr->n_ioaddrs);
                        return false;
                    }
                }
                else
                {
                    geminifs_error("GPU Controller: type2_prp_dma_ptr is invalid for Group[%zu] Sub-slice[%zu]\n",
                                   group_idx, sub_idx);
                    return false;
                }

                // 填充type2_prp_gpu_memory: 将剩余数据页的ioaddrs复制到GPU内存
                size_t remaining_pages = slice_pages - 1;    // 除去PRP1指向的第一页
                std::vector<uint64_t> prp_list_host(512, 0); // 4KB / 8字节 = 512个entry

                for (size_t j = 0; j < remaining_pages && j < 511; j++)
                { // 最多511个entry
                    prp_list_host[j] = dma_ctx->dma_ptr->ioaddrs[current_ioaddr_index + 1 + j];
                    // geminifs_debug("GPU Controller: Group[%zu] Sub-slice[%zu] PRP List[%zu] = 0x%lx (ioaddr_index=%zu)\n",
                    //                group_idx, sub_idx, j, prp_list_host[j], current_ioaddr_index + 1 + j);
                }

                // 将PRP List复制到GPU内存
                cudaError_t err = cudaMemcpy(
                    static_cast<char *>(dma_ctx->type2_prp_dma_ptr->vaddr) + type2_gpu_memory_offset,
                    prp_list_host.data(),
                    PRP_SIZE_SINGLE_PAGE,
                    cudaMemcpyHostToDevice);

                if (err != cudaSuccess)
                {
                    geminifs_error("GPU Controller: Failed to copy PRP list to GPU memory for Group[%zu] Sub-slice[%zu]: %s\n",
                                   group_idx, sub_idx, cudaGetErrorString(err));
                    return false;
                }

                // geminifs_info("GPU Controller: Group[%zu] Sub-slice[%zu] Type2: PRP1=0x%lx, PRP2=0x%lx, filled %zu ioaddrs into GPU memory\n",
                            //   group_idx, sub_idx, group_entry.prp1, group_entry.prp2, remaining_pages);

                type2_gpu_memory_offset += PRP_SIZE_SINGLE_PAGE; // 移动到下一个4K空间
            }

            current_ioaddr_index += slice_pages; // 移动到下一个切片的起始ioaddr
        }

        // geminifs_info("GPU Controller: Group[%zu] PRP initialization complete: %zu entries processed\n",
                    //   group_idx, group.prp_mappings.size());
    }

    // geminifs_info("GPU Controller: Successfully configured PRP mappings for %zu granularity groups\n",
                //   dma_ctx->granularity_groups.size());

    return true;
}

bool GPUController::addPRPMappingsToGPU(geminifs_dma *dma_ctx, void *gpu_buffer, uint64_t gpu_buffer_size)
{
    if (!dma_ctx)
    {
        geminifs_error("GPU Controller: DMA context is null for PRP mappings registration\n");
        return false;
    }

    if (!memory_mapper_)
    {
        geminifs_error("GPU Controller: Memory mapper not initialized\n");
        return false;
    }

    if (dma_ctx->granularity_groups.empty())
    {
        geminifs_error("GPU Controller: No granularity groups found for PRP mappings registration\n");
        return false;
    }

    geminifs_debug("GPU Controller: Adding PRP mappings to GPU memory for %zu granularity groups, gpu_buffer %p, buffer_size %ld\n",
                  dma_ctx->granularity_groups.size(), gpu_buffer, gpu_buffer_size);

    // 遍历每个granularity group，将其PRP映射注册到GPU内存
	uint64_t buffer_offset = 0;
    for (size_t group_idx = 0; group_idx < dma_ctx->granularity_groups.size(); group_idx++) {
        const auto &group = dma_ctx->granularity_groups[group_idx];
        if (group.prp_mappings.empty()) {
            geminifs_warn("GPU Controller: Group[%zu] has no PRP mappings to register\n", group_idx);
            continue;
        }

		const uint64_t group_bytes = group.prp_mappings.size() * sizeof(PRPMappingEntry);
		if (gpu_buffer && gpu_buffer_size != 0) {
			if (buffer_offset + group_bytes > gpu_buffer_size) {
				geminifs_error(
					"GPU Controller: Insufficient GPU buffer for Group[%zu] "
					"(need %llu, left %llu)\n",
					group_idx,
					(unsigned long long)group_bytes,
					(unsigned long long)(gpu_buffer_size - buffer_offset));
				return false;
			}
		}

        auto cur_gpu_buffer = gpu_buffer != nullptr ? (void *)((uint64_t)gpu_buffer + buffer_offset) : nullptr;
        bool success = memory_mapper_->addBatchMappings(group.gpu_tensor_ptr, group.granularity_size, 
														group.prp_mappings, cur_gpu_buffer, 
														group_bytes);

        if (!success) {
            geminifs_error("GPU Controller: Failed to add batch mappings for Group[%zu]\n", group_idx);
            return false;
        }

		buffer_offset += group_bytes;
        // geminifs_info("GPU Controller: Successfully registered %zu PRP mappings for Group[%zu]\n",
        //               group.prp_mappings.size(), group_idx);
    }

    // geminifs_info("GPU Controller: Successfully registered PRP mappings for all %zu granularity groups\n",
    //               dma_ctx->granularity_groups.size());

    return true;
}

// === GPUControllerRegistry Implementation ===

GPUControllerRegistry &GPUControllerRegistry::getInstance()
{
    static GPUControllerRegistry instance;
    return instance;
}

bool GPUControllerRegistry::registerGPUController(int device_id, GPUControllerPtr controller)
{
    if (!controller)
    {
        geminifs_error("GPU Controller Registry: Invalid controller provided for device %d\n", device_id);
        return false;
    }

    std::lock_guard<std::mutex> lock(registry_mutex_);

    if (gpu_controllers_.find(device_id) != gpu_controllers_.end())
    {
        geminifs_warn("GPU Controller Registry: Device %d already has a registered controller\n", device_id);
        return false;
    }

    gpu_controllers_[device_id] = controller;
    geminifs_debug("GPU Controller Registry: Registered controller for device %d\n", device_id);
    return true;
}

bool GPUControllerRegistry::unregisterGPUController(int device_id)
{
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = gpu_controllers_.find(device_id);
    if (it == gpu_controllers_.end())
    {
        geminifs_warn("GPU Controller Registry: No controller registered for device %d\n", device_id);
        return false;
    }

    gpu_controllers_.erase(it);
    geminifs_debug("GPU Controller Registry: Unregistered controller for device %d\n", device_id);
    return true;
}

GPUControllerPtr GPUControllerRegistry::getGPUController(int device_id)
{
    std::lock_guard<std::mutex> lock(registry_mutex_);

    auto it = gpu_controllers_.find(device_id);
    return (it != gpu_controllers_.end()) ? it->second : nullptr;
}

const std::unordered_map<int, GPUControllerPtr> &GPUControllerRegistry::getAllGPUControllers() const
{
    return gpu_controllers_;
}

void GPUControllerRegistry::clearAll()
{
    std::lock_guard<std::mutex> lock(registry_mutex_);
    gpu_controllers_.clear();
    geminifs_debug("GPU Controller Registry: Cleared all registered controllers\n");
}

/**
 * 设备端安全检查函数：检查文件读写操作的安全性
 * @param d_fd NVMe文件描述符
 * @param tensor_size 找到的tensor大小
 * @param offset 文件偏移量
 * @param len 读写长度
 * @return true表示安全，false表示不安全
 */
__device__ bool check_file_access_safety(NVMe_File *d_fd,
                                         uint64_t tensor_size,
                                         size_t offset,
                                         size_t len)
{
    if (!d_fd)
    {
        geminifs_error("Safety Check Error: NVMe_File pointer is null\n");
        return false;
    }

    // 获取文件大小（通过公共方法）
    uint64_t file_max_size = d_fd->get_file_size();
    if (file_max_size == 0)
    {
        geminifs_error("Safety Check Error: Could not get file size (hdr might be null)\n");
        return false;
    }

    // 检查tensor_size是否足够容纳要读取的数据
    if (tensor_size < len)
    {
        geminifs_error("Safety Check Error: tensor_size (%lu) < read length (%zu)\n",
               tensor_size, len);
        return false;
    }

    // 检查文件访问边界
    if (offset + len > file_max_size)
    {
        geminifs_error("Safety Check Error: access beyond file boundary - "
               "offset (%zu) + len (%zu) = %zu > file_max_size (%lu)\n",
               offset, len, offset + len, file_max_size);
        return false;
    }

    // 检查offset是否有效
    if (offset >= file_max_size)
    {
        geminifs_error("Safety Check Error: offset (%zu) >= file_max_size (%lu)\n",
               offset, file_max_size);
        return false;
    }

    // 检查len是否为0
    if (len == 0)
    {
        geminifs_error("Safety Check Warning: read length is 0\n");
        return false;
    }

    // 检查4K对齐 - NVMe设备通常要求4K对齐访问
    if (offset % 4096 != 0)
    {
        printf("Safety Check Error: offset (%zu) is not 4K aligned (remainder: %zu)\n",
               offset, offset % 4096);
        return false;
    }

    if (len % 4096 != 0)
    {
        geminifs_error("Safety Check Error: length (%zu) is not 4K aligned (remainder: %zu)\n",
               len, len % 4096);
        return false;
    }

    // // 打印成功信息
    // printf("Safety Check Passed: tensor_size=%lu, offset=%zu, len=%zu, file_max_size=%lu (all 4K aligned)\n",
    //        tensor_size, offset, len, file_max_size);

    return true;
}


// 所有的检查都在host端进行
__global__ void nvme_xfer_kernel(NVMeFilesSpan nvme_file,
                                 PRPMappingEntrySpan prp_entries,
                                 uint64_t tensor_size,
                                 size_t base_file_offset,
                                 bool is_read) {

    const uint32_t tid   = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t total = prp_entries.size();
    if (tid >= total) return;

    auto entry = &prp_entries[tid];
    const uint64_t blk_size = entry->data_length;
    const uint64_t gpu_blk  = (base_file_offset / blk_size) + tid;
    const uint64_t fd_idx   = gpu_blk % nvme_file.size();
    const uint64_t file_off = (gpu_blk / nvme_file.size()) * blk_size;

    // geminifs_info("tid=%u fd_idx=%lu prp1=0x%lx prp2=0x%lx file_off=%lu len=%lu\n",
    //               tid, fd_idx, entry->prp1, entry->prp2, file_off, blk_size);
    

    auto file = nvme_file[fd_idx];
    if (!check_file_access_safety(file, tensor_size, file_off, blk_size)) {
        geminifs_error("nvme_xfer_kernel: Safety check failed for tid=%u\n", tid);
        return;
    }

    if (is_read) {
        nvme_controller_g_read(file, entry->prp1, entry->prp2, file_off, blk_size);
    } else {
        nvme_controller_g_write(file, entry->prp1, entry->prp2, file_off, blk_size);
    }
}

// 这里为了避免拷贝直接用了两个span，可以直接调用长度为1的batch发起。
__global__ void nvme_kv_xfer_kernel(NVMeFilesSpan nvme_file,
                                    PRPMappingEntrySpan k_mappings,
                                    PRPMappingEntrySpan v_mappings,
                                    uint64_t tensor_size,
                                    size_t base_file_offset,
                                    bool is_read) {

    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto k_size = k_mappings.size();
    const auto v_size = v_mappings.size();
    const auto total  = k_size + v_size;

    if (tid >= total || nvme_file.empty()) return;

    auto entry = tid < k_size ? &k_mappings[tid] : &v_mappings[tid - k_size];
    const uint64_t blk_size = entry->data_length;
    const uint64_t gpu_blk  = (base_file_offset / blk_size) + tid;
    const uint64_t fd_idx   = gpu_blk % nvme_file.size();
    const uint64_t file_off = (gpu_blk / nvme_file.size()) * blk_size;

    // geminifs_info("tid=%u fd_idx=%lu prp1=0x%lx prp2=0x%lx file_off=%lu len=%lu\n",
    //               tid, fd_idx, entry->prp1, entry->prp2, file_off, blk_size);

    auto file = nvme_file[fd_idx];
    if (!check_file_access_safety(file, tensor_size, file_off, blk_size)) {
        geminifs_error("nvme_kv_xfer_kernel: Safety check failed for tid=%u\n", tid);
        return;
    }

    if (is_read) {
        nvme_controller_g_read(file, entry->prp1, entry->prp2, file_off, blk_size);
    } else {
        nvme_controller_g_write(file, entry->prp1, entry->prp2, file_off, blk_size);
    }
}


__global__ void nvme_batch_xfer_kernel(BatchIoEntry* io_ctx,
                                       uint64_t tensor_size,
                                       uint32_t total_count,
                                       bool is_read) {
    
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= total_count || !io_ctx) return;

    // const auto local_tid = 

    auto ctx = io_ctx->d_ioctxs[tid];
    auto entry = ctx.prp_entry;
    if (!entry) return;

    auto nvme_file = ctx.nvme_files;
    if (nvme_file.size() == 0) return;

    const uint64_t blk_size = entry->data_length;
    const uint64_t gpu_blk  = (ctx.file_offset / blk_size) + ctx.prp_idx;
    const uint64_t fd_idx   = gpu_blk % nvme_file.size();
    const uint64_t file_off = (gpu_blk / nvme_file.size()) * blk_size;      
    
    // geminifs_info("tid=%u fd_idx=%lu prp1=0x%lx prp2=0x%lx file_off=%lu len=%lu\n",
    //               tid, fd_idx, entry->prp1, entry->prp2, file_off, blk_size);
    
    auto file = nvme_file[fd_idx];
    if (!check_file_access_safety(file, tensor_size, file_off, blk_size)) {
        geminifs_error("nvme_batch_xfer_kernel: Safety check failed for tid=%u\n", tid);
        return;
    }

    if (is_read) {
        nvme_controller_g_read(file, entry->prp1, entry->prp2, file_off, blk_size);
    } else {
        nvme_controller_g_write(file, entry->prp1, entry->prp2, file_off, blk_size);
    }

}