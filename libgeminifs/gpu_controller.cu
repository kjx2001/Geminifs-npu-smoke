#include "geminifs_helper.h"
#include "geminifs_mem.h"
#include "buffer.h"
#include <cuda_runtime.h>
#include <cassert>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <unordered_set>
#include "gpu_controller.cuh"

__device__ uint32_t gpu_lookup_all_prp_mappings(uint64_t tensor_ptr,
                                                GPUHashEntry *hash_table,
                                                GPUMappingNode *mapping_nodes,
                                                PRPMappingEntry *mapping_entries,
                                                PRPMappingEntry **result_ptrs,
                                                uint32_t max_results,
                                                uint64_t *tensor_size)
{
    uint32_t hash_index = gpu_hash(tensor_ptr);

    GPUHashEntry &hash_entry = hash_table[hash_index];

    // printf("tensor_ptr is %lx, hash id is %u\n",tensor_ptr, hash_index);
    // 检查是否找到对应的tensor
    if (hash_entry.GPU_virtual_ptr != tensor_ptr || hash_entry.first_node == GPUMemoryMapper::INVALID_INDEX)
    {
        if (tensor_size)
        {
            *tensor_size = 0; // 未找到时设置tensor_size为0
        }
        return 0; // 未找到
    }

    // 如果找到了tensor，返回其tensor_size
    if (tensor_size)
    {
        *tensor_size = hash_entry.tensor_size;
    }

    uint32_t found_count = 0;
    uint32_t current_node = hash_entry.first_node;

    // 遍历映射链表，只返回mapping_entries的地址指针
    while (current_node != GPUMemoryMapper::INVALID_INDEX && found_count < max_results)
    {
        GPUMappingNode &node = mapping_nodes[current_node];

        if (node.entry_index != GPUMemoryMapper::INVALID_INDEX)
        {
            result_ptrs[found_count] = &mapping_entries[node.entry_index];
            printf("Found mapping entry %d:%p, prp1: %lx, prp2: %lx, len: %u\n",
                 found_count, result_ptrs[found_count], result_ptrs[found_count]->prp1,
                 result_ptrs[found_count]->prp2, result_ptrs[found_count]->data_length);
            found_count++;
        }

        current_node = node.next_node;
    }

    return found_count;
}

// === GPU mem mapper Implementation ===
GPUMemoryMapper::GPUMemoryMapper(int device_id)
    : d_mapping_entries_(nullptr), d_mapping_nodes_(nullptr), d_hash_table_(nullptr),
      d_free_entry_list_(nullptr), d_free_node_list_(nullptr),
      d_free_entry_count_(nullptr), d_free_node_count_(nullptr), d_device_view_(nullptr),
      is_initialized_(false), device_id_(device_id)
{
}

GPUMemoryMapper::~GPUMemoryMapper()
{
    cleanup();
}

bool GPUMemoryMapper::initialize()
{
    std::lock_guard<std::mutex> lock(mapper_mutex_);

    if (is_initialized_)
    {
        return true;
    }

    // 设置CUDA设备
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to set device %d: %s\n",
                       device_id_, cudaGetErrorString(err));
        return false;
    }

    // 分配映射条目数组
    err = cudaMalloc(&d_mapping_entries_, sizeof(PRPMappingEntry) * MAX_MAPPINGS);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate mapping entries: %s\n",
                       cudaGetErrorString(err));
        return false;
    }

    // 分配映射节点数组
    err = cudaMalloc(&d_mapping_nodes_, sizeof(GPUMappingNode) * MAX_MAPPING_NODES);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate mapping nodes: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 分配哈希表
    err = cudaMalloc(&d_hash_table_, sizeof(GPUHashEntry) * HASH_TABLE_SIZE);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate hash table: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 分配空闲条目列表
    err = cudaMalloc(&d_free_entry_list_, sizeof(uint32_t) * MAX_MAPPINGS);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate free entry list: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 分配空闲节点列表
    err = cudaMalloc(&d_free_node_list_, sizeof(uint32_t) * MAX_MAPPING_NODES);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate free node list: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 分配计数器
    err = cudaMalloc(&d_free_entry_count_, sizeof(uint32_t));
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate free entry counter: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    err = cudaMalloc(&d_free_node_count_, sizeof(uint32_t));
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate free node counter: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 分配Device侧视图结构体
    err = cudaMalloc(&d_device_view_, sizeof(GPUMemoryMapperDeviceView));
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate device view: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 初始化所有数据结构
    err = cudaMemset(d_hash_table_, 0, sizeof(GPUHashEntry) * HASH_TABLE_SIZE);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to initialize hash table: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    err = cudaMemset(d_mapping_entries_, 0, sizeof(PRPMappingEntry) * MAX_MAPPINGS);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to initialize mapping entries: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    err = cudaMemset(d_mapping_nodes_, 0, sizeof(GPUMappingNode) * MAX_MAPPING_NODES);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to initialize mapping nodes: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 初始化空闲条目列表
    std::vector<uint32_t> free_entry_indices(MAX_MAPPINGS);
    for (uint32_t i = 0; i < MAX_MAPPINGS; ++i)
    {
        free_entry_indices[i] = i;
    }

    err = cudaMemcpy(d_free_entry_list_, free_entry_indices.data(),
                     sizeof(uint32_t) * MAX_MAPPINGS, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to initialize free entry list: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 初始化空闲节点列表
    std::vector<uint32_t> free_node_indices(MAX_MAPPING_NODES);
    for (uint32_t i = 0; i < MAX_MAPPING_NODES; ++i)
    {
        free_node_indices[i] = i;
    }

    err = cudaMemcpy(d_free_node_list_, free_node_indices.data(),
                     sizeof(uint32_t) * MAX_MAPPING_NODES, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to initialize free node list: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 初始化计数器
    uint32_t initial_entry_count = MAX_MAPPINGS;
    err = cudaMemcpy(d_free_entry_count_, &initial_entry_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to initialize free entry counter: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    uint32_t initial_node_count = MAX_MAPPING_NODES;
    err = cudaMemcpy(d_free_node_count_, &initial_node_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to initialize free node counter: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    // 初始化Device侧视图结构体
    GPUMemoryMapperDeviceView host_view;
    host_view.d_mapping_entries = d_mapping_entries_;
    host_view.d_mapping_nodes = d_mapping_nodes_;
    host_view.d_hash_table = d_hash_table_;
    host_view.d_free_entry_list = d_free_entry_list_;
    host_view.d_free_node_list = d_free_node_list_;
    host_view.d_free_entry_count = d_free_entry_count_;
    host_view.d_free_node_count = d_free_node_count_;

    err = cudaMemcpy(d_device_view_, &host_view, sizeof(GPUMemoryMapperDeviceView), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to initialize device view: %s\n",
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }

    is_initialized_ = true;
    geminifs_debug("GPU Memory Mapper: Successfully initialized for device %d\n", device_id_);
    return true;
}

void GPUMemoryMapper::cleanup()
{
    if (d_mapping_entries_)
    {
        cudaFree(d_mapping_entries_);
        d_mapping_entries_ = nullptr;
    }

    if (d_mapping_nodes_)
    {
        cudaFree(d_mapping_nodes_);
        d_mapping_nodes_ = nullptr;
    }

    if (d_hash_table_)
    {
        cudaFree(d_hash_table_);
        d_hash_table_ = nullptr;
    }

    if (d_free_entry_list_)
    {
        cudaFree(d_free_entry_list_);
        d_free_entry_list_ = nullptr;
    }

    if (d_free_node_list_)
    {
        cudaFree(d_free_node_list_);
        d_free_node_list_ = nullptr;
    }

    if (d_free_entry_count_)
    {
        cudaFree(d_free_entry_count_);
        d_free_entry_count_ = nullptr;
    }

    if (d_free_node_count_)
    {
        cudaFree(d_free_node_count_);
        d_free_node_count_ = nullptr;
    }

    if (d_device_view_)
    {
        cudaFree(d_device_view_);
        d_device_view_ = nullptr;
    }

    is_initialized_ = false;
}

// CUDA kernel for batch adding mappings
__global__ void kernel_add_batch_mappings(uint64_t tensor_ptr,
                                          uint64_t tensor_size,
                                          PRPMappingEntry *new_mappings,
                                          uint32_t mapping_count,
                                          GPUHashEntry *hash_table,
                                          GPUMappingNode *mapping_nodes,
                                          PRPMappingEntry *mapping_entries,
                                          uint32_t *free_entry_list,
                                          uint32_t *free_node_list,
                                          uint32_t *free_entry_count,
                                          uint32_t *free_node_count,
                                          bool *success)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *success = false;

        // 检查是否有足够的空闲资源
        uint32_t available_entries = *free_entry_count;
        uint32_t available_nodes = *free_node_count;

        if (available_entries < mapping_count || available_nodes < mapping_count)
        {
            return; // 资源不足
        }

        // 分配资源
        uint32_t entry_start = atomicSub(free_entry_count, mapping_count);
        uint32_t node_start = atomicSub(free_node_count, mapping_count);

        printf("Available entries: %u, nodes: %u\n", available_entries, available_nodes);
        printf("Requested mappings: %u, entries start: %u, nodes start: %u\n",
               mapping_count, entry_start, node_start);
        if (entry_start < mapping_count || node_start < mapping_count)
        {
            // 恢复计数器并退出
            atomicAdd(free_entry_count, mapping_count);
            atomicAdd(free_node_count, mapping_count);
            return;
        }

        // 填充映射条目和节点
        uint32_t first_node_idx = GPUMemoryMapper::INVALID_INDEX;
        uint32_t last_node_idx = GPUMemoryMapper::INVALID_INDEX;
        for (uint32_t i = 0; i < mapping_count; ++i)
        {
            uint32_t entry_idx = free_entry_list[entry_start - 1 - i];
            uint32_t node_idx = free_node_list[node_start - 1 - i];

            // 填充条目
            mapping_entries[entry_idx] = new_mappings[i];

            // 构建链表（尾插入）
            mapping_nodes[node_idx] = GPUMappingNode(entry_idx);
            if (i == 0)
            {
                first_node_idx = node_idx;
                last_node_idx = node_idx;
            }
            else
            {
                mapping_nodes[last_node_idx].next_node = node_idx;
                last_node_idx = node_idx;
            }
        }

        // 更新哈希表
        uint32_t hash_index = gpu_hash(tensor_ptr);
        printf("Hash index for tensor 0x%lx: %u\n", tensor_ptr, hash_index);
        // 线性探测找到合适的槽位
        for (uint32_t i = 0; i < GPUMemoryMapper::HASH_TABLE_SIZE; ++i)
        {
            uint32_t probe_index = (hash_index + i) % GPUMemoryMapper::HASH_TABLE_SIZE;
            GPUHashEntry &entry = hash_table[probe_index];

            if (entry.GPU_virtual_ptr == 0)
            {
                // 空槽位，创建新条目
                entry.GPU_virtual_ptr = tensor_ptr;
                entry.first_node = first_node_idx;
                entry.mapping_count = mapping_count;
                entry.tensor_size = tensor_size;
                *success = true;
                break;
            }
            else if (entry.GPU_virtual_ptr == tensor_ptr)
            {
                // 已存在的tensor，追加到链表
                // 找到链表尾部
                uint32_t current = entry.first_node;
                while (mapping_nodes[current].next_node != GPUMemoryMapper::INVALID_INDEX)
                {
                    current = mapping_nodes[current].next_node;
                }
                mapping_nodes[current].next_node = first_node_idx;
                entry.mapping_count += mapping_count;
                *success = true;
                break;
            }
        }
    }
}

bool GPUMemoryMapper::addBatchMappings(uint64_t tensor_ptr, uint64_t tensor_size, const std::vector<PRPMappingEntry> &mappings)
{
    if (!is_initialized_)
    {
        geminifs_error("GPU Memory Mapper: Not initialized\n");
        return false;
    }

    if (mappings.empty())
    {
        geminifs_warn("GPU Memory Mapper: Empty mappings provided\n");
        return true;
    }

    std::lock_guard<std::mutex> lock(mapper_mutex_);

    // 分配设备端内存
    PRPMappingEntry *d_mappings;
    bool *d_success;

    cudaError_t err = cudaMalloc(&d_mappings, sizeof(PRPMappingEntry) * mappings.size());
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate device mappings: %s\n",
                       cudaGetErrorString(err));
        return false;
    }

    err = cudaMalloc(&d_success, sizeof(bool));
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to allocate success flag: %s\n",
                       cudaGetErrorString(err));
        cudaFree(d_mappings);
        return false;
    }

    // 复制映射到设备
    err = cudaMemcpy(d_mappings, mappings.data(), sizeof(PRPMappingEntry) * mappings.size(),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to copy mappings to device: %s\n",
                       cudaGetErrorString(err));
        cudaFree(d_mappings);
        cudaFree(d_success);
        return false;
    }

    // 启动内核
    kernel_add_batch_mappings<<<1, 1>>>(tensor_ptr, tensor_size, d_mappings, static_cast<uint32_t>(mappings.size()),
                                        d_hash_table_, d_mapping_nodes_, d_mapping_entries_,
                                        d_free_entry_list_, d_free_node_list_,
                                        d_free_entry_count_, d_free_node_count_, d_success);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Kernel execution failed: %s\n",
                       cudaGetErrorString(err));
        cudaFree(d_mappings);
        cudaFree(d_success);
        return false;
    }

    // 获取结果
    bool success;
    err = cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_mappings);
    cudaFree(d_success);

    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to copy result: %s\n",
                       cudaGetErrorString(err));
        return false;
    }

    if (success)
    {
        geminifs_debug("GPU Memory Mapper: Added %zu batch mappings for tensor 0x%lx\n",
                       mappings.size(), tensor_ptr);
    }
    else
    {
        geminifs_error("GPU Memory Mapper: Failed to add batch mappings - insufficient resources\n");
    }

    return success;
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GPUMemoryMapper::getStats() const
{
    if (!is_initialized_)
    {
        return std::make_tuple(0, 0, 0, 0);
    }

    uint32_t free_entry_count, free_node_count;

    cudaError_t err = cudaMemcpy(&free_entry_count, d_free_entry_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to get entry stats: %s\n", cudaGetErrorString(err));
        return std::make_tuple(0, 0, 0, 0);
    }

    err = cudaMemcpy(&free_node_count, d_free_node_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        geminifs_error("GPU Memory Mapper: Failed to get node stats: %s\n", cudaGetErrorString(err));
        return std::make_tuple(0, 0, 0, 0);
    }

    uint32_t used_entries = MAX_MAPPINGS - free_entry_count;
    uint32_t used_nodes = MAX_MAPPING_NODES - free_node_count;

    return std::make_tuple(used_entries, MAX_MAPPINGS, used_nodes, MAX_MAPPING_NODES);
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

    // Check 4K alignment for tensor pointer
    if (tensor_ptr % 4096 != 0)
    {
        geminifs_error("GPU Controller: Tensor pointer 0x%lx is not 4K aligned. Memory registration failed.\n", tensor_ptr);
        return false;
    }

    // Check 4K alignment for tensor size
    if (tensor_size % 4096 != 0)
    {
        geminifs_error("GPU Controller: Tensor size %zu is not 4K aligned. Memory registration failed.\n", tensor_size);
        return false;
    }

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

// === File Operations ===

// void* GPUController::openFile(GPUFileId gpu_file_id,
//                                 size_t file_size,
//                                 uint32_t o_flag, // device or host
//                                 const std::vector<nvme_ctrl_param>& nvme_params,
//                                 GPUFileManager& gpu_file_manager) {
// if (!isInitialized()) {
//     geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
//     return nullptr;
// }

// if (nvme_controllers_.empty() || nvme_params.empty() || nvme_controllers_.size() != nvme_params.size()) {
//     geminifs_error("GPU Controller: Mismatch between NVMe controllers and parameters, or none provided.\n");
//     return nullptr;
// }

// std::vector<std::string> all_nvme_file_names;
// std::vector<size_t> all_controller_indexes;
// std::vector<size_t> all_nvme_file_sizes; // per-link sizes

// size_t remaining_file_size = file_size;
// for (size_t i = 0; i < nvme_controllers_.size(); ++i) {
//     auto& nvme_controller = nvme_controllers_[i];
//     const auto& nvme_param = nvme_params[i];

//     size_t size_on_this_controller = std::min(remaining_file_size, file_size / nvme_controllers_.size());
//     remaining_file_size -= size_on_this_controller;

//     size_t max_io_size = nvme_controller->maxIOsize;
//     if (max_io_size == 0) {
//         geminifs_warn("GPU Controller: maxIOsize for controller %zu is zero, skipping.\n", i);
//         continue;
//     }

//     size_t remaining_size = size_on_this_controller;

//     while (remaining_size > 0) {
//         size_t chunk_size = std::min(remaining_size, max_io_size);

//         std::string filename = std::to_string(nvme_controller->next_nvme_file_id());

//         void* file_handle = nvme_controller->g_open(filename, chunk_size, o_flag);
//         if (!file_handle) {
//             geminifs_error("GPU Controller: Failed to create NVMe file '%s' on controller %zu\n", filename.c_str(), i);
//             // In a real scenario, we might want to clean up already created files
//             return nullptr;
//         }

//         all_nvme_file_names.push_back(filename);
//         all_controller_indexes.push_back(i);
//         all_nvme_file_sizes.push_back(chunk_size);

//         remaining_size -= chunk_size;
//     }
// }

// GPUFileDesc out_desc;
// GPU_File* new_gpu_file = gpu_file_manager.createGPUFile(
//     gpu_file_id,
//     file_size,
//     4096,
//     all_nvme_file_names,
//     all_controller_indexes,
//     all_nvme_file_sizes,
//     out_desc
// );

// if (new_gpu_file == nullptr) {
//     geminifs_error("GPU Controller: Failed to register GPUFile with GPUFileManager.\n");
//     // Cleanup logic for created files should be here
//     return nullptr;
// }

// // This is a placeholder, as the exact return type would depend on how the user wants to identify/use the opened file.
// // Returning the new_file_id cast to void* is one option.
// return reinterpret_cast<void*>(new_gpu_file);
// }

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

                geminifs_info("GPU Controller: Sub-slice[%zu]: local_offset=%zu, size=%zu, global_offset=%zu\n",
                              sub_index, current_offset, slice_size, current_offset);

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

    // 创建 geminifs_dma 结构
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
    // TODO (YJQ): move thie mapping to CPU
    if (!addPRPMappingsToGPU(dma_ctx))
    {
        delete dma_ctx;
        return nullptr;
    }

    return dma_ctx;
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

        geminifs_info("GPU Controller: Group[%zu] created %zu PRP mappings (%zu type2)\n",
                      group_idx, group.prp_mappings.size(), group_type2_count);
    }

    // 记录全局第三种类型的数量
    dma_ctx->type2_prp_count = total_type2_count;

    // 为第三种类型的PRP分配4KB对齐的GPU内存
    if (total_type2_count > 0)
    {
        size_t memory_size = total_type2_count * 4096; // 每个第三种类型需要4KB
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

        geminifs_info("GPU Controller: Initializing PRP entries for Group[%zu]: gpu_ptr=0x%lx\n",
                      group_idx, group.gpu_tensor_ptr);

        // 处理该group的每个子切片
        for (size_t sub_idx = 0; sub_idx < group.sub_slices.size(); sub_idx++)
        {
            const auto &sub_slice = group.sub_slices[sub_idx];
            size_t slice_size = sub_slice.size;
            size_t slice_pages = (slice_size + 4095) / 4096; // 切片需要的4K页数（向上取整）

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
                    size_t type2_page_index = type2_gpu_memory_offset / 4096;
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
                    4096,
                    cudaMemcpyHostToDevice);

                if (err != cudaSuccess)
                {
                    geminifs_error("GPU Controller: Failed to copy PRP list to GPU memory for Group[%zu] Sub-slice[%zu]: %s\n",
                                   group_idx, sub_idx, cudaGetErrorString(err));
                    return false;
                }

                geminifs_info("GPU Controller: Group[%zu] Sub-slice[%zu] Type2: PRP1=0x%lx, PRP2=0x%lx, filled %zu ioaddrs into GPU memory\n",
                              group_idx, sub_idx, group_entry.prp1, group_entry.prp2, remaining_pages);

                type2_gpu_memory_offset += 4096; // 移动到下一个4K空间
            }

            current_ioaddr_index += slice_pages; // 移动到下一个切片的起始ioaddr
        }

        geminifs_info("GPU Controller: Group[%zu] PRP initialization complete: %zu entries processed\n",
                      group_idx, group.prp_mappings.size());
    }

    geminifs_info("GPU Controller: Successfully configured PRP mappings for %zu granularity groups\n",
                  dma_ctx->granularity_groups.size());

    return true;
}

bool GPUController::addPRPMappingsToGPU(geminifs_dma *dma_ctx)
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

    geminifs_info("GPU Controller: Adding PRP mappings to GPU memory for %zu granularity groups\n",
                  dma_ctx->granularity_groups.size());

    // 遍历每个granularity group，将其PRP映射注册到GPU内存
    for (size_t group_idx = 0; group_idx < dma_ctx->granularity_groups.size(); group_idx++)
    {
        const auto &group = dma_ctx->granularity_groups[group_idx];

        if (group.prp_mappings.empty())
        {
            geminifs_warn("GPU Controller: Group[%zu] has no PRP mappings to register\n", group_idx);
            continue;
        }

        geminifs_info("GPU Controller: Registering %zu PRP mappings for Group[%zu] (gpu_ptr=0x%lx, size=%zu)\n",
                      group.prp_mappings.size(), group_idx, group.gpu_tensor_ptr, group.granularity_size);

        // 使用addBatchMappings批量添加该group的所有PRP映射
        bool success = memory_mapper_->addBatchMappings(group.gpu_tensor_ptr, group.granularity_size, group.prp_mappings);

        if (!success)
        {
            geminifs_error("GPU Controller: Failed to add batch mappings for Group[%zu]\n", group_idx);
            return false;
        }

        geminifs_info("GPU Controller: Successfully registered %zu PRP mappings for Group[%zu]\n",
                      group.prp_mappings.size(), group_idx);
    }

    geminifs_info("GPU Controller: Successfully registered PRP mappings for all %zu granularity groups\n",
                  dma_ctx->granularity_groups.size());

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
 * 子kernel：每个线程查询一个granularity地址的PRP映射
 */
__global__ void gpu_lookup_granularity_prp_mappings_kernel(GPUMemoryMapperDeviceView *device_view,
                                                           uint64_t base_tensor_ptr,
                                                           uint64_t granularity,
                                                           uint32_t total_granularities)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 确保线程ID在有效范围内
    if (tid >= total_granularities)
    {
        return;
    }

    // 计算当前线程负责的tensor地址
    uint64_t current_tensor_ptr = base_tensor_ptr + tid * granularity;

    // 优化：只存储mapping_entries的地址指针，大幅减少栈内存使用
    PRPMappingEntry *mapping_entry_ptrs[64]; // 每个granularity最多存储64个mapping_entries指针
    uint64_t found_tensor_size = 0;          // 用于接收tensor_size

    // 查询当前地址的PRP映射，返回mapping_entries的地址
    uint32_t found_count = gpu_lookup_all_prp_mappings(
        current_tensor_ptr,
        device_view->d_hash_table,
        device_view->d_mapping_nodes,
        device_view->d_mapping_entries,
        mapping_entry_ptrs,
        64,
        &found_tensor_size);

    // 打印查询结果（通过指针访问mapping_entries内容）
    if (found_count > 0)
    {
        // printf("Thread[%u] Tensor 0x%lx (offset: %lu): Found %u PRP mappings, tensor_size: %lu\n",
        //        tid, current_tensor_ptr, tid * granularity, found_count, found_tensor_size);

        // for (uint32_t i = 0; i < found_count; ++i) {
        //     PRPMappingEntry* entry_ptr = mapping_entry_ptrs[i];
        //     printf("  [%u.%u] GPU_VAddr: 0x%lx, Type: %u, PRP1: 0x%lx, PRP2: 0x%lx\n",
        //            tid, i, current_tensor_ptr, entry_ptr->transfer_type,
        //            entry_ptr->prp1, entry_ptr->prp2);
        // }
    }
    else
    {
        printf("Thread[%u] Tensor 0x%lx (offset: %lu): No PRP mappings found, tensor_size: %lu\n",
               tid, current_tensor_ptr, tid * granularity, found_tensor_size);
    }
}

/**
 * 主kernel：根据granularity计算所有需要查询的地址，并动态启动子kernel
 */
__global__ void gpu_debug_prp_mappings_kernel(GPUMemoryMapperDeviceView *device_view,
                                              uint64_t tensor_ptr,
                                              size_t tensor_size,
                                              uint64_t granularity)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        if (!device_view)
        {
            printf("GPU Debug: Device view is null\n");
            return;
        }

        // 验证tensor_size是granularity的整数倍
        if (granularity == 0 || tensor_size % granularity != 0)
        {
            printf("GPU Debug Error: tensor_size (%zu) must be a multiple of granularity (%lu)\n",
                   tensor_size, granularity);
            return;
        }

        // 计算需要查询的granularity数量
        uint32_t total_granularities = tensor_size / granularity;

        // printf("=== GPU PRP Mapping Debug (Dynamic Parallel) ===\n");
        // printf("Base Tensor Ptr: 0x%lx, Size: %zu, Granularity: %lu\n",
        //        tensor_ptr, tensor_size, granularity);
        // printf("Total granularities to query: %u\n", total_granularities);

        // 计算最优的线程块配置
        // 使用32线程为一个warp，尽量降低SM使用
        const uint32_t THREADS_PER_BLOCK = 32; // 一个warp
        uint32_t blocks_needed = (total_granularities + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // 限制最大block数量以避免过度使用SM资源
        const uint32_t MAX_BLOCKS = 64; // 限制最大block数量
        if (blocks_needed > MAX_BLOCKS)
        {
            printf("GPU Debug Warning: Need %u blocks, limiting to %u blocks\n",
                   blocks_needed, MAX_BLOCKS);
            blocks_needed = MAX_BLOCKS;
        }

        // printf("Launching dynamic kernel with %u blocks × %u threads = %u total threads\n",
        //        blocks_needed, THREADS_PER_BLOCK, blocks_needed * THREADS_PER_BLOCK);

        // 动态启动子kernel进行并行查询
        gpu_lookup_granularity_prp_mappings_kernel<<<blocks_needed, THREADS_PER_BLOCK>>>(
            device_view,
            tensor_ptr,
            granularity,
            total_granularities);

        // 在device代码中，我们不能显式同步子kernel
        // 子kernel会自动完成并返回结果
        // printf("GPU Debug: Launched dynamic kernel with %u blocks\n", blocks_needed);
        // printf("=== End PRP Mapping Debug ===\n");
    }
}

/**
 * Host端函数用于调用GPU kernel查询和打印PRP映射
 */
void debug_prp_mappings_from_gpu(GPUMemoryMapper *mapper,
                                 uint64_t tensor_ptr,
                                 size_t tensor_size,
                                 uint64_t granularity)
{
    if (!mapper)
    {
        geminifs_error("Debug PRP Mappings: Invalid mapper\n");
        return;
    }

    GPUMemoryMapperDeviceView *device_view = mapper->getDeviceViewPtr();
    if (!device_view)
    {
        geminifs_error("Debug PRP Mappings: Invalid device view\n");
        return;
    }

    // 启动kernel - 使用单个线程块和单个线程
    gpu_debug_prp_mappings_kernel<<<1, 1>>>(device_view, tensor_ptr, tensor_size, granularity);

    // 同步等待kernel完成
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        geminifs_error("Debug PRP Mappings: Kernel execution failed: %s\n",
                       cudaGetErrorString(err));
    }
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
        printf("Safety Check Error: NVMe_File pointer is null\n");
        return false;
    }

    // 获取文件大小（通过公共方法）
    uint64_t file_max_size = d_fd->get_file_size();
    if (file_max_size == 0)
    {
        printf("Safety Check Error: Could not get file size (hdr might be null)\n");
        return false;
    }

    // 检查tensor_size是否足够容纳要读取的数据
    if (tensor_size < len)
    {
        printf("Safety Check Error: tensor_size (%lu) < read length (%zu)\n",
               tensor_size, len);
        return false;
    }

    // 检查文件访问边界
    if (offset + len > file_max_size)
    {
        printf("Safety Check Error: access beyond file boundary - "
               "offset (%zu) + len (%zu) = %zu > file_max_size (%lu)\n",
               offset, len, offset + len, file_max_size);
        return false;
    }

    // 检查offset是否有效
    if (offset >= file_max_size)
    {
        printf("Safety Check Error: offset (%zu) >= file_max_size (%lu)\n",
               offset, file_max_size);
        return false;
    }

    // 检查len是否为0
    if (len == 0)
    {
        printf("Safety Check Warning: read length is 0\n");
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
        printf("Safety Check Error: length (%zu) is not 4K aligned (remainder: %zu)\n",
               len, len % 4096);
        return false;
    }

    // // 打印成功信息
    // printf("Safety Check Passed: tensor_size=%lu, offset=%zu, len=%zu, file_max_size=%lu (all 4K aligned)\n",
    //        tensor_size, offset, len, file_max_size);

    return true;
}

__global__ void GPU_Read_kernel(NVMe_File *d_fd,
                                uint64_t tensor_ptr,
                                size_t offset,
                                size_t len,
                                GPUMemoryMapperDeviceView *device_view)
{

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        if (!device_view)
        {
            printf("GPU Debug: Device view is null\n");
            return;
        }

        PRPMappingEntry *mapping_entry_ptrs[32]; // 每个granularity最多存储32个mapping_entries指针
        uint64_t found_tensor_size = 0;          // 用于接收tensor_size

        // 查询当前地址的PRP映射，返回mapping_entries的地址
        uint32_t found_count = gpu_lookup_all_prp_mappings(
            tensor_ptr,
            device_view->d_hash_table,
            device_view->d_mapping_nodes,
            device_view->d_mapping_entries,
            mapping_entry_ptrs,
            32,
            &found_tensor_size);

        if (found_count == 0)
        {
            printf("GPU_Read_kernel: No PRP mappings found for tensor 0x%lx\n", tensor_ptr);
            return;
        }

        if (!check_file_access_safety(d_fd, found_tensor_size, offset, len))
        {
            printf("GPU_Read_kernel: Safety check failed, aborting read operation\n");
            return;
        }

        // 计算线程块配置
        const uint32_t THREADS_PER_BLOCK = 32; // 一个warp
        uint32_t blocks_needed = (found_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // 限制最大block数量以避免过度使用SM资源
        const uint32_t MAX_BLOCKS = 64;
        if (blocks_needed > MAX_BLOCKS)
        {
            printf("GPU_Read_kernel Warning: Need %u blocks, limiting to %u blocks\n",
                   blocks_needed, MAX_BLOCKS);
            blocks_needed = MAX_BLOCKS;
        }
        // 动态并行调用批量NVMe读取kernel，传递必要参数让子kernel自行查询
        nvme_batch_read_kernel_v2<<<blocks_needed, THREADS_PER_BLOCK>>>(
            d_fd, tensor_ptr, found_count, offset, device_view);
    }
}

/**
 * 批量NVMe读取kernel V2：每个线程直接查询自己的PRP映射条目
 * 解决动态并行内存访问问题，避免传递指针数组
 * @param d_fd NVMe文件描述符
 * @param tensor_ptr GPU tensor指针
 * @param total_count 总的映射条目数量
 * @param base_file_offset 文件基础偏移量
 * @param device_view GPU内存映射器的设备视图
 */
__global__ void nvme_batch_read_kernel_v2(NVMe_File *d_fd,
                                          uint64_t tensor_ptr,
                                          uint32_t total_count,
                                          size_t base_file_offset,
                                          GPUMemoryMapperDeviceView *device_view)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 确保线程ID在有效范围内
    if (tid >= total_count)
    {
        return;
    }

    // 每个线程重新查询PRP映射，但只取自己需要的那一个
    PRPMappingEntry *temp_mapping_ptrs[64]; // 栈数组，足够大
    uint64_t found_tensor_size = 0;

    // 查询当前tensor的所有PRP映射
    uint32_t found_count = gpu_lookup_all_prp_mappings(
        tensor_ptr,
        device_view->d_hash_table,
        device_view->d_mapping_nodes,
        device_view->d_mapping_entries,
        temp_mapping_ptrs,
        64,
        &found_tensor_size);

    // 检查是否有足够的映射条目
    if (tid >= found_count)
    {
        printf("Thread[%u]: Index exceeds found mappings (%u)\n", tid, found_count);
        return;
    }

    // 获取当前线程对应的PRP映射条目
    PRPMappingEntry *entry_ptr = temp_mapping_ptrs[tid];
    if (!entry_ptr)
    {
        printf("Thread[%u]: Null mapping entry pointer\n", tid);
        return;
    }

    // printf("Thread[%u]: Found mapping entry: Type: %u, PRP1: 0x%lx, PRP2: 0x%lx, tensor_offset: %lu, len : %lu\n",
    //        tid, entry_ptr->transfer_type, entry_ptr->prp1, entry_ptr->prp2, entry_ptr->tensor_offset, entry_ptr->data_length);

    // 从PRP映射条目中获取必要信息
    uint64_t prp1 = entry_ptr->prp1;
    uint64_t prp2 = entry_ptr->prp2;
    uint32_t data_length = entry_ptr->data_length;
    uint64_t tensor_offset = entry_ptr->tensor_offset;

    // 计算当前切片的文件偏移量
    // base_file_offset: 对应granularity在文件中的起始偏移
    // tensor_offset: 该PRP entry在granularity内的本地偏移
    size_t slice_file_offset = base_file_offset + tensor_offset;

    // 调用NVMe读取函数
    nvme_controller_g_read(d_fd, prp1, prp2, slice_file_offset, data_length);

    // 打印调试信息
    // printf("Thread[%u]: NVMe read completed - tensor_offset=%lu, file_offset=%zu, len=%u, prp1=0x%lx, prp2=0x%lx\n",
    //        tid, tensor_offset, slice_file_offset, data_length, prp1, prp2);
}

/**
 * 批量NVMe写入kernel：每个线程处理一个PRP映射条目
 * @param d_fd NVMe文件描述符
 * @param mapping_entry_ptrs PRP映射条目指针数组
 * @param found_count 找到的映射条目数量
 * @param base_file_offset 文件基础偏移量
 */
__global__ void nvme_batch_write_kernel_v2(NVMe_File *d_fd,
                                           uint64_t tensor_ptr,
                                           uint32_t total_count,
                                           size_t base_file_offset,
                                           GPUMemoryMapperDeviceView *device_view)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 确保线程ID在有效范围内
    if (tid >= total_count)
    {
        return;
    }

    // 每个线程重新查询PRP映射，但只取自己需要的那一个
    PRPMappingEntry *temp_mapping_ptrs[64]; // 栈数组，足够大
    uint64_t found_tensor_size = 0;

    // 查询当前tensor的所有PRP映射
    uint32_t found_count = gpu_lookup_all_prp_mappings(
        tensor_ptr,
        device_view->d_hash_table,
        device_view->d_mapping_nodes,
        device_view->d_mapping_entries,
        temp_mapping_ptrs,
        64,
        &found_tensor_size);

    // 检查是否有足够的映射条目
    if (tid >= found_count)
    {
        printf("Thread[%u]: Index exceeds found mappings (%u)\n", tid, found_count);
        return;
    }

    // 获取当前线程对应的PRP映射条目
    PRPMappingEntry *entry_ptr = temp_mapping_ptrs[tid];
    if (!entry_ptr)
    {
        printf("Thread[%u]: Null mapping entry pointer\n", tid);
        return;
    }

    printf("Thread[%u]: Found mapping entry: Type: %u, PRP1: 0x%lx, PRP2: 0x%lx, tensor_offset: %lu\n",
           tid, entry_ptr->transfer_type, entry_ptr->prp1, entry_ptr->prp2, entry_ptr->tensor_offset);

    // 从PRP映射条目中获取必要信息
    uint64_t prp1 = entry_ptr->prp1;
    uint64_t prp2 = entry_ptr->prp2;
    uint32_t data_length = entry_ptr->data_length;
    uint64_t tensor_offset = entry_ptr->tensor_offset;

    // 计算当前切片的文件偏移量
    // base_file_offset: 对应granularity在文件中的起始偏移
    // tensor_offset: 该PRP entry在granularity内的本地偏移
    size_t slice_file_offset = base_file_offset + tensor_offset;

    // 调用NVMe读取函数
    nvme_controller_g_write(d_fd, prp1, prp2, slice_file_offset, data_length);

    // 打印调试信息
    printf("Thread[%u]: NVMe write completed - tensor_offset=%lu, file_offset=%zu, len=%u, prp1=0x%lx, prp2=0x%lx\n",
           tid, tensor_offset, slice_file_offset, data_length, prp1, prp2);
}

/**
 * GPU写入kernel：查询PRP映射并动态并行发起NVMe IO
 * @param d_fd NVMe文件描述符
 * @param tensor_ptr GPU tensor指针
 * @param offset 文件偏移量
 * @param len 写入长度
 * @param device_view GPU内存映射器的设备视图
 */
__global__ void GPU_Write_kernel(NVMe_File *d_fd,
                                 uint64_t tensor_ptr,
                                 size_t offset,
                                 size_t len,
                                 GPUMemoryMapperDeviceView *device_view)
{

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        if (!device_view)
        {
            printf("GPU Write: Device view is null\n");
            return;
        }

        PRPMappingEntry *mapping_entry_ptrs[32]; // 每个granularity最多存储32个mapping_entries指针
        uint64_t found_tensor_size = 0;          // 用于接收tensor_size

        // 查询当前地址的PRP映射，返回mapping_entries的地址
        uint32_t found_count = gpu_lookup_all_prp_mappings(
            tensor_ptr,
            device_view->d_hash_table,
            device_view->d_mapping_nodes,
            device_view->d_mapping_entries,
            mapping_entry_ptrs,
            32,
            &found_tensor_size);

        if (found_count == 0)
        {
            printf("GPU_Write_kernel: No PRP mappings found for tensor 0x%lx\n", tensor_ptr);
            return;
        }

        if (!check_file_access_safety(d_fd, found_tensor_size, offset, len))
        {
            printf("GPU_Write_kernel: Safety check failed, aborting write operation\n");
            return;
        }

        // 计算线程块配置
        const uint32_t THREADS_PER_BLOCK = 32; // 一个warp
        uint32_t blocks_needed = (found_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // 限制最大block数量以避免过度使用SM资源
        const uint32_t MAX_BLOCKS = 64;
        if (blocks_needed > MAX_BLOCKS)
        {
            printf("GPU_Write_kernel Warning: Need %u blocks, limiting to %u blocks\n",
                   blocks_needed, MAX_BLOCKS);
            blocks_needed = MAX_BLOCKS;
        }

        printf("GPU_Write_kernel: Launching batch NVMe write with %u blocks × %u threads for %u mappings\n",
               blocks_needed, THREADS_PER_BLOCK, found_count);

        // 动态并行调用批量NVMe写入kernel V2，传递必要参数让子kernel自行查询
        nvme_batch_write_kernel_v2<<<blocks_needed, THREADS_PER_BLOCK>>>(
            d_fd, tensor_ptr, found_count, offset, device_view);

        printf("GPU_Write_kernel: Batch NVMe write kernel launched\n");
    }
}

__global__ void GPU_Read_kernel_multi(GPUIoContext *io_ctx,
                                      uint64_t tensor_ptr,
                                      size_t offset,
                                      size_t len,
                                      GPUMemoryMapperDeviceView *device_view)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("GPU_Read_kernel_multi: Launching with tensor_ptr=0x%lx, offset=%lu, len=%lu\n",
               tensor_ptr, (uint64_t)offset, (uint64_t)len);

        if (!device_view)
        {
            printf("GPU_Write_kernel_multi: invalid device_view\n");
            return;
        }
        uint64_t tensor_size = 0;
        uint32_t found_count = gpu_lookup_all_prp_mappings(
            tensor_ptr,
            device_view->d_hash_table,
            device_view->d_mapping_nodes,
            device_view->d_mapping_entries,
            io_ctx->mapping_entries,
            1024,
            &tensor_size);
        if (found_count == 0)
        {
            printf("GPU_Read_kernel_multi: no mappings for tensor 0x%lx\n", tensor_ptr);
            return;
        }

        // 启动 v3 批量 kernel，让每个线程处理一个映射条目，并按 tid%num_fds 选择 fd
        const uint32_t THREADS_PER_BLOCK = 32;
        uint32_t blocks = (found_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        printf("GPU_Read_kernel_multi: Launching nvme_batch_read_kernel_v3 with blocks=%u, threads_per_block=%u\n",
               blocks, THREADS_PER_BLOCK);
        __threadfence();

        nvme_batch_read_kernel_v3<<<blocks, THREADS_PER_BLOCK>>>(io_ctx, tensor_ptr, found_count, offset);

        printf("GPU_Read_kernel_multi: nvme_batch_read_kernel_v3 launched\n");
    }
}

__global__ void GPU_Write_kernel_multi(GPUIoContext *io_ctx,
                                       uint64_t tensor_ptr,
                                       size_t offset,
                                       size_t len,
                                       GPUMemoryMapperDeviceView *device_view)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("GPU_Write_kernel_multi: Launching with tensor_ptr=0x%lx, offset=%lu, len=%lu\n",
               tensor_ptr, offset, len);

        if (!device_view)
        {
            printf("GPU_Write_kernel_multi: invalid device_view\n");
            return;
        }
        uint64_t tensor_size = 0;
        uint32_t found_count = gpu_lookup_all_prp_mappings(
            tensor_ptr,
            device_view->d_hash_table,
            device_view->d_mapping_nodes,
            device_view->d_mapping_entries,
            io_ctx->mapping_entries,
            1024,
            &tensor_size);
        printf("GPU_Write_kernel_multi: found_count=%u, tensor_size=%lu\n", found_count, tensor_size);

        if (found_count == 0)
        {
            printf("GPU_Write_kernel_multi: no mappings for tensor 0x%lx\n", tensor_ptr);
            return;
        }
        const uint32_t THREADS_PER_BLOCK = 32;
        uint32_t blocks = (found_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        printf("GPU_Write_kernel_multi: Launching nvme_batch_write_kernel_v3 with blocks=%u, threads_per_block=%u\n",
               blocks, THREADS_PER_BLOCK);
        __threadfence();

        nvme_batch_write_kernel_v3<<<blocks, THREADS_PER_BLOCK>>>(io_ctx, tensor_ptr, found_count, offset);

        printf("GPU_Write_kernel_multi: nvme_batch_write_kernel_v3 launched\n");
    }
}

__global__ void nvme_batch_read_kernel_v3(GPUIoContext *io_ctx,
                                          uint64_t tensor_ptr,
                                          uint32_t total_count,
                                          size_t base_file_offset)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= total_count)
    {
        printf("WARNING, read kernel tid %d >= total count %d !\n", tid, total_count);
        return;
    }

    if (io_ctx->num_files == 0)
    {
        printf("WARNING, read kernel num_files=0\n");
        return;
    }

    // 获取本线程的映射条目
    PRPMappingEntry *entry = io_ctx->mapping_entries[tid];
    if (!entry)
    {
        printf("WARNING, read kernel tid %d no prp entry!\n", tid);
        return;
    }

    uint64_t thread_logical_offset = base_file_offset + tid * entry->data_length;
    uint64_t fd_idx = (thread_logical_offset % (io_ctx->num_files * entry->data_length)) / entry->data_length;
    uint64_t thread_physical_offset = (thread_logical_offset / (io_ctx->num_files * entry->data_length)) * entry->data_length + fd_idx * entry->data_length;
    NVMe_File *file = io_ctx->nvme_files[fd_idx];

    printf("nvme_batch_read_kernel_v3: tid=%u, fd_idx=%u, prp1=0x%lx, prp2=0x%lx, file_offset=%lu, len=%u\n",
           tid, fd_idx, entry->prp1, entry->prp2, thread_physical_offset, entry->data_length);

    nvme_controller_g_read(file, entry->prp1, entry->prp2, thread_physical_offset, entry->data_length);
}

__global__ void nvme_batch_write_kernel_v3(GPUIoContext *io_ctx,
                                           uint64_t tensor_ptr,
                                           uint32_t total_count,
                                           size_t base_file_offset)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= total_count)
    {
        printf("WARNING, write kernel tid %d >= total count %d !\n", tid, total_count);
        return;
    }

    if (io_ctx->num_files == 0)
    {
        printf("WARNING, write kernel invalid device_view or num_files=0\n");
        return;
    }

    PRPMappingEntry *entry = io_ctx->mapping_entries[tid];
    if (!entry)
    {
        printf("WARNING, write kernel tid %d no prp entry!\n", tid);
        return;
    }

    uint64_t thread_logical_offset = base_file_offset + tid * entry->data_length;
    uint64_t fd_idx = (thread_logical_offset % (io_ctx->num_files * entry->data_length)) / entry->data_length;
    uint64_t thread_physical_offset = (thread_logical_offset / (io_ctx->num_files * entry->data_length)) * entry->data_length + fd_idx * entry->data_length;
    NVMe_File *file = io_ctx->nvme_files[fd_idx];

    printf("nvme_batch_write_kernel_v3: tid=%u, fd_idx=%u, prp1=0x%lx, prp2=0x%lx, file_offset=%lu, len=%u\n",
           tid, fd_idx, entry->prp1, entry->prp2, thread_physical_offset, entry->data_length);

    nvme_controller_g_write(file, entry->prp1, entry->prp2, thread_physical_offset, entry->data_length);
}