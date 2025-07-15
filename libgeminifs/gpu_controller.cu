#include "geminifs_helper.h"
#include "geminifs_mem.h"
#include "buffer.h"
#include <cuda_runtime.h>
#include <cassert>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include "gpu_controller.cuh"



__device__ uint32_t gpu_lookup_all_prp_mappings(uint64_t tensor_ptr,
                                                GPUHashEntry* hash_table,
                                                GPUMappingNode* mapping_nodes,
                                                PRPMappingEntry* mapping_entries,
                                                PRPMappingEntry* results,
                                                uint32_t max_results) {
    uint32_t hash_index = gpu_hash(tensor_ptr);
    GPUHashEntry& hash_entry = hash_table[hash_index];
    
    // 检查是否找到对应的tensor
    if (hash_entry.GPU_virtual_ptr != tensor_ptr || hash_entry.first_node == GPUMemoryMapper::INVALID_INDEX) {
        return 0; // 未找到
    }
    
    uint32_t found_count = 0;
    uint32_t current_node = hash_entry.first_node;
    
    // 遍历映射链表
    while (current_node != GPUMemoryMapper::INVALID_INDEX && found_count < max_results) {
        GPUMappingNode& node = mapping_nodes[current_node];
        
        if (node.entry_index != GPUMemoryMapper::INVALID_INDEX) {
            results[found_count] = mapping_entries[node.entry_index];
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
      d_free_entry_count_(nullptr), d_free_node_count_(nullptr),
      is_initialized_(false), device_id_(device_id) {
}

GPUMemoryMapper::~GPUMemoryMapper() {
    cleanup();
}

bool GPUMemoryMapper::initialize() {
    std::lock_guard<std::mutex> lock(mapper_mutex_);
    
    if (is_initialized_) {
        return true;
    }
    
    // 设置CUDA设备
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to set device %d: %s\n", 
                       device_id_, cudaGetErrorString(err));
        return false;
    }
    
    // 分配映射条目数组
    err = cudaMalloc(&d_mapping_entries_, sizeof(PRPMappingEntry) * MAX_MAPPINGS);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate mapping entries: %s\n", 
                       cudaGetErrorString(err));
        return false;
    }
    
    // 分配映射节点数组
    err = cudaMalloc(&d_mapping_nodes_, sizeof(GPUMappingNode) * MAX_MAPPING_NODES);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate mapping nodes: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    // 分配哈希表
    err = cudaMalloc(&d_hash_table_, sizeof(GPUHashEntry) * HASH_TABLE_SIZE);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate hash table: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    // 分配空闲条目列表
    err = cudaMalloc(&d_free_entry_list_, sizeof(uint32_t) * MAX_MAPPINGS);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate free entry list: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    // 分配空闲节点列表
    err = cudaMalloc(&d_free_node_list_, sizeof(uint32_t) * MAX_MAPPING_NODES);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate free node list: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    // 分配计数器
    err = cudaMalloc(&d_free_entry_count_, sizeof(uint32_t));
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate free entry counter: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    err = cudaMalloc(&d_free_node_count_, sizeof(uint32_t));
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate free node counter: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    // 初始化所有数据结构
    err = cudaMemset(d_hash_table_, 0, sizeof(GPUHashEntry) * HASH_TABLE_SIZE);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to initialize hash table: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    err = cudaMemset(d_mapping_entries_, 0, sizeof(PRPMappingEntry) * MAX_MAPPINGS);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to initialize mapping entries: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    err = cudaMemset(d_mapping_nodes_, 0, sizeof(GPUMappingNode) * MAX_MAPPING_NODES);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to initialize mapping nodes: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    // 初始化空闲条目列表
    std::vector<uint32_t> free_entry_indices(MAX_MAPPINGS);
    for (uint32_t i = 0; i < MAX_MAPPINGS; ++i) {
        free_entry_indices[i] = i;
    }
    
    err = cudaMemcpy(d_free_entry_list_, free_entry_indices.data(), 
                     sizeof(uint32_t) * MAX_MAPPINGS, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to initialize free entry list: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    // 初始化空闲节点列表
    std::vector<uint32_t> free_node_indices(MAX_MAPPING_NODES);
    for (uint32_t i = 0; i < MAX_MAPPING_NODES; ++i) {
        free_node_indices[i] = i;
    }
    
    err = cudaMemcpy(d_free_node_list_, free_node_indices.data(), 
                     sizeof(uint32_t) * MAX_MAPPING_NODES, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to initialize free node list: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    // 初始化计数器
    uint32_t initial_entry_count = MAX_MAPPINGS;
    err = cudaMemcpy(d_free_entry_count_, &initial_entry_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to initialize free entry counter: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    uint32_t initial_node_count = MAX_MAPPING_NODES;
    err = cudaMemcpy(d_free_node_count_, &initial_node_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to initialize free node counter: %s\n", 
                       cudaGetErrorString(err));
        cleanup();
        return false;
    }
    
    is_initialized_ = true;
    geminifs_debug("GPU Memory Mapper: Successfully initialized for device %d\n", device_id_);
    return true;
}

void GPUMemoryMapper::cleanup() {
    if (d_mapping_entries_) {
        cudaFree(d_mapping_entries_);
        d_mapping_entries_ = nullptr;
    }
    
    if (d_mapping_nodes_) {
        cudaFree(d_mapping_nodes_);
        d_mapping_nodes_ = nullptr;
    }
    
    if (d_hash_table_) {
        cudaFree(d_hash_table_);
        d_hash_table_ = nullptr;
    }
    
    if (d_free_entry_list_) {
        cudaFree(d_free_entry_list_);
        d_free_entry_list_ = nullptr;
    }
    
    if (d_free_node_list_) {
        cudaFree(d_free_node_list_);
        d_free_node_list_ = nullptr;
    }
    
    if (d_free_entry_count_) {
        cudaFree(d_free_entry_count_);
        d_free_entry_count_ = nullptr;
    }
    
    if (d_free_node_count_) {
        cudaFree(d_free_node_count_);
        d_free_node_count_ = nullptr;
    }
    
    is_initialized_ = false;
}

// CUDA kernel for adding mapping
__global__ void kernel_add_mapping(uint64_t tensor_ptr, uint32_t transfer_type,
                                  uint64_t prp1, uint64_t prp2,
                                  GPUHashEntry* hash_table,
                                  GPUMappingNode* mapping_nodes,
                                  PRPMappingEntry* mapping_entries,
                                  uint32_t* free_entry_list,
                                  uint32_t* free_node_list,
                                  uint32_t* free_entry_count,
                                  uint32_t* free_node_count,
                                  bool* success) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *success = false;
        
        // 获取空闲条目索引
        uint32_t old_entry_count = atomicSub(free_entry_count, 1);
        if (old_entry_count == 0) {
            atomicAdd(free_entry_count, 1); // 恢复计数
            return;
        }
        
        // 获取空闲节点索引
        uint32_t old_node_count = atomicSub(free_node_count, 1);
        if (old_node_count == 0) {
            atomicAdd(free_entry_count, 1); // 恢复条目计数
            atomicAdd(free_node_count, 1);  // 恢复节点计数
            return;
        }
        
        uint32_t entry_index = free_entry_list[old_entry_count - 1];
        uint32_t node_index = free_node_list[old_node_count - 1];
        
        // 填充映射条目
        mapping_entries[entry_index] = PRPMappingEntry(transfer_type, prp1, prp2);
        
        // 填充映射节点
        mapping_nodes[node_index] = GPUMappingNode(entry_index);
        
        // 计算哈希索引
        uint32_t hash_index = gpu_hash(tensor_ptr);
        GPUHashEntry& hash_entry = hash_table[hash_index];
        
        if (hash_entry.GPU_virtual_ptr == 0) {
            // 新的tensor，创建新的哈希条目
            hash_entry.GPU_virtual_ptr = tensor_ptr;
            hash_entry.first_node = node_index;
            hash_entry.mapping_count = 1;
        } else if (hash_entry.GPU_virtual_ptr == tensor_ptr) {
            // 已存在的tensor，添加到链表头
            mapping_nodes[node_index].next_node = hash_entry.first_node;
            hash_entry.first_node = node_index;
            hash_entry.mapping_count++;
        } else {
            // 哈希冲突，使用线性探测
            for (uint32_t i = 1; i < GPUMemoryMapper::HASH_TABLE_SIZE; ++i) {
                uint32_t probe_index = (hash_index + i) % GPUMemoryMapper::HASH_TABLE_SIZE;
                GPUHashEntry& probe_entry = hash_table[probe_index];
                
                if (probe_entry.GPU_virtual_ptr == 0) {
                    // 找到空槽位
                    probe_entry.GPU_virtual_ptr = tensor_ptr;
                    probe_entry.first_node = node_index;
                    probe_entry.mapping_count = 1;
                    break;
                } else if (probe_entry.GPU_virtual_ptr == tensor_ptr) {
                    // 找到相同tensor的条目
                    mapping_nodes[node_index].next_node = probe_entry.first_node;
                    probe_entry.first_node = node_index;
                    probe_entry.mapping_count++;
                    break;
                }
            }
        }
        
        *success = true;
    }
}

// CUDA kernel for batch adding mappings
__global__ void kernel_add_batch_mappings(uint64_t tensor_ptr, 
                                          PRPMappingEntry* new_mappings,
                                          uint32_t mapping_count,
                                          GPUHashEntry* hash_table,
                                          GPUMappingNode* mapping_nodes,
                                          PRPMappingEntry* mapping_entries,
                                          uint32_t* free_entry_list,
                                          uint32_t* free_node_list,
                                          uint32_t* free_entry_count,
                                          uint32_t* free_node_count,
                                          bool* success) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *success = false;
        
        // 检查是否有足够的空闲资源
        uint32_t available_entries = *free_entry_count;
        uint32_t available_nodes = *free_node_count;
        
        if (available_entries < mapping_count || available_nodes < mapping_count) {
            return; // 资源不足
        }
        
        // 分配资源
        uint32_t entry_start = atomicSub(free_entry_count, mapping_count);
        uint32_t node_start = atomicSub(free_node_count, mapping_count);
        
        if (entry_start < mapping_count || node_start < mapping_count) {
            // 恢复计数器并退出
            atomicAdd(free_entry_count, mapping_count);
            atomicAdd(free_node_count, mapping_count);
            return;
        }
        
        // 填充映射条目和节点
        uint32_t first_node_idx = GPUMemoryMapper::INVALID_INDEX;
        for (uint32_t i = 0; i < mapping_count; ++i) {
            uint32_t entry_idx = free_entry_list[entry_start - 1 - i];
            uint32_t node_idx = free_node_list[node_start - 1 - i];
            
            // 填充条目
            mapping_entries[entry_idx] = new_mappings[i];
            
            // 构建链表
            mapping_nodes[node_idx] = GPUMappingNode(entry_idx);
            if (i == 0) {
                first_node_idx = node_idx;
            } else {
                mapping_nodes[node_idx].next_node = first_node_idx;
                first_node_idx = node_idx;
            }
        }
        
        // 更新哈希表
        uint32_t hash_index = gpu_hash(tensor_ptr);
        
        // 线性探测找到合适的槽位
        for (uint32_t i = 0; i < GPUMemoryMapper::HASH_TABLE_SIZE; ++i) {
            uint32_t probe_index = (hash_index + i) % GPUMemoryMapper::HASH_TABLE_SIZE;
            GPUHashEntry& entry = hash_table[probe_index];
            
            if (entry.GPU_virtual_ptr == 0) {
                // 空槽位，创建新条目
                entry.GPU_virtual_ptr = tensor_ptr;
                entry.first_node = first_node_idx;
                entry.mapping_count = mapping_count;
                *success = true;
                break;
            } else if (entry.GPU_virtual_ptr == tensor_ptr) {
                // 已存在的tensor，追加到链表
                // 找到链表尾部
                uint32_t current = entry.first_node;
                while (mapping_nodes[current].next_node != GPUMemoryMapper::INVALID_INDEX) {
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

bool GPUMemoryMapper::addMapping(uint64_t tensor_ptr, uint32_t transfer_type, uint64_t prp1, uint64_t prp2) {
    if (!is_initialized_) {
        geminifs_error("GPU Memory Mapper: Not initialized\n");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mapper_mutex_);
    
    // 分配设备端成功标志
    bool* d_success;
    cudaError_t err = cudaMalloc(&d_success, sizeof(bool));
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate success flag: %s\n", 
                       cudaGetErrorString(err));
        return false;
    }
    
    // 启动内核
    kernel_add_mapping<<<1, 1>>>(tensor_ptr, transfer_type, prp1, prp2,
                                 d_hash_table_, d_mapping_nodes_, d_mapping_entries_,
                                 d_free_entry_list_, d_free_node_list_,
                                 d_free_entry_count_, d_free_node_count_, d_success);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Kernel execution failed: %s\n", 
                       cudaGetErrorString(err));
        cudaFree(d_success);
        return false;
    }
    
    // 获取结果
    bool success;
    err = cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_success);
    
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to copy result: %s\n", 
                       cudaGetErrorString(err));
        return false;
    }
    
    if (success) {
        geminifs_debug("GPU Memory Mapper: Added mapping for tensor 0x%lx -> transfer_type %u, PRP1: 0x%lx, PRP2: 0x%lx\n",
                       tensor_ptr, transfer_type, prp1, prp2);
    } else {
        geminifs_error("GPU Memory Mapper: Failed to add mapping - no free space\n");
    }
    
    return success;
}

bool GPUMemoryMapper::addBatchMappings(uint64_t tensor_ptr, const std::vector<PRPMappingEntry>& mappings) {
    if (!is_initialized_) {
        geminifs_error("GPU Memory Mapper: Not initialized\n");
        return false;
    }
    
    if (mappings.empty()) {
        geminifs_warn("GPU Memory Mapper: Empty mappings provided\n");
        return true;
    }
    
    std::lock_guard<std::mutex> lock(mapper_mutex_);
    
    // 分配设备端内存
    PRPMappingEntry* d_mappings;
    bool* d_success;
    
    cudaError_t err = cudaMalloc(&d_mappings, sizeof(PRPMappingEntry) * mappings.size());
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate device mappings: %s\n", 
                       cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc(&d_success, sizeof(bool));
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to allocate success flag: %s\n", 
                       cudaGetErrorString(err));
        cudaFree(d_mappings);
        return false;
    }
    
    // 复制映射到设备
    err = cudaMemcpy(d_mappings, mappings.data(), sizeof(PRPMappingEntry) * mappings.size(), 
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to copy mappings to device: %s\n", 
                       cudaGetErrorString(err));
        cudaFree(d_mappings);
        cudaFree(d_success);
        return false;
    }
    
    // 启动内核
    kernel_add_batch_mappings<<<1, 1>>>(tensor_ptr, d_mappings, static_cast<uint32_t>(mappings.size()),
                                        d_hash_table_, d_mapping_nodes_, d_mapping_entries_,
                                        d_free_entry_list_, d_free_node_list_,
                                        d_free_entry_count_, d_free_node_count_, d_success);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
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
    
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to copy result: %s\n", 
                       cudaGetErrorString(err));
        return false;
    }
    
    if (success) {
        geminifs_debug("GPU Memory Mapper: Added %zu batch mappings for tensor 0x%lx\n",
                       mappings.size(), tensor_ptr);
    } else {
        geminifs_error("GPU Memory Mapper: Failed to add batch mappings - insufficient resources\n");
    }
    
    return success;
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GPUMemoryMapper::getStats() const {
    if (!is_initialized_) {
        return std::make_tuple(0, 0, 0, 0);
    }
    
    uint32_t free_entry_count, free_node_count;
    
    cudaError_t err = cudaMemcpy(&free_entry_count, d_free_entry_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to get entry stats: %s\n", cudaGetErrorString(err));
        return std::make_tuple(0, 0, 0, 0);
    }
    
    err = cudaMemcpy(&free_node_count, d_free_node_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        geminifs_error("GPU Memory Mapper: Failed to get node stats: %s\n", cudaGetErrorString(err));
        return std::make_tuple(0, 0, 0, 0);
    }
    
    uint32_t used_entries = MAX_MAPPINGS - free_entry_count;
    uint32_t used_nodes = MAX_MAPPING_NODES - free_node_count;
    
    return std::make_tuple(used_entries, MAX_MAPPINGS, used_nodes, MAX_MAPPING_NODES);
}
 

// === PRPContext Implementation ===

void PRPContext::cleanup() {
    if (prp_pages) {
        for (size_t i = 0; i < num_prp_pages; ++i) {
            if (prp_pages[i]) {
                cudaFree(prp_pages[i]);
                prp_pages[i] = nullptr;
            }
        }
        delete[] prp_pages;
        prp_pages = nullptr;
    }
    
    if (prp_page_addrs) {
        delete[] prp_page_addrs;
        prp_page_addrs = nullptr;
    }
    
    num_prp_pages = 0;
    data_size = 0;
    transfer_type = PRP_TYPE_SINGLE_PAGE;
}

bool PRPContext::allocatePRPPages(size_t num_pages) {
    if (num_pages == 0) {
        geminifs_error("PRP Context: Cannot allocate 0 pages\n");
        return false;
    }
    
    cleanup(); // 清理之前的分配
    
    // 分配页面指针数组
    prp_pages = new void*[num_pages];
    prp_page_addrs = new uint64_t[num_pages];
    
    if (!prp_pages || !prp_page_addrs) {
        geminifs_error("PRP Context: Failed to allocate page arrays\n");
        cleanup();
        return false;
    }
    
    // 初始化为空
    memset(prp_pages, 0, sizeof(void*) * num_pages);
    memset(prp_page_addrs, 0, sizeof(uint64_t) * num_pages);
    
    // 分配每个 PRP 页面
    for (size_t i = 0; i < num_pages; ++i) {
        cudaError_t err = cudaMalloc(&prp_pages[i], PRP_PAGE_SIZE);
        if (err != cudaSuccess) {
            geminifs_error("PRP Context: Failed to allocate PRP page %zu: %s\n", 
                          i, cudaGetErrorString(err));
            cleanup();
            return false;
        }
        
        // 清零页面
        err = cudaMemset(prp_pages[i], 0, PRP_PAGE_SIZE);
        if (err != cudaSuccess) {
            geminifs_error("PRP Context: Failed to clear PRP page %zu: %s\n", 
                          i, cudaGetErrorString(err));
            cleanup();
            return false;
        }
        
        // 获取页面的设备地址 (这里简化处理，实际可能需要更复杂的地址获取)
        prp_page_addrs[i] = reinterpret_cast<uint64_t>(prp_pages[i]);
    }
    
    num_prp_pages = num_pages;
    geminifs_debug("PRP Context: Successfully allocated %zu PRP pages\n", num_pages);
    return true;
}

bool PRPContext::buildPRPList(const std::vector<uint64_t>& ioaddrs) {
    if (ioaddrs.empty()) {
        geminifs_error("PRP Context: Cannot build PRP list with empty ioaddrs\n");
        return false;
    }
    
    data_size = ioaddrs.size() * PRP_PAGE_SIZE;
    
    // 检查数据大小限制
    if (data_size > MAX_TRANSFER_SIZE) {
        geminifs_error("PRP Context: Data size %zu exceeds maximum transfer size %zu\n", 
                      data_size, MAX_TRANSFER_SIZE);
        return false;
    }
    
    // 确定传输类型
    if (data_size <= PRP_PAGE_SIZE) {
        // 单页传输
        transfer_type = PRP_TYPE_SINGLE_PAGE;
        
        if (!allocatePRPPages(1)) {
            return false;
        }
        
        // 创建 PRP 页面结构
        PRPListPage host_page;
        host_page.prp_entries[0] = ioaddrs[0];
        host_page.transfer_type = PRP_TYPE_SINGLE_PAGE;
        
        // 复制到设备
        cudaError_t err = cudaMemcpy(prp_pages[0], &host_page, sizeof(PRPListPage), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            geminifs_error("PRP Context: Failed to copy single page PRP to device: %s\n", 
                          cudaGetErrorString(err));
            return false;
        }
        
        geminifs_debug("PRP Context: Built single page PRP list with ioaddr 0x%lx\n", ioaddrs[0]);
        
    } else if (data_size <= 2 * PRP_PAGE_SIZE) {
        // 双页传输
        transfer_type = PRP_TYPE_DUAL_PAGE;
        
        if (!allocatePRPPages(1)) {
            return false;
        }
        
        // 创建 PRP 页面结构
        PRPListPage host_page;
        host_page.prp_entries[0] = ioaddrs[0];
        host_page.prp_entries[1] = ioaddrs.size() > 1 ? ioaddrs[1] : 0;
        host_page.transfer_type = PRP_TYPE_DUAL_PAGE;
        
        // 复制到设备
        cudaError_t err = cudaMemcpy(prp_pages[0], &host_page, sizeof(PRPListPage), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            geminifs_error("PRP Context: Failed to copy dual page PRP to device: %s\n", 
                          cudaGetErrorString(err));
            return false;
        }
        
        geminifs_debug("PRP Context: Built dual page PRP list with ioaddrs 0x%lx, 0x%lx\n", 
                      ioaddrs[0], ioaddrs.size() > 1 ? ioaddrs[1] : 0);
        
    } else {
        // PRP List 传输
        transfer_type = PRP_TYPE_LIST;
        
        // 计算需要的 PRP 页面数量
        size_t total_entries = ioaddrs.size();
        size_t pages_needed = (total_entries + PRP_ENTRIES_PER_PAGE - 1) / PRP_ENTRIES_PER_PAGE;
        
        if (!allocatePRPPages(pages_needed)) {
            return false;
        }
        
        // 构建多个 PRP 页面
        size_t entry_index = 0;
        for (size_t page_idx = 0; page_idx < pages_needed; ++page_idx) {
            PRPListPage host_page;
            
            // 填充当前页面的 entries
            size_t entries_in_this_page = std::min(PRP_ENTRIES_PER_PAGE, total_entries - entry_index);
            
            for (size_t i = 0; i < entries_in_this_page; ++i) {
                host_page.prp_entries[i] = ioaddrs[entry_index + i];
            }
            
            // 如果不是最后一页，最后一个 entry 指向下一个 PRP 页面
            if (page_idx < pages_needed - 1) {
                host_page.prp_entries[PRP_ENTRIES_PER_PAGE - 1] = prp_page_addrs[page_idx + 1];
                entries_in_this_page--; // 最后一个 entry 用于链接，减少实际数据 entries
            }
            
            host_page.transfer_type = PRP_TYPE_LIST;
            
            // 复制到设备
            cudaError_t err = cudaMemcpy(prp_pages[page_idx], &host_page, sizeof(PRPListPage), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                geminifs_error("PRP Context: Failed to copy PRP list page %zu to device: %s\n", 
                              page_idx, cudaGetErrorString(err));
                return false;
            }
            
            entry_index += entries_in_this_page;
        }
        
        geminifs_debug("PRP Context: Built PRP list with %zu pages, %zu total entries\n", 
                      pages_needed, total_entries);
    }
    
    return true;
}



/**
 * 获取 PRP 传输类型字符串
 */
const char* getPRPTransferTypeString(PRPTransferType type) {
    switch (type) {
        case PRP_TYPE_SINGLE_PAGE: return "Single Page";
        case PRP_TYPE_DUAL_PAGE:   return "Dual Page";
        case PRP_TYPE_LIST:        return "PRP List";
        default:                   return "Unknown";
    }
}



// === GPUController Implementation ===

GPUController::GPUController(int device_id, const std::string& mount_base_path)
    : device_id_(device_id), mount_base_path_(mount_base_path), is_initialized_(false) {
    
    geminifs_debug("GPU Controller: Initializing for device %d with mount path '%s'\n", 
                   device_id, mount_base_path.c_str());
    
    // Set the CUDA device context
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        geminifs_error("GPU Controller: Failed to set CUDA device %d: %s\n", 
                       device_id_, cudaGetErrorString(err));
        return;
    }
    
    // Initialize the controller
    if (!initialize()) {
        geminifs_error("GPU Controller: Failed to initialize for device %d\n", device_id_);
        return;
    }
    
    is_initialized_.store(true);
    geminifs_debug("GPU Controller: Successfully initialized for device %d\n", device_id_);
}

GPUController::~GPUController() {
    cleanup();
}

bool GPUController::initialize() {
    // Create base mount directory if it doesn't exist
    std::filesystem::create_directories(mount_base_path_);
    
    // Initialize containers
    dma_contexts_.clear();
    nvme_controllers_.clear();

    // Initialize memory mapper
    memory_mapper_ = std::make_unique<GPUMemoryMapper>(device_id_);
    if (!memory_mapper_->initialize()) {
        geminifs_error("GPU Controller: Failed to initialize memory mapper for device %d\n", device_id_);
        return false;
    }
    
    geminifs_debug("GPU Controller: Successfully initialized memory mapper for device %d\n", device_id_);
    
    return true;
}

void GPUController::cleanup() {
    geminifs_debug("GPU Controller: Cleaning up device %d\n", device_id_);
    
    // Cleanup memory mapper first (before clearing DMA contexts that might use it)
    if (memory_mapper_) {
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
        for (auto& nvme_controller : nvme_controllers_) {
            if (nvme_controller) {
                geminifs_debug("GPU Controller: Resetting NVMe controller\n");
                nvme_controller.reset();
            }
        }
        
        nvme_controllers_.clear();
    }

    
    is_initialized_.store(false);
}

// === Memory Management Methods ===

bool GPUController::registerTensorMemory(const torch::Tensor& tensor, uint64_t granularity) {
    if (!isInitialized()) {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }
    
    if (!validateTensor(tensor)) {
        return false;
    }
    
    uint64_t tensor_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());
    auto tensor_size = tensor.numel() * tensor.element_size();
    
    // Check 4K alignment for tensor pointer
    if (tensor_ptr % 4096 != 0) {
        geminifs_error("GPU Controller: Tensor pointer 0x%lx is not 4K aligned. Memory registration failed.\n", tensor_ptr);
        return false;
    }
    
    // Check 4K alignment for tensor size
    if (tensor_size % 4096 != 0) {
        geminifs_error("GPU Controller: Tensor size %zu is not 4K aligned. Memory registration failed.\n", tensor_size);
        return false;
    }
    
    // Check 4K alignment for granularity (if specified)
    if (granularity > 0 && granularity % 4096 != 0) {
        geminifs_error("GPU Controller: Granularity %llu is not 4K aligned. Memory registration failed.\n", granularity);
        return false;
    }
    
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        
        // Check if tensor is already registered
        if (dma_contexts_.find(tensor_ptr) != dma_contexts_.end()) {
            geminifs_warn("GPU Controller: Tensor at %p is already registered\n", tensor.data_ptr());
            return true;
        }
        
        // Create DMA context
        geminifs_dma* dma_ctx = createDMAContext(tensor, granularity);
        if (dma_ctx == nullptr) {
            geminifs_error("GPU Controller: Failed to create DMA context for tensor at %p\n", tensor.data_ptr());
            return false;
        }
        
        dma_contexts_[tensor_ptr] = dma_ctx;
    }
    
    geminifs_debug("GPU Controller: Successfully registered tensor at %p for device %d\n", 
                   tensor.data_ptr(), device_id_);
    return true;
}

bool GPUController::unregisterTensorMemory(void* tensor_ptr) {
    if (!isInitialized()) {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }
    
    uint64_t ptr_key = reinterpret_cast<uint64_t>(tensor_ptr);
    
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        
        auto it = dma_contexts_.find(ptr_key);
        if (it == dma_contexts_.end()) {
            geminifs_warn("GPU Controller: Tensor at %p is not registered\n", tensor_ptr);
            return false;
        }
        
        // Clean up DMA context (但不释放 CUDA 内存)
        geminifs_dma* dma_ctx = it->second;
        // 注意: 不调用 cudaFree(dma_ctx->ioaddrs)，因为 CUDA 内存由应用进程管理
        delete dma_ctx;
        
        dma_contexts_.erase(it);
    }
    
    geminifs_debug("GPU Controller: Successfully unregistered tensor at %p for device %d\n", 
                   tensor_ptr, device_id_);
    return true;
}

geminifs_dma* GPUController::getDMAContext(void* tensor_ptr) {
    uint64_t ptr_key = reinterpret_cast<uint64_t>(tensor_ptr);
    
    std::lock_guard<std::mutex> lock(memory_mutex_);
    auto it = dma_contexts_.find(ptr_key);
    return (it != dma_contexts_.end()) ? it->second : nullptr;
}

const std::unordered_map<uint64_t, geminifs_dma*>& GPUController::getAllDMAContexts() const {
    return dma_contexts_;
}

void GPUController::clearAllDMAContexts() {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    for (auto& pair : dma_contexts_) {
        geminifs_dma* dma_ctx = pair.second;
        // 注意: 不调用 cudaFree，因为 CUDA 内存由应用进程管理
        // PRP 上下文会在 geminifs_dma 的析构函数中自动清理
        delete dma_ctx;
    }
    
    dma_contexts_.clear();
    geminifs_debug("GPU Controller: Cleared all DMA contexts for device %d\n", device_id_);
}

// === Storage Management Methods ===

bool GPUController::addNVMeController(NVMeControllerPtr nvme_controller) {
    if (!isInitialized()) {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }
    
    if (!nvme_controller) {
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

bool GPUController::removeNVMeController(size_t index) {
    if (!isInitialized()) {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }
    
    {
        std::lock_guard<std::mutex> lock(storage_mutex_);
        
        if (index >= nvme_controllers_.size()) {
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

NVMeControllerPtr GPUController::getNVMeController(size_t index) {
    std::lock_guard<std::mutex> lock(storage_mutex_);
    
    if (index >= nvme_controllers_.size()) {
        return nullptr;
    }
    
    return nvme_controllers_[index];
}

const std::vector<NVMeControllerPtr>& GPUController::getAllNVMeControllers() const {
    return nvme_controllers_;
}

size_t GPUController::getControllerCount() const {
    std::lock_guard<std::mutex> lock(storage_mutex_);
    return nvme_controllers_.size();
}

// === File Operations ===

void* GPUController::openFile(const std::string& filename, size_t file_size, uint32_t o_flag, size_t controller_index) {
    if (!isInitialized()) {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return nullptr;
    }
    
    auto controller = getNVMeController(controller_index);
    if (!controller) {
        geminifs_error("GPU Controller: Invalid controller index %zu\n", controller_index);
        return nullptr;
    }
    
    return controller->g_open(filename, file_size, o_flag);
}

// === Utility Methods ===

std::pair<size_t, size_t> GPUController::getMemoryStats() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);
    
    size_t total_memory = 0;
    size_t total_tensors = dma_contexts_.size();
    
    for (const auto& pair : dma_contexts_) {
        const geminifs_dma* dma_ctx = pair.second;
        if (dma_ctx->dma_ptr) {
            // Calculate memory based on DMA size (this might need adjustment based on actual DMA structure)
            total_memory += dma_ctx->dma_ptr->n_ioaddrs * GPU_PAGE_SIZE;
        }
    }
    
    return std::make_pair(total_memory, total_tensors);
}

// === Private Methods ===

bool GPUController::validateTensor(const torch::Tensor& tensor) const {
    // Check if tensor is on the correct device
    if (!tensor.is_cuda()) {
        geminifs_error("GPU Controller: Tensor is not on CUDA device\n");
        return false;
    }
    
    if (tensor.device().index() != device_id_) {
        geminifs_error("GPU Controller: Tensor is on device %d, expected device %d\n", 
                       tensor.device().index(), device_id_);
        return false;
    }
    
    // Check alignment
    if (!is_ptr_aligned(tensor.data_ptr())) {
        geminifs_error("GPU Controller: Tensor data pointer %p is not aligned to GPU_PAGE_SIZE\n", 
                       tensor.data_ptr());
        return false;
    }
    
    auto tensor_size = tensor.numel() * tensor.element_size();
    if (!is_aligned(tensor_size)) {
        geminifs_error("GPU Controller: Tensor size %ld is not aligned to GPU_PAGE_SIZE\n", tensor_size);
        return false;
    }
    
    return true;
}

bool GPUController::performDMASlicing(geminifs_dma* dma_ctx, size_t tensor_size, uint64_t granularity) {
    // 获取所有NVMe控制器maxIOsize的最小值（用于第二级切割）
    uint64_t min_max_io_size = UINT64_MAX;
    size_t num_nvme_controllers = nvme_controllers_.size();
    
    for (const auto& nvme_ctrl : nvme_controllers_) {
        if (nvme_ctrl && nvme_ctrl->maxIOsize > 0) {
            min_max_io_size = std::min(min_max_io_size, nvme_ctrl->maxIOsize);
        }
    }
    
    // 如果没有找到有效的maxIOsize，报错并返回
    if (min_max_io_size == UINT64_MAX) {
        geminifs_error("GPU Controller: No valid maxIOsize found in any NVMe controller. Memory registration failed.\n");
        return false;
    }
    
    // 记录切片粒度信息
    dma_ctx->slice_granularity = (granularity > 0) ? granularity : min_max_io_size;
    
    // 实现两级切割逻辑
    if (granularity > 0) {
        // 第一级：按照外部传入的granularity进行切割
        std::vector<std::pair<size_t, size_t>> primary_slices;  // (offset, size)
        size_t remaining_size = tensor_size;
        size_t current_offset = 0;
        
        while (remaining_size > 0) {
            size_t slice_size = std::min(remaining_size, (size_t)granularity);
            primary_slices.push_back(std::make_pair(current_offset, slice_size));
            
            current_offset += slice_size;
            remaining_size -= slice_size;
        }
        
        geminifs_debug("GPU Controller: First-level slicing: %zu slices by granularity %llu\n", 
                       primary_slices.size(), granularity);
        
        // 第二级：对每个第一级切片再按照maxIOsize进行切割
        for (const auto& primary_slice : primary_slices) {
            size_t slice_offset = primary_slice.first;
            size_t slice_size = primary_slice.second;
            
            if (slice_size <= min_max_io_size) {
                // 当前切片小于等于maxIOsize，不需要进一步切割
                dma_ctx->slice_sizes.push_back(slice_size);
                dma_ctx->slice_offsets.push_back(slice_offset);
            } else {
                // 当前切片需要按照maxIOsize进一步切割
                size_t sub_remaining = slice_size;
                size_t sub_offset = slice_offset;
                
                while (sub_remaining > 0) {
                    size_t sub_slice_size = std::min(sub_remaining, (size_t)min_max_io_size);
                    dma_ctx->slice_sizes.push_back(sub_slice_size);
                    dma_ctx->slice_offsets.push_back(sub_offset);
                    
                    sub_offset += sub_slice_size;
                    sub_remaining -= sub_slice_size;
                }
            }
        }
        
        dma_ctx->num_slices = dma_ctx->slice_sizes.size();
        
        geminifs_debug("GPU Controller: Two-level slicing complete: %zu final slices "
                       "(granularity %llu -> maxIOsize %llu)\n", 
                       dma_ctx->num_slices, granularity, min_max_io_size);
        
    } else {
        // 没有外部粒度，只按照maxIOsize进行切割
        if (tensor_size <= min_max_io_size) {
            // 数据小于等于maxIOsize，不进行切片
            dma_ctx->num_slices = 1;
            dma_ctx->slice_sizes.push_back(tensor_size);
            dma_ctx->slice_offsets.push_back(0);
            
            geminifs_debug("GPU Controller: Tensor size %zu <= maxIOsize %llu, no slicing needed\n", 
                           tensor_size, min_max_io_size);
        } else {
            // 需要按照maxIOsize进行切片
            size_t remaining_size = tensor_size;
            size_t current_offset = 0;
            
            while (remaining_size > 0) {
                size_t slice_size = std::min(remaining_size, (size_t)min_max_io_size);
                dma_ctx->slice_sizes.push_back(slice_size);
                dma_ctx->slice_offsets.push_back(current_offset);
                
                current_offset += slice_size;
                remaining_size -= slice_size;
            }
            
            dma_ctx->num_slices = dma_ctx->slice_sizes.size();
            
            geminifs_debug("GPU Controller: Single-level slicing: %zu slices by maxIOsize %llu\n", 
                           dma_ctx->num_slices, min_max_io_size);
        }
    }
    
    // 对所有切片的size和offset进行4K对齐检查
    const uint64_t alignment_4k = 4096;
    for (size_t i = 0; i < dma_ctx->num_slices; i++) {
        size_t slice_size = dma_ctx->slice_sizes[i];
        size_t slice_offset = dma_ctx->slice_offsets[i];
        
        // 检查slice size是否4K对齐
        if (slice_size % alignment_4k != 0) {
            geminifs_error("GPU Controller: Slice %zu size %zu is not 4K-aligned\n", i, slice_size);
            return false;
        }
        
        // 检查slice offset是否4K对齐
        if (slice_offset % alignment_4k != 0) {
            geminifs_error("GPU Controller: Slice %zu offset %zu is not 4K-aligned\n", i, slice_offset);
            return false;
        }
    }
    
    geminifs_debug("GPU Controller: All %zu slices are 4K-aligned (size and offset)\n", 
                   dma_ctx->num_slices);
    
    return true;
}

geminifs_dma* GPUController::createDMAContext(const torch::Tensor& tensor, uint64_t granularity) {
    auto tensor_size = tensor.numel() * tensor.element_size();
    
    // 如果指定了切割粒度（非0），检查tensor大小是否为粒度的整数倍
    if (granularity > 0) {
        if (tensor_size % granularity != 0) {
            geminifs_error("GPU Controller: Tensor size %zu is not a multiple of granularity %llu. Memory registration failed.\n", 
                          tensor_size, granularity);
            return nullptr;
        }
        geminifs_debug("GPU Controller: Using external granularity %llu for tensor size %zu\n", granularity, tensor_size);
    }
    
    // For now, we'll assume we have at least one NVMe controller to get the ctrl pointer
    if (nvme_controllers_.empty()) {
        geminifs_error("GPU Controller: No NVMe controllers available for DMA context creation\n");
        return nullptr;
    }
    
    // Use the first controller for DMA creation
    auto first_controller = nvme_controllers_[0];
    if (!first_controller || !first_controller->controller) {
        geminifs_error("GPU Controller: Invalid NVMe controller for DMA context creation\n");
        return nullptr;
    }
    
    DmaPtr dma_ptr = getDeviceDma(first_controller->controller->ctrl, 
                                  tensor.data_ptr(), tensor_size, device_id_);
    if (dma_ptr == nullptr) {
        geminifs_error("GPU Controller: Failed to get DMA pointer for tensor\n");
        return nullptr;
    }
    
    // 创建 geminifs_dma 结构
    geminifs_dma* dma_ctx = new geminifs_dma();
    dma_ctx->dma_ptr = dma_ptr;
    
    // 执行 DMA 切片
    if (!performDMASlicing(dma_ctx, tensor_size, granularity)) {
        delete dma_ctx;
        return nullptr;
    }
    
    // 创建对应数量的PRPMappingEntry
    dma_ctx->prp_mappings.reserve(dma_ctx->num_slices);
    
    // 统计第三种类型(transfer_type=2)的数量
    size_t type2_count = 0;
    
    for (size_t i = 0; i < dma_ctx->num_slices; i++) {
        size_t slice_size = dma_ctx->slice_sizes[i];
        uint32_t transfer_type = 0;  // 默认类型0
        
        // 根据slice_size确定NVMe IO cmd类型
        if (slice_size <= 4096) {
            transfer_type = 0;  // 小于等于4K
        } else if (slice_size <= 8192) {
            transfer_type = 1;  // 大于4K小于等于8K
        } else {
            transfer_type = 2;  // 大于8K
            type2_count++;       // 统计第三种类型的数量
        }
        
        // 创建PRPMappingEntry (prp1和prp2先不初始化)
        PRPMappingEntry entry(transfer_type, 0, 0);
        dma_ctx->prp_mappings.push_back(entry);
        
        geminifs_debug("GPU Controller: Created PRPMappingEntry[%zu]: transfer_type=%u for slice_size=%zu\n", 
                       i, transfer_type, slice_size);
    }
    
    // 记录第三种类型的数量
    dma_ctx->type2_prp_count = type2_count;
    
    // 为第三种类型的PRP分配4KB对齐的GPU内存
    if (type2_count > 0) {
        size_t memory_size = type2_count * 4096;  // 每个第三种类型需要4KB
        
        cudaError_t err = cudaMalloc(&dma_ctx->type2_prp_gpu_memory, memory_size);
        if (err != cudaSuccess) {
            geminifs_error("GPU Controller: Failed to allocate type2 PRP GPU memory (%zu bytes): %s\n", 
                           memory_size, cudaGetErrorString(err));
            delete dma_ctx;
            return nullptr;
        }
        
        // 检查4KB对齐
        uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(dma_ctx->type2_prp_gpu_memory);
        if (ptr_addr % 4096 != 0) {
            geminifs_error("GPU Controller: Allocated type2 PRP GPU memory is not 4KB aligned (addr: 0x%lx)\n", ptr_addr);
            delete dma_ctx;
            return nullptr;
        }
        
        geminifs_info("GPU Controller: Allocated %zu bytes of 4KB-aligned GPU memory for %zu type2 PRP entries at 0x%lx\n", 
                      memory_size, type2_count, ptr_addr);
        
        // 获取type2_prp_gpu_memory的DMA地址
        dma_ctx->type2_prp_dma_ptr = getDeviceDma(first_controller->controller->ctrl, 
                                                   dma_ctx->type2_prp_gpu_memory, memory_size, device_id_);
        if (dma_ctx->type2_prp_dma_ptr == nullptr) {
            geminifs_error("GPU Controller: Failed to get DMA pointer for type2 PRP GPU memory\n");
            delete dma_ctx;
            return nullptr;
        }
        
        geminifs_debug("GPU Controller: Successfully obtained DMA pointer for type2 PRP GPU memory (size: %zu bytes)\n", 
                       memory_size);
    } else {
        geminifs_debug("GPU Controller: No type2 PRP entries found, no GPU memory allocation needed\n");
    }
    
    geminifs_info("GPU Controller: DMA Context Created Successfully\n");
    geminifs_info("  Tensor size: %zu bytes\n", tensor_size);
    geminifs_info("  Slice granularity: %llu bytes\n", dma_ctx->slice_granularity);
    geminifs_info("  Total slices: %zu\n", dma_ctx->num_slices);
    
    // Print detailed slice information
    for (size_t i = 0; i < dma_ctx->num_slices; i++) {
        geminifs_info("  Slice[%zu]: offset=%zu, size=%zu\n", 
                      i, dma_ctx->slice_offsets[i], dma_ctx->slice_sizes[i]);
    }
    
    // Print DMA pointer information
    if (dma_ctx->dma_ptr) {
        geminifs_info("  DMA ptr contiguous: %s\n", dma_ctx->dma_ptr->contiguous ? "Yes" : "No");
        geminifs_info("  DMA ptr n_ioaddrs: %zu\n", dma_ctx->dma_ptr->n_ioaddrs);
        if (dma_ctx->dma_ptr->n_ioaddrs > 0) {
            geminifs_info("  DMA ptr first ioaddr: 0x%lx\n", dma_ctx->dma_ptr->ioaddrs[0]);
        }
    }
    
    return dma_ctx;
}

// === GPUControllerRegistry Implementation ===

GPUControllerRegistry& GPUControllerRegistry::getInstance() {
    static GPUControllerRegistry instance;
    return instance;
}

bool GPUControllerRegistry::registerGPUController(int device_id, GPUControllerPtr controller) {
    if (!controller) {
        geminifs_error("GPU Controller Registry: Invalid controller provided for device %d\n", device_id);
        return false;
    }
    
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    if (gpu_controllers_.find(device_id) != gpu_controllers_.end()) {
        geminifs_warn("GPU Controller Registry: Device %d already has a registered controller\n", device_id);
        return false;
    }
    
    gpu_controllers_[device_id] = controller;
    geminifs_debug("GPU Controller Registry: Registered controller for device %d\n", device_id);
    return true;
}

bool GPUControllerRegistry::unregisterGPUController(int device_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = gpu_controllers_.find(device_id);
    if (it == gpu_controllers_.end()) {
        geminifs_warn("GPU Controller Registry: No controller registered for device %d\n", device_id);
        return false;
    }
    
    gpu_controllers_.erase(it);
    geminifs_debug("GPU Controller Registry: Unregistered controller for device %d\n", device_id);
    return true;
}

GPUControllerPtr GPUControllerRegistry::getGPUController(int device_id) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = gpu_controllers_.find(device_id);
    return (it != gpu_controllers_.end()) ? it->second : nullptr;
}

const std::unordered_map<int, GPUControllerPtr>& GPUControllerRegistry::getAllGPUControllers() const {
    return gpu_controllers_;
}

void GPUControllerRegistry::clearAll() {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    gpu_controllers_.clear();
    geminifs_debug("GPU Controller Registry: Cleared all registered controllers\n");
}
