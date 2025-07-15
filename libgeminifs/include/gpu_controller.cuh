#ifndef __GPU_CONTROLLER_H__
#define __GPU_CONTROLLER_H__

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <string>
#include <torch/all.h>
#include "nvme_controller.cuh"
#include "geminifs_mem.h"

/**
 * PRP映射条目结构 (20字节)
 */
struct PRPMappingEntry {
    uint32_t transfer_type; // NVMe cmd transfer type (4字节)
    uint64_t prp1;          // PRP1
    uint64_t prp2;          // PRP2 may be NULL
    __device__ __host__ PRPMappingEntry() : transfer_type(0), prp1(0), prp2(0)  {}
    __device__ __host__ PRPMappingEntry(uint32_t transfer_type, uint64_t p1, uint64_t p2 ) 
        : transfer_type(transfer_type), prp1(p1), prp2(p2) {}
};

/**
 * GPU端映射链表节点 (16字节)
 */
struct GPUMappingNode {
    uint32_t entry_index;     // 4字节 - 在映射条目数组中的索引
    uint32_t next_node;       // 4字节 - 下一个节点的索引 (链表)
    uint64_t reserved;        // 8字节 - 保留字段，用于对齐
    
    __device__ __host__ GPUMappingNode() : entry_index(0xFFFFFFFF), next_node(0xFFFFFFFF), reserved(0) {}
    __device__ __host__ GPUMappingNode(uint32_t entry_idx) : entry_index(entry_idx), next_node(0xFFFFFFFF), reserved(0) {}
};


/**
 * GPU端哈希表条目 (16字节) - 支持链表
 */
struct GPUHashEntry {
    uint64_t GPU_virtual_ptr;    // 8字节 - tensor指针 (作为key)
    uint32_t first_node;         // 4字节 - 第一个映射节点的索引
    uint32_t mapping_count;      // 4字节 - 该tensor的映射数量
    
    __device__ __host__ GPUHashEntry() : GPU_virtual_ptr(0), first_node(0xFFFFFFFF), mapping_count(0) {}
};

/**
 * GPU内存映射管理器
 */
class GPUMemoryMapper {
public:
    static constexpr size_t MAX_MAPPINGS = 1024 * 1024;          // 1M个PRP映射条目
    static constexpr size_t MAX_MAPPING_NODES = 4 * 1024 * 1024; // 2M个映射节点 (支持平均每个tensor 4个映射)
    static constexpr size_t HASH_TABLE_SIZE = 512 * 1024;        // 512K个哈希槽位
    static constexpr uint32_t INVALID_INDEX = 0xFFFFFFFF;
    
private:
    // GPU内存指针
    PRPMappingEntry* d_mapping_entries_;     // PRP映射条目数组
    GPUMappingNode* d_mapping_nodes_;        // 映射节点数组 (用于链表)
    GPUHashEntry* d_hash_table_;             // 哈希表
    uint32_t* d_free_entry_list_;            // 空闲映射条目列表
    uint32_t* d_free_node_list_;             // 空闲映射节点列表
    uint32_t* d_free_entry_count_;           // 空闲映射条目计数器
    uint32_t* d_free_node_count_;            // 空闲映射节点计数器
    
    // 主机端管理
    mutable std::mutex mapper_mutex_;
    bool is_initialized_;
    int device_id_;
    
public:
    GPUMemoryMapper(int device_id);
    ~GPUMemoryMapper();
    
    /**
     * 初始化GPU内存映射器
     */
    bool initialize();
    
    /**
     * 清理所有资源
     */
    void cleanup();
    
    /**
     * 添加单个映射到已存在的tensor
     * @param tensor_ptr Tensor的GPU虚拟内存指针
     * @param transfer_type 传输类型
     * @param prp1 PRP1地址
     * @param prp2 PRP2地址
     * @return 成功返回true
     */
    bool addMapping(uint64_t tensor_ptr, uint32_t transfer_type, uint64_t prp1, uint64_t prp2);
    
    
    /**
     * 批量添加多个映射到同一个tensor
     * @param tensor_ptr Tensor的GPU虚拟内存指针
     * @param mappings 映射条目向量
     * @return 成功返回true
     */
    bool addBatchMappings(uint64_t tensor_ptr, const std::vector<PRPMappingEntry>& mappings);
 
    
    /**
     * 移除tensor的所有映射
     * @param tensor_ptr Tensor的GPU虚拟内存指针
     * @return 成功返回true
     */
    bool removeAllMappings(uint64_t tensor_ptr);
    
   /**
     * 获取映射条目数组指针 (用于GPU kernel)
     */
    PRPMappingEntry* getMappingEntriesPtr() const { return d_mapping_entries_; }
    
    /**
     * 获取映射节点数组指针 (用于GPU kernel)
     */
    GPUMappingNode* getMappingNodesPtr() const { return d_mapping_nodes_; }
    
    /**
     * 获取哈希表指针 (用于GPU kernel)
     */
    GPUHashEntry* getHashTablePtr() const { return d_hash_table_; }
    
    /**
     * 获取统计信息
     */
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> getStats() const; // (used_entries, total_entries, used_nodes, total_nodes)
};


// === GPU mem to dma maping Implementation ===


/**
 * GPU端哈希函数
 */
__device__ __forceinline__ uint32_t gpu_hash(uint64_t key) {
    // 使用FNV-1a哈希算法的简化版本
    uint64_t hash = 14695981039346656037ULL;
    hash ^= key;
    hash *= 1099511628211ULL;
    return static_cast<uint32_t>(hash % GPUMemoryMapper::HASH_TABLE_SIZE);
}

// === GPU设备端查找函数 ===

/**
 * GPU端查找tensor的所有PRP映射
 * @param tensor_ptr Tensor的GPU虚拟内存指针
 * @param hash_table 哈希表指针
 * @param mapping_nodes 映射节点数组指针
 * @param mapping_entries 映射条目数组指针
 * @param results 输出的PRP映射条目数组 (调用者分配)
 * @param max_results 最大结果数量
 * @return 实际找到的映射数量
 */
__device__ uint32_t gpu_lookup_all_prp_mappings(uint64_t tensor_ptr,
                                                GPUHashEntry* hash_table,
                                                GPUMappingNode* mapping_nodes,
                                                PRPMappingEntry* mapping_entries,
                                                PRPMappingEntry* results,
                                                uint32_t max_results);

/**
 * GPU Controller class for managing a single GPU device's memory and storage
 * Handles both GPU memory management and multiple NVMe controllers
 */
class GPUController {
public:
    /**
     * Constructor for GPU Controller
     * @param device_id CUDA device ID
     * @param mount_base_path Base mount path for all NVMe controllers under this GPU
     */
    GPUController(int device_id, const std::string& mount_base_path);
    
    
    /**
     * Destructor - cleans up all resources
     */
    ~GPUController();
    
    // === Memory Management Methods ===
    
    /**
     * Register a tensor's memory for DMA operations
     * @param tensor PyTorch tensor to register
     * @return true if successful, false otherwise
     */
    bool registerTensorMemory(const torch::Tensor& tensor, uint64_t granularity = 0);
    
    /**
     * Unregister a tensor's memory
     * @param tensor_ptr Pointer to tensor data
     * @return true if successful, false otherwise
     */
    bool unregisterTensorMemory(void* tensor_ptr);
    
    /**
     * Get DMA context for a registered tensor
     * @param tensor_ptr Pointer to tensor data
     * @return DMA context or nullptr if not found
     */
    struct geminifs_dma* getDMAContext(void* tensor_ptr);
    
    /**
     * Get all registered memory contexts
     * @return Map of all registered DMA contexts
     */
    const std::unordered_map<uint64_t, struct geminifs_dma*>& getAllDMAContexts() const;
    
    /**
     * Clear all registered memory contexts
     */
    void clearAllDMAContexts();
    
    // === Storage Management Methods ===
    
    /**
     * Add an NVMe controller to this GPU
     * @param nvme_controller Shared pointer to NVMe controller
     * @return true if successful, false otherwise
     */
    bool addNVMeController(NVMeControllerPtr nvme_controller);
    
    /**
     * Remove an NVMe controller by index
     * @param index Index of the controller to remove
     * @return true if successful, false otherwise
     */
    bool removeNVMeController(size_t index);
    
    /**
     * Get NVMe controller by index
     * @param index Index of the controller
     * @return Shared pointer to controller or nullptr if not found
     */
    NVMeControllerPtr getNVMeController(size_t index);
    
    /**
     * Get all NVMe controllers
     * @return Vector of all NVMe controllers
     */
    const std::vector<NVMeControllerPtr>& getAllNVMeControllers() const;
    
    /**
     * Get number of NVMe controllers
     * @return Number of controllers
     */
    size_t getControllerCount() const;
    
    // === File Operations ===
    
    /**
     * Open a file using one of the managed NVMe controllers
     * @param filename Name of the file to open
     * @param file_size Size of the file
     * @param o_flag Open flags
     * @param controller_index Index of the controller to use (default: 0)
     * @return File descriptor or nullptr if failed
     */
    void* openFile(const std::string& filename, size_t file_size, uint32_t o_flag, size_t controller_index = 0);
    
    // === GPU Management Methods ===
    
    /**
     * Get the CUDA device ID
     * @return Device ID
     */
    int getDeviceId() const { return device_id_; }
    
    /**
     * Get the mount base path
     * @return Mount base path
     */
    const std::string& getMountBasePath() const { return mount_base_path_; }
    
    /**
     * Check if the GPU controller is initialized
     * @return true if initialized, false otherwise
     */
    bool isInitialized() const { return is_initialized_.load(); }
    
    GPUMemoryMapper* getMemoryMapper() const { return memory_mapper_.get(); }

    /**
     * Get memory usage statistics
     * @return Pair of (used_memory, total_registered_tensors)
     */
    std::pair<size_t, size_t> getMemoryStats() const;

private:
    // === Private Members ===
    
    int device_id_;                                                          // CUDA device ID
    std::string mount_base_path_;                                           // Base mount path
    std::atomic<bool> is_initialized_;                                      // Initialization status
    
    // Memory management
    std::unordered_map<uint64_t, struct geminifs_dma*> dma_contexts_;      // DMA contexts for registered memory
    mutable std::mutex memory_mutex_;                                       // Mutex for memory operations
    
    // Storage management  
    std::vector<NVMeControllerPtr> nvme_controllers_;                       // NVMe controllers
    mutable std::mutex storage_mutex_;                                      // Mutex for storage operations
    
    std::unique_ptr<GPUMemoryMapper> memory_mapper_;
    
    /**
     * Initialize the GPU controller
     * @return true if successful, false otherwise
     */
    bool initialize();
    
    /**
     * Cleanup all resources
     */
    void cleanup();
    
    /**
     * Validate tensor for memory registration
     * @param tensor Tensor to validate
     * @return true if valid, false otherwise
     */
    bool validateTensor(const torch::Tensor& tensor) const;
    
    /**
     * Create DMA context for tensor
     * @param tensor Tensor to create context for
     * @return DMA context or nullptr if failed
     */
    struct geminifs_dma* createDMAContext(const torch::Tensor& tensor, uint64_t granularity = 0);
    
    /**
     * Perform DMA memory slicing for a given tensor
     * @param dma_ctx DMA context to populate with slice information
     * @param tensor_size Size of the tensor
     * @param granularity Optional granularity for slicing (0 means use maxIOsize only)
     * @return true if successful, false otherwise
     */
    bool performDMASlicing(geminifs_dma* dma_ctx, size_t tensor_size, uint64_t granularity = 0);
    
    // === PRP List Management Methods ===
    


};

// Smart pointer for GPUController
using GPUControllerPtr = std::shared_ptr<GPUController>;

/**
 * Global GPU Controller Registry for managing multiple GPU controllers
 */
class GPUControllerRegistry {
public:
    /**
     * Get the singleton instance
     * @return Reference to the singleton instance
     */
    static GPUControllerRegistry& getInstance();
    
    /**
     * Register a GPU controller
     * @param device_id CUDA device ID
     * @param controller Shared pointer to GPU controller
     * @return true if successful, false otherwise
     */
    bool registerGPUController(int device_id, GPUControllerPtr controller);
    
    /**
     * Unregister a GPU controller
     * @param device_id CUDA device ID
     * @return true if successful, false otherwise
     */
    bool unregisterGPUController(int device_id);
    
    /**
     * Get GPU controller by device ID
     * @param device_id CUDA device ID
     * @return Shared pointer to controller or nullptr if not found
     */
    GPUControllerPtr getGPUController(int device_id);
    
    /**
     * Get all registered GPU controllers
     * @return Map of all GPU controllers
     */
    const std::unordered_map<int, GPUControllerPtr>& getAllGPUControllers() const;
    
    /**
     * Clear all registered GPU controllers
     */
    void clearAll();
    
private:
    std::unordered_map<int, GPUControllerPtr> gpu_controllers_;
    mutable std::mutex registry_mutex_;
    
    // Singleton pattern
    GPUControllerRegistry() = default;
    ~GPUControllerRegistry() = default;
    GPUControllerRegistry(const GPUControllerRegistry&) = delete;
    GPUControllerRegistry& operator=(const GPUControllerRegistry&) = delete;
};

#endif // __GPU_CONTROLLER_H__
