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
 * PRP映射条目结构 (32字节)
 */
struct PRPMappingEntry {
    uint32_t transfer_type; // NVMe cmd transfer type (4字节)
    uint32_t data_length;   // 数据长度 (4字节)
    uint64_t prp1;          // PRP1 (8字节)
    uint64_t prp2;          // PRP2 may be NULL (8字节)
    uint64_t tensor_offset; // 该PRP entry在granularity内的偏移位置 (8字节)
    
    __device__ __host__ PRPMappingEntry() : transfer_type(0), data_length(0), prp1(0), prp2(0), tensor_offset(0) {}
    __device__ __host__ PRPMappingEntry(uint32_t transfer_type, uint32_t data_len, uint64_t p1, uint64_t p2) 
        : transfer_type(transfer_type), data_length(data_len), prp1(p1), prp2(p2), tensor_offset(0) {}
    __device__ __host__ PRPMappingEntry(uint32_t transfer_type, uint32_t data_len, uint64_t p1, uint64_t p2, uint64_t offset) 
        : transfer_type(transfer_type), data_length(data_len), prp1(p1), prp2(p2), tensor_offset(offset) {}
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
    uint64_t tensor_size;        // 8字节 - 注册的GPU虚拟内存总长度
    
    __device__ __host__ GPUHashEntry() : GPU_virtual_ptr(0), first_node(0xFFFFFFFF), mapping_count(0), tensor_size(0) {}
};

/**
 * GPU内存映射管理器的Device侧视图结构体
 * 用于在GPU kernel中访问映射数据结构
 */
struct GPUMemoryMapperDeviceView {
    PRPMappingEntry* d_mapping_entries;     // PRP映射条目数组
    GPUMappingNode* d_mapping_nodes;        // 映射节点数组
    GPUHashEntry* d_hash_table;             // 哈希表
    uint32_t* d_free_entry_list;            // 空闲映射条目列表
    uint32_t* d_free_node_list;             // 空闲映射节点列表
    uint32_t* d_free_entry_count;           // 空闲映射条目计数器
    uint32_t* d_free_node_count;            // 空闲映射节点计数器
    
    __device__ __host__ GPUMemoryMapperDeviceView() 
        : d_mapping_entries(nullptr), d_mapping_nodes(nullptr), d_hash_table(nullptr),
          d_free_entry_list(nullptr), d_free_node_list(nullptr), 
          d_free_entry_count(nullptr), d_free_node_count(nullptr) {}
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
    
    // Device侧视图结构体指针 - 在GPU内存中
    GPUMemoryMapperDeviceView* d_device_view_;
    
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
     * 批量添加多个映射到同一个tensor
     * @param tensor_ptr Tensor的GPU虚拟内存指针
     * @param mappings 映射条目向量
     * @return 成功返回true
     */
    bool addBatchMappings(uint64_t tensor_ptr, uint64_t tensor_size, const std::vector<PRPMappingEntry>& mappings);
 
    
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
     * 获取Device侧视图结构体指针 (用于GPU kernel)
     */
    GPUMemoryMapperDeviceView* getDeviceViewPtr() const { return d_device_view_; }
    
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
    // 改进的哈希函数，解决指针哈希冲突问题
    // 灵感来源于MurmurHash和xorshift
    key = (key >> 12); // 指针通常是4K对齐的，右移12位可以消除低位的0，增加有效信息
    key ^= (key >> 33);
    key *= 0xff51afd7ed558ccdULL;
    key ^= (key >> 33);
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= (key >> 33);
    return static_cast<uint32_t>(key % GPUMemoryMapper::HASH_TABLE_SIZE);
}

// === GPU设备端查找函数 ===

/**
 * GPU端查找tensor的所有PRP映射
 * @param tensor_ptr Tensor的GPU虚拟内存指针
 * @param hash_table 哈希表指针
 * @param mapping_nodes 映射节点数组指针
 * @param mapping_entries 映射条目数组指针
 * @param result_ptrs 输出的PRP映射条目指针数组 (调用者分配)
 * @param max_results 最大结果数量
 * @return 实际找到的映射数量
 */
__device__ uint32_t gpu_lookup_all_prp_mappings(uint64_t tensor_ptr,
                                                GPUHashEntry* hash_table,
                                                GPUMappingNode* mapping_nodes,
                                                PRPMappingEntry* mapping_entries,
                                                PRPMappingEntry** result_ptrs,
                                                uint32_t max_results,
                                                uint64_t* tensor_size);

/**
 * GPU kernel用于查询和打印tensor的PRP映射信息
 * @param device_view Device侧视图结构体指针
 * @param tensor_ptr Tensor指针
 * @param tensor_size Tensor大小
 * @param granularity 粒度大小
 */
__global__ void gpu_debug_prp_mappings_kernel(GPUMemoryMapperDeviceView* device_view,
                                              uint64_t tensor_ptr, 
                                              size_t tensor_size,
                                              uint64_t granularity);

/**
 * 批量NVMe读取kernel：每个线程处理一个PRP映射条目
 * @param d_fd NVMe文件描述符
 * @param mapping_entry_ptrs PRP映射条目指针数组
 * @param found_count 找到的映射条目数量
 * @param base_file_offset 文件基础偏移量
 */
__global__ void nvme_batch_read_kernel(NVMe_File* d_fd,
                                      PRPMappingEntry** mapping_entry_ptrs,
                                      uint32_t found_count,
                                      size_t base_file_offset);

/**
 * 批量NVMe写入kernel：每个线程处理一个PRP映射条目
 * @param d_fd NVMe文件描述符
 * @param mapping_entry_ptrs PRP映射条目指针数组
 * @param found_count 找到的映射条目数量
 * @param base_file_offset 文件基础偏移量
 */
__global__ void nvme_batch_write_kernel(NVMe_File* d_fd,
                                       PRPMappingEntry** mapping_entry_ptrs,
                                       uint32_t found_count,
                                       size_t base_file_offset);

/**
 * GPU读取kernel：查询PRP映射并动态并行发起NVMe IO
 * @param d_fd NVMe文件描述符
 * @param tensor_ptr GPU tensor指针
 * @param offset 文件偏移量
 * @param len 读取长度
 * @param device_view GPU内存映射器的设备视图
 */
__global__ void GPU_Read_kernel(NVMe_File* d_fd,
                               uint64_t tensor_ptr,
                               size_t offset,
                               size_t len, 
                               GPUMemoryMapperDeviceView* device_view);

/**
 * GPU写入kernel：查询PRP映射并动态并行发起NVMe IO
 * @param d_fd NVMe文件描述符
 * @param tensor_ptr GPU tensor指针
 * @param offset 文件偏移量
 * @param len 写入长度
 * @param device_view GPU内存映射器的设备视图
 */
__global__ void GPU_Write_kernel(NVMe_File* d_fd,
                                uint64_t tensor_ptr,
                                size_t offset,
                                size_t len, 
                                GPUMemoryMapperDeviceView* device_view);

/**
 * Host端函数用于调用GPU kernel查询和打印PRP映射
 * @param mapper GPUMemoryMapper指针
 * @param tensor_ptr Tensor指针
 * @param tensor_size Tensor大小
 * @param granularity 粒度大小
 */
void debug_prp_mappings_from_gpu(GPUMemoryMapper* mapper, 
                                 uint64_t tensor_ptr, 
                                 size_t tensor_size, 
                                 uint64_t granularity);

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
    
    /**
     * Initialize PRP entries for all slices in DMA context
     * @param dma_ctx DMA context with slice information
     * @return true if successful, false otherwise
     */
    bool initializePRPEntries(geminifs_dma* dma_ctx);

    /**
     * Add PRP mappings to GPU memory for all granularity groups
     * @param dma_ctx DMA context with initialized PRP entries
     * @return true if successful, false otherwise
     */
    bool addPRPMappingsToGPU(geminifs_dma* dma_ctx);


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
