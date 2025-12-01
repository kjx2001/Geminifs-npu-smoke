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
#include "gpu_file_manager.cuh"
#include "prp_mapping_entry.h"
#include <cuda/std/span>



class GPUMemoryMapper {
public:
    struct GPUMappingNode {
        uint32_t entry_index;     // 4字节 - 在映射条目数组中的索引
        uint32_t next_node;       // 4字节 - 下一个节点的索引 (链表)
        uint64_t reserved;        // 8字节 - 保留字段，用于对齐
        __device__ __host__ GPUMappingNode(uint32_t entry_idx = 0) : entry_index(entry_idx), next_node(0xFFFFFFFF), reserved(0) {}
    };

    struct GPUHashEntry {
        uint64_t dev_ptr;
        uint64_t tensor_size;
        PRPMappingEntrySpan mappings;
        GPUHashEntry* next; 
    };

public:
    static constexpr size_t MAX_MAPPINGS = 1024 * 1024;          // 1M个PRP映射条目
    static constexpr size_t MAX_MAPPING_NODES = 4 * 1024 * 1024; // 2M个映射节点 (支持平均每个tensor 4个映射)
    static constexpr size_t HASH_TABLE_SIZE = 512 * 1024;        // 512K个哈希槽位
    static constexpr uint32_t INVALID_INDEX = 0xFFFFFFFF;
    
private:
    // 主机端管理
    mutable std::mutex mapper_mutex_;
    std::atomic<bool> initialized_;
    std::unordered_map<uint64_t, GPUHashEntry*> host_mappings_; // 主机端的映射存储
    int device_id_;
public:
    GPUMemoryMapper(int device_id) : initialized_(false), device_id_(device_id) {}
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
    bool addBatchMappings(uint64_t tensor_ptr, uint64_t tensor_size, 
                          const std::vector<PRPMappingEntry>& mappings, 
                          void *gpu_buffer = nullptr, size_t gpu_buffer_size = 0);
    

    GPUHashEntry* lookupMappings(uint64_t tensor_ptr) const;
    
    /**
     * 移除tensor的所有映射
     * @param tensor_ptr Tensor的GPU虚拟内存指针
     * @return 成功返回true
     */
    bool removeAllMappings(uint64_t tensor_ptr);

    void debugPrintMappings(const GPUHashEntry * entry, bool inDevice) const;
};


__global__ void nvme_xfer_kernel(NVMeFilesSpan nvme_file,
                                 PRPMappingEntrySpan prp_entries,
                                 uint64_t tensor_size,
                                 size_t base_file_offset,
                                 bool is_read);

                    
__global__ void nvme_kv_xfer_kernel(NVMeFilesSpan nvme_file,
                                    PRPMappingEntrySpan k_mappings,
                                    PRPMappingEntrySpan v_mappings,
                                    uint64_t tensor_size,
                                    size_t base_file_offset,
                                    bool is_read);

__global__ void nvme_batch_xfer_kernel(BatchIoEntry* io_ctx,
                                       uint64_t tensor_size,
                                       uint32_t total_count,
                                       bool is_read);

__global__ void nvme_batch_loop_xfer_kernel(BatchIoEntry* io_ctx,
                                            uint32_t total_count,
                                            bool is_read);

__global__ void nvme_batch_read_kernel(cuda::std::span<GPUIoContext> io_ctx,
                                       uint32_t total_count,
                                       size_t base_file_offset);

__global__ void nvme_batch_write_kernel(GPUIoContext* io_ctx,
                                           uint64_t tensor_ptr,
                                           uint32_t total_count,
                                           size_t base_file_offset);


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

    bool registerTesnsorList(const std::vector<torch::Tensor> &tensor_list, uint64_t granularity = 0);
    
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
     * @param gpu_file_id GPU file ID
     * @param file_size Size of the file
     * @param o_flag Open flags
     * @param nvme_params Vector of NVMe controller parameters
     * @param gpu_file_manager GPU file manager
     * @return File descriptor or nullptr if failed
     */
    // void* openFile(GPUFileId gpu_file_id, uint32_t o_flag, const std::vector<nvme_ctrl_param>& nvme_params, GPUFileManager& gpu_file_manager);
    
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

    std::vector<struct geminifs_dma*> createDMAContexts(const std::vector<torch::Tensor>& tensor_list, uint64_t granularity = 0);
    
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

    bool initializePRPList(std::vector<geminifs_dma*> &dma_ctxs);

    bool doInitializePRPList(geminifs_dma* dma_ctx, void *gpu_addr = nullptr, size_t gpu_size = 0);

    /**
     * Add PRP mappings to GPU memory for all granularity groups
     * @param dma_ctx DMA context with initialized PRP entries
     * @return true if successful, false otherwise
     */
    bool addPRPMappingsToGPU(geminifs_dma* dma_ctx, void *gpu_addr = nullptr, size_t gpu_size = 0);


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
