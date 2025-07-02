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
#include "nvme_controller.h"
#include "geminifs_mem.h"

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
    bool registerTensorMemory(const torch::Tensor& tensor);
    
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
    
    // === Private Methods ===
    
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
    struct geminifs_dma* createDMAContext(const torch::Tensor& tensor);
    
    // === PRP List Management Methods ===
    
    /**
     * Build PRP list for tensor DMA
     */
    bool buildPRPList(geminifs_dma* dma_ctx, const torch::Tensor& tensor);
    
    /**
     * Build single PRP entry
     */
    bool buildSinglePRP(geminifs_dma* dma_ctx, DmaPtr dma_ptr);
    
    /**
     * Build dual PRP entries
     */
    bool buildDualPRP(geminifs_dma* dma_ctx, DmaPtr dma_ptr);
    
    /**
     * Build PRP list for large transfers
     */
    bool buildListPRP(geminifs_dma* dma_ctx, DmaPtr dma_ptr);
    
    /**
     * Cleanup PRP list resources
     */
    void cleanupPRPList(geminifs_dma* dma_ctx);
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
