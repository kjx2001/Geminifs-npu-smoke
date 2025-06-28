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
#include "memory.h"

using ControllerPtr = std::shared_ptr<Controller>;

// Structure to manage host to device file descriptor mapping
struct DeviceFileHandle {
    host_fd_t host_fd;      // Host file descriptor
    dev_fd_t device_fd;     // Device file descriptor (GPU memory)
    size_t hdr_size;        // Size of the header structure
    std::string filename;   // Original filename
    
    DeviceFileHandle(host_fd_t h_fd, dev_fd_t d_fd, size_t size, const std::string& name)
        : host_fd(h_fd), device_fd(d_fd), hdr_size(size), filename(name) {}
};

class NVMeController {
public:
    ControllerPtr controller;      // NVMe controller smart pointer
    std::unique_ptr<FileManager> file_manager; // File manager for this mount
    std::string mount_path;        // Mount path
    
    NVMeController(const ControllerPtr& ctrl, std::unique_ptr<FileManager> fm, const std::string& path)
        : controller(ctrl), file_manager(std::move(fm)), mount_path(path), is_initialized_(false) {}

    NVMeController(const nvme_ctrl_param& params);
    
    // Destructor
    ~NVMeController();

    void* g_open(std::string filename, size_t file_size, uint32_t o_flag);

    // Managed file operations - automatically handle file descriptor tracking
    host_fd_t host_file_create_managed(int block_size, size_t file_size, const std::string& filename);
    host_fd_t host_file_open_managed(const std::string& filepath, uint32_t o_flag);
    void host_file_close_managed(host_fd_t fd);

    // Device file operations - manage host to device fd mapping
    dev_fd_t device_file_create_managed(int block_size, size_t file_size, const std::string& filename);
    dev_fd_t device_file_open_managed(const std::string& filename, size_t file_size);
    void device_file_close_managed(dev_fd_t device_fd);
    // Check if controller is properly initialized
    bool is_initialized() const { return is_initialized_; }

private:
    // Private helper methods
    bool check_sys_config_exists();
    bool check_snvme_control_exists();
    ControllerPtr open_single_controller(const std::string& pci_addr, const nvme_ctrl_param& params);
    
    // Device file management helper methods

    dev_fd_t copy_host_fd_to_device(host_fd_t host_fd, size_t hdr_size);
    void cleanup_device_files();
    
    // Initialization state
    bool is_initialized_;
    
    // Device file descriptor management
    std::vector<DeviceFileHandle> device_files_;
    mutable std::mutex device_files_mtx_;
};
// Smart pointer for MountController
using NVMeControllerPtr = std::shared_ptr<NVMeController>;

struct geminifs_dma{
    uint64_t *ioaddrs;
    DmaPtr dma_ptr;
};

struct geminifs_metadata{
    std::vector<ControllerPtr> ctrls;
    std::atomic<bool> is_init{false};
    thrust::device_ptr<GPUFilePool> global_pool; //device_ptr
    GPUPoolId pool_id;
    uint64_t file_size;
    uint64_t file_block_size;
    std::vector<cudaStream_t> streams;
};


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
__host__ GPUFilePool* geminifs_init_fds(size_t nr_files, size_t file_size);
__host__ struct geminifs_metadata* geminifs_get_metadata(int device_id);
__host__ struct geminifs_dma* geminifs_get_dma(const torch::Tensor &tensor);
__host__ bool geminifs_create_dma(const torch::Tensor& tensor);


__host__ std::vector<ControllerPtr> 
host_open_ctrls(struct geminifs_ctrl_params *ctrl_params);

__host__ std::vector<ControllerPtr> 
geminifs_nvme_host_open_ctrls(struct geminifs_ctrl_params *ctrl_params);


__host__ void 
geminifs_nvme_host_close_ctrls(std::vector<ControllerPtr> &ctrls); 


// geminifs_batch_create(int nr_device, int nr_files, size_t block_size, size_t file_size, int  cudaDevice);

__host__ GPUFile* 
geminifs_file_batch_create(std::vector<ControllerPtr> &ctrls, GPUPoolId pool_id, int nr_files, 
                        size_t block_size, size_t file_size, int cudaDevice);

__host__ DmaPtr createDmaTest(int idx, size_t block_size, int device);

__host__
DmaPtr getdeviceDmaTest(int idx, void* buffer, size_t size, int cudaDevice);




/*------------------------------xfer function-----------------------------------*/
__host__ void 
geminifs_device_xfer(GPUFile *file, uint64_t *ioaddr, 
                        size_t file_offset, size_t nbytes,
                        enum FileXferType type);
                        
__host__ void
geminifs_device_batch_xfer(GPUFile *files, uint64_t **ioaddr, 
                            size_t file_offset, size_t nbytes,
                            enum FileXferType type, size_t batch_size);

__global__ void 
__geminifs_device_batch_xfer_once(GPUFilePool *global_pool, 
                            GPUFileId file_id, uint64_t ioaddr,
                            size_t file_offset, size_t nbytes, 
                            enum FileXferType type);


__global__ void 
__geminifs_device_batch_xfer_once2(GPUFilePool *global_pool, 
                            GPUFileId file_id, cuda::std::span<uint64_t> ioaddr,
                            size_t file_offset, size_t nbytes, 
                            enum FileXferType type);




/*------------------------------xfer test-----------------------------------*/
bool geminifs_init_fds_wrapper_cuda_test(int64_t nr_files, int64_t file_size, int64_t device_id, 
                                        const std::string& mount_path, const std::string& pcie_addr);

bool geminifs_device_xfer_wrapper_test(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    const torch::Tensor &key_cache,         // shape = [max_num_block, block_size, num_heads, head_size]
    const torch::Tensor &value_cache,       // shape = [max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, enum FileXferType type);

bool geminifs_get_dma_wrapper_cuda_test(const torch::Tensor& tensor);

bool geminifs_device_xfer_wrapper_test2(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, int64_t num_layers, enum FileXferType type);

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
__host__ bool geminifs_register_tensor_with_gpu(const torch::Tensor& tensor);

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

#endif