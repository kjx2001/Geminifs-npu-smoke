#include "geminifs.cuh"
#include "geminifs_helper.h"
#include "utils.cuh"
#include "buffer.h"
#include <cuda_runtime.h>
#include <cassert>

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
    
    return true;
}

void GPUController::cleanup() {
    geminifs_debug("GPU Controller: Cleaning up device %d\n", device_id_);
    
    // Clear all DMA contexts
    clearAllDMAContexts();
    
    // Clear all NVMe controllers
    {
        std::lock_guard<std::mutex> lock(storage_mutex_);
        nvme_controllers_.clear();
    }
    
    is_initialized_.store(false);
}

// === Memory Management Methods ===

bool GPUController::registerTensorMemory(const torch::Tensor& tensor) {
    if (!isInitialized()) {
        geminifs_error("GPU Controller: Device %d is not initialized\n", device_id_);
        return false;
    }
    
    if (!validateTensor(tensor)) {
        return false;
    }
    
    uint64_t tensor_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());
    
    {
        std::lock_guard<std::mutex> lock(memory_mutex_);
        
        // Check if tensor is already registered
        if (dma_contexts_.find(tensor_ptr) != dma_contexts_.end()) {
            geminifs_warn("GPU Controller: Tensor at %p is already registered\n", tensor.data_ptr());
            return true;
        }
        
        // Create DMA context
        geminifs_dma* dma_ctx = createDMAContext(tensor);
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
        
        // Clean up DMA context
        geminifs_dma* dma_ctx = it->second;
        if (dma_ctx->ioaddrs != nullptr) {
            cudaFree(dma_ctx->ioaddrs);
        }
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
        if (dma_ctx->ioaddrs != nullptr) {
            cudaFree(dma_ctx->ioaddrs);
        }
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

geminifs_dma* GPUController::createDMAContext(const torch::Tensor& tensor) {
    auto tensor_size = tensor.numel() * tensor.element_size();
    
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
    
    uint64_t* ioaddrs = nullptr;
    if (!dma_ptr->contiguous) {
        // If the ioaddr of dma is not contiguous, allocate device buffer
        cudaError_t err = cudaMalloc(&ioaddrs, sizeof(uint64_t) * dma_ptr->n_ioaddrs);
        if (err != cudaSuccess) {
            geminifs_error("GPU Controller: Failed to allocate device memory for ioaddrs: %s\n", 
                           cudaGetErrorString(err));
            return nullptr;
        }
        
        err = cudaMemcpy(ioaddrs, dma_ptr->ioaddrs, sizeof(uint64_t) * dma_ptr->n_ioaddrs, 
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            geminifs_error("GPU Controller: Failed to copy ioaddrs to device: %s\n", 
                           cudaGetErrorString(err));
            cudaFree(ioaddrs);
            return nullptr;
        }
    }
    
    geminifs_debug("GPU Controller: Created DMA context for tensor %p, size %ld, ioaddr %lx, n_ioaddrs %ld, contiguous %d\n", 
                   tensor.data_ptr(), tensor_size, dma_ptr->ioaddrs[0], dma_ptr->n_ioaddrs, dma_ptr->contiguous);
    
    return new geminifs_dma{
        .ioaddrs = ioaddrs,
        .dma_ptr = dma_ptr
    };
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
