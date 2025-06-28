#include "geminifs.cuh"
#include "geminifs_helper.h"
#include "geminifs_mem.h"
#include "utils.cuh"
#include "buffer.h"
#include <cuda_runtime.h>
#include <cassert>
#include <filesystem>
#include <cstring>
#include <algorithm>

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

// === PRP 辅助函数实现 ===

/**
 * 创建 PRP 上下文
 */
PRPContext* createPRPContext(const std::vector<uint64_t>& ioaddrs) {
    PRPContext* context = new PRPContext();
    
    if (!context->buildPRPList(ioaddrs)) {
        delete context;
        return nullptr;
    }
    
    return context;
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

/**
 * 验证 PRP 上下文
 */
bool validatePRPContext(const PRPContext* context) {
    if (!context) {
        return false;
    }
    
    if (context->num_prp_pages == 0 || !context->prp_pages || !context->prp_page_addrs) {
        return false;
    }
    
    if (context->data_size > MAX_TRANSFER_SIZE) {
        return false;
    }
    
    return true;
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
    
    // 创建 geminifs_dma 结构
    geminifs_dma* dma_ctx = new geminifs_dma();
    dma_ctx->dma_ptr = dma_ptr;
    
    // 处理 ioaddrs
    uint64_t* ioaddrs = nullptr;
    if (!dma_ptr->contiguous) {
        // If the ioaddr of dma is not contiguous, allocate device buffer
        cudaError_t err = cudaMalloc(&ioaddrs, sizeof(uint64_t) * dma_ptr->n_ioaddrs);
        if (err != cudaSuccess) {
            geminifs_error("GPU Controller: Failed to allocate device memory for ioaddrs: %s\n", 
                           cudaGetErrorString(err));
            delete dma_ctx;
            return nullptr;
        }
        
        err = cudaMemcpy(ioaddrs, dma_ptr->ioaddrs, sizeof(uint64_t) * dma_ptr->n_ioaddrs, 
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            geminifs_error("GPU Controller: Failed to copy ioaddrs to device: %s\n", 
                           cudaGetErrorString(err));
            cudaFree(ioaddrs);
            delete dma_ctx;
            return nullptr;
        }
    }
    dma_ctx->ioaddrs = ioaddrs;
    
    // 创建 PRP 上下文
    std::vector<uint64_t> ioaddr_vector;
    
    // 将 DMA 地址复制到 vector 中
    if (dma_ptr->contiguous && dma_ptr->n_ioaddrs > 0) {
        // 连续内存，计算所有页面地址
        uint64_t base_addr = dma_ptr->ioaddrs[0];
        size_t num_pages = (tensor_size + PRP_PAGE_SIZE - 1) / PRP_PAGE_SIZE;
        
        for (size_t i = 0; i < num_pages; ++i) {
            ioaddr_vector.push_back(base_addr + i * PRP_PAGE_SIZE);
        }
    } else {
        // 非连续内存，使用所有提供的地址
        for (size_t i = 0; i < dma_ptr->n_ioaddrs; ++i) {
            ioaddr_vector.push_back(dma_ptr->ioaddrs[i]);
        }
    }
    
    // 检查数据大小限制
    if (tensor_size > MAX_TRANSFER_SIZE) {
        geminifs_error("GPU Controller: Tensor size %zu exceeds maximum transfer size %zu\n", 
                      tensor_size, MAX_TRANSFER_SIZE);
        delete dma_ctx;
        return nullptr;
    }
    
    // 创建 PRP 上下文
    dma_ctx->prp_context = new PRPContext();
    if (!dma_ctx->prp_context->buildPRPList(ioaddr_vector)) {
        geminifs_error("GPU Controller: Failed to build PRP list for tensor\n");
        delete dma_ctx;
        return nullptr;
    }
    
    geminifs_debug("GPU Controller: Created DMA context for tensor %p, size %zu, transfer_type: %s, "
                   "ioaddr 0x%lx, n_ioaddrs %zu, contiguous %d, prp_pages %zu\n", 
                   tensor.data_ptr(), tensor_size, 
                   getPRPTransferTypeString(dma_ctx->prp_context->transfer_type),
                   dma_ptr->ioaddrs[0], dma_ptr->n_ioaddrs, dma_ptr->contiguous,
                   dma_ctx->prp_context->num_prp_pages);
    
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
