#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <sys/types.h> 
#include <sys/stat.h>
#include <sys/resource.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include <ctrl.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#include <sys/file.h>
#include <fcntl.h>

#include <torch/library.h>
#include <torch/torch.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>          // 包含 device_of 和其他张量工具
#include <thrust/device_ptr.h>
#include <cuda/std/span>


#include "buffer.h"
#include "geminifs.h"
#include "geminifs_helper.h"
#include "torch/types.h"
#include "utils.cuh"
#include "nvm_error.h"
#include "file.cuh"
#include "utils.cuh"
#include "geminifs.cuh"





static char snvme_control_path[] = "/dev/snvm_control";
static char sys_config_path[] = "/mnt/sys_GPU_NVMe_topology.json";

NVMeController::NVMeController(const nvme_ctrl_param& params) : is_initialized_(false) {
    // Set mount path
    mount_path = params.mount_path;
    
    // Create mount directory if it doesn't exist
    std::filesystem::create_directories(mount_path);
    
    // Initialize single controller using the provided PCI address
    controller = open_single_controller(params.pci_addr, params);
    
    // Initialize file manager with log file in the controller's actual mount path
    std::string log_file_path = controller->dev_mount_path + "/nvme_file_log.dat";
    file_manager = std::make_unique<FileManager>(log_file_path, 1000); // 1000 is persistence threshold
    
    // Set initialization state to true after successful initialization
    is_initialized_ = true;
}

NVMeController::~NVMeController() {
    // Clean up device files first
    cleanup_device_files();
    
    if (file_manager) {
        // File manager will automatically clean up resources
        file_manager.reset();
    }
    if (controller) {
        // Close the controller
        controller.reset();
    }

    // Destructor automatically cleans up smart pointers
    // No explicit cleanup needed for shared_ptr and unique_ptr
}

__host__
void * NVMeController::g_open(std::string filename, size_t file_size, uint32_t o_flag)
{
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("g_open: NVMeController is not properly initialized. Please ensure the constructor completed successfully.\n");
        return nullptr;
    }
    
    // Check if the file size is a multiple of the ctrl block size
    assert(file_size % controller->blk_size == 0);
    
    // Check if both O_HOST and O_DEVICE flags are set (invalid)
    if ((o_flag & O_HOST) && (o_flag & O_DEVICE)) {
        geminifs_error("g_open: Cannot specify both O_HOST and O_DEVICE flags\n");
        return nullptr;
    }
    
    // Default to O_HOST if no flag is specified
    if (!(o_flag & O_HOST) && !(o_flag & O_DEVICE)) {
        o_flag |= O_HOST;
    }
    
    geminifs_debug("g_open: Opening file '%s' with size %zu bytes, flags 0x%x\n", 
                  filename.c_str(), file_size, o_flag);
    
    // Check if file exists in the file_manager log
    NVMeFileDesc file_desc;
    bool file_exists_in_log = file_manager->getFileByFilename(filename, file_desc);
    
    void* result_fd = nullptr;
    
    if (file_exists_in_log) {
        geminifs_debug("g_open: File '%s' found in log at slot %u\n", 
                      filename.c_str(), file_desc.slot_index);
        
        // Check if the file size matches the requested size
        if (file_desc.size != file_size) {
            geminifs_error("g_open: File '%s' exists with size %zu bytes, but requested size is %zu bytes. "
                          "Please delete the existing file if you want to create a new one with different size.\n", 
                          filename.c_str(), file_desc.size, file_size);
            return nullptr;
        }
        // Open the existing file
        if (o_flag & O_HOST) {
            // Open for host-side operations
            std::filesystem::path file_path = controller->dev_mount_path;
            file_path = file_path / filename;
            
            // Check if physical file exists
            if (!std::filesystem::exists(file_path)) {
                geminifs_error("g_open: File '%s' exists in log but physical file not found. Removing from log.\n", filename.c_str());
                // Remove the inconsistent record from log
                file_manager->deleteFile(filename);
                return nullptr;
            } else {
                // Open existing file using managed version
                result_fd = host_file_open_managed(file_path.string(), o_flag);
                if (result_fd == nullptr) {
                    geminifs_error("g_open: Failed to open existing file '%s'\n", file_path.c_str());
                    return nullptr;
                }
            }
        } else if (o_flag & O_DEVICE) {
            // Open for device-side operations - open existing file and copy to GPU
            result_fd = device_file_open_managed(filename, file_size);
            if (result_fd == nullptr) {
                geminifs_error("g_open: Failed to open device file '%s'\n", filename.c_str());
                return nullptr;
            }
        }
    } else {
        geminifs_debug("g_open: File '%s' not found in log, creating new file\n", filename.c_str());
        
        // File doesn't exist in log, check if physical file exists
        std::filesystem::path file_path = controller->dev_mount_path;
        file_path = file_path / filename;
        
        if (std::filesystem::exists(file_path)) {
            geminifs_debug("g_open: Physical file exists but not in log, recreating and adjusting size\n");
            // Remove existing physical file as required by spec
            std::filesystem::remove(file_path);
        }
        
        // Create new file
        if (o_flag & O_HOST) {
            // Create for host-side operations
            result_fd = host_file_create_managed(controller->page_size, file_size, filename);
            if (result_fd == nullptr) {
                geminifs_error("g_open: Failed to create host file '%s'\n", filename.c_str());
                return nullptr;
            }
            
            // Create file record in log
            NVMeFileDesc new_desc;
            if (!file_manager->createFile(filename, new_desc, file_size)) {
                geminifs_error("g_open: Failed to create file record in log for '%s'\n", filename.c_str());
                // Clean up the created file using managed close function
                host_file_close_managed((host_fd_t)result_fd);
                std::filesystem::remove(file_path);
                return nullptr;
            }
            
            geminifs_debug("g_open: Created new file '%s' with log slot %u\n", 
                          filename.c_str(), new_desc.slot_index);
        } else if (o_flag & O_DEVICE) {
            // Create for device-side operations
            result_fd = device_file_create_managed(controller->page_size, file_size, filename);
            if (result_fd == nullptr) {
                geminifs_error("g_open: Failed to create device file '%s'\n", filename.c_str());
                return nullptr;
            }
            
            // Create file record in log
            NVMeFileDesc new_desc;
            if (!file_manager->createFile(filename, new_desc, file_size)) {
                geminifs_error("g_open: Failed to create file record in log for '%s'\n", filename.c_str());
                // Clean up the created device file
                device_file_close_managed((dev_fd_t)result_fd);
                std::filesystem::remove(file_path);
                return nullptr;
            }
            
            geminifs_debug("g_open: Created new device file '%s' with log slot %u\n", 
                          filename.c_str(), new_desc.slot_index);
        }
    }
    
    geminifs_debug("g_open: Successfully opened file '%s', returning fd %p\n", 
                  filename.c_str(), result_fd);
    
    return result_fd;
}

// Helper function for binary bit counting (needed by NVMeController member functions)
static int one_nr__of__binary_int(unsigned long long i) {
    int count = 0;
    while (i != 0) {
        if ((i & 1) == 1)
            count++;
        i = i >> 1;
    }
    return count;
}

/**
 * NVMeController member function to create a file with automatic FileManager integration
 */
host_fd_t NVMeController::host_file_create_managed(int block_size, size_t file_size, const std::string& filename) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("host_file_create_managed: NVMeController is not properly initialized\n");
        return nullptr;
    }
    
    assert(file_size % block_size == 0);

    auto nvpage_size = controller->page_size;
    assert(block_size % nvpage_size == 0);

    auto hdr_size = ROUND_UP(sizeof(struct geminiFS_hdr) + 
                                    sizeof(nvme_ofst_t) * (file_size / block_size), block_size);

    // Allocate host memory for the header
    struct geminiFS_hdr *hdr = (struct geminiFS_hdr *)malloc(hdr_size);
    if (!hdr) {
        geminifs_error("host_file_create_managed: Failed to allocate memory for header\n");
        return nullptr;
    }
    
    std::filesystem::path dev_mount_path(controller->dev_mount_path);
    std::filesystem::path dir_path = dev_mount_path;
    std::filesystem::create_directories(dir_path);
    std::filesystem::path file_path = dir_path / filename;
    
    // Create the file structure
    hdr->magic_num = the_geminiFS_magic.magic_num;
    hdr->virtual_space_size = ROUND_UP(file_size, block_size);
    hdr->block_bit = one_nr__of__binary_int(block_size - 1);
    hdr->nr_l1 = file_size / block_size;
    hdr->first_block_base = hdr_size;
    
    // Create the physical file
    int fd = open(file_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        // Provide detailed error information
        geminifs_error("host_file_create_managed: Failed to create file '%s'\n", file_path.c_str());
        return nullptr;
    }
    
    // Allocate space for the file
    if (fallocate(fd, 0, 0, hdr->first_block_base + hdr->virtual_space_size) != 0) {
        geminifs_error("host_file_create_managed: Failed to allocate space for file '%s'\n", file_path.c_str());
        close(fd);
        free(hdr);
        return nullptr;
    }
    
    // Write the header
    if (lseek(fd, 0, SEEK_SET) == (off_t)(-1)) {
        geminifs_error("host_file_create_managed: Failed to seek to beginning of file\n");
        close(fd);
        free(hdr);
        return nullptr;
    }
    
    if (write(fd, hdr, sizeof(*hdr)) != sizeof(*hdr)) {
        geminifs_error("host_file_create_managed: Failed to write header to file\n");
        close(fd);
        free(hdr);
        return nullptr;
    }
    
    hdr->fd = fd;
    
    // Refine NVMe offsets
    host_refine_nvmeofst(hdr);
    
    // Register with FileManager for automatic cleanup
    if (file_manager != nullptr) {
        file_manager->registerOpenFile(hdr, filename, hdr_size);
    }
    
    geminifs_debug("host_file_create_managed: Created file '%s' with size %zu, hdr_size %zu\n", 
                   filename.c_str(), file_size, hdr_size);
    
    return hdr;
}

/**
 * NVMeController member function to open a file with automatic FileManager integration
 */
host_fd_t NVMeController::host_file_open_managed(const std::string& filepath, uint32_t o_flag) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("host_file_open_managed: NVMeController is not properly initialized\n");
        return nullptr;
    }
    
    // Remove O_DEVICE and O_HOST flags before passing to host_open_geminifs_file
    // These are GeminiFS-specific flags that shouldn't be passed to the underlying file operations
    uint32_t file_flags = o_flag & ~(O_HOST | O_DEVICE);
    
    // For now, we'll still use the existing host_open_geminifs_file function
    // In the future, this could be extended to accept different flags
    host_fd_t result = host_open_geminifs_file(filepath.c_str());
    
    // Check for read-only file access errors
    if (result == nullptr) {
        // Check if the failure might be due to read-only access requirements
        if ((file_flags & O_ACCMODE) == O_RDONLY) {
            geminifs_error("host_file_open_managed: Failed to open file '%s' in read-only mode. "
                          "GeminiFS files currently require read-write access for proper operation.\n", 
                          filepath.c_str());
        } else {
            geminifs_error("host_file_open_managed: Failed to open file '%s'\n", filepath.c_str());
        }
        return nullptr;
    }
    
    if (file_manager != nullptr) {
        // Calculate the size of the allocated header for registration
        size_t hdr_size = ROUND_UP(sizeof(struct geminiFS_hdr) + sizeof(nvme_ofst_t) * result->nr_l1, result->first_block_base);
        file_manager->registerOpenFile(result, filepath, hdr_size);
    }
    
    return result;
}

/**
 * NVMeController member function to close a file with automatic FileManager cleanup
 */    
void NVMeController::host_file_close_managed(host_fd_t fd) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("host_file_close_managed: NVMeController is not properly initialized\n");
        return;
    }
    
    if (file_manager != nullptr) {
        file_manager->unregisterOpenFile(fd);
    }
    
    // Close and free the file descriptor
    close(fd->fd);
    free(fd);
}

/**
 * NVMeController member function to create a device file with host-to-device mapping
 */
dev_fd_t NVMeController::device_file_create_managed(int block_size, size_t file_size, const std::string& filename) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("device file create managed: NVMeController is not properly initialized\n");
        return nullptr;
    }
    
    // First create the host file
    host_fd_t host_fd = host_file_create_managed(block_size, file_size, filename);
    if (host_fd == nullptr) {
        geminifs_error("device file create managed: Failed to create host file '%s'\n", filename.c_str());
        return nullptr;
    }
    
    // Calculate header size
    size_t hdr_size = ROUND_UP(sizeof(struct geminiFS_hdr) + 
                               sizeof(nvme_ofst_t) * (file_size / block_size), block_size);
    
    // Copy host file descriptor to device
    dev_fd_t device_fd = copy_host_fd_to_device(host_fd, hdr_size);
    if (device_fd == nullptr) {
        geminifs_error("device file create managed: Failed to copy host file to device for '%s'\n", filename.c_str());
        // Clean up the host file
        host_file_close_managed(host_fd);
        return nullptr;
    }
    
    // Store the mapping for management
    {
        std::lock_guard<std::mutex> lock(device_files_mtx_);
        device_files_.emplace_back(host_fd, device_fd, hdr_size, filename);
    }
    
    geminifs_debug("device file create managed: Created device file '%s' with device_fd %p\n", 
                   filename.c_str(), device_fd);
    
    return device_fd;
}

/**
 * NVMeController private function to open an existing file as a device file
 */
dev_fd_t NVMeController::device_file_open_managed(const std::string& filename, size_t file_size) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("device file open managed: NVMeController is not properly initialized\n");
        return nullptr;
    }
    
    // Build file path
    std::filesystem::path file_path = controller->dev_mount_path;
    file_path = file_path / filename;
    
    // Check if physical file exists
    if (!std::filesystem::exists(file_path)) {
        geminifs_error("device file open managed: Physical file '%s' not found\n", file_path.c_str());
        return nullptr;
    }
    
    // Open the host file
    host_fd_t host_fd = host_file_open_managed(file_path.string(), O_RDWR);
    if (host_fd == nullptr) {
        geminifs_error("device file open managed: Failed to open host file '%s'\n", file_path.c_str());
        return nullptr;
    }
    
    // Validate file size
    if (host_fd->virtual_space_size != file_size) {
        geminifs_error("device file open managed: File size mismatch. Expected %zu, got %zu\n", 
                       file_size, host_fd->virtual_space_size);
        host_file_close_managed(host_fd);
        return nullptr;
    }
    
    // Calculate header size
    size_t hdr_size = ROUND_UP(sizeof(struct geminiFS_hdr) + 
                               sizeof(nvme_ofst_t) * host_fd->nr_l1, host_fd->first_block_base);
    
    // Copy host file descriptor to device
    dev_fd_t device_fd = copy_host_fd_to_device(host_fd, hdr_size);
    if (device_fd == nullptr) {
        geminifs_error("device file open managed: Failed to copy host file to device for '%s'\n", filename.c_str());
        host_file_close_managed(host_fd);
        return nullptr;
    }
    
    // Store the mapping for management
    {
        std::lock_guard<std::mutex> lock(device_files_mtx_);
        device_files_.emplace_back(host_fd, device_fd, hdr_size, filename);
    }
    
    geminifs_debug("device file open managed: Opened device file '%s' with device_fd %p\n", 
                   filename.c_str(), device_fd);
    
    return device_fd;
}

/**
 * NVMeController private function to close a device file and clean up resources
 */
void NVMeController::device_file_close_managed(dev_fd_t device_fd) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("device file close managed: NVMeController is not properly initialized\n");
        return;
    }
    
    std::lock_guard<std::mutex> lock(device_files_mtx_);
    
    // Find the device file handle
    auto it = std::find_if(device_files_.begin(), device_files_.end(),
                          [device_fd](const DeviceFileHandle& handle) {
                              return handle.device_fd == device_fd;
                          });
    
    if (it != device_files_.end()) {
        geminifs_debug("device file close managed: Closing device file '%s'\n", it->filename.c_str());
        
        // Free device memory
        cudaError_t err = cudaFree(device_fd);
        if (err != cudaSuccess) {
            geminifs_error("device file close managed: Failed to free device memory: %s\n", 
                          cudaGetErrorString(err));
        }
        
        // Close host file
        host_file_close_managed(it->host_fd);
        
        // Remove from tracking
        device_files_.erase(it);
    } else {
        geminifs_error("device file close managed: Device file descriptor %p not found\n", device_fd);
    }
}

/**
 * NVMeController private helper function to copy host file descriptor to device memory
 */
dev_fd_t NVMeController::copy_host_fd_to_device(host_fd_t host_fd, size_t hdr_size) {
    void* device_fd = nullptr;
    
    // Allocate device memory
    cudaError_t err = cudaMalloc(&device_fd, hdr_size);
    if (err != cudaSuccess) {
        geminifs_error("copy host fd to device: Failed to allocate device memory: %s\n", 
                      cudaGetErrorString(err));
        return nullptr;
    }
    
    // Copy header from host to device
    err = cudaMemcpy(device_fd, host_fd, hdr_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("copy host fd to device: Failed to copy memory to device: %s\n", 
                      cudaGetErrorString(err));
        cudaFree(device_fd);
        return nullptr;
    }
    
    // Synchronize to ensure copy is complete
    err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        geminifs_error("copy host fd to device: Failed to synchronize stream: %s\n", 
                      cudaGetErrorString(err));
        cudaFree(device_fd);
        return nullptr;
    }
    
    geminifs_debug("copy host fd to device: Successfully copied %zu bytes to device %p\n", 
                   hdr_size, device_fd);
    
    return device_fd;
}

/**
 * NVMeController private helper function to clean up all device files
 */
void NVMeController::cleanup_device_files() {
    std::lock_guard<std::mutex> lock(device_files_mtx_);
    
    geminifs_debug("cleanup device files: Cleaning up %zu device files\n", device_files_.size());
    
    for (auto& handle : device_files_) {
        geminifs_debug("cleanup device files: Cleaning up device file '%s'\n", handle.filename.c_str());
        
        // Free device memory
        if (handle.device_fd != nullptr) {
            cudaError_t err = cudaFree(handle.device_fd);
            if (err != cudaSuccess) {
                geminifs_error("cleanup device files: Failed to free device memory for '%s': %s\n", 
                              handle.filename.c_str(), cudaGetErrorString(err));
            }
        }
    }
    
    device_files_.clear();
}

    
// Global helper functions for checking system components
static inline bool check_snvme_control_exists() {
    if (access(snvme_control_path, F_OK) != 0) {
        geminifs_error("SNVM control device '%s' does not exist. Please ensure the kernel module is properly installed.\n", snvme_control_path);
        return false;
    }
    return true;
}

static inline bool check_sys_config_exists() {
    if (access(sys_config_path, F_OK) != 0) {
        geminifs_error("Sys GPU-NVMe topology '%s' does not exist. Please ensure the kernel module is properly installed.\n", sys_config_path);
        return false;
    }
    return true;
}

static inline void host_close_ctrls(struct geminifs_metadata *metadata){
    for (auto &ctrl : metadata->ctrls) {
        ctrl.reset();
    }
    metadata->ctrls.clear();
}

void host_close_all(){
    for (auto &kv : global_metadata) {
        auto metadata = kv.second;
        if (metadata->is_init) {
            host_close_ctrls(metadata);
            metadata->is_init = false;
        }
    }
    global_metadata.clear();
}

// force close all controllers when the program exits
__attribute__((destructor))
static void clean_geminifs() {
    host_close_all();
}



static inline void *host_batch_create(std::vector<ControllerPtr> &ctrls, GPUPoolId pool_id,
                                        int block_size, int nr_files, size_t file_size) {
    assert(file_size % block_size == 0);
    assert(nr_files > 0);

    auto nvpage_size = ctrls[0]->page_size;
    assert(block_size % nvpage_size == 0);

    void *host_fds_base;
    auto hdr_size = ROUND_UP(sizeof(struct geminiFS_hdr) + 
                                    sizeof(nvme_ofst_t) * (file_size / block_size), block_size);
    auto nr_device = ctrls.size();
    cuda_check_error(cudaMallocHost(&host_fds_base, hdr_size * nr_files * nr_device));
    
  
    for (int idx = 0; idx < nr_device; idx++) {
        // fixme
        auto ctrl = ctrls[idx].get();
        std::filesystem::path dev_mount_path(ctrl->dev_mount_path);
        std::filesystem::path dir_path = dev_mount_path / std::to_string(pool_id) / std::to_string(idx);
        std::filesystem::create_directories(dir_path);

        for (uint32_t file_idx = 0; file_idx < nr_files; file_idx++) {
            std::filesystem::path file_path = dir_path / std::to_string(file_idx);
            // geminifs_debug("create file %s\n", file_path.c_str());
            auto hdr = (struct geminiFS_hdr *)(
                                (uintptr_t)host_fds_base + (file_idx * nr_device + idx) * hdr_size);
            host_create_geminifs_file(hdr, std::string(file_path).c_str(), block_size, file_size);
            geminifs_debug("hdr info: block_size %d, file_size %lu, first_block_base %ld, file_path %s\n", 
                            hdr->block_bit, hdr->virtual_space_size, hdr->first_block_base, file_path.c_str());
        }

    }
    
    void *dev_fds_base;
    cuda_check_error(cudaMalloc(&dev_fds_base, hdr_size * nr_files * nr_device));
    cuda_check_error(cudaMemcpy(dev_fds_base, host_fds_base, hdr_size * nr_files * nr_device, cudaMemcpyHostToDevice));
    cuda_check_error(cudaStreamSynchronize(0));
    cuda_check_error(cudaFreeHost(host_fds_base));

    geminifs_debug("geminifs_batch_create: allocated %ld bytes for device fds base\n", 
                    hdr_size * nr_files * nr_device);

    return dev_fds_base;
}




static 
NVMeFile *device_batch_open(std::vector<ControllerPtr> &ctrls, GPUPoolId pool_id, size_t block_size, 
                            int nr_files, size_t file_size, int cudaDevice) {
    struct nvme_cmd__addr *total_nvme_cmds;
    QueueAcquireHelper *queue_acquire_helper;
    NVMeFile *files;
    uint16_t *total_cids;
    uint16_t *total_sq_poss;
    void **ctrl_ptrs;

    // cuda_check_error(cudaSetDevice(cudaDevice));
    void *dev_fds_base = host_batch_create(ctrls, pool_id, block_size, nr_files, file_size);
    size_t nvme_page_size = ctrls[0]->page_size;
    size_t per_file_size = file_size / ctrls.size();
    size_t hdr_size = ROUND_UP(sizeof(struct geminiFS_hdr) + sizeof(nvme_ofst_t) * (file_size / block_size), block_size);
    size_t max_nvme_cmds = file_size / nvme_page_size;
    size_t nr_device = ctrls.size();
    size_t total_cnt = nr_files * nr_device;
    
    cuda_check_error(cudaMalloc(&files, sizeof(NVMeFile) * total_cnt));
    cuda_check_error(cudaMalloc(&total_cids, sizeof(uint16_t) * max_nvme_cmds * total_cnt));
    cuda_check_error(cudaMalloc(&total_sq_poss, sizeof(uint16_t) * max_nvme_cmds * total_cnt));
    cuda_check_error(cudaMalloc(&total_nvme_cmds, sizeof(struct nvme_cmd__addr) * max_nvme_cmds * total_cnt));
    cuda_check_error(cudaMalloc(&queue_acquire_helper, sizeof(QueueAcquireHelper) * nr_device));
    cuda_check_error(cudaMallocManaged(&ctrl_ptrs, sizeof(void *) * nr_device));

    for (int dev_idx = 0; dev_idx < nr_device; dev_idx++) {
        auto *dev_ctrl = ctrls[dev_idx]->d_ctrl_ptr;
        ctrl_ptrs[dev_idx] = dev_ctrl;
    }

    // assume that all the devices have the same block size
    auto block_log = ctrls[0]->h_qps[0]->block_size_log;
    auto nr_queues = ctrls[0]->n_qps;

    RUN_ON_DEVICE({
        for (int dev_idx = 0; dev_idx < nr_device; dev_idx++) {
            auto q_helper = queue_acquire_helper + dev_idx;
            new (q_helper) QueueAcquireHelper(nr_queues);
        }
        for (int file_idx = 0; file_idx < nr_files; file_idx ++){    
            for (int dev_idx = 0; dev_idx < nr_device; dev_idx++) {
                auto q_helper = queue_acquire_helper + dev_idx;
                auto this_file = files + file_idx * nr_device + dev_idx;
                auto dev_ctrl = (Controller *)ctrl_ptrs[dev_idx];
                auto *hdr = (struct geminiFS_hdr *)((uintptr_t)dev_fds_base  
                                                        + (file_idx * nr_device + dev_idx) * hdr_size);
                new (this_file) NVMeFile(dev_ctrl, hdr);
                this_file->max_nvme_cmds = max_nvme_cmds;
                this_file->nvme_cmds = total_nvme_cmds + (file_idx * nr_device + dev_idx) * max_nvme_cmds;
                this_file->cids = total_cids + (file_idx * nr_device + dev_idx) * max_nvme_cmds;
                this_file->sq_poss = total_sq_poss + (file_idx * nr_device + dev_idx) * max_nvme_cmds;
                this_file->hqps_block_size_log = block_log;
                this_file->queue_acquire_helper = q_helper;
                this_file->file_size = per_file_size;
                this_file->nvme_page_size = dev_ctrl->page_size;
                this_file->block_size = block_size;
            }
        }
    })
    cudaFree(ctrl_ptrs);

    return files;

}


struct DMAInfo{
    uint64_t *vaddr;
    uint64_t ioaddr_base;
    DmaPtr dma_ptr;
};

__host__ GPUFile* 
geminifs_batch_create(std::vector<ControllerPtr> &ctrls, GPUPoolId pool_id, int nr_files, 
                        size_t block_size, size_t file_size, int cudaDevice){
    
    // cuda_check_error(cudaSetDevice(cudaDevice));
    
    auto nr_device = ctrls.size();
    auto nv_file_size = file_size / nr_device;
    auto nvme_page_size = ctrls[0].get()->page_size;
    NVMeFile *nv_files = device_batch_open(ctrls, pool_id, block_size, nr_files, file_size, cudaDevice);

    GPUFile * gpu_files__ptr;
    void *dma_info__ptr;    

    auto dma__per_nvfile = GPU_PAGE_SIZE / (sizeof(uint64_t) * nv_file_size / nvme_page_size);
    auto total_dma_size = ROUND_UP(nr_files, dma__per_nvfile) / dma__per_nvfile;
    cuda_check_error(cudaMalloc(&gpu_files__ptr, sizeof(GPUFile) * nr_files));
    cuda_check_error(cudaMallocManaged(&dma_info__ptr, sizeof(struct DMAInfo) * total_dma_size));

    for (auto idx = 0;idx < total_dma_size; idx ++) {
        auto *dma_info = (struct DMAInfo *)((uintptr_t)dma_info__ptr + idx * sizeof(struct DMAInfo));
        dma_info->dma_ptr = createDma(ctrls[idx % nr_device].get()->ctrl, 
                                        GPU_PAGE_SIZE, cudaDevice);
        dma_info->vaddr = (uint64_t *)dma_info->dma_ptr->vaddr;
        dma_info->ioaddr_base = dma_info->dma_ptr->ioaddrs[0];
    }

    RUN_ON_DEVICE({
        for (int idx = 0; idx < nr_files; idx++){
            auto gpu_file = gpu_files__ptr + idx;
            auto dma_info = (struct DMAInfo *)((uintptr_t)dma_info__ptr + 
                                                    (idx / dma__per_nvfile) * sizeof(struct DMAInfo));

            // auto dma_info = (struct DMAInfo *)((uintptr_t)dma_info__ptr + idx * sizeof(struct DMAInfo));
            new (gpu_file)GPUFile(nv_files + idx * nr_device, nr_device, file_size);
            // new (gpu_file)GPUFile(nv_files + idx * nr_device, static_cast<size_t>(file_size));
            gpu_file->nvme_page_size = nvme_page_size;
            gpu_file->block_size = block_size;
            gpu_file->prp_list__of_total_pages_vaddr = dma_info->vaddr + 
                                                        (idx % dma__per_nvfile) * (nv_file_size / nvme_page_size);
            gpu_file->prp_list_ioaddr_base = dma_info->ioaddr_base + 
                                                (idx % dma__per_nvfile) * (nv_file_size / nvme_page_size) * sizeof(uint64_t);
            gpu_file->file_id = idx;
            // geminifs_debug("allocate %d gpu_file, file_id is: %lld\n", idx, gpu_file->file_id);
        }
    });

    return gpu_files__ptr;
}

__host__ GPUFile* 
geminifs_file_batch_create(std::vector<ControllerPtr> &ctrls, GPUPoolId pool_id, int nr_files, 
                        size_t block_size, size_t file_size, int cudaDevice){   
    auto nr_device = ctrls.size();
    auto nv_file_size = file_size / nr_device;
    auto nvme_page_size = ctrls[0].get()->page_size;
    NVMeFile *nv_files = device_batch_open(ctrls, pool_id, block_size, nr_files, file_size, cudaDevice);

    GPUFile * gpu_files__ptr;
    void *dma_info__ptr;    

    auto dma__per_nvfile = GPU_PAGE_SIZE / (sizeof(uint64_t) * nv_file_size / nvme_page_size);
    auto total_dma_size = ROUND_UP(nr_files, dma__per_nvfile) / dma__per_nvfile;
    cuda_check_error(cudaMalloc(&gpu_files__ptr, sizeof(GPUFile) * nr_files));
    cuda_check_error(cudaMallocManaged(&dma_info__ptr, sizeof(struct DMAInfo) * total_dma_size));

    for (auto idx = 0;idx < total_dma_size; idx ++) {
        auto *dma_info = (struct DMAInfo *)((uintptr_t)dma_info__ptr + idx * sizeof(struct DMAInfo));
        dma_info->dma_ptr = createDma(ctrls[idx % nr_device].get()->ctrl, 
                                        GPU_PAGE_SIZE, cudaDevice);
        dma_info->vaddr = (uint64_t *)dma_info->dma_ptr->vaddr;
        dma_info->ioaddr_base = dma_info->dma_ptr->ioaddrs[0];
    }

    RUN_ON_DEVICE({
        for (int idx = 0; idx < nr_files; idx++){
            auto gpu_file = gpu_files__ptr + idx;
            auto dma_info = (struct DMAInfo *)((uintptr_t)dma_info__ptr + 
                                                    (idx / dma__per_nvfile) * sizeof(struct DMAInfo));

            // auto dma_info = (struct DMAInfo *)((uintptr_t)dma_info__ptr + idx * sizeof(struct DMAInfo));
            new (gpu_file)GPUFile(nv_files + idx * nr_device, nr_device, file_size);
            // new (gpu_file)GPUFile(nv_files + idx * nr_device, static_cast<size_t>(file_size));
            gpu_file->nvme_page_size = nvme_page_size;
            gpu_file->block_size = block_size;
            gpu_file->prp_list__of_total_pages_vaddr = dma_info->vaddr + 
                                                        (idx % dma__per_nvfile) * (nv_file_size / nvme_page_size);
            gpu_file->prp_list_ioaddr_base = dma_info->ioaddr_base + 
                                                (idx % dma__per_nvfile) * (nv_file_size / nvme_page_size) * sizeof(uint64_t);
            gpu_file->file_id = idx;
            // geminifs_debug("allocate %d gpu_file, file_id is: %lld\n", idx, gpu_file->file_id);
        }
    });

    return gpu_files__ptr;
}

__host__ GPUFile* 
geminifs_file_create(std::vector<ControllerPtr> &ctrls, GPUPoolId pool_id, int nr_files, 
                        size_t block_size, size_t file_size, int cudaDevice){
    
    // cuda_check_error(cudaSetDevice(cudaDevice));
    
    auto nr_device = ctrls.size();
    auto nv_file_size = file_size / nr_device;
    auto nvme_page_size = ctrls[0].get()->page_size;
    NVMeFile *nv_files = device_batch_open(ctrls, pool_id, block_size, nr_files, file_size, cudaDevice);

    GPUFile * gpu_files__ptr;
    void *dma_info__ptr;    

    auto dma__per_nvfile = GPU_PAGE_SIZE / (sizeof(uint64_t) * nv_file_size / nvme_page_size);
    auto total_dma_size = ROUND_UP(nr_files, dma__per_nvfile) / dma__per_nvfile;
    cuda_check_error(cudaMalloc(&gpu_files__ptr, sizeof(GPUFile) * nr_files));
    cuda_check_error(cudaMallocManaged(&dma_info__ptr, sizeof(struct DMAInfo) * total_dma_size));

    for (auto idx = 0;idx < total_dma_size; idx ++) {
        auto *dma_info = (struct DMAInfo *)((uintptr_t)dma_info__ptr + idx * sizeof(struct DMAInfo));
        dma_info->dma_ptr = createDma(ctrls[idx % nr_device].get()->ctrl, 
                                        GPU_PAGE_SIZE, cudaDevice);
        dma_info->vaddr = (uint64_t *)dma_info->dma_ptr->vaddr;
        dma_info->ioaddr_base = dma_info->dma_ptr->ioaddrs[0];
    }

    RUN_ON_DEVICE({
        for (int idx = 0; idx < nr_files; idx++){
            auto gpu_file = gpu_files__ptr + idx;
            auto dma_info = (struct DMAInfo *)((uintptr_t)dma_info__ptr + 
                                                    (idx / dma__per_nvfile) * sizeof(struct DMAInfo));

            // auto dma_info = (struct DMAInfo *)((uintptr_t)dma_info__ptr + idx * sizeof(struct DMAInfo));
            new (gpu_file)GPUFile(nv_files + idx * nr_device, nr_device, file_size);
            // new (gpu_file)GPUFile(nv_files + idx * nr_device, static_cast<size_t>(file_size));
            gpu_file->nvme_page_size = nvme_page_size;
            gpu_file->block_size = block_size;
            gpu_file->prp_list__of_total_pages_vaddr = dma_info->vaddr + 
                                                        (idx % dma__per_nvfile) * (nv_file_size / nvme_page_size);
            gpu_file->prp_list_ioaddr_base = dma_info->ioaddr_base + 
                                                (idx % dma__per_nvfile) * (nv_file_size / nvme_page_size) * sizeof(uint64_t);
            gpu_file->file_id = idx;
            // geminifs_debug("allocate %d gpu_file, file_id is: %lld\n", idx, gpu_file->file_id);
        }
    });

    return gpu_files__ptr;
}

ControllerPtr NVMeController::open_single_controller(const std::string& pci_addr, const nvme_ctrl_param& params) {
    // Create mount path for this specific controller
    std::filesystem::path mount_path(params.mount_path);
    std::filesystem::path this_mount_path = mount_path / 
        ("cuda" + std::to_string(params.cudaDevice) + '-' + pci_addr);

    if (!std::filesystem::exists(this_mount_path)) {
        std::filesystem::create_directories(this_mount_path);
    }
    
    // Create and initialize controller
    ControllerPtr ctrl = std::make_shared<Controller>(
        snvme_control_path, 
        pci_addr.c_str(), 
        this_mount_path.c_str(), 
        params.ns_id, 
        params.cudaDevice, 
        params.queueDepth, 
        params.numQueues);
    return ctrl;
}

std::vector<ControllerPtr> host_open_ctrls(struct geminifs_ctrl_params *ctrl_params){
    

    std::vector<ControllerPtr> ctrls;

    // Check if SNVM control device exists
    if (!check_snvme_control_exists()) {
        geminifs_error("Failed to initialize controllers: SNVM kernel module not loaded\n");
        return ctrls;  // Return empty vector to indicate error
    }

    // Check if Sys config file exists
    if (!check_sys_config_exists()) {
        geminifs_error("Failed to initialize controllers: SNVM kernel module not loaded\n");
        return ctrls;  // Return empty vector to indicate error
    }

    std::filesystem::create_directories(ctrl_params->mount_path);
    std::filesystem::path mount_path(ctrl_params->mount_path);

    for (size_t idx = 0; idx < ctrl_params->pci_addr.size(); idx++){
        std::filesystem::path this_mount_path = mount_path / 
                                    ("cuda" + std::to_string(ctrl_params->cudaDevice) + '-' + std::to_string(idx));
        if (!std::filesystem::exists(this_mount_path)) {
            std::filesystem::create_directories(this_mount_path);
        }
        auto ctrl = new Controller(
            snvme_control_path,
            ctrl_params->pci_addr[idx].c_str(),
            this_mount_path,
            ctrl_params->ns_id,
            ctrl_params->cudaDevice,
            ctrl_params->queueDepth,
            ctrl_params->numQueues);
        nvm_info("Opening controller %ld: pci addr %s, mount path %s", 
                            idx, ctrl_params->pci_addr[idx].c_str(), this_mount_path.c_str());  
        ctrls.push_back(std::shared_ptr<Controller>(ctrl));
    }

#ifdef DEBUG
    for (size_t idx = 0; idx < ctrl_params->pci_addr.size(); idx++) {
      auto ctrl = ctrls[idx].get();
      nvm_debug("Opening controller %d: dev path %s, mount path %s", idx,
               ctrl->dev_path, ctrl->dev_mount_path);
    }
#endif

    return std::move(ctrls);
}



void  geminifs_nvme_host_close_ctrls(std::vector<ControllerPtr> &ctrls)
{
    for (auto &ctrl : ctrls) {
        if (ctrl) {
            ctrl.reset();
        }
    }
    ctrls.clear();
}


static inline geminifs_metadata* __geminifs_init(struct geminifs_ctrl_params &ctrl_params, 
                                                size_t nr_files, size_t file_size, size_t file_block_size) {
    assert(nr_files > 0);

    // check current device 
    int current_device;
    cuda_check_error(cuda_getDevice(&current_device));
    if (current_device != ctrl_params.cudaDevice) {
        geminifs_warn("geminifs_init_fds_wrapper_cuda: current device %d is not the same as ctrl_params.cudaDevice %d\n", 
                        current_device, ctrl_params.cudaDevice);
        // cuda_check_error(cudaSetDevice(ctrl_params.cudaDevice));
    }

    geminifs_debug("geminifs_init_fds_wrapper_cuda: current device %d, ctrl_params.cudaDevice %d\n", 
                        current_device, ctrl_params.cudaDevice);
    file_size = ROUND_UP(file_size, file_block_size);
    GPUPoolId this_pool_id = (GPUPoolId)time(NULL); //unique pool id
    std::vector<ControllerPtr> ctrls = host_open_ctrls(&ctrl_params);
    auto files = geminifs_batch_create(ctrls, this_pool_id, nr_files, 
        file_block_size, file_size, ctrl_params.cudaDevice);

    GPUFilePool *pool;
    uint16_t *is_allocated;
    cuda_check_error(cudaMalloc(&pool, sizeof(GPUFilePool)));
    cuda_check_error(cudaMalloc(&is_allocated, nr_files  * sizeof(uint16_t)));
    cuda_check_error(cudaMemset(is_allocated, 0x0, nr_files * sizeof(uint16_t)));
    
    
    RUN_ON_DEVICE({
        new (pool) GPUFilePool(files, is_allocated, file_size, file_block_size, nr_files);
        pool->set_pool_id(this_pool_id);
    });

    std::vector<cudaStream_t> streams(32);
    for (size_t i = 0; i < 32; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    return new geminifs_metadata{
        .ctrls = std::move(ctrls),
        .is_init = true,
        .global_pool = thrust::device_pointer_cast(pool),
        .pool_id = this_pool_id,
        .file_size = file_size,
        .file_block_size = file_block_size,
        .streams = std::move(streams)
    };
}


/*----------------------Xfer-------------------*/
__global__ void 
__geminifs_device_batch_xfer(GPUFilePool *global_pool, 
                            cuda::std::span<GPUFileId> file_ids,
                            cuda::std::span<uint64_t> ioaddr,
                            size_t file_offset, 
                            size_t nbytes, enum FileXferType type){
    size_t nr_block = gridDim.x;
    size_t nr_thread_per_block = blockDim.x;
    assert(nr_block == file_ids.size());
    assert(nr_thread_per_block == 32);
    assert(nbytes % GPU_PAGE_SIZE == 0);
    
    size_t nbytes__per_thread = std::max(nbytes / 32, GPU_PAGE_SIZE);

    int lane = my_lane_id();
    geminifs_debug("file_ids[blockIdx.x] %ld\n", file_ids[blockIdx.x]);
    auto file = global_pool->get_file(file_ids[blockIdx.x]);
    if (lane == 0) {
        file->scatter_ioaddrs(ioaddr, file_offset, nbytes);
    }

    __syncwarp();
    size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
    if (lane * nbytes__per_thread  + nbytes__per_thread > nbytes) {
        return;
    }

    if (type == FILE_XFER_READ) {
        file->read_in(this_thread_file_offset, nbytes__per_thread);
    } else {
        file->write_out(this_thread_file_offset, nbytes__per_thread);
    }
}

__global__ void 
__geminifs_device_batch_xfer_once(GPUFilePool *global_pool, 
                            GPUFileId file_id, uint64_t ioaddr,
                            size_t file_offset, size_t nbytes, 
                            enum FileXferType type){
    size_t nr_block = gridDim.x;
    size_t nr_thread_per_block = blockDim.x;
    assert(nr_block == 1);
    assert(nr_thread_per_block == 32);
    assert(nbytes % GPU_PAGE_SIZE == 0);
    
    size_t nbytes__per_thread = std::max(nbytes / 32, GPU_PAGE_SIZE);

    int lane = my_lane_id();
    auto file = global_pool->get_file(file_id);
    if (lane == 0) {
        file->scatter_ioaddrs(ioaddr, file_offset, nbytes);
    }

    __syncwarp();
    size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
    if (lane * nbytes__per_thread  + nbytes__per_thread > nbytes) {
        return;
    }

    if (type == FILE_XFER_READ) {
        file->read_in(this_thread_file_offset, nbytes__per_thread);
    } else {
        file->write_out(this_thread_file_offset, nbytes__per_thread);
    }
}

__global__ void 
__geminifs_device_batch_xfer_once2(GPUFilePool *global_pool, 
                            GPUFileId file_id, cuda::std::span<uint64_t> ioaddr,
                            size_t file_offset, size_t nbytes, 
                            enum FileXferType type){
    size_t nr_block = gridDim.x;
    size_t nr_thread_per_block = blockDim.x;
    assert(nr_block == 1);
    assert(nr_thread_per_block == 32);
    assert(nbytes % GPU_PAGE_SIZE == 0);
    
    size_t nbytes__per_thread = std::max(nbytes / 32, GPU_PAGE_SIZE);

    int lane = my_lane_id();
    auto file = global_pool->get_file(file_id);
    if (lane == 0) {
        file->scatter_ioaddrs(ioaddr, file_offset, nbytes);
    }

    __syncwarp();
    size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
    if (lane * nbytes__per_thread  + nbytes__per_thread > nbytes) {
        return;
    }

    if (type == FILE_XFER_READ) {
        file->read_in(this_thread_file_offset, nbytes__per_thread);
    } else {
        file->write_out(this_thread_file_offset, nbytes__per_thread);
    }
}

__global__ void 
__geminifs_device_batch_xfer(GPUFilePool *global_pool, 
                            cuda::std::span<GPUFileId> file_ids,
                            cuda::std::span<uint64_t> block_ids,
                            cuda::std::span<uint64_t> ioaddr, 
                            size_t per_chuck_size, // per_chuck_size = chuck_size * block_size
                            size_t file_offset, enum FileXferType type){
    size_t nr_block = gridDim.x;
    size_t nr_thread_per_block = blockDim.x;
    assert(nr_block == file_ids.size());
    assert(nr_thread_per_block == 32);
    assert(per_chuck_size % global_pool->file_block_size == 0);
    
    size_t nbytes__per_thread = std::max(per_chuck_size / 32, global_pool->file_block_size);

    int lane = my_lane_id();
    auto file = global_pool->get_file(file_ids[blockIdx.x]);
    if (lane == 0) {
        // fixme
        file->scatter_ioaddrs(ioaddr, file_offset, per_chuck_size);
        // file
    }

    __syncwarp();
    size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
    if (lane * nbytes__per_thread  + nbytes__per_thread > per_chuck_size) {
        return;
    }

    if (type == FILE_XFER_READ) {
        file->read_in(this_thread_file_offset, nbytes__per_thread);
    } else {
        file->write_out(this_thread_file_offset, nbytes__per_thread);
    }
}


__global__ void 
__geminifs_device_batch_xfer(GPUFilePool *global_pool, 
                            cuda::std::span<GPUFileId> file_ids,
                            cuda::std::span<uint64_t> block_ids,
                            uint64_t ioaddr, size_t per_chuck_size, // per_chuck_size = chuck_size * block_size
                            size_t file_offset, enum FileXferType type){
    size_t nr_block = gridDim.x;
    size_t nr_thread_per_block = blockDim.x;
    assert(nr_block == file_ids.size());
    assert(nr_thread_per_block == 32);
    assert(per_chuck_size % global_pool->file_block_size == 0);
    
    size_t nbytes__per_thread = std::max(per_chuck_size / 32, global_pool->file_block_size);
    int lane = my_lane_id();
    auto file = global_pool->get_file(file_ids[blockIdx.x]);
    
    if (lane == 0) {
        auto this_block_ioaddr = ioaddr + block_ids[blockIdx.x] * per_chuck_size;
        file->scatter_ioaddrs(this_block_ioaddr, file_offset, per_chuck_size);
    }
    __syncwarp();


    size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
    if (lane * nbytes__per_thread  + nbytes__per_thread > per_chuck_size) {
        return;
    }
    if (type == FILE_XFER_READ) {
        // geminifs_debug("read_in, file[%p]:this_thread_file_offset %ld, nbytes__per_thread %ld\n", file, this_thread_file_offset, nbytes__per_thread);
        file->read_in(this_thread_file_offset, nbytes__per_thread);
        // geminifs_debug("read_in done\n");
    } else {
        file->write_out(this_thread_file_offset, nbytes__per_thread);
    }
    global_pool->put_file(file_ids[blockIdx.x]);
}


/**
 * 检查指针是否为CUDA设备指针
 */
static inline bool is_device_pointer(const void* ptr, const char* error_msg = nullptr) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    
    bool is_device = (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice);
    
    if (error_msg != nullptr && !is_device) {
        TORCH_CHECK(false, error_msg);
    }
    
    return is_device;
}

static inline bool is_aligned(const uint64_t value, size_t alignment = GPU_PAGE_SIZE) {
    return (value & (alignment - 1)) == 0;
}

static inline bool is_ptr_aligned(const void* ptr, size_t alignment = GPU_PAGE_SIZE) {
    return is_aligned(reinterpret_cast<uint64_t>(ptr), alignment);
}




static inline bool __geimifs_device_one_layer_xfer(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    const torch::Tensor &key_cache,         // shape = [max_num_block, block_size, num_heads, head_size]
    const torch::Tensor &value_cache,       // shape = [max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, enum FileXferType type,
    struct geminifs_metadata *metadata, 
    cudaStream_t stream) {
    
    int max_num_block = key_cache.size(0);
    int block_size = key_cache.size(1);
    int num_heads = key_cache.size(2);
    int head_size = key_cache.size(3);
    
    uint64_t block_nbytes = block_size * num_heads * head_size * key_cache.element_size();
    uint64_t layer_stride = 2 * block_nbytes;
    uint64_t key_file_offset = start_layer_idx * layer_stride;
    uint64_t value_file_offset = key_file_offset + block_nbytes;

    if (block_nbytes & (metadata->file_block_size - 1)) { // to avoid xfer to other page
        geminifs_error("block_nbytes %ld is not aligned to file block size\n", block_nbytes);
        return false;
    }

    struct geminifs_dma *key_dma_ctx, *value_dma_ctx;
    if ((key_dma_ctx = geminifs_get_dma(key_cache)) == nullptr) {
        geminifs_error("geminifs_device_xfer_wrapper_cuda: key_cache.data_ptr() %p has not been initialized\n", key_cache.data_ptr());
        return false;
    }

    if ((value_dma_ctx = geminifs_get_dma(value_cache)) == nullptr) {
        geminifs_error("geminifs_device_xfer_wrapper_cuda: value_cache.data_ptr() %p has not been initialized\n", value_cache.data_ptr());
        return false;
    }

    dim3 grid(cached_file_ids.numel());
    dim3 block(32);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));

    cuda::std::span<GPUFileId> file_ids = {(GPUFileId *)cached_file_ids.data_ptr(), (size_t)cached_file_ids.numel()};
    cuda::std::span<uint64_t> block_ids = {(uint64_t *)inner_block_ids.data_ptr(), (size_t)inner_block_ids.numel()};

    auto * pool = metadata->global_pool.get();
    if (key_dma_ctx->dma_ptr->contiguous) {
        __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
                        (pool, file_ids, block_ids, key_dma_ctx->dma_ptr->ioaddrs[0], 
                            block_nbytes, key_file_offset, type);
    }
    
    if (value_dma_ctx->dma_ptr->contiguous) {
        __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
                        (pool, file_ids, block_ids, value_dma_ctx->dma_ptr->ioaddrs[0], 
                            block_nbytes, value_file_offset, type);
    }
    
    if (!key_dma_ctx->dma_ptr->contiguous) { // assert that ioaddrs is not null
        __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
                        (pool, file_ids, block_ids, 
                        {key_dma_ctx->ioaddrs, key_dma_ctx->dma_ptr->n_ioaddrs}, 
                        block_nbytes, key_file_offset, type);
    }   
    
    if (!value_dma_ctx->dma_ptr->contiguous) { // assert that ioaddrs is not null
        __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
                        (pool, file_ids, block_ids, 
                        {value_dma_ctx->ioaddrs, value_dma_ctx->dma_ptr->n_ioaddrs}, 
                        block_nbytes, value_file_offset, type);
    }

    return true;
}

static inline bool geminifs_device_mutiple_layer_xfer(
    const torch::Tensor& cached_file_ids,  // shape = [num_cached_files,]
    const torch::Tensor& inner_block_ids,  // shape = [num_cached_files,]
    const std::vector<torch::Tensor>& key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    const std::vector<torch::Tensor>& value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, int64_t num_layers,
    enum FileXferType type) {

    // Input validation
    if (num_layers <= 0) {
        geminifs_error("geminifs_device_mutiple_layer_xfer: num_layers must be positive, got %lld\n", num_layers);
        return false;
    }
    // printf("num_layers %lld, key_caches.size() %zu, value_caches.size() %zu\n", num_layers, key_caches.size(), value_caches.size());
    if (key_caches.size() != num_layers || value_caches.size() != num_layers) {
        geminifs_error("geminifs_device_mutiple_layer_xfer: key_caches (%zu) or value_caches (%zu) size mismatch with num_layers (%lld)\n", key_caches.size(), value_caches.size(), num_layers);
        return false;
    }
    if (cached_file_ids.numel() != inner_block_ids.numel()) {
        geminifs_error("geminifs_device_mutiple_layer_xfer: cached_file_ids and inner_block_ids must have same number of elements (%lld vs %lld)\n", cached_file_ids.numel(), inner_block_ids.numel());
        return false;
    }
    if (key_caches.empty() || !key_caches[0].is_cuda()) {
        geminifs_error("geminifs_device_mutiple_layer_xfer: key_caches are empty or not on a CUDA device.\n");
        return false;
    }

    auto device = key_caches[0].device().index();
    struct geminifs_metadata* metadata = geminifs_get_metadata(device);

    if (metadata == nullptr) {
        geminifs_error("geminifs_device_mutiple_layer_xfer: device %d has not been initialized.\n", device);
        return false;
    }

    const size_t num_available_streams = metadata->streams.size();
    if (num_available_streams == 0) {
        geminifs_error("geminifs_device_mutiple_layer_xfer: No CUDA streams available in metadata for device %d.\n", device);
        return false;
    }

    // Create a vector to hold CUDA events for each layer's transfer
    std::vector<cudaEvent_t> completion_events(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        // Ensure event creation is successful
        cudaError_t err = cudaEventCreate(&completion_events[i]);
        if (err != cudaSuccess) {
            geminifs_error("geminifs_device_mutiple_layer_xfer: Failed to create CUDA event %d: %s\n", i, cudaGetErrorString(err));
            // Clean up already created events before returning
            for (int j = 0; j < i; ++j) {
                cudaEventDestroy(completion_events[j]);
            }
            return false;
        }
    }

    // Launch transfers on different streams
    for (int i = 0; i < num_layers; ++i) {
        cudaStream_t stream = metadata->streams[i % num_available_streams];

        // Perform the transfer for the current layer
        __geimifs_device_one_layer_xfer(
            cached_file_ids,
            inner_block_ids,
            key_caches[i],
            value_caches[i],
            i + start_layer_idx, // Absolute layer index
            type,
            metadata,
            stream
        );

        // Record an event in the stream after the transfer is submitted
        cudaError_t err = cudaEventRecord(completion_events[i], stream);
        if (err != cudaSuccess) {
            geminifs_error("geminifs_device_mutiple_layer_xfer: Failed to record CUDA event %d for stream %p: %s\n", i, (void*)stream, cudaGetErrorString(err));
            // This is a critical error, might need more robust cleanup
            for (auto& event : completion_events) { // Clean up all events
                if (event) cudaEventDestroy(event);
            }
            return false;
        }
    }


    return true; // All transfers are guaranteed to be complete on the GPU
}


static inline bool geminifs_device_xfer_wrapper_cuda(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    const torch::Tensor &key_cache,         // shape = [max_num_block, block_size, num_heads, head_size]
    const torch::Tensor &value_cache,       // shape = [max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, enum FileXferType type) {
    
    int max_num_block = key_cache.size(0);
    int block_size = key_cache.size(1);
    int num_heads = key_cache.size(2);
    int head_size = key_cache.size(3);
    int device = key_cache.device().index();

    // layer -> file offset, block id -> key/value cache offset
    uint64_t block_nbytes = block_size * num_heads * head_size * key_cache.element_size();
    uint64_t layer_stride = 2 * block_nbytes;
    uint64_t key_file_offset = start_layer_idx * layer_stride;
    uint64_t value_file_offset = key_file_offset + block_nbytes;

    struct geminifs_metadata *metadata;
    if ((metadata = geminifs_get_metadata(device)) == nullptr) {
        geminifs_error("geminifs_device_xfer_wrapper_cuda: device %d has not been initialized\n", device);
        return false;
    }
    
    if (block_nbytes & (metadata->file_block_size - 1)) { // to avoid xfer to other page
        geminifs_error("block_nbytes %ld is not aligned to file block size\n", block_nbytes);
        return false;
    }
    
    auto * pool = metadata->global_pool.get();
    is_device_pointer(pool, "global_pool must be device ptr");

    if (!is_ptr_aligned(key_cache.data_ptr())) {
        geminifs_error("key_cache.data_ptr() %p, block_nbytes %ld\n", key_cache.data_ptr(), block_nbytes);
        return false;
    }

    if (!is_ptr_aligned(value_cache.data_ptr())) {
        geminifs_error("value_cache.data_ptr() %p, block_nbytes %ld\n", value_cache.data_ptr(), block_nbytes);
        return false;
    }

    struct geminifs_dma *key_dma_ctx, *value_dma_ctx;
    if ((key_dma_ctx = geminifs_get_dma(key_cache)) == nullptr) {
        geminifs_error("geminifs_device_xfer_wrapper_cuda: key_cache.data_ptr() %p has not been initialized\n", key_cache.data_ptr());
        return false;
    }

    if ((value_dma_ctx = geminifs_get_dma(value_cache)) == nullptr) {
        geminifs_error("geminifs_device_xfer_wrapper_cuda: value_cache.data_ptr() %p has not been initialized\n", value_cache.data_ptr());
        return false;
    }

    dim3 grid(cached_file_ids.numel());
    dim3 block(32);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
    // const at::cuda::OptionalCUDAGuard device_guard(device_of(value_cache));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // geminifs_debug("tensor info: key_cache.data_ptr() %p, value_cache.data_ptr() %p, block_nbytes %ld, key_file_offset %ld, value_file_offset %ld\n", 
    //                 key_cache.data_ptr(), value_cache.data_ptr(), block_nbytes, key_file_offset, value_file_offset);

    cuda::std::span<GPUFileId> file_ids = {(GPUFileId *)cached_file_ids.data_ptr(), (size_t)cached_file_ids.numel()};
    cuda::std::span<uint64_t> block_ids = {(uint64_t *)inner_block_ids.data_ptr(), (size_t)inner_block_ids.numel()};
    if (key_dma_ctx->dma_ptr->contiguous) {
        // geminifs_debug("key dma is contiguous, ready to transfer key\n");
        __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
                        (pool, file_ids, block_ids, key_dma_ctx->dma_ptr->ioaddrs[0], 
                            block_nbytes, key_file_offset, type);
        // geminifs_debug("key transfer done\n");
    }

    if (value_dma_ctx->dma_ptr->contiguous) {
        // geminifs_debug("value dma is contiguous, ready to transfer value\n");
        __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
                        (pool, file_ids, block_ids, value_dma_ctx->dma_ptr->ioaddrs[0], 
                            block_nbytes, value_file_offset, type);
        // geminifs_debug("value transfer done\n");
    }

    if (!key_dma_ctx->dma_ptr->contiguous) { // assert that ioaddrs is not null
        __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
                        (pool, file_ids, block_ids, 
                        {key_dma_ctx->ioaddrs, key_dma_ctx->dma_ptr->n_ioaddrs}, 
                        block_nbytes, key_file_offset, type);
    }

    if (!value_dma_ctx->dma_ptr->contiguous) { // assert that ioaddrs is not null
        __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
                        (pool, file_ids, block_ids, 
                        {value_dma_ctx->ioaddrs, value_dma_ctx->dma_ptr->n_ioaddrs}, 
                        block_nbytes, value_file_offset, type);
    }
    cudaDeviceSynchronize();
    // geminifs_debug("xfer done\n");
    return true;
}

bool batch_write_direct(const torch::Tensor& cached_file_ids,   //shape = [num_cached_files,]
                        const torch::Tensor& inner_block_ids,  // shape = [num_cached_files,]
                        const torch::Tensor& key_cache,               // shape = [max_num_block, block_size, num_heads, head_size]
                        const torch::Tensor& value_cache,             // shape = [max_num_block, block_size, num_heads, head_size]
                        int64_t start_layer_idx) {

    return geminifs_device_xfer_wrapper_cuda(cached_file_ids, inner_block_ids, 
                                            key_cache, value_cache, 
                                            start_layer_idx, FILE_XFER_WRITE);
}

bool batch_read_direct(const torch::Tensor& cached_file_ids,   //shape = [num_cached_files,]
                        const torch::Tensor& inner_block_ids,  // shape = [num_cached_files,]
                        const torch::Tensor& key_cache,               // shape = [max_num_block, block_size, num_heads, head_size]
                        const torch::Tensor& value_cache,             // shape = [max_num_block, block_size, num_heads, head_size]
                        int64_t start_layer_idx) {

    return geminifs_device_xfer_wrapper_cuda(cached_file_ids, inner_block_ids, 
                                            key_cache, value_cache, 
                                            start_layer_idx, FILE_XFER_READ);
}

bool geminifs_device_mutiple_layer_read_wrapper_cuda(
                                        const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
                                        const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
                                        const std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
                                        const std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
                                        ints64_t start_layer_idx, int64_t num_layers) {
    return geminifs_device_mutiple_layer_xfer(cached_file_ids, inner_block_ids, 
                                            key_caches, value_caches, 
                                            start_layer_idx, num_layers, FILE_XFER_READ);
}

bool geminifs_device_mutiple_layer_write_wrapper_cuda(
                                        const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
                                        const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
                                        const std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
                                        const std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
                                        int64_t start_layer_idx, int64_t num_layers) {
    return geminifs_device_mutiple_layer_xfer(cached_file_ids, inner_block_ids, 
                                            key_caches, value_caches,   
                                            start_layer_idx, num_layers, FILE_XFER_WRITE);
}

bool geminifs_init_fds_wrapper_cuda(const torch::Tensor& file_meta, 
                                    const std::string& mount_path, 
                                    const std::string& pcie_addr) {
    

    TORCH_CHECK(file_meta.dim() == 1 && file_meta.numel() >= 2, "file_meta should be a 1D tensor with at least 2 elements");
    TORCH_CHECK(file_meta.scalar_type() == torch::kUInt64, "file_meta should have kUInt64 type");
    
    
    torch::Tensor file_meta_host = file_meta.cpu();
    const uint64_t* file_meta_ptr = file_meta_host.data_ptr<uint64_t>();
    uint64_t nr_files = file_meta_ptr[0];
    uint64_t file_size = file_meta_ptr[1];
    int64_t device_id = file_meta.device().index();

    geminifs_debug("geminifs_init_fds_wrapper_cuda: nr_files %ld, file_size %ld, device_id %ld\n", 
                    nr_files, file_size, device_id);

    assert(nr_files > 0 && file_size > 0 && device_id >= 0);

    std::vector<string> pcie_addr_vec = split(pcie_addr, ',');
    auto size = pcie_addr_vec.size();
    if (size == 0 || (size & (size - 1)) != 0) {
        geminifs_error("geminifs_init_fds_wrapper_cuda: pcie_addr %s is not a power of 2\n", pcie_addr.c_str());
        return false;
    }
    
    struct geminifs_ctrl_params ctrl_params = {
        .mount_path = mount_path,
        .snvme_control_path = "/dev/snvm_control",
        .pci_addr = pcie_addr_vec,
        .cudaDevice = (int)device_id,
        .ns_id = 1,
        .queueDepth = 1024,
        .numQueues = 64
    };

    if (geminifs_get_metadata(ctrl_params.cudaDevice) != nullptr) {
        geminifs_error("geminifs_init_fds_wrapper_cuda: device %d has been initialized\n", ctrl_params.cudaDevice);
        return false;
    } else {
        global_metadata[ctrl_params.cudaDevice] = __geminifs_init(ctrl_params, nr_files,
                                                                    file_size,  __4KB__);
    }

    return true;
}




bool geminifs_init_fds_wrapper_cuda_test(int64_t nr_files, int64_t file_size, int64_t device_id, 
                                            const std::string& mount_path, const std::string& pcie_addr){
    // 指定 TensorOptions 来设置设备和数据类型为 int64
    torch::Tensor file_meta = torch::tensor({nr_files, file_size}, 
                                            torch::TensorOptions().device(torch::kCUDA, device_id).dtype(torch::kUInt64));
    return geminifs_init_fds_wrapper_cuda(file_meta, mount_path, pcie_addr);
}

bool geminifs_device_xfer_wrapper_test(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    const torch::Tensor &key_cache,         // shape = [max_num_block, block_size, num_heads, head_size]
    const torch::Tensor &value_cache,       // shape = [max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, enum FileXferType type) {
    
    return geminifs_device_xfer_wrapper_cuda(cached_file_ids, inner_block_ids, 
                                            key_cache, value_cache,
                                            start_layer_idx, type);
}

bool geminifs_device_xfer_wrapper_test2(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, int64_t num_layers, enum FileXferType type) {

    return geminifs_device_mutiple_layer_xfer(cached_file_ids, inner_block_ids, 
                                            key_caches, value_caches,
                                            start_layer_idx, num_layers, type);
}

/**
 * Create and register a GPU controller for a specific device
 */
__host__ GPUControllerPtr geminifs_create_gpu_controller(int device_id, const std::string& mount_base_path) {
    auto gpu_controller = std::make_shared<GPUController>(device_id, mount_base_path);
    
    if (!gpu_controller->isInitialized()) {
        geminifs_error("geminifs_create_gpu_controller: Failed to initialize GPU controller for device %d\n", device_id);
        return nullptr;
    }
    
    // Register with the global registry
    auto& registry = GPUControllerRegistry::getInstance();
    if (!registry.registerGPUController(device_id, gpu_controller)) {
        geminifs_error("geminifs_create_gpu_controller: Failed to register GPU controller for device %d\n", device_id);
        return nullptr;
    }
    
    geminifs_debug("geminifs_create_gpu_controller: Successfully created and registered GPU controller for device %d\n", device_id);
    return gpu_controller;
}

/**
 * Get GPU controller for a specific device
 */
__host__ GPUControllerPtr geminifs_get_gpu_controller(int device_id) {
    auto& registry = GPUControllerRegistry::getInstance();
    return registry.getGPUController(device_id);
}

/**
 * Add an NVMe controller to a GPU controller
 */
__host__ bool geminifs_add_nvme_to_gpu(int device_id, const nvme_ctrl_param& params) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_add_nvme_to_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    // Create NVMe controller
    auto nvme_controller = std::make_shared<NVMeController>(params);
    if (!nvme_controller->is_initialized()) {
        geminifs_error("geminifs_add_nvme_to_gpu: Failed to initialize NVMe controller\n");
        return false;
    }
    
    // Add to GPU controller
    if (!gpu_controller->addNVMeController(nvme_controller)) {
        geminifs_error("geminifs_add_nvme_to_gpu: Failed to add NVMe controller to GPU %d\n", device_id);
        return false;
    }
    
    geminifs_debug("geminifs_add_nvme_to_gpu: Successfully added NVMe controller to GPU %d\n", device_id);
    return true;
}

/**
 * Register tensor memory with GPU controller
 */
__host__ bool geminifs_register_tensor_with_gpu(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_register_tensor_with_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    return gpu_controller->registerTensorMemory(tensor);
}

/**
 * Unregister tensor memory from GPU controller
 */
__host__ bool geminifs_unregister_tensor_from_gpu(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_unregister_tensor_from_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    return gpu_controller->unregisterTensorMemory(tensor.data_ptr());
}

/**
 * Get DMA context from GPU controller
 */
__host__ struct geminifs_dma* geminifs_get_tensor_dma_from_gpu(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_get_tensor_dma_from_gpu: No GPU controller found for device %d\n", device_id);
        return nullptr;
    }
    
    return gpu_controller->getDMAContext(tensor.data_ptr());
}

/**
 * Open file using GPU controller
 */
__host__ void* geminifs_gpu_open_file(int device_id, const std::string& filename, size_t file_size, uint32_t o_flag, size_t controller_index = 0) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_gpu_open_file: No GPU controller found for device %d\n", device_id);
        return nullptr;
    }
    
    return gpu_controller->openFile(filename, file_size, o_flag, controller_index);
}

/**
 * Cleanup all GPU controllers
 */
__host__ void geminifs_cleanup_all_gpu_controllers() {
    auto& registry = GPUControllerRegistry::getInstance();
    registry.clearAll();
    geminifs_debug("geminifs_cleanup_all_gpu_controllers: Cleaned up all GPU controllers\n");
}

/**
 * Get PRP list info for a registered tensor
 */
__host__ const prp_list_info* geminifs_get_tensor_prp_info(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_get_tensor_prp_info: No GPU controller found for device %d\n", device_id);
        return nullptr;
    }
    
    auto dma_ctx = gpu_controller->getDMAContext(tensor.data_ptr());
    if (!dma_ctx) {
        geminifs_error("geminifs_get_tensor_prp_info: No DMA context found for tensor at %p\n", tensor.data_ptr());
        return nullptr;
    }
    
    return &dma_ctx->prp_info;
}
