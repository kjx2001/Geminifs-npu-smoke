#include "nvme_controller.cuh"
#include "helper.cuh"
#include "geminifs_helper.h"
#include "nvm_cmd.h"
#include "geminifs.h"

#include <cuda_runtime.h>
#include <unistd.h>
#include <cassert>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <filesystem>
#include <fcntl.h>  // For fallocate

using namespace std;

// Static paths for system components
static char snvme_control_path[] = "/dev/snvm_control";
static char sys_config_path[] = "/mnt/sys_GPU_NVMe_topology.json";

// CUDA kernels for GPU-side initialization

// CUDA kernel to initialize QueueAcquireHelper on GPU
__global__ void init_queue_acquire_helper_kernel(QueueAcquireHelper* d_helper, int num_queues) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (d_helper) QueueAcquireHelper(num_queues);
    }
}

// CUDA kernel to initialize NVMeFile on GPU
__global__ void init_nvme_file_kernel(NVMe_File* d_nvme_file, 
                                       Controller* d_ctrl_ptr,
                                       struct geminiFS_hdr* device_fd,
                                       QueueAcquireHelper* d_queue_acquire_helper,
                                       uint32_t nvme_page_size,
                                       uint32_t block_size,
                                       uint32_t hqps_block_size_log) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (d_nvme_file) NVMe_File(d_ctrl_ptr, device_fd);
        d_nvme_file->queue_acquire_helper = d_queue_acquire_helper;
        d_nvme_file->nvme_page_size = nvme_page_size;
        d_nvme_file->hqps_block_size_log = hqps_block_size_log;
    }
}



NVMeController::NVMeController(const nvme_ctrl_param& params) : is_initialized_(false), d_queue_acquire_helper(nullptr) {
    // Set mount path
    mount_path = params.mount_path;
    
    // Set maximum I/O size (convert from KB to bytes)
    maxIOsize = params.maxIOsize * 1024;
    
    // Check if maxIOsize is within supported limits (params.maxIOsize is in KB)
    if (params.maxIOsize > 1024) {
        geminifs_error("NVMeController initialization failed: maxIOsize (%lu KB) exceeds maximum supported size (1024 KB). Current system only supports up to 1MB NVMe I/O\n", params.maxIOsize);
        throw std::runtime_error("maxIOsize exceeds supported limit of 1024 KB");
    }
    
    // Check if maxIOsize is 4K aligned
    if (maxIOsize % 4096 != 0) {
        geminifs_error("NVMeController initialization failed: maxIOsize (%lu bytes) is not 4K aligned. maxIOsize must be a multiple of 4096 bytes\n", maxIOsize);
        throw std::runtime_error("maxIOsize is not 4K aligned");
    }
    
    // Create mount directory if it doesn't exist
    std::filesystem::create_directories(mount_path);
    
    // Initialize single controller using the provided PCI address
    controller = open_single_controller(params.pci_addr, params);
    
    // Allocate and initialize QueueAcquireHelper on GPU
    if (controller) {
        cudaError_t cuda_err = cudaMalloc(&d_queue_acquire_helper, sizeof(QueueAcquireHelper));
        if (cuda_err != cudaSuccess) {
            geminifs_error("Failed to allocate QueueAcquireHelper on GPU: %s\n", cudaGetErrorString(cuda_err));
            throw std::runtime_error("Failed to allocate QueueAcquireHelper on GPU");
        }
        
        // Initialize QueueAcquireHelper on GPU using CUDA kernel
        init_queue_acquire_helper_kernel<<<1, 1>>>(d_queue_acquire_helper, params.numQueues);
        
        // Check for any CUDA errors after initialization
        cuda_err = cudaDeviceSynchronize();
        if (cuda_err != cudaSuccess) {
            geminifs_error("Failed to synchronize after QueueAcquireHelper initialization: %s\n", cudaGetErrorString(cuda_err));
            cudaFree(d_queue_acquire_helper);
            d_queue_acquire_helper = nullptr;
            throw std::runtime_error("Failed to initialize QueueAcquireHelper on GPU");
        }
        
        cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            geminifs_error("Failed to initialize QueueAcquireHelper on GPU: %s\n", cudaGetErrorString(cuda_err));
            cudaFree(d_queue_acquire_helper);
            d_queue_acquire_helper = nullptr;
            throw std::runtime_error("Failed to initialize QueueAcquireHelper on GPU");
        }
        
        geminifs_debug("Successfully allocated and initialized QueueAcquireHelper on GPU at %p for %lu queues\n", 
                       d_queue_acquire_helper, params.numQueues);
    }
    
    // Initialize file manager with log file in the controller's actual mount path
    std::string log_file_path = controller->dev_mount_path + "/nvme_file_log.dat";
    file_manager = std::make_unique<FileManager>(log_file_path, 1000); // 1000 is persistence threshold
    
    // Set initialization state to true after successful initialization
    is_initialized_ = true;
}

NVMeController::~NVMeController() {
    // Clean up device files first
    cleanup_device_files();
    
    // Clean up GPU QueueAcquireHelper if allocated
    if (d_queue_acquire_helper != nullptr) {
        cudaError_t err = cudaFree(d_queue_acquire_helper);
        if (err != cudaSuccess) {
            geminifs_error("Failed to free GPU QueueAcquireHelper: %s\n", cudaGetErrorString(err));
        }
        d_queue_acquire_helper = nullptr;
    }
    
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
    uint32_t file_id = file_manager->getFileIdByFilename(filename);
    bool file_exists_in_log = (file_id != UINT32_MAX);
    NVMeFileDesc file_desc;
    if (file_exists_in_log) {
        file_manager->getFileById(file_id, file_desc);
    }
    
    void* result_fd = nullptr;
    
    if (file_exists_in_log) {
        // File exists in log, check if physical file exists too
        std::filesystem::path file_path = controller->dev_mount_path;
        file_path = file_path / filename;
        
        if (std::filesystem::exists(file_path)) {
            // Both exist, open existing file
            if (o_flag & O_HOST) {
                result_fd = host_file_open_managed(file_path, O_RDWR);
                if (result_fd == nullptr) {
                    geminifs_error("g_open: Failed to open existing host file '%s'\n", filename.c_str());
                    return nullptr;
                }
            } else if (o_flag & O_DEVICE) {
                result_fd = device_file_open_managed(filename);
                if (result_fd == nullptr) {
                    geminifs_error("g_open: Failed to open existing device file '%s'\n", filename.c_str());
                    return nullptr;
                }
            }
            
            geminifs_debug("g_open: Opened existing file '%s' from log slot %u\n", 
                          filename.c_str(), file_desc.slot_index);
        } else {
            // File exists in log but physical file missing, remove from log and recreate
            geminifs_debug("g_open: File '%s' exists in log but physical file missing, recreating\n", filename.c_str());
            file_manager->deleteFile(file_id);
            file_exists_in_log = false;  // Force recreation below
        }
    }
    
    if (!file_exists_in_log) {
        // File doesn't exist in log, create new file
        std::filesystem::path file_path = controller->dev_mount_path;
        file_path = file_path / filename;

        if (std::filesystem::exists(file_path)) {
            geminifs_debug("g_open: Physical file exists but not in log, recreating and adjusting size\n");
            // Remove existing physical file as required by spec
            std::filesystem::remove(file_path);
        }
        
        // Create new file
        if (o_flag & O_HOST) {
            // Create for host-side operations using internal method to get host_fd_t
            result_fd = create_host_fd_internal(controller->page_size, file_size, filename);
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
                // Build file path and remove it
                string file_path_str = string(controller->dev_mount_path) + "/" + filename;
                unlink(file_path_str.c_str());
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

// // Helper function for binary bit counting (needed by NVMeController member functions)
// static int one_nr__of__binary_int(unsigned long long i) {
//     int count = 0;
//     while (i != 0) {
//         if ((i & 1) == 1)
//             count++;
//         i = i >> 1;
//     }
//     return count;
// }

/**
 * NVMeController member function to create a file with automatic FileManager integration
 * Returns the slot_index from NVMeFileDesc as the file ID
 */
uint32_t NVMeController::host_file_create_managed(int block_size, size_t file_size, const std::string& filename) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("host_file_create_managed: NVMeController is not properly initialized\n");
        return UINT32_MAX; // Invalid file ID
    }
    
    assert(file_size % block_size == 0);

    auto nvpage_size = controller->page_size;
    assert(block_size % nvpage_size == 0);

    auto hdr_size = GEMINI_HDR_MAX_SIZE;

    // Allocate host memory for the header
    struct geminiFS_hdr *hdr = (struct geminiFS_hdr *)malloc(hdr_size);
    if (!hdr) {
        geminifs_error("host_file_create_managed: Failed to allocate memory for header\n");
        return UINT32_MAX;
    }
    
    // Build file path using string concatenation to avoid filesystem issues
    string file_path_str = string(controller->dev_mount_path) + "/" + filename;
    
    // Initialize header
    hdr->magic_num = the_geminiFS_magic.magic_num;
    hdr->first_block_base = hdr_size;
    hdr->virtual_space_size = file_size;
    hdr->block_bit = __builtin_ctzll(block_size); // Block size in bits

    // Open file
    int fd = open(file_path_str.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        geminifs_error("host_file_create_managed: Failed to create file '%s': %s\n", file_path_str.c_str(), strerror(errno));
        free(hdr);
        return UINT32_MAX;
    }
    
    // Set file size using fallocate to actually allocate space
    if (fallocate(fd, 0, 0, hdr_size + file_size) != 0) {
        geminifs_error("host_file_create_managed: Failed to allocate file space: %s\n", strerror(errno));
        close(fd);
        free(hdr);
        return UINT32_MAX;
    }
    
    hdr->fd = fd;
    
    // Refine NVMe offsets
    host_refine_nvmeofst(hdr);
   
    // // Write header to file
    // if (write(fd, hdr, hdr_size) != (ssize_t)hdr_size) {
    //     geminifs_error("host_file_create_managed: Failed to write header to file '%s': %s\n", 
    //                   file_path_str.c_str(), strerror(errno));
    //     close(fd);
    //     free(hdr);
    //     return UINT32_MAX;
    // }
    
    // Close file immediately - we only create it, don't keep it open
    close(fd);
    
    // Create file record in FileManager and get the slot_index
    NVMeFileDesc file_desc;
    if (file_manager == nullptr || !file_manager->createFile(filename, file_desc, file_size)) {
        geminifs_error("host_file_create_managed: Failed to create file record in FileManager for '%s'\n", filename.c_str());
        // Clean up the created physical file
        unlink(file_path_str.c_str());
        free(hdr);
        return UINT32_MAX;
    }
    
    // Clean up header memory
    free(hdr);
    
    geminifs_debug("host_file_create_managed: Created file '%s' with size %zu, slot_index %u (file closed)\n", 
                   filename.c_str(), file_size, file_desc.slot_index);
    
    return file_desc.slot_index; // Return the slot_index as file ID
}

/**
 * Create a new host file with auto-generated filename based on NVMe ID naming convention
 * @param block_size The block size for the file
 * @param file_size The size of the file to create
 * @return The file ID (slot_index), or UINT32_MAX if creation failed
 */
uint32_t NVMeController::host_file_create_managed(int block_size, size_t file_size) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("host_file_create_managed: NVMeController is not properly initialized\n");
        return UINT32_MAX; // Invalid file ID
    }
    
    if (!file_manager) {
        geminifs_error("host_file_create_managed: FileManager is not available\n");
        return UINT32_MAX;
    }
    
    // Use FileManager's auto-filename generation feature
    NVMeFileDesc file_desc;
    if (!file_manager->createFile(file_desc, file_size)) {
        geminifs_error("host_file_create_managed: Failed to create file in FileManager\n");
        return UINT32_MAX;
    }
    
    // The filename is automatically generated based on slot_index (file ID)
    std::string filename(file_desc.filename);
    
    assert(file_size % block_size == 0);

    auto nvpage_size = controller->page_size;
    assert(block_size % nvpage_size == 0);

    auto hdr_size = GEMINI_HDR_MAX_SIZE;

    // Allocate host memory for the header
    struct geminiFS_hdr *hdr = (struct geminiFS_hdr *)malloc(hdr_size);
    if (!hdr) {
        geminifs_error("host_file_create_managed: Failed to allocate memory for header\n");
        // Clean up the FileManager entry
        file_manager->deleteFile(file_desc.slot_index);
        return UINT32_MAX;
    }
    
    // Build file path using string concatenation
    string file_path_str = string(controller->dev_mount_path) + "/" + filename;
    
    // Initialize header
    hdr->magic_num = the_geminiFS_magic.magic_num;
    hdr->first_block_base = hdr_size;
    hdr->virtual_space_size = file_size;
    hdr->block_bit = __builtin_ctzll(block_size); // Block size in bits

    // Create and write file, then immediately close
    int fd = open(file_path_str.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        geminifs_error("host_file_create_managed: Failed to create file '%s': %s\n", 
                       file_path_str.c_str(), strerror(errno));
        free(hdr);
        // Clean up the FileManager entry
        file_manager->deleteFile(file_desc.slot_index);
        return UINT32_MAX;
    }
    
    // Set file size using fallocate to actually allocate space
    if (fallocate(fd, 0, 0, hdr_size + file_size) != 0) {
        geminifs_error("host_file_create_managed: Failed to allocate file space for '%s': %s\n", 
                       file_path_str.c_str(), strerror(errno));
        close(fd);
        free(hdr);
        // Clean up the FileManager entry
        file_manager->deleteFile(file_desc.slot_index);
        return UINT32_MAX;
    }
    
    // Set temporary fd for header operations
    hdr->fd = fd;
    
    // Refine NVMe offsets
    host_refine_nvmeofst(hdr);

    // // Write header to file
    // if (write(fd, hdr, hdr_size) != (ssize_t)hdr_size) {
    //     geminifs_error("host_file_create_managed: Failed to write header to file '%s': %s\n", 
    //                    file_path_str.c_str(), strerror(errno));
    //     close(fd);
    //     unlink(file_path_str.c_str()); // Delete the incomplete file
    //     free(hdr);
    //     // Clean up the FileManager entry
    //     file_manager->deleteFile(file_desc.slot_index);
    //     return UINT32_MAX;
    // }

    // Close file immediately - we only created it, don't need to keep it open
    close(fd);
    free(hdr);
    
    geminifs_debug("host_file_create_managed: Created file '%s' with auto-generated name, size %zu, slot_index %u\n", 
                   filename.c_str(), file_size, file_desc.slot_index);
    
    return file_desc.slot_index; // Return the slot_index as file ID
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
        size_t hdr_size = result->first_block_base;
        file_manager->registerOpenFile(result, filepath, hdr_size);
    }
    
    return result;
}

/**
 * NVMeController member function to open a file with automatic FileManager integration
 */
host_fd_t NVMeController::host_file_open_managed(uint32_t id, uint32_t o_flag) {
    NVMeFileDesc file_desc;
    file_manager->getFileById(id, file_desc);

    std::string filepath = get_file_path(file_desc);
    
    return host_file_open_managed(filepath, o_flag);
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
 * Internal helper function to create a host file descriptor
 * This is used when we need the actual host_fd_t for device operations
 */
host_fd_t NVMeController::create_host_fd_internal(int block_size, size_t file_size, const std::string& filename) {
    assert(file_size % block_size == 0);

    auto nvpage_size = controller->page_size;
    assert(block_size % nvpage_size == 0);

    auto hdr_size = GEMINI_HDR_MAX_SIZE;

    // Allocate host memory for the header
    struct geminiFS_hdr *hdr = (struct geminiFS_hdr *)malloc(hdr_size);
    if (!hdr) {
        geminifs_error("create_host_fd_internal: Failed to allocate memory for header\n");
        return nullptr;
    }
    
    // Build file path using string concatenation
    string file_path_str = string(controller->dev_mount_path) + "/" + filename;
    
    // Initialize header
    hdr->magic_num = the_geminiFS_magic.magic_num;
    hdr->first_block_base = hdr_size;
    hdr->virtual_space_size = file_size;
    hdr->block_bit = __builtin_ctzll(block_size);

    // Open file
    int fd = open(file_path_str.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        geminifs_error("create_host_fd_internal: Failed to create file '%s': %s\n", file_path_str.c_str(), strerror(errno));
        free(hdr);
        return nullptr;
    }
    
    // Set file size using fallocate
    if (fallocate(fd, 0, 0, hdr_size + file_size) != 0) {
        geminifs_error("create_host_fd_internal: Failed to allocate file space: %s\n", strerror(errno));
        close(fd);
        free(hdr);
        return nullptr;
    }
    
    hdr->fd = fd;
    
    // Refine NVMe offsets
    host_refine_nvmeofst(hdr);
    
    // Register with FileManager for cleanup tracking
    if (file_manager != nullptr) {
        file_manager->registerOpenFile(hdr, filename, hdr_size);
    }
    
    return hdr;
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
    
    // First create the host file using internal method
    host_fd_t host_fd = create_host_fd_internal(block_size, file_size, filename);
    if (host_fd == nullptr) {
        geminifs_error("device file create managed: Failed to create host file '%s'\n", filename.c_str());
        return nullptr;
    }
    
    // Create file record in FileManager
    NVMeFileDesc file_desc;
    if (file_manager == nullptr || !file_manager->createFile(filename, file_desc, file_size)) {
        geminifs_error("device file create managed: Failed to create file record in FileManager for '%s'\n", filename.c_str());
        host_file_close_managed(host_fd);
        return nullptr;
    }
    
    // Calculate header size
    size_t hdr_size = GEMINI_HDR_MAX_SIZE;
    
    // Copy host file descriptor to device
    dev_fd_t device_fd = copy_host_fd_to_device(host_fd, hdr_size);
    if (device_fd == nullptr) {
        geminifs_error("device file create managed: Failed to copy host file to device for '%s'\n", filename.c_str());
        // Clean up the host file
        host_file_close_managed(host_fd);
        return nullptr;
    }
    
    // Allocate and initialize NVMeFile on GPU
    NVMe_File* d_nvme_file = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_nvme_file, sizeof(NVMe_File));
    if (cuda_err != cudaSuccess) {
        geminifs_error("device file create managed: Failed to allocate NVMeFile on GPU: %s\n", cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        return nullptr;
    }
    
    // Initialize NVMeFile on GPU using CUDA kernel
    init_nvme_file_kernel<<<1, 1>>>(d_nvme_file,
                                     (Controller*)controller->d_ctrl_ptr,
                                     (struct geminiFS_hdr*)device_fd,
                                     d_queue_acquire_helper,
                                     controller->page_size,
                                     block_size,
                                     controller->h_qps[0]->block_size_log);
    
    // Check for any CUDA errors after initialization
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        geminifs_error("device file create managed: Failed to synchronize after NVMeFile initialization: %s\n", cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        cudaFree(d_nvme_file);
        return nullptr;
    }
    
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        geminifs_error("device file create managed: Failed to initialize NVMeFile on GPU: %s\n", cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        cudaFree(d_nvme_file);
        return nullptr;
    }
    
    // Store the mapping for management
    {
        std::lock_guard<std::mutex> lock(device_files_mtx_);
        device_files_.emplace_back(host_fd, device_fd, hdr_size, filename, d_nvme_file);
    }
    
    geminifs_debug("device file create managed: Created device file '%s' with device_fd %p\n", 
                   filename.c_str(), device_fd);
    
    return device_fd;
}

/**
 * NVMeController private function to open an existing file as a device file
 */
dev_fd_t NVMeController::device_file_open_managed(const std::string& filename) {
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
    host_fd_t host_fd = host_file_open_managed(file_path, O_RDWR);
    if (host_fd == nullptr) {
        geminifs_error("device file open managed: Failed to open host file '%s'\n", file_path.c_str());
        return nullptr;
    }
    
    // Validate file size
    // if (host_fd->virtual_space_size != file_size) {
    //     geminifs_error("device file open managed: File size mismatch. Expected %zu, got %zu\n", 
    //                    file_size, host_fd->virtual_space_size);
    //     host_file_close_managed(host_fd);
    //     return nullptr;
    // }
    
    // Calculate header size
    size_t hdr_size = host_fd->first_block_base;
    
    // Copy host file descriptor to device
    dev_fd_t device_fd = copy_host_fd_to_device(host_fd, hdr_size);
    if (device_fd == nullptr) {
        geminifs_error("device file open managed: Failed to copy host file to device for '%s'\n", filename.c_str());
        host_file_close_managed(host_fd);
        return nullptr;
    }
    
    // Allocate and initialize NVMeFile on GPU
    NVMe_File* d_nvme_file = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_nvme_file, sizeof(NVMe_File));
    if (cuda_err != cudaSuccess) {
        geminifs_error("device file open managed: Failed to allocate NVMeFile on GPU: %s\n", cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        return nullptr;
    }
    
    // Initialize NVMeFile on GPU using CUDA kernel
    init_nvme_file_kernel<<<1, 1>>>(d_nvme_file,
                                     (Controller*)controller->d_ctrl_ptr,
                                     (struct geminiFS_hdr*)device_fd,
                                     d_queue_acquire_helper,
                                     controller->page_size,
                                     controller->blk_size,
                                     controller->blk_size_log);
    
    // Check for any CUDA errors after initialization
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        geminifs_error("device file open managed: Failed to synchronize after NVMeFile initialization: %s\n", cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        cudaFree(d_nvme_file);
        return nullptr;
    }
    
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        geminifs_error("device file open managed: Failed to initialize NVMeFile on GPU: %s\n", cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        cudaFree(d_nvme_file);
        return nullptr;
    }
    
    // Store the mapping for management
    {
        std::lock_guard<std::mutex> lock(device_files_mtx_);
        device_files_.emplace_back(host_fd, device_fd, hdr_size, filename, d_nvme_file);
    }
    
    geminifs_debug("device file open managed: Opened device file '%s' with device_fd %p\n", 
                   filename.c_str(), device_fd);
    
    return d_nvme_file;
}

dev_fd_t NVMeController::device_file_open_managed(uint32_t id) {
    NVMeFileDesc file_desc;
    file_manager->getFileById(id, file_desc);

    std::string filepath = get_file_path(file_desc);
    
    return device_file_open_managed(filepath);
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
        
        // Free GPU NVMeFile if allocated
        if (it->d_nvme_file != nullptr) {
            cudaError_t err = cudaFree(it->d_nvme_file);
            if (err != cudaSuccess) {
                geminifs_error("device file close managed: Failed to free NVMeFile on GPU: %s\n", 
                              cudaGetErrorString(err));
            }
        }
        
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
        geminifs_error("copy_host_fd_to_device: Failed to allocate device memory: %s\n", 
                      cudaGetErrorString(err));
        return nullptr;
    }
    
    // Copy header to device
    err = cudaMemcpy(device_fd, host_fd, hdr_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("copy_host_fd_to_device: Failed to copy header to device: %s\n", 
                      cudaGetErrorString(err));
        cudaFree(device_fd);
        return nullptr;
    }
    
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
        
        // Free GPU NVMeFile if allocated
        if (handle.d_nvme_file != nullptr) {
            cudaError_t err = cudaFree(handle.d_nvme_file);
            if (err != cudaSuccess) {
                geminifs_error("cleanup device files: Failed to free NVMeFile for '%s': %s\n", 
                              handle.filename.c_str(), cudaGetErrorString(err));
            }
        }
        
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

string NVMeController::get_file_path(const NVMeFileDesc& file_desc) {
    std::filesystem::path file_path = std::filesystem::path(controller->dev_mount_path) / file_desc.filename;
    return file_path.string();
}

ControllerPtr NVMeController::open_single_controller(const std::string& pci_addr, const nvme_ctrl_param& params) {
    // Create mount path for this specific controller
    std::filesystem::path mount_path_param(params.mount_path);
    std::filesystem::path this_mount_path = mount_path_param;

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

// Private helper functions for checking system components
bool NVMeController::check_snvme_control_exists() {
    if (access(snvme_control_path, F_OK) != 0) {
        geminifs_error("SNVM control device '%s' does not exist. Please ensure the kernel module is properly installed.\n", snvme_control_path);
        return false;
    }
    return true;
}

bool NVMeController::check_sys_config_exists() {
    if (access(sys_config_path, F_OK) != 0) {
        geminifs_error("Sys GPU-NVMe topology '%s' does not exist. Please ensure the kernel module is properly installed.\n", sys_config_path);
        return false;
    }
    return true;
}

/**
 * Clean up all files managed by this NVMe controller
 * This function will:
 * 1. Get all filenames from the FileManager log
 * 2. Delete each file from both the log and physical storage
 * 3. Close any open device file descriptors
 */
bool NVMeController::device_file_delete_all_files_managed() {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("device_file_delete_all_files_managed: NVMeController is not properly initialized\n");
        return false;
    }
    
    if (!file_manager) {
        geminifs_error("device_file_delete_all_files_managed: FileManager is not available\n");
        return false;
    }
    
    geminifs_debug("device_file_delete_all_files_managed: Starting cleanup for mount path '%s'\n", 
                   controller->dev_mount_path.c_str());
    
    // Get all file IDs from the FileManager log
    std::vector<uint32_t> all_file_ids = file_manager->getAllFileIds();
    
    if (all_file_ids.empty()) {
        geminifs_debug("device_file_delete_all_files_managed: No files found in log, cleanup complete\n");
        return true;
    }
    
    geminifs_debug("device_file_delete_all_files_managed: Found %zu files to clean up\n", all_file_ids.size());
    
    size_t files_deleted = 0;
    size_t files_failed = 0;
    
    // Process each file
    for (const auto& file_id : all_file_ids) {
        // Get file descriptor to get filename
        NVMeFileDesc file_desc;
        if (!file_manager->getFileById(file_id, file_desc)) {
            geminifs_error("device_file_delete_all_files_managed: Failed to get file descriptor for ID %u\n", file_id);
            files_failed++;
            continue;
        }
        
        std::string filename(file_desc.filename);
        geminifs_debug("device_file_delete_all_files_managed: Processing file '%s' (ID: %u)\n", filename.c_str(), file_id);
        
        // Build full path to the physical file
        string file_path_str = string(controller->dev_mount_path) + "/" + filename;
        const char* file_path_cstr = file_path_str.c_str();
        
        bool physical_file_deleted = false;
        bool log_entry_deleted = false;
        
        // Try to delete the physical file if it exists
        struct stat st;
        if (stat(file_path_cstr, &st) == 0) {
            if (unlink(file_path_cstr) == 0) {
                geminifs_debug("device_file_delete_all_files_managed: Successfully deleted physical file '%s'\n", 
                               file_path_cstr);
                physical_file_deleted = true;
            } else {
                geminifs_error("device_file_delete_all_files_managed: Failed to delete physical file '%s': %s\n", 
                               file_path_cstr, strerror(errno));
            }
        } else {
            geminifs_debug("device_file_delete_all_files_managed: Physical file '%s' does not exist\n", 
                           file_path_cstr);
            physical_file_deleted = true; // Consider it as "successfully deleted" if it doesn't exist
        }
        
        // Delete the entry from FileManager log
        if (file_manager->deleteFile(file_id)) {
            geminifs_debug("device_file_delete_all_files_managed: Successfully deleted log entry for '%s'\n", 
                           filename.c_str());
            log_entry_deleted = true;
        } else {
            geminifs_error("device_file_delete_all_files_managed: Failed to delete log entry for '%s'\n", 
                           filename.c_str());
        }
        
        // Count success/failure
        if (physical_file_deleted && log_entry_deleted) {
            files_deleted++;
        } else {
            files_failed++;
        }
    }
    
    // Force persistence of the log changes
    file_manager->forcePersist();
    
    // Clean up any remaining device file descriptors
    cleanup_device_files();
    
    // Report results
    geminifs_debug("device_file_delete_all_files_managed: Cleanup complete. "
                   "Successfully deleted: %zu, Failed: %zun", 
                   files_deleted, files_failed);
    
    if (files_failed > 0) {
        geminifs_error("device_file_delete_all_files_managed: %zu files could not be completely cleaned up\n", 
                       files_failed);
        return false;
    }
    
    geminifs_debug("device_file_delete_all_files_managed: All files successfully cleaned up\n");
    return true;
}

/**
 * Delete a single file managed by this NVMe controller by filename
 * @param filename The name of the file to delete
 * @return true if successful, false otherwise
 */
bool NVMeController::device_file_delete_single_managed(const std::string& filename) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("device_file_delete_single_managed: NVMeController is not properly initialized\n");
        return false;
    }
    
    if (!file_manager) {
        geminifs_error("device_file_delete_single_managed: FileManager is not available\n");
        return false;
    }
    
    geminifs_debug("device_file_delete_single_managed: Deleting file '%s' from mount path '%s'\n", 
                   filename.c_str(), controller->dev_mount_path.c_str());
    
    // Parse file ID directly from filename (since filename = nvme_id.toString())
    uint32_t file_id = FileManager::parseFileIdFromFilename(filename);
    if (file_id == UINT32_MAX) {
        geminifs_error("device_file_delete_single_managed: Cannot parse file ID from filename '%s'\n", filename.c_str());
        return false;
    }
    
    // Build full path to the physical file using string concatenation
    string file_path_str = string(controller->dev_mount_path) + "/" + filename;
    
    bool physical_file_deleted = false;
    bool log_entry_deleted = false;
    
    // Try to delete the physical file if it exists
    struct stat stat_buf;
    if (stat(file_path_str.c_str(), &stat_buf) == 0) {
        if (unlink(file_path_str.c_str()) == 0) {
            geminifs_debug("device_file_delete_single_managed: Successfully deleted physical file '%s'\n", 
                           file_path_str.c_str());
            physical_file_deleted = true;
        } else {
            geminifs_error("device_file_delete_single_managed: Failed to delete physical file '%s': %s\n", 
                           file_path_str.c_str(), strerror(errno));
        }
    } else {
        geminifs_debug("device_file_delete_single_managed: Physical file '%s' does not exist\n", 
                       file_path_str.c_str());
        physical_file_deleted = true; // Consider it as "successfully deleted" if it doesn't exist
    }
    
    // Delete the entry from FileManager log
    if (file_manager->deleteFile(file_id)) {
        geminifs_debug("device_file_delete_single_managed: Successfully deleted log entry for '%s'\n", 
                       filename.c_str());
        log_entry_deleted = true;
    } else {
        geminifs_error("device_file_delete_single_managed: Failed to delete log entry for '%s'\n", 
                       filename.c_str());
    }
    
    // Remove from device_files_ if it exists there
    {
        std::lock_guard<std::mutex> lock(device_files_mtx_);
        auto it = std::find_if(device_files_.begin(), device_files_.end(),
                              [&filename](const DeviceFileHandle& handle) {
                                  return handle.filename == filename;
                              });
        
        if (it != device_files_.end()) {
            geminifs_debug("device_file_delete_single_managed: Removing device file handle for '%s'\n", 
                           filename.c_str());
            
            // Free GPU NVMeFile if allocated
            if (it->d_nvme_file != nullptr) {
                cudaError_t err = cudaFree(it->d_nvme_file);
                if (err != cudaSuccess) {
                    geminifs_error("device_file_delete_single_managed: Failed to free NVMeFile for '%s': %s\n", 
                                  filename.c_str(), cudaGetErrorString(err));
                }
            }
            
            // Free device memory
            if (it->device_fd != nullptr) {
                cudaError_t err = cudaFree(it->device_fd);
                if (err != cudaSuccess) {
                    geminifs_error("device_file_delete_single_managed: Failed to free device memory for '%s': %s\n", 
                                  filename.c_str(), cudaGetErrorString(err));
                }
            }
            
            device_files_.erase(it);
        }
    }
    
    // Force persistence of the log changes
    file_manager->forcePersist();
    
    bool success = physical_file_deleted && log_entry_deleted;
    if (success) {
        geminifs_debug("device_file_delete_single_managed: Successfully deleted file '%s'\n", filename.c_str());
    } else {
        geminifs_error("device_file_delete_single_managed: Failed to completely delete file '%s'\n", filename.c_str());
    }
    
    return success;
}

/**
 * Delete a single file managed by this NVMe controller by NVMe file ID
 * This method looks up the filename by file ID and then calls device_file_delete_single_managed
 * @param nvme_file_id The NVMe file ID (usually the file descriptor or index)
 * @return true if successful, false otherwise
 */
bool NVMeController::host_file_delete_managed(uint32_t nvme_file_id) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("host_file_delete_managed: NVMeController is not properly initialized\n");
        return false;
    }
    
    if (!file_manager) {
        geminifs_error("host_file_delete_managed: FileManager is not available\n");
        return false;
    }
    
    geminifs_debug("host_file_delete_managed: Looking for file with NVMe ID %u\n", nvme_file_id);
    
    // Get file descriptor to check if file exists
    NVMeFileDesc file_desc;
    if (!file_manager->getFileById(nvme_file_id, file_desc)) {
        geminifs_error("host_file_delete_managed: File with ID %u not found in log\n", nvme_file_id);
        return false;
    }
    
    std::string filename(file_desc.filename);
    
    // Use the single file deletion method
    return device_file_delete_single_managed(filename);
}

/**
 * Get the count of files managed by this NVMe controller
 */
size_t NVMeController::device_file_get_managed_file_count() const {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("device_file_get_managed_file_count: NVMeController is not properly initialized\n");
        return 0;
    }
    
    if (!file_manager) {
        geminifs_error("device_file_get_managed_file_count: FileManager is not initialized\n");
        return 0;
    }
    
    // Get all file IDs from the FileManager log
    std::vector<uint32_t> all_file_ids = file_manager->getAllFileIds();
    
    geminifs_debug("device_file_get_managed_file_count: Found %zu managed files\n", all_file_ids.size());
    
    return all_file_ids.size();
}

/**
 * Validate that all managed files have the expected size
 */
size_t NVMeController::device_file_validate_sizes(size_t expected_size) const {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("device_file_validate_sizes: NVMeController is not properly initialized\n");
        return 0;
    }
    
    if (!file_manager) {
        geminifs_error("device_file_validate_sizes: FileManager is not initialized\n");
        return 0;
    }
    
    // Get all file IDs from the FileManager log
    std::vector<uint32_t> all_file_ids = file_manager->getAllFileIds();
    
    if (all_file_ids.empty()) {
        geminifs_debug("device_file_validate_sizes: No files to validate\n");
        return 0;
    }
    
    geminifs_debug("device_file_validate_sizes: Validating %zu files for expected size %zu bytes\n", 
                   all_file_ids.size(), expected_size);
    
    size_t valid_files = 0;
    
    for (const auto& file_id : all_file_ids) {
        // Get file descriptor to get filename
        NVMeFileDesc file_desc;
        if (!file_manager->getFileById(file_id, file_desc)) {
            geminifs_error("device_file_validate_sizes: Failed to get file descriptor for ID %u\n", file_id);
            continue;
        }
        
        std::string filename(file_desc.filename);
        // Build file path using C-style string operations
        std::string file_path = controller->dev_mount_path;
        if (file_path.back() != '/') {
            file_path += "/";
        }
        file_path += filename;
        
        // Use stat to get file size
        struct stat file_stat;
        if (stat(file_path.c_str(), &file_stat) == 0) {
            size_t file_size = static_cast<size_t>(file_stat.st_size);
            
            // For GeminiFS files, we need to account for the header size
            // The actual data size should be expected_size, but the file includes header
            size_t expected_total_size = expected_size + GEMINI_HDR_MAX_SIZE;
            
            if (file_size == expected_total_size) {
                valid_files++;
                geminifs_debug("device_file_validate_sizes: File '%s' has correct size %zu bytes\n", 
                              filename.c_str(), file_size);
            } else {
                geminifs_debug("device_file_validate_sizes: File '%s' has incorrect size %zu bytes (expected %zu bytes), skipping\n", 
                              filename.c_str(), file_size, expected_total_size);
            }
        } else {
            geminifs_debug("device_file_validate_sizes: Failed to get size of file '%s': %s, skipping\n", 
                          file_path.c_str(), strerror(errno));
        }
    }
    
    geminifs_debug("device_file_validate_sizes: Validation complete. Valid files with expected size: %zu\n", 
                   valid_files);
    
    return valid_files;
}

// GPU device-side read/write interface using NVMeFile pointer

// Global wrapper for external calling from host
__global__
void nvme_controller_g_read_kernel(dev_fd_t device_fd, uint64_t prp1, uint64_t prp2, size_t file_offset, size_t nbytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        nvme_controller_g_read(device_fd, prp1, prp2, file_offset, nbytes);
    }
}

__global__
void nvme_controller_g_write_kernel(dev_fd_t device_fd, uint64_t prp1, uint64_t prp2, size_t file_offset, size_t nbytes)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        nvme_controller_g_write(device_fd, prp1, prp2, file_offset, nbytes);
    }
}

__device__
void nvme_controller_g_read(dev_fd_t device_fd, uint64_t prp1, uint64_t prp2, size_t file_offset, size_t nbytes)
{
    auto *nvme_file = (NVMe_File*)device_fd;
    // Call the read method on the NVMe_File instance
    nvme_file->read_in(prp1, prp2, file_offset, nbytes);
}

__device__
void nvme_controller_g_write(dev_fd_t device_fd, uint64_t prp1, uint64_t prp2, size_t file_offset, size_t nbytes)
{
    auto *nvme_file = (NVMe_File*)device_fd;
    // Call the write method on the NVMe_File instance
    nvme_file->write_out(prp1, prp2, file_offset, nbytes);
}

/**
 * Create a managed file without opening it - only creates the file with proper header and size
 * This function is optimized for file creation without the overhead of maintaining file descriptors
 */
bool NVMeController::host_file_create_only_managed(int block_size, size_t file_size, const std::string& filename) {
    // Check if controller is properly initialized
    if (!is_initialized()) {
        geminifs_error("host_file_create_only_managed: NVMeController is not properly initialized\n");
        return false;
    }
    
    assert(file_size % block_size == 0);

    auto nvpage_size = controller->page_size;
    assert(block_size % nvpage_size == 0);

    size_t hdr_size = GEMINI_HDR_MAX_SIZE;

    // Allocate host memory for the header
    struct geminiFS_hdr *hdr = (struct geminiFS_hdr *)malloc(hdr_size);
    if (!hdr) {
        geminifs_error("host_file_create_only_managed: Failed to allocate memory for header\n");
        return false;
    }
    
    // Build file path using helper function (avoiding std::filesystem)
    std::string file_path = build_file_path(controller->dev_mount_path, filename);
    
    // Create directory if it doesn't exist
    if (!create_directories(controller->dev_mount_path)) {
        geminifs_error("host_file_create_only_managed: Failed to create directory '%s'\n", controller->dev_mount_path.c_str());
        free(hdr);
        return false;
    }
    
    // Initialize header
    hdr->magic_num = the_geminiFS_magic.magic_num;
    hdr->first_block_base = hdr_size;
    hdr->virtual_space_size = file_size;
    hdr->block_bit = __builtin_ctzll(block_size); // Block size in bits

    // Create and write file, then immediately close
    int fd = open(file_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        geminifs_error("host_file_create_only_managed: Failed to create file '%s': %s\n", 
                      file_path.c_str(), strerror(errno));
        free(hdr);
        return false;
    }
    
    // Set file size using fallocate to actually allocate space
    if (fallocate(fd, 0, 0, hdr_size + file_size) != 0) {
        geminifs_error("host_file_create_only_managed: Failed to allocate file space for '%s': %s\n", 
                      file_path.c_str(), strerror(errno));
        close(fd);
        free(hdr);
        return false;
    }
    
    // Set temporary fd for header operations
    hdr->fd = fd;
    
    // Refine NVMe offsets
    host_refine_nvmeofst(hdr);
    
    // Write header to file
    if (write(fd, hdr, hdr_size) != (ssize_t)hdr_size) {
        geminifs_error("host_file_create_only_managed: Failed to write header to file '%s': %s\n", 
                      file_path.c_str(), strerror(errno));
        close(fd);
        free(hdr);
        return false;
    }
    
    // Close file immediately - we only created it, don't need to keep it open
    close(fd);
    
    // Create file record in log for consistency with g_open behavior
    if (file_manager != nullptr) {
        NVMeFileDesc new_desc;
        if (!file_manager->createFile(filename, new_desc, file_size)) {
            geminifs_error("host_file_create_only_managed: Failed to create file record in log for '%s'\n", filename.c_str());
            // Clean up the created physical file
            unlink(file_path.c_str());
            free(hdr);
            return false;
        }
        geminifs_debug("host_file_create_only_managed: Created file record in log for '%s' with slot %u\n", 
                      filename.c_str(), new_desc.slot_index);
    }
    
    // Clean up header memory
    free(hdr);
    
    geminifs_debug("host_file_create_only_managed: Created file '%s' with size %zu, hdr_size %zu (file closed)\n", 
                   filename.c_str(), file_size, hdr_size);
    
    return true;
}

