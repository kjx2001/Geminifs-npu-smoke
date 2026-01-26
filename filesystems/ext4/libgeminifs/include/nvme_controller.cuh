#ifndef __NVME_CONTROLLER_H__
#define __NVME_CONTROLLER_H__

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <filesystem>
#include "ctrl.h"
#include "nvme_file.h"
#include "geminifs.h"
#include "file.cuh"
#include <cuda/std/span>

using ControllerPtr = std::shared_ptr<Controller>;

// Forward declarations
struct nvme_ctrl_param;
struct DeviceFileHandle;

// Structure to manage host to device file descriptor mapping
struct DeviceFileHandle {
    host_fd_t host_fd;      // Host file descriptor
    dev_fd_t device_fd;     // Device file descriptor (GPU memory)
    size_t hdr_size;        // Size of the header structure
    std::string filename;   // Original filename
    NVMe_File* d_nvme_file; // GPU-side NVMeFile pointer
    
    DeviceFileHandle(host_fd_t h_fd, dev_fd_t d_fd, size_t size, const std::string& name, NVMe_File* nvme_file)
        : host_fd(h_fd), device_fd(d_fd), hdr_size(size), filename(name), d_nvme_file(nvme_file) {}
};

class NVMeController {
public:
    ControllerPtr controller;      // NVMe controller smart pointer
    std::unique_ptr<FileManager> file_manager; // File manager for this mount
    std::string mount_path;        // Mount path
    uint64_t maxIOsize;            // Maximum I/O size
    
    // GPU-side queue management helper
    QueueAcquireHelper *d_queue_acquire_helper;

    NVMeController(const ControllerPtr& ctrl, std::unique_ptr<FileManager> fm, const std::string& path)
        : controller(ctrl), file_manager(std::move(fm)), mount_path(path), maxIOsize(0), is_initialized_(false), d_queue_acquire_helper(nullptr) {}

    NVMeController(const nvme_ctrl_param& params);
    
    // Destructor
    ~NVMeController();

    void* g_open(std::string filename, size_t file_size, uint32_t o_flag);

    // Managed file operations - automatically handle file descriptor tracking
    uint32_t host_file_create_managed(int block_size, size_t file_size, const std::string& filename);
    uint32_t host_file_create_managed(int block_size, size_t file_size);  // Auto-generate filename based on file ID
    bool host_file_create_only_managed(int block_size, size_t file_size, const std::string& filename);
    host_fd_t host_file_open_managed(const std::string& filepath, uint32_t o_flag);
    host_fd_t host_file_open_managed(uint32_t nvme_file_id, uint32_t o_flag);
    void host_file_close_managed(host_fd_t fd);

    // Device file operations - manage host to device fd mapping
    dev_fd_t device_file_create_managed(int block_size, size_t file_size, const std::string& filename);
    dev_fd_t device_file_open_managed(const std::string& filename);
    dev_fd_t device_file_open_managed(uint32_t nvme_file_id);
    void device_file_close_managed(dev_fd_t device_fd);
    
    // delete up all files managed by this controller
    bool device_file_delete_all_files_managed();
    
    // Delete a single file managed by this controller by filename
    bool device_file_delete_single_managed(const std::string& filename);
    
    // Delete a single file managed by this controller by NVMe file ID
    bool host_file_delete_managed(uint32_t nvme_file_id);
    
    // Get the count of managed files
    size_t device_file_get_managed_file_count() const;
    
    // Validate files with expected size and return count of valid files
    size_t device_file_validate_sizes(size_t expected_size) const;
    
    // Check if controller is properly initialized
    bool is_initialized() const { return is_initialized_; }

    size_t next_nvme_file_id() { return next_nvme_file_id_++; }


private:
    // Private helper methods
    bool check_sys_config_exists();
    bool check_snvme_control_exists();
    ControllerPtr open_single_controller(const std::string& pci_addr, const nvme_ctrl_param& params);
    
    // Internal helper to create host_fd_t (for device operations)
    host_fd_t create_host_fd_internal(int block_size, size_t file_size, const std::string& filename);
    
    // Device file management helper methods
    dev_fd_t copy_host_fd_to_device(host_fd_t host_fd, size_t hdr_size);
    void cleanup_device_files();
    string get_file_path(const NVMeFileDesc& file_desc);
    
    // Initialization state
    bool is_initialized_;
    size_t next_nvme_file_id_ = 0;
    // Device file descriptor management
    std::vector<DeviceFileHandle> device_files_;
    mutable std::mutex device_files_mtx_;
};

// Smart pointer for NVMeController
using NVMeControllerPtr = std::shared_ptr<NVMeController>;

__device__
void nvme_controller_g_read(dev_fd_t device_fd, uint64_t prp1, uint64_t prp2, size_t file_offset, size_t nbytes);

__device__
void nvme_controller_g_write(dev_fd_t device_fd, uint64_t prp1, uint64_t prp2, size_t file_offset, size_t nbytes);

__global__
void nvme_controller_g_read_kernel(dev_fd_t device_fd, uint64_t prp1, uint64_t prp2, size_t file_offset, size_t nbytes);

__global__
void nvme_controller_g_write_kernel(dev_fd_t device_fd, uint64_t prp1, uint64_t prp2, size_t file_offset, size_t nbytes);
#endif // __NVME_CONTROLLER_H__
