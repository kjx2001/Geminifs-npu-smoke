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
    
    // Clean up all files managed by this controller
    bool device_file_clean_all_files_managed();
    
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

// Smart pointer for NVMeController
using NVMeControllerPtr = std::shared_ptr<NVMeController>;

#endif // __NVME_CONTROLLER_H__
