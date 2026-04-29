#ifndef __BLOCK_DEVICE_MANAGER_CUH__
#define __BLOCK_DEVICE_MANAGER_CUH__

/**
 * block_device_manager.cuh -- Host-side block device manager for device_manager layer.
 *
 * Canonical home: device_manager/include/block_device_manager.cuh
 * Legacy origin:  filesystems/ext4/libgeminifs/include/nvme_controller.cuh
 *
 * Responsibilities:
 *   - Open/close a single NVMe controller via libnvm
 *   - Allocate and GPU-initialise BlockAddressTranslator objects
 *   - Manage file lifecycle: create / open / close / delete
 *   - Track host<->device file handle mappings in DeviceFileHandle
 *   - Persist file metadata through FileManager (append-only log)
 *   - Expose controller resources (QueuePair*, page_size, etc.) for io_engine
 *
 * Layer placement:
 *   This class is the device_manager layer's primary host-side object.
 *   It owns the Controller lifetime and file metadata, but does NOT own
 *   queue scheduling or IO submission -- those belong to io_engine.
 *
 * Dependencies:
 *   libnvm ctrl.h               -- Controller, QueuePair
 *   nvme_file.h                 -- FileManager, NVMeFileDesc, host_fd_t
 *   geminifs.h                  -- nvme_ctrl_param, dev_fd_t, geminiFS_hdr
 *   block_address_translator.cuh -- BlockAddressTranslator (GPU-side)
 */

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <filesystem>

#include "ctrl.h"
#include "nvme_file.h"
#include "geminifs.h"
#include "block_address_translator.cuh"

using ControllerPtr = std::shared_ptr<Controller>;

// Forward declarations
struct nvme_ctrl_param;
struct DeviceFileHandle;

/**
 * DeviceFileHandle -- tracks a single open device file across host and GPU.
 *
 * When a file is opened for device-side (GPU) access:
 *   - host_fd      holds the mapped host header (geminiFS_hdr in host memory)
 *   - device_fd    holds the same header copied to GPU device memory
 *   - d_translator is the GPU-resident BlockAddressTranslator object
 */
struct DeviceFileHandle {
    host_fd_t               host_fd;        ///< Host-side geminiFS_hdr pointer
    dev_fd_t                device_fd;      ///< GPU-side copy of the header
    size_t                  hdr_size;       ///< Header allocation size (bytes)
    std::string             filename;       ///< Logical filename (for lookup and debug)
    BlockAddressTranslator* d_translator;   ///< GPU-resident address translator

    DeviceFileHandle(host_fd_t h_fd, dev_fd_t d_fd,
                     size_t size, const std::string& name,
                     BlockAddressTranslator* translator)
        : host_fd(h_fd), device_fd(d_fd), hdr_size(size),
          filename(name), d_translator(translator) {}
};

/**
 * BlockDeviceManager -- host-side manager for one block device (NVMe controller).
 *
 * One instance corresponds to one physical NVMe controller (one PCI address).
 * Multiple BlockDeviceManagers can be aggregated by a higher-level GPU controller.
 *
 * Thread safety: all public methods that touch device_files_ or the controller
 * state are guarded by internal mutexes.
 */
class BlockDeviceManager {
public:
    ControllerPtr                controller;     ///< libnvm controller handle
    std::unique_ptr<FileManager> file_manager;   ///< Persistent file metadata log
    std::string                  mount_path;     ///< EXT4 mount point for this controller
    uint64_t                     maxIOsize;      ///< Maximum single-IO size (bytes)

    BlockDeviceManager(const ControllerPtr& ctrl,
                       std::unique_ptr<FileManager> fm,
                       const std::string& path)
        : controller(ctrl), file_manager(std::move(fm)), mount_path(path),
          maxIOsize(0), is_initialized_(false) {}

    /**
     * Primary constructor -- open one NVMe controller from config params.
     * Throws std::runtime_error on failure.
     */
    explicit BlockDeviceManager(const nvme_ctrl_param& params);

    ~BlockDeviceManager();

    // --- Unified open (host or device) ---
    void* g_open(std::string filename, size_t file_size, uint32_t o_flag);

    // --- Host-side file operations ---
    uint32_t  host_file_create_managed(int block_size, size_t file_size, const std::string& filename);
    uint32_t  host_file_create_managed(int block_size, size_t file_size);
    bool      host_file_create_only_managed(int block_size, size_t file_size, const std::string& filename);
    host_fd_t host_file_open_managed(const std::string& filepath, uint32_t o_flag);
    host_fd_t host_file_open_managed(uint32_t file_id, uint32_t o_flag);
    void      host_file_close_managed(host_fd_t fd);
    bool      host_file_delete_managed(uint32_t file_id);

    // --- Device-side file operations ---
    dev_fd_t device_file_create_managed(int block_size, size_t file_size, const std::string& filename);
    dev_fd_t device_file_open_managed(const std::string& filename);
    dev_fd_t device_file_open_managed(uint32_t file_id);
    void     device_file_close_managed(dev_fd_t device_fd);
    bool     device_file_delete_all_files_managed();
    bool     device_file_delete_single_managed(const std::string& filename);

    // --- Metadata queries ---
    size_t device_file_get_managed_file_count() const;
    size_t device_file_validate_sizes(size_t expected_size) const;

    bool   is_initialized() const { return is_initialized_; }
    size_t next_file_id() { return next_file_id_++; }

    // --- io_engine integration ---
    // io_engine can access controller (public) to obtain:
    //   controller->d_qps      -- device-side QueuePair array
    //   controller->numQueues  -- queue count for scheduling
    //   controller->blk_size_log -- for LBA calculation
    //   controller->page_size  -- for alignment checks

private:
    bool   is_initialized_;
    size_t next_file_id_ = 0;

    std::vector<DeviceFileHandle> device_files_;
    mutable std::mutex            device_files_mtx_;

    // --- Private helpers ---
    bool          check_sys_config_exists();
    bool          check_snvme_control_exists();
    ControllerPtr open_single_controller(const std::string& pci_addr,
                                         const nvme_ctrl_param& params);

    host_fd_t create_host_fd_internal(int block_size, size_t file_size,
                                       const std::string& filename);
    dev_fd_t  copy_host_fd_to_device(host_fd_t host_fd, size_t hdr_size);
    void      cleanup_device_files();
    std::string get_file_path(const NVMeFileDesc& file_desc);
};

using BlockDeviceManagerPtr = std::shared_ptr<BlockDeviceManager>;

// --- GPU init kernel ---

/**
 * CUDA kernel to initialise a BlockAddressTranslator on GPU via placement new.
 * Called by BlockDeviceManager when opening a file for device-side access.
 */
__global__ void init_block_translator_kernel(BlockAddressTranslator* d_translator,
                                              struct geminiFS_hdr* device_hdr);

#endif // __BLOCK_DEVICE_MANAGER_CUH__
