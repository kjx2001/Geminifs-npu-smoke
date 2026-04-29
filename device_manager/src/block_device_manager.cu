#include "block_device_manager.cuh"
#include "geminifs_helper.h"
#include "geminifs.h"

#include <cuda_runtime.h>
#include <unistd.h>
#include <cassert>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <filesystem>
#include <fcntl.h>

using namespace std;

// Static paths for system components
static char snvme_control_path[] = "/dev/snvm_control";
static char sys_config_path[] = "/mnt/sys_GPU_NVMe_topology.json";

// ---------------------------------------------------------------------------
// GPU kernel: initialise BlockAddressTranslator via placement new
// ---------------------------------------------------------------------------

__global__ void init_block_translator_kernel(BlockAddressTranslator* d_translator,
                                              struct geminiFS_hdr* device_hdr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (d_translator) BlockAddressTranslator(device_hdr);
    }
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

BlockDeviceManager::BlockDeviceManager(const nvme_ctrl_param& params)
    : is_initialized_(false)
{
    mount_path = params.mount_path;

    // Convert maxIOsize from KB to bytes and validate
    maxIOsize = params.maxIOsize * 1024;

    if (params.maxIOsize > 1024) {
        geminifs_error("BlockDeviceManager init failed: maxIOsize (%lu KB) exceeds maximum supported size (1024 KB)\n",
                       params.maxIOsize);
        throw std::runtime_error("maxIOsize exceeds supported limit of 1024 KB");
    }

    if (maxIOsize % 4096 != 0) {
        geminifs_error("BlockDeviceManager init failed: maxIOsize (%lu bytes) is not 4K aligned\n", maxIOsize);
        throw std::runtime_error("maxIOsize is not 4K aligned");
    }

    // Create mount directory if it doesn't exist
    std::filesystem::create_directories(mount_path);

    // Open the NVMe controller via libnvm
    controller = open_single_controller(params.pci_addr, params);

    // NOTE: QueueAcquireHelper allocation is NOT done here.
    // Queue scheduling is io_engine's responsibility.

    // Initialise persistent file metadata log
    std::string log_file_path = controller->dev_mount_path + "/nvme_file_log.dat";
    file_manager = std::make_unique<FileManager>(log_file_path, 1000);

    is_initialized_ = true;
}

BlockDeviceManager::~BlockDeviceManager() {
    cleanup_device_files();

    if (file_manager) {
        file_manager.reset();
    }
    if (controller) {
        controller.reset();
    }
}

// ---------------------------------------------------------------------------
// Unified open
// ---------------------------------------------------------------------------

__host__
void* BlockDeviceManager::g_open(std::string filename, size_t file_size, uint32_t o_flag) {
    if (!is_initialized()) {
        geminifs_error("g_open: BlockDeviceManager is not properly initialized\n");
        return nullptr;
    }

    assert(file_size % controller->blk_size == 0);

    if ((o_flag & O_HOST) && (o_flag & O_DEVICE)) {
        geminifs_error("g_open: Cannot specify both O_HOST and O_DEVICE flags\n");
        return nullptr;
    }
    if (!(o_flag & O_HOST) && !(o_flag & O_DEVICE)) {
        o_flag |= O_HOST;
    }

    geminifs_debug("g_open: Opening file '%s' with size %zu bytes, flags 0x%x\n",
                   filename.c_str(), file_size, o_flag);

    // Check if file exists in log
    uint32_t file_id = file_manager->getFileIdByFilename(filename);
    bool file_exists_in_log = (file_id != UINT32_MAX);
    NVMeFileDesc file_desc;
    if (file_exists_in_log) {
        file_manager->getFileById(file_id, file_desc);
    }

    void* result_fd = nullptr;

    if (file_exists_in_log) {
        std::filesystem::path file_path = controller->dev_mount_path;
        file_path = file_path / filename;

        if (std::filesystem::exists(file_path)) {
            if (o_flag & O_HOST) {
                result_fd = host_file_open_managed(file_path, O_RDWR);
            } else if (o_flag & O_DEVICE) {
                result_fd = device_file_open_managed(filename);
            }
            if (result_fd == nullptr) {
                geminifs_error("g_open: Failed to open existing file '%s'\n", filename.c_str());
                return nullptr;
            }
            geminifs_debug("g_open: Opened existing file '%s' from log slot %u\n",
                           filename.c_str(), file_desc.slot_index);
        } else {
            geminifs_debug("g_open: File '%s' exists in log but physical file missing, recreating\n",
                           filename.c_str());
            file_manager->deleteFile(file_id);
            file_exists_in_log = false;
        }
    }

    if (!file_exists_in_log) {
        std::filesystem::path file_path = controller->dev_mount_path;
        file_path = file_path / filename;

        if (std::filesystem::exists(file_path)) {
            geminifs_debug("g_open: Physical file exists but not in log, recreating\n");
            std::filesystem::remove(file_path);
        }

        if (o_flag & O_HOST) {
            result_fd = create_host_fd_internal(controller->page_size, file_size, filename);
            if (result_fd == nullptr) {
                geminifs_error("g_open: Failed to create host file '%s'\n", filename.c_str());
                return nullptr;
            }
            NVMeFileDesc new_desc;
            if (!file_manager->createFile(filename, new_desc, file_size)) {
                geminifs_error("g_open: Failed to create file record in log for '%s'\n", filename.c_str());
                host_file_close_managed((host_fd_t)result_fd);
                string file_path_str = string(controller->dev_mount_path) + "/" + filename;
                unlink(file_path_str.c_str());
                return nullptr;
            }
            geminifs_debug("g_open: Created new file '%s' with log slot %u\n",
                           filename.c_str(), new_desc.slot_index);
        } else if (o_flag & O_DEVICE) {
            result_fd = device_file_create_managed(controller->page_size, file_size, filename);
            if (result_fd == nullptr) {
                geminifs_error("g_open: Failed to create device file '%s'\n", filename.c_str());
                return nullptr;
            }
            NVMeFileDesc new_desc;
            if (!file_manager->createFile(filename, new_desc, file_size)) {
                geminifs_error("g_open: Failed to create file record in log for '%s'\n", filename.c_str());
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

// ---------------------------------------------------------------------------
// Host-side file operations
// ---------------------------------------------------------------------------

uint32_t BlockDeviceManager::host_file_create_managed(int block_size, size_t file_size,
                                                       const std::string& filename) {
    if (!is_initialized()) {
        geminifs_error("host_file_create_managed: not initialized\n");
        return UINT32_MAX;
    }

    assert(file_size % block_size == 0);
    auto nvpage_size = controller->page_size;
    assert(block_size % nvpage_size == 0);
    auto hdr_size = GEMINI_HDR_MAX_SIZE;

    struct geminiFS_hdr* hdr = (struct geminiFS_hdr*)malloc(hdr_size);
    if (!hdr) {
        geminifs_error("host_file_create_managed: Failed to allocate header\n");
        return UINT32_MAX;
    }

    string file_path_str = string(controller->dev_mount_path) + "/" + filename;

    hdr->magic_num = the_geminiFS_magic.magic_num;
    hdr->first_block_base = hdr_size;
    hdr->virtual_space_size = file_size;
    hdr->block_bit = __builtin_ctzll(block_size);

    int fd = open(file_path_str.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        geminifs_error("host_file_create_managed: Failed to create file '%s': %s\n",
                       file_path_str.c_str(), strerror(errno));
        free(hdr);
        return UINT32_MAX;
    }

    if (fallocate(fd, 0, 0, hdr_size + file_size) != 0) {
        geminifs_error("host_file_create_managed: Failed to allocate file space: %s\n", strerror(errno));
        close(fd);
        free(hdr);
        return UINT32_MAX;
    }

    hdr->fd = fd;
    host_refine_nvmeofst(hdr);
    close(fd);

    NVMeFileDesc file_desc;
    if (file_manager == nullptr || !file_manager->createFile(filename, file_desc, file_size)) {
        geminifs_error("host_file_create_managed: Failed to create file record for '%s'\n", filename.c_str());
        unlink(file_path_str.c_str());
        free(hdr);
        return UINT32_MAX;
    }

    free(hdr);
    geminifs_debug("host_file_create_managed: Created file '%s' size %zu, slot %u\n",
                   filename.c_str(), file_size, file_desc.slot_index);
    return file_desc.slot_index;
}

uint32_t BlockDeviceManager::host_file_create_managed(int block_size, size_t file_size) {
    if (!is_initialized()) {
        geminifs_error("host_file_create_managed: not initialized\n");
        return UINT32_MAX;
    }
    if (!file_manager) {
        geminifs_error("host_file_create_managed: FileManager not available\n");
        return UINT32_MAX;
    }

    NVMeFileDesc file_desc;
    if (!file_manager->createFile(file_desc, file_size)) {
        geminifs_error("host_file_create_managed: Failed to create file in FileManager\n");
        return UINT32_MAX;
    }

    std::string filename(file_desc.filename);
    assert(file_size % block_size == 0);
    auto nvpage_size = controller->page_size;
    assert(block_size % nvpage_size == 0);
    auto hdr_size = GEMINI_HDR_MAX_SIZE;

    struct geminiFS_hdr* hdr = (struct geminiFS_hdr*)malloc(hdr_size);
    if (!hdr) {
        geminifs_error("host_file_create_managed: Failed to allocate header\n");
        file_manager->deleteFile(file_desc.slot_index);
        return UINT32_MAX;
    }

    string file_path_str = string(controller->dev_mount_path) + "/" + filename;

    hdr->magic_num = the_geminiFS_magic.magic_num;
    hdr->first_block_base = hdr_size;
    hdr->virtual_space_size = file_size;
    hdr->block_bit = __builtin_ctzll(block_size);

    int fd = open(file_path_str.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        geminifs_error("host_file_create_managed: Failed to create file '%s': %s\n",
                       file_path_str.c_str(), strerror(errno));
        free(hdr);
        file_manager->deleteFile(file_desc.slot_index);
        return UINT32_MAX;
    }

    if (fallocate(fd, 0, 0, hdr_size + file_size) != 0) {
        geminifs_error("host_file_create_managed: Failed to allocate space for '%s': %s\n",
                       file_path_str.c_str(), strerror(errno));
        close(fd);
        free(hdr);
        file_manager->deleteFile(file_desc.slot_index);
        return UINT32_MAX;
    }

    hdr->fd = fd;
    host_refine_nvmeofst(hdr);
    close(fd);
    free(hdr);

    geminifs_debug("host_file_create_managed: Created file '%s' auto-name, size %zu, slot %u\n",
                   filename.c_str(), file_size, file_desc.slot_index);
    return file_desc.slot_index;
}

host_fd_t BlockDeviceManager::host_file_open_managed(const std::string& filepath, uint32_t o_flag) {
    if (!is_initialized()) {
        geminifs_error("host_file_open_managed: not initialized\n");
        return nullptr;
    }

    uint32_t file_flags = o_flag & ~(O_HOST | O_DEVICE);

    host_fd_t result = host_open_geminifs_file(filepath.c_str());
    if (result == nullptr) {
        if ((file_flags & O_ACCMODE) == O_RDONLY) {
            geminifs_error("host_file_open_managed: Failed to open '%s' in read-only mode\n", filepath.c_str());
        } else {
            geminifs_error("host_file_open_managed: Failed to open '%s'\n", filepath.c_str());
        }
        return nullptr;
    }

    if (file_manager != nullptr) {
        size_t hdr_size = result->first_block_base;
        file_manager->registerOpenFile(result, filepath, hdr_size);
    }

    return result;
}

host_fd_t BlockDeviceManager::host_file_open_managed(uint32_t id, uint32_t o_flag) {
    NVMeFileDesc file_desc;
    file_manager->getFileById(id, file_desc);
    std::string filepath = get_file_path(file_desc);
    return host_file_open_managed(filepath, o_flag);
}

void BlockDeviceManager::host_file_close_managed(host_fd_t fd) {
    if (!is_initialized()) {
        geminifs_error("host_file_close_managed: not initialized\n");
        return;
    }
    if (file_manager != nullptr) {
        file_manager->unregisterOpenFile(fd);
    }
    close(fd->fd);
    free(fd);
}

bool BlockDeviceManager::host_file_delete_managed(uint32_t file_id) {
    if (!is_initialized()) {
        geminifs_error("host_file_delete_managed: not initialized\n");
        return false;
    }
    if (!file_manager) {
        geminifs_error("host_file_delete_managed: FileManager not available\n");
        return false;
    }

    NVMeFileDesc file_desc;
    if (!file_manager->getFileById(file_id, file_desc)) {
        geminifs_error("host_file_delete_managed: File ID %u not found\n", file_id);
        return false;
    }

    std::string filename(file_desc.filename);
    return device_file_delete_single_managed(filename);
}

// ---------------------------------------------------------------------------
// Device-side file operations
// ---------------------------------------------------------------------------

dev_fd_t BlockDeviceManager::device_file_create_managed(int block_size, size_t file_size,
                                                         const std::string& filename) {
    if (!is_initialized()) {
        geminifs_error("device_file_create_managed: not initialized\n");
        return nullptr;
    }

    // Create host file with metadata
    host_fd_t host_fd = create_host_fd_internal(block_size, file_size, filename);
    if (host_fd == nullptr) {
        geminifs_error("device_file_create_managed: Failed to create host file '%s'\n", filename.c_str());
        return nullptr;
    }

    // Create file record in FileManager
    NVMeFileDesc file_desc;
    if (file_manager == nullptr || !file_manager->createFile(filename, file_desc, file_size)) {
        geminifs_error("device_file_create_managed: Failed to create file record for '%s'\n", filename.c_str());
        host_file_close_managed(host_fd);
        return nullptr;
    }

    size_t hdr_size = GEMINI_HDR_MAX_SIZE;

    // Copy extent metadata to GPU
    dev_fd_t device_fd = copy_host_fd_to_device(host_fd, hdr_size);
    if (device_fd == nullptr) {
        geminifs_error("device_file_create_managed: Failed to copy header to device for '%s'\n", filename.c_str());
        host_file_close_managed(host_fd);
        return nullptr;
    }

    // Allocate BlockAddressTranslator on GPU
    BlockAddressTranslator* d_translator = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_translator, sizeof(BlockAddressTranslator));
    if (cuda_err != cudaSuccess) {
        geminifs_error("device_file_create_managed: Failed to allocate translator on GPU: %s\n",
                       cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        return nullptr;
    }

    // Initialise translator on GPU
    init_block_translator_kernel<<<1, 1>>>(d_translator, (struct geminiFS_hdr*)device_fd);

    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        geminifs_error("device_file_create_managed: Translator init sync failed: %s\n",
                       cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        cudaFree(d_translator);
        return nullptr;
    }

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        geminifs_error("device_file_create_managed: Translator init failed: %s\n",
                       cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        cudaFree(d_translator);
        return nullptr;
    }

    // Track the handle
    {
        std::lock_guard<std::mutex> lock(device_files_mtx_);
        device_files_.emplace_back(host_fd, device_fd, hdr_size, filename, d_translator);
    }

    geminifs_debug("device_file_create_managed: Created '%s' translator=%p\n",
                   filename.c_str(), d_translator);
    return d_translator;
}

dev_fd_t BlockDeviceManager::device_file_open_managed(const std::string& filename) {
    if (!is_initialized()) {
        geminifs_error("device_file_open_managed: not initialized\n");
        return nullptr;
    }

    std::filesystem::path file_path = controller->dev_mount_path;
    file_path = file_path / filename;

    if (!std::filesystem::exists(file_path)) {
        geminifs_error("device_file_open_managed: Physical file '%s' not found\n", file_path.c_str());
        return nullptr;
    }

    // Open host file
    host_fd_t host_fd = host_file_open_managed(file_path, O_RDWR);
    if (host_fd == nullptr) {
        geminifs_error("device_file_open_managed: Failed to open host file '%s'\n", file_path.c_str());
        return nullptr;
    }

    size_t hdr_size = host_fd->first_block_base;

    // Copy extent metadata to GPU
    dev_fd_t device_fd = copy_host_fd_to_device(host_fd, hdr_size);
    if (device_fd == nullptr) {
        geminifs_error("device_file_open_managed: Failed to copy header to device for '%s'\n", filename.c_str());
        host_file_close_managed(host_fd);
        return nullptr;
    }

    // Allocate BlockAddressTranslator on GPU
    BlockAddressTranslator* d_translator = nullptr;
    cudaError_t cuda_err = cudaMalloc(&d_translator, sizeof(BlockAddressTranslator));
    if (cuda_err != cudaSuccess) {
        geminifs_error("device_file_open_managed: Failed to allocate translator on GPU: %s\n",
                       cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        return nullptr;
    }

    // Initialise translator on GPU
    init_block_translator_kernel<<<1, 1>>>(d_translator, (struct geminiFS_hdr*)device_fd);

    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        geminifs_error("device_file_open_managed: Translator init sync failed: %s\n",
                       cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        cudaFree(d_translator);
        return nullptr;
    }

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        geminifs_error("device_file_open_managed: Translator init failed: %s\n",
                       cudaGetErrorString(cuda_err));
        host_file_close_managed(host_fd);
        cudaFree(device_fd);
        cudaFree(d_translator);
        return nullptr;
    }

    // Track the handle
    {
        std::lock_guard<std::mutex> lock(device_files_mtx_);
        device_files_.emplace_back(host_fd, device_fd, hdr_size, filename, d_translator);
    }

    geminifs_debug("device_file_open_managed: Opened '%s' translator=%p\n",
                   filename.c_str(), d_translator);
    return d_translator;
}

dev_fd_t BlockDeviceManager::device_file_open_managed(uint32_t id) {
    NVMeFileDesc file_desc;
    file_manager->getFileById(id, file_desc);
    std::string filepath = get_file_path(file_desc);
    return device_file_open_managed(filepath);
}

void BlockDeviceManager::device_file_close_managed(dev_fd_t device_fd) {
    if (!is_initialized()) {
        geminifs_error("device_file_close_managed: not initialized\n");
        return;
    }

    std::lock_guard<std::mutex> lock(device_files_mtx_);

    // Search by translator pointer (device_fd is the translator returned by open/create)
    auto it = std::find_if(device_files_.begin(), device_files_.end(),
                           [device_fd](const DeviceFileHandle& handle) {
                               return handle.d_translator == (BlockAddressTranslator*)device_fd;
                           });

    if (it != device_files_.end()) {
        geminifs_debug("device_file_close_managed: Closing '%s'\n", it->filename.c_str());

        // Free GPU translator
        if (it->d_translator != nullptr) {
            cudaError_t err = cudaFree(it->d_translator);
            if (err != cudaSuccess) {
                geminifs_error("device_file_close_managed: Failed to free translator: %s\n",
                               cudaGetErrorString(err));
            }
        }

        // Free GPU header copy
        if (it->device_fd != nullptr) {
            cudaError_t err = cudaFree(it->device_fd);
            if (err != cudaSuccess) {
                geminifs_error("device_file_close_managed: Failed to free device memory: %s\n",
                               cudaGetErrorString(err));
            }
        }

        // Close host file
        host_file_close_managed(it->host_fd);

        device_files_.erase(it);
    } else {
        geminifs_error("device_file_close_managed: descriptor %p not found\n", device_fd);
    }
}

// ---------------------------------------------------------------------------
// Delete operations
// ---------------------------------------------------------------------------

bool BlockDeviceManager::device_file_delete_all_files_managed() {
    if (!is_initialized()) {
        geminifs_error("device_file_delete_all: not initialized\n");
        return false;
    }
    if (!file_manager) {
        geminifs_error("device_file_delete_all: FileManager not available\n");
        return false;
    }

    std::vector<uint32_t> all_file_ids = file_manager->getAllFileIds();
    if (all_file_ids.empty()) {
        geminifs_debug("device_file_delete_all: No files found, done\n");
        return true;
    }

    geminifs_debug("device_file_delete_all: Found %zu files to clean up\n", all_file_ids.size());

    size_t files_deleted = 0;
    size_t files_failed = 0;

    for (const auto& fid : all_file_ids) {
        NVMeFileDesc file_desc;
        if (!file_manager->getFileById(fid, file_desc)) {
            geminifs_error("device_file_delete_all: Failed to get descriptor for ID %u\n", fid);
            files_failed++;
            continue;
        }

        std::string filename(file_desc.filename);
        string file_path_str = string(controller->dev_mount_path) + "/" + filename;
        const char* file_path_cstr = file_path_str.c_str();

        bool physical_deleted = false;
        bool log_deleted = false;

        struct stat st;
        if (stat(file_path_cstr, &st) == 0) {
            if (unlink(file_path_cstr) == 0) {
                physical_deleted = true;
            } else {
                geminifs_error("device_file_delete_all: Failed to delete '%s': %s\n",
                               file_path_cstr, strerror(errno));
            }
        } else {
            physical_deleted = true; // doesn't exist = ok
        }

        if (file_manager->deleteFile(fid)) {
            log_deleted = true;
        } else {
            geminifs_error("device_file_delete_all: Failed to delete log entry for '%s'\n", filename.c_str());
        }

        if (physical_deleted && log_deleted) {
            files_deleted++;
        } else {
            files_failed++;
        }
    }

    file_manager->forcePersist();
    cleanup_device_files();

    geminifs_debug("device_file_delete_all: deleted=%zu, failed=%zu\n", files_deleted, files_failed);

    if (files_failed > 0) {
        geminifs_error("device_file_delete_all: %zu files could not be cleaned up\n", files_failed);
        return false;
    }
    return true;
}

bool BlockDeviceManager::device_file_delete_single_managed(const std::string& filename) {
    if (!is_initialized()) {
        geminifs_error("device_file_delete_single: not initialized\n");
        return false;
    }
    if (!file_manager) {
        geminifs_error("device_file_delete_single: FileManager not available\n");
        return false;
    }

    uint32_t file_id = FileManager::parseFileIdFromFilename(filename);
    if (file_id == UINT32_MAX) {
        geminifs_error("device_file_delete_single: Cannot parse file ID from '%s'\n", filename.c_str());
        return false;
    }

    string file_path_str = string(controller->dev_mount_path) + "/" + filename;

    bool physical_deleted = false;
    bool log_deleted = false;

    struct stat stat_buf;
    if (stat(file_path_str.c_str(), &stat_buf) == 0) {
        if (unlink(file_path_str.c_str()) == 0) {
            physical_deleted = true;
        } else {
            geminifs_error("device_file_delete_single: Failed to delete '%s': %s\n",
                           file_path_str.c_str(), strerror(errno));
        }
    } else {
        physical_deleted = true;
    }

    if (file_manager->deleteFile(file_id)) {
        log_deleted = true;
    } else {
        geminifs_error("device_file_delete_single: Failed to delete log entry for '%s'\n", filename.c_str());
    }

    // Remove from device_files_ if tracked
    {
        std::lock_guard<std::mutex> lock(device_files_mtx_);
        auto it = std::find_if(device_files_.begin(), device_files_.end(),
                               [&filename](const DeviceFileHandle& handle) {
                                   return handle.filename == filename;
                               });

        if (it != device_files_.end()) {
            if (it->d_translator != nullptr) {
                cudaFree(it->d_translator);
            }
            if (it->device_fd != nullptr) {
                cudaFree(it->device_fd);
            }
            device_files_.erase(it);
        }
    }

    file_manager->forcePersist();

    bool success = physical_deleted && log_deleted;
    if (!success) {
        geminifs_error("device_file_delete_single: Failed to completely delete '%s'\n", filename.c_str());
    }
    return success;
}

// ---------------------------------------------------------------------------
// Metadata queries
// ---------------------------------------------------------------------------

size_t BlockDeviceManager::device_file_get_managed_file_count() const {
    if (!is_initialized() || !file_manager) {
        return 0;
    }
    std::vector<uint32_t> all_ids = file_manager->getAllFileIds();
    return all_ids.size();
}

size_t BlockDeviceManager::device_file_validate_sizes(size_t expected_size) const {
    if (!is_initialized() || !file_manager) {
        return 0;
    }

    std::vector<uint32_t> all_ids = file_manager->getAllFileIds();
    if (all_ids.empty()) {
        return 0;
    }

    size_t valid_files = 0;

    for (const auto& fid : all_ids) {
        NVMeFileDesc file_desc;
        if (!file_manager->getFileById(fid, file_desc)) {
            continue;
        }

        std::string filename(file_desc.filename);
        std::string file_path = controller->dev_mount_path;
        if (file_path.back() != '/') {
            file_path += "/";
        }
        file_path += filename;

        struct stat file_stat;
        if (stat(file_path.c_str(), &file_stat) == 0) {
            size_t file_size = static_cast<size_t>(file_stat.st_size);
            size_t expected_total = expected_size + GEMINI_HDR_MAX_SIZE;

            if (file_size == expected_total) {
                valid_files++;
            }
        }
    }

    return valid_files;
}

// ---------------------------------------------------------------------------
// Internal create helper (create file + fill extent metadata)
// ---------------------------------------------------------------------------

bool BlockDeviceManager::host_file_create_only_managed(int block_size, size_t file_size,
                                                        const std::string& filename) {
    if (!is_initialized()) {
        geminifs_error("host_file_create_only_managed: not initialized\n");
        return false;
    }

    assert(file_size % block_size == 0);
    auto nvpage_size = controller->page_size;
    assert(block_size % nvpage_size == 0);
    size_t hdr_size = GEMINI_HDR_MAX_SIZE;

    struct geminiFS_hdr* hdr = (struct geminiFS_hdr*)malloc(hdr_size);
    if (!hdr) {
        geminifs_error("host_file_create_only_managed: Failed to allocate header\n");
        return false;
    }

    std::string file_path = build_file_path(controller->dev_mount_path, filename);

    if (!create_directories(controller->dev_mount_path)) {
        geminifs_error("host_file_create_only_managed: Failed to create directory '%s'\n",
                       controller->dev_mount_path.c_str());
        free(hdr);
        return false;
    }

    hdr->magic_num = the_geminiFS_magic.magic_num;
    hdr->first_block_base = hdr_size;
    hdr->virtual_space_size = file_size;
    hdr->block_bit = __builtin_ctzll(block_size);

    int fd = open(file_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        geminifs_error("host_file_create_only_managed: Failed to create '%s': %s\n",
                       file_path.c_str(), strerror(errno));
        free(hdr);
        return false;
    }

    if (fallocate(fd, 0, 0, hdr_size + file_size) != 0) {
        geminifs_error("host_file_create_only_managed: Failed to allocate space: %s\n", strerror(errno));
        close(fd);
        free(hdr);
        return false;
    }

    hdr->fd = fd;
    host_refine_nvmeofst(hdr);

    if (write(fd, hdr, hdr_size) != (ssize_t)hdr_size) {
        geminifs_error("host_file_create_only_managed: Failed to write header: %s\n", strerror(errno));
        close(fd);
        free(hdr);
        return false;
    }

    close(fd);

    if (file_manager != nullptr) {
        NVMeFileDesc new_desc;
        if (!file_manager->createFile(filename, new_desc, file_size)) {
            geminifs_error("host_file_create_only_managed: Failed to create log record for '%s'\n",
                           filename.c_str());
            unlink(file_path.c_str());
            free(hdr);
            return false;
        }
    }

    free(hdr);
    geminifs_debug("host_file_create_only_managed: Created '%s' size %zu\n", filename.c_str(), file_size);
    return true;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

host_fd_t BlockDeviceManager::create_host_fd_internal(int block_size, size_t file_size,
                                                       const std::string& filename) {
    assert(file_size % block_size == 0);
    auto nvpage_size = controller->page_size;
    assert(block_size % nvpage_size == 0);
    auto hdr_size = GEMINI_HDR_MAX_SIZE;

    struct geminiFS_hdr* hdr = (struct geminiFS_hdr*)malloc(hdr_size);
    if (!hdr) {
        geminifs_error("create_host_fd_internal: Failed to allocate header\n");
        return nullptr;
    }

    string file_path_str = string(controller->dev_mount_path) + "/" + filename;

    hdr->magic_num = the_geminiFS_magic.magic_num;
    hdr->first_block_base = hdr_size;
    hdr->virtual_space_size = file_size;
    hdr->block_bit = __builtin_ctzll(block_size);

    int fd = open(file_path_str.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd < 0) {
        geminifs_error("create_host_fd_internal: Failed to create '%s': %s\n",
                       file_path_str.c_str(), strerror(errno));
        free(hdr);
        return nullptr;
    }

    if (fallocate(fd, 0, 0, hdr_size + file_size) != 0) {
        geminifs_error("create_host_fd_internal: Failed to allocate space: %s\n", strerror(errno));
        close(fd);
        free(hdr);
        return nullptr;
    }

    hdr->fd = fd;
    host_refine_nvmeofst(hdr);

    if (file_manager != nullptr) {
        file_manager->registerOpenFile(hdr, filename, hdr_size);
    }

    return hdr;
}

dev_fd_t BlockDeviceManager::copy_host_fd_to_device(host_fd_t host_fd, size_t hdr_size) {
    void* device_fd = nullptr;

    cudaError_t err = cudaMalloc(&device_fd, hdr_size);
    if (err != cudaSuccess) {
        geminifs_error("copy_host_fd_to_device: Failed to allocate device memory: %s\n",
                       cudaGetErrorString(err));
        return nullptr;
    }

    err = cudaMemcpy(device_fd, host_fd, hdr_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("copy_host_fd_to_device: Failed to copy header to device: %s\n",
                       cudaGetErrorString(err));
        cudaFree(device_fd);
        return nullptr;
    }

    return device_fd;
}

void BlockDeviceManager::cleanup_device_files() {
    std::lock_guard<std::mutex> lock(device_files_mtx_);

    for (auto& handle : device_files_) {
        if (handle.d_translator != nullptr) {
            cudaFree(handle.d_translator);
        }
        if (handle.device_fd != nullptr) {
            cudaFree(handle.device_fd);
        }
    }

    device_files_.clear();
}

string BlockDeviceManager::get_file_path(const NVMeFileDesc& file_desc) {
    std::filesystem::path file_path = std::filesystem::path(controller->dev_mount_path) / file_desc.filename;
    return file_path.string();
}

ControllerPtr BlockDeviceManager::open_single_controller(const std::string& pci_addr,
                                                          const nvme_ctrl_param& params) {
    std::filesystem::path mount_path_param(params.mount_path);
    std::filesystem::path this_mount_path = mount_path_param;

    if (!std::filesystem::exists(this_mount_path)) {
        std::filesystem::create_directories(this_mount_path);
    }

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

bool BlockDeviceManager::check_snvme_control_exists() {
    if (access(snvme_control_path, F_OK) != 0) {
        geminifs_error("SNVM control device '%s' does not exist\n", snvme_control_path);
        return false;
    }
    return true;
}

bool BlockDeviceManager::check_sys_config_exists() {
    if (access(sys_config_path, F_OK) != 0) {
        geminifs_error("Sys GPU-NVMe topology '%s' does not exist\n", sys_config_path);
        return false;
    }
    return true;
}
