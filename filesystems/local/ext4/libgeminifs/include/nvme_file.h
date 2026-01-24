#ifndef NVME_FILE_H
#define NVME_FILE_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <memory>
#include <cstdio> // For FILE*


constexpr size_t BITMAP_SIZE_BYTES = 2 * 128 * 1024; // 128 KB
constexpr size_t BITS_PER_BYTE = 8;
constexpr size_t MAX_RECORDS = BITMAP_SIZE_BYTES * BITS_PER_BYTE; // 1,048,576 records

// Forward declaration
struct LogHeader;

using NVMeFileId = uint32_t;
using NVMeCtrlId = uint32_t;



/*host file*/
// Forward declaration for host_fd_t
typedef struct geminiFS_hdr* host_fd_t;

// Structure to track opened host file descriptors for automatic cleanup
struct OpenFileHandle {
    host_fd_t fd;           // The file descriptor
    std::string filename;   // Filename for logging purposes
    size_t hdr_size;        // Size of the allocated header (including l1 array)
    
    OpenFileHandle(host_fd_t fd_, const std::string& name, size_t size)
        : fd(fd_), filename(name), hdr_size(size) {}
};

// File descriptor structure for NVMe files (Optimized)
struct NVMeFileDesc {
    uint32_t slot_index;     // The index of this record in the log area (4 bytes)
    uint8_t  padding[20];    // Padding to make it 64 bytes total (20 bytes)
    char filename[16];       // File name, less than 16 chars (16 bytes)
    size_t size;             // File size in bytes (8 bytes)
    uint64_t create_time;    // Creation timestamp (8 bytes)
    uint64_t modify_time;    // Last modification timestamp (8 bytes)
};

// Compile-time check to ensure the struct size is exactly 64 bytes.
static_assert(sizeof(NVMeFileDesc) == 64, "NVMeFileDesc size must be 64 bytes!");

// Log file header structure (512 bytes)
struct LogHeader {
    uint64_t magic_num;
    uint64_t log_file_size;
    uint64_t active_record_count;
    uint64_t total_record_capacity;
    uint64_t header_size;
    uint64_t bitmap_offset;
    uint64_t bitmap_size;
    uint64_t records_offset;
    uint64_t record_size; // This will be sizeof(NVMeFileDesc)
    char reserved[456];
};

class FileManager {
public:
    explicit FileManager(const std::string& log_path, size_t persistence_threshold = 1000);
    ~FileManager();
    
    FileManager(const FileManager&) = delete;
    FileManager& operator=(const FileManager&) = delete;
    FileManager(FileManager&&) = delete;
    FileManager& operator=(FileManager&&) = delete;

    bool createFile(const std::string& filename, NVMeFileDesc& out_desc, size_t file_size = 0);
    bool createFile(NVMeFileDesc& out_desc, size_t file_size = 0);  // Auto-generate filename based on slot ID
    bool deleteFile(uint32_t file_id);
    bool getFileById(uint32_t file_id, NVMeFileDesc& out_desc) const;
    uint32_t getFileIdByFilename(const std::string& filename) const;
    std::vector<uint32_t> getAllFileIds() const;
    
    // Utility methods for NVMe ID <-> filename conversion
    static std::string generateFilenameFromId(uint32_t file_id);
    static uint32_t parseFileIdFromFilename(const std::string& filename);

    // Host file descriptor management
    void registerOpenFile(host_fd_t fd, const std::string& filename, size_t hdr_size);
    bool unregisterOpenFile(host_fd_t fd);
    void closeAllOpenFiles();

    void forcePersist();

private:
    void initializeLogFile();
    void loadFromFile();
    void persistBitmap();
    long findNextFreeSlot();
    bool writeRecordToSlot(const NVMeFileDesc& desc, uint64_t slot_index);

    // private helper function
    static uint64_t getCurrentTimestamp();

    static const uint64_t MAGIC_NUMBER = 0x4E564D4546494C45; // "NVMEFILE"

    std::string log_file_path_;
    FILE* log_file_handle_ = nullptr; // Using C-style file handle
    LogHeader header_;

    std::vector<bool> dirty_bitmap_;
    
    mutable std::mutex mtx_;
    std::unordered_map<uint32_t, NVMeFileDesc> nvme_file_id_to_file_map_;  // ID to file mapping for efficiency

    // Host file descriptor tracking for automatic cleanup
    std::vector<OpenFileHandle> open_files_;
    mutable std::mutex open_files_mtx_;

    size_t persistence_threshold_;
    size_t pending_writes_count_;
};

#ifdef NVME_LAYER_DEBUG
#define nvme_layer_debug(fmt, ...) \
    printf("[DEBUG][%s:%d]: " fmt "", __func__, __LINE__, ##__VA_ARGS__);
#else
#define nvme_layer_debug(fmt, ...) \

#endif

#endif