/**
 * @file    geminifs_nvme_file.cpp
 * @brief   File Creation Monitor and Attribute Indexer for an NVMe Directory.
 *
 * @details
 * This program is designed to continuously monitor a specified target directory,
 * which is expected to be the mount point of an NVMe controller, and to
 * meticulously record all file creation events that occur within it.
 *
 * Its core functionalities include:
 *
 * 1.  **Event Monitoring**: Utilizes a low-level filesystem notification
 * mechanism (e.g., inotify on Linux) to efficiently listen for
 * `IN_CREATE` events.
 * 2.  **Attribute Recording**: For each newly created file, the program
 * immediately captures its essential attributes, such as file size,
 * permission mode, owner information, and creation/modification timestamps.
 * 3.  **UUID Assignment & Indexing**: Assigns a unique Universally Unique
 * Identifier (UUID) to each tracked file. The program maintains a
 * high-performance, in-memory index structure (e.g., `std::unordered_map`)
 * to provide an instantaneous mapping from a file's UUID back to its
 * original filename.
 *
 * This tool is primarily intended for applications that require rapid tracking,
 * management, and retrieval of a large number of files on high-speed storage
 * devices, such as data acquisition systems, log aggregation services, or the
 * underlying implementation of a content-addressable storage system.
 *
 * @author      [Your Name or Team Name]
 * @version     1.0.0
 * @date        2025-06-21
 *
 * @copyright   Copyright (c) 2025, [Your Organization or Company Name]. All rights reserved.
 * (Alternatively, use an open-source license, e.g., "Distributed under the MIT License.")
 *
 * @note
 * - This program relies on POSIX-compliant filesystem interfaces.
 * - For optimal performance on Linux, using `inotify` is recommended.
 * - Requires linking against a UUID library (e.g., libuuid) during compilation.
 *
 * @par Build Example (Linux):
 * @code
 * g++ nvme_file_tracker.cpp -o nvme_tracker -std=c++17 -luuid -lrt -pthread
 * @endcode
 *
 * @par Usage:
 * main.cpp
C++

#include "file_manager.h"
#include <iostream>
#include <vector>
#include <unistd.h> // for sleep

void printFileDesc(const NVMeFileDesc& desc) {
    std::cout << "  Slot Index: " << desc.slot_index << std::endl;
    std::cout << "  Filename: " << desc.filename << std::endl;
    std::cout << "  Size: " << desc.size << " bytes" << std::endl;
    std::cout << "  Create Time: " << desc.create_time << std::endl;
    std::cout << "  Modify Time: " << desc.modify_time << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

int main() {
    const std::string log_file = "./nvme_file_log_final.dat";

    // --- First Run: Initialize with a low persistence threshold for demonstration ---
    std::cout << "--- Initializing FileManager (persistence threshold = 5) ---" << std::endl;
    auto manager = std::make_unique<FileManager>(log_file, 5);

    std::vector<NVMeFileDesc> created_files;
    std::cout << "\n--- Creating 6 files (should trigger one persistence) ---" << std::endl;
    for (int i = 0; i < 6; ++i) {
        NVMeFileDesc desc;
        std::string filename = "file_" + std::to_string(i) + ".dat";
        if (manager->createFile(filename, desc)) {
            std::cout << "Created " << filename << " at slot " << desc.slot_index << std::endl;
            created_files.push_back(desc);
        }
    }

    // --- Deleting some files ---
    std::cout << "\n--- Deleting file_1.dat and file_3.dat ---" << std::endl;
    manager->deleteFile("file_1.dat");
    std::cout << "Deleted file_1.dat" << std::endl;
    manager->deleteFile("file_3.dat");
    std::cout << "Deleted file_3.dat" << std::endl;

    // --- Create a new file, it should occupy the first free slot (slot 1) ---
    std::cout << "\n--- Creating a new file ---" << std::endl;
    NVMeFileDesc new_file_desc;
    if (manager->createFile("new_file.txt", new_file_desc)) {
        std::cout << "Created new_file.txt, it should be in the first free slot." << std::endl;
        printFileDesc(new_file_desc);
    }

    std::cout << "\n--- Releasing manager instance, forcing final persistence ---" << std::endl;
    manager.reset();


    // --- Second Run: Simulate a restart ---
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "--- Simulating restart: Initializing a new FileManager ---" << std::endl;
    std::cout << "=======================================================" << std::endl;

    auto manager2 = std::make_unique<FileManager>(log_file, 1000);

    std::cout << "\n--- Listing all active files in new instance ---" << std::endl;
    auto all_files = manager2->getAllFilenames()
    ;
    for (const auto& name : all_files) {
        NVMeFileDesc desc;
        if(manager2->getFileByFilename(name, desc)) {
            std::cout << " - " << name << " (at slot " << desc.slot_index << ")" << std::endl;
        }
    }

    NVMeFileDesc dummy;
    if (!manager2->getFileByFilename("file_1.dat", dummy)) {
         std::cout << "\nCorrectly verified that 'file_1.dat' does not exist." << std::endl;
    }

    manager2.reset();
    return 0;
}
 * @code
 * ./nvme_tracker /mnt/nvme_drive
 * @endcode
 */

// =====================================================================================
//                             C++ Code Implementation Starts Here
// =====================================================================================

#include "nvme_file.h"
#include "geminifs.h"  // For geminiFS_hdr definition
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <algorithm>  // For std::find_if
#include <unistd.h>   // For close()


FileManager::FileManager(const std::string& log_path, size_t persistence_threshold)
    : log_file_path_(log_path),
      dirty_bitmap_(MAX_RECORDS, false),
      persistence_threshold_(persistence_threshold),
      pending_writes_count_(0) {

    bool need_initialize = false;
    
    // Check if file exists and is valid
    FILE* file = fopen(log_file_path_.c_str(), "rb");
    if (file == nullptr) {
        // File doesn't exist
        nvme_layer_debug("Log file does not exist. Creating new file...\n");
        need_initialize = true;
    } else {
        // File exists, check if it's valid
        
        // Check file size first
        fseek(file, 0, SEEK_END);
        long file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        
        if (file_size == 0) {
            // File is empty
            nvme_layer_debug("Log file is empty. Reinitializing...\n");
            need_initialize = true;
        } else if (file_size < static_cast<long>(sizeof(LogHeader))) {
            // File too small to contain a valid header
            nvme_layer_debug("Log file is too small to contain valid header. Reinitializing...\n");
            need_initialize = true;
        } else {
            // Try to read and validate header
            LogHeader temp_header;
            if (fread(&temp_header, sizeof(LogHeader), 1, file) != 1) {
                nvme_layer_debug("Cannot read header from log file. Reinitializing...\n");
                need_initialize = true;
            } else if (temp_header.magic_num != MAGIC_NUMBER) {
                std::cout << "Invalid magic number in log file. Expected: 0x" << std::hex << MAGIC_NUMBER 
                         << ", got: 0x" << temp_header.magic_num << std::dec << ". Reinitializing..." << std::endl;
                need_initialize = true;
            } else if (temp_header.record_size != sizeof(NVMeFileDesc)) {
                std::cout << "Incompatible record size in log file. Expected: " << sizeof(NVMeFileDesc)
                         << ", got: " << temp_header.record_size << ". Reinitializing..." << std::endl;
                need_initialize = true;
            } else if (temp_header.header_size != sizeof(LogHeader)) {
                std::cout << "Incompatible header size in log file. Expected: " << sizeof(LogHeader)
                         << ", got: " << temp_header.header_size << ". Reinitializing..." << std::endl;
                need_initialize = true;
            } else if (temp_header.active_record_count > MAX_RECORDS) {
                std::cout << "Invalid active record count in log file. Count: " << temp_header.active_record_count
                         << " exceeds maximum: " << MAX_RECORDS << ". Reinitializing..." << std::endl;
                need_initialize = true;
            } else {
                // Validate expected file size based on header
                long expected_size = sizeof(LogHeader) + BITMAP_SIZE_BYTES + (MAX_RECORDS * sizeof(NVMeFileDesc));
                if (file_size != expected_size) {
                    std::cout << "File size mismatch. Expected: " << expected_size 
                             << " bytes, got: " << file_size << " bytes. Reinitializing..." << std::endl;
                    need_initialize = true;
                } else {
                    // Header and basic structure look valid
                    std::cout << "Found valid log file with " << temp_header.active_record_count << " records." << std::endl;
                }
            }
        }
        
        fclose(file);
    }

    // Initialize file if needed
    if (need_initialize) {
        initializeLogFile();
    }

    // Open the file in binary read/write update mode for the lifetime of the object.
    log_file_handle_ = fopen(log_file_path_.c_str(), "r+b");
    if (log_file_handle_ == nullptr) {
         throw std::runtime_error("FATAL: Could not open log file for update: " + log_file_path_);
    }

    loadFromFile();
}

FileManager::~FileManager() {
    std::cout << "FileManager shutting down. Closing all open files..." << std::endl;
    closeAllOpenFiles();
    std::cout << "FileManager shutting down. Forcing persistence..." << std::endl;
    forcePersist();
    if (log_file_handle_ != nullptr) {
        fclose(log_file_handle_);
    }
}

void FileManager::initializeLogFile() {
    std::lock_guard<std::mutex> lock(mtx_);

    // "w+b" creates a new file for read/write, truncating if it exists.
    FILE* file = fopen(log_file_path_.c_str(), "w+b");
    if (file == nullptr) {
        throw std::runtime_error("FATAL: Could not create and initialize log file at " + log_file_path_);
    }

    // Setup header
    header_ = {};
    header_.magic_num = MAGIC_NUMBER;
    header_.header_size = sizeof(LogHeader);
    header_.bitmap_offset = sizeof(LogHeader);
    header_.bitmap_size = BITMAP_SIZE_BYTES;
    header_.records_offset = header_.bitmap_offset + header_.bitmap_size;
    header_.record_size = sizeof(NVMeFileDesc); // This will be 64
    header_.total_record_capacity = MAX_RECORDS;
    header_.active_record_count = 0;
    header_.log_file_size = header_.records_offset + (MAX_RECORDS * sizeof(NVMeFileDesc));

    // Write header
    if (fwrite(&header_, sizeof(LogHeader), 1, file) != 1) {
        fclose(file);
        throw std::runtime_error("FATAL: Failed to write header during initialization.");
    }

    // Write empty bitmap
    std::vector<char> empty_bitmap(BITMAP_SIZE_BYTES, 0);
    if (fwrite(empty_bitmap.data(), BITMAP_SIZE_BYTES, 1, file) != 1) {
        fclose(file);
        throw std::runtime_error("FATAL: Failed to write bitmap during initialization.");
    }

    // Write empty records space (all MAX_RECORDS)
    std::vector<char> empty_records(MAX_RECORDS * sizeof(NVMeFileDesc), 0);
    if (fwrite(empty_records.data(), MAX_RECORDS * sizeof(NVMeFileDesc), 1, file) != 1) {
        fclose(file);
        throw std::runtime_error("FATAL: Failed to write records space during initialization.");
    }

    fclose(file);
    std::cout << "Log file initialized at: " << log_file_path_ << std::endl;
}


void FileManager::loadFromFile() {
    std::lock_guard<std::mutex> lock(mtx_);

    // Read and validate header
    fseek(log_file_handle_, 0, SEEK_SET);
    if (fread(&header_, sizeof(LogHeader), 1, log_file_handle_) != 1) {
        throw std::runtime_error("FATAL: Could not read header from log file.");
    }

    if (header_.magic_num != MAGIC_NUMBER || header_.record_size != sizeof(NVMeFileDesc)) {
        throw std::runtime_error("FATAL: Log file is corrupt or incompatible. Expected record size: "
            + std::to_string(sizeof(NVMeFileDesc)) + ", but got: " + std::to_string(header_.record_size));
    }

    // Read bitmap into memory
    fseek(log_file_handle_, header_.bitmap_offset, SEEK_SET);
    std::vector<uint8_t> bitmap_buffer(BITMAP_SIZE_BYTES);
    if (fread(bitmap_buffer.data(), BITMAP_SIZE_BYTES, 1, log_file_handle_) != 1) {
        throw std::runtime_error("FATAL: Could not read bitmap from log file.");
    }

    for (size_t i = 0; i < MAX_RECORDS; ++i) {
        dirty_bitmap_[i] = (bitmap_buffer[i / BITS_PER_BYTE] >> (i % BITS_PER_BYTE)) & 1;
    }

    // Load records based on bitmap
    NVMeFileDesc desc;
    for (uint64_t i = 0; i < MAX_RECORDS; ++i) {
        if (dirty_bitmap_[i]) {
            fseek(log_file_handle_, header_.records_offset + i * header_.record_size, SEEK_SET);
            if (fread(&desc, sizeof(NVMeFileDesc), 1, log_file_handle_) != 1) {
                 std::cerr << "Warning: Failed to read record at slot " << i << ". Marking as invalid." << std::endl;
                 dirty_bitmap_[i] = false;
                 pending_writes_count_++;
                 continue;
            }

            if (desc.slot_index != i) {
                 std::cerr << "Warning: Corrupt record found at slot " << i << ". Slot index mismatch. Ignoring." << std::endl;
                 dirty_bitmap_[i] = false;
                 pending_writes_count_++;
                 continue;
            }

            nvme_file_id_to_file_map_[desc.slot_index] = desc;
        }
    }

    if(nvme_file_id_to_file_map_.size() != header_.active_record_count) {
        std::cerr << "Warning: Header count mismatch. Correcting..." << std::endl;
        header_.active_record_count = nvme_file_id_to_file_map_.size();
        pending_writes_count_++; // Mark for persistence
    }

    std::cout << "Successfully loaded " << nvme_file_id_to_file_map_.size() << " active file records." << std::endl;
}

void FileManager::forcePersist() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (pending_writes_count_ > 0 || persistence_threshold_ == 0) {
        persistBitmap();
    }
}

void FileManager::persistBitmap() {
    if (log_file_handle_ == nullptr) 
    {
        std::cerr << "Error: The file is NULL." << std::endl;
        return;
    }
    std::vector<uint8_t> bitmap_buffer(BITMAP_SIZE_BYTES, 0);
    for (size_t i = 0; i < MAX_RECORDS; ++i) {
        if (dirty_bitmap_[i]) {
            bitmap_buffer[i / BITS_PER_BYTE] |= (1 << (i % BITS_PER_BYTE));
        }
    }

    header_.active_record_count = nvme_file_id_to_file_map_.size();

    fseek(log_file_handle_, 0, SEEK_SET);
    if (fwrite(&header_, sizeof(LogHeader), 1, log_file_handle_) != 1) {
        std::cerr << "Error: Failed to write header during persistence." << std::endl;
    }

    fseek(log_file_handle_, header_.bitmap_offset, SEEK_SET);
    if (fwrite(bitmap_buffer.data(), BITMAP_SIZE_BYTES, 1, log_file_handle_) != 1) {
        std::cerr << "Error: Failed to write bitmap during persistence." << std::endl;
    }

    fflush(log_file_handle_);

    // std::cout << "Bitmap and header persisted. " << pending_writes_count_ << " pending writes cleared." << std::endl;
    pending_writes_count_ = 0;
}

long FileManager::findNextFreeSlot() {
    for (uint32_t i = 0; i < MAX_RECORDS; ++i) {
        if (!dirty_bitmap_[i]) {
            return i;
        }
    }
    return -1;
}

bool FileManager::writeRecordToSlot(const NVMeFileDesc& desc, uint64_t slot_index) {
    long offset = header_.records_offset + slot_index * header_.record_size;
    fseek(log_file_handle_, offset, SEEK_SET);
    return fwrite(&desc, sizeof(NVMeFileDesc), 1, log_file_handle_) == 1;
}

bool FileManager::createFile(const std::string& filename, NVMeFileDesc& out_desc, size_t file_size) {
    if (filename.length() >= 16) {
        std::cerr << "Error: Filename too long. Must be less than 16 characters." << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(mtx_);

    long slot = findNextFreeSlot();
    if (slot == -1) {
        std::cerr << "Error: Log file is full. Cannot create new file." << std::endl;
        return false;
    }

    // If filename follows NVMe ID rule (pure number), verify it matches the slot
    uint32_t parsed_id = parseFileIdFromFilename(filename);
    if (parsed_id != UINT32_MAX) {
        if (parsed_id != static_cast<uint32_t>(slot)) {
            std::cerr << "Warning: Filename '" << filename << "' doesn't match allocated slot " << slot 
                     << ". Using slot-based naming." << std::endl;
        }
    } else {
        // Non-numeric filename is okay, but warn about naming convention
        std::cerr << "Warning: Filename '" << filename << "' doesn't follow NVMe ID naming convention (should be numeric)." << std::endl;
    }

    // Check if the slot is already in use (this should not happen, but be safe)
    if (nvme_file_id_to_file_map_.find(slot) != nvme_file_id_to_file_map_.end()) {
        std::cerr << "Error: Slot " << slot << " is already occupied (this should not happen)." << std::endl;
        return false;
    }

    NVMeFileDesc new_desc = {};
    strncpy(new_desc.filename, filename.c_str(), sizeof(new_desc.filename) - 1);

    uint64_t now = getCurrentTimestamp();
    new_desc.slot_index = slot;
    new_desc.create_time = now;
    new_desc.modify_time = now;
    new_desc.size = file_size;

    if (!writeRecordToSlot(new_desc, slot)) {
        std::cerr << "Error: Failed to write new record to log file." << std::endl;
        return false;
    }

    dirty_bitmap_[slot] = true;
    nvme_file_id_to_file_map_[slot] = new_desc;
    out_desc = new_desc;

    pending_writes_count_++;
    if (persistence_threshold_ == 0 || (persistence_threshold_ > 0 && pending_writes_count_ >= persistence_threshold_)) {
        persistBitmap();
    }

    return true;
}

bool FileManager::createFile(NVMeFileDesc& out_desc, size_t file_size) {
    std::lock_guard<std::mutex> lock(mtx_);

    long slot = findNextFreeSlot();
    if (slot == -1) {
        std::cerr << "Error: Log file is full. Cannot create new file." << std::endl;
        return false;
    }

    // Generate filename based on NVMe ID rule: filename = slot_id.toString()
    std::string auto_filename = generateFilenameFromId(static_cast<uint32_t>(slot));
    
    if (auto_filename.length() >= 16) {
        std::cerr << "Error: Auto-generated filename too long: " << auto_filename << std::endl;
        return false;
    }

    NVMeFileDesc new_desc = {};
    strncpy(new_desc.filename, auto_filename.c_str(), sizeof(new_desc.filename) - 1);

    uint64_t now = getCurrentTimestamp();
    new_desc.slot_index = slot;
    new_desc.create_time = now;
    new_desc.modify_time = now;
    new_desc.size = file_size;

    if (!writeRecordToSlot(new_desc, slot)) {
        std::cerr << "Error: Failed to write new record to log file." << std::endl;
        return false;
    }

    dirty_bitmap_[slot] = true;
    nvme_file_id_to_file_map_[slot] = new_desc;
    out_desc = new_desc;

    pending_writes_count_++;
    if (persistence_threshold_ == 0 || (persistence_threshold_ > 0 && pending_writes_count_ >= persistence_threshold_)) {
        persistBitmap();
    }

    return true;
}

bool FileManager::deleteFile(uint32_t file_id) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto file_it = nvme_file_id_to_file_map_.find(file_id);
    if (file_it == nvme_file_id_to_file_map_.end()) return false;

    const NVMeFileDesc& desc_to_delete = file_it->second;
    uint64_t slot = desc_to_delete.slot_index;

    dirty_bitmap_[slot] = false;
    nvme_file_id_to_file_map_.erase(file_it);

    pending_writes_count_++;
    if (persistence_threshold_ == 0 || (persistence_threshold_ > 0 && pending_writes_count_ >= persistence_threshold_)) {
        persistBitmap();
    }

    return true;
}

// ---- Other public and helper methods (unchanged) ----

uint64_t FileManager::getCurrentTimestamp() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

bool FileManager::getFileById(uint32_t file_id, NVMeFileDesc& out_desc) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = nvme_file_id_to_file_map_.find(file_id);
    if (it != nvme_file_id_to_file_map_.end()) {
        out_desc = it->second;
        return true;
    }
    return false;
}

uint32_t FileManager::getFileIdByFilename(const std::string& filename) const {
    // Since NVMe file naming rule is: filename = nvme_id.toString()
    // We can directly parse the filename to get the ID
    uint32_t file_id = parseFileIdFromFilename(filename);
    
    if (file_id != UINT32_MAX) {
        // Verify the file actually exists in our map
        std::lock_guard<std::mutex> lock(mtx_);
        if (nvme_file_id_to_file_map_.find(file_id) != nvme_file_id_to_file_map_.end()) {
            return file_id;
        }
    }
    
    // If parsing fails or file not found, fall back to scanning (for non-standard filenames)
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& pair : nvme_file_id_to_file_map_) {
        if (std::string(pair.second.filename) == filename) {
            return pair.first;
        }
    }
    
    return UINT32_MAX; // Return invalid ID if not found
}

std::vector<uint32_t> FileManager::getAllFileIds() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<uint32_t> file_ids;
    file_ids.reserve(nvme_file_id_to_file_map_.size());
    for (const auto& pair : nvme_file_id_to_file_map_) {
        file_ids.push_back(pair.first);
    }
    return file_ids;
}

// Host file descriptor management methods
void FileManager::registerOpenFile(host_fd_t fd, const std::string& filename, size_t hdr_size) {
    std::lock_guard<std::mutex> lock(open_files_mtx_);
    open_files_.emplace_back(fd, filename, hdr_size);
    // std::cout << "Registered open file: " << filename << " (fd: " << fd << ", size: " << hdr_size << ")" << std::endl;
}

bool FileManager::unregisterOpenFile(host_fd_t fd) {
    std::lock_guard<std::mutex> lock(open_files_mtx_);
    auto it = std::find_if(open_files_.begin(), open_files_.end(), 
                          [fd](const OpenFileHandle& handle) { return handle.fd == fd; });
    
    if (it != open_files_.end()) {
        // std::cout << "Unregistered open file: " << it->filename << " (fd: " << fd << ")" << std::endl;
        open_files_.erase(it);
        return true;
    }
    return false;
}

void FileManager::closeAllOpenFiles() {
    std::lock_guard<std::mutex> lock(open_files_mtx_);
    
    for (const auto& handle : open_files_) {
        // std::cout << "Auto-closing file: " << handle.filename << " (fd: " << handle.fd << ")" << std::endl;
        
        // Close the file descriptor first
        if (handle.fd->fd > 0) {
            close(handle.fd->fd);
        }
        
        // Free the allocated memory (accounting for variable size)
        free(handle.fd);
    }
    
    open_files_.clear();
    std::cout << "All open files have been automatically closed." << std::endl;
}

// Static utility methods for NVMe ID <-> filename conversion
std::string FileManager::generateFilenameFromId(uint32_t file_id) {
    return std::to_string(file_id);
}

uint32_t FileManager::parseFileIdFromFilename(const std::string& filename) {
    try {
        return std::stoul(filename);
    } catch (const std::exception& e) {
        return UINT32_MAX; // Return invalid ID if parsing fails
    }
}



