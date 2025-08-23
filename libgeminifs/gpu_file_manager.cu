#include "include/gpu_file_manager.cuh"
#include "geminifs_helper.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstring>
#include <chrono>
#include <algorithm>

// === GPUFileManager - Custom Persistence and GPU Heap Implementation ===

GPUFileManager::GPUFileManager(const std::string& file_prefix, size_t gpu_heap_size_gb, uint32_t max_files)
    : metadata_path_(file_prefix + "_metadata.db"),
      links_path_(file_prefix + "_links.db"),
      metadata_file_handle_(nullptr),
      links_file_handle_(nullptr),
      d_heap_memory_(nullptr),
      heap_size_bytes_(gpu_heap_size_gb * 1024 * 1024 * 1024),
      d_lookup_table_(nullptr),
      max_gpu_files_(max_files),
      next_file_id_(1) {
    
    if (!initialize()) {
        geminifs_error("GPUFileManager: Failed to initialize.\n");
        // In a real application, we might want to throw an exception here.
    }
}

GPUFileManager::~GPUFileManager() {
    if (d_lookup_table_) {
        cudaFree(d_lookup_table_);
    }
    if (d_heap_memory_) {
        cudaFree(d_heap_memory_);
    }
    if (metadata_file_handle_) {
        fclose(metadata_file_handle_);
    }
    if (links_file_handle_) {
        fclose(links_file_handle_);
    }
    geminifs_info("GPUFileManager: Cleanly shut down.\n");
}

bool GPUFileManager::initialize() {
    std::lock_guard<std::mutex> lock(mtx_);

    // 1. Open or create persistent files (same as before)
    metadata_file_handle_ = fopen(metadata_path_.c_str(), "rb+");
    if (!metadata_file_handle_) metadata_file_handle_ = fopen(metadata_path_.c_str(), "wb+");
    if (!metadata_file_handle_) {
        geminifs_error("GPUFileManager: Failed to open or create metadata file: %s\n", metadata_path_.c_str());
        return false;
    }

    links_file_handle_ = fopen(links_path_.c_str(), "rb+");
    if (!links_file_handle_) links_file_handle_ = fopen(links_path_.c_str(), "wb+");
    if (!links_file_handle_) {
        geminifs_error("GPUFileManager: Failed to open or create links file: %s\n", links_path_.c_str());
        return false;
    }

    // 2. Allocate GPU memory for the main heap AND the lookup table
    cudaError_t err = cudaMalloc(&d_heap_memory_, heap_size_bytes_);
    if (err != cudaSuccess) {
        geminifs_error("GPUFileManager: Failed to allocate %zu MB on GPU for heap: %s\n", heap_size_bytes_ / (1024*1024), cudaGetErrorString(err));
        return false;
    }
    geminifs_info("GPUFileManager: Allocated %zu MB for metadata heap on GPU.\n", heap_size_bytes_ / (1024*1024));
    
    err = cudaMalloc(&d_lookup_table_, max_gpu_files_ * sizeof(GPU_File*));
    if (err != cudaSuccess) {
        geminifs_error("GPUFileManager: Failed to allocate lookup table for %u files on GPU: %s\n", max_gpu_files_, cudaGetErrorString(err));
        cudaFree(d_heap_memory_);
        d_heap_memory_ = nullptr;
        return false;
    }
    cudaMemset(d_lookup_table_, 0, max_gpu_files_ * sizeof(GPU_File*)); // Initialize all pointers to null
    geminifs_info("GPUFileManager: Allocated GPU lookup table for %u files.\n", max_gpu_files_);


    // 3. Initialize free block list for the heap
    gpu_free_blocks_.push_back({0, heap_size_bytes_});

    // 4. Load existing data from files into memory and reconstruct GPU state
    loadFromFiles();

    return true;
}

void GPUFileManager::loadFromFiles() {
    // This function is called from initialize(), which already holds the lock.
    fseek(metadata_file_handle_, 0, SEEK_SET);

    GPUFileDesc read_desc; // Descriptor read from disk
    std::vector<char> link_data_buffer;

    while (fread(&read_desc, sizeof(GPUFileDesc), 1, metadata_file_handle_) == 1) {
        if (read_desc.status == GPUFileDescStatus::VALID) {
            if (read_desc.file_id >= max_gpu_files_) {
                geminifs_error("GPUFileManager: Loaded file ID %lu which exceeds max configured files %u. Skipping.\n", read_desc.file_id, max_gpu_files_);
                continue;
            }

            // The descriptor for in-memory state. We will populate its gpu offsets with new values.
            GPUFileDesc current_desc = read_desc;

            // Allocate new space on the GPU heap for links and the handle.
            // The old offsets from disk are ignored; we are rebuilding the heap from scratch.
            if (!gpu_heap_alloc(current_desc.gpu_link_size, current_desc.gpu_link_offset)) {
                geminifs_error("GPUFileManager: Failed to allocate %lu bytes on GPU heap for links of file %lu during loading.\n", current_desc.gpu_link_size, current_desc.file_id);
                continue; // Skip this file
            }
            if (!gpu_heap_alloc(sizeof(GPU_File), current_desc.gpu_handle_offset)) {
                geminifs_error("GPUFileManager: Failed to allocate %lu bytes on GPU heap for handle of file %lu during loading.\n", sizeof(GPU_File), current_desc.file_id);
                continue; // Skip this file
            }

            // Now that we have new GPU offsets, let's get the device pointers.
            NVMe_Link* d_links = reinterpret_cast<NVMe_Link*>(static_cast<char*>(d_heap_memory_) + current_desc.gpu_link_offset);
            GPU_File* d_handle = reinterpret_cast<GPU_File*>(static_cast<char*>(d_heap_memory_) + current_desc.gpu_handle_offset);

            // Load link data from links.db (using the original disk_offset) and copy to the newly allocated GPU space.
            if (read_desc.gpu_link_size > 0) {
                link_data_buffer.resize(read_desc.gpu_link_size);
                fseek(links_file_handle_, read_desc.disk_offset, SEEK_SET);
                if (fread(link_data_buffer.data(), read_desc.gpu_link_size, 1, links_file_handle_) != 1) {
                     geminifs_error("GPUFileManager: Failed to read link data for file %lu from disk.\n", current_desc.file_id);
                     gpu_heap_free(current_desc.gpu_handle_offset, sizeof(GPU_File));
                     gpu_heap_free(current_desc.gpu_link_offset, read_desc.gpu_link_size);
                     continue;
                }
                cudaMemcpy(d_links, link_data_buffer.data(), read_desc.gpu_link_size, cudaMemcpyHostToDevice);
            }
            
            // Create a temporary host-side GPU_File object to copy to the device handle.
            // It needs to point to the device location of the links.
            GPU_File h_gpu_file(d_links, current_desc.nr_nvme_files, current_desc.total_file_size, current_desc.block_size);
            cudaMemcpy(d_handle, &h_gpu_file, sizeof(GPU_File), cudaMemcpyHostToDevice);
            
            // Update the GPU-side lookup table with the pointer to the new handle location.
            cudaMemcpy(&d_lookup_table_[current_desc.file_id], &d_handle, sizeof(GPU_File*), cudaMemcpyHostToDevice);

            // Finally, store the descriptor with the new GPU offsets in our in-memory map.
            file_id_to_desc_map_[current_desc.file_id] = current_desc;
            next_file_id_ = std::max(next_file_id_, current_desc.file_id + 1);
        }
    }
    
    geminifs_info("GPUFileManager: Loaded and reconstructed %zu GPUFiles on the GPU.\n", file_id_to_desc_map_.size());
}


// === Core GPUFile Management ===

GPU_File* GPUFileManager::createGPUFile(size_t file_id, size_t total_file_size, size_t block_size,
                                       const std::vector<std::string>& nvme_file_names,
                                       const std::vector<size_t>& controller_indexes,
                                       const std::vector<size_t>& nvme_file_sizes,
                                       GPUFileDesc& out_desc) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    if (file_id >= max_gpu_files_) {
        geminifs_error("GPUFileManager: Cannot create new file, lookup table is full (max %u files).\n", max_gpu_files_);
        next_file_id_--; // Rollback ID allocation
        return 0;
    }
    
    if (nvme_file_names.size() != controller_indexes.size() || nvme_file_names.size() != nvme_file_sizes.size()) {
        geminifs_error("GPUFileManager: nvme_file_names, controller_indexes and nvme_file_sizes size mismatch\n");
        return 0;
    }
    
    // 1. Prepare link data on host
    uint64_t links_total_size = nvme_file_names.size() * sizeof(NVMe_Link);
    std::vector<NVMe_Link> links(nvme_file_names.size());
    uint64_t now = getCurrentTimestamp();

    for (size_t i = 0; i < nvme_file_names.size(); ++i) {
        NVMe_Link& link = links[i];
        strncpy(link.name, nvme_file_names[i].c_str(), sizeof(link.name) - 1);
        link.name[sizeof(link.name) - 1] = '\0';
        link.controller_index = controller_indexes[i];
        link.file_size = nvme_file_sizes[i];
    }

    // 2. Allocate space on GPU heap for BOTH links and the handle
    uint64_t links_gpu_offset, handle_gpu_offset;
    if (!gpu_heap_alloc(links_total_size, links_gpu_offset)) {
        geminifs_error("GPUFileManager: Not enough space on GPU metadata heap for links.\n");
        return 0;
    }
    if (!gpu_heap_alloc(sizeof(GPU_File), handle_gpu_offset)) {
        geminifs_error("GPUFileManager: Not enough space on GPU metadata heap for handle.\n");
        return 0;
    }
    
    // 3. Write link data to links.db file (append-only)
    fseek(links_file_handle_, 0, SEEK_END);
    long disk_offset = ftell(links_file_handle_);
    if (fwrite(links.data(), links_total_size, 1, links_file_handle_) != 1) {
        geminifs_error("GPUFileManager: Failed to write links to disk.\n");
        gpu_heap_free(links_gpu_offset, links_total_size);
        gpu_heap_free(handle_gpu_offset, sizeof(GPU_File));
        return 0;
    }
    fflush(links_file_handle_);

    // 4. Copy link data to GPU heap
    NVMe_Link* d_links = reinterpret_cast<NVMe_Link*>(static_cast<char*>(d_heap_memory_) + links_gpu_offset);
    cudaError_t err = cudaMemcpy(d_links, links.data(), links_total_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("GPUFileManager: Failed to copy links to GPU: %s\n", cudaGetErrorString(err));
        gpu_heap_free(links_gpu_offset, links_total_size);
        gpu_heap_free(handle_gpu_offset, sizeof(GPU_File));
        return 0;
    }

    // 5. Create GPU_File handle object ON THE GPU
    GPU_File* d_handle = reinterpret_cast<GPU_File*>(static_cast<char*>(d_heap_memory_) + handle_gpu_offset);
    GPU_File h_gpu_file(d_links, nvme_file_names.size(), total_file_size, block_size);
    err = cudaMemcpy(d_handle, &h_gpu_file, sizeof(GPU_File), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("GPUFileManager: Failed to copy handle to GPU: %s\n", cudaGetErrorString(err));
        gpu_heap_free(links_gpu_offset, links_total_size);
        gpu_heap_free(handle_gpu_offset, sizeof(GPU_File));
        return 0;
    }
    
    // 6. Update the GPU-side lookup table to point to the new handle
    err = cudaMemcpy(&d_lookup_table_[file_id], &d_handle, sizeof(GPU_File*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        geminifs_error("GPUFileManager: Failed to update GPU lookup table: %s\n", cudaGetErrorString(err));
        gpu_heap_free(links_gpu_offset, links_total_size);
        gpu_heap_free(handle_gpu_offset, sizeof(GPU_File));
        return 0;
    }

    // 7. Create and persist descriptor
    out_desc = {};
    out_desc.status = GPUFileDescStatus::VALID;
    out_desc.file_id = file_id;
    out_desc.nr_nvme_files = nvme_file_names.size();
    out_desc.total_file_size = total_file_size;
    out_desc.block_size = block_size;
    out_desc.create_time = now;
    out_desc.modify_time = now;
    out_desc.disk_offset = disk_offset;
    out_desc.disk_size = links_total_size;
    out_desc.gpu_link_offset = links_gpu_offset;
    out_desc.gpu_link_size = links_total_size;
    out_desc.gpu_handle_offset = handle_gpu_offset;

    fseek(metadata_file_handle_, 0, SEEK_END);
    long desc_offset = ftell(metadata_file_handle_);
    if (fwrite(&out_desc, sizeof(GPUFileDesc), 1, metadata_file_handle_) != 1) {
        geminifs_error("GPUFileManager: Failed to write descriptor to disk.\n");
        // Rollback all GPU state
        gpu_heap_free(links_gpu_offset, links_total_size);
        gpu_heap_free(handle_gpu_offset, sizeof(GPU_File));
        GPU_File* null_handle = nullptr;
        cudaMemcpy(&d_lookup_table_[file_id], &null_handle, sizeof(GPU_File*), cudaMemcpyHostToDevice);
        return 0;
    }
    fflush(metadata_file_handle_);

    // 8. Update CPU cache
    file_id_to_desc_map_[file_id] = out_desc;

    geminifs_info("GPUFileManager: Created GPU file %lu with %zu NVMe files. Handle at heap offset %lu, links at %lu\n",
                  file_id, nvme_file_names.size(), handle_gpu_offset, links_gpu_offset);
    return d_handle;
}

bool GPUFileManager::deleteGPUFile(GPUFileId file_id) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = file_id_to_desc_map_.find(file_id);
    if (it == file_id_to_desc_map_.end()) {
        geminifs_warn("GPUFileManager: Attempted to delete non-existent file ID %lu\n", file_id);
        return false;
    }

    GPUFileDesc desc = it->second;

    // 1. Mark as DELETED on disk
    desc.status = GPUFileDescStatus::DELETED;
    desc.modify_time = getCurrentTimestamp();

    // To find the offset, we must scan the metadata file.
    fseek(metadata_file_handle_, 0, SEEK_SET);
    GPUFileDesc temp_desc;
    long current_offset = 0;
    bool found_on_disk = false;
    while (fread(&temp_desc, sizeof(GPUFileDesc), 1, metadata_file_handle_) == 1) {
        if (temp_desc.file_id == file_id) {
            persistGPUFileDesc(desc, current_offset);
            found_on_disk = true;
            break;
        }
        current_offset += sizeof(GPUFileDesc);
    }
    
    if (!found_on_disk) {
        geminifs_error("GPUFileManager: Descrepancy! File %lu in cache but not found on disk for deletion.\n", file_id);
    }

    // 2. Free the GPU memory blocks for both the handle and the links
    gpu_heap_free(desc.gpu_handle_offset, sizeof(GPU_File));
    gpu_heap_free(desc.gpu_link_offset, desc.gpu_link_size);
    geminifs_info("GPUFileManager: Freed handle (%zuB) and links (%luB) on GPU heap.\n", sizeof(GPU_File), desc.gpu_link_size);

    // 3. Nullify the pointer in the GPU lookup table
    GPU_File* null_handle = nullptr;
    cudaMemcpy(&d_lookup_table_[file_id], &null_handle, sizeof(GPU_File*), cudaMemcpyHostToDevice);

    // 4. Remove from CPU cache
    file_id_to_desc_map_.erase(it);

    geminifs_info("GPUFileManager: Deleted GPU file %lu.\n", file_id);
    return true;
}

// getGPUFileDesc, getAllGPUFileDescs, getRegisteredFileCount remain the same

bool GPUFileManager::getGPUFileDesc(GPUFileId file_id, GPUFileDesc& out_desc) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = file_id_to_desc_map_.find(file_id);
    if (it != file_id_to_desc_map_.end()) {
        out_desc = it->second;
        return true;
    }
    return false;
}

std::vector<GPUFileDesc> GPUFileManager::getAllGPUFileDescs() const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<GPUFileDesc> descs;
    descs.reserve(file_id_to_desc_map_.size());
    for (const auto& pair : file_id_to_desc_map_) {
        descs.push_back(pair.second);
    }
    return descs;
}

bool GPUFileManager::getLinksForFile(GPUFileId file_id, std::vector<NVMe_Link>& out_links) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = file_id_to_desc_map_.find(file_id);
    if (it == file_id_to_desc_map_.end()) {
        return false;
    }
    const GPUFileDesc& desc = it->second;
    if (desc.gpu_link_size == 0) {
        out_links.clear();
        return true;
    }
    // 计算链接数量并从GPU堆复制到主机
    size_t num_links = desc.gpu_link_size / sizeof(NVMe_Link);
    std::vector<NVMe_Link> host_links(num_links);
    const void* d_links = static_cast<const char*>(d_heap_memory_) + desc.gpu_link_offset; 
    cudaError_t err = cudaMemcpy(host_links.data(), d_links, desc.gpu_link_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        geminifs_error("GPUFileManager::getLinksForFile: cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    out_links = std::move(host_links);
    return true;
}

size_t GPUFileManager::getRegisteredFileCount() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return file_id_to_desc_map_.size();
}


// gpu_heap_alloc, gpu_heap_free, and helper methods remain the same

bool GPUFileManager::gpu_heap_alloc(uint64_t size, uint64_t& out_offset) {
    // Simple first-fit algorithm.
    for (auto it = gpu_free_blocks_.begin(); it != gpu_free_blocks_.end(); ++it) {
        if (it->size >= size) {
            out_offset = it->offset;
            if (it->size == size) {
                // Exact match, remove the whole block
                gpu_free_blocks_.erase(it);
            } else {
                // Split the block
                it->offset += size;
                it->size -= size;
            }
            return true;
        }
    }
    // Not found
    out_offset = (uint64_t)-1;
    return false;
}

void GPUFileManager::gpu_heap_free(uint64_t offset, uint64_t size) {
    if (size == 0) return;

    // Insert the new free block, maintaining sorted order by offset
    auto it = std::lower_bound(gpu_free_blocks_.begin(), gpu_free_blocks_.end(), offset,
        [](const FreeBlock& block, uint64_t value) {
            return block.offset < value;
        });
    
    it = gpu_free_blocks_.insert(it, {offset, size});

    // Merge with previous block if adjacent
    if (it != gpu_free_blocks_.begin()) {
        auto prev = std::prev(it);
        if (prev->offset + prev->size == it->offset) {
            prev->size += it->size;
            it = gpu_free_blocks_.erase(it);
            --it; // after erasing, `it` is invalid, point it back to the merged block.
        }
    }

    // Merge with next block if adjacent
    auto next = std::next(it);
    if (next != gpu_free_blocks_.end()) {
        if (it->offset + it->size == next->offset) {
            it->size += next->size;
            gpu_free_blocks_.erase(next);
        }
    }
}


// === Helper Methods ===

GPUFileId GPUFileManager::allocateNewFileId() {
    // This is not thread-safe if called outside a locked context.
    return next_file_id_++;
}

uint64_t GPUFileManager::getCurrentTimestamp() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void GPUFileManager::persistGPUFileDesc(const GPUFileDesc& desc, long file_offset) {
    // This function assumes the caller holds the lock and knows the correct offset.
    fseek(metadata_file_handle_, file_offset, SEEK_SET);
    fwrite(&desc, sizeof(GPUFileDesc), 1, metadata_file_handle_);
    fflush(metadata_file_handle_);
}