#ifndef GPU_FILE_MANAGER_H
#define GPU_FILE_MANAGER_H

#include <cstddef>
#include <string>
#include <sys/types.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <utility>

#include "file.cuh"
#include "nvme_file.h"  // 引入NVMeFileDesc
#include "geminifs.h"
#include "prp_mapping_entry.h"

// 前向声明
class GPUController;
using GPUControllerPtr = std::shared_ptr<GPUController>;

// GPUFile的唯一标识符
using GPUFileId = uint32_t;

constexpr size_t GPU_BITMAP_SIZE_BYTES = 2 * 128 * 1024; // 128 KB
constexpr size_t GPU_BITS_PER_BYTE = 8;
constexpr size_t GPU_MAX_RECORDS = GPU_BITMAP_SIZE_BYTES * GPU_BITS_PER_BYTE; // 1,048,576 records

// GPU文件日志头结构 (512 bytes)
struct GPULogHeader {
    uint64_t magic_num;
    uint64_t log_file_size;
    uint64_t active_record_count;
    uint64_t total_record_capacity;
    uint64_t header_size;
    uint64_t bitmap_offset;
    uint64_t bitmap_size;
    uint64_t records_offset;
    uint64_t record_size; // This will be sizeof(GPUFileDesc)
    char reserved[456];
};


// 紧凑的NVMe文件映射结构  
struct CompactNVMeMapping {
    uint32_t nvme_file_ids[4];     // 最多4个NVMe文件的slot_index (16 bytes)
    uint8_t mapping_flags;         // 映射标志位 (1 byte)
                                   // 位7-4: 对应4个NVMe控制器是否有文件 (bit7对应controller0, bit6对应controller1, etc.)
                                   // 位3-1: 保留位
                                   // 位0: 该文件是否有效
    
    // 辅助方法
    inline bool is_valid() const { return mapping_flags & 0x01; }
    inline void set_valid(bool valid) { 
        mapping_flags = valid ? (mapping_flags | 0x01) : (mapping_flags & 0xFE); 
    }
    
    inline bool has_nvme_controller(int controller_index) const { 
        return (controller_index >= 0 && controller_index < 4) && (mapping_flags & (0x80 >> controller_index)); 
    }
    inline void set_nvme_controller_exists(int controller_index, bool exists) {
        if (controller_index >= 0 && controller_index < 4) {
            uint8_t mask = 0x80 >> controller_index;
            mapping_flags = exists ? (mapping_flags | mask) : (mapping_flags & ~mask);
        }
    }
    
    CompactNVMeMapping() : mapping_flags(0) {
        for (int i = 0; i < 4; i++) nvme_file_ids[i] = 0;
    }
};



// GPUFile 的元数据描述结构 - 简化版本
struct GPUFileDesc {
    GPUFileId file_id;             // GPUFile的全局唯一ID，同时也是slot_index (4 bytes)
    size_t total_file_size;        // GPUFile的总大小 (8 bytes)
    size_t block_size;             // 块大小，即第三个维度tensor内存对象的大小 (8 bytes)
    uint32_t tensor_shape[3];      // tensor的三维形状: [K/V, Layer, tensor_object_size] (12 bytes)
    CompactNVMeMapping nvme_mapping; // 紧凑的NVMe文件映射 (17 bytes)
};

template<typename T>
using cudaSpan = cuda::std::span<T>;
using NVMeFilesSpan = cudaSpan<NVMe_File*>;
using PRPMappingEntrySpan = cudaSpan<PRPMappingEntry>;

struct GPUIoContext {
    NVMeFilesSpan nvme_files;
    PRPMappingEntry *prp_entry;
    off_t prp_idx;
    off_t file_offset;
};

#define MAX_IOCTX_PER_BATCH 512u
#define MAX_LAYER_PER_BATCH MAX_IOCTX_PER_BATCH

struct BatchIoEntry {
    GPUIoContext d_ioctxs[MAX_IOCTX_PER_BATCH];
};

class BatchIoPool {
public:
    struct BatchIoNode {
        BatchIoEntry *entry;
        BatchIoNode* next;
        BatchIoNode(BatchIoEntry* e) : entry(e), next(nullptr) {}
        BatchIoNode() : entry(nullptr), next(nullptr) {}
    };
public:
    explicit __host__ BatchIoPool(size_t capacity) {
        auto cudaErr = cudaMalloc(&entries_, capacity * sizeof(BatchIoEntry));
        if (cudaErr != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for BatchIoPool");
        }
        nodes_.resize(capacity);
        for (size_t i = 0; i < capacity; ++i) {
            nodes_[i].entry = entries_ + i;
            release(entries_ + i);
        }
    }

    ~BatchIoPool() {
        cudaFree(entries_);
        nodes_.clear();
    }

    // 分配资源
    BatchIoEntry* acquire() {
        BatchIoNode* node;
        do {
            node = head_.load(std::memory_order_acquire);
            if (!node) return nullptr;
        } while (!head_.compare_exchange_weak(node, node->next,
                                             std::memory_order_acquire,
                                             std::memory_order_relaxed));
        return node->entry;
    }

    // 回收资源
    void release(BatchIoEntry* obj) {
        size_t idx = obj - entries_;   // entries_ 是数组首指针
        BatchIoNode* node = &nodes_[idx];
        BatchIoNode* oldHead;
        do {
            oldHead = head_.load(std::memory_order_relaxed);
            node->next = oldHead;
        } while (!head_.compare_exchange_weak(oldHead, node,
                                             std::memory_order_release,
                                             std::memory_order_relaxed));
    }

private:
    BatchIoEntry* entries_; // device point
    std::atomic<BatchIoNode*> head_ {nullptr};
    std::vector<BatchIoNode> nodes_;
};



class GPUFileManager {
public:
    explicit GPUFileManager(const std::string& log_path, GPUControllerPtr gpu_controller, size_t persistence_threshold = 1000);
    ~GPUFileManager();
    
    // 禁用拷贝和移动构造函数
    GPUFileManager(const GPUFileManager&) = delete;
    GPUFileManager& operator=(const GPUFileManager&) = delete;
    GPUFileManager(GPUFileManager&&) = delete;
    GPUFileManager& operator=(GPUFileManager&&) = delete;

    // 核心文件操作接口 - 重新设计为直接管理NVMe文件
    bool createGPUFiles(size_t total_file_size, size_t to_create, const std::vector<size_t>& tensor_shape, std::vector<GPUFileId> &out_files);

    bool createGPUFile(size_t total_file_size, const std::vector<size_t>& tensor_shape, GPUFileId& out_file_id);
    bool deleteGPUFile(GPUFileId file_id);
    bool initGPUFile(GPUFileId file_id);
    bool openGPUFile(GPUFileId& file_id);
    bool closeGPUFile(GPUFileId file_id);
    bool getGPUFileById(GPUFileId file_id, GPUFileDesc& out_desc) const;
    bool getDevFdById(GPUFileId file_id, std::vector<dev_fd_t>& dev_fds) const;
    bool getNVMeFilesSpanById(GPUFileId file_id, NVMeFilesSpan& nvme_files) const;

    std::vector<GPUFileId> getAllGPUFileIds() const;

    BatchIoEntry* allocateIoContexts();
    void releaseIoContexts(BatchIoEntry* entry);

    // 持久化管理
    void forcePersist();

private:
    // 初始化和清理
    void initializeLogFile();
    void loadFromFile();
    void persistBitmap();
    long findNextFreeSlot();
    std::vector<long> findMultipleFreeSlots(size_t count);
    bool writeRecordToSlot(const GPUFileDesc& desc, uint64_t slot_index);

    static const uint64_t MAGIC_NUMBER = 0x4750554649544C45; // "GPUFILE"

    // 持久化相关
    std::string log_file_path_;
    FILE* log_file_handle_ = nullptr; // 使用C风格文件句柄
    GPULogHeader header_;

    std::vector<bool> dirty_bitmap_;
    
    mutable std::mutex mtx_;
    // free_list_是**已打开的**GPUFile池
    std::vector<GPUFileId> free_list_;
    std::unordered_map<GPUFileId, NVMeFilesSpan> file_id_to_ctx_map_;
    std::unordered_map<GPUFileId, GPUFileDesc> file_id_to_desc_map_;
    std::map<std::pair<NVMeCtrlId, NVMeFileId>, dev_fd_t> nvme_file_id_to_dev_fd_map_;
    std::unique_ptr<BatchIoPool> io_ctx_pool_; // IO上下文池

    size_t persistence_threshold_;
    size_t pending_writes_count_;
    
    // GPU控制器管理
    GPUControllerPtr gpu_controller_;
    
    // 快速为一个nvme设备分配多个文件
    bool createNVMeFilesPerDevice(size_t file_size, size_t to_create, size_t nvme_idx, std::vector<uint32_t>& nvme_file_ids);
    // 辅助方法：管理NVMe文件
    bool createNVMeFilesForGPUFile(size_t file_size, std::vector<uint32_t>& nvme_file_ids);
    bool deleteNVMeFilesForGPUFile(const CompactNVMeMapping& nvme_mapping);
};


#endif // GPU_FILE_MANAGER_H 