#ifndef GPU_FILE_MANAGER_H
#define GPU_FILE_MANAGER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstdint>
#include <cstdio>

#include "file.cuh"

// GPUFile的唯一标识符
using GPUFileId = uint64_t;

// 持久化文件的前缀
static const std::string default_db_path_prefix = "/home/zwh/Geminifs/db/geminifs";


// GPUFile 在磁盘和GPU内存中的状态
enum class GPUFileDescStatus : uint32_t {
    VALID = 0,      // 有效
    DELETED = 1     // 已被标记为删除
};

// GPUFile 的元数据描述结构 - 现在包含在磁盘和GPU内存中的位置信息
struct GPUFileDesc {
    GPUFileDescStatus status;      // 描述符状态 (4 bytes)
    GPUFileId file_id;             // GPUFile的全局唯一ID (8 bytes)
    uint32_t nr_nvme_files;        // NVMe文件数量 (4 bytes)
    size_t total_file_size;        // GPUFile的总大小 (8 bytes)
    size_t block_size;             // 块大小 (8 bytes)
    uint64_t create_time;          // 创建时间戳 (8 bytes)
    uint64_t modify_time;          // 最后修改时间戳 (8 bytes)

    uint64_t disk_offset;          // 在links.db文件中的偏移量 (8 bytes)
    uint64_t disk_size;            // 在links.db文件中的大小 (8 bytes)
    uint64_t gpu_link_offset;      // 在GPU显存堆中的偏移量 (8 bytes)
    uint64_t gpu_link_size;        // 在GPU显存堆中的大小 (8 bytes)

    // The offset of the GPU_File object itself in the heap.
    // The NVMe_Link array will be at gpu_link_offset.
    uint64_t gpu_handle_offset;    // (8 bytes)

    uint8_t reserved[0];          // 保留空间 (0 bytes now)
};


// 批量操作请求类型 - 保持不变
enum class BatchOperationType {
    ADD = 0,
    UPDATE = 1,
    REMOVE = 2
};

// 批量操作请求 - 保持不变
struct BatchNVMeLinkRequest {
    BatchOperationType operation;
    size_t link_index;
    NVMe_Link link_data;  // 仅ADD和UPDATE时使用
};


__device__ GPU_File* get_gpu_file(GPUFileId file_id, GPU_File** d_lookup_table);
/**
 * GPUFileManager - 基于自定义持久化和显存堆的元数据管理器
 * 
 * 设计特点:
 * - 启动时将所有元数据加载到预分配的GPU显存中，供Kernel直接访问。
 * - 使用自定义的空闲块链表管理GPU显存堆，支持高效的分配和回收。
 * - 使用双文件系统（元数据文件 + 数据文件）进行持久化。
 * - 写入操作为线性追加，读取为GPU内存直接访问。
 */
class GPUFileManager {
public:
    /**
     * 构造函数
     * @param file_prefix 持久化文件的前缀 (e.g., /data/geminifs)
     * @param gpu_heap_size_gb 在GPU上预分配的元数据堆大小 (GB)
     * @param max_files GPU端查找表能容纳的最大文件数
     */
    explicit GPUFileManager(const std::string& file_prefix = default_db_path_prefix, 
                            size_t gpu_heap_size_gb = 1,
                            uint32_t max_files = NUM_FILES); // Default to 100k files
    
    /**
     * 析构函数 - 释放GPU内存并关闭文件
     */
    ~GPUFileManager();

    // 禁止拷贝和赋值
    GPUFileManager(const GPUFileManager&) = delete;
    GPUFileManager& operator=(const GPUFileManager&) = delete;

    // === 核心GPUFile管理接口 ===
    
    /**
     * 创建新的GPUFile，并将其元数据加载到GPU
     * @param total_file_size 文件总大小
     * @param block_size 块大小
     * @param nvme_file_names NVMe文件名列表
     * @param controller_indexes 控制器索引列表
     * @param nvme_file_sizes 每个 NVMe 链接对应的文件大小（chunk 大小），与 names 对齐
     * @param out_desc [输出] 创建成功后的描述符
     * @return 新创建的GPUFileId，失败返回0
     */
    GPU_File* createGPUFile(size_t file_id, size_t total_file_size, size_t block_size,
                           const std::vector<std::string>& nvme_file_names,
                           const std::vector<size_t>& controller_indexes,
                           const std::vector<size_t>& nvme_file_sizes,
                           GPUFileDesc& out_desc);

    /**
     * 删除GPUFile
     * @param file_id 要删除的GPUFile ID
     * @return 成功返回true
     */
    bool deleteGPUFile(GPUFileId file_id);

    /**
     * 获取GPUFile描述符 (从CPU缓存)
     * @param file_id GPUFile ID
     * @param out_desc [输出] 描述符
     * @return 找到返回true
     */
    bool getGPUFileDesc(GPUFileId file_id, GPUFileDesc& out_desc) const;

    /**
     * 获取所有GPUFile的描述符
     */
    std::vector<GPUFileDesc> getAllGPUFileDescs() const;

    // 新增：获取指定文件的 NVMe_Link 列表（文件名与控制器索引）
    bool getLinksForFile(GPUFileId file_id, std::vector<NVMe_Link>& out_links) const;

    // === GPU Kernel接口支持 ===

    /**
     * 获取GPU端查找表的基地址
     * @return GPU内存指针 (GPU_File**)。Kernel可以使用此指针和File ID直接查找GPU_File。
     */
    GPU_File** getGPULookupTablePtr() const { return d_lookup_table_; }

    /**
     * 获取查找表的最大容量
     */
    uint32_t getMaxGpuFiles() const { return max_gpu_files_; }


    /**
     * 获取已注册文件数量
     * @return 文件数量
     */
    size_t getRegisteredFileCount() const;

private:
    // 持久化相关
    std::string metadata_path_;
    std::string links_path_;
    FILE* metadata_file_handle_;
    FILE* links_file_handle_;
    
    // GPU显存堆相关
    void* d_heap_memory_;
    size_t heap_size_bytes_;

    // GPU端查找表
    GPU_File** d_lookup_table_;
    uint32_t max_gpu_files_;

    // GPU显存堆空闲块管理
    struct FreeBlock {
        uint64_t offset;
        uint64_t size;
    };
    std::vector<FreeBlock> gpu_free_blocks_;
    bool gpu_heap_alloc(uint64_t size, uint64_t& out_offset);
    void gpu_heap_free(uint64_t offset, uint64_t size);

    mutable std::mutex mtx_;
    
    // 内存缓存
    mutable std::unordered_map<GPUFileId, GPUFileDesc> file_id_to_desc_map_;
    
    // 统计信息
    GPUFileId next_file_id_;
    
    // === 初始化和清理 ===
    bool initialize();
    void loadFromFiles();
    
    // === 辅助方法 ===
    GPUFileId allocateNewFileId();
    uint64_t getCurrentTimestamp() const;
    void persistGPUFileDesc(const GPUFileDesc& desc, long file_offset);

    // 序列化/反序列化不再需要，直接读写struct
};

// === Device helper ===
// 从调用方传入的 GPU 侧查找表中取 GPU_File*；调用方需保证 file_id 合法
__device__ inline GPU_File* get_gpu_file(GPUFileId file_id, GPU_File** d_lookup_table) {
    if (d_lookup_table == nullptr) {
        return nullptr;
    }
    return d_lookup_table[file_id];
}

#endif // GPU_FILE_MANAGER_H 