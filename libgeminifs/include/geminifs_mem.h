#ifndef GEMINIFS_MEM_H
#define GEMINIFS_MEM_H

#include <vector>
#include <atomic>
#include <cstdint>
#include <cstring>
#include "buffer.h"

// PRP List 相关常量
constexpr size_t PRP_PAGE_SIZE = 4096;                           // 4KB 页面大小
constexpr size_t PRP_ENTRY_SIZE = 8;                             // 每个 PRP entry 8 字节
constexpr size_t PRP_ENTRIES_PER_PAGE = (PRP_PAGE_SIZE - PRP_ENTRY_SIZE) / PRP_ENTRY_SIZE;  // 511 entries (最后8B存类型)
constexpr size_t MAX_TRANSFER_SIZE = 128 * 1024 * 1024;          // 最大传输 128MB
constexpr size_t MAX_DATA_PER_PRP_PAGE = PRP_ENTRIES_PER_PAGE * PRP_PAGE_SIZE;  // 每个 PRP 页面可寻址的最大数据

// PRP 传输类型
enum PRPTransferType : uint64_t {
    PRP_TYPE_SINGLE_PAGE = 1,      // 单页传输 (≤ 4KB)
    PRP_TYPE_DUAL_PAGE = 2,        // 双页传输 (4KB < size ≤ 8KB) 
    PRP_TYPE_LIST = 3              // PRP List 传输 (> 8KB)
};

// PRP List 页面结构
struct PRPListPage {
    uint64_t prp_entries[PRP_ENTRIES_PER_PAGE];  // 511 个 PRP entries
    PRPTransferType transfer_type;                // 传输类型 (存储在页面最后 8 字节)
    
    PRPListPage() : transfer_type(PRP_TYPE_SINGLE_PAGE) {
        memset(prp_entries, 0, sizeof(prp_entries));
    }
};

// PRP 上下文结构
struct PRPContext {
    PRPTransferType transfer_type;                // 传输类型
    size_t data_size;                            // 数据总大小
    size_t num_prp_pages;                        // PRP 页面数量
    void** prp_pages;                            // PRP 页面指针数组 (GPU 内存)
    uint64_t* prp_page_addrs;                    // PRP 页面的物理地址数组
    
    PRPContext() : transfer_type(PRP_TYPE_SINGLE_PAGE), data_size(0), 
                   num_prp_pages(0), prp_pages(nullptr), prp_page_addrs(nullptr) {}
    
    ~PRPContext() {
        cleanup();
    }
    
    void cleanup();
    bool allocatePRPPages(size_t num_pages);
    bool buildPRPList(const std::vector<uint64_t>& ioaddrs);
};

// struct geminifs_metadata{
//     std::vector<ControllerPtr> ctrls;
//     std::atomic<bool> is_init{false};
//     GPUPoolId pool_id;
//     uint64_t file_size;
//     uint64_t file_block_size;
//     std::vector<cudaStream_t> streams;
// };

struct geminifs_dma{
    uint64_t *ioaddrs;                          // 原始 IO 地址数组
    DmaPtr dma_ptr;                             // DMA 指针
    PRPContext* prp_context;                    // PRP 上下文
    
    // Slice 相关字段
    std::vector<size_t> slice_sizes;            // 每个切片的大小
    std::vector<size_t> slice_offsets;          // 每个切片在原始数据中的偏移
    uint64_t slice_granularity;                 // 切片粒度（最小maxIOsize）
    size_t num_slices;                          // 切片数量
    
    geminifs_dma() : ioaddrs(nullptr), dma_ptr(nullptr), prp_context(nullptr), 
                     slice_granularity(0), num_slices(0) {}
    
    ~geminifs_dma() {
        if (prp_context) {
            delete prp_context;
            prp_context = nullptr;
        }
    }
};




#endif