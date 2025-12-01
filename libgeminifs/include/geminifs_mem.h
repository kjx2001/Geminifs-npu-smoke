#ifndef GEMINIFS_MEM_H
#define GEMINIFS_MEM_H

#include <vector>
#include <atomic>
#include <cstdint>
#include <cstring>
#include "buffer.h"
#include "prp_mapping_entry.h"

// 前向声明
// struct PRPMappingEntry;


// PRP List 相关常量
constexpr size_t PRP_SIZE_SINGLE_PAGE = 4096;                           // 4KB 页面大小
constexpr size_t PRP_SIZE_DUAL_PAGE = 2 * PRP_SIZE_SINGLE_PAGE; // 双页传输大小 8KB
constexpr size_t PRP_ENTRY_SIZE = 8;                             // 每个 PRP entry 8 字节
constexpr size_t PRP_ENTRIES_PER_PAGE = (PRP_SIZE_SINGLE_PAGE - PRP_ENTRY_SIZE) / PRP_ENTRY_SIZE;  // 511 entries (最后8B存类型)
constexpr size_t MAX_TRANSFER_SIZE = 128 * 1024 * 1024;          // 最大传输 128MB
constexpr size_t MAX_DATA_PER_PRP_PAGE = PRP_ENTRIES_PER_PAGE * PRP_SIZE_SINGLE_PAGE;  // 每个 PRP 页面可寻址的最大数据

// PRP 传输类型
enum PRPTransferType : uint64_t {
    PRP_TYPE_SINGLE_PAGE = 0,      // 单页传输 (≤ 4KB)
    PRP_TYPE_DUAL_PAGE = 1,        // 双页传输 (4KB < size ≤ 8KB) 
    PRP_TYPE_LIST = 2              // PRP List 传输 (> 8KB)
};

// PRP List 页面结构
struct PRPListPage {
    uint64_t prp_entries[PRP_ENTRIES_PER_PAGE];  // 511 个 PRP entries
    PRPTransferType transfer_type;                // 传输类型 (存储在页面最后 8 字节)
    
    PRPListPage() : transfer_type(PRP_TYPE_SINGLE_PAGE) {
        memset(prp_entries, 0, sizeof(prp_entries));
    }
};

inline int getPrpTransferType(size_t slice_size) {
    if (slice_size <= PRP_SIZE_SINGLE_PAGE) {
        return PRP_TYPE_SINGLE_PAGE;
    } else if (slice_size <= PRP_SIZE_DUAL_PAGE) {
        return PRP_TYPE_DUAL_PAGE; // 大于4K小于等于8K
    } else {
        return PRP_TYPE_LIST; // 大于8K
    }
}



// struct geminifs_metadata{
//     std::vector<ControllerPtr> ctrls;
//     std::atomic<bool> is_init{false};
//     GPUPoolId pool_id;
//     uint64_t file_size;
//     uint64_t file_block_size;
//     std::vector<cudaStream_t> streams;
// };

// 子切片信息结构
struct SubSliceInfo {
    size_t offset;                              // 切片在granularity内的偏移
    size_t size;                                // 切片大小
    size_t global_offset;                       // 切片在整个tensor中的偏移
    
    SubSliceInfo(size_t o, size_t s, size_t go) : offset(o), size(s), global_offset(go) {}
};

// 粒度级别的切片信息
struct GranularitySliceGroup {
    uint64_t gpu_tensor_ptr;                    // 该granularity对应的GPU tensor数据指针
    size_t granularity_offset;                  // 该granularity在整个tensor中的起始偏移
    size_t granularity_size;                    // 该granularity的大小
    std::vector<SubSliceInfo> sub_slices;       // 该granularity内的所有子切片
    std::vector<PRPMappingEntry> prp_mappings;  // 该granularity对应的PRP映射条目
    
    GranularitySliceGroup(uint64_t ptr, size_t offset, size_t size) 
        : gpu_tensor_ptr(ptr), granularity_offset(offset), granularity_size(size) {}
};

struct geminifs_dma{
    uint64_t *ioaddrs;                          // 原始 IO 地址数组
    DmaPtr dma_ptr;                             // DMA 指针
    
    // 切片粒度信息
    uint64_t slice_granularity;                 // 切片粒度（最小maxIOsize）
    
    // 二级切片组织结构
    std::vector<GranularitySliceGroup> granularity_groups;  // 每个granularity对应的切片组
    
    // Type 2 PRP List GPU内存相关字段
    size_t type2_prp_count;                     // 第三种类型PRP的数量
    DmaPtr type2_prp_dma_ptr;                   // 第三种类型PRP GPU内存的DMA指针
    
    geminifs_dma() : ioaddrs(nullptr), dma_ptr(nullptr), 
                     slice_granularity(0), type2_prp_count(0), type2_prp_dma_ptr(nullptr) {}
    
    ~geminifs_dma() {
        // DMA指针和GPU内存由系统自动管理，无需手动释放
    }
};




#endif