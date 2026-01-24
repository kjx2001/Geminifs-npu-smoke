#ifndef PRP_MAPPING_ENTRY_H
#define PRP_MAPPING_ENTRY_H

#include <cstdint>

#ifndef __CUDACC__
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#endif

// PRP映射条目结构 (32字节)
struct PRPMappingEntry {
    uint32_t transfer_type; // NVMe cmd transfer type (4字节)
    uint32_t data_length;   // 数据长度 (4字节)
    uint64_t prp1;          // PRP1 (8字节)
    uint64_t prp2;          // PRP2 may be NULL (8字节)
    uint64_t tensor_offset; // 该PRP entry在granularity内的偏移位置 (8字节)
    
    __device__ __host__ PRPMappingEntry() : transfer_type(0), data_length(0), prp1(0), prp2(0), tensor_offset(0) {}
    __device__ __host__ PRPMappingEntry(uint32_t transfer_type, uint32_t data_len, uint64_t p1, uint64_t p2) 
        : transfer_type(transfer_type), data_length(data_len), prp1(p1), prp2(p2), tensor_offset(0) {}
    __device__ __host__ PRPMappingEntry(uint32_t transfer_type, uint32_t data_len, uint64_t p1, uint64_t p2, uint64_t offset) 
        : transfer_type(transfer_type), data_length(data_len), prp1(p1), prp2(p2), tensor_offset(offset) {}
};

#endif // PRP_MAPPING_ENTRY_H 