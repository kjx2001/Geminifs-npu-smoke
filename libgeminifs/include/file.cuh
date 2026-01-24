#ifndef __FILE_CUH__
#define __FILE_CUH__


#include "nvm_error.h"
#include "nvm_types.h"
#include "utils.cuh"
#include "geminifs.h"
#include "ctrl.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdint.h>
#include <cuda/std/span>
#include "helper.cuh"
#include <cuda/atomic>
#include <vector> // Added for std::vector
#include <string> // Added for std::string
#include <algorithm> // Added for std::max, std::min

#include "geminifs_helper.h"
#include "geminifs.h"

constexpr uint32_t NUM_FILES = 2000000;

typedef enum FileXferType {
    FILE_XFER_READ = 0,
    FILE_XFER_WRITE = 1,
    FILE_XFER_INVALID = 2
}FileXferType;

typedef enum FileStatus {
    FILE_CLEAN = 0,
    FILE_DIRTY = 1,
    FILE_INVALID = 2
}FileStatus;


struct nvme_cmd__addr {
    nvme_ofst_t nvme_ofst;
    uint64_t ioaddr;
    size_t size;
};

class File {
private:
    virtual __forceinline__ __device__ void __xfer(size_t file_offset, size_t nbytes, FileXferType type) = 0;
    virtual __forceinline__ __device__ void __xfer(FileXferType type) = 0;
public:
    __forceinline__ __device__ void read_in(size_t file_offset, size_t nbytes) {
        __xfer(file_offset, nbytes, FILE_XFER_READ);
    }
    __forceinline__ __device__ void write_out(size_t file_offset, size_t nbytes) {
        __xfer(file_offset, nbytes, FILE_XFER_WRITE);
    }
};


class NVMe_File{

private: 
    Controller *ctrl; // represent one NVMe controller
    struct geminiFS_hdr *hdr;  // static header for get NVMeFile cls
    friend class GPU_File;

    __forceinline__ __device__ nvme_ofst_t __get_nvmeofst(vaddr_t va) const {
        assert(hdr);
        uint64_t blk_id = ((uint64_t)va) >> hdr->block_bit;
        uint64_t start_blk_id = 0;
        uint64_t end_blk_id = 0;

        geminifs_debug("__get_nvmeofst: va=0x%llx, blk_id=%llu, block_bit=%u, extent_count=%llu\n",
            (unsigned long long)va,
            (unsigned long long)blk_id,
            (unsigned)hdr->block_bit,
            (unsigned long long)hdr->extent_count);

        for (size_t i = 0; i < hdr->extent_count; ++i) {
            uint64_t add = ((uint64_t)hdr->extents[i].fe_length) >> hdr->block_bit;
            end_blk_id += add;

            geminifs_debug("va=0x%llx, extent[%llu], fe_length=%llu, fe_physical=0x%llx, start_blk_id=%llu, end_blk_id=%llu\n",
                (unsigned long long)va,
                (unsigned long long)i,
                (unsigned long long)hdr->extents[i].fe_length,
                (unsigned long long)hdr->extents[i].fe_physical,
                (unsigned long long)start_blk_id,
                (unsigned long long)end_blk_id);

            /* 检查是否在这个 extent 范围内： [start_blk_id, end_blk_id) */
            if (blk_id >= start_blk_id && blk_id < end_blk_id) {
                uint64_t offset_blk = (uint64_t)(blk_id - start_blk_id);
                uint64_t byte_offset = offset_blk << hdr->block_bit; /* 保证 64-bit */
                nvme_ofst_t result = (nvme_ofst_t)( (uint64_t)hdr->extents[i].fe_physical + byte_offset );

                geminifs_debug("Match: va=0x%llx, i=%llu, fe_physical=0x%llx, offset_blk=%llu, byte_offset=%llu, result=0x%llx\n",
                    (unsigned long long)va,
                    (unsigned long long)i,
                    (unsigned long long)hdr->extents[i].fe_physical,
                    (unsigned long long)offset_blk,
                    (unsigned long long)byte_offset,
                    (unsigned long long)result);

                return result;
            }
            start_blk_id = end_blk_id;
        }

        geminifs_error("Invalid virtual address for NVMe offset calculation: va=0x%llx\n", (unsigned long long)va);
        assert(false);
        return 0;
    }

    __forceinline__ __device__ void nvme_xfer(size_t file_offset, size_t nbytes,
         uint64_t prp1, uint64_t prp2, FileXferType type)
    {
        auto queue_acquire_helper = this->queue_acquire_helper;
        assert(nbytes % this->nvme_page_size == 0);
        assert(file_offset % this->nvme_page_size == 0);
        nvme_ofst_t nvme_ofst = __get_nvmeofst(file_offset);
        // uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint64_t starting_lba = nvme_ofst >> hqps_block_size_log;
        // printf("NVMe_File: tid:%d, offset: %lx, nvme_ofst: %lx, starting_lba: %lx, nbytes: %lu\n", 
        //        tid, file_offset, (unsigned long) nvme_ofst, (unsigned long) starting_lba, (unsigned long) nbytes);
        int queue = queue_acquire_helper->acquire_queue();
        QueuePair* qp = &ctrl->d_qps[queue];

        uint64_t n_blocks = nbytes >> hqps_block_size_log;
        uint16_t cid;

        queue_acquire_helper->issue_nvme_cmd(qp,
            prp1,
            prp2, // fixme
            n_blocks,
            starting_lba,
            type == FILE_XFER_READ ? NVM_IO_READ : NVM_IO_WRITE,
            &cid);
        queue_acquire_helper->poll(qp,cid);
    }


public:
    void *parent;
    size_t nvme_page_size;

   
    int hqps_block_size_log;
    QueueAcquireHelper *queue_acquire_helper;

    __forceinline__ __device__ NVMe_File(Controller * ctrl_, 
                                        struct geminiFS_hdr *hdr_): ctrl(ctrl_), hdr(hdr_) { }


    // Public method to get NVMe offset (wrapper for private __get_nvmeofst)
    __forceinline__ __device__ nvme_ofst_t get_nvme_offset(vaddr_t va) const {
        return __get_nvmeofst(va);
    }
    
    // Public method to get file virtual space size
    __forceinline__ __device__ uint64_t get_file_size() const {
        return hdr ? hdr->virtual_space_size : 0;
    }
    
    __forceinline__ __device__ void read_in(uint64_t prp1, uint64_t prp2 ,size_t file_offset, size_t nbytes) {
        nvme_xfer(file_offset, nbytes, prp1, prp2, FILE_XFER_READ);
    }
    __forceinline__ __device__ void write_out(uint64_t prp1, uint64_t prp2 , size_t file_offset, size_t nbytes) {
        nvme_xfer(file_offset, nbytes, prp1, prp2, FILE_XFER_WRITE);
    }
};
struct NVMe_Link {
    char name[16];
    size_t   controller_index;
    size_t   file_size;       // per-link NVMe file size (chunk size)
};
// /**
//  * @class GPU_File
//  * @brief A __host__ __device__ compatible class that represents a file in GPU memory.
//  *
//  * This class acts as a handle. An instance of this class is stored in the GPU heap,
//  * and it points to an array of NVMeFileLink objects, which are also in the GPU heap.
//  * Kernels can access file metadata through this object.
//  */
// class GPU_File {
// private:
//     NVMe_Link* links_;          // Pointer to the array of links in the GPU heap
//     uint32_t      num_links_;      // Number of links in the array
//     size_t        total_file_size_; // Total size of the GPU file (sum of all NVMe file sizes) 

// public:
//     // Constructor usable from both host and device
//     __host__ __device__ GPU_File(NVMe_Link* links, uint32_t num_links, size_t total_size, size_t blk_size)
//         : links_(links), num_links_(num_links), total_file_size_(total_size) {}

//     // --- Device-side Accessors for Kernels ---

//     __device__ uint32_t getNrFiles() const {
//         return num_links_;
//     }

//     /**
//      * @brief Retrieves a specific NVMe file link by index.
//      * @param index The index of the link to retrieve.
//      * @param out_link [out] The retrieved link data.
//      * @return True if the index is valid, false otherwise.
//      */
//     __device__ bool getNVMeFileLink(uint32_t index, NVMe_Link& out_link) const {
//         if (links_ == nullptr || index >= num_links_) {
//             return false;
//         }
//         // Direct memory copy from the GPU heap (pointed to by links_) to the kernel's local memory (out_link)
//         out_link = links_[index];
//         return true;
//     }

//     __device__ size_t getTotalFileSize() const { return total_file_size_; }
//     __device__ size_t getBlockSize() const { return block_size_; }
// };



#endif