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

constexpr uint32_t NUM_FILES = 1000000;

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
        uint64_t blk_id = va >> hdr->block_bit;
        uint64_t start_blk_id = 0;
        uint64_t end_blk_id = 0;

        for (size_t i = 0; i < hdr->extent_count; ++i) {
            end_blk_id += hdr->extents[i].fe_length >> hdr->block_bit;
            assert(blk_id < end_blk_id);
            if (blk_id >= start_blk_id) {
                return hdr->extents[i].fe_physical + ((blk_id - start_blk_id) << hdr->block_bit);
            }
            start_blk_id = end_blk_id;
        }

        assert(false && "Invalid virtual address for NVMe offset calculation");
        return 0;
    }

    __forceinline__ __device__ void nvme_xfer(size_t file_offset, size_t nbytes,
         uint64_t prp1, uint64_t prp2, FileXferType type)
    {
        auto queue_acquire_helper = this->queue_acquire_helper;
        assert(nbytes % this->nvme_page_size == 0);
        assert(file_offset % this->nvme_page_size == 0);
        nvme_ofst_t nvme_ofst = __get_nvmeofst(file_offset);
        uint64_t starting_lba = nvme_ofst >> hqps_block_size_log;
        // printf("NVMe_File: nvme_ofst: %lx, starting_lba: %lx, nbytes: %zu\n", 
        //        (unsigned long) nvme_ofst, (unsigned long) starting_lba, nbytes);
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
        nvme_xfer(file_offset, nbytes, prp1, prp2 ,FILE_XFER_READ);
    }
    __forceinline__ __device__ void write_out(uint64_t prp1, uint64_t prp2 , size_t file_offset, size_t nbytes) {
        nvme_xfer(file_offset, nbytes, prp1, prp2 , FILE_XFER_WRITE);
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