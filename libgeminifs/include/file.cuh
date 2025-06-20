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


class NVMeFile : public File {
private:
    Controller *ctrl; // represent one NVMe controller
    struct geminiFS_hdr *hdr;  // static header for get NVMeFile cls
    friend class GPUFile;


    // need optimized use extend tree
    // va: 0 -> file size (except for the header)
    __forceinline__ __device__ nvme_ofst_t __get_nvmeofst(vaddr_t va) {
        assert(hdr != nullptr);
        uint64_t l1_idx = va >> hdr->block_bit;
        return l1_idx < hdr->nr_l1 ? hdr->l1[l1_idx] : 0;
    }

        /*
    * @brief: transfer data between NVMe and GPU memory
    * @param buf_ioaddrs: the buffer of GPU memory, aligned with nvme page size
    * @param file_offset: the offset of the file
    * @param nbytes: the number of bytes to transfer
    * @param type: the type of transfer
    */                                   
    __forceinline__ __device__ void __xfer(size_t file_offset, size_t nbytes, FileXferType type) {
        auto nvme_page_size = this->nvme_page_size;
        auto file_block_size = this->block_size;

        auto queue_acquire_helper = this->queue_acquire_helper;
        auto nr_nvpage__per_block = file_block_size / nvme_page_size;
        auto nr_blocks = nbytes / file_block_size;
        assert(file_block_size % nvme_page_size == 0);

        assert(nbytes % file_block_size == 0);
        assert(file_offset % nvme_page_size == 0);

        auto metadata_offset = file_offset / nvme_page_size;
        //Todo: fix bug cids may same within different requests
        auto cur_cids = this->cids + metadata_offset;
        auto cur_sq_poss = this->sq_poss + metadata_offset;
        auto cur_nvme_cmds = this->nvme_cmds + metadata_offset;
        auto cur_buf_ioaddrs = this->prp_list_vaddr_base__of_cur_file + metadata_offset;
        auto cur_prp_list_addr = this->prp_list_ioaddr_base__of_cur_file + metadata_offset * sizeof(uint64_t);

        vaddr_t va = file_offset;
        size_t nr_nvme_cmds = 0;
        for (int blocks_idx = 0; blocks_idx < nr_blocks; blocks_idx ++){
            vaddr_t fileblock_va = va + blocks_idx * file_block_size;
            nvme_ofst_t nvme_ofst = __get_nvmeofst(fileblock_va);
            uint64_t ioaddr = cur_buf_ioaddrs[blocks_idx * nr_nvpage__per_block];
            // geminifs_info("fileblock_va %lx, nvme_ofst %lx, ioaddr %lx\n", fileblock_va, nvme_ofst, ioaddr);
            if (nr_nvme_cmds != 0 &&
                (cur_nvme_cmds[nr_nvme_cmds - 1].nvme_ofst +
                cur_nvme_cmds[nr_nvme_cmds - 1].size == nvme_ofst)) {
                cur_nvme_cmds[nr_nvme_cmds - 1].size += file_block_size;
            } else {
                cur_nvme_cmds[nr_nvme_cmds].nvme_ofst = nvme_ofst;
                cur_nvme_cmds[nr_nvme_cmds].ioaddr = ioaddr;
                cur_nvme_cmds[nr_nvme_cmds].size = file_block_size;
                nr_nvme_cmds++;
            }
        }

        int prp_list_offset = 0;
        int queue = queue_acquire_helper->acquire_queue();
        for (int cmd_idx = 0; cmd_idx < nr_nvme_cmds; cmd_idx++) {
            uint64_t prp2 = 0;
            if (cur_nvme_cmds[cmd_idx].size == nvme_page_size){
                prp2 = 0;
            } else if (cur_nvme_cmds[cmd_idx].size == 2 * nvme_page_size) {
                prp2 = cur_buf_ioaddrs[prp_list_offset + 1];
                geminifs_debug("prp1 %lx, prp2 %lx\n", cur_buf_ioaddrs[prp_list_offset], prp2);
            } else { // > 2 * nvme_page_size
                prp2 = cur_prp_list_addr + prp_list_offset * sizeof(uint64_t) + __WORD__;
            }
 
            // geminifs_info("nvme_ofst %lx, ioaddr %lx, prp2 %lx, size %ld\n", 
                // cur_nvme_cmds[cmd_idx].nvme_ofst, cur_nvme_cmds[cmd_idx].ioaddr, prp2, cur_nvme_cmds[cmd_idx].size);

            queue_acquire_helper->issue_nvme_cmd(ctrl, queue,
                cur_nvme_cmds[cmd_idx].nvme_ofst,
                cur_nvme_cmds[cmd_idx].ioaddr,
                prp2, // fixme
                cur_nvme_cmds[cmd_idx].size,
                this->hqps_block_size_log,
                type == FILE_XFER_READ ? NVM_IO_READ : NVM_IO_WRITE,
                cur_cids + cmd_idx, cur_sq_poss + cmd_idx);
            // prp_list_offset += (cur_nvme_cmds[cmd_idx].size / nvme_page_size) * sizeof(uint64_t);
            prp_list_offset += cur_nvme_cmds[cmd_idx].size / nvme_page_size;
        }

        for (int cmd_idx = 0; cmd_idx < nr_nvme_cmds; cmd_idx++) {
            queue_acquire_helper->poll(ctrl, queue, cur_cids[cmd_idx], cur_sq_poss[cmd_idx]);
        }
        queue_acquire_helper->release_queue(queue);

    }

    __forceinline__ __device__ void __xfer(FileXferType type) {
        this-> __xfer( 0, file_size, type);
    }

public:
    void *parent;
    void *prp_list_dev_ptr;
    uint64_t prp_list_ioaddr_base__of_cur_file;
    uint64_t *prp_list_vaddr_base__of_cur_file;
    struct nvme_cmd__addr *nvme_cmds;
    size_t max_nvme_cmds;
    size_t nvme_page_size;
    size_t block_size;
    size_t file_size;

    uint16_t *cids;
    uint16_t *sq_poss;
    int hqps_block_size_log;
    QueueAcquireHelper *queue_acquire_helper;

    __forceinline__ __device__ NVMeFile(Controller * ctrl_, 
                                        struct geminiFS_hdr *hdr_): ctrl(ctrl_), hdr(hdr_) { }

    __forceinline__ __device__ bool check_file_status(void) {
        return this->ctrl != nullptr && this->hdr != nullptr && 
                this->nvme_cmds != nullptr && this->cids != nullptr && this->sq_poss != nullptr;
    }

    __forceinline__ __device__ void print_file_info(void){
        geminifs_info("NVMeFile: %p, ctrl %p, hdr %p, file_size %ld, block_size %ld\n", 
            this, this->ctrl, this->hdr, this->file_size, this->block_size);
        geminifs_info("nvme_cmds %p, max_nvme_cmds %ld, nvme_page_size %ld\n", 
            this->nvme_cmds, this->max_nvme_cmds, this->nvme_page_size);
        geminifs_info("cids %p, sq_poss %p\n", this->cids, this->sq_poss);
    }
};

class GPUFile : public File{
private:
    NVMeFile *files;
    size_t nr_files;
    size_t total_file_size;

    bool prp_list_initialized = false;

    /*
    * @brief: transfer data between GPUFile(constituted by NVMeFiles) and GPU memory
    * @param buf_ioaddrs: the buffer of GPU memory, aligned with gpu page size (64KB)
    * @param file_offset: the offset of the file
    * @param nbytes: the number of bytes to transfer
    * @param type: the type of transfer
    */
    __forceinline__ __device__ void __xfer(size_t file_offset, size_t nbytes, FileXferType type) {
        // auto nvpage_size = this->nvme_page_size;
        auto nr_blocks = nbytes / this->block_size;
        
        auto block_start = file_offset / this->block_size;
        auto block_end = block_start + nr_blocks;

        auto first_window_size = min(this->nr_files - block_start % this->nr_files, nr_blocks);
        auto first_window_idx = block_start % this->nr_files;
        auto last_window_size = first_window_size == nr_blocks ? 0 : block_end % this->nr_files;
        auto remaining = (nr_blocks - first_window_size - last_window_size) / this->nr_files;

        auto __xfered_size = 0;
        geminifs_debug("__xfer: file_offset %ld, nbytes %ld, block_start %ld, block_end %ld, first_window_size %ld, first_window_idx %ld, last_window_size %ld, remaining %ld\n", 
            file_offset, nbytes, block_start, block_end, first_window_size, first_window_idx, last_window_size, remaining);

        for (auto idx = 0; idx < this->nr_files && __xfered_size != nbytes; idx ++) {
            auto file_idx = (first_window_idx + idx) % this->nr_files;
            auto xfer_blocks = remaining;
            auto xfer_offset = (file_offset / this->block_size / this->nr_files) * this->block_size;

            if (file_idx >= first_window_idx && file_idx < first_window_idx + first_window_size){
                xfer_blocks ++;
            } else {
                xfer_offset = (file_offset / this->block_size / this->nr_files + 1) * this->block_size;
            }

            if (file_idx < last_window_size) xfer_blocks ++; 
            if (xfer_blocks > 0) {
                auto file = this->files + file_idx;
                geminifs_debug("File[%ld](%p) xfer_offset %ld, xfer_size %ld, file_offset %ld, nbytes %ld\n", 
                    file_idx, file, xfer_offset, xfer_blocks * this->block_size, file_offset, nbytes);
                file->__xfer(xfer_offset , xfer_blocks * this->block_size, type);
                __xfered_size += xfer_blocks * this->block_size;
            }
        }
    }

    __forceinline__ __device__ void __xfer(FileXferType type) {
        this->__xfer(0, this->total_file_size, type);
    }

public:
    GPUFileId file_id;
    cuda_device_ref ref;
    // DmaPtr prp_list__of_total_pages;
    uint64_t *prp_list__of_total_pages_vaddr;
    uint64_t prp_list_ioaddr_base;
    size_t nvme_page_size;
    size_t block_size;

    int64_t hash_value;

    __device__ GPUFile(NVMeFile *files_, int nr_files_, size_t total_file_size)
                            : files(files_), nr_files(nr_files_), total_file_size(total_file_size) {
        ref = 0;
        geminifs_debug("GPUFile: %p, files %p, nr_files %ld, total_file_size %ld\n", 
            this, this->files, nr_files, total_file_size);
    }
    
    /*
    * @brief: scatter the ioaddr of GPU memory to the ioaddr of NVMe files
    * @param buf_ioaddrs: the ioaddr of GPU memory
    * @param nbytes: the number of bytes to reshape
    * @return: the ioaddr of reshaped buffer
    */
    __forceinline__ __device__ void scatter_ioaddrs(cuda::std::span<uint64_t> buf_ioaddrs, size_t file_offset, 
                                                            size_t nbytes, size_t ioaddr_size = __4KB__) {
        assert(buf_ioaddrs.size() * ioaddr_size == nbytes);

        auto nvpage_size = this->nvme_page_size;
        auto block_size = this->block_size;
        auto block_start = file_offset / block_size;
        auto nr_blocks = nbytes / block_size;
        auto nr_nvpage__per_block = block_size / nvpage_size;
        
        assert(nbytes % block_size == 0);
        assert(block_size % nvpage_size == 0);
        assert(nbytes + file_offset <= this->total_file_size);
        if (!prp_list_initialized)this->__set_prp_list(nvme_page_size);

        auto nr_nvme_pages__per_file = this->total_file_size / nvme_page_size / this->nr_files;
        for (auto block_idx = block_start; block_idx < block_start + nr_blocks; block_idx++) {
            auto file_index = block_idx % this->nr_files;
            auto block_offset = block_idx / this->nr_files;
            for (auto nvpage_idx = 0; nvpage_idx < nr_nvpage__per_block; nvpage_idx++) {
                this->prp_list__of_total_pages_vaddr[file_index * nr_nvme_pages__per_file + block_offset * 
                    nr_nvpage__per_block + nvpage_idx] =  buf_ioaddrs[(block_idx - block_start) * nr_nvpage__per_block + nvpage_idx];
            }
        }

    }

    __forceinline__ __device__ void scatter_ioaddrs(uint64_t buf_ioaddrs_start, size_t file_offset, 
                                                            size_t nbytes, size_t ioaddr_size = __4KB__) {
                                                            
        auto nvpage_size = this->nvme_page_size;
        auto block_size = this->block_size;
        auto block_start = file_offset / block_size;
        auto nr_blocks = nbytes / block_size;
        auto nr_nvpage__per_block = block_size / nvpage_size;

        assert(nbytes % block_size == 0);
        assert(block_size % nvpage_size == 0);
        assert(nbytes + file_offset <= this->total_file_size);


        if (!prp_list_initialized)this->__set_prp_list(nvme_page_size);
        
        // auto nr_blocks__per_file = this->total_file_size / block_size / this->nr_files;
        auto nr_nvme_pages__per_file = this->total_file_size / nvme_page_size / this->nr_files;
        for (auto block_idx = block_start; block_idx < block_start + nr_blocks; block_idx++) {
            auto file_index = block_idx % this->nr_files;
            auto block_offset = block_idx / this->nr_files;
            for (auto nvpage_idx = 0; nvpage_idx < nr_nvpage__per_block; nvpage_idx++) {
                this->prp_list__of_total_pages_vaddr[file_index * nr_nvme_pages__per_file + block_offset * 
                    nr_nvpage__per_block + nvpage_idx] =  buf_ioaddrs_start + 
                    ((block_idx - block_start) * nr_nvpage__per_block + nvpage_idx) * ioaddr_size;
            }
        }

    }

    __forceinline__ __device__ void __set_prp_list(size_t nvme_page_size) {
        auto nr_nvme_pages__per_file = this->total_file_size / nvme_page_size / this->nr_files;
        for (int i = 0; i < this->nr_files; i++) {
            auto file = this->files + i;
            file->prp_list_vaddr_base__of_cur_file = this->prp_list__of_total_pages_vaddr + 
                i * nr_nvme_pages__per_file;
            file->prp_list_ioaddr_base__of_cur_file = this->prp_list_ioaddr_base + 
                i * nr_nvme_pages__per_file * sizeof(uint64_t);
            geminifs_debug("prp_list_vaddr_base__of_cur_file %p, prp_list_ioaddr_base__of_cur_file %lx\n", 
                file->prp_list_vaddr_base__of_cur_file, file->prp_list_ioaddr_base__of_cur_file);
        }
        this->prp_list_initialized = true;
    }

    __device__ void print_file_info(void){
        geminifs_info("GPUFile: %p, files %p, nr_files %ld, total_file_size %ld\n", 
            this, this->files, nr_files, total_file_size);
        for (int i = 0; i < nr_files; i++) {
            auto file = this->files + i;
            geminifs_info("NVMeFile[%d]: %p, %p, %p\n", i, &file, file->ctrl, file->hdr);
        }
    }
};


// 得干掉重写
class GPUFilePool {
private:
    GPUFile *files;
    GPUPoolId pool_id; // using creating time to indicate the this file pool;
    size_t gpu_file_size;
    size_t pool_size;
    uint16_t *is_allocated;

public:
    size_t file_block_size;
    
    __forceinline__ __device__ uint64_t to_file_id(int64_t hash_value) {
        return (hash_value >> 63) ^ (hash_value << 1) + 1;
    }

    __forceinline__ __device__ uint64_t to_hash_value(uint64_t file_id) {
        uint64_t zz = file_id - 1;
        return (zz >> 1) ^ -static_cast<int64_t>(zz & 1);
    }

    __device__ GPUFilePool(GPUFile *files, size_t file_size, size_t pool_size) {
        this->files = files;
        this->gpu_file_size = file_size;
        this->pool_size = pool_size;
        this->pool_id = 0;
    }

    __device__ GPUFilePool(GPUFile *files, uint16_t *is_allocated, size_t file_size, size_t file_block_size, size_t pool_size) {
        this->files = files;
        this->is_allocated = is_allocated;
        this->gpu_file_size = file_size;
        this->file_block_size = file_block_size;
        this->pool_size = pool_size;
        this->pool_id = 0;
    }
    
    __device__ GPUFileId allocate_file(int64_t hash_value) {
        uint64_t file_id = this->to_file_id(hash_value) % this->pool_size;
        int retry = pool_size - 1;

        do {
            if (atomicCAS(&this->is_allocated[file_id], 0, 1)== 0) {
                break;
            }
            file_id = (file_id + 1) % pool_size;
        } while (true && --retry > 0);

        if (retry <= 0) {
            geminifs_error("GPUFilePool: no available file\n");
            geminifs_error("pool_size %ld, file_id %ld\n", pool_size, file_id);
            return UINT64_MAX;
        }

        auto file = files + file_id;
        file->ref++;
        file->hash_value = hash_value;
        file->file_id = file_id;
        return file_id;
    }
    // Todo: optimize the file index
    __device__ GPUFile *get_allocated_file(int64_t hash_value, bool ref = false) {
        auto file_id = this->to_file_id(hash_value);
        assert(file_id < pool_size);
        auto file = files + file_id;
        int retry = pool_size - 1;
        if (this->is_allocated[file_id] && file->hash_value == hash_value) {
            if (ref) file->ref++;
            return file;
        } else {
            while (true && --retry > 0) {
                file_id = (file_id + 1) % pool_size;
                file = files + file_id;
                if (this->is_allocated[file_id] && file->hash_value == hash_value) {
                    if (ref) file->ref++;
                    return file;
                }
            }
            geminifs_error("GPUFilePool: no allocated file\n");
            return nullptr;
        }
    }

    __device__ bool free_file(int64_t hash_value) {
        auto file = this->get_allocated_file(hash_value);
        if (file != nullptr) {
            if (atomicCAS(&this->is_allocated[file->file_id], 1, 0)) {
                geminifs_error("GPUFilePool: free file failed\n");
                return false;
            }
            file->hash_value = 0;
            file->file_id = 0;
            file->ref = 0;
            return true;
        }
        geminifs_error("GPUFilePool: no allocated file\n");
        return false;
    }

    __device__ GPUFile *get_file(GPUFileId file_id) {
        assert(file_id < pool_size);
        auto file = files + file_id;
        file->ref++;
        return file;
    }

    __device__ void put_file(GPUFileId file_id) {
        assert(file_id < pool_size);
        auto file = files + file_id;
        file->ref--;
    }

    __device__ void set_pool_id(GPUPoolId pool_id) {
        this->pool_id = pool_id;
    }

    __device__ GPUPoolId get_pool_id() {
        return this->pool_id;
    }
};

#endif