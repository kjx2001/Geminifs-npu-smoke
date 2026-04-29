#ifndef __BLOCK_IO_CHANNEL_CUH__
#define __BLOCK_IO_CHANNEL_CUH__

/**
 * block_io_channel.cuh -- GPU-side per-file IO channel.
 *
 * Canonical home: io_engine/include/block_io_channel.cuh
 * Legacy origin:  NVMe_File::nvme_xfer / read_in / write_out in
 *                 filesystems/ext4/libgeminifs/include/file.cuh
 *
 * Responsibilities:
 *   - Bind a file's BlockAddressTranslator (from device_manager) with an
 *     NvmeQueueScheduler (from io_engine) into a single callable handle
 *   - Expose read_in / write_out for GPU kernels
 *   - Drive the IO path: translate file_offset -> nvme_ofst -> LBA, then
 *     issue NVMe command and poll for completion
 *
 * Layer placement:
 *   Lives entirely in io_engine. Consumes device_manager's translator +
 *   QueuePair array (memory objects only; does not own them).
 *
 * GPU-only: all methods are __device__. The object is allocated in device
 * memory and initialised via a placement-new CUDA kernel on the host.
 *
 * Dependencies:
 *   block_address_translator.cuh  -- BlockAddressTranslator (device_manager)
 *   nvme_queue_scheduler.cuh      -- NvmeQueueScheduler      (io_engine)
 *   ctrl.h                        -- QueuePair
 *   nvm_io.h                      -- NVM_IO_READ / NVM_IO_WRITE opcodes
 *   geminifs.h                    -- vaddr_t, nvme_ofst_t
 *   utils.cuh                     -- geminifs_debug / geminifs_error
 */

#include "block_address_translator.cuh"
#include "nvme_queue_scheduler.cuh"
#include "ctrl.h"
#include "nvm_io.h"
#include "geminifs.h"
#include "utils.cuh"
#include <cassert>
#include <cstdint>

enum BlockIoType : uint8_t {
    BLOCK_IO_READ    = 0,
    BLOCK_IO_WRITE   = 1,
    BLOCK_IO_INVALID = 2,
};

class BlockIoChannel {
private:
    BlockAddressTranslator* translator;     ///< From device_manager (borrowed)
    QueuePair*              d_qps;          ///< From device_manager's Controller (borrowed)
    NvmeQueueScheduler*     scheduler;      ///< Owned by io_engine

    /**
     * Core IO path: translate file_offset -> LBA, pick a queue, issue command, poll.
     */
    __forceinline__ __device__
    void xfer(size_t file_offset, size_t nbytes,
              uint64_t prp1, uint64_t prp2,
              BlockIoType type) {
        assert(nbytes      % this->nvme_page_size == 0);
        assert(file_offset % this->nvme_page_size == 0);

        // 1. Translate virtual address -> physical NVMe offset (device_manager)
        nvme_ofst_t nvme_ofst   = translator->translate(file_offset);
        uint64_t    starting_lba = nvme_ofst >> hqps_block_size_log;
        uint64_t    n_blocks     = nbytes    >> hqps_block_size_log;

        // 2. Pick a queue (io_engine scheduling)
        int        queue = scheduler->acquire_queue();
        QueuePair* qp    = &d_qps[queue];

        // 3. Build + submit + poll (io_engine)
        uint16_t cid;
        scheduler->issue_nvme_cmd(qp, prp1, prp2, n_blocks, starting_lba,
                                   type == BLOCK_IO_READ ? NVM_IO_READ : NVM_IO_WRITE,
                                   &cid);
        scheduler->poll(qp, cid);
    }

public:
    size_t nvme_page_size;       ///< NVMe page granularity (bytes), for alignment checks
    int    hqps_block_size_log;  ///< log2 of hardware queue-pair block size (for LBA calc)

    __forceinline__ __device__
    BlockIoChannel(BlockAddressTranslator* translator_,
                   QueuePair*              d_qps_,
                   NvmeQueueScheduler*     scheduler_)
        : translator(translator_), d_qps(d_qps_), scheduler(scheduler_),
          nvme_page_size(0), hqps_block_size_log(0) {}

    // --- Public IO interface ---

    __forceinline__ __device__
    void read_in(uint64_t prp1, uint64_t prp2,
                 size_t file_offset, size_t nbytes) {
        xfer(file_offset, nbytes, prp1, prp2, BLOCK_IO_READ);
    }

    __forceinline__ __device__
    void write_out(uint64_t prp1, uint64_t prp2,
                   size_t file_offset, size_t nbytes) {
        xfer(file_offset, nbytes, prp1, prp2, BLOCK_IO_WRITE);
    }

    // --- Metadata accessors (delegate to translator) ---

    __forceinline__ __device__
    nvme_ofst_t get_nvme_offset(vaddr_t va) const {
        return translator->translate(va);
    }

    __forceinline__ __device__
    uint64_t get_file_size() const {
        return translator->get_file_size();
    }
};

#endif // __BLOCK_IO_CHANNEL_CUH__