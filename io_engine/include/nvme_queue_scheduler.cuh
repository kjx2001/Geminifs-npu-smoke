#ifndef __NVME_QUEUE_SCHEDULER_CUH__
#define __NVME_QUEUE_SCHEDULER_CUH__

/**
 * nvme_queue_scheduler.cuh -- GPU-side NVMe queue scheduler and command submitter.
 *
 * Canonical home: io_engine/include/nvme_queue_scheduler.cuh
 * Legacy origin:  filesystems/ext4/libgeminifs/include/helper.cuh (QueueAcquireHelper)
 *
 * Responsibilities:
 *   - Pick an NVMe submission queue for the calling GPU thread (round-robin)
 *   - Build an NVMe read/write command (nvm_cmd_t) and enqueue it on the SQ
 *   - Poll the completion queue until the command's CID completes
 *
 * Layer placement:
 *   io_engine owns this class. It is the NVMe protocol-level executor:
 *   given a QueuePair (from device_manager) and an LBA, it issues the command.
 *   The Block address translation and file metadata are device_manager's job.
 *
 * GPU-only: all methods are __device__. The object is allocated in device
 * memory and initialised via a placement-new CUDA kernel on the host.
 *
 * Dependencies:
 *   ctrl.h                 -- QueuePair
 *   nvm_cmd.h              -- nvm_cmd_t, nvm_cmd_header / data_ptr / rw_blks
 *   nvm_parallel_queue.h   -- get_cid, sq_enqueue, cq_poll, cq_dequeue, put_cid
 */

#include "ctrl.h"
#include "nvm_cmd.h"
#include "nvm_parallel_queue.h"
#include <cstdint>

class NvmeQueueScheduler {
private:
    int nr_queues;

public:
    __device__
    NvmeQueueScheduler(int nr_queues_) : nr_queues(nr_queues_) {}

    /**
     * Pick an NVMe submission queue for the calling thread.
     *
     * Current policy: round-robin based on a warp-scaled thread index.
     * Different policies (priority-based, dedicated, etc.) can be added
     * as alternative schedulers later.
     */
    __forceinline__ __device__
    int acquire_queue() {
        return (int)((blockDim.x * 32 + threadIdx.x) % this->nr_queues);
    }

    /**
     * Release a queue previously acquired via acquire_queue().
     *
     * Currently a no-op -- the round-robin policy does not track
     * per-queue ownership. Kept for API symmetry and future schedulers.
     */
    __forceinline__ __device__
    void release_queue(int queue_id) {
        (void)queue_id;
    }

    /**
     * Build an NVMe read/write command and enqueue it on the submission queue.
     *
     * @param qp            QueuePair to submit on (from device_manager's Controller)
     * @param prp1, prp2    PRP list pointers (pre-built by caller)
     * @param n_blocks      Number of logical blocks (each 512B or configured LBA size)
     * @param starting_lba  Starting logical block address
     * @param opcode        NVM_IO_READ or NVM_IO_WRITE
     * @param cid           [out] command ID used for polling completion
     */
    __forceinline__ __device__
    void issue_nvme_cmd(QueuePair* qp,
                        uint64_t prp1, uint64_t prp2,
                        uint64_t n_blocks, uint64_t starting_lba,
                        uint8_t opcode, uint16_t* cid) {
        nvm_cmd_t cmd;
        *cid = get_cid(&(qp->sq));
        nvm_cmd_header(&cmd, *cid, opcode, qp->nvmNamespace);
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
        sq_enqueue(&qp->sq, &cmd);
    }

    /**
     * Wait for the command with the given CID to complete, then release the CID.
     */
    __forceinline__ __device__
    void poll(QueuePair* qp, uint16_t cid) {
        uint32_t cq_pos = cq_poll(&qp->cq, cid);
        cq_dequeue(&qp->cq, cq_pos, &qp->sq);
        put_cid(&qp->sq, cid);
    }
};

#endif // __NVME_QUEUE_SCHEDULER_CUH__
