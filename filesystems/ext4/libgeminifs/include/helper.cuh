#ifndef __HELPER_CUH__
#define __HELPER_CUH__

#include "geminifs.h"
#include "nvm_cmd.h"
#include "utils.cuh"
#include <cstdio>
#include <cuda/atomic>
#include <cuda/semaphore>
#include <ctrl.h>
#include <nvm_parallel_queue.h>

//Todo: need to intergrate into GPU block layer

class QueueAcquireHelper {
private:
    int nr_queues;
public:
    __device__ QueueAcquireHelper(int nr_queues) {
        this->nr_queues = nr_queues;
    }
    __forceinline__ __device__ int acquire_queue() {
        return (int)((blockDim.x * 32 + threadIdx.x) % this->nr_queues);
        // return (my_lane_id() * ((get_smid() % 25) + 1)) % this->nr_queues;
    }
    __forceinline__ __device__ void release_queue(int queue_id) {

    }
    __forceinline__ __device__ void issue_nvme_cmd(
                                        QueuePair* qp,
                                        uint64_t prp1,
                                        uint64_t prp2,
                                        uint64_t n_blocks, // 512B per block
                                        uint64_t starting_lba,
                                        uint8_t opcode,
                                        uint16_t *cid) {
        nvm_cmd_t cmd;
        // assert(nr_byte % nvme_page_size == 0);
        *cid = get_cid(&(qp->sq));
        nvm_cmd_header(&cmd, *cid, opcode, qp->nvmNamespace);
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
        sq_enqueue(&qp->sq, &cmd); 
        // printf("Queue(%p):cid: %d, sq_pos: %d, queue: %d, prp_list: %lx, ioaddr: %lx, nr_bytesa %ld\n", 
        //         this, *cid, *sq_pos, queue, prp2, prp1, nr_byte);
    }

    __forceinline__ __device__ void poll(
                                        QueuePair* qp,
                                        uint16_t cid) {
        uint32_t cq_pos = cq_poll(&qp->cq, cid);
    
        cq_dequeue(&qp->cq, cq_pos, &qp->sq);
        put_cid(&qp->sq, cid);
        // geminifs_info("ctrl[%p]: queue %d, cid %d, sq-pos %d done!", ctrl, queue, cid, sq_pos);
    }
};

#endif