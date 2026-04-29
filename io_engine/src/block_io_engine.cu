#include "block_io_engine.cuh"
#include "geminifs_helper.h"

#include <cuda_runtime.h>
#include <stdexcept>

// ---------------------------------------------------------------------------
// GPU init kernels
// ---------------------------------------------------------------------------

__global__ void init_nvme_queue_scheduler_kernel(NvmeQueueScheduler* d_scheduler,
                                                  int num_queues) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (d_scheduler) NvmeQueueScheduler(num_queues);
    }
}

__global__ void init_block_io_channel_kernel(BlockIoChannel*         d_channel,
                                              BlockAddressTranslator* d_translator,
                                              QueuePair*              d_qps,
                                              NvmeQueueScheduler*     d_scheduler,
                                              size_t                  nvme_page_size,
                                              int                     hqps_block_size_log) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        new (d_channel) BlockIoChannel(d_translator, d_qps, d_scheduler);
        d_channel->nvme_page_size      = nvme_page_size;
        d_channel->hqps_block_size_log = hqps_block_size_log;
    }
}

// ---------------------------------------------------------------------------
// One-shot IO kernels
// ---------------------------------------------------------------------------

__global__ void block_io_channel_read_kernel(BlockIoChannel* d_channel,
                                              uint64_t prp1, uint64_t prp2,
                                              size_t file_offset, size_t nbytes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_channel->read_in(prp1, prp2, file_offset, nbytes);
    }
}

__global__ void block_io_channel_write_kernel(BlockIoChannel* d_channel,
                                               uint64_t prp1, uint64_t prp2,
                                               size_t file_offset, size_t nbytes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_channel->write_out(prp1, prp2, file_offset, nbytes);
    }
}

// ---------------------------------------------------------------------------
// BlockIoEngine: constructor / destructor
// ---------------------------------------------------------------------------

BlockIoEngine::BlockIoEngine(int num_queues)
    : d_scheduler_(nullptr), num_queues_(num_queues)
{
    cudaError_t err = cudaMalloc(&d_scheduler_, sizeof(NvmeQueueScheduler));
    if (err != cudaSuccess) {
        geminifs_error("BlockIoEngine: Failed to allocate NvmeQueueScheduler on GPU: %s\n",
                       cudaGetErrorString(err));
        throw std::runtime_error("Failed to allocate NvmeQueueScheduler on GPU");
    }

    init_nvme_queue_scheduler_kernel<<<1, 1>>>(d_scheduler_, num_queues);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        geminifs_error("BlockIoEngine: scheduler init sync failed: %s\n",
                       cudaGetErrorString(err));
        cudaFree(d_scheduler_);
        d_scheduler_ = nullptr;
        throw std::runtime_error("Failed to initialise NvmeQueueScheduler on GPU");
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        geminifs_error("BlockIoEngine: scheduler init failed: %s\n",
                       cudaGetErrorString(err));
        cudaFree(d_scheduler_);
        d_scheduler_ = nullptr;
        throw std::runtime_error("Failed to initialise NvmeQueueScheduler on GPU");
    }

    geminifs_debug("BlockIoEngine: scheduler=%p, num_queues=%d\n", d_scheduler_, num_queues);
}

BlockIoEngine::~BlockIoEngine() {
    if (d_scheduler_ != nullptr) {
        cudaError_t err = cudaFree(d_scheduler_);
        if (err != cudaSuccess) {
            geminifs_error("~BlockIoEngine: Failed to free scheduler: %s\n",
                           cudaGetErrorString(err));
        }
        d_scheduler_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Channel create / destroy
// ---------------------------------------------------------------------------

BlockIoChannel* BlockIoEngine::create_channel(BlockAddressTranslator* d_translator,
                                               QueuePair*              d_qps,
                                               size_t                  nvme_page_size,
                                               int                     hqps_block_size_log) {
    if (d_translator == nullptr || d_qps == nullptr || d_scheduler_ == nullptr) {
        geminifs_error("create_channel: null input (translator=%p, qps=%p, scheduler=%p)\n",
                       d_translator, d_qps, d_scheduler_);
        return nullptr;
    }

    BlockIoChannel* d_channel = nullptr;
    cudaError_t err = cudaMalloc(&d_channel, sizeof(BlockIoChannel));
    if (err != cudaSuccess) {
        geminifs_error("create_channel: Failed to allocate channel on GPU: %s\n",
                       cudaGetErrorString(err));
        return nullptr;
    }

    init_block_io_channel_kernel<<<1, 1>>>(d_channel, d_translator, d_qps, d_scheduler_,
                                            nvme_page_size, hqps_block_size_log);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        geminifs_error("create_channel: channel init sync failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_channel);
        return nullptr;
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        geminifs_error("create_channel: channel init failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_channel);
        return nullptr;
    }

    geminifs_debug("create_channel: channel=%p translator=%p qps=%p scheduler=%p\n",
                   d_channel, d_translator, d_qps, d_scheduler_);
    return d_channel;
}

void BlockIoEngine::destroy_channel(BlockIoChannel* d_channel) {
    if (d_channel == nullptr) return;

    cudaError_t err = cudaFree(d_channel);
    if (err != cudaSuccess) {
        geminifs_error("destroy_channel: Failed to free channel %p: %s\n",
                       d_channel, cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// One-shot launchers
// ---------------------------------------------------------------------------

void BlockIoEngine::launch_read(BlockIoChannel* d_channel,
                                 uint64_t prp1, uint64_t prp2,
                                 size_t file_offset, size_t nbytes) {
    block_io_channel_read_kernel<<<1, 1>>>(d_channel, prp1, prp2, file_offset, nbytes);
}

void BlockIoEngine::launch_write(BlockIoChannel* d_channel,
                                  uint64_t prp1, uint64_t prp2,
                                  size_t file_offset, size_t nbytes) {
    block_io_channel_write_kernel<<<1, 1>>>(d_channel, prp1, prp2, file_offset, nbytes);
}
