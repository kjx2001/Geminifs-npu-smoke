#ifndef __BLOCK_IO_ENGINE_CUH__
#define __BLOCK_IO_ENGINE_CUH__

/**
 * block_io_engine.cuh -- Host-side IO engine for the io_engine layer.
 *
 * Canonical home: io_engine/include/block_io_engine.cuh
 * Legacy origin:  scattered across filesystems/ext4/libgeminifs/nvme_controller.cu
 *                 (init_queue_acquire_helper_kernel, init_nvme_file_kernel,
 *                  g_read_kernel, g_write_kernel)
 *
 * Responsibilities:
 *   - Own the GPU-resident NvmeQueueScheduler (one per block device)
 *   - Create/destroy BlockIoChannel objects on GPU by binding device_manager's
 *     BlockAddressTranslator + QueuePair array with this engine's scheduler
 *   - Provide host-side kernel launchers for one-shot read/write
 *
 * Layer placement:
 *   Entirely in io_engine. Consumes resources from device_manager:
 *     - BlockAddressTranslator*  (per-file translator, GPU-resident)
 *     - QueuePair*               (shared queue array from Controller)
 *   The lifetime of those resources is device_manager's responsibility.
 *
 * Dependencies:
 *   block_io_channel.cuh        -- BlockIoChannel (GPU)
 *   nvme_queue_scheduler.cuh    -- NvmeQueueScheduler (GPU)
 *   block_address_translator.cuh -- BlockAddressTranslator (device_manager)
 *   ctrl.h                      -- QueuePair
 */

#include <memory>
#include <cstdint>
#include <cstddef>

#include "ctrl.h"
#include "block_address_translator.cuh"
#include "nvme_queue_scheduler.cuh"
#include "block_io_channel.cuh"

/**
 * BlockIoEngine -- host-side IO engine bound to one block device's queue array.
 *
 * One instance per BlockDeviceManager. Owns the scheduler on GPU and vends
 * per-file BlockIoChannel objects.
 *
 * Thread safety: the scheduler is immutable after construction. Channel
 * create/destroy are independent CUDA allocations; callers serialise their
 * own channel lifecycle.
 */
class BlockIoEngine {
public:
    /**
     * Construct and GPU-initialise an NvmeQueueScheduler for num_queues.
     * Throws std::runtime_error on CUDA failure.
     */
    explicit BlockIoEngine(int num_queues);

    ~BlockIoEngine();

    BlockIoEngine(const BlockIoEngine&)            = delete;
    BlockIoEngine& operator=(const BlockIoEngine&) = delete;

    /**
     * Create a per-file IO channel on GPU.
     *
     * @param d_translator       GPU-resident translator (owned by device_manager)
     * @param d_qps              GPU-resident QueuePair array (from Controller)
     * @param nvme_page_size     NVMe page granularity (bytes)
     * @param hqps_block_size_log  log2 of hardware block size (for LBA calc)
     * @return BlockIoChannel* on GPU, or nullptr on failure. Caller owns
     *         the returned pointer and must release it via destroy_channel().
     */
    BlockIoChannel* create_channel(BlockAddressTranslator* d_translator,
                                   QueuePair*              d_qps,
                                   size_t                  nvme_page_size,
                                   int                     hqps_block_size_log);

    /**
     * Destroy a channel previously returned by create_channel().
     */
    void destroy_channel(BlockIoChannel* d_channel);

    /**
     * One-shot host-side kernel launchers. Use these when a host caller
     * wants a synchronous read/write without writing its own kernel.
     *
     * For inline use from a user kernel, call channel->read_in() / write_out()
     * directly instead.
     */
    void launch_read(BlockIoChannel* d_channel,
                     uint64_t prp1, uint64_t prp2,
                     size_t file_offset, size_t nbytes);

    void launch_write(BlockIoChannel* d_channel,
                      uint64_t prp1, uint64_t prp2,
                      size_t file_offset, size_t nbytes);

    NvmeQueueScheduler* get_device_scheduler() const { return d_scheduler_; }
    int                 get_num_queues()       const { return num_queues_; }

private:
    NvmeQueueScheduler* d_scheduler_;  ///< GPU-resident scheduler (owned)
    int                 num_queues_;
};

using BlockIoEnginePtr = std::shared_ptr<BlockIoEngine>;

// --- GPU init kernels (declarations; defined in block_io_engine.cu) ---

/**
 * Initialise NvmeQueueScheduler on GPU via placement new.
 */
__global__ void init_nvme_queue_scheduler_kernel(NvmeQueueScheduler* d_scheduler,
                                                  int num_queues);

/**
 * Initialise BlockIoChannel on GPU via placement new, then populate the
 * metadata fields (nvme_page_size, hqps_block_size_log).
 */
__global__ void init_block_io_channel_kernel(BlockIoChannel*         d_channel,
                                              BlockAddressTranslator* d_translator,
                                              QueuePair*              d_qps,
                                              NvmeQueueScheduler*     d_scheduler,
                                              size_t                  nvme_page_size,
                                              int                     hqps_block_size_log);

// --- One-shot GPU kernels for host-launched IO ---

__global__ void block_io_channel_read_kernel(BlockIoChannel* d_channel,
                                              uint64_t prp1, uint64_t prp2,
                                              size_t file_offset, size_t nbytes);

__global__ void block_io_channel_write_kernel(BlockIoChannel* d_channel,
                                               uint64_t prp1, uint64_t prp2,
                                               size_t file_offset, size_t nbytes);

#endif // __BLOCK_IO_ENGINE_CUH__