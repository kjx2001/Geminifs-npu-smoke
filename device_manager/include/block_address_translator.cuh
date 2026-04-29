#ifndef __BLOCK_ADDRESS_TRANSLATOR_CUH__
#define __BLOCK_ADDRESS_TRANSLATOR_CUH__

/**
 * block_address_translator.cuh -- GPU-side block address translator.
 *
 * Canonical home: device_manager/include/block_address_translator.cuh
 *
 * Responsibilities:
 *   - Translate a virtual file address (vaddr_t) to a physical block offset
 *     using the extent map stored in geminiFS_hdr
 *   - Provide file size metadata query
 *
 * This class is the interface contract between device_manager and io_engine.
 * device_manager allocates and populates the extent metadata (via FIEMAP ioctl
 * on the host); io_engine calls translate() during IO to obtain the physical
 * offset for NVMe command submission.
 *
 * GPU-only: all methods are __device__. The object is initialised on the host
 * via a placement-new CUDA kernel in the device manager.
 *
 * Dependencies:
 *   geminifs.h  -- geminiFS_hdr, vaddr_t, nvme_ofst_t
 *   utils.cuh   -- geminifs_debug / geminifs_error macros
 */

#include "geminifs.h"
#include "utils.cuh"
#include <cassert>
#include <cstdint>

class BlockAddressTranslator {
private:
    struct geminiFS_hdr* hdr;  ///< File header with extent map (owned by device_manager)

public:
    __forceinline__ __device__
    BlockAddressTranslator(struct geminiFS_hdr* hdr_) : hdr(hdr_) {}

    /**
     * Translate a virtual file address to a physical block offset.
     *
     * Walks the FIEMAP extent table in the file header linearly.
     * Extent counts are small (typically < 16) for the tensor-sized files
     * this runtime targets, so linear scan is sufficient.
     *
     * @param va  Virtual file address (byte offset within the logical file)
     * @return    Physical block offset in bytes on the NVMe device
     */
    __forceinline__ __device__
    nvme_ofst_t translate(vaddr_t va) const {
        assert(hdr);
        uint64_t blk_id       = ((uint64_t)va) >> hdr->block_bit;
        uint64_t start_blk_id = 0;
        uint64_t end_blk_id   = 0;

        geminifs_debug(
            "BlockAddressTranslator::translate: va=0x%llx blk_id=%llu block_bit=%u extent_count=%llu\n",
            (unsigned long long)va,
            (unsigned long long)blk_id,
            (unsigned)hdr->block_bit,
            (unsigned long long)hdr->extent_count);

        for (size_t i = 0; i < hdr->extent_count; ++i) {
            uint64_t add = ((uint64_t)hdr->extents[i].fe_length) >> hdr->block_bit;
            end_blk_id += add;

            if (blk_id >= start_blk_id && blk_id < end_blk_id) {
                uint64_t offset_blk  = (uint64_t)(blk_id - start_blk_id);
                uint64_t byte_offset = offset_blk << hdr->block_bit;
                return (nvme_ofst_t)((uint64_t)hdr->extents[i].fe_physical + byte_offset);
            }
            start_blk_id = end_blk_id;
        }

        geminifs_error(
            "BlockAddressTranslator::translate: va=0x%llx not covered by any extent\n",
            (unsigned long long)va);
        assert(false);
        return 0;
    }

    /**
     * Get the logical file size in bytes.
     */
    __forceinline__ __device__
    uint64_t get_file_size() const {
        return hdr ? hdr->virtual_space_size : 0;
    }
};

#endif // __BLOCK_ADDRESS_TRANSLATOR_CUH__
