#ifndef __TUTTI_RUNTIME_STORAGE_TARGET_H__
#define __TUTTI_RUNTIME_STORAGE_TARGET_H__

/**
 * storage_target.h -- the addressable thing a BatchRequest reads from / writes to.
 *
 * Layer: Core Runtime (Roadmap.md §3 object model).
 *
 * Role:
 *   - A StorageTarget is the runtime-level abstraction over "where the
 *     data lives": a file on a local filesystem, a raw block range on
 *     an NVMe namespace, an RDMA remote region, a future object-store
 *     key. Backends translate StorageTarget into their own concrete
 *     locator at IO submission time.
 *   - Roadmap.md §3 explicitly says "File is treated as one storage
 *     object form, not the only storage abstraction." StorageTarget is
 *     the type that captures that.
 *
 * Why a tagged union instead of a base class:
 *   - The set of target shapes is small and stable; a tagged union is
 *     easier to construct (no allocation), pass through C-shaped APIs,
 *     and serialise. New shapes append.
 *   - Backends only see shapes they understand: a local_nvme backend
 *     handles FILE / BLOCK_RANGE; an RDMA backend handles RDMA_REGION;
 *     an object-store backend handles OBJECT_KEY. Mismatches are
 *     reported at SPI entry, not at runtime dispatch time.
 *
 * Lifetime:
 *   - StorageTarget is a value type. Caller-owned. The runtime makes
 *     copies as needed; nothing inside owns heap memory.
 *   - The string fields use std::string for ergonomics; if a future
 *     backend needs to embed targets in device-side structures,
 *     introduce a parallel POD shape rather than weakening this one.
 */

#include <cstdint>
#include <string>

namespace tutti {

struct Device;  // device.h

/**
 * Tag for which variant the StorageTarget carries.
 * Numeric values are stable; new shapes append.
 */
enum class StorageTargetKind : uint32_t {
    FILE        = 0,   // local file on a backend-mounted filesystem
    BLOCK_RANGE = 1,   // raw byte range on a namespace (no filesystem)
    RDMA_REGION = 2,   // remote virtual-address range reachable via an RDMA QP
    OBJECT_KEY  = 3,   // object-store key (for future S3-shape backends)
};

/**
 * File on a backend-mounted filesystem. The path is relative to the
 * backend's view of the mount (e.g. "/mnt/gpu0/snvm_nvme0n1/foo" for
 * local_nvme via NVMeService).
 */
struct StorageFileTarget {
    std::string path;
    uint64_t    file_id;       // backend-assigned, stable across opens; 0 if unknown
};

/**
 * Raw block range on a namespace. Used when the backend bypasses the
 * filesystem (e.g. zero-copy raw NVMe namespace IO).
 */
struct StorageBlockRangeTarget {
    uint32_t namespace_id;     // NVMe namespace number, or backend-defined id
    uint64_t lba_start;        // logical block address (units of block_size)
    uint64_t lba_count;        // number of LBAs
    uint32_t block_size;       // bytes per LBA
};

/**
 * Remote RDMA memory region. Resolved at IO time by the RDMA backend
 * to a concrete (qp, rkey, vaddr) tuple.
 */
struct StorageRdmaRegionTarget {
    uint64_t remote_va;        // remote virtual base address
    uint64_t length;           // bytes
    uint32_t rkey;             // remote memory key
    uint32_t qp_handle;        // backend-private handle to the QP
};

/**
 * Generic object-store key. Reserved for future backends.
 */
struct StorageObjectKeyTarget {
    std::string bucket;
    std::string key;
};

/**
 * Tagged union of the variants. Exactly one variant applies, selected
 * by `kind`. `device_id` ties the target to a runtime Device.
 *
 * Reading the wrong variant is undefined; consumers MUST switch on
 * `kind` before accessing any of the *_target fields.
 */
struct StorageTarget {
    int32_t            device_id;
    StorageTargetKind  kind;

    // C++ `std::string` cannot live in a C-style union; use a struct
    // with all variants populated optionally. The size cost (~120 B)
    // is negligible compared to one IO submission.
    StorageFileTarget        file_target;
    StorageBlockRangeTarget  block_range_target;
    StorageRdmaRegionTarget  rdma_region_target;
    StorageObjectKeyTarget   object_key_target;
};

} // namespace tutti

#endif // __TUTTI_RUNTIME_STORAGE_TARGET_H__
