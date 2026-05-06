#ifndef __TUTTI_MEMORY_MEMORY_SUBSYSTEM_H__
#define __TUTTI_MEMORY_MEMORY_SUBSYSTEM_H__

/**
 * memory_subsystem.h -- the IMemorySubsystem interface.
 *
 * Layer: Memory Layer (independent, parallel to Device Manager per Roadmap §3).
 *
 * Role:
 *   - Single source of truth for "the runtime knows about this piece of
 *     memory". Every region the runtime handles in any data-plane path
 *     (IO Engine, COOP channels, Device Manager queue rings, ...)
 *     enters through one of the register_*/allocate_* methods here and
 *     leaves through unregister/free.
 *   - Decouples WHERE memory came from (app malloc, vLLM slab, IPC,
 *     shm, fd-mapped, runtime-allocated) from HOW backends use it
 *     (DMA, RDMA, IPC, peer-to-peer). The subsystem fills in the
 *     RegistrationMetadata that backends consume.
 *
 * Allocation vs registration:
 *   - allocate_*: convenience for callers that don't already own a
 *     buffer. The runtime allocates with the appropriate primitive,
 *     registers the result, returns a MemoryRegion handle. free()
 *     releases both registration AND the underlying allocation.
 *   - register_*: the caller already has a buffer. The runtime records
 *     metadata and prepares backend-side mappings on demand.
 *     unregister() drops the runtime's view; the underlying buffer is
 *     the caller's to free.
 *
 * Threading:
 *   - Implementations MUST be thread-safe. Multiple data-plane threads
 *     concurrently look up MemoryRegions during IO submission.
 *
 * Lifetime:
 *   - The MemoryRegion* returned by allocate_*/register_* is valid
 *     until the matching free()/unregister(). The Memory Layer keeps
 *     internal state alive across that interval; callers may freely
 *     hand the pointer to other layers.
 *   - After free()/unregister(), the pointer is invalid. Callers
 *     MUST drop their copies.
 *
 * What's NOT in v0.1:
 *   - No reference counting. Callers cooperate to avoid double-free.
 *   - No region splitting / sub-allocation; one register call yields
 *     one MemoryRegion. Sub-allocation is an upper-layer concern.
 *   - No cross-process registry; single-process today. CUDA IPC is
 *     supported per-process via register_external(CUDA_IPC).
 */

#include <cstdint>
#include <cstddef>

#include "memory_kind.h"
#include "memory_region.h"

namespace tutti {

/**
 * Look-up key for `lookup()`. The application typically only knows the
 * pointer it gave the runtime; the subsystem walks its index by either
 * the host or device pointer.
 *
 * Containing a single discriminated key keeps `lookup` from needing
 * two overloads with the same signature shape.
 */
struct MemoryLookupKey {
    enum class By : uint32_t { HOST_PTR, DEVICE_PTR, REGION_ID };
    By       by;
    union {
        const void* ptr;        // for HOST_PTR / DEVICE_PTR
        uint64_t    region_id;  // for REGION_ID
    };
};

class IMemorySubsystem {
public:
    virtual ~IMemorySubsystem() = default;

    // ------------------------------------------------------------------
    // Allocation (runtime-driven)
    // ------------------------------------------------------------------

    /// Allocate `size` bytes of host memory of `kind` (HOST or PINNED_HOST).
    /// Returns nullptr on allocation failure.
    virtual MemoryRegion* allocate_host(std::size_t size, MemoryKind kind) = 0;

    /// Allocate `size` bytes of CUDA device memory on `device_id` (DEVICE) or
    /// CUDA managed memory (MANAGED, ignore device_id beyond placement hint).
    virtual MemoryRegion* allocate_device(std::size_t size,
                                           MemoryKind  kind,
                                           int         device_id) = 0;

    /// Release a region the runtime allocated. After return the pointer
    /// is invalid. Must NOT be called on regions that came from
    /// register_* (use unregister there).
    virtual void free(MemoryRegion* region) = 0;

    // ------------------------------------------------------------------
    // Registration (caller-allocated)
    // ------------------------------------------------------------------

    /// Register a host buffer the caller already allocated. The runtime
    /// can pin / IOMMU-map it for backends that need it (e.g. NVMe DMA).
    virtual MemoryRegion* register_host(void*       host_ptr,
                                         std::size_t size) = 0;

    /// Register a CUDA device buffer the caller already allocated.
    virtual MemoryRegion* register_device(void*       device_ptr,
                                           std::size_t size,
                                           int         device_id) = 0;

    /// Register memory whose origin is outside both the runtime and the
    /// CUDA allocator. The ExternalMemorySpec selects the variant
    /// (APP_MANAGED slab, CUDA IPC import, host shm, fd-backed mmap).
    /// At least one of (host_ptr, device_ptr) must be non-null.
    virtual MemoryRegion* register_external(void*                       host_ptr,
                                             void*                       device_ptr,
                                             std::size_t                 size,
                                             const ExternalMemorySpec&   spec) = 0;

    /// Drop the runtime's registration metadata for `region`. The
    /// underlying buffer remains intact and is the caller's to free.
    /// After return, the pointer is invalid.
    virtual void unregister(MemoryRegion* region) = 0;

    // ------------------------------------------------------------------
    // Backend-facing prepare (lazy, on first use)
    // ------------------------------------------------------------------

    /// Ensure RegistrationMetadata has the NVMe DMA mapping (per-page
    /// IO addresses + page size) populated for `region`. No-op if
    /// already prepared. Returns false if the underlying memory cannot
    /// be DMA-mapped (e.g. plain non-pinned HOST without IOMMU support).
    virtual bool prepare_nvme_dma(MemoryRegion* region) = 0;

    /// Same shape, RDMA flavour. Returns false if no RDMA backend is
    /// active or the memory cannot be registered with an HCA.
    virtual bool prepare_rdma_mr(MemoryRegion* region) = 0;

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    /// Find the MemoryRegion that contains `key`. Returns nullptr if
    /// not registered.
    virtual MemoryRegion* lookup(const MemoryLookupKey& key) const = 0;
};

} // namespace tutti

#endif // __TUTTI_MEMORY_MEMORY_SUBSYSTEM_H__
