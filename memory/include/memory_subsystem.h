#ifndef __TUTTI_MEMORY_MEMORY_SUBSYSTEM_H__
#define __TUTTI_MEMORY_MEMORY_SUBSYSTEM_H__

/**
 * memory_subsystem.h -- the IMemorySubsystem interface.
 *
 * Layer: Memory.  See doc/refactor/LegacyDecomposition.md §2 / §3.5
 * for the full layer cake.  This layer is the single source of truth
 * for "the runtime knows about this piece of memory" and for
 * translating tensors into NVMe-readable address descriptors
 * (PRP today, SGL when supported cluster-wide).
 *
 * Hard invariants (mirror of LegacyDecomposition.md §2.2):
 *   - The descriptor format (PRP vs SGL) is **cluster-wide**, set
 *     once at coordinator boot via set_descriptor_format().  If any
 *     attached controller lacks SGL support, the entire runtime
 *     falls back to PRP.  Memory therefore stores at most one
 *     descriptor format per registration, never a mix.
 *   - Memory is per-process, never per-controller.  A single
 *     IMemorySubsystem instance services all attached devices; DMA
 *     mappings are tracked internally as a (region, device) table.
 *   - Memory does not own queue pairs and does not submit IO.
 *     Producing a PRP/SGL list is a memory job; submitting it is
 *     nvme_storage's.
 *
 * Allocation vs registration:
 *   - allocate_X (X = host / device): convenience for callers that
 *     don't already own a buffer.  The runtime allocates with the
 *     appropriate primitive, records a MemoryRegion, returns it.
 *     free() releases both the registration and the underlying
 *     allocation.
 *   - register_X (X = host / device / external): the caller already
 *     has a buffer.  The runtime records metadata; the underlying
 *     buffer stays the caller's responsibility.
 *   - register_tensor: the high-level entry-point used by upper
 *     layers (block_storage, io_engine).  Spec.target_devices
 *     drives DMA mapping for every device the buffer will be IO'd
 *     against.  May be called more than once on the same buffer
 *     (idempotent on (region, device) pairs).
 *
 * Threading:
 *   - Implementations MUST be thread-safe.  Multiple data-plane
 *     threads concurrently look up MemoryRegions during IO submission.
 *
 * Lifetime:
 *   - The MemoryRegion pointer returned by allocate_X / register_X
 *     is valid until the matching free() / unregister().
 *   - After free() / unregister(), the pointer is invalid.  Callers
 *     MUST drop their copies.
 *
 * Out of scope for v0.1:
 *   - No reference counting on regions.
 *   - No region splitting / sub-allocation.
 *   - register_external CUDA_IPC / HOST_SHM / HOST_FD_MAP variants
 *     are stubbed (return nullptr).
 *   - descriptor_slice() ships an Unimplemented stub; the real PRP
 *     builder lands in R7, the SGL builder in R8.
 */

#include <cstdint>
#include <cstddef>
#include <vector>

#include "memory_kind.h"
#include "memory_region.h"

namespace tutti {

struct Device;   // runtime/include/device.h

// ---------------------------------------------------------------------------
// Cluster-wide descriptor format.
//
// Coordinator probes every controller it ever attaches and ANDs the
// capability bits.  If every ctrl supports SGL, the runtime can use
// SGL; otherwise it falls back to PRP for the whole run.
// ---------------------------------------------------------------------------
enum class DescriptorFormat : uint8_t {
    UNSET = 0,   // before set_descriptor_format() is called
    PRP   = 1,
    SGL   = 2,
};

// ---------------------------------------------------------------------------
// One row of an NVMe-readable description of part of a buffer.
//
// PRP path uses prp1 (and optional prp2 for cross-page IO); SGL path
// reinterprets the same struct.  The slice walker emits as many of
// these as needed to cover the requested byte range.
// ---------------------------------------------------------------------------
struct AddressDescriptor {
    uint64_t prp1;          // valid iff format == PRP
    uint64_t prp2;          // valid iff format == PRP, second page in pair
    uint64_t data_length;   // bytes covered by this descriptor row
    // SGL fields collapsed for now; expanded in R7.
};

// ---------------------------------------------------------------------------
// Caller intent at register_tensor() time.
//
// `target_devices` lists every device the buffer must be IO-mappable
// to.  Memory will lazily DMA-map (via libnvm or its successor) on
// first use, but registers the *intent* now so descriptor_slice()
// can answer questions about page_size, etc.
// ---------------------------------------------------------------------------
struct TensorRegistrationSpec {
    void*                       ptr;           // host or device pointer
    std::size_t                 size;          // bytes
    std::vector<std::size_t>    shape;         // optional tensor shape
    std::vector<const Device*>  target_devices; // see invariant above
};

// ---------------------------------------------------------------------------
// Look-up key for `lookup()`.
// ---------------------------------------------------------------------------
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
    // Cluster-wide capability handshake (called by coordinator at boot)
    // ------------------------------------------------------------------

    /// Freeze the descriptor format.  Coordinator probes every ctrl
    /// it attaches and calls this once with the AND of their caps.
    /// Any later register_tensor / descriptor_slice produces this
    /// format.  Calling twice with different values is a logic
    /// error in v0.1 (implementations may assert).
    virtual void set_descriptor_format(DescriptorFormat fmt) = 0;

    /// Inspect the currently-frozen format.
    virtual DescriptorFormat descriptor_format() const = 0;

    // ------------------------------------------------------------------
    // Allocation (runtime-driven)
    // ------------------------------------------------------------------

    /// Allocate `size` bytes of host memory of `kind` (HOST or PINNED_HOST).
    virtual MemoryRegion* allocate_host(std::size_t size, MemoryKind kind) = 0;

    /// Allocate `size` bytes of CUDA device memory on `device_id`
    /// (DEVICE) or CUDA managed memory (MANAGED).
    virtual MemoryRegion* allocate_device(std::size_t size,
                                           MemoryKind  kind,
                                           int         device_id) = 0;

    /// Release a region the runtime allocated.
    virtual void free(MemoryRegion* region) = 0;

    // ------------------------------------------------------------------
    // Low-level registration (caller-allocated, no DMA mapping)
    //
    // These are the thin entry-points used internally by
    // register_tensor() and by tests; upper layers should normally
    // call register_tensor() with target_devices populated.
    // ------------------------------------------------------------------

    virtual MemoryRegion* register_host(void*       host_ptr,
                                         std::size_t size) = 0;
    virtual MemoryRegion* register_device(void*       device_ptr,
                                           std::size_t size,
                                           int         device_id) = 0;

    /// External-source (vLLM slab, CUDA IPC import, ...).
    virtual MemoryRegion* register_external(void*                       host_ptr,
                                             void*                       device_ptr,
                                             std::size_t                 size,
                                             const ExternalMemorySpec&   spec) = 0;

    /// Drop the runtime's metadata for `region`.
    virtual void unregister(MemoryRegion* region) = 0;

    // ------------------------------------------------------------------
    // High-level registration
    // ------------------------------------------------------------------

    /// Register a buffer that will be DMA'd to one or more NVMe
    /// devices.  Side effect: every device in spec.target_devices
    /// gets a DMA mapping created for this buffer (if it doesn't
    /// already have one).  Calling repeatedly with the same ptr
    /// is fine; existing mappings are reused.
    ///
    /// Returns a stable MemoryRegion* tied to spec.ptr.  If spec.ptr
    /// was not previously registered, an internal register_host /
    /// register_device is performed first, classifying by whether
    /// the pointer maps to host or device memory.
    virtual MemoryRegion* register_tensor(const TensorRegistrationSpec& spec) = 0;

    // ------------------------------------------------------------------
    // Address descriptor extraction (consumed by io_engine kernel +
    // nvme_storage host-side IO).
    // ------------------------------------------------------------------

    /// Walk `region`'s DMA mapping for `device` and emit address
    /// descriptors covering byte_range [byte_offset, byte_offset + byte_length).
    ///
    /// On entry *inout_count tells the maximum number of descriptors
    /// the caller can accept; on success *inout_count is set to the
    /// number actually emitted.  Returns false if the format hasn't
    /// been set, the region has no mapping for `device`, or the
    /// caller's buffer is too small.
    ///
    /// v0.1: returns false (Unimplemented).  R7 lands the PRP
    /// builder; R8 lands the SGL fallback.
    virtual bool descriptor_slice(MemoryRegion*       region,
                                   const Device*       device,
                                   uint64_t            byte_offset,
                                   uint64_t            byte_length,
                                   AddressDescriptor*  out,
                                   std::size_t*        inout_count) = 0;

    // ------------------------------------------------------------------
    // Query
    // ------------------------------------------------------------------

    /// Find the MemoryRegion that contains `key`.
    virtual MemoryRegion* lookup(const MemoryLookupKey& key) const = 0;
};

} // namespace tutti

#endif // __TUTTI_MEMORY_MEMORY_SUBSYSTEM_H__
