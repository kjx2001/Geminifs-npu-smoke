#ifndef __TUTTI_MEMORY_HOST_DEVICE_MEMORY_SUBSYSTEM_H__
#define __TUTTI_MEMORY_HOST_DEVICE_MEMORY_SUBSYSTEM_H__

/**
 * host_device_memory_subsystem.h -- the v0.1 IMemorySubsystem
 * implementation.
 *
 * Layer: Memory.
 *
 * R4.5 changes (vs the original R3 cut):
 *   - Constructor no longer takes a single nvm_ctrl_t*.  The
 *     subsystem now serves multiple devices; DMA mappings are
 *     tracked per (region_id, Device*).  Coordinator is expected
 *     to wire each Device in via register_tensor() (whose
 *     spec.target_devices drives mapping creation).
 *   - prepare_nvme_dma() / prepare_rdma_mr() are gone.  DMA
 *     mapping happens implicitly during register_tensor().
 *   - set_descriptor_format() / descriptor_format() implemented.
 *     PRP/SGL builder lives in v0.1 as a Unimplemented stub
 *     (descriptor_slice returns false); the real builder lands
 *     in R7 (PRP) / R8 (SGL).
 *
 * Coverage in v0.1
 *   - allocate_host(HOST | PINNED_HOST)        => malloc / cudaMallocHost
 *   - allocate_device(DEVICE)                  => cudaMalloc on a target GPU
 *   - allocate_device(MANAGED)                 => cudaMallocManaged
 *   - free                                     => matching free / cudaFree*
 *   - register_host                            => no-copy view, MR id assigned
 *   - register_device                          => same, on a GPU buffer
 *   - register_external(APP_MANAGED)           => same, source recorded
 *   - register_external(CUDA_IPC | HOST_SHM | HOST_FD_MAP) => unimplemented;
 *                                                returns nullptr.
 *   - register_tensor                          => idempotent; finds (or
 *                                                creates) the region for
 *                                                spec.ptr and DMA-maps it
 *                                                to each target device
 *                                                via libnvm.
 *   - descriptor_slice                         => Unimplemented stub
 *                                                (returns false until R7).
 *   - lookup                                   => walks an internal table.
 *
 * Threading
 *   - All public methods are protected by a single std::mutex; v0.1
 *     is deliberately simple.  When tracing shows real contention
 *     we'll swap in a striped table.
 *
 * Lifetime
 *   - Caller-allocated regions: subsystem records metadata only.
 *     unregister() releases metadata + any nvm_dma_t handles
 *     created on its behalf.  The buffer stays the caller's.
 *   - Subsystem-allocated regions: free() releases metadata,
 *     nvm_dma_t handles, AND the underlying buffer.
 */

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "memory_kind.h"
#include "memory_region.h"
#include "memory_subsystem.h"

#include <nvm_types.h>   // nvm_ctrl_t, nvm_dma_t

namespace tutti {

struct Device;   // runtime/include/device.h -- forward-decl mirrored
                 // here so this header doesn't need to pull in the
                 // full Device definition.

class HostDeviceMemorySubsystem : public IMemorySubsystem {
public:
    HostDeviceMemorySubsystem();
    ~HostDeviceMemorySubsystem() override;

    HostDeviceMemorySubsystem(const HostDeviceMemorySubsystem&)            = delete;
    HostDeviceMemorySubsystem& operator=(const HostDeviceMemorySubsystem&) = delete;

    // --- IMemorySubsystem -----------------------------------------

    void              set_descriptor_format(DescriptorFormat fmt) override;
    DescriptorFormat  descriptor_format() const override;

    MemoryRegion* allocate_host(std::size_t size, MemoryKind kind) override;
    MemoryRegion* allocate_device(std::size_t size,
                                   MemoryKind  kind,
                                   int         device_id) override;
    void          free(MemoryRegion* region) override;

    MemoryRegion* register_host(void* host_ptr, std::size_t size) override;
    MemoryRegion* register_device(void*       device_ptr,
                                   std::size_t size,
                                   int         device_id) override;
    MemoryRegion* register_external(void*                     host_ptr,
                                     void*                     device_ptr,
                                     std::size_t               size,
                                     const ExternalMemorySpec& spec) override;
    void          unregister(MemoryRegion* region) override;

    MemoryRegion* register_tensor(const TensorRegistrationSpec& spec) override;

    bool descriptor_slice(MemoryRegion*       region,
                          const Device*       device,
                          uint64_t            byte_offset,
                          uint64_t            byte_length,
                          AddressDescriptor*  out,
                          std::size_t*        inout_count) override;

    MemoryRegion* lookup(const MemoryLookupKey& key) const override;

    // --- Test-only knobs ------------------------------------------

    /// Number of regions currently tracked.
    std::size_t region_count() const;

    /// Test-friendly query: does `region` have a DMA mapping for
    /// `device`?  Returns the per-page IO address count via
    /// out_count, the page size via out_page_size, and the first
    /// IO address via out_first_ioaddr.  Returns false if no
    /// mapping exists.  Smokes use this to validate that
    /// register_tensor wired DMA mapping correctly without leaking
    /// nvm_dma_t* details.
    bool query_nvme_mapping(const MemoryRegion* region,
                            const Device*       device,
                            std::size_t*        out_count,
                            std::size_t*        out_page_size,
                            uint64_t*           out_first_ioaddr) const;

private:
    struct Slot {
        std::unique_ptr<MemoryRegion> region;
        bool        owns_host_alloc   = false;
        bool        owns_device_alloc = false;
        // (Device* -> nvm_dma_t*) DMA mappings created lazily by
        // register_tensor().  unique_ptr-style: subsystem owns the
        // nvm_dma_t and unmaps on erase.
        std::unordered_map<const Device*, nvm_dma_t*> dma_per_device;
    };

    MemoryRegion* register_into_table(std::unique_ptr<MemoryRegion> r,
                                       bool owns_host_alloc,
                                       bool owns_device_alloc);
    void          erase_locked(uint64_t region_id);

    /// Find the slot that owns `ptr` (host or device pointer).
    /// Returns nullptr if not registered.  Caller must hold mtx_.
    Slot*         slot_by_ptr_locked(const void* ptr);

    /// Map `region` to `device` if not already mapped.  Caller must
    /// hold mtx_.  Returns true on success or already-mapped.
    bool          ensure_mapping_locked(Slot& slot, const Device* device);

    DescriptorFormat                       fmt_ = DescriptorFormat::UNSET;
    mutable std::mutex                     mtx_;
    uint64_t                               next_region_id_ = 1;
    std::unordered_map<uint64_t, Slot>     regions_;
};

} // namespace tutti

#endif // __TUTTI_MEMORY_HOST_DEVICE_MEMORY_SUBSYSTEM_H__
