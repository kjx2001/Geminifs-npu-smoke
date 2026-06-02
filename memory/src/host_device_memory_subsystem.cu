/**
 * host_device_memory_subsystem.cu -- v0.1 IMemorySubsystem
 * implementation backed by malloc / cudaMalloc + libnvm DMA mapping.
 *
 * See host_device_memory_subsystem.h for the design rationale.
 *
 * R4.5 changes vs the original R3 cut:
 *   - No nvm_ctrl_t* in the constructor; DMA mappings are tracked
 *     per-(region, Device*) and are created on demand by
 *     register_tensor() walking spec.target_devices.
 *   - prepare_nvme_dma() / prepare_rdma_mr() are gone.
 *   - set_descriptor_format() and descriptor_slice() are wired up
 *     (the latter as an Unimplemented stub).
 */

#include "host_device_memory_subsystem.h"
#include "cuda_helpers.cuh"

#include "../../device_manager/include/local_nvme_device.h"
#include "../../runtime/include/device.h"

#include <cuda_runtime.h>
#include <nvm_dma.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <cstdio>

namespace tutti {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Build a fresh, zeroed MemoryRegion ready to be filled in by the caller.
std::unique_ptr<MemoryRegion> make_region(uint64_t id) {
    auto r = std::make_unique<MemoryRegion>();
    std::memset(r.get(), 0, sizeof(MemoryRegion));
    r->region_id   = id;
    r->cuda_device = -1;
    return r;
}

// Pull the libnvm ctrl out of a tutti::Device that carries a
// LocalNvmeDevice payload.  Returns nullptr if the device is the
// wrong shape (which means caller is misusing target_devices).
nvm_ctrl_t* ctrl_for(const Device* dev) {
    if (dev == nullptr || dev->backend_private == nullptr) return nullptr;
    auto* lnd = static_cast<LocalNvmeDevice*>(dev->backend_private);
    return lnd->ctrl;
}

} // namespace

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

HostDeviceMemorySubsystem::HostDeviceMemorySubsystem() = default;

HostDeviceMemorySubsystem::~HostDeviceMemorySubsystem() {
    // Tear down whatever the user forgot to free.  We loop locally
    // because erase_locked() mutates regions_.
    std::lock_guard<std::mutex> lock(mtx_);
    while (!regions_.empty()) {
        auto it = regions_.begin();
        Slot& slot = it->second;

        // Unmap every per-device DMA handle.
        for (auto& kv : slot.dma_per_device) {
            if (kv.second != nullptr) nvm_dma_unmap(kv.second);
        }
        slot.dma_per_device.clear();

        if (slot.owns_host_alloc && slot.region->host_ptr != nullptr) {
            if (slot.region->kind == MemoryKind::PINNED_HOST) {
                cudaFreeHost(slot.region->host_ptr);
            } else {
                std::free(slot.region->host_ptr);
            }
        }
        if (slot.owns_device_alloc && slot.region->device_ptr != nullptr) {
            cudaFree(slot.region->device_ptr);
        }
        regions_.erase(it);
    }
}

// ---------------------------------------------------------------------------
// Descriptor format handshake
// ---------------------------------------------------------------------------

void HostDeviceMemorySubsystem::set_descriptor_format(DescriptorFormat fmt) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (fmt_ != DescriptorFormat::UNSET && fmt_ != fmt) {
        std::fprintf(stderr,
            "[memory] set_descriptor_format: changing %u -> %u (logic error)\n",
            (unsigned)fmt_, (unsigned)fmt);
        // Don't assert; in v0.1 just take the new value and warn.
    }
    fmt_ = fmt;
}

DescriptorFormat HostDeviceMemorySubsystem::descriptor_format() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return fmt_;
}

// ---------------------------------------------------------------------------
// register_into_table -- the common path that adds a new MemoryRegion
// ---------------------------------------------------------------------------

MemoryRegion* HostDeviceMemorySubsystem::register_into_table(
    std::unique_ptr<MemoryRegion> r,
    bool owns_host_alloc,
    bool owns_device_alloc)
{
    std::lock_guard<std::mutex> lock(mtx_);

    uint64_t id = r->region_id;
    Slot slot;
    slot.region            = std::move(r);
    slot.owns_host_alloc   = owns_host_alloc;
    slot.owns_device_alloc = owns_device_alloc;

    auto [it, ok] = regions_.emplace(id, std::move(slot));
    if (!ok) {
        std::fprintf(stderr, "[memory] register_into_table: id=%lu collision\n",
                     (unsigned long)id);
        return nullptr;
    }
    return it->second.region.get();
}

void HostDeviceMemorySubsystem::erase_locked(uint64_t region_id) {
    auto it = regions_.find(region_id);
    if (it == regions_.end()) return;

    Slot& slot = it->second;
    for (auto& kv : slot.dma_per_device) {
        if (kv.second != nullptr) nvm_dma_unmap(kv.second);
    }
    slot.dma_per_device.clear();

    if (slot.owns_host_alloc && slot.region->host_ptr != nullptr) {
        if (slot.region->kind == MemoryKind::PINNED_HOST) {
            cudaFreeHost(slot.region->host_ptr);
        } else {
            std::free(slot.region->host_ptr);
        }
    }
    if (slot.owns_device_alloc && slot.region->device_ptr != nullptr) {
        cudaFree(slot.region->device_ptr);
    }
    regions_.erase(it);
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

MemoryRegion* HostDeviceMemorySubsystem::allocate_host(
    std::size_t size, MemoryKind kind)
{
    if (size == 0) return nullptr;
    if (kind != MemoryKind::HOST && kind != MemoryKind::PINNED_HOST) {
        std::fprintf(stderr, "[memory] allocate_host: unsupported kind=%u\n",
                     (unsigned)kind);
        return nullptr;
    }

    void* ptr = nullptr;
    if (kind == MemoryKind::PINNED_HOST) {
        if (cudaMallocHost(&ptr, size) != cudaSuccess) return nullptr;
    } else {
        ptr = std::malloc(size);
        if (ptr == nullptr) return nullptr;
    }

    uint64_t id;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        id = next_region_id_++;
    }
    auto r = make_region(id);
    r->kind     = kind;
    r->host_ptr = ptr;
    r->size     = size;

    return register_into_table(std::move(r),
                                /*owns_host_alloc=*/true,
                                /*owns_device_alloc=*/false);
}

MemoryRegion* HostDeviceMemorySubsystem::allocate_device(
    std::size_t size, MemoryKind kind, int device_id)
{
    if (size == 0) return nullptr;

    if (kind == MemoryKind::DEVICE) {
        if (device_id < 0) return nullptr;
        if (cudaSetDevice(device_id) != cudaSuccess) return nullptr;
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, size) != cudaSuccess) return nullptr;

        uint64_t id;
        { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
        auto r = make_region(id);
        r->kind        = MemoryKind::DEVICE;
        r->cuda_device = device_id;
        r->device_ptr  = ptr;
        r->size        = size;
        return register_into_table(std::move(r), false, true);
    }

    if (kind == MemoryKind::MANAGED) {
        void* ptr = nullptr;
        if (cudaMallocManaged(&ptr, size) != cudaSuccess) return nullptr;

        uint64_t id;
        { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
        auto r = make_region(id);
        r->kind        = MemoryKind::MANAGED;
        r->cuda_device = device_id;
        r->host_ptr    = ptr;
        r->device_ptr  = ptr;
        r->size        = size;
        return register_into_table(std::move(r), false, true);
    }

    std::fprintf(stderr, "[memory] allocate_device: unsupported kind=%u\n",
                 (unsigned)kind);
    return nullptr;
}

void HostDeviceMemorySubsystem::free(MemoryRegion* region) {
    if (region == nullptr) return;
    std::lock_guard<std::mutex> lock(mtx_);
    erase_locked(region->region_id);
}

// ---------------------------------------------------------------------------
// Registration of caller-allocated buffers
// ---------------------------------------------------------------------------

MemoryRegion* HostDeviceMemorySubsystem::register_host(
    void* host_ptr, std::size_t size)
{
    if (host_ptr == nullptr || size == 0) return nullptr;
    uint64_t id;
    { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
    auto r = make_region(id);
    r->kind     = MemoryKind::HOST;
    r->host_ptr = host_ptr;
    r->size     = size;
    return register_into_table(std::move(r), false, false);
}

MemoryRegion* HostDeviceMemorySubsystem::register_device(
    void* device_ptr, std::size_t size, int device_id)
{
    if (device_ptr == nullptr || size == 0 || device_id < 0) return nullptr;
    uint64_t id;
    { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
    auto r = make_region(id);
    r->kind        = MemoryKind::DEVICE;
    r->cuda_device = device_id;
    r->device_ptr  = device_ptr;
    r->size        = size;
    return register_into_table(std::move(r), false, false);
}

MemoryRegion* HostDeviceMemorySubsystem::register_external(
    void* host_ptr, void* device_ptr,
    std::size_t size, const ExternalMemorySpec& spec)
{
    if (size == 0) return nullptr;
    if (host_ptr == nullptr && device_ptr == nullptr) return nullptr;

    if (spec.source != ExternalMemorySource::APP_MANAGED) {
        std::fprintf(stderr,
            "[memory] register_external: source=%u not implemented in v0.1\n",
            (unsigned)spec.source);
        return nullptr;
    }

    uint64_t id;
    { std::lock_guard<std::mutex> lock(mtx_); id = next_region_id_++; }
    auto r = make_region(id);
    r->kind       = MemoryKind::EXTERNAL;
    r->host_ptr   = host_ptr;
    r->device_ptr = device_ptr;
    r->size       = size;
    r->external   = spec;
    return register_into_table(std::move(r), false, false);
}

void HostDeviceMemorySubsystem::unregister(MemoryRegion* region) {
    if (region == nullptr) return;
    std::lock_guard<std::mutex> lock(mtx_);
    erase_locked(region->region_id);
}

// ---------------------------------------------------------------------------
// register_tensor -- high-level entry-point that drives DMA mapping
// ---------------------------------------------------------------------------

HostDeviceMemorySubsystem::Slot*
HostDeviceMemorySubsystem::slot_by_ptr_locked(const void* ptr) {
    if (ptr == nullptr) return nullptr;
    const auto* needle = static_cast<const uint8_t*>(ptr);
    for (auto& [id, slot] : regions_) {
        const auto* hbase = static_cast<const uint8_t*>(slot.region->host_ptr);
        if (hbase != nullptr &&
            needle >= hbase && needle < hbase + slot.region->size) {
            return &slot;
        }
        const auto* dbase = static_cast<const uint8_t*>(slot.region->device_ptr);
        if (dbase != nullptr &&
            needle >= dbase && needle < dbase + slot.region->size) {
            return &slot;
        }
    }
    return nullptr;
}

bool HostDeviceMemorySubsystem::ensure_mapping_locked(Slot& slot,
                                                       const Device* device)
{
    if (device == nullptr) return false;

    auto it = slot.dma_per_device.find(device);
    if (it != slot.dma_per_device.end() && it->second != nullptr) {
        return true;   // already mapped; idempotent.
    }

    nvm_ctrl_t* ctrl = ctrl_for(device);
    if (ctrl == nullptr) {
        std::fprintf(stderr,
            "[memory] ensure_mapping: device %d has no libnvm ctrl\n",
            device->device_id);
        return false;
    }

    nvm_dma_t* dma = nullptr;
    int rc = -1;
    if (slot.region->device_ptr != nullptr) {
        rc = nvm_dma_map_data_device(&dma, ctrl,
                                      slot.region->device_ptr,
                                      slot.region->size);
    } else if (slot.region->host_ptr != nullptr) {
        rc = nvm_dma_map_data_host(&dma, ctrl,
                                    slot.region->host_ptr,
                                    slot.region->size);
    } else {
        return false;
    }

    if (rc != 0 || dma == nullptr) {
        std::fprintf(stderr,
            "[memory] ensure_mapping: nvm_dma_map_data_* rc=%d dma=%p\n",
            rc, (void*)dma);
        return false;
    }

    slot.dma_per_device[device] = dma;
    return true;
}

MemoryRegion* HostDeviceMemorySubsystem::register_tensor(
    const TensorRegistrationSpec& spec)
{
    if (spec.ptr == nullptr || spec.size == 0) return nullptr;

    // Step 1: find or create a region for spec.ptr.
    MemoryRegion* region = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (Slot* existing = slot_by_ptr_locked(spec.ptr)) {
            region = existing->region.get();
        }
    }

    if (region == nullptr) {
        // Not registered yet -- classify by pointer attributes.
        cudaPointerAttributes attr{};
        cudaError_t cerr = cudaPointerGetAttributes(&attr, spec.ptr);
        if (cerr != cudaSuccess) {
            // Plain host pointer that CUDA doesn't know about.
            // Treat as host buffer.
            (void)cudaGetLastError();   // clear sticky error
            region = register_host(spec.ptr, spec.size);
        } else if (attr.type == cudaMemoryTypeDevice) {
            int dev = attr.device;
            region = register_device(spec.ptr, spec.size, dev);
        } else if (attr.type == cudaMemoryTypeHost ||
                   attr.type == cudaMemoryTypeManaged) {
            region = register_host(spec.ptr, spec.size);
        } else {
            region = register_host(spec.ptr, spec.size);
        }
        if (region == nullptr) return nullptr;
    }

    // Step 2: ensure DMA mapping for every target device.
    {
        std::lock_guard<std::mutex> lock(mtx_);
        Slot* slot = slot_by_ptr_locked(spec.ptr);
        if (slot == nullptr) return nullptr;
        for (const Device* dev : spec.target_devices) {
            if (!ensure_mapping_locked(*slot, dev)) {
                // Mapping failure leaves any previously-created
                // mappings in place; caller can retry or unregister.
                std::fprintf(stderr,
                    "[memory] register_tensor: mapping failed for device %d\n",
                    dev != nullptr ? dev->device_id : -1);
                return nullptr;
            }
        }
    }

    return region;
}

// ---------------------------------------------------------------------------
// descriptor_slice -- v0.1 stub.
// ---------------------------------------------------------------------------

bool HostDeviceMemorySubsystem::descriptor_slice(
    MemoryRegion*       /*region*/,
    const Device*       /*device*/,
    uint64_t            /*byte_offset*/,
    uint64_t            /*byte_length*/,
    AddressDescriptor*  /*out*/,
    std::size_t*        /*inout_count*/)
{
    // R7 lands the real PRP builder; R8 the SGL fallback.
    // Until then upper layers query the raw mapping via
    // query_nvme_mapping() (test-only) or through nvme_storage's
    // own NVMe-format helpers.
    return false;
}

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

MemoryRegion* HostDeviceMemorySubsystem::lookup(
    const MemoryLookupKey& key) const
{
    std::lock_guard<std::mutex> lock(mtx_);

    for (const auto& [id, slot] : regions_) {
        switch (key.by) {
        case MemoryLookupKey::By::REGION_ID:
            if (id == key.region_id) return slot.region.get();
            break;
        case MemoryLookupKey::By::HOST_PTR: {
            const auto* base = static_cast<const uint8_t*>(slot.region->host_ptr);
            if (base == nullptr) break;
            const auto* needle = static_cast<const uint8_t*>(key.ptr);
            if (needle >= base && needle < base + slot.region->size)
                return slot.region.get();
            break;
        }
        case MemoryLookupKey::By::DEVICE_PTR: {
            const auto* base = static_cast<const uint8_t*>(slot.region->device_ptr);
            if (base == nullptr) break;
            const auto* needle = static_cast<const uint8_t*>(key.ptr);
            if (needle >= base && needle < base + slot.region->size)
                return slot.region.get();
            break;
        }
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Test-only knobs
// ---------------------------------------------------------------------------

std::size_t HostDeviceMemorySubsystem::region_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return regions_.size();
}

bool HostDeviceMemorySubsystem::query_nvme_mapping(
    const MemoryRegion* region,
    const Device*       device,
    std::size_t*        out_count,
    std::size_t*        out_page_size,
    uint64_t*           out_first_ioaddr) const
{
    if (region == nullptr || device == nullptr) return false;

    std::lock_guard<std::mutex> lock(mtx_);
    auto it = regions_.find(region->region_id);
    if (it == regions_.end()) return false;
    const Slot& slot = it->second;

    auto kit = slot.dma_per_device.find(device);
    if (kit == slot.dma_per_device.end() || kit->second == nullptr) return false;
    nvm_dma_t* dma = kit->second;

    if (out_count)        *out_count        = (std::size_t)dma->n_ioaddrs;
    if (out_page_size)    *out_page_size    = (std::size_t)dma->page_size;
    if (out_first_ioaddr) *out_first_ioaddr = (dma->n_ioaddrs > 0)
                                                ? (uint64_t)dma->ioaddrs[0]
                                                : 0ULL;
    return true;
}

} // namespace tutti
