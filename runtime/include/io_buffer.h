#ifndef __TUTTI_RUNTIME_IO_BUFFER_H__
#define __TUTTI_RUNTIME_IO_BUFFER_H__

/**
 * io_buffer.h -- the data buffer side of an IO operation.
 *
 * Layer: Core Runtime (Roadmap.md §3 object model).
 *
 * Role:
 *   - An IOBuffer is the runtime's view of "the bytes the IO reads or
 *     writes." It is a thin wrapper around a MemoryRegion with an
 *     offset/length window into it, so the same registered region can
 *     be reused as many IOBuffers (think: one tensor, many slices).
 *   - StorageTarget describes WHERE; IOBuffer describes WITH WHAT.
 *
 * Why not just hand the backend a (MemoryRegion*, offset, length)?
 *   - Same data, but the runtime needs a stable "buffer identity" to
 *     attach BufferDescriptors to (one set of PRP/SGL/RDMA-key
 *     metadata per IOBuffer). The IOBuffer is the cache key.
 *   - Adapters (LMCache / Mooncake) want to pass IOBuffer handles
 *     around without leaking MemoryRegion internals.
 *
 * Lifetime:
 *   - The runtime keeps the underlying MemoryRegion alive while the
 *     IOBuffer is referenced. IOBuffer destruction does NOT free the
 *     MemoryRegion -- that follows the application's
 *     register/unregister lifecycle in the Memory Layer.
 */

#include <cstdint>
#include <cstddef>

namespace tutti {

struct MemoryRegion;          // memory/include/memory_region.h
struct BufferDescriptor;      // io_engine/include/buffer_descriptor.h
struct BufferDescriptorBatch; // io_engine/include/buffer_descriptor.h

/**
 * One IO buffer. (region, byte_offset, byte_length) defines the window
 * into the registered region; `descriptors` points at the prepared
 * BufferDescriptor batch the IO Engine will submit on this buffer's
 * behalf.
 *
 * `descriptors` MAY be null if descriptors haven't been prepared yet
 * (the application registered the buffer but never asked for IO).
 * Backends that need them call IBackendProvider::prepare_descriptors()
 * lazily and the runtime caches the result here.
 */
struct IOBuffer {
    uint64_t            buffer_id;     // unique within the Runtime
    const MemoryRegion* region;        // app-registered home
    uint64_t            byte_offset;   // window start, relative to region
    uint64_t            byte_length;   // window size

    // Cached backend-specific descriptor batch covering this buffer.
    // Filled in by the runtime after prepare_descriptors(); null until
    // first use.
    const BufferDescriptorBatch* descriptors;
};

} // namespace tutti

#endif // __TUTTI_RUNTIME_IO_BUFFER_H__
