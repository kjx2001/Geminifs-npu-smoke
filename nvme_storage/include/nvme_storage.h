#ifndef __TUTTI_NVME_STORAGE_NVME_STORAGE_H__
#define __TUTTI_NVME_STORAGE_NVME_STORAGE_H__

/**
 * nvme_storage.h -- the INvmeStorage interface.
 *
 * Layer: nvme_storage.  See doc/refactor/LegacyDecomposition.md §3.6.
 *
 * Role:
 *   - Owns named LBA ranges on top of one or more NVMe namespaces.
 *   - Knows how to mount / format / FIEMAP host filesystems on top
 *     of the snvme block device, but exposes only a flat "named
 *     file" API.  No filesystem-specific concept leaks out.
 *   - Provides host-side blocking IO (for bootstrap / metadata /
 *     tests) and -- in a follow-up R5b -- device-side submission
 *     primitives that io_engine kernels call inline.
 *
 * Layer boundary:
 *   - This header MUST NOT pull in libnvm headers; we want callers
 *     (block_storage) to consume the abstraction without leaking
 *     libnvm types.  Implementations are free to use libnvm
 *     internally.
 *   - Capacity / file directory metadata is persisted on the host
 *     filesystem (via PersistentFileLog).  That backing store is
 *     deliberately not exposed.
 *
 * Lifetime:
 *   - bootstrap() takes a list of Device*; for each device it
 *     mounts the host fs, prepares a directory, and reloads the
 *     PersistentFileLog.  Failure rolls back partial mounts.
 *   - shutdown() flushes everything and unmounts.  Idempotent.
 *
 * Multi-NVMe:
 *   - One INvmeStorage instance services multiple devices.
 *     Per-Device state is keyed off `Device*` everywhere.
 *
 * Threading:
 *   - Public methods are thread-safe; v0.1 uses a single std::mutex.
 *
 * Deferred to R5b:
 *   - acquire_queue_pair / release_queue_pair (host-side handle to
 *     a libnvm SQ/CQ pair the GPU kernel will write into).
 *   - __device__ submit_read_one / submit_write_one (device-side
 *     submit functions called inline from io_engine kernels).
 *   - NvmeFileDeviceHandle (GPU-resident mirror of NvmeFile).
 */

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <sys/types.h>     // ssize_t
#include <vector>

#include "nvme_file.h"

namespace tutti {

struct Device;

class INvmeStorage {
public:
    virtual ~INvmeStorage() = default;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Mount + prepare the on-disk directory for every device.
    /// Returns false on any failure; partial state rolled back.
    /// MUST be called before any other method.
    virtual bool bootstrap(const std::vector<const Device*>& devices) = 0;

    /// Unmount everything and release internal state.  Safe to call
    /// twice.  Returns false if any unmount failed (state is still
    /// cleared on best-effort).
    virtual bool shutdown() = 0;

    // ------------------------------------------------------------------
    // Capacity (per device)
    // ------------------------------------------------------------------

    virtual uint64_t total_capacity   (const Device*) const = 0;
    virtual uint64_t available_capacity(const Device*) const = 0;

    // ------------------------------------------------------------------
    // Directory
    // ------------------------------------------------------------------

    /// Create a named NvmeFile of `size_bytes` user-visible bytes on
    /// `device`.  Internally allocates `sizeof(NvmeFileHeader) +
    /// size_bytes` on the underlying host filesystem, embeds the
    /// header at byte 0, and stores the FIEMAP extent list in both
    /// the header and the persistent file log.
    /// Returns nullptr if a file with the same name already exists,
    /// or on allocation / FIEMAP failure.
    virtual NvmeFile* create_file(const Device*  device,
                                   std::string_view name,
                                   uint64_t        size_bytes) = 0;

    /// Re-open an existing file (host fd reopened).  Returns nullptr
    /// if not found.
    virtual NvmeFile* open_file(const Device*    device,
                                 std::string_view name) = 0;

    /// fsync the host fd, close it, and persist the file log.  The
    /// NvmeFile pointer remains valid for re-open / list_files but
    /// its host_fd becomes < 0.
    virtual bool      close_file(NvmeFile* file) = 0;

    /// Remove the file from the directory + delete the underlying
    /// host file.  Returns false if not found or unlink fails.
    virtual bool      delete_file(NvmeFile* file) = 0;

    /// All currently-known files for `device`.
    virtual std::vector<NvmeFile*> list_files(const Device*) const = 0;

    // ------------------------------------------------------------------
    // Host-side blocking IO (R5a)
    //
    // Useful for:
    //   - bootstrap / metadata writes that don't need GPU paths
    //   - the smoke test (verify byte content roundtrip)
    //   - cooperative CPU side of mixed CPU+GPU IO
    //
    // Implementation walks pread/pwrite over the host fd.  It is
    // NOT the high-throughput path -- that's the GPU device-side
    // submit (R5b).
    //
    // `byte_offset` is the LOGICAL offset (excludes the 4 KiB
    // NvmeFileHeader at byte 0).  The implementation adjusts.
    // ------------------------------------------------------------------

    virtual ssize_t read_blocking (NvmeFile* file, uint64_t byte_offset,
                                    void*        dst, size_t len) = 0;
    virtual ssize_t write_blocking(NvmeFile* file, uint64_t byte_offset,
                                    const void* src, size_t len) = 0;

    /// fsync the file (data + metadata).  Useful when callers want
    /// a flush point without closing.
    virtual bool    sync(NvmeFile* file) = 0;
};

} // namespace tutti

#endif // __TUTTI_NVME_STORAGE_NVME_STORAGE_H__
