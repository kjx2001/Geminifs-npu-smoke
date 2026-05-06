#ifndef __TUTTI_API_RUNTIME_H__
#define __TUTTI_API_RUNTIME_H__

/**
 * runtime.h -- the API-layer Runtime entry point.
 *
 * Layer: API (Roadmap.md §3 layered architecture).
 *
 * Role:
 *   - The single class application code creates to talk to Tutti.
 *   - Owns / coordinates the four subsystems described in Roadmap.md:
 *        * Memory (`IMemorySubsystem`)
 *        * Device Manager (`IDeviceRegistry` + per-device
 *          `ILeaseManager` instances)
 *        * IO Engine + Backend SPI (`IBackendProvider` per backend)
 *        * Filesystems (`IFilesystem` keyed by alias)
 *   - Exposes a thin convenience surface (`submit_batch`,
 *     `acquire_lease`) and direct subsystem accessors so adapters
 *     can compose at whatever granularity they want.
 *
 * Threading:
 *   - `initialize()` and `shutdown()` are NOT thread-safe; the caller
 *     coordinates lifecycle.
 *   - All other methods are safe to call concurrently from multiple
 *     threads, subject to the underlying subsystem contracts.
 *   - Concurrent calls into a single ICompletionSink are sequenced
 *     by the runtime's dispatch logic (sink sees one on_complete()
 *     per entry, in arbitrary order).
 *
 * Lifetime:
 *   - The application owns the `Runtime` object. Subsystems and
 *     handles returned by accessors are valid until shutdown()
 *     completes; callers MUST NOT delete them and MUST NOT use them
 *     after shutdown.
 */

#include <cstdint>
#include <string>

#include "error.h"
#include "runtime_config.h"
#include "../../io_engine/include/backend_type.h"
#include "../../io_engine/include/io_submit_mode.h"

namespace tutti {

// Forward declarations — kept out of the include graph to keep
// api/runtime.h cheap to include from application code.
struct Device;
struct BatchRequest;
struct Lease;
class  IBackendProvider;
class  IDeviceRegistry;
class  ILeaseManager;
class  IMemorySubsystem;
class  IFilesystem;

/**
 * The runtime entry point.
 *
 * Construction does NOT bring up any subsystem; call `initialize()`
 * with a `RuntimeConfig` to actually wire things up. This split lets
 * tests construct a `Runtime` and inject mocks via the manual
 * registration methods (`add_backend`, `add_filesystem`, ...) without
 * triggering daemon attach or device discovery.
 */
class Runtime {
public:
    Runtime();
    ~Runtime();

    Runtime(const Runtime&)            = delete;
    Runtime& operator=(const Runtime&) = delete;

    // ===== Lifecycle =================================================

    /// Bring up the runtime per `cfg`: connect to the device manager,
    /// load backends, register filesystems, prime the memory
    /// subsystem. Returns the first failure encountered; partial
    /// state is rolled back before returning.
    Status initialize(const RuntimeConfig& cfg);

    /// Tear down all subsystems in reverse-init order. Safe to call
    /// multiple times; the second call is a no-op.
    Status shutdown();

    /// Whether `initialize()` has run successfully and `shutdown()`
    /// has not.
    bool is_initialized() const noexcept;

    // ===== Manual registration =======================================
    //
    // Useful for tests, embedded use, or any composition that
    // bypasses the dlopen-based RuntimeConfig path.
    //
    // The runtime takes BORROWED pointers; the caller keeps ownership
    // and must outlive the Runtime.

    Status add_backend(BackendType type, IBackendProvider* backend);
    Status add_filesystem(const std::string& name, IFilesystem* fs);
    Status add_lease_manager(int32_t device_id, ILeaseManager* mgr);

    // ===== Subsystem accessors ========================================
    //
    // All accessors return borrowed pointers, valid until shutdown().
    // nullptr means "not registered" — never an error condition for
    // optional ones (e.g. lease manager on a device that doesn't
    // need one).

    IDeviceRegistry*  device_registry();
    IMemorySubsystem* memory_subsystem();

    IBackendProvider* backend_for(BackendType type);
    IBackendProvider* backend_for_device(int32_t device_id);
    IFilesystem*      filesystem(const std::string& name);
    ILeaseManager*    lease_manager(int32_t device_id);

    // ===== Submission =================================================

    /**
     * Dispatch a BatchRequest.
     *
     * Internally:
     *   1. Resolve each entry's StorageTarget -> Device via the
     *      device registry. Reject NO_DEVICE.
     *   2. Verify all entries hit the same backend (v0.1 limitation;
     *      MIXED_BACKEND_BATCH otherwise).
     *   3. Verify the device's CapabilitySet advertises `mode` and
     *      that the backend supports every entry's
     *      `target.kind` (CAPABILITY_NOT_SUPPORTED /
     *      TARGET_KIND_NOT_SUPPORTED).
     *   4. If `req.lease_id` is set, ask the device's lease manager
     *      whether it's still valid. Reject LEASE_INVALID /
     *      LEASE_EXPIRED.
     *   5. Lower the BatchRequest into the SPI pair
     *      (BufferDescriptorBatch, IORequestBatch). Descriptor
     *      preparation goes through IMemorySubsystem and is cached
     *      per (MemoryRegion, BackendType).
     *   6. Call the appropriate IBackendProvider::submit_batch_*
     *      based on `mode`. Completions land on `req.sink`.
     *
     * Synchronous vs asynchronous behaviour follows the IOSubmitMode:
     *   - BATCH_CPU_SYNC blocks until all completions have fired
     *     into `req.sink`.
     *   - BATCH_GPU_STREAM / BATCH_CPU_ASYNC / COOP return as soon
     *     as the work is enqueued; completions arrive on `req.sink`
     *     later (callback / future fire / proxy thread).
     */
    Status submit_batch(const BatchRequest& req, IOSubmitMode mode);

    // ===== Lease helpers (thin pass-through to ILeaseManager) =========

    /// Refresh a lease's heartbeat. Returns LEASE_INVALID if the
    /// device or lease isn't recognised.
    Status heartbeat_lease(int32_t device_id, const std::string& lease_id);

    /// Release a lease early.
    Status release_lease(int32_t device_id,
                         const std::string& lease_id,
                         uint32_t client_pid);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace tutti

#endif // __TUTTI_API_RUNTIME_H__
