#ifndef __TUTTI_API_ERROR_H__
#define __TUTTI_API_ERROR_H__

/**
 * error.h -- the runtime's status / error model.
 *
 * Layer: API (Roadmap.md §3 layered architecture).
 *
 * Why a return-value Status instead of exceptions:
 *   - Tutti's submission path crosses GPU launches, async futures,
 *     RPC boundaries (NVMeService and future RDMA daemons), and
 *     C-ABI callable shims for non-C++ adapters. Exceptions don't
 *     cross those cleanly. Status returns do.
 *   - The cost (verbose call sites) is paid by the runtime
 *     internals; adapter-facing helpers can wrap Status in
 *     `std::optional` / exceptions if a particular adapter prefers.
 *
 * Backend-code passthrough:
 *   - When a backend reports a failure (e.g. NVMe completion
 *     status non-zero, ibverbs WC error), the original numeric code
 *     is preserved in `backend_code`. This lets advanced callers
 *     differentiate transient from fatal without sniffing the
 *     `message` string.
 */

#include <cstdint>
#include <string>
#include <utility>

namespace tutti {

/**
 * Coarse error categories. Numeric values are stable; new categories
 * append. Keep the set small — backend specifics go in `Status::message`
 * and `Status::backend_code`.
 */
enum class StatusCode : int32_t {
    OK                         = 0,

    // Argument / state validation
    INVALID_ARG                = 1,
    NOT_INITIALIZED            = 2,
    ALREADY_INITIALIZED        = 3,

    // Lookup failures
    NO_DEVICE                  = 10,   // device_id not in registry
    NO_BACKEND                 = 11,   // BackendType has no registered provider
    NO_FILESYSTEM              = 12,   // filesystem alias not registered
    NO_LEASE_MANAGER           = 13,   // device has no ILeaseManager

    // Capability mismatches (caught at dispatch time)
    CAPABILITY_NOT_SUPPORTED   = 20,   // device doesn't advertise the requested IOSubmitMode
    TARGET_KIND_NOT_SUPPORTED  = 21,   // backend doesn't speak this StorageTargetKind
    MIXED_BACKEND_BATCH        = 22,   // BatchRequest entries hit >1 backend (v0.1 limitation)

    // Lease / resource problems
    LEASE_INVALID              = 30,   // lease_id not recognised
    LEASE_EXPIRED              = 31,   // recognised but past timeout / heartbeat
    OUT_OF_RESOURCE            = 32,   // queues / QPs / buffers exhausted

    // Pass-through failures
    MEMORY_REGISTRATION_FAILED = 40,   // prepare_descriptors / MR setup failed
    BACKEND_FAILED             = 41,   // generic backend submission/completion failure

    // Catch-all
    INTERNAL                   = 99,
};

/**
 * One status value. Default-constructed Status is OK.
 *
 * Construction patterns:
 *   - `Status::ok()` -- explicit success
 *   - `Status::invalid_arg("missing field X")` -- common shorthand
 *   - `Status{StatusCode::BACKEND_FAILED, errno, "submit failed"}` -- raw
 */
struct Status {
    StatusCode  code         = StatusCode::OK;
    int32_t     backend_code = 0;     // passthrough numeric code; 0 if N/A
    std::string message;

    bool ok() const noexcept { return code == StatusCode::OK; }

    // ---- Common factories --------------------------------------------
    static Status OK();
    static Status invalid_arg(std::string msg);
    static Status not_initialized();
    static Status no_device(int32_t device_id);
    static Status no_backend(int32_t backend_type);
    static Status capability_not_supported(std::string msg);
    static Status target_kind_not_supported(std::string msg);
    static Status mixed_backend_batch();
    static Status lease_invalid(std::string lease_id);
    static Status backend_failed(int32_t backend_code, std::string msg);
    static Status internal(std::string msg);
};

inline Status Status::OK() {
    return Status{};
}
inline Status Status::invalid_arg(std::string msg) {
    return Status{StatusCode::INVALID_ARG, 0, std::move(msg)};
}
inline Status Status::not_initialized() {
    return Status{StatusCode::NOT_INITIALIZED, 0, "Runtime is not initialized"};
}
inline Status Status::no_device(int32_t device_id) {
    return Status{StatusCode::NO_DEVICE, device_id,
                   "No device with id=" + std::to_string(device_id)};
}
inline Status Status::no_backend(int32_t backend_type) {
    return Status{StatusCode::NO_BACKEND, backend_type,
                   "No backend registered for backend_type=" +
                   std::to_string(backend_type)};
}
inline Status Status::capability_not_supported(std::string msg) {
    return Status{StatusCode::CAPABILITY_NOT_SUPPORTED, 0, std::move(msg)};
}
inline Status Status::target_kind_not_supported(std::string msg) {
    return Status{StatusCode::TARGET_KIND_NOT_SUPPORTED, 0, std::move(msg)};
}
inline Status Status::mixed_backend_batch() {
    return Status{StatusCode::MIXED_BACKEND_BATCH, 0,
                   "BatchRequest entries span more than one backend; "
                   "v0.1 requires one backend per batch"};
}
inline Status Status::lease_invalid(std::string lease_id) {
    return Status{StatusCode::LEASE_INVALID, 0,
                   "lease_id not recognised: " + lease_id};
}
inline Status Status::backend_failed(int32_t backend_code, std::string msg) {
    return Status{StatusCode::BACKEND_FAILED, backend_code, std::move(msg)};
}
inline Status Status::internal(std::string msg) {
    return Status{StatusCode::INTERNAL, 0, std::move(msg)};
}

} // namespace tutti

#endif // __TUTTI_API_ERROR_H__
