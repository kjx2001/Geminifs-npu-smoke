#ifndef __TUTTI_API_RUNTIME_CONFIG_H__
#define __TUTTI_API_RUNTIME_CONFIG_H__

/**
 * runtime_config.h -- the configuration the application hands to
 * Runtime::initialize().
 *
 * Layer: API (Roadmap.md §3 layered architecture).
 *
 * Design principles:
 *   - The config describes *intent*, not in-memory pointers. It is
 *     YAML / JSON / proto -friendly so a deployment tool can write
 *     it to disk and the Runtime can parse it on startup.
 *   - Per-backend specifics are kept opaque (`config_blob`) so the
 *     api/ layer never needs to know each backend's tuning knobs.
 *     Each backend factory parses its own blob.
 *   - For app-driven composition (no on-disk config), use
 *     `Runtime::add_backend()` / `add_filesystem()` directly and
 *     leave the corresponding lists empty in the config.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "../../io_engine/include/backend_type.h"

namespace tutti {

/**
 * Where the device fleet lives.
 *
 * IN_PROCESS: this process performs device discovery itself
 *             (libnvm-style direct controller open). Suitable for
 *             single-tenant deployments.
 *
 * SERVICE_CLIENT: this process attaches to a daemon (NVMeService
 *             today; future RDMA-pool daemon analogously) and gets
 *             device handles + leases over RPC. Suitable for the
 *             multi-tenant `Service-owned bootstrap` path described
 *             in Roadmap.md §3.
 */
enum class DeviceManagerMode : uint32_t {
    IN_PROCESS     = 0,
    SERVICE_CLIENT = 1,
};

/**
 * Cross-process device manager wiring. Only consulted when
 * `mode == SERVICE_CLIENT`.
 */
struct DeviceManagerConfig {
    DeviceManagerMode mode             = DeviceManagerMode::IN_PROCESS;

    /// e.g. "unix:///var/run/tutti/nvmeservice.sock" or "127.0.0.1:50051".
    /// Empty for IN_PROCESS.
    std::string       daemon_endpoint;

    /// Stable identity used by the daemon for lease bookkeeping.
    /// Typically `<hostname>-<pid>` if left empty at runtime startup.
    std::string       client_id;

    /// Heartbeat interval the client uses against this daemon.
    /// 0 lets the daemon pick a default.
    uint32_t          heartbeat_interval_sec = 0;
};

/**
 * One backend the runtime should bring up at startup.
 *
 * If `library_path` is non-empty, the runtime dlopens it and looks
 * up `extern "C" tutti_backend_<lower(type)>_create(const RuntimeConfig*)`.
 * If empty, the runtime expects the backend to have been linked
 * statically and registered through a generated dispatch table.
 */
struct BackendConfig {
    BackendType type;

    /// Optional shared-object path. Empty = statically linked.
    std::string library_path;

    /// Backend-private config, parsed by the backend factory.
    /// v0.1 uses an opaque YAML / JSON string; future versions may
    /// switch to a typed proto. The api/ layer never inspects it.
    std::string config_blob;
};

/**
 * One filesystem the runtime should bring up at startup.
 *
 * `name` is the alias the runtime registers it under (e.g. "ext4",
 * "tutti_layout", "3fs"). Applications then refer to it by alias
 * when they call `Runtime::filesystem(name)`.
 */
struct FilesystemConfig {
    std::string name;          // alias used at runtime
    std::string library_path;  // empty = statically linked
    std::string config_blob;   // FS-private config
};

/**
 * Memory subsystem hints. Allocation is application-driven (Roadmap
 * memory ownership policy: app allocates, runtime registers) so this
 * is purely advisory — pool sizes, default registration cache size,
 * GPU pinning preferences, etc. v0.1 leaves it empty; reserved for
 * future use.
 */
struct MemorySubsystemConfig {
    // Reserved for v0.2+: registration cache cap, NUMA hints, ...
    uint32_t reserved = 0;
};

/**
 * The full runtime startup config.
 *
 * Lifecycle:
 *   - Constructed by the application (parsed from YAML / programmatic).
 *   - Passed to `Runtime::initialize(cfg)`. The runtime captures what
 *     it needs and does not retain a reference after initialize()
 *     returns; subsequent edits to `cfg` are ignored.
 */
struct RuntimeConfig {
    DeviceManagerConfig             device_manager;
    MemorySubsystemConfig           memory;

    /// Backends to bring up. Order is preserved; backends are
    /// initialised in order so later backends can depend on earlier
    /// ones if needed (e.g. an RDMA-NVMe hybrid depending on RDMA).
    std::vector<BackendConfig>      backends;

    /// Filesystems to register. Order is preserved.
    std::vector<FilesystemConfig>   filesystems;

    /// Free-form log level for the runtime itself. Backend log levels
    /// are typically passed via the per-backend `config_blob`.
    /// Recognised values: "trace", "debug", "info", "warn", "error".
    /// Empty = "info".
    std::string                     log_level;
};

} // namespace tutti

#endif // __TUTTI_API_RUNTIME_CONFIG_H__
