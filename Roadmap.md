# GeminiFS Unified Storage Runtime Roadmap

## Status

- Current active version: `v0.1`
- This document describes the `v0.1` architecture baseline and the active roadmap.
- Future version number changes are maintainer-driven and must not be advanced automatically.
- The project name may change in a future version from `GeminiFS` to a more general runtime-oriented name.
- Historical roadmap snapshots are archived under [`doc/history/`](doc/history/).
- Every version roadmap must preserve:
  - a `Feature Snapshot`
  - a `Known Bugs Snapshot`

## v0.1 Positioning

`v0.1` defines GeminiFS as a `Unified Storage Runtime` rather than only a GPU file abstraction.

Naming note:

- `GeminiFS` is currently treated as the repository and legacy implementation name
- the long-term product name may be replaced by a more general and easier-to-understand runtime name
- new architecture, API, and directory decisions should avoid unnecessarily hard-coding the `GeminiFS` name

The runtime is intended to provide:

- Stable upper-layer APIs for systems such as `LMCache` and `Mooncake`
- Both `CPU-side` and `GPU-side` read/write paths
- A standalone memory subsystem for host/device allocation and registration
- A backend SPI that can evolve toward `Local NVMe`, `GDS`, `RDMA`, and vendor-specific backends
- Clear separation between `device manager`, `IO engine`, and `memory model`

`v0.1` explicitly does **not** define `cooperative submit`.

Current assumption:

- CPU and GPU do not normally access the same data region at the same time
- Submission mode is either `CPU_SUBMIT` or `GPU_SUBMIT`
- Any future CPU/GPU hardware-cooperative path will be introduced only after an explicit version decision

## v0.1 Feature Snapshot

The `v0.1` version baseline is intended to include the following core features:

- reposition the project as a `Unified Storage Runtime`
- define stable runtime-facing abstractions instead of exposing file-system-only internals
- provide both `CPU-side` and `GPU-side` read/write paths
- support exactly two submission modes:
  - `CPU_SUBMIT`
  - `GPU_SUBMIT`
- treat memory management as a first-class subsystem with host/device allocation and registration semantics
- define a backend SPI suitable for future `Local NVMe`, `GDS`, and `RDMA` evolution
- treat the modified NVMe Linux kernel module as part of the `local_nvme` backend baseline
- support both direct GPU-owned bootstrap and `NVMeService`-owned bootstrap
- keep upper-layer adaptation for systems such as `LMCache` and `Mooncake` outside the core runtime

Note:

- this snapshot records the `v0.1` feature baseline and roadmap intent
- it does not imply that every item is already fully implemented in the current codebase

## v0.1 Known Bugs Snapshot

Known bugs and unstable areas currently tracked for `v0.1`:

- `GPU file persistence` has known correctness/stability issues and must not yet be treated as a stable persistence contract

## Current Repository Problems

The current codebase does not yet match the `v0.1` target architecture. Main issues:

- Initialization is over-coupled: controller discovery, GPU setup, NVMe setup, file management, and runtime bootstrap are mixed together
- Configuration is split across incompatible formats and parsing paths
- Public interfaces leak backend details such as controller internals, CUDA details, and file-layout assumptions
- Memory registration, mapping, and allocation are spread across storage-facing classes instead of a dedicated subsystem
- The current file-oriented model is too narrow for upper-layer cache/block/object runtimes
- The modified NVMe kernel module lifecycle is not yet treated as a formal deployment and compatibility concern

## v0.1 Target Architecture

### Layering

`v0.1` uses the following logical layers:

1. `API Layer`
   - Stable runtime API exposed to applications and adapters
   - Provides CPU read/write, GPU read/write, batch IO, and lifecycle management

2. `Adapter Layer`
   - Integration adapters for frameworks such as `LMCache` and `Mooncake`
   - Converts framework-specific concepts into runtime-neutral requests

3. `Core Runtime Layer`
   - Defines runtime object model, request model, error model, lifecycle, and capability queries
   - Must not depend on a concrete backend implementation

4. `Memory Layer` *(independent — parallel to Device Manager)*
   - Manages host/device allocation, registration, deregistration, and region metadata
   - Owns the memory model used by both the IO engine and upper-layer integrations
   - Has no dependency on the Device Manager

5. `Device Manager Layer` *(independent — parallel to Memory Layer)*
   - Device discovery
   - topology and capability reporting
   - queue/resource lease management
   - process attach metadata
   - health and lifecycle management
   - Has no dependency on the Memory Layer

6. `IO Engine Layer` *(depends on both Memory Layer and Device Manager)*
   - Read/write submission
   - mapping and buffer preparation
   - completion handling
   - batch execution
   - CPU_SUBMIT and GPU_SUBMIT execution paths

7. `Backend SPI Layer`
   - Formal backend extension interface
   - Supports pluggable backends without changing upper-layer APIs

8. `Backend Implementations`
   - `local_nvme` as the first reference backend
   - future candidates include `gds_nvme`, `rdma`, and hybrid backends

### Kernel Module Baseline

The current `backend/kernel_modules` area contains a modified Linux NVMe kernel module lineage used to support CPU-side and GPU-side access to NVMe queues.

This must be treated as a formal architecture dependency rather than an implementation detail.

Rules for `v0.1`:

- the kernel module is part of the `local_nvme` backend baseline
- queue-level CPU/GPU simultaneous access capability is a backend/driver capability, not a public API promise of cooperative submit
- runtime semantics must still assume explicit ownership and clear submission mode boundaries
- kernel-facing logic must be isolated enough that future Linux version support can evolve without rewriting upper-layer APIs

### Driver Lifecycle and Bootstrap Model

The system must support two runtime bootstrap paths:

- `GPU-owned bootstrap`
  - a GPU process initializes the data path directly when deployment chooses a process-local ownership model

- `Service-owned bootstrap`
  - `NVMeService` initializes and manages shared controller/queue resources
  - GPU processes attach later without owning low-level initialization

`v0.1` should treat these as deployment modes over the same backend/device-manager model, not as separate architectures.

### Deployment and Compatibility Constraints

For the modified NVMe kernel module, the roadmap must account for:

- Linux version compatibility strategy
  - the module will need an explicit support matrix for targeted kernel versions
  - internal adaptation points should be isolated for kernel API drift

- installation timing
  - the module is expected to be installed before runtime use
  - preferred deployment model is system startup installation rather than ad hoc runtime build/load

- operational packaging
  - deployment should consider package-based install, DKMS-style rebuild strategy, or another explicit lifecycle model
  - startup integration should consider `systemd`, boot-time module loading, and service ordering

- runtime prerequisites
  - service startup, GPU attach flow, permissions, device nodes, and module readiness must be checkable before data-path use

- failure handling
  - deployment design must define what happens when the module is missing, kernel ABI is incompatible, or service bootstrap fails

These constraints are part of architecture planning because they directly affect portability, operability, and contributor usability.

### Runtime Object Model

`v0.1` should converge on the following core objects:

- `Runtime`
- `RuntimeConfig`
- `Device`
- `StorageTarget`
- `MemoryRegion`
- `IOBuffer`
- `IORequest`
- `BatchRequest`
- `Completion`
- `Lease`
- `CapabilitySet`

Rules:

- Upper layers must depend on these abstract objects instead of controller/file implementation details
- Backend implementations may extend internals, but not the public object model
- File is treated as one storage object form, not the only storage abstraction

### Submission Model

`v0.1` supports exactly two submission modes:

- `CPU_SUBMIT`
  - CPU prepares and submits IO
  - GPU may consume or produce the data buffer, but submission ownership remains on CPU

- `GPU_SUBMIT`
  - GPU prepares and initiates the IO path defined by the backend
  - CPU may assist with control or completion plumbing, but not as a cooperative execution model

Non-goals in `v0.1`:

- No `COOPERATIVE_SUBMIT`
- No requirement that CPU and GPU simultaneously operate on the same logical region
- No implicit concurrency semantics beyond explicit API contracts

Note:

- backend driver support for CPU/GPU queue access does not by itself justify exposing a cooperative runtime submission model
- queue-sharing capability and runtime ownership semantics must remain distinct concepts

### Memory Model

The memory subsystem is a first-class part of `v0.1`.

Supported memory categories:

- `HOST`
- `PINNED_HOST`
- `DEVICE`
- `MANAGED`
- `EXTERNAL`

Memory operations must be separated by semantics:

- Allocation
  - `allocate_host`
  - `allocate_pinned_host`
  - `allocate_device`
  - `free`

- Registration
  - `register_host`
  - `register_device`
  - `unregister`

- Query
  - `query_region`
  - `query_capabilities`

- Exchange
  - reserved for future import/export and cross-process sharing

Every `MemoryRegion` should describe at least:

- address
- size
- alignment
- location
- ownership
- registration state
- access capabilities
- backend-visible attributes

### API Direction for Upper Layers

`v0.1` runtime APIs should serve both general applications and cache/object systems.

Required API classes:

- Runtime lifecycle API
- Capability and topology query API
- CPU read/write API
- GPU read/write API
- Batch IO API
- Memory allocation and registration API
- Queue/resource lease API

API constraints:

- Do not expose backend-private types in the public API
- Do not bind public APIs to a specific file-layout implementation
- Do not hardcode `LMCache` or `Mooncake` structures into the core runtime
- Framework-specific adaptation belongs in dedicated adapters

## v0.1 Recommended Directory Direction

This is a design target, not a completed repository state.

```text
GeminiFS/
├── api/                # public runtime API definitions
├── runtime/            # core runtime objects and orchestration
├── memory/             # allocation, registration, region model
├── device_manager/      # daemon/client/protocol for device manager
├── io_engine/         # submission, mapping, completion, batching
├── backends/           # backend SPI and backend implementations
├── adapters/           # LMCache, Mooncake, and future integrations
└── doc/
    ├── architecture/   # architecture descriptions
    ├── rfcs/           # design RFCs
    ├── ai/             # AI-facing subsystem docs
    └── history/        # archived roadmap snapshots
```

## Active Roadmap

### Phase 0: Freeze the Architectural Baseline

Goals:

- Define `Unified Storage Runtime` as the official product direction
- Stop extending the old monolithic initialization path
- Freeze core concepts, naming, and boundaries before moving directories

Deliverables:

- `v0.1` architecture document in this roadmap
- stable terminology for runtime, memory, device manager, IO engine, and backend SPI
- explicit rejection of `cooperative submit` in this version
- naming transition requirement recorded so future APIs are not forced to retain the `GeminiFS` label

### Phase 1: Define Stable Core Interfaces

Goals:

- Define the public runtime API surface before rewriting internals
- Remove direct exposure of controller/file-system internals from future public headers

Deliverables:

- public object model
- request/response model
- error model
- lifecycle model
- capability query model

### Phase 2: Extract the Memory Subsystem

Goals:

- Separate memory ownership and memory registration from storage logic
- Make host and device memory first-class runtime resources

Deliverables:

- `MemoryRegion` model
- host/device allocation APIs
- host/device registration APIs
- clear ownership and teardown semantics

### Phase 3: Split Device Manager and IO Engine

Goals:

- Move discovery, topology, leases, and shared-resource metadata into the device manager
- Keep read/write execution in the IO engine
- Separate service-owned bootstrap and process-owned bootstrap from upper-layer APIs

Deliverables:

- device manager responsibilities and interfaces
- IO engine responsibilities and interfaces
- runtime bootstrap mode definition
- attach/init boundary for GPU process and `NVMeService`
- explicit attach path between device manager and IO engine

### Phase 4: Introduce Backend SPI

Goals:

- Make backend replacement and extension possible without rewriting upper layers
- Ensure future `RDMA` and `GDS` work is additive rather than invasive

Design contract: [`doc/design/backend-spi.md`](doc/design/backend-spi.md)

Deliverables:

- `IBackendProvider` interface (`prepare_descriptors`, `acquire_queue`, `release_queue`, `launch_io_kernel`)
- `BufferDescriptor` tagged union (NVMe + RDMA placeholder)
- `BackendRegistry` wiring
- `local_nvme` refactored to implement `IBackendProvider`

### Phase 5: Land the First Reference Backend

Goals:

- Provide one complete backend that validates the architecture
- Use `local_nvme` as the reference implementation

Deliverables:

- reference local backend
- CPU_SUBMIT path
- GPU_SUBMIT path
- capability reporting
- explicit kernel-module dependency contract
- bootstrap support for direct init and service-managed init

### Phase 6: Build Upper-Layer Adapters

Goals:

- Make the runtime consumable by external application stacks
- Keep framework-specific logic outside the core runtime

Deliverables:

- `LMCache` adapter plan
- `Mooncake` adapter plan
- adapter boundary rules

### Phase 7: Documentation and Governance

Goals:

- Make the architecture maintainable by both humans and AI contributors
- Ensure later backends can be implemented independently

Deliverables:

- architecture docs
- AI-facing subsystem docs
- RFC templates and review rules
- backend extension guidance
- deployment guide for module install, boot ordering, and service startup
- kernel compatibility policy for supported Linux versions

## Versioning Rules

- `Roadmap.md` is always the active roadmap for the current version selected by the maintainer
- Version changes are not made automatically
- Project naming changes are also maintainer-driven and should be handled explicitly rather than implicitly during refactors
- Every active and archived version roadmap must retain a per-version `Feature Snapshot` and `Known Bugs Snapshot`
- When a new version is opened, the previous active roadmap snapshot should be copied into [`doc/history/`](doc/history/)
- Archive file naming should follow:
  - `roadmap-v0.1.md`
  - `roadmap-v0.2.md`
  - `roadmap-v1.0.md`

## Out of Scope for v0.1

- cooperative CPU/GPU submit model
- simultaneous CPU/GPU access optimization for the same logical region
- committing to a single remote transport design before the backend SPI is stabilized
- binding core runtime APIs directly to one framework's internal data structures
