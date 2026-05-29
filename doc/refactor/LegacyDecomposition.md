# Legacy Decomposition Map

> Read-only exploration produced before any code is moved.  Tracks
> what lives in `filesystems/ext4/libgeminifs/` (a.k.a. **legacy**),
> where each piece is going, and in what order to extract it.
>
> Lifecycle: once every entry below is **DONE** and
> `filesystems/ext4/libgeminifs/` is empty, this whole
> `doc/refactor/` directory is deleted along with the legacy tree.

---

## 0. TL;DR

| Topic | Decision |
|-------|----------|
| Project name | Stays `Tutti` (matches `Roadmap.md`, `tutti::` namespace).  Repo directory name `Geminifs/` is unchanged. |
| `runtime/` → `coordinator/` | Coordinator is "the layer GPU programs talk to; stitches memory / device_manager / io_engine together."  Both the directory **and** the class are renamed (`class Runtime` → `class Coordinator`, `RuntimeConfig` → `CoordinatorConfig`).  Happens late — see "Refactor order" below. |
| Bottom-up ordering | We refactor **from the bottom up**: first `memory/` and `device_manager/` land in their final shape with their own smoke tests, then `io_engine/`, then the coordinator rename at the top.  The legacy tree stays buildable the whole time as a reference. |
| Legacy tree | `filesystems/ext4/libgeminifs/` is **frozen** for the duration.  Read-only logic reference.  Deleted in one shot at the end. |
| Build during refactor | Legacy continues to build until R1 lands a CMake target switch that drops it from the default build (no file is deleted). |
| Naming style | Layer name + functional noun.  No `geminifs` / `Gemini` prefix in any new symbol, file, or namespace. |
| Torch / heavy KV-cache APIs | Stay out of core layers; future home is `adapters/` (per `Roadmap.md` §3 layer 2). |
| `FilesystemDecomposition.md` | Written **after R5**, when we know what the device-side bring-up actually looks like.  Until then `filesystems/` is unplanned territory. |

---

## 1. Current state of the four core layers

There are actually **two generations** of code in the new layers and
the decomposition has to reconcile them:

### Generation A — Tutti SPI baseline (Roadmap-aligned, header-only)

Pure C++ headers in `tutti::` namespace, no implementations yet.

```
api/include/{runtime.h, runtime_config.h, error.h}
runtime/include/{device.h, lease.h, io_buffer.h, batch_request.h,
                 storage_target.h, capability_set.h}
memory/include/{memory_kind.h, memory_region.h, memory_subsystem.h}
device_manager/include/{device_registry.h, lease_manager.h}
io_engine/include/{backend_type.h, backend_provider.h, io_request.h,
                   buffer_descriptor.h, queue.h, queue_provider.h,
                   io_future.h, coop_channel.h, io_submit_mode.h}
```

These are the contracts the refactor must converge on.

### Generation B — early extraction from legacy (transitional, NOT `tutti::`)

```
device_manager/include/block_device_manager.cuh   // wraps legacy Controller
device_manager/include/block_address_translator.cuh // GPU-side FIEMAP walker
device_manager/src/block_device_manager.cu        // depends on geminifs_helper.h
io_engine/include/block_io_engine.cuh             // BlockIoEngine
io_engine/include/block_io_channel.cuh            // GPU-side IO channel
io_engine/include/nvme_queue_scheduler.cuh        // GPU-side QID picker + cmd issue
io_engine/src/block_io_engine.cu                  // depends on geminifs_helper.h
```

These are the "halfway out of legacy" residues from a previous attempt.
They are **not** in `tutti::` namespace, they hold raw pointers, and
they include `geminifs.h` / `geminifs_helper.h`.  They will be
reconciled against Generation A in step R2.

### Generation C — Backend implementations (already aligned, untouched by this refactor)

```
backends/local/nvme/libnvm/       // owner / client split done (4a)
backends/local/NVMeService/       // session broker (4b)
backends/local/kernel_modules/    // snvme (B3 + B6)
```

These are the L1/L2 work products.  Refactor does **not** touch
them except to add a thin "Tutti-side IBackendProvider impl"
(`backends/local/nvme/src/local_nvme_backend.cu`, future) that wraps
libnvm into the SPI from Generation A.

---

## 2. Legacy inventory

Twelve files in `filesystems/ext4/libgeminifs/`, grouped by purpose.

Sizes are rough KB.

### 2.1 Block-IO leaf code (translation + queue scheduling) — already partially extracted

| File | KB | What it is | Status |
|------|----|-----------|--------|
| `include/file.cuh` | 7.8 | `NVMe_File` class: `__get_nvmeofst`, `nvme_xfer`, `read_in`, `write_out` | **Generation B** copy lives in `io_engine/include/block_io_channel.cuh` and `device_manager/include/block_address_translator.cuh`.  Logic equivalent. |
| `include/helper.cuh` | 2.0 | `QueueAcquireHelper`: `acquire_queue` / `issue_nvme_cmd` / `poll` | **Generation B** copy lives in `io_engine/include/nvme_queue_scheduler.cuh`.  Logic equivalent. |
| `include/utils.cuh` | 2.1 | `cuda_assert`, `ROUND_UP`, `RUN_ON_DEVICE` macros | Spreads across many files; needs a clean home — proposal R2.4. |
| `include/prp_mapping_entry.h` | 1.1 | `PRPMappingEntry` (32 B POD) | Reusable; proposal R2.2 moves it under `io_engine/`. |

### 2.2 Storage-controller orchestration

| File | KB | What it is | Status |
|------|----|-----------|--------|
| `nvme_controller.cu` | 54 | host-side `NVMeController` impl: `g_open`, host/device file managed APIs, FIEMAP refining, copy header to device, kernel launches | Bulk logic already lifted to `device_manager/src/block_device_manager.cu` (Generation B).  Differences: legacy uses `Controller(SharedControllerSpec)` IPC path (post-4b dead).  Recheck under R2.3. |
| `include/nvme_controller.cuh` | 4.9 | `NVMeController` header | Replaced by `device_manager/include/block_device_manager.cuh`. |
| `nvme_file.cpp` | 23 | `FileManager`: append-only metadata log (bitmap + records), `OpenFileHandle` registry | Reusable.  Proposal R2.3 — move under `device_manager/` as the per-controller metadata log. |
| `include/nvme_file.h` | 4.2 | matching header | Same. |

### 2.3 GPU-side controller / memory mapper / KV-cache batched IO (Torch-flavoured)

| File | KB | What it is | Status |
|------|----|-----------|--------|
| `gpu_controller.cu` | 59 | `GPUController` + `GPUMemoryMapper`.  Tensor registration with PRP entries, multiple `nvme_*_kernel`s for KV-cache batched IO, GPU-side hash table of (tensor_ptr → PRP slice list). | **Torch-coupled adapter logic.**  Stays out of core. |
| `include/gpu_controller.cuh` | 12.3 | matching header | Same. |
| `gpu_file_manager.cu` | 31 | `GPUFileManager`: GPU-resident file table + bitmap log + `BatchIoPool` (BatchIoEntry GPU ring) | Same — adapter territory. |
| `include/gpu_file_manager.cuh` | 8.0 | matching header | Same. |

### 2.4 Top-level façade ("GeminiFS the noun") + adapter glue

| File | KB | What it is | Status |
|------|----|-----------|--------|
| `geminifs.cu` | 41 | `class GeminiFS` (Torch facade).  `geminifs_batched_{read,write}`, `geminifs_GPU_{read,write}_kernel`, register/unregister tensors, parse config, owns `GPUController` + `nvme_ctrl_param` list. | **Torch adapter facade.**  Drops out at end of refactor.  Replaced by `coordinator::Runtime` (Generation A `Runtime`) + an `adapters/torch_kvcache/` shim. |
| `geminifs.cpp` | 9.1 | Host-side `host_open_all` / `host_create_geminifs_file` / `host_open_geminifs_file` / `host_xfer_geminifs_file` / `host_refine_nvmeofst` — the C-callable open-all helper | Move pieces into `device_manager/` (FIEMAP refining + header build) and `coordinator/` (open-all bootstrap). |
| `geminifs_helper.cpp` | 17 | YAML / JSON config parsers; `PCI_BDF`; `parse_system_config`; logging macros (`geminifs_info` / `error` / `warn` / `debug`); fd-limit / cuda-aligned helpers | Split: parsers → become `coordinator::Config` next to current `nvmeservice_config`.  Logging macros → become a tiny `runtime/log.h` (or just `tutti_log.h`).  Cuda helpers → `memory/` utils. |
| `include/geminifs_helper.h` | 3.8 | matching header | Same. |
| `include/geminifs.h` | 3.7 | C-shaped public API: `host_fd_t`, `dev_fd_t`, `O_HOST`, `O_DEVICE`, `geminiFS_hdr`, `nvme_ctrl_param`, `geminifs_ctrl_params` | Split.  `geminiFS_hdr` (= the per-file metadata in 512 B with FIEMAP extents) is the actual filesystem-on-NVMe format and **must** survive in a renamed form — it's persisted on disk.  Proposal R2.1 below. |
| `include/geminifs.cuh` | 6.7 | Torch-coupled façade header (the `GeminiFS` class) | Adapter territory. |
| `include/geminifs_mem.h` | 3.9 | `PRPListPage`, `PRPTransferType`, `geminifs_dma`, `SubSliceInfo`, `GranularitySliceGroup` | Mixed.  `PRP*` constants → `io_engine/`.  `SubSliceInfo` already referenced from `io_engine/include/backend_provider.h` (it's currently a forward decl pointing here).  Resolve in R2.2. |
| `include/gemini_fiemap.h` | 5.2 | Kernel-ABI FIEMAP definitions (`fiemap`, `fiemap_extent`) + Gemini-side variants + `convert_*` helpers | Survives — it's a kernel ABI.  Renamed to `tutti_fiemap.h` and moved to a host-side helpers directory under `filesystems/ext4_fiemap/` (per `Roadmap.md`). |
| `ops.h` | 3.2 | Torch-Python binding declarations (`geminifs_init_fds_wrapper_cuda`, `batch_read_direct`, etc.) | Adapter territory. |
| `memory.cpp` | 3.0 | host malloc / free wrappers + small alignment helpers | Fold into `memory/` (host kind). |
| `backtrace.cpp` | 2.3 | libunwind stack trace pretty-printer for fatal aborts | Move to a tiny `runtime/util/backtrace.cpp` or drop (debug-only). |
| `include/backtrace.h` | 0 B (likely a 23-byte stub) | header | Same. |

### 2.5 Dead-for-good

| File | Action |
|------|--------|
| `geminifs.cu.bak` | DELETE in R0 cleanup commit. |
| `json.h` (already gone) | DONE — deleted at user request. |

---

## 3. Reconciliation: Generation B → Generation A

The hardest part isn't the legacy → new move; it's deciding what
shape the **canonical** new code takes once Generations A and B both
land in the same directory.  Recommendation per layer:

### `device_manager/`

Generation A says "interfaces only" (`IDeviceRegistry`,
`ILeaseManager`).  Generation B says "concrete `BlockDeviceManager`
that does libnvm bring-up and file lifecycle."

**Decision**: keep both but rename Generation B so its role is
clear.

- `device_manager/include/device_registry.h` — Generation A.
- `device_manager/include/lease_manager.h` — Generation A.
- `device_manager/include/block_device.h` — renamed from
  `block_device_manager.cuh`.  Becomes the local_nvme-specific
  concrete device backend, owned by an `IDeviceRegistry`
  implementation rather than directly by `coordinator/`.
- `device_manager/include/block_address_translator.cuh` — stays,
  but no longer includes `geminifs.h`.  The header it walks
  (`geminiFS_hdr` from legacy) is renamed (see R2.1) and moved.

The legacy `NVMeController` API surface (`g_open`, `host_*_managed`,
`device_*_managed`) is **dropped**.  Generation A's
`IDeviceRegistry::find_by_id() -> Device*` is the new entry point;
file management moves into `filesystems/ext4_fiemap/`.

### `io_engine/`

Generation A says "SPI interface (`IBackendProvider`,
`IQueueProvider`) plus value types (`IORequest`, `BufferDescriptor`,
...)."  Generation B says "`BlockIoEngine` that owns the scheduler
and vends per-file channels."

**Decision**: Generation B becomes the
`backends/local_nvme/`-internal implementation, not a `tutti::`-namespace
public class.  Specifically:

- `io_engine/include/*.h` from Generation A — stays, becomes the
  SPI all backends implement.
- `io_engine/include/block_io_engine.cuh`,
  `block_io_channel.cuh`, `nvme_queue_scheduler.cuh` — moved to
  `backends/local/nvme/src/` (the libnvm side).  These are the
  GPU-side guts behind one specific backend's
  `IBackendProvider::launch_batch_gpu_stream()` implementation.
- The current `io_engine/src/block_io_engine.cu` is split: the
  scheduler-init / kernel-launch glue follows the channel headers
  out of `io_engine/`; nothing else needs an `io_engine/src/` file
  in v0.1.

This makes `io_engine/` a pure header-SPI module ‑ much closer to
`Roadmap.md`'s spec.

### `memory/`

No reconciliation needed — Generation B has nothing here.
Population happens in R2.5 by moving `geminifs_mem.h`'s small,
non-Torch pieces (PRP entry / SubSliceInfo) into `io_engine/` and
the host-malloc helpers (`memory.cpp`) into `memory/`.

### `runtime/` → `coordinator/`

Generation A says "`Runtime` class is the GPU program's entry
point."  Generation B does not exist here.

**Decision**: rename `runtime/` → `coordinator/` in R7 (the **last**
non-cleanup step, after memory + device_manager + io_engine have
all settled).  Both directory and class get renamed —
`tutti::Runtime` → `tutti::Coordinator`,
`tutti::RuntimeConfig` → `tutti::CoordinatorConfig`.

---

## 4. Proposed sequence of commits

Bottom-up: every step ends with a buildable tree + a smoke that
exercises the just-landed layer.  Steps are independent commits.

```
        +---- R8 final cleanup (delete legacy + this file)
        |
        +---- R7 coordinator/ rename (top of the stack last)
        |
        +---- R6 io_engine/ becomes pure header SPI
        |     (push Generation-B residue down into backends/local/nvme)
        |
        +---- R4 device_manager/ rebuilt + smoke
        +---- R3 memory/ implementation + smoke
        |     (these two are the "bottom" — both are parallel-safe)
        |
        +---- R2.* small POD/util headers move (PRP / FileManager /
        |     fiemap / utils macros / on-disk header)
        |
        +---- R1 carve legacy out of default build (EXCLUDE_FROM_ALL +
        |     temporary include shim)
        |
        +---- R0 this map + delete .bak (done in this commit)
```

### R0 — Prep (THIS commit)

- Add `doc/refactor/LegacyDecomposition.md`.
- Delete `filesystems/ext4/libgeminifs/json.h` (already done).
- Delete `filesystems/ext4/libgeminifs/geminifs.cu.bak`.
- **No** code moves.

### R1 — Carve legacy out of the default build

- New CMake target `libgeminifs_legacy` with `EXCLUDE_FROM_ALL`.
- Generation-B residue (`device_manager/src/`, `io_engine/src/`)
  currently `#include "geminifs.h"` / `#include "geminifs_helper.h"`.
  Keep a temporary `INTERFACE` shim that re-exposes only those two
  headers from `filesystems/ext4/libgeminifs/include/` to those two
  layers, so they still build.  Document the shim as "removed by R6."
- **Default `make` no longer builds the legacy `.cu`/`.cpp`.**
  Sources remain on disk for reference.

### R2 — Move the leaf POD / utility headers

These are small, ownerless types referenced from multiple places.
Move first so the bigger refactors don't have to chase headers.

| Sub-step | What moves | Destination |
|----------|------------|-------------|
| R2.1 | `geminiFS_hdr` (on-disk file header, FIEMAP-bearing) | `filesystems/ext4_fiemap/include/ext4_block_header.h`; struct renamed `Ext4BlockFileHeader`; magic number unchanged (binary on-disk compat) |
| R2.2 | `PRPMappingEntry`, `PRPListPage`, `PRP_*` constants, `SubSliceInfo` | `io_engine/include/nvme_prp_descriptor.h` (canonical home; `backend_provider.h`'s `SubSliceInfo` forward decl points here) |
| R2.3 | `FileManager` (append-only metadata log) | `device_manager/include/persistent_file_log.h` + `device_manager/src/persistent_file_log.cpp`; class renamed `PersistentFileLog` |
| R2.4 | `utils.cuh` macros | split: CUDA helpers → `memory/include/cuda_helpers.cuh`; non-CUDA macros (`ROUND_UP`) → `coordinator/include/tutti_macros.h` (lives under the still-named `runtime/` directory until R7 rename) |
| R2.5 | `gemini_fiemap.h` (FIEMAP ABI wrapper) | `filesystems/ext4_fiemap/include/ext4_fiemap.h`; `gemini_fiemap_extent` → `Ext4FiemapExtent` |

Legacy still compiles via the R1 shim; Gen-B residue redirected to
the new locations.

### R3 — `memory/` implementation + smoke   **(BOTTOM-UP STARTS HERE)**

Goal: stand up `IMemorySubsystem` so layers above can hand it a
buffer and get back a `MemoryRegion*`.

- Implement `tutti::HostDeviceMemorySubsystem` (working title) in
  `memory/src/`:
  - `allocate_host` / `allocate_device` / `allocate_*` —
    plain `malloc` / `cudaMalloc` paths.
  - `register_host` / `register_device` — record a `MemoryRegion`
    with `RegistrationMetadata` filled lazily.
  - `prepare_nvme_dma` — call libnvm's `nvm_dma_map_host` /
    `nvm_dma_map_device` to populate `dma_ioaddrs`.  This is the
    only place memory talks to libnvm.
  - `register_external` for `APP_MANAGED` first; `CUDA_IPC` /
    `HOST_SHM` / `HOST_FD_MAP` are stubbed (`return nullptr`)
    until R6 if we need them.
- Move `filesystems/ext4/libgeminifs/memory.cpp` (host malloc /
  alignment helpers) into `memory/src/` and refit symbol names.
- **Smoke**: `memory/test/memory_smoke.cu`.
  - allocate_host + register + free
  - allocate_device + register + free
  - register_external(APP_MANAGED) on a caller-`cudaMalloc`'d
    buffer
  - prepare_nvme_dma against an `nvm_ctrl_t*` produced by the
    libnvm role smoke pattern (parent forks before CUDA, child
    runs subsystem)
  - lookup by HOST_PTR / DEVICE_PTR / REGION_ID
- After R3, `memory/` has both header (Gen-A) and implementation
  with its own validated test path.  Nothing above has to change.

### R4 — `device_manager/` rebuilt + smoke

Goal: stand up `IDeviceRegistry` and `ILeaseManager` with **two**
concrete `IDeviceRegistry` implementations (per your request).

- `LocalNvmeNvmeServiceRegistry` (or just `NvmeServiceBackedRegistry`)
  — talks to the NVMeService daemon via `nvmeservice_client`, turns
  `ListDevices()` responses into `tutti::Device`s.  Lease lifecycle
  drives `ILeaseManager` via `Connect` / `Heartbeat` / `Disconnect`.
- `LocalNvmeDirectRegistry` — GPU-owned bootstrap path: directly
  calls `nvm_controller_init_b3` / `nvm_create_group` /
  `nvm_add_user_queue` itself.  No daemon, no RPC.  Suitable for
  single-tenant testing / micro-benchmarks.
- Pick which one the coordinator uses via
  `DeviceManagerMode::IN_PROCESS` vs `SERVICE_CLIENT` in
  `RuntimeConfig` (already there in Gen-A).
- Rebuild what was `BlockDeviceManager` as `LocalNvmeBlockDevice`
  (the per-`Device` concrete state behind both registries).
  Move `block_address_translator.cuh` here; drop its include of
  `geminifs.h` once R2.1 + R2.5 are done.
- **Smoke**: `device_manager/test/registry_smoke.cu` — both
  registry impls bring up one NVMe + enumerate + acquire a lease
  + release.  Validates that both bootstrap modes produce a
  consistent `Device` shape.

After R4, the bottom two layers (memory + device_manager) are in
their final canonical form with their own smoke tests.  Everything
above is still the unfinished mix of Gen-A headers + Gen-B residue,
but the foundation it'll need is now solid.

### R5 — Pause for `FilesystemDecomposition.md`

- No code moves.  Before tackling `io_engine/` cleanup we draft a
  second decomposition map covering the `filesystems/` tree
  (`ext4_fiemap/`, `tutti_layout/` for legacy on-device layout,
  `IFilesystem` SPI).  Saves us guessing what `io_engine/` should
  consume.

### R6 — `io_engine/` becomes pure header SPI

- `git mv io_engine/include/block_io_engine.cuh
       io_engine/include/block_io_channel.cuh
       io_engine/include/nvme_queue_scheduler.cuh`  →
  `backends/local/nvme/src/`.  These are guts behind
  `IBackendProvider::launch_batch_gpu_stream()` for the libnvm
  backend, not public SPI.
- `git mv io_engine/src/block_io_engine.cu` → same.
- Write the actual `IBackendProvider` implementation
  (`backends/local/nvme/src/local_nvme_backend.cu`) that exposes
  the relocated kernels through the `tutti::` SPI.
- `io_engine/src/` ends up empty; `io_engine/` is header-only.
- **Smoke**: `io_engine/test/spi_smoke.cu` — instantiate the libnvm
  backend, feed it a `BufferDescriptorBatch` + `IORequestBatch`,
  validate end-to-end through both `launch_batch_gpu_stream` and
  `submit_batch_cpu_sync`.

### R7 — `runtime/` → `coordinator/` + class rename

By now memory / device_manager / io_engine are all settled.

- `git mv runtime/ coordinator/`.
- Rename `api/include/runtime.h` → `api/include/coordinator.h`.
- `class Runtime` → `class Coordinator`,
  `struct RuntimeConfig` → `struct CoordinatorConfig`,
  `tutti::Runtime` → `tutti::Coordinator` everywhere.
- Drop the R1 include shim (nothing references
  `filesystems/ext4/libgeminifs/include/` any more by this point).
- Write the actual `Coordinator::initialize` / `submit_batch` body
  on top of the now-stable lower layers.
- **Smoke**: `coordinator/test/coordinator_smoke.cu` — end-to-end:
  Coordinator boots, picks libnvm backend via direct registry, does
  one read + one write via `submit_batch(BATCH_GPU_STREAM)`,
  shuts down clean.

### R8 — Delete the legacy tree + this file

- `rm -rf filesystems/ext4/libgeminifs/`
- `rm -rf doc/refactor/`
- `Todolist.md` / `Roadmap.md` / `README.md` cleanup pass.
- Final commit: `cleanup: drop legacy libgeminifs (R8)`.

---

## 5. Decisions already made (was "open questions")

These were settled at R0 time:

- **R7**: rename **both** the directory (`runtime/` →
  `coordinator/`) and the class (`Runtime` → `Coordinator`,
  `RuntimeConfig` → `CoordinatorConfig`).  No half-rename.
- **R4**: ship **two** `IDeviceRegistry` impls, not one:
  - `NvmeServiceBackedRegistry` for `DeviceManagerMode::SERVICE_CLIENT`.
  - `LocalNvmeDirectRegistry` for `DeviceManagerMode::IN_PROCESS`.
  Both produce identically-shaped `tutti::Device` instances; the
  backend code below the registry doesn't know which mode it's
  running in.
- **Refactor order**: bottom-up.  `memory/` (R3) and
  `device_manager/` (R4) land first with smokes, then we pause for
  `FilesystemDecomposition.md` (R5), then `io_engine/` cleanup (R6),
  then the coordinator rename at the top (R7).  Reason: changing
  the bottom is cheap because nothing in the new tree depends on
  it yet; changing the top first forces sed-style rewrites every
  time a lower layer moves.

Still open, deferred to their respective steps:

- **R3**: exact name of the `IMemorySubsystem` impl class — try
  `HostDeviceMemorySubsystem` first; rename if a clearer one
  surfaces while writing the smoke.
- **R4**: how `LocalNvmeDirectRegistry` learns about the kernel
  module being loaded — probably stat `/dev/snvm_control` at
  construction, fail fast.
- **R5**: do we keep `filesystems/ext4_fiemap/` as one directory or
  split it into a header-only "FIEMAP ABI" sub-target?  Defer to
  the dedicated decomposition doc.

---

## 6. Per-step "what counts as DONE" checklist

To be filled in as we land each step.

- [x] R0 — this file lands, `.bak` deleted, `json.h` deleted.
- [ ] R1 — `make` doesn't touch legacy `.cu`/`.cpp` files; legacy
      `EXCLUDE_FROM_ALL`; include shim limited to `geminifs.h` /
      `geminifs_helper.h`.
- [ ] R2.1 — `Ext4BlockFileHeader` exists, on-disk magic unchanged.
- [ ] R2.2 — `nvme_prp_descriptor.h` exists, `backend_provider.h`
      forward decl resolves there.
- [ ] R2.3 — `PersistentFileLog` exists.
- [ ] R2.4 — `cuda_helpers.cuh` + `tutti_macros.h` exist.
- [ ] R2.5 — `ext4_fiemap.h` exists, names converted.
- [ ] R3 — `IMemorySubsystem` impl in `memory/src/` + green
      `memory_smoke` (host alloc, device alloc, register external,
      prepare_nvme_dma against a libnvm ctrl).
- [ ] R4 — both `LocalNvmeDirectRegistry` and
      `NvmeServiceBackedRegistry` exist; green `registry_smoke`
      from each; `LocalNvmeBlockDevice` replaces `BlockDeviceManager`.
- [ ] R5 — `doc/refactor/FilesystemDecomposition.md` lands.
- [ ] R6 — `io_engine/src/` empty; libnvm `IBackendProvider` impl
      in `backends/local/nvme/src/`; green `spi_smoke`.
- [ ] R7 — `coordinator/` exists, `runtime/` gone; class names
      renamed; R1 shim deleted; green `coordinator_smoke`
      (end-to-end one read + one write).
- [ ] R8 — `filesystems/ext4/libgeminifs/` and `doc/refactor/`
      deleted.
