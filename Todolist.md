# Todolist

This file tracks the current active work for the repository.

It is not a long-term vision document.
Use [`Roadmap.md`](Roadmap.md) for versioned architecture and roadmap planning.

## Current Focus

- Refactor the project from a GPU-file-oriented implementation into a `Unified Storage Runtime`
- Freeze `v0.1` architecture boundaries before major code movement
- Keep interface and directory changes reviewable and explicit

## Active Tasks

- [x] Finalize the `v0.1` target top-level directory structure
      (Roadmap.md `v0.1 Recommended Directory Direction` updated with the
      backend × filesystem two-axis split and the runtime / device_manager
      noun-vs-service rule)
- [ ] Finalize the `v0.1` top-level public API boundaries
      (Slice 4 — `api/` Runtime entry, RuntimeConfig, error model — pending)
- [x] Define the core runtime object model
      (`runtime/include/`: `Device`, `IOBuffer`, `BatchRequest`,
      `StorageTarget`, `CapabilitySet`, `Lease` value type)
- [x] Define the standalone memory subsystem API
      (`memory/include/`: `MemoryKind`, `MemoryRegion`, `IMemorySubsystem`)
- [x] Define the device manager and IO engine boundary
      (`device_manager/include/`: `IDeviceRegistry`, `ILeaseManager`;
      `io_engine/include/`: `IBackendProvider`, `IQueueProvider`,
      `BufferDescriptor`, `IORequest`, batches)
- [ ] Define the backend SPI for the first `local_nvme` backend
      (SPI shape exists; `local_nvme` refactor to implement
      `IBackendProvider` is pending)
- [ ] Define how `LMCache` and `Mooncake` adapters will attach to the runtime
- [ ] Unify the configuration strategy and remove split config semantics over time
- [ ] Define deployment flow for the modified NVMe kernel module
- [ ] Define Linux version compatibility policy for the kernel module
- [ ] Define startup ordering for kernel module, service, and runtime attach flow
- [ ] Add AI-consumable architecture docs under `doc/`
      (architecture and backend-spi docs landed; per-subsystem AI docs pending)

## NVMeService Rewrite — Remaining Work

Control-plane code, state, server, client, and `libnvm` shared-resource
reconstruction are in place. Outstanding items to make it buildable and
runnable end-to-end:

- [x] `backends/local/NVMeService/examples/nvmeservice_daemon.cpp` — daemon
      entry point (parse `sys_config.yaml`, construct `ServiceState`,
      start gRPC server, start reaper, wait for SIGINT)
- [x] `backends/local/NVMeService/examples/nvmeservice_client.cpp` — smoke
      test that connects, lists devices, allocates, sleeps, releases
- [x] `backends/local/NVMeService/examples/sys_config.yaml` — example config
      matching the new YAML schema (gpus / nvmes with `queue_groups` for
      per-NVMe multi-GPU queue split)
- [x] `backends/local/NVMeService/examples/CMakeLists.txt` — build the two
      example executables against the new `nvmeservice` library
- [x] Update root `CMakeLists.txt` NVMeService section: compile new file
      layout (`nvmeservice_config.cpp`, `nvmeservice_state.cu`,
      `nvmeservice_server.cpp`, `nvmeservice_client.cpp`) and the new
      `backends/local/nvme/libnvm/src/shared_ctrl.cu`; version-aware
      protoc detection patched (commit `df4f2c8`)
- [ ] Add `BlockDeviceManager` second constructor that takes a
      `std::shared_ptr<Controller>` plus mount path (skips own controller
      init so it can consume a shared Controller from NVMeService)
      — folded into Slice 4
- [ ] Verify CUDA IPC works for PRP memory; if not, wire client-side PRP
      allocation fallback (the server already honours `has_prp=false`)
- [ ] Verify `cudaHostRegister(BAR0, cudaHostRegisterIoMemory)` works in a
      non-daemon process (same SNVMe device was already mmapped by daemon)
- [ ] End-to-end smoke: daemon + one client, run a trivial read/write via
      `BlockDeviceManager` built on the shared `Controller`

## SNVMe Queue-Budget Tuning — Phase 2 / 3 (kernel ABI landed; upper layers pending)

Phase 1 (kernel ABI + libnvm wrappers + smoke verification) is in.
`NVM_SET_IOQ_NUM` now carries a `struct nvm_ioctl_setup` whose
`cap_kernel_ioq` and `groups[]` fields let userspace tell snvme how to
slice an MSI-X-limited NVMe between kernel-IOQs and GPU-direct user
IOQs.  Verified on HGX H20 + Intel DC SSD (MSI-X=136 vs 192 vCPUs):
smoke now reports `nr_user_q=1` and dmesg shows `queue split: kernel=31
user=1` instead of the silent fallback to `dma_alloc_coherent`.

The two upper layers were intentionally left for a follow-up so that
Phase 1 could land cleanly:

- [ ] **Phase 2 — NVMeService daemon: parse `queue_setup` from
      `sys_config.yaml` and feed it to snvme.**
      Add a `queue_setup` block to the per-NVMe schema in
      `backends/local/NVMeService/src/nvmeservice_config.{h,cpp}` with
      these fields (one-to-one mapping to `struct nvm_ioctl_setup`):
        `kernel_ioq_cap` (uint, required)
        `user_ioq_total` (uint, required, must equal sum of group counts)
        `on_host`        (bool, default false)
        `nr_write`       (uint, default 0 = use module param)
        `nr_poll`        (uint, default 0 = use module param)
        `queue_groups: [{ owner_id|gpu_id: uint, count: uint }]`
                         (size <= 8; sum(count) must equal `user_ioq_total`)
      Validate at daemon-startup time:
        - `kernel_ioq_cap + user_ioq_total <= total_queues`
        - `sum(groups[].count) == user_ioq_total`
        - `groups.size() <= NVM_MAX_QUEUE_GROUPS`
      Then in `nvmeservice_state.cu` (or whichever class owns the
      per-NVMe `Controller`), call `nvm_queue_setup(ctrl, &setup)`
      from libnvm BEFORE the existing `NVM_MAP_*`/`NVM_SET_SHARE_REG`
      sequence.  Existing daemon callers that don't set `queue_setup`
      should keep working (kernel falls back to `cap_kernel_ioq=0` =
      `num_possible_cpus()` default).

- [ ] **Phase 3 — Operator documentation.**
      a. `sys_config.yaml`: add a worked example of `queue_setup` under
         the `nvmes:` section, with comments explaining when to set
         `kernel_ioq_cap` (= "the controller's MSI-X count is below
         host CPU count" rule of thumb) and the per-GPU split.
      b. `backends/local/kernel_modules/PORTING.md` §8.1: add a row
         to the troubleshooting cheat sheet for the dmesg signature
         `queue squeeze: kernel=N user=M (controller granted ...)`,
         pointing operators at `cap_kernel_ioq` as the tunable.
      c. `README.md`: add a short "Queue budget tuning" subsection
         under the existing kernel-module notes that links to the
         sys_config.yaml example and the PORTING.md cheat-sheet row.

- [ ] **(Optional) snvme_smoke_gpu `--cap-kernel N` flag.**
      Currently the smoke binaries hard-code `cap_kernel_ioq = 32`,
      which forces case B (split) on a generous controller.  A
      `--cap-kernel N` CLI flag would let regression runs explicitly
      exercise either case A2 (squeeze, `N` >> controller MSI-X) or
      case A1 (full fallback, `N=0` AND user_cq > grant) without
      recompiling.  Low priority — case B coverage is what production
      cares about, A1/A2 are review gates.

## Discussion Required Before Major Refactor

- [x] Decide the future runtime/product name — `Tutti`, recorded in
      `Roadmap.md` and `README.md`
- [x] Decide the final naming style for runtime-facing APIs — `tutti::`
      C++ namespace, applied to all new headers under
      `runtime/`, `memory/`, `device_manager/`, `io_engine/`
- [x] Decide the initial target directory migration plan — backend ×
      filesystem two-axis split documented in `Roadmap.md`
- [ ] Decide what must remain temporarily compatible during the first refactor wave

## Known Bugs To Track

- [ ] `GPU file persistence` correctness/stability issue

## Current Non-Goals

- [ ] Do not introduce `cooperative submit` in `v0.1`
- [ ] Do not treat simultaneous CPU/GPU access semantics as a stable runtime contract
- [ ] Do not hardcode framework-specific types into the core runtime

## Collaboration Notes

- Update this file when current priorities change
- Keep completed items visible until the next version snapshot is archived
- If a new active version is created, move the old roadmap snapshot into [`doc/history/`](doc/history/)
