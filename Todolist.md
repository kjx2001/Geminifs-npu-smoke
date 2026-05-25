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
      test that connects, lists devices, allocates, sleeps, releases.
      Now also runs a hand-off probe: prints `alloc->mount_path`, lists
      it (verifies the daemon-installed GPU-view symlink resolves from
      this process), and walks the first few imported QueuePairs to
      confirm shared SQ/CQ/PRP IPC pointers and doorbell GPU VAs are
      live before the heartbeat hold.
- [x] `sys_config.yaml` — single canonical config at the repo root
      (the per-examples copy was removed; `examples/CMakeLists.txt` now
      stages the root file into `build/bin/`). Schema annotated with
      QueuePair-unit conventions; `queue_setup` worked example present.
- [x] `backends/local/NVMeService/examples/CMakeLists.txt` — build the two
      example executables against the new `nvmeservice` library
- [x] Update root `CMakeLists.txt` NVMeService section: compile new file
      layout (`nvmeservice_config.cpp`, `nvmeservice_state.cu`,
      `nvmeservice_server.cpp`, `nvmeservice_client.cpp`) and the new
      `backends/local/nvme/libnvm/src/shared_ctrl.cu`; version-aware
      protoc detection patched (commit `df4f2c8`)
- [x] `BlockDeviceManager` second constructor that takes a
      `std::shared_ptr<Controller>` plus mount path (skips own controller
      init so it can consume a shared Controller from NVMeService).
      Lives in `device_manager/include/block_device_manager.cuh` as
      `BlockDeviceManager(const ControllerPtr&, std::unique_ptr<FileManager>, const std::string&)`.
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

- [x] **Phase 2 — NVMeService daemon: parse `queue_setup` from
      `sys_config.yaml` and feed it to snvme.**
      Added `QueueSetup` (`kernel_ioq_cap` / `on_host` / `nr_write` /
      `nr_poll`) plus `queue_groups` to `nvmeservice_config.{h,cpp}`.
      `validate_config` enforces, in QueuePair units, the local
      invariant `Σqueue_groups[].count + queue_setup.kernel_ioq_cap <=
      total_queues`, plus `len(queue_groups) <= NVM_MAX_QUEUE_GROUPS`
      and the `on_host=false` constraint. `nvmeservice_state.cu::
      init_device` translates the YAML block into `struct
      nvm_ioctl_setup` (per-group `count *= 2` for the kernel
      SQ+CQ-entry unit) and hands it to libnvm's new `Controller(...
      const nvm_ioctl_setup&)` ctor. libnvm internally calls
      `nvm_queue_setup(ctrl, &setup)` BEFORE `NVM_MAP_*` /
      `NVM_SET_SHARE_REG`, mirroring the smoke-gpu reference bring-up.
      Old single-GPU callers (GeminiFS, BlockDeviceManager) keep the
      legacy 7-arg ctor signature — internally synthesises a
      one-group setup with `cap_kernel_ioq=0`, byte-equivalent to the
      pre-Phase-2 `nvm_queue_set()` submit. The historical hard cap
      `max_queue=75` in `Controller::init_queues` was removed; the
      caller's QueuePair count is the source of truth (only
      `MAX_QUEUES=1024` structural ceiling left).

- [x] **Phase 3 — Operator documentation.**
      a. `sys_config.yaml`: full `queue_setup` worked example with
         field-by-field comments; QueuePair-unit conventions stated
         throughout. The default profile is now NUMA-0 single-NVMe +
         single-GPU smoke (with TODO markers for the host-specific
         BDF / GPU id), and the dual-GPU profile is preserved as a
         commented reference at the bottom.
      b. `backends/local/kernel_modules/PORTING.md` §8.1: row added
         for the `queue squeeze: kernel=N user=M (controller granted
         ...)` dmesg signature, pointing operators at
         `queue_setup.kernel_ioq_cap` as the tunable.
      c. `README.md`: "Queue budget tuning" subsection added under the
         existing kernel-module notes, with a worked YAML example,
         the MSI-X-vs-vCPU rule of thumb, and the dmesg signature for
         the silent fallback.

## Share-mode queue recycle gap (NVMeService)

### Problem

`NVMeService::release_range` (and the reaper path) marks a queue
range as free in `DeviceQueueGroup::queue_allocated`, but does NOT
reset the underlying NVMe queue state. The next client that
allocates the same range will:

- Inherit the controller-side `SQHD/SQT/CQH/CQT/phase` from the
  previous tenant (no Delete/Create I/O SQ/CQ has been issued).
- See stale SQEs that the previous client wrote into the SQ ring
  (cudaMalloc'd SQ memory is never re-zeroed on release).
- See stale CQEs in the CQ ring with the previous tenant's phase
  bit -- the new client's phase tracking starts at phase=1 and
  will mis-classify these as fresh completions, or never poll the
  real new ones because the CID never matches.

Symptoms on hardware: silent data corruption (controller fetches a
stale SQE from an unexpected SQT position and executes its LBA /
opcode) or hangs (phase tag mismatch -> `cq_poll` busy-loops
forever).

`init_gpu_specific_struct` re-allocates the GPU coordination state
(tickets / cid bitmap / marks / pos_locks) per client, so cross-
process state pollution is limited to controller-side and DMA-ring
state. The kernel-side / GPU-coord-side bitmap is fine.

### NVMe spec gives a clean fix

NVMe 1.4 §4.1 + §5.4 / §5.5: `Delete I/O SQ` (opcode 0x00), `Delete
I/O CQ` (opcode 0x04), `Create I/O CQ` (opcode 0x05), `Create I/O
SQ` (opcode 0x01) are per-queue admin commands. They:

- Affect only the named qid.
- Do NOT require Controller Reset (CC.EN=0) -- the rest of the
  controller (kernel-side queues, other user queues, other clients
  on the same NVMe) keeps running.
- Reset `SQHD/SQT/CQH/CQT/phase` to spec defaults on the next
  Create.
- Allow PRP1 to point at the same physical pages -- userspace can
  keep the existing SQ/CQ DMA mappings and just re-issue
  Create with the same addresses.

snvme already has the kernel-side helpers (`adapter_alloc_cq_user`,
`adapter_alloc_sq_user`, `adapter_delete_cq`, `adapter_delete_sq`,
`snvme_disable_user_io_queues`) but only invokes them on probe /
disable -- there is NO userspace ioctl that triggers per-queue
recycle.

### Mitigation (until kernel changes land)

**Today's NVMeService MUST NOT actually recycle queues.** The
reaper detects dead clients and frees the lease metadata, but the
queue range should stay reserved (excluded from future
allocations) until the daemon restarts. This avoids the silent
corruption case while we do the kernel work properly.

Required follow-ups:
- [ ] `ServiceState::release_range` / `reaper_loop` need a "do not
      return to pool" path: mark the range as permanently consumed
      for this daemon lifetime, and log a clear warning that
      queue pool capacity has dropped.
- [ ] `NVMeService.md` must document this explicit limitation
      (operators may run out of queues if clients churn).

### Real fix: kernel + libnvm + daemon

Plan, executed in this order so each step is independently
testable:

1. [ ] **kernel: `NVM_RAW_ADMIN_CMD` ioctl** -- a generic 64-byte
       admin SQE forwarder. Userspace fills a `struct nvme_command`
       and the kernel runs `snvme_submit_sync_cmd(dev->ctrl.admin_q,
       &c, NULL, 0)`. Returns the CQE status. This is the building
       block for everything else (Delete/Create I/O, Abort, vendor
       commands). Touches:
        - `snvme-5.4.241-1-tlinux4-0017/pci.c` (new ioctl handler)
        - `backends/local/nvme/libnvm/include/ioctl.h` (UAPI)
        - PORTING.md (note added)

2. [ ] **smoke T2 (test/snvme_smoke_recycle.c)**: drive a real
       Delete I/O SQ -> Delete I/O CQ -> Create I/O CQ -> Create
       I/O SQ sequence via `NVM_RAW_ADMIN_CMD` against a bound
       controller. Verify (a) the admin commands succeed and (b)
       a subsequent NVMe read on the recycled queue completes
       correctly. **This is the empirical proof that NVMe spec
       per-queue reset works on our target SSD firmware.**

3. [ ] **smoke T1 (test/snvme_smoke_recycle.c)** -- optional
       counter-test: skip the Delete/Create dance, just re-use
       the queue with fresh host-side state, run a read, and
       expect either an NVMe error completion or a hang. This
       documents what the current NVMeService bug actually
       looks like on hardware. Run BEFORE T2 in the same binary
       so the report shows broken-then-fixed.

4. [ ] **kernel: `NVM_RECYCLE_USER_QUEUE` ioctl** -- convenience
       wrapper that runs the four-command sequence + does the
       SQ/CQ ring cudaMemset(0) + reset `nvme_dev`'s per-qid
       bookkeeping. Optional; daemon can equally call T2's path
       four times. Defer until M1+T2 are green.

5. [ ] **libnvm**: implement the stub declarations in
       `nvm_admin.h` (`nvm_admin_sq_create`, `nvm_admin_sq_delete`,
       `nvm_admin_cq_create`, `nvm_admin_cq_delete`,
       `nvm_admin_abort`). Each is a thin
       `ioctl(NVM_RAW_ADMIN_CMD, struct nvme_command)` wrapper.
       Add a single `Controller::recycle_queue(uint16_t qid)`
       helper that runs the four-command sequence + host-side
       resets + ring cudaMemset.

6. [ ] **NVMeService daemon**: invoke
       `controller->recycle_queue(qid)` for every queue in the
       just-released range, BEFORE returning the range to the
       pool. Today's "permanent consumption" mitigation is
       removed at this step.

7. [ ] **NVMeService reaper test (`05_reaper.sh`)**: extend to
       verify that the *recycled* queue range is reusable -- run
       client B after A dies, check B's IOs complete (currently
       the test only checks that `avail` returns to its initial
       count, which the mitigation above already satisfies
       through a different mechanism).

### Related independent items

- [ ] `init_gpu_specific_struct` (`backends/local/nvme/libnvm/include/queue.h`)
      should consolidate the 5 separate `BufferPtr` allocations
      (`sq_tickets`, `sq_tail_mark`, `sq_cid`, `cq_head_mark`,
      `cq_pos_locks`) into a single contiguous cudaMalloc with
      segmented offsets, and `cudaMemset(0)` it explicitly.
      Rationale: (a) reduce GPU heap fragmentation (sq_cid alone
      is 2 MiB per QP; today 5 allocations per QP); (b) cudaMalloc
      does NOT guarantee zero-initialised memory, and the cid
      bitmap MUST start at 0 for `get_cid()`'s `fetch_or(LOCKED)`
      lock-claim protocol to work. Pure local refactor, no API
      change. Applies to single-GPU / share / future local modes.

- [ ] `nvm_queue_t::qs_log2` is set via `(uint32_t)std::log2(qs)`
      which silently loses precision when `qs` is not a power of
      two (NVMe MQES is `MaxQueueEntries - 1`, often e.g. 1023 ->
      qs=1024 is fine but a controller reporting odd MQES would
      break the lock-free ring math). Switch to
      `__builtin_ctzll(qs)` with an explicit power-of-two assert,
      same change in `shared_ctrl.cu`.

- [ ] `device.cpp:468` `cudaHostRegister(BAR0, ..., IoMemory)`
      lacks `cudaHostRegisterPortable`. Single-process multi-GPU
      smoke (local-mode multi-GPU Controller, future Todolist
      item) will hit this -- the doorbell GPU VA is only valid on
      the GPU that was current at register time.

- [ ] **local-mode multi-GPU ctor** -- new
      `Controller::MultiGpuMode::LOCAL` variant of
      `init_queues_multi_gpu_*` that does resolve doorbell GPU
      VAs (per-queue `cudaHostGetDevicePointer` after
      `cudaSetDevice(per_queue_dev[i])`) and cudaMalloc's d_qps
      on `deviceId`. Used for single-process multi-GPU smoke
      tests. SHARE variant (current default) stays unchanged.

- [ ] **per-group d_qps** -- when the local-mode multi-GPU
      smoke shows cross-GPU `d_qps[queue]` access becomes a
      bottleneck (P2P / UVA fallback), split `d_qps` into
      `d_qps_per_group[NVM_MAX_QUEUE_GROUPS]`, each cudaMalloc'd
      on the matching `groups[g].owner_id`. Today's single-GPU
      consumers (GeminiFS, BlockDeviceManager) keep reading
      `d_qps == d_qps_per_group[0]`. Multi-GPU kernels read the
      group-local array.

- [ ] **NVM_ADD_USER_QUEUE map-type discrimination** -- today the
      ADD_USER_QUEUE handler resolves `(sq_vaddr, cq_vaddr)` to
      maps by `vaddr & PAGE_MASK` lookup against `g->maps`, with
      NO type tag on the map.  Means: if userspace accidentally
      passes a data-buffer vaddr where it meant an SQ/CQ ring
      vaddr, the kernel will happily Create I/O SQ with PRP1 =
      that data buffer's dma_addr; the controller then reads
      garbage as SQEs and the failure mode is silent corruption /
      controller hang rather than a clean -EINVAL.
      Fix sketch: add `uint8_t map_type` to `struct map`
      (RING_SQ / RING_CQ / DATA), set at NVM_MAP_HOST_MEMORY time
      via a new flag in `struct nvm_ioctl_map`; ADD_USER_QUEUE
      then rejects mismatched types up front.

- [x] **NVM_ADD_USER_QUEUE vaddr-mask alignment hazard for GPU maps.**
      pci.c:6279 used to mask with the host PAGE_MASK (4 KiB)
      regardless of whether the map was created via
      `NVM_MAP_HOST_MEMORY` (vaddr aligned to PAGE_SIZE) or
      `NVM_MAP_DEVICE_MEMORY` (vaddr aligned to GPU_PAGE_SIZE,
      64 KiB).  Host smoke worked by accident because page-aligned
      stayed page-aligned; GPU smoke would have miss-matched.
      Fixed by deriving the lookup mask from `cursor->page_size`
      so both routings work.

- [ ] **Decouple data-buffer maps from the queue group lifecycle.**
      Right now the only well-supported NVM_MAP_HOST_MEMORY mode
      is "bind to a queue group" (group_id != 0).  The legacy
      group_id == 0 path exists but had a sibling-fd reaping bug
      (fixed by limiting purge_by_owner to group_id == 0) and is
      best treated as deprecated.  For data buffers that outlive
      the queues they're submitted on (typical: client allocates
      a 4 MiB DMA pool once, sends a million IOs through it,
      eventually destroys the queue group) we want a third mode:
      "bind to fd, NOT to a queue group".
      Three implementation options to evaluate:
        a) Reserve a sentinel group_id (e.g. UINT32_MAX) meaning
           "fd-scoped".  Smallest ABI delta.
        b) Add a per-fd implicit default group, allocated at
           open(), destroyed at close().  Cleanest semantics; data
           buffers go there by default if user passes group_id=0.
        c) Add a `map_kind` field to struct nvm_ioctl_map
           (RING / DATA), and route DATA maps to a per-fd list
           regardless of group_id.  Pairs naturally with the
           map-type discrimination Todolist item above.
      Smoke note: snvme_smoke_io.c currently registers data
      buffers under the same queue group as the rings.  This works
      but couples lifecycles -- it's a smoke-correctness shortcut,
      not a recommendation for NVMeService production use.

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
