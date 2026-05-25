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
- [ ] Add AI-consumable architecture docs under `doc/`
      (architecture and backend-spi docs landed; per-subsystem AI docs pending)

## SNVMe — Map-type discrimination + fd-scoped data buffer maps

This is the next kernel-side work item. The two changes are
intentionally bundled because they share a single ABI break (one
new flag in `struct nvm_ioctl_map`) and resolve two related
hazards at once:

**Hazard 1 — map-type discrimination.** Today
`NVM_ADD_USER_QUEUE` resolves `(sq_vaddr, cq_vaddr)` against
`g->maps` by vaddr lookup with no type tag on the map. If
userspace accidentally passes a data-buffer vaddr where it meant
an SQ/CQ ring vaddr, the kernel happily issues `Create I/O SQ`
with PRP1 = that data buffer's dma_addr; the controller then
reads garbage as SQEs and the failure mode is silent corruption /
controller hang rather than a clean `-EINVAL`.

**Hazard 2 — data-buffer lifecycle coupling.** The only well-
supported mode today binds every map to a queue group
(`group_id != 0`). The `group_id == 0` legacy path had a sibling-
fd reaping bug (fixed by limiting `purge_by_owner` to
`group_id == 0`) and is best treated as deprecated. For data
buffers that outlive the queues they're submitted on (typical:
client allocates a 4 MiB DMA pool once, sends a million IOs
through it, eventually destroys the queue group) we want a
third mode: "bind to fd, NOT to a queue group".

### Plan

- [ ] **Kernel ABI.** Add a `map_kind` field to
      `struct nvm_ioctl_map` (values: `RING_SQ`, `RING_CQ`,
      `DATA`). `RING_SQ` / `RING_CQ` maps MUST carry a non-zero
      `group_id` and link onto `g->maps` as today. `DATA` maps
      ignore `group_id` and link onto a per-fd `data_maps` list
      instead, so they survive `NVM_DESTROY_QUEUE_GROUP` and only
      get released on fd close.
- [ ] **Kernel: tag every `struct map` with its kind** at
      `NVM_MAP_HOST_MEMORY` / `NVM_MAP_DEVICE_MEMORY` time.
- [ ] **Kernel: `NVM_ADD_USER_QUEUE` rejects mismatched kinds**
      up front with `-EINVAL`. The vaddr lookup additionally
      requires `map_kind == RING_SQ` for `pairs[i].sq_vaddr` and
      `map_kind == RING_CQ` for `pairs[i].cq_vaddr`.
- [ ] **Kernel: per-fd `data_maps` list** with its own purge path
      on `snvm_dev_release`. `NVM_DESTROY_QUEUE_GROUP` MUST NOT
      walk this list.
- [ ] **libnvm wrappers**: extend `nvm_dma_map_*` family to take
      an explicit kind argument; add helper variants for the
      three common cases.
- [ ] **smoke**: extend `snvme_smoke_io.c` so data buffers are
      registered as `DATA` (fd-scoped), rings as `RING_SQ` /
      `RING_CQ` (group-scoped). Add a phase that destroys the
      queue group while data buffers are still mapped, then
      re-creates the group + new rings + reuses the same data
      buffer maps for fresh IO. `snvme_smoke_gpu.cu --rounds N`
      should also exercise this: the rounds loop currently
      tears down everything per round; switch it to
      "tear down rings, keep data maps" for at least half the
      rounds.
- [ ] **PORTING.md update**: §4.3.1 needs the new flag, §5.1
      needs the "DATA maps survive group destroy" rule, §7.3.1
      gets a new trap entry "data map registered with
      `RING_SQ` kind silently passes ADD_USER_QUEUE".

## NVMeService Rewrite — Remaining Work

Control-plane code, state, server, client, and `libnvm` shared-resource
reconstruction are in place. Outstanding items to make it buildable
and runnable end-to-end:

- [x] daemon entry point, client smoke, root `sys_config.yaml`,
      example CMakeLists, root CMakeLists protoc patch,
      `BlockDeviceManager` shared-Controller ctor — all landed.
- [ ] Verify CUDA IPC works for PRP memory; if not, wire client-side PRP
      allocation fallback (the server already honours `has_prp=false`).
- [ ] Verify `cudaHostRegister(BAR0, cudaHostRegisterIoMemory)` works in a
      non-daemon process (same SNVMe device was already mmapped by daemon).
- [ ] End-to-end smoke: daemon + one client, run a trivial read/write via
      `BlockDeviceManager` built on the shared `Controller`.

### Reaper / dead-client policy (settled)

When the reaper detects a dead client (PID gone, or PID reused via
starttime mismatch), the daemon **destroys the dead client's entire
queue group** via `NVM_DESTROY_QUEUE_GROUP`, not by recycling
individual qids. This means:

- `NVM_DESTROY_QUEUE_GROUP` cascades through `Delete I/O SQ` +
  `Delete I/O CQ` + map purge for every queue / ring in the
  group, on the controller side. There is no stale SQHD/SQT/CQH/
  CQT/phase to inherit.
- The next client's `allocate()` request goes through the normal
  `NVM_CREATE_QUEUE_GROUP` + `NVM_ADD_USER_QUEUE` path and gets
  freshly-allocated qids from the user QID pool. **Client B
  never inherits client A's qids.**
- The "share-mode queue recycle gap" / `NVM_RECYCLE_USER_QUEUE`
  plan in earlier revisions of this file is therefore obsolete
  and has been removed.

Outstanding daemon work to realise this:

- [ ] **Daemon owns one queue group per allocation.**
      `ServiceState::allocate` should call
      `NVM_CREATE_QUEUE_GROUP` + `NVM_ADD_USER_QUEUE` for the
      requested QueuePair count, store the resulting `group_id`
      on the `Allocation` record, and return the QID range +
      doorbell offsets (or just the imported QueuePair handles)
      to the client.
- [ ] **`release_range` and reaper destroy the group.** Replace
      the current `queue_allocated[i] = false` flip in
      `ServiceState::release_range` with a
      `NVM_DESTROY_QUEUE_GROUP(allocation.group_id)` call. The
      `queue_allocated` bitmap stays as a soft accounting view
      for the gRPC `list_devices` response, but the source of
      truth becomes the kernel-side group + user QID pool.
- [ ] **Pool-vs-group accounting.** Decide whether
      `total_queues` in `DeviceState` reflects the pre-cap
      controller MSI-X grant (current) or the dynamic
      `start_cq_idx..max_user_qid` window reported by
      `NVM_GET_DEV_INFO`. The latter is more accurate now that
      groups can come and go at runtime; the schema in
      `sys_config.yaml`'s `queue_setup` block already implies
      this.

## SNVMe Queue-Budget Tuning — DONE

Phase 1 (kernel ABI + libnvm wrappers + smoke verification),
Phase 2 (NVMeService daemon parses `queue_setup` from
`sys_config.yaml` and feeds it to snvme via the new `Controller`
ctor), Phase 3 (operator documentation in `sys_config.yaml`,
`PORTING.md` §8.1, `README.md` "Queue budget tuning") all
landed and verified on HGX H20 + Intel DC SSD (MSI-X=136 vs 192
vCPUs).  The historical `max_queue=75` hard cap in
`Controller::init_queues` was removed; only the structural
`MAX_QUEUES=1024` ceiling remains.

## SNVMe B1/B2/B3 — DONE

Per-fd queue groups, GPU-resident SQ/CQ rings via
`NVM_MAP_DEVICE_MEMORY`, dynamic `NVM_ADD_USER_QUEUE` /
`NVM_DESTROY_QUEUE_GROUP`, and `NVM_SET_KERNEL_IOQ_CAP` cap-only
path all landed and verified end-to-end:

- `snvme_smoke_qgroup` (B1), `snvme_smoke_addq` (B3 host),
  `snvme_smoke_recycle` (B4 raw admin) — all green.
- `snvme_smoke_io` — 23 phases / 200 IOs / 768 KiB byte-by-byte
  verified across PRP1 / PRP1+PRP2 / PRP_List / SGL (auto-skip)
  + SQ-tail-wrap.
- `snvme_smoke_gpu --rounds 4` — 56 steps / 4 rounds of full
  alloc/free of (queue group + GPU rings + GPU data + PRP_List),
  same QIDs (37/38) and doorbell offsets recycled every round.
- `PORTING.md` rewritten end-to-end for the B3 flow, line-number
  anchors removed in favour of symbol-name anchors.

The vaddr-mask hazard (host PAGE_MASK vs GPU page mask in
`NVM_ADD_USER_QUEUE`) was found and fixed during this round and
is documented as a §7.3.1 trap.

## SNVMe — Smaller follow-ups

- [ ] **`nvm_queue_t::qs_log2` power-of-two assert.** Today the
      log2 is computed via `(uint32_t)std::log2(qs)` which would
      silently lose precision if `qs` were not a power of two.
      In practice NVMe queue size is always a power of two
      (CAP.MQES + 1), so the failure is hypothetical, but the
      lack of a check makes that assumption invisible. Add an
      explicit `assert((qs & (qs - 1)) == 0)` on construction in
      both `queue.h` and `shared_ctrl.cu`. No runtime change.

- [ ] **`device.cpp` `cudaHostRegister(BAR0, ..., IoMemory)`
      should also pass `cudaHostRegisterPortable`.**
      Single-process multi-GPU smoke (the local-mode multi-GPU
      Controller item below) will hit this — the doorbell GPU
      VA is otherwise only valid on the GPU that was current at
      register time.

- [ ] **local-mode multi-GPU ctor.** A new
      `Controller::MultiGpuMode::LOCAL` variant of
      `init_queues_multi_gpu_*` that resolves doorbell GPU VAs
      per queue (`cudaHostGetDevicePointer` after
      `cudaSetDevice(per_queue_dev[i])`) and `cudaMalloc`s
      `d_qps` on `deviceId`. Used for single-process multi-GPU
      smoke tests. SHARE variant (current default) stays
      unchanged.

- [ ] **per-group `d_qps`.** When the local-mode multi-GPU smoke
      shows cross-GPU `d_qps[queue]` access becomes a bottleneck
      (P2P / UVA fallback), split `d_qps` into
      `d_qps_per_group[NVM_MAX_QUEUE_GROUPS]`, each `cudaMalloc`'d
      on the matching `groups[g].owner_id`. Today's single-GPU
      consumers (GeminiFS, BlockDeviceManager) keep reading
      `d_qps == d_qps_per_group[0]`. Multi-GPU kernels read the
      group-local array.

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
