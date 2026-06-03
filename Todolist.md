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

## NVMeService — DONE (L1 Commit 4b)

The daemon was rewritten to act as a *session broker* on top of the
B3/B6 kernel ABI, not a queue host or a quota ledger:

- `nvmeservice_state.h/.cu` no longer holds `std::shared_ptr<Controller>`
  per device.  It holds an `nvm_ctrl_t*` brought up via
  `nvm_controller_init_b3` (chrdev_create + cap + bind + probe).  The
  daemon does **not** maintain any per-GPU queue ledger -- the kernel's
  user QID pool is the single source of truth.
- YAML schema collapsed to "what the daemon really needs":
  `total_queues`, `queue_depth`, `queue_groups[].count`, the entire
  `queue_setup` block all removed.  What remains: `kernel_ioq_cap`
  (NVM_SET_KERNEL_IOQ_CAP hint), `allowed_gpus[]` (NUMA / PCIe-switch
  ACL), `queue_pool.{default,max}_per_client` (daemon-side guidance
  upper bound on Connect grants).
- `nvmeservice.proto` simplified: `AllocateQueues` -> `Connect`,
  `ReleaseQueues` -> `Disconnect`, all `cudaIpcMemHandle_t` /
  `QueueSharedMem` plumbing removed.  `DeviceInfo.quotas[]` replaced
  by `allowed_gpus[]` (cuda_device + GPU-view symlink path).
  `ConnectResponse.queue_quota` renamed `granted_queues` (policy
  guidance, no daemon-side accounting).
- Client library is a thin gRPC session holder: `client.connect()`
  returns a `Session` with the metadata; the caller drives libnvm
  themselves (`nvm_ctrl_attach_client` -> `nvm_create_group` ->
  ring/data maps -> `nvm_add_user_queue`).
- Reaper just drops stale lease records on PID-dead detection.  No
  kernel-side cleanup needed: the dead client's fd close already
  cascades through `snvm_dev_release` (B6 fd-scoped DATA invariant).
- `libnvm/src/shared_ctrl.cu` and `include/shared_ctrl.h` deleted.
  `Controller::is_shared`, `QueuePair::is_shared` and the IPC import
  ctor/dtor branches all removed.
- Examples rewritten: `nvmeservice_daemon` brings up the chrdev
  owner; `nvmeservice_client` (split into `.cpp` for gRPC and `.cu`
  for the IO smoke to keep protobuf headers out of nvcc) does
  Connect + attach_client + create_group + GPU IO + destroy_group +
  free_client + Disconnect.
- Legacy `examples/tests/*.sh` integration test scripts deleted; the
  two binaries (`nvmeservice_daemon` + `nvmeservice_client`) now
  cover the smoke surface end-to-end.

### Open follow-ups

- [ ] `disk.ns_id` is still left at 0 by `NVM_GET_DEV_INFO`; daemon
  currently propagates whatever YAML says. Long-term fix: extend the
  kernel ioctl to ship the namespace id + use it everywhere.
- [ ] `chrdev` minor reconstruction in `init_device` parses
  `disk.disk_name` (e.g. "snvme0n1" -> "0"). When the
  `pci_device_addr.domain` overload is fixed (entry below), pull the
  minor through that explicit out-param instead.


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
      `queue.h`. No runtime change.

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

- [ ] **Stop overloading `pci_device_addr.domain` as the
      CHRDEV_CREATE out-param.** `SNVM_CHRDEV_CREATE` /
      `SNVM_CHRDEV_REMOVE` currently `memset(dev_addr, 0, ...)` and
      stuff the assigned chrdev minor into `dev_addr->domain` on
      return (see `snvme-*/pci.c::snvm_chrdev_helper`), forcing
      userspace to do `snprintf("/dev/ssnvme%d", bdf.domain)` with
      a field that no longer means PCIe domain. Works today only
      because every NVMe card in scope sits at domain 0; breaks
      the moment a multi-domain host shows up, and is just
      confusing to read regardless. Fix is a layout-compatible
      ABI bump: define a dedicated `struct snvm_ioctl_chrdev {
      struct pci_device_addr addr; uint32_t out_minor; uint32_t
      reserved[3]; }` for opcodes 3/4, leave `pci_device_addr` as
      a pure in-param everywhere else. Touches:
      `include/ioctl.h`, both `snvme-*/pci.c` chrdev helpers,
      `libnvm/src/linux/device.cpp::nvm_controller_init`,
      `snvme_smoke_libnvm.c` / `snvme_smoke_io.c` /
      `snvme_smoke_addq.c` / `snvme_smoke_gpu.cu` / `snvme_smoke_qgroup.c`,
      and PORTING.md §4.2 (chrdev minor allocation).

## nvme_storage — Durability follow-ups

`PersistentFileLog::persist()` is atomic per write (tmp → fsync →
rename), and bootstrap runs a C0 reconcile pass that drops
tombstone entries (log entry but no `<name>.bin`) and unlinks
ghost `.bin` files (file but no log entry). That covers the
practical user-visible damage from a crash mid-create / mid-delete.

R5a.1 also added `create_file(persist_now, sync_now)` + a
`flush_metadata(device)` API so bulk-init workloads (e.g. LMCache
provisioning N=10^6 KV-shard files at startup) can collapse the
otherwise-O(N²) total log write and per-file fsync down to one
syncfs(2) + one log rewrite.  See
`nvme_storage/test/nvme_storage_bulk_smoke.cu` for the
"per-call durable vs deferred" wall-time comparison.

What's still missing is genuine transactionality on the
`(host_fs op ↔ log persist)` pair. The two follow-ups below close
that hole at increasing levels of strictness; do them only when
real workload pressure shows up (concurrent create+crash, or
write-amp from rewrite-on-every-persist).

- [ ] **C1: tombstone-style intent on the entry.** Extend
      `OnDiskEntry` with a `status` byte:
      `PENDING_CREATE` / `COMMITTED` / `PENDING_DELETE`. Reorder
      `create_file` to log.add(PENDING_CREATE) + persist BEFORE
      the host pwrite/fsync, then flip to COMMITTED and persist
      again after fsync succeeds; reorder `delete_file`
      symmetrically (PENDING_DELETE first, ::unlink, then
      log.remove + persist). Reconcile then has authoritative
      state to drive: `PENDING_CREATE` means "host fs not
      committed, drop entry"; `PENDING_DELETE` means "complete the
      ::unlink, then drop entry". One ABI bump
      (`OnDiskHeader::version` 1 → 2 + a forward-compat reader for
      v1). Estimated ~150 LoC + a recovery-mode smoke
      (kill -9 mid-create / mid-delete, rerun bootstrap).

- [ ] **C2: append-only write-ahead log (WAL).** Replace
      `rewrite-on-every-persist` with a `<mount>/.tutti/file_log.wal`
      append-only stream of `{op, file_id, name, extents}` records
      plus periodic compaction into `file_log.bin`. Each `add` /
      `remove` writes one record + fsync (constant-cost, no
      O(N) rewrite); recovery replays WAL onto the snapshot.
      Required if the directory ever gets large (thousands of
      entries) and the rewrite cost shows up in create_file
      latency. Carries the C1 status field semantics inside each
      WAL record. Estimated ~400 LoC + replay smoke.

Both items strictly subsume C0 — the bootstrap reconcile becomes
unnecessary once the on-disk format is intent-logged. Until then,
C0 is good enough to keep the user-visible directory clean across
process crashes.

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
