<img src="doc/pics/tardis_logo.png" align="left" width="32" />

## Tutti

`Tutti` (Italian for "all instruments together") is a `CPU/GPU companion
storage software stack`: a unified storage runtime where CPU and GPU paths
cooperate on top of a shared memory subsystem and a pluggable backend SPI.

The active architecture baseline is tracked in [`Roadmap.md`](Roadmap.md).
New abstractions, namespaces (`tutti::`), and headers use the `Tutti` name.

## Current Status

This repository currently contains:

- the existing storage/runtime implementation
- a modified local NVMe stack, including kernel-module changes
- an `NVMeService` device-manager prototype
- architecture and refactor planning documents for `v0.1`

The repository is in a transition stage:

- current code layout reflects historical implementation boundaries
- target architecture is organised around `api`, `runtime`, `memory`, `device_manager`, `io_engine`, `backends`, and `adapters`
- interface and directory changes should be discussed before major code movement

## Start Here

If you are a new contributor or another AI agent, read these files first:

1. [`Roadmap.md`](Roadmap.md)
   Current `v0.1` architecture baseline, roadmap, feature snapshot, and known bugs.

2. [`Todolist.md`](Todolist.md)
   Current active work items, discussion items, and immediate priorities.

3. [`CONTRIBUTING.md`](CONTRIBUTING.md)
   Coding rules, interface discipline, collaboration rules, and commit naming requirements.

4. [`doc/history/README.md`](doc/history/README.md)
   Rules for versioned roadmap history and archive handling.

5. [`intergration/intergration.md`](intergration/intergration.md)
   Current upper-layer integration intent, especially for framework-facing APIs.

After that, read the implementation-specific documents relevant to your area:

- [`doc/design/backend-spi.md`](doc/design/backend-spi.md)
  Backend SPI contract — read this before implementing any new storage backend.

- [`doc/architecture/system-architecture.md`](doc/architecture/system-architecture.md)
  Full system architecture with component diagrams and data flow.

- [`backends/local/NVMeService/NVMeService.md`](backends/local/NVMeService/NVMeService.md)
- [`filesystems/ext4/README.md`](filesystems/ext4/README.md)

## Knowledge Keywords

These keywords help human contributors and AI agents search and understand the codebase faster.

### Domain Keywords

- `NVMe Specification`
- `PCIe`
- `NVMe queue`
- `submission queue`
- `completion queue`
- `queue pair`
- `doorbell`
- `PRP`
- `DMA`
- `pinned memory`
- `CUDA`
- `CUDA IPC`
- `GPU Direct Storage`
- `GDS`
- `RDMA`
- `Linux kernel module`
- `DKMS`
- `systemd`
- `EXT4 file system`
- `FIEMAP`
- `LMCache`
- `Mooncake`
- `device manager`
- `IO engine`
- `memory registration`
- `Unified Storage Runtime`

### Code Search Keywords

- `GeminiFS::init`
- `parse_and_setup_controllers`
- `GPUController`
- `NVMeController`
- `GPUFileManager`
- `GPUMemoryMapper`
- `FsAllocQueues`
- `FsReleaseQueues`
- `LeaseHeartbeat`
- `Controller::init_queues`
- `d_qps`
- `d_ctrl_ptr`
- `d_queue_acquire_helper`

### Search Strategy

When modifying or reviewing code, it is usually best to search in this order:

1. the architecture and roadmap terms in [`Roadmap.md`](Roadmap.md)
2. the subsystem documents
3. the exact code identifiers above
4. the local backend implementation under [`backends/local/`](backends/local/)
5. the current monolithic path under [`filesystems/ext4/libgeminifs`](filesystems/ext4/libgeminifs)

## Repository Map

This is the current repository structure as it exists today.

### Core Implementation Areas

- [`backends/local/nvme/libnvm`](backends/local/nvme/libnvm)
  User-space NVMe support library used by the local backend path.

- [`backends/local/kernel_modules/snvme-5.15.0`](backends/local/kernel_modules/snvme-5.15.0)
  Modified Linux NVMe kernel-module lineage used to support CPU/GPU access to NVMe queue resources.
  Additional kernel baselines live alongside this one (see
  [Supported Linux Kernels](#supported-linux-kernels)).

- [`backends/local/NVMeService`](backends/local/NVMeService)
  Local device-manager prototype for controller initialization, queue leasing, and process attach flow.

- [`filesystems/ext4/libgeminifs`](filesystems/ext4/libgeminifs)
  Current monolithic implementation area. This is where GPU controller logic, NVMe controller logic, memory mapping, and GPU file management are currently mixed together.

### Supporting Areas

- [`examples/`](examples)
  Validation and example programs.

- [`scripts/`](scripts)
  Environment preparation, reset, binding, and operational scripts.

- [`doc/`](doc)
  Project documents, including roadmap history and future architecture material.

- [`build/`](build)
  Local generated build output. Treat as generated artifacts, not source-of-truth documentation.

## Current Collaboration Rules

When working in this repository:

- treat [`Roadmap.md`](Roadmap.md) as the active architecture source
- treat [`Todolist.md`](Todolist.md) as the active task board
- preserve version snapshots in [`doc/history/`](doc/history/) when the maintainer changes versions
- avoid assuming the current directory layout is the final layout
- use the `Tutti` name (and `tutti::` namespace) in new abstractions and public APIs
- discuss top-level interface changes and directory reshaping before large code edits

## Current Known Problem

The currently tracked `v0.1` known bug is:

- `GPU file persistence` is not yet stable and must not be treated as a reliable persistence contract

See [`Roadmap.md`](Roadmap.md) for the version snapshot that tracks this.

## Build and Runtime Notes

The current build still reflects the existing implementation, not the final target architecture.

Relevant build/runtime entry points:

- root [`CMakeLists.txt`](CMakeLists.txt)
- [`filesystems/ext4/README.md`](filesystems/ext4/README.md)
- root configuration sample (single source of truth):
  - [`sys_config.yaml`](sys_config.yaml)

Important operational constraint:

- the modified NVMe kernel module is part of the local backend baseline and must be considered in deployment, Linux-version compatibility, and startup sequencing

### Module signing (locked-down deployment kernels)

Some deployment kernels — notably **TencentOS Server 5.4.241-1-tlinux4-0017**
— ship with `CONFIG_MODULE_SIG_FORCE=y`, which makes the kernel reject
any unsigned module at `insmod` / `modprobe` time:

```
kernel: Loading of unsigned module is rejected
insmod: ERROR: could not insert module snvme-core.ko: Required key not available
```

`CONFIG_MODULE_SIG_FORCE=y` is compiled-in and **cannot** be disabled at
runtime (sysctl `kernel.modules_sig_enforce`, kernel cmdline
`module.sig_enforce=0`, `insmod --force`, lockdown bypass — none of
them apply).  Three workflows are available; choose by environment:

1. **Signing service (production)** — send `snvme-core.ko` and
   `snvme.ko` to the deployment team's module-signing service, drop
   the returned signed artefacts back into `build/module/`, then
   `insmod`. This is the path that the locked-down kernel was built
   for.
2. **Development kernel (development hosts)** — install a kernel
   that does NOT set `CONFIG_MODULE_SIG_FORCE=y` (mainline / OSS
   kernels typically do not), select it via GRUB, and let
   `SNVME_KERNEL_VERSION` pick a matching `snvme-<tag>` baseline.
   Verify the active kernel is signing-free with:

   ```bash
   grep CONFIG_MODULE_SIG_FORCE /boot/config-$(uname -r)
   ```

   A `# CONFIG_MODULE_SIG_FORCE is not set` line means `insmod`
   accepts unsigned modules.  This is the recommended bring-up
   environment for editing the snvme baselines.
3. **Self-signing (advanced, host-specific)** — only feasible if
   either (a) the running kernel was compiled with
   `CONFIG_SECONDARY_TRUSTED_KEYRING=y` AND the keyring is writable
   by the current process, or (b) you have access to the same
   signing CA whose public half is in `.builtin_trusted_keys`.
   On TencentOS 5.4.241-1-tlinux4-0017 (a) is `# ... is not set`
   and (b) requires the central signing service, so self-signing
   degenerates back to workflow (1) for that image.

The `scripts/sign-file` helper that comes with the kernel build tree
is the canonical signing tool in all three workflows:

```bash
/usr/src/kernels/$(uname -r)/scripts/sign-file \
    sha256 <private_key.pem> <public_key.x509> snvme-core.ko
/usr/src/kernels/$(uname -r)/scripts/sign-file \
    sha256 <private_key.pem> <public_key.x509> snvme.ko
```

### Queue budget tuning

The snvme kernel module splits each NVMe controller's I/O queue budget
between the **kernel-side blk-mq path** (so the disk is still mountable
and `read(2)`/`write(2)` works) and the **user-side share** that
NVMeService hands to GPU clients via CUDA IPC. The split is operator-
controlled through one block in [`sys_config.yaml`](sys_config.yaml):

```yaml
nvmes:
  - pci_addr: "0000:50:00.0"
    total_queues: 64                # NVMe IOQ budget the operator commits
    queue_groups:
      - { gpu_id: 0, count: 32 }    # user share (per-GPU partitions)
    queue_setup:
      kernel_ioq_cap: 32            # kernel-side cap (QueuePair units)
      on_host: false
      nr_write: 0
      nr_poll: 0
```

All counts are in **QueuePair units** (1 pair = 1 SQ + 1 CQ). The daemon
enforces the local invariant
`Σqueue_groups[].count + queue_setup.kernel_ioq_cap <= total_queues`
at startup; the kernel additionally checks the result against the
controller's actual `Identify Controller` IOQ ceiling at
`NVM_SET_IOQ_NUM` time.

**When to set `kernel_ioq_cap` explicitly.** If the controller's MSI-X
vector count is smaller than `num_possible_cpus()` on the host, the
kernel's default ask (`nr_io_queues = num_possible_cpus()`) consumes
every vector and leaves zero room for the user-allocated share. The
GPU-direct path then silently falls back to `dma_alloc_coherent`, the
smoke test reports `nr_user_q=0`, and dmesg carries the signature

```
queue squeeze: kernel=N user=M (controller granted ...)
```

The fix is to lower `kernel_ioq_cap` so that the controller's MSI-X
grant has room left for the user share. Verified on HGX H20 + Intel DC
SSD (MSI-X=136 vs 192 vCPUs): pre-fix smoke shows `nr_user_q=0`;
post-fix dmesg becomes `queue split: kernel=K user=M` and smoke reports
the requested user-queue count. The `snvme_smoke_gpu.cu` reference test
hard-codes `kernel_ioq_cap = 32` for the same reason; production
callers should match their controller's MSI-X grant.

See:

- [`sys_config.yaml`](sys_config.yaml) — full schema with field-by-field
  comments.
- [`backends/local/kernel_modules/PORTING.md`](backends/local/kernel_modules/PORTING.md)
  §8.1 — troubleshooting cheat sheet row for the `queue squeeze` dmesg
  signature.

## Supported Linux Kernels

The modified NVMe kernel module (`snvme`) is maintained as one directory
per supported Linux kernel baseline under
[`backends/local/kernel_modules/`](backends/local/kernel_modules). The
baseline is selected at CMake configure time via
`-DSNVME_KERNEL_VERSION=<tag>` (the `<tag>` is the directory suffix,
e.g. `5.15.0-public` or `5.4.241-1-tlinux4-0017`). If omitted, CMake
auto-detects from `uname -r` via longest-prefix match.

| Baseline directory | Kernel version | Status | Upstream source |
|---|---|---|---|
| [`snvme-5.15.0-public`](backends/local/kernel_modules/snvme-5.15.0-public) | Linux 5.15.0 (mainline) | Active — full snvme baseline | [torvalds/linux](https://github.com/torvalds/linux) |
| [`snvme-5.4.241-1-tlinux4-0017`](backends/local/kernel_modules/snvme-5.4.241-1-tlinux4-0017) | Linux 5.4.241-1-tlinux4-0017 (TencentOS Server / OpenCloudOS LTS) | Active — full snvme baseline (`snvme-core.ko` + `snvme.ko`) | [OpenCloudOS-Kernel `linux-5.4/lts/5.4.241-30.0017`](https://gitee.com/OpenCloudOS/OpenCloudOS-Kernel/tree/linux-5.4%2Flts%2F5.4.241-30.0017/) |

Notes:

- `snvme-5.15.0-public` is the reference baseline and contains the full
  modified NVMe driver sources, including the CPU/GPU IO-queue sharing
  hooks in `pci.c` and the `/dev/snvm_control` + `/dev/ssnvme*` ioctl
  surface that libnvm consumes.
- `snvme-5.4.241-1-tlinux4-0017` carries a **complete** snvme port onto
  the upstream nvme-5.4.241 host driver as shipped by TencentOS / the
  OpenCloudOS LTS 5.4.241-30.0017 kernel. It builds both
  `snvme-core.ko` and `snvme.ko`, exposes the same `/dev/snvm_control`
  + `/dev/ssnvme<N>` UAPI as `snvme-5.15.0-public`, and ships several
  small bug fixes vs the 5.15 baseline (documented at every affected
  site; see PORTING.md §7.3.1 for the trap list they map to).
  Concretely:
  - The upstream nvme-5.4.241 host driver (`core.c`, `fabrics.c`,
    `multipath.c`, `nvme.h`, `pci.c`, `rdma.c`, `tcp.c`) is included
    with the snvme symbol-rename pass applied (see
    [`snvme-5.4.241-1-tlinux4-0017/snvme-rename.sed`](backends/local/kernel_modules/snvme-5.4.241-1-tlinux4-0017/snvme-rename.sed)).
    The rename set is derived from the actual 5.4
    `EXPORT_SYMBOL_GPL` surface, not blindly copied from 5.15 — e.g.
    it includes the 5.4-only `nvme_init_identify` rename and
    intentionally omits the 5.15-only `nvme_alloc_request_qid`,
    `nvme_init_ctrl_finish`, `__nvme_check_ready`, and
    `nvme_fail_nonready_command` renames.
  - For exported helpers that snvme does not call across module
    boundaries (e.g. `nvme_reset_ctrl_sync`, `nvme_delete_ctrl`,
    `nvme_cancel_tagset`, `nvme_cancel_admin_tagset`,
    `nvme_stop_keep_alive`, `nvme_sync_io_queues`), the upstream name
    is kept and the `EXPORT_SYMBOL_GPL` line is commented out — the
    same approach used in `snvme-5.15.0-public`.
  - `admin_timeout` is renamed to `s_admin_timeout` uniformly (the
    `module_param` site, the `EXPORT_SYMBOL_GPL`, the `nvme.h` extern
    declaration, the `ADMIN_TIMEOUT` macro body, and every
    open-coded reference). This is slightly more thorough than
    `snvme-5.15.0-public`, which renames only the definition and
    relies on the in-tree `nvme-core.ko` to satisfy the macro's
    extern reference at load time.
  - PORTING.md §2 namespace-separation rule is enforced for every
    string literal that snvme-rename.sed cannot touch:
    - in `core.c`: `alloc_workqueue("snvme-wq", ...)` and the matching
      `snvme-reset-wq` / `snvme-delete-wq` (the in-tree driver
      already owns `nvme-wq` etc. under `/sys/devices/virtual/
      workqueue/`, so duplicates make `insmod` fail in
      `nvme_core_init`); `alloc_chrdev_region(..., "snvme")`;
      `class_create(THIS_MODULE, "snvme")` and
      `class_create(THIS_MODULE, "snvme-subsystem")`;
      `dev_set_name(ctrl->device, "snvme%d", ...)`; and
      `dev_set_name(&subsys->dev, "snvme-subsys%d", ...)`;
    - in `multipath.c` and `nvme.h`:
      `sprintf(disk_name, "snvme%dn%d", ...)` and
      `"snvme%dc%dn%d"` (gendisk names → `/dev/snvme0n1`);
    - in `pci.c`: `pci_request_irq(..., "snvme%dq%d", ...)` (IRQ
      names in `/proc/interrupts`) and
      `pci_request_mem_regions(pdev, "snvme")` (`/proc/iomem`
      owner tag).
    Without these renames the in-tree `nvme.ko` and `snvme.ko`
    collide on workqueue sysfs, `/sys/class/nvme/`, `/dev/nvme*`,
    `/proc/interrupts` and `/proc/iomem`. The NVMe wwid prefix
    `"nvme.%04x-..."` in `wwid_show` is part of the userspace ABI
    and is intentionally **not** renamed (udev and multipath-tools
    match on it).
  - The kernel-version-agnostic snvme helpers are copied verbatim
    from `snvme-5.15.0-public/`: `ctrl.{c,h}`, `list.{c,h}`,
    `map.{c,h}`, `nvfs-core.h`, `nvfs-p2p.{c,h}`, `nvfs-pci.{c,h}`.
  - 5.15-only features deliberately **not** ported into 5.4: the
    `struct nvme_gpu_map` / `struct GPU_io_queue_info` declarations
    in `nvme.h` (dead code in 5.15 too — only declared, never used),
    `ioctl.c` as a separate translation unit (5.4 keeps the ioctl
    handlers inside `core.c`), `zns.c`, `hwmon.c`, and any 5.15-only
    `nvme_*` helpers.
  - The full snvme increment for `pci.c` (CPU/GPU queue-share hooks,
    `/dev/snvm_control` + `/dev/ssnvme<N>` chrdev plumbing, BAR0
    mmap, all `SNVM_*` / `NVM_*` ioctl handlers, segment-7 user
    queue teardown) has been re-expressed against the 5.4
    `struct nvme_dev` / `struct nvme_queue` layout. The path the
    user-pinned queues take diverges intentionally from 5.15: in 5.4
    they are bound to the controller through a separate
    `nvme_create_user_queue()` flow that does NOT allocate a
    `struct nvme_queue` and therefore cannot pollute the admin
    queue's `dma_alloc_coherent` path (PORTING.md §7.2 invariant).
  - Bug fixes vs `snvme-5.15.0-public/pci.c`, all hitting PORTING.md
    §7.3.1 traps that are easy to reintroduce on the next uplift:
    - `snvm_dev_map_ioctl()`: `int ret = 0;` at declaration AND
      explicit `ret = 0;` on every success break, including
      `NVM_UNMAP_*`, `NVM_SET_SHARE_REG` and `NVM_CLEAR_IOQ_NUM`
      (trap #2).
    - `NVM_MAP_HOST_MEMORY` / `NVM_MAP_DEVICE_QUEUE_MEMORY`:
      `copy_to_user()` failure on the IO-address writeback rolls
      back `ctrl->ioq_map_num`, `ctrl->cq_num` and calls
      `unmap_and_release(map)` before returning `-EFAULT` (trap #4).
    - `NVM_MAP_DEVICE_MEMORY`: data-path `copy_to_user()` failure
      calls `unmap_and_release(map)` instead of leaking the pinned
      GPU pages.
    - `NVM_MAP_DEVICE_QUEUE_MEMORY` with `ioq_idx < 0`:
      `unmap_and_release(map)` before returning `-EFAULT` (5.15
      leaks the fresh p2p mapping).
    - `nvme_create_user_queue()`: every non-zero `adapter_alloc_
      sq_user()` result rolls back the controller-side CQ (5.15
      only rolled back for `result > 0`, leaking the CQ for
      transport / submit_sync failures, which is the much more
      common case).
    - `snvme_disable_user_io_queues()` is a new teardown helper
      called from `nvme_dev_disable()` that issues Delete-IO-{SQ,CQ}
      for every user QID before the reset cycle proceeds. Without
      it the next `SNVM_DEVICE_BIND` fails with NVMe status `0x101`
      (Invalid Queue Identifier) and the only symptom is a probe
      that "just fails" — `snvme-5.15.0-public` carries the same
      latent defect (its smoke tests only do a single bind, so the
      symptom never surfaces there).
  - [`snvme-5.4.241-1-tlinux4-0017/snvme-pci-5.15-incremental.diff`](backends/local/kernel_modules/snvme-5.4.241-1-tlinux4-0017/snvme-pci-5.15-incremental.diff)
    — the unified diff between upstream `nvme-5.15.0/host/pci.c` and
    `snvme-5.15.0-public/pci.c`, kept in tree as the canonical
    reference for the snvme increment. Re-consult it on the next
    baseline uplift.
  - [`backends/local/kernel_modules/PORTING.md`](backends/local/kernel_modules/PORTING.md)
    — prescriptive porting guide (UAPI surface, bring-up sequence,
    known regression traps). Read §7.3.1 ("Known regression traps")
    before editing any `snvme-*/pci.c`: it is the canonical list of
    bugs that have been fixed in snvme and are easy to reintroduce
    during a 3-way merge.
  - [`backends/local/kernel_modules/test/`](backends/local/kernel_modules/test)
    — `snvme_smoke{,_gpu}` end-to-end UAPI tests. These exercise
    every ioctl listed in `libnvm/include/ioctl.h`; passing both is
    the verification gate after any baseline port.
  - The userspace contract from
    [`backends/local/nvme/libnvm/include/ioctl.h`](backends/local/nvme/libnvm/include/ioctl.h)
    (the `NVM_*` and `SNVM_*` ioctl numbers, `nvm_ioctl_map`,
    `nvm_ioctl_dev`, `pci_device_addr` structs, the
    `/dev/snvm_control` + `/dev/ssnvme<domain>` device-node names) is
    treated as the cross-baseline ABI: any `snvme-<tag>` baseline
    must implement exactly those numbers and structs unchanged so
    `libnvm` works against either kernel without recompilation.
- New baselines must follow the `snvme-<tag>` naming convention, where
  `<tag>` is a prefix of the target kernel's `uname -r` output, so the
  build can locate them via `SNVME_KERNEL_VERSION` or the automatic
  longest-prefix match.
- Kernel-API adaptation points must be isolated inside each baseline
  directory; upper-layer APIs and the `local_nvme` backend contract must
  not depend on the selected kernel version (see
  [`Roadmap.md`](Roadmap.md) → "Kernel Module Baseline").
