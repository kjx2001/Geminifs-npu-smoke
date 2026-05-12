# SNVMe Kernel Module — Porting Guide

> Audience: developers who want to port the **SNVMe** kernel module
> (`backends/local/kernel_modules/snvme/`) to a different upstream Linux
> kernel version (the current code base tracks the **Linux 5.15 LTS**
> NVMe driver tree). This document is a **prescriptive specification** of
> the modifications SNVMe applies on top of the stock `drivers/nvme/host`
> tree, the kernel/user-space contract it exposes, and the bring-up
> sequence the user-space side (`libnvm`) relies on.
>
> Read it together with:
>
> - `snvme/pci.c`          — SNVMe core (modified `nvme/host/pci.c`)
> - `snvme/ctrl.{h,c}`     — per-PCI controller bookkeeping + chrdev factory
> - `snvme/ioctl.c`        — original NVMe user ioctl path (unchanged)
> - `snvme/Makefile.in`    — out-of-tree build glue (configured by CMake)
> - `../nvme/libnvm/include/ioctl.h` — shared kernel/user UAPI header
> - `../nvme/libnvm/src/linux/device.cpp` — userspace counterpart
> - `../nvme/libnvm/include/ctrl.h` — `Controller` C++ wrapper (entry point)

---

## 1. What SNVMe is, in one paragraph

SNVMe is a fork of the in-tree Linux NVMe host driver that allows
**user-space and CUDA kernels to own NVMe IO submission/completion
queues directly**. The stock NVMe driver allocates SQ/CQ ring buffers
from `dma_alloc_coherent()` (kernel pages). SNVMe instead lets a user
process

1. allocate SQ/CQ memory itself — in **host-pinned memory** or in
   **GPU memory** (via `cudaMalloc` + `nvidia_p2p_get_pages`),
2. hand the IO addresses of those pages to the kernel through
   `ioctl(/dev/snvm_<N>)`,
3. ask the kernel to drive a normal NVMe `Identify`/`Create IO SQ`/
   `Create IO CQ` admin sequence using **the user-supplied memory** as
   the queue backing store,
4. and finally `mmap()` BAR0 of the device so the doorbell registers
   are accessible from CUDA kernels (via `cudaHostRegister`/
   `cudaHostGetDevicePointer`).

Once those steps complete, the same NVMe controller is simultaneously:

- a **regular block device** (`/dev/snvme<X>n<Y>` — note the leading
  `s`, set in `core.c:3806` / `multipath.c:58`; e.g. `/dev/snvme0n1`),
  mounted by the kernel like any other NVMe SSD, and
- a **direct-IO submission target** for user-space / GPU code, which
  rings the doorbells without round-tripping through the block layer.

This is the same idea behind BaM / GPUDirect Storage; SNVMe is the
in-house implementation that lives next to a stock `nvme` driver
without conflicting with it.

---

## 2. Co-existence with the in-tree `nvme` driver

> **Hard requirement.** SNVMe must coexist with the upstream `nvme`
> driver. The two modules differ only in which PCI device each one is
> currently bound to.

The upstream module provides the symbols `nvme_*` and registers a PCI
driver named `"nvme"`. To avoid clashes:

| Concern                     | Convention                                                |
| --------------------------- | --------------------------------------------------------- |
| Kernel module name          | `snvme.ko`, `snvme-core.ko`                               |
| `pci_driver.name`           | `"snvme"` (`PCI_DRIVER_NAME` in `pci.c`)                  |
| Internal symbol prefix      | `snvm_*` / `s_nvme_*` for anything SNVMe-specific         |
| Functions kept from upstream | rename with `s_` prefix when their semantics changed     |
| chrdev class name           | `"libsnvm helper"` (`DRIVER_NAME` in `pci.c`)             |
| Control device              | `/dev/snvm_control` (single-instance, factory)            |
| Per-controller chrdev       | `/dev/ssnvme<N>` — note the **double 's'** (`ctrl.c:36`); used for BAR0 mmap + queue ioctls |
| Block device on success     | `/dev/snvme<X>n<Y>` — single 's', set in `core.c:3806` and `multipath.c:58` (e.g. `/dev/snvme0n1`) |

> **Naming gotcha.** SNVMe exposes **two** `/dev` objects per bound
> NVMe controller, and they look superficially similar:
>
> | Path                | Type      | Created by                                | fops / role                                  |
> | ------------------- | --------- | ----------------------------------------- | -------------------------------------------- |
> | `/dev/ssnvme<N>`    | char dev  | `SNVM_CHRDEV_CREATE` ioctl                | `snvm_dev_fops`: BAR0 mmap, queue ioctls     |
> | `/dev/snvme<X>n<Y>` | block dev | `s_nvme_probe()` → `device_add_disk()`    | regular block layer; what you actually mount |
>
> The names differ by **one letter** (`ssnvme` vs `snvme`). Both are
> namespaced away from the in-tree `nvme` driver: the upstream module
> would create `/dev/nvme<X>n<Y>`, never `/dev/snvme<X>n<Y>`.

When porting:

1. **Do not** rename the file operations vtables that `module_init`
   exports (`snvm_fops`, `snvm_dev_fops`); userspace finds them by
   device name.
2. The `snvme-core.ko` ↔ `snvme.ko` split mirrors the upstream
   `nvme-core.ko` ↔ `nvme.ko` split. Keep the layering — promoting
   helper code from `snvme-core` into `snvme` will cause symbol
   collisions on systems where the in-tree `nvme` is also loaded.
3. Stay strict on `EXPORT_SYMBOL_GPL` selection. If the in-tree
   `nvme-core.ko` has already exported a name SNVMe needs, do **not**
   re-export it from `snvme-core.ko`; rename the SNVMe one instead.

---

## 3. The two modifications, in detail

The SNVMe diff against upstream `drivers/nvme/host/` boils down to
**(A) renaming everything to `snvm_*`** and **(B) replacing the
implicit `nvme_probe()`-driven initialization with an explicit,
user-driven bring-up flow**. Section 3.1 covers (A); 3.2 covers (B).

### 3.1 Symbol renames (mechanical)

For every translation unit in `nvme/host/` that SNVMe carries
(`core.c`, `pci.c`, `ioctl.c`, `multipath.c`, `zns.c`, `hwmon.c`,
`fabrics.c`, `rdma.c`, `tcp.c`, `nvme.h`, …):

- Rename the PCI driver registration: `nvme_driver` → `snvme_driver`,
  `.name = "nvme"` → `.name = PCI_DRIVER_NAME` (= `"snvme"`).
- Rename `module_init`/`module_exit` entry points:
  `nvme_init` → keep the name (it is `static`) but make sure it's the
  **SNVMe** init, calling `snvm_cdev_init()` and `pci_register_driver(&snvme_driver)`.
- Rename module-wide globals that the in-tree module also defines
  (e.g. workqueues — `nvme_wq` becomes `s_nvme_wq`, the per-controller
  cache becomes the `ctrl_list` we maintain in `snvme/list.c`).
- Functions whose semantics change (queue allocation, queue pair
  allocation, IRQ setup) are prefixed with `s_` in this fork
  (`s_nvme_alloc_queue`, `s_nvme_setup_io_queues`, …). The upstream
  names are never re-defined in the SNVMe build.

When porting to a newer kernel, do the renames mechanically first,
then re-apply the surgical changes from §3.2.

### 3.2 User-driven bring-up: the `snvm_control` interface

This is the only **functional** change. The intuition:

- Stock NVMe: bind PCI device → `nvme_probe()` → `nvme_alloc_queue()`
  uses `dma_alloc_coherent` for SQ/CQ → `nvme_setup_io_queues()` is
  done before `nvme_probe()` returns.
- SNVMe: PCI device is **explicitly bound** by user-space request,
  *after* user-space has already published its own queue memory.
  `s_nvme_setup_io_queues()` then uses the IO addresses the user
  supplied instead of calling `dma_alloc_coherent`.

The state machine lives in the per-controller `struct ctrl`
(`snvme/ctrl.h`). The salient fields are:

```c
unsigned int ioq_num;       /* number of IO queues the user promised */
unsigned int cq_num;        /* of those, how many are completion queues */
unsigned int ioq_map_num;   /* how many user pages have been registered so far */
unsigned int use_sreg;      /* "share registers" flag — when set, probe()
                             * must consume user pages instead of dma_alloc */
```

The flag `use_sreg` plus the equality `ioq_num == ioq_map_num` is what
gates the deviation from the upstream probe path. See `pci.c:3227`:

```c
if (ctrl && ctrl->ioq_num == ctrl->ioq_map_num && ctrl->use_sreg) {
    dev->nr_user_allocated_queues = ctrl->ioq_num;
    dev->nr_user_allocated_cq     = ctrl->cq_num;
    /* ...skip dma_alloc, use ctrl->user pages... */
}
```

When porting to a newer kernel:

1. Find the upstream equivalents of `nvme_alloc_queue` /
   `nvme_setup_io_queues` / `nvme_create_io_queues`.
2. Inject a branch at the same logical points: if the per-controller
   `ctrl` sentinel says "user already published queues", skip
   `dma_alloc_coherent` and copy the pre-registered IO addresses out
   of the controller's `map` list (`snvme/map.c`) into the SQ/CQ
   descriptors instead.
3. Leave the admin queue path on `dma_alloc_coherent`. Only the IO
   queues are user-supplied.
4. Make sure `nvme_dev_add()` / `nvme_alloc_admin_tags()` continue to
   produce a regular `/dev/snvme<X>n<Y>` block device — the user-space
   side relies on it for `mount()` (`Host_file_system_int`).

---

## 4. UAPI surface (kernel ↔ user contract)

This is the public ABI. **Changes here break libnvm and any
NVMeService client.** Header: `backends/local/nvme/libnvm/include/ioctl.h`.

### 4.1 Device files

| Path                | Created by                            | fops                | Purpose                                   |
| ------------------- | ------------------------------------- | ------------------- | ----------------------------------------- |
| `/dev/snvm_control` | `snvm_cdev_init()` (module load)      | `snvm_fops`         | bind/unbind PCI devices, factory of chrdev |
| `/dev/ssnvme<N>`    | `SNVM_CHRDEV_CREATE` ioctl            | `snvm_dev_fops`     | per-controller: BAR0 mmap + queue ioctls   |

The minor `<N>` is allocated from `snvm_chrdev_minor_ida` and is
returned to user-space in `pci_device_addr.domain` of the `_IOWR`
result (see §4.3).

### 4.2 `/dev/snvm_control` ioctls

```c
enum snvm_ctrl_ioctl_type {
    SNVM_DEVICE_BIND        = _IOW(0x90, 1, struct pci_device_addr),
    SNVM_DEVICE_UNBIND      = _IOW(0x90, 2, struct pci_device_addr),
    SNVM_CHRDEV_CREATE      = _IOWR(0x90, 3, struct pci_device_addr),
    SNVM_CHRDEV_REMOVE      = _IOW(0x90, 4, struct pci_device_addr),
};
```

Semantics:

- **`SNVM_CHRDEV_CREATE` / `_REMOVE`**: idempotent; (un)registers
  `/dev/ssnvme<N>` for the BDF described by `pci_device_addr`.
  `_CREATE` writes the allocated minor back into `addr.domain` (this
  field is reused as an out-parameter — see `pci.c:4204`).
- **`SNVM_DEVICE_BIND`**: detaches whatever PCI driver currently owns
  the BDF (typically the in-tree `nvme`), registers `snvme_driver` if
  not already registered, and force-attaches it. Triggers
  `s_nvme_probe()`, which honours the per-controller `use_sreg` flag.
- **`SNVM_DEVICE_UNBIND`**: counterpart; only succeeds if the device
  is currently bound to `snvme`.

### 4.3 `/dev/ssnvme<N>` ioctls

```c
enum nvm_ioctl_type {
    NVM_MAP_HOST_MEMORY             = _IOW(0x80, 1, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_MEMORY           = _IOW(0x80, 2, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_QUEUE_MEMORY     = _IOW(0x80, 3, struct nvm_ioctl_map),
    NVM_UNMAP_HOST_MEMORY           = _IOW(0x80, 4, uint64_t),
    NVM_UNMAP_DEVICE_MEMORY         = _IOW(0x80, 5, uint64_t),
    NVM_UNMAP_DEVICE_QUEUE_MEMORY   = _IOW(0x80, 6, uint64_t),
    NVM_SET_IOQ_NUM                 = _IOW(0x80, 7, struct nvm_ioctl_dev),
    NVM_SET_SHARE_REG               = _IOW(0x80, 8, struct nvm_ioctl_dev),
    NVM_GET_DEV_INFO                = _IOR(0x80, 9, struct nvm_ioctl_dev),
    NVM_CLEAR_IOQ_NUM               = _IOW(0x80, 10, struct nvm_ioctl_dev),
};
```

`NVM_MAP_*` requests carry a `struct nvm_ioctl_map`:

```c
struct nvm_ioctl_map {
    uint64_t  vaddr_start;   /* userspace VA of the queue buffer */
    size_t    n_pages;
    uint64_t* ioaddrs;       /* OUT: kernel writes IO addresses here */
    int       ioq_idx;       /* >=0: this page belongs to IO queue ioq_idx */
    int       is_cq;         /* 1 = completion queue ring, 0 = submission */
};
```

The kernel pins the pages (via `get_user_pages_fast` for HOST, via
`nvidia_p2p_get_pages` for DEVICE) and either records IO addresses
(when `ioq_idx >= 0` ⇒ "this is a queue ring") or sets up a generic
DMA mapping (when `ioq_idx < 0` ⇒ "this is a PRP buffer").

Each `MAP_*` call increments `ctrl->ioq_map_num` when `ioq_idx >= 0`;
`NVM_SET_IOQ_NUM` declared the target value upfront, and
`NVM_SET_SHARE_REG` flips `use_sreg=1` so that the next
`SNVM_DEVICE_BIND` / `nvme_probe()` consumes the user pages.

### 4.4 `mmap()` on `/dev/ssnvme<N>`

`snvm_dev_fops.mmap = svm_mmap_registers` (`pci.c:3931`) maps **BAR0**
of the bound NVMe controller into the calling process. Userspace then
uses `cudaHostRegister(..., cudaHostRegisterIoMemory)` so CUDA kernels
can read/write the doorbells directly. This is the source of the
`mm_ptr` value flowing through `_nvm_ctrl_init()` and ultimately into
each `QueuePair::sq.db` / `cq.db`.

---

## 5. Bring-up sequence (the canonical flow)

This is **the** flow that user-space must implement; it is also the
flow your port must keep working. Reference implementation:
`Controller::Controller(...)` ⇢ `nvm_controller_init()` ⇢
`nvm_device_init()` in `backends/local/nvme/libnvm/`.

```
  USER (libnvm)                                      KERNEL (snvme.ko)
  ------------                                       -----------------
  open("/dev/snvm_control")          ──ioctl──▶
  SNVM_CHRDEV_CREATE(BDF)                            allocate minor N
                                     ◀──return──    /dev/ssnvme<N> exists
                                                   addr.domain := N

  open("/dev/ssnvme<N>")
  mmap(fd, 0, BAR0_size)             ──mmap──▶
                                     ◀──return──   BAR0 mapped (mm_ptr)
  cudaHostRegister(mm_ptr, IoMemory)

  --- initialise queue rings in user space ---
  for each IO queue i in 0..n_qps-1:
      allocate SQ ring   (cudaMalloc or cudaHostAlloc)
      allocate CQ ring   (cudaMalloc or cudaHostAlloc)
      allocate PRP list  (cudaMalloc or cudaHostAlloc)

  NVM_SET_IOQ_NUM(n_sqs + n_cqs)     ──ioctl──▶    ctrl->ioq_num := N

  for each ring buffer R:
      NVM_MAP_HOST_MEMORY(R, ioq_idx=i, is_cq=…)
      or NVM_MAP_DEVICE_QUEUE_MEMORY ──ioctl──▶    pin pages, fill ioaddrs[],
                                                  ctrl->ioq_map_num++

  NVM_SET_SHARE_REG(1)               ──ioctl──▶    ctrl->use_sreg := 1

  --- now ask the kernel to actually start the controller ---
  SNVM_DEVICE_BIND(BDF)              ──ioctl──▶    pci_register_driver(snvme)
                                                   force-attach BDF
                                                   s_nvme_probe():
                                                     ↳ admin queue (kernel DMA)
                                                     ↳ IO queues use user pages
                                                     ↳ /dev/snvme<X>n<Y> appears
                                     ◀──return──   bind ok

  NVM_GET_DEV_INFO                   ──ioctl──▶    fills nr_user_q,
                                                   max_data_size, block_size,
                                                   disk_name (e.g. "snvme0n1")
                                     ◀──return──

  mount /dev/snvme<X>n<Y> at <mount_path> (regular VFS)
```

Tear-down is the strict reverse; `nvm_ctrl_free()` runs:

```
  NVM_CLEAR_IOQ_NUM        (resets ioq_map_num/cq_num)
  SNVM_DEVICE_UNBIND       (s_nvme_remove → /dev/snvme<X>n<Y> disappears)
  SNVM_CHRDEV_REMOVE       (releases minor)
```

> **Invariant.** `SNVM_DEVICE_BIND` must only be issued **after**
> `ioq_map_num == ioq_num` AND `use_sreg == 1`. Failing this, the
> kernel still binds, but `s_nvme_probe()` falls back to the upstream
> `dma_alloc_coherent` path and the user's queue rings are silently
> ignored — the symptom is "everything looks fine but the doorbells
> don't ring anything".

---

## 6. Build & install

`Makefile.in` is rendered by the top-level CMake (`CMakeLists.txt`,
target `module_output`). To build the module out-of-tree by hand:

```bash
cd build/module
make KERNEL_SRC=/lib/modules/$(uname -r)/build
sudo insmod snvme-core.ko
sudo insmod snvme.ko
ls -l /dev/snvm_control          # should appear, mode 0666
```

Module parameters:

| Param            | Default | Meaning                                 |
| ---------------- | ------- | --------------------------------------- |
| `max_num_ctrls`  | 64      | size of the chrdev minor pool           |

Unload:

```bash
sudo rmmod snvme
sudo rmmod snvme_core
```

`rmmod snvme` will:
- clear all outstanding host/device memory mappings,
- `pci_unregister_driver(snvme_driver)`, releasing every BDF it had
  bound, and
- destroy `/dev/snvm_control`.

Any process that still holds `/dev/ssnvme<N>` open will see further
ioctls fail with `-EBADF` because `ctrl_find_by_inode()` returns NULL.

---

## 7. Porting to a new kernel version

> **Reality check.** The in-tree `drivers/nvme/host/` is refactored
> **almost every LTS cycle** — allocator helpers get renamed, `struct
> nvme_dev` gains fields, admin-queue / tagset setup gets reshuffled,
> `blk_alloc_disk()` vs `blk_mq_alloc_disk()` switch in 5.14, queue
> limits in 6.0, mpath hashing in 6.6, ... A three-way `diff` between
> **old upstream / new upstream / SNVMe fork** is a *starting point*,
> not a complete recipe. What diff misses:
>
> - **Struct-layout surgery.** SNVMe injects ~6 fields into
>   `struct nvme_dev` (`nr_user_allocated_queues`,
>   `nr_user_use_cq`, `user_start_qid`, …). When upstream adds its own
>   fields, merge-three-way puts them in adjacent lines but cannot tell
>   you whether the resulting layout is still consistent with every
>   accessor (some are in inline helpers, some in admin path, some in
>   IO path).
> - **Semantic drift inside "unchanged" helpers.** A function whose
>   signature is identical between versions can acquire new implicit
>   preconditions (e.g. "caller must hold `ctrl->lock`", or "tagset
>   must already be live"). Diff stays silent; you debug at runtime.
> - **Helper refactors that move your hook point.** If upstream
>   replaces three `dma_alloc_coherent` call sites with a single
>   `nvme_alloc_queue_mem()`, the SNVMe `use_sreg` branch has to follow
>   the new split point — sometimes by hooking inside the new helper,
>   sometimes by not calling it at all.
> - **Lock-order / flow changes.** `nvme_reset_work` and `nvme_probe`
>   in particular are re-ordered every other release. A patch that
>   re-applies cleanly can still run with the wrong IRQ set up.
>
> Treat the steps below as the minimum; the **Phase 3 semantic audit**
> is the one people skip and regret.

### 7.1 Phase 1 — Mechanical merge (this is what `diff` gets you)

- [ ] **Anchor the old baseline.** Write down the exact upstream
      tag SNVMe currently tracks. Grep for it:
      `git log --grep="nvme: " drivers/nvme/host/ | head` in that tag's
      tree. Save the three SHAs that touched `pci.c`, `core.c`,
      `multipath.c` most recently — they're your 3-way-merge left side.
- [ ] **Sync the new baseline.** Check out `drivers/nvme/host/` from
      the target kernel tag (`v6.x`). This is the merge right side.
- [ ] **3-way merge into SNVMe.** For each file SNVMe carries
      (`core.c`, `pci.c`, `ioctl.c`, `multipath.c`, `zns.c`, `hwmon.c`,
      `fabrics.c`, `rdma.c`, `tcp.c`, `nvme.h`), run a 3-way merge
      (old-upstream → new-upstream → SNVMe-fork). **Never** rebase
      by just applying the `old→new` upstream patch on top of SNVMe
      — conflict resolution without the fork as the third input hides
      struct-layout bugs.
- [ ] **Re-apply renames (§3.1)** on anything upstream added.
      `grep -n '\bnvme_[a-z_]*\(' snvme/*.c` — any new match that is
      also exported or referenced cross-module gets a `snvm_` /
      `s_nvme_` prefix.

### 7.2 Phase 2 — Structural surgery (diff cannot do this for you)

- [ ] **Audit `struct nvme_dev` layout.** Open `pci.c` side-by-side
      with the **new** upstream `nvme.h` definition. Every SNVMe-added
      field should go *after* all upstream fields (ABI doesn't matter
      inside the module, but bisecting crashes later is easier if
      SNVMe additions are in one contiguous block). Any upstream field
      that was removed must be removed from SNVMe access sites too.
      Confirm with `pahole snvme.ko` if you can.
- [ ] **Audit `struct ctrl`** (`snvme/ctrl.h`). This one is
      SNVMe-owned so upstream won't touch it, but verify
      `ctrl->pdev` lifetime matches the new probe/remove ordering
      (see §5 and §7.3).
- [ ] **Relocate the `use_sreg` branch (§3.2).** In the NEW upstream
      tree, find the call site that allocates SQ/CQ memory for IO
      queues. Call-graph to trace from:
      `nvme_probe → nvme_setup_io_queues → nvme_create_io_queues → nvme_alloc_queue → …dma_alloc_coherent…`
      Whatever that chain looks like in the target kernel, the
      `ctrl->use_sreg && ioq_num == ioq_map_num` test has to sit
      **immediately before** the `dma_alloc_coherent` it is replacing.
      Not one layer up, not one layer down.
- [ ] **Audit every `pci.c` branch that references user-allocated
      fields.** At time of writing these are
      `nr_user_allocated_{queues,cq,sq}`, `nr_user_use_{cq,sq}`,
      `user_start_qid`, `online_user_queues`, `max_qid`,
      `use_user_allocated`. If `nvme_setup_io_queues` changed how
      `nr_io_queues` is computed, each arithmetic site must be
      re-derived from first principles (see `pci.c:2470–2492`).
- [ ] **Verify admin-queue path is untouched.** The user-pages branch
      MUST only apply to IO queues. If upstream merged admin+IO queue
      allocation, split them back out in SNVMe.
- [ ] **Verify block-device registration still uses
      `"snvme%dn%d"`** (`core.c:3806`, `multipath.c:58/62`). Upstream
      naming of `disk->disk_name` has been touched by several
      releases; don't let a merge silently revert it to `"nvme..."` —
      that breaks the §2 namespace-separation guarantee.

### 7.3 Phase 3 — Semantic audit (this is where bugs actually live)

- [ ] **Re-read every upstream commit message** on `drivers/nvme/`
      between the old and new baseline. In practice:
      `git log --oneline v<OLD>..v<NEW> -- drivers/nvme/host/ | wc -l`
      — if this is more than ~50, budget at least a day. Flag any
      commit whose subject contains `lock`, `refcount`, `probe`,
      `reset`, `queue`, `tagset`, `irq`, `reinit`, or `remove` — these
      are the ones that silently change SNVMe's assumptions.
- [ ] **Lock-order check.** In the new kernel, walk `nvme_probe` and
      `nvme_reset_work` top-to-bottom and verify that every mutex /
      rw_sem SNVMe touches (admin_q, shutdown_lock, namespaces_rwsem,
      subsys_lock, `snvm_control_lock`) is still taken in the same
      order relative to each other. New kernels occasionally move
      `mutex_lock` calls across function boundaries.
- [ ] **Reset / live-migration path.** Run `nvme reset-controller
      /dev/snvme0n1` after a successful bind. If the reset path in
      the new kernel reallocates IO queues, the `use_sreg` flag
      must be re-honored — otherwise the reset silently falls back to
      `dma_alloc_coherent` and your SQ/CQ pointers on the GPU go
      stale. This is a common regression and **smoke tests will NOT
      catch it**; write a dedicated reset test.
- [ ] **IRQ affinity.** If `nvme_setup_irqs` /
      `pci_alloc_irq_vectors_affinity` changed, verify that the MSI-X
      vectors for user-allocated queues are still routed to the GPU's
      NUMA node (or at least not pinned to a CPU that disagrees with
      what `init_queues` assumed).
- [ ] **`nvidia_p2p_*` compatibility.** These are loaded from the
      proprietary NVIDIA driver and are sensitive to **driver
      version**, not kernel version. But the function-signature
      shims in `snvme/nvfs-p2p.c` + `nvfs-pci.{c,h}` are
      kernel-version-sensitive (they use `get_user_pages_fast`-family
      helpers whose signatures shift). Rebuild the NVIDIA driver
      against the new kernel first, then SNVMe on top.

#### 7.3.1 Known regression traps (re-audit these every uplift)

This is a list of bugs that have been found and fixed in SNVMe in the
past — they are easy to reintroduce during a 3-way merge because the
surrounding code changes but the **bug pattern** is invisible to diff.
Re-audit each one after §7.1:

- **`svm_mmap_registers` null-check must be `||`, not `&&`.**
  (`snvme/pci.c`.) `ctrl_find_by_inode()` can legitimately return
  `NULL`; if it does, the `&&` form then dereferences it. A merge
  conflict in this function has historically lost the fix.
- **`snvm_dev_map_ioctl` `ret` must be initialised at declaration.**
  (`snvme/pci.c`.) Several `case` branches return through
  `ret = ...; break;`, but a few happy paths (`NVM_UNMAP_*`,
  `NVM_SET_SHARE_REG`, `NVM_CLEAR_IOQ_NUM`) used to fall through to
  the final `return ret;` without setting `ret`, leaking stack
  garbage as the ioctl return value. Keep `int ret = 0;` at the top
  AND set `ret = 0;` on every success `break;` — the redundancy is
  the point.
- **`NVM_MAP_*` must check `IS_ERR_OR_NULL(map)` *before* any
  dereference or counter bump.** (`snvme/pci.c`.) The tempting
  shape "bump `ioq_map_num` → check bound → write `map->ioq_idx`"
  oopses when `map_userspace()` / `map_device_ioqueue_memory()`
  return an `ERR_PTR`. When helpers are refactored in an uplift the
  check-after-deref pattern often creeps back. The correct order
  is: (1) call the mapper, (2) `IS_ERR_OR_NULL` guard, (3) bound
  check, (4) commit state, (5) `copy_to_user` with rollback-on-fail.
- **`ioq_map_num` counter must roll back on every failure path.**
  (`snvme/pci.c` `NVM_MAP_*` + `copy_to_user` error branches.)
  Failing to roll back poisons the `use_sreg` gate: subsequent
  `SNVM_DEVICE_BIND` sees `ioq_map_num > ioq_num` and silently falls
  back to `dma_alloc_coherent` (the exact Phase-3-class bug).
- **`snvm_chrdev_helper(remove)` teardown order:
  `ctrl_put()` FIRST, `ida_simple_remove()` SECOND.** (`snvme/pci.c`.)
  `ctrl_put()` uses `ctrl->number` internally (`device_destroy()` /
  `cdev_del()` through the minor-encoded `dev_t`). Returning the
  minor to the IDA pool first opens a window where a concurrent
  `SNVM_CHRDEV_CREATE` picks up the same minor and races our still-
  live `cdev`. Same rule for `snvm_chrdev_create()`'s error unwind.
- **`NVM_MAP_DEVICE_QUEUE_MEMORY` must reject `ioq_idx < 0` *before*
  calling `map_device_ioqueue_memory()`.** Otherwise you pay the cost
  of `nvidia_p2p_get_pages()` (which is slow and can fail partially)
  only to throw the result away on the next line.
- **`NVM_MAP_DEVICE_MEMORY` (data path) must `unmap_and_release()`
  on `copy_to_user` failure.** Otherwise a crash in userspace between
  `ioctl()` and receiving the IO addresses leaks pinned GPU pages
  for the lifetime of the module.
- **`nvme_probe()` must gate on `ctrl_find_by_pci_dev(&ctrl_list, pdev) != NULL`
  at the very top, returning `-ENODEV` otherwise.** (`snvme/pci.c`.)
  `pci_register_driver(&snvme_driver)` inside `snvm_rebind_driver()`
  asks the PCI core to call `.probe()` for **every** matching NVMe on
  the host, not just the BDF the user passed to `SNVM_DEVICE_BIND`.
  Without the gate, the *first* `SNVM_DEVICE_BIND` on a multi-NVMe
  host hijacks every unbound NVMe it finds, even ones the in-tree
  `nvme` driver was supposed to own. Worse, those extra NVMes have
  no `ctrl` record, so they skip the `use_sreg` branch and come up
  on kernel-DMA queues — looking "fine" to probe but broken for
  SNVMe IO. Keep the gate at the top of `nvme_probe()`; do NOT
  collapse it into the later "`if (ctrl && ...)`" check.
- **User queue indices (`ioq_idx`) are 0-based.** `nvme_create_io_queues_mix()`
  (`pci.c`) walks the user queues with `count = 0; i = online_queues ..`
  and calls `map_find_by_pci_dev_and_idx(list, pdev, uqid=count, is_cq=1)`.
  If userspace registers its first SQ/CQ with `ioq_idx=1`, the lookup
  misses and probe dies with `map_find_by_pci_dev_and_idx cq error!`.
  libnvm gets this right (`queue.h` passes `qp_id` starting at 0);
  anything that talks to SNVMe directly (smoke tests, external tools)
  MUST start `ioq_idx` at 0 too.
- **`NVM_SET_IOQ_NUM` field-name landmine: `request.is_cq` is *not* a
  CQ flag here, it's the `on_host` flag.** (`pci.c` `NVM_SET_IOQ_NUM`
  case copies `request.is_cq` into `ctrl->on_host`.) Later, in
  `nvme_create_user_queue()`, `dev->queue_on_host` decides which list
  is searched:
  ```
  if (dev->queue_on_host) list = &host_list;
  else                    list = &device_queue_list;
  ```
  So a userspace program that pins SQ/CQ rings via `NVM_MAP_HOST_MEMORY`
  MUST pass `is_cq = 1` (= on_host=1) to `NVM_SET_IOQ_NUM`, otherwise
  probe silently looks in `device_queue_list`, fails to find the rings,
  and returns the same `map_find_by_pci_dev_and_idx cq error!` as the
  off-by-one trap above. The two failure modes look identical in dmesg
  but are independent. Same applies to `NVM_MAP_HOST_MEMORY` /
  `NVM_MAP_DEVICE_QUEUE_MEMORY` — pick the ioctl that matches your
  `on_host` decision.

None of these are detected by the smoke tests as written — the
smoke tests run the happy path. They are detected by (a) reading
this list during the merge, and (b) the **reset + stress** workload
described in §7.4.

### 7.4 Phase 4 — Verification (the mandatory gate)

- [ ] `snvme_smoke` returns 0. *Necessary, not sufficient.*
- [ ] `snvme_smoke_gpu` returns 0 (with `--bind` on a throw-away
      NVMe). Exercises the `NVM_MAP_DEVICE_*` paths that Phase 3
      lock-order bugs manifest in.
- [ ] **Reset test.** After `snvme_smoke_gpu --bind` succeeds,
      issue `nvme reset-controller` against the resulting
      `/dev/snvmeXnY`. It must either complete cleanly or fail
      loudly; **silent fallback to kernel-DMA queues is a bug**.
- [ ] **Co-existence test.** With both `nvme.ko` and `snvme.ko`
      loaded, bind one NVMe to each, mount both, `fio` them
      simultaneously for ≥5 min. The two drivers use **disjoint**
      block-device namespaces (in-tree → `/dev/nvme*`, SNVMe →
      `/dev/snvme*`), so collisions on the device-node side should not
      happen — verify that, and watch for kernel oopses or shared
      workqueue/IRQ name clashes (`s_nvme_wq` vs `nvme_wq`, etc.).
- [ ] **Stress test under reset.** Run the co-existence workload
      while looping `nvme reset-controller /dev/snvme0n1` every 30 s
      for 10 iterations. If this stays clean, you've caught most
      Phase 3 semantic drift.

### 7.5 How far to go?

For a **patch-level** uplift (5.15.x → 5.15.y): Phase 1 + 4 usually
suffices. For a **minor-version** uplift (5.15 → 5.19): Phase 1–3
are all required. For a **cross-LTS** uplift (5.15 → 6.6): assume
Phase 2 and Phase 3 together are a week of work; do NOT skip the
reset test. If upstream merged a large NVMe refactor (e.g. the
`queue_limits` transition in 6.0), plan for a rewrite of the
`use_sreg` branch from scratch, not a patch re-apply.

---

## 8. Sanity test programs

Two end-to-end tests, both self-contained (no Geminifs filesystem, no
gRPC daemon). They live at:

```
backends/local/kernel_modules/test/snvme_smoke.c        # libc-only, no CUDA
backends/local/kernel_modules/test/snvme_smoke_gpu.cu   # adds the GPU paths
backends/local/kernel_modules/test/run_snvme_smoke.sh
backends/local/kernel_modules/test/Makefile
```

| Binary             | Covers                                                                  | Built when     |
| ------------------ | ----------------------------------------------------------------------- | -------------- |
| `snvme_smoke`      | `NVM_MAP_HOST_MEMORY` path + chrdev create/remove + BAR0 mmap           | always         |
| `snvme_smoke_gpu`  | adds `NVM_MAP_DEVICE_MEMORY` and `NVM_MAP_DEVICE_QUEUE_MEMORY` (cudaMalloc + nvidia_p2p_get_pages) | when `nvcc` is on `$PATH` |

Both binaries support a default **UAPI-smoke** mode (does not trigger
`s_nvme_probe()`, completely safe to run while another NVMe is mounted)
and a `--bind` mode that additionally runs the destructive bring-up.

Run via the wrapper:

```bash
cd backends/local/kernel_modules/test

# host (libc) UAPI smoke -- safe even on production hosts
./run_snvme_smoke.sh

# GPU path UAPI smoke -- requires NVIDIA driver loaded, still safe
./run_snvme_smoke.sh --gpu
./run_snvme_smoke.sh --gpu --gpu-id 1            # pick CUDA device 1

# full bring-up (destructive: detaches in-tree nvme from the BDF)
./run_snvme_smoke.sh --bind
./run_snvme_smoke.sh --gpu --bind
```

Either binary exits with code `0` only when every UAPI step round-trips
cleanly. Any failure prints `[FAIL] step=<N> ... errno=<E>` and stops.

> **Tip.** During a kernel uplift, run `snvme_smoke` first. Only when
> it passes should you try `snvme_smoke_gpu` — a failure there usually
> means the NVIDIA driver / `nvfs_nvidia_p2p_*` glue is broken
> (kernel-side issue lives in `snvme/nvfs-p2p.c`), not the SNVMe core.
