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
4. **`snvme-rename.sed` only rewrites C identifiers; it cannot touch
   `printf`-style format strings.** Every site that constructs a
   device, IRQ, workqueue, sysfs class or chrdev region name from a
   literal must be renamed by hand. The complete list — re-audit
   after every uplift:

   `core.c` (in `nvme_core_init` and friends):
   - `alloc_workqueue("snvme-wq", ...)`,
     `alloc_workqueue("snvme-reset-wq", ...)`,
     `alloc_workqueue("snvme-delete-wq", ...)`
     — workqueues are exposed under
     `/sys/devices/virtual/workqueue/` because of `WQ_SYSFS`; a
     duplicate name makes `alloc_workqueue()` fail with
     `kobject_add_internal failed for nvme-wq with -EEXIST` and
     `insmod` aborts in `nvme_core_init`.  This is the **first**
     symptom of a baseline that forgot the §2 string-literal
     audit.
   - `alloc_chrdev_region(..., "snvme")`
     — owner tag in `/proc/devices`; does not fail on duplicates,
     but the two modules end up sharing one line and udev rules
     that match on the chrdev name break.
   - `class_create(THIS_MODULE, "snvme")` and
     `class_create(THIS_MODULE, "snvme-subsystem")`
     — sysfs class names under `/sys/class/`; behavior on
     duplicates is kernel-version dependent (silent shared-pointer
     on some, `EEXIST` on others — never rely on either).
   - `dev_set_name(ctrl->device, "snvme%d", ...)` — per-controller
     sysfs name.
   - `dev_set_name(&subsys->dev, "snvme-subsys%d", ...)` — per-
     subsystem sysfs name (the class is already separate, so this
     is for grep-friendliness and uniform "snvme..." output, not
     a hard collision).

   `multipath.c`:
   - `sprintf(disk_name, "snvme%dn%d", ...)` (non-multipath fallback
     and multipath head),
     `sprintf(disk_name, "snvme%dc%dn%d", ...)` (hidden multipath
     leg).
     A leftover `"nvme%dn%d"` collides with the in-tree
     `/dev/nvme0n1`; `device_add_disk()` then fails and probe
     unwinds.

   `nvme.h`:
   - the non-multipath inline fallback for `nvme_set_disk_name()`:
     `sprintf(disk_name, "snvme%dn%d", ...)`.  Same hazard as
     above, only on kernels built without `CONFIG_NVME_MULTIPATH`.

   `pci.c`:
   - `pci_request_irq(..., "snvme%dq%d", ...)` — IRQ description
     string in `/proc/interrupts`; duplicates merely confuse
     debugging, no hard failure.
   - `pci_request_mem_regions(pdev, "snvme")` — `/proc/iomem`
     owner tag; same effect.

   **Do NOT** rename the NVMe wwid prefix used by the `wwid_show`
   sysfs attribute (`"nvme.%04x-..."` in `core.c`).  That prefix is
   part of the NVMe userspace contract (udev / multipath-tools
   matches on it); keep it byte-for-byte identical to upstream.

   Missing any one of the renamable sites above silently
   re-introduces a `/dev`, `/proc/interrupts` or workqueue-sysfs
   collision with the in-tree `nvme.ko`.  This regressed on the
   `snvme-5.4.241-1-tlinux4-0017` baseline initial port (the
   upstream-5.4 literals were carried verbatim) and was only
   caught by trying to `insmod` while `nvme.ko` was already loaded.

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

### 6.1 Module signing on locked-down kernels

Deployment kernels that ship with `CONFIG_MODULE_SIG_FORCE=y` (e.g.
TencentOS Server 5.4.241-1-tlinux4-0017) reject every unsigned
module with `Loading of unsigned module is rejected` and
`insmod: ... Required key not available`.  Three observations
matter for porting:

1. `CONFIG_MODULE_SIG_FORCE=y` is a **compile-time** enforcement.
   It cannot be cleared at runtime via `sysctl
   kernel.modules_sig_enforce`, kernel cmdline `module.sig_enforce=0`,
   `insmod --force`, or a Secure Boot toggle — those knobs only
   apply to `CONFIG_MODULE_SIG_FORCE=n` kernels.  Confirm with:

   ```bash
   grep CONFIG_MODULE_SIG_FORCE /boot/config-$(uname -r)
   ```

2. Self-signing requires either
   `CONFIG_SECONDARY_TRUSTED_KEYRING=y` plus a writable secondary
   keyring (Secure Boot + MOK, or `keyctl add asymmetric` if
   integrity policy allows), OR access to the CA whose public
   half is baked into `.builtin_trusted_keys`.  On the TencentOS
   image above neither holds, so the only production path is the
   central signing service (kmod upload → signed kmod download
   → `insmod`).
3. For **active porting work** (editing snvme baselines, running
   the §7.4 verification gate), prefer a development host whose
   running kernel does not set `CONFIG_MODULE_SIG_FORCE=y` (any
   stock mainline kernel, the upstream `temp/kernel-5.4.241-1.0017.7`
   rebuilt with `CONFIG_MODULE_SIG_FORCE=n`, etc.).  The signing
   workflow is a deployment concern, not a porting concern, and
   trying to iterate on snvme with a "edit → build → upload →
   wait → download → insmod → dmesg" loop is impractical.

The same `scripts/sign-file` helper that ships with the kernel
build tree is used in all signing workflows:

```bash
/usr/src/kernels/$(uname -r)/scripts/sign-file \
    sha256 <priv_key.pem> <pub_key.x509> snvme-core.ko
/usr/src/kernels/$(uname -r)/scripts/sign-file \
    sha256 <priv_key.pem> <pub_key.x509> snvme.ko
```

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
Re-audit each one after §7.1.

> **Per-baseline status note.** Where a trap was specifically verified
> against a baseline different from `snvme-5.15.0-public`, the affected
> file calls it out at the patch site with a `PORTING.md §7.3.1` cross
> reference. The `snvme-5.4.241-1-tlinux4-0017` baseline was audited
> in full against this list and additionally fixes traps #4, the
> `NVM_MAP_DEVICE_MEMORY` `copy_to_user` leak, and the
> **`snvm_dev_fops` missing `.release`** hook (all three of which are
> still latent in `snvme-5.15.0-public`); see that directory's `pci.c`
> banner for the full bug-fix list. When uplifting to a new kernel,
> diff against `snvme-5.4.241-1-tlinux4-0017/pci.c` for the cleanest
> version of these fixes.

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
  back to `dma_alloc_coherent` (the exact Phase-3-class bug). The
  rollback set is: `ctrl->ioq_map_num--`, `ctrl->cq_num--` if
  `map->is_cq`, then `unmap_and_release(map)`. This applies to both
  the budget-overflow path (just after `ioq_map_num += 1`) AND the
  final `copy_to_user(request.ioaddrs, ...)` path. Fixed in
  `snvme-5.4.241-1-tlinux4-0017/pci.c` (`NVM_MAP_HOST_MEMORY` and
  `NVM_MAP_DEVICE_QUEUE_MEMORY` cases); still latent in
  `snvme-5.15.0-public/pci.c`.
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
  for the lifetime of the module. Fixed in
  `snvme-5.4.241-1-tlinux4-0017/pci.c`; still latent in
  `snvme-5.15.0-public/pci.c`.
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
- **`snvm_dev_fops` MUST have `.open` + `.release` hooks so that an
  abnormal userspace exit cannot leak host pins, GPU p2p references,
  or per-ctrl IO-queue accounting counters.** Upstream snvme-5.15.0
  and the original 5.4 port ship `snvm_dev_fops` with only `.owner +
  .unlocked_ioctl + .mmap` — there is no automatic cleanup if the
  process holding `/dev/ssnvme*` open dies between `NVM_MAP_*` and the
  matching `NVM_UNMAP_*`. Symptoms on TencentOS 5.4.241 (reproducible
  in `/var/log/messages`):

    1. Next `SNVM_DEVICE_BIND` after a test crash logs
       `snvme: ctrl exist, ioq_num=N cq_num=M map_num=K` — the
       controller is reused **dirty**, with counters carried over
       from the dead process.
    2. `nvidia.ko` refcount accumulates because nobody calls
       `nvidia_p2p_put_pages()`; eventually `rmmod snvme` says
       "module in use" forever and the box requires a reboot.

  Note: the `snvme: snvme_find_get_ns(nsid=1) failed` 3x log line
  often appears nearby in `/var/log/messages` but is a **separate**
  bug (the `NVM_GET_DEV_INFO` vs `nvme_scan_work` race documented in
  the next trap entry). The two are independently reproducible and
  must be fixed independently — the dirty-rebind path makes the
  scan race **more likely** by short-circuiting the probe-side
  delays, but the scan race exists on a fresh module load too.

  Fix (recorded at `snvme-5.4.241-1-tlinux4-0017/pci.c` ~lines
  4749-4920 and `map.c` `map_purge_by_owner`):

  - `.open` allocates a `struct snvm_dev_owner { ctrl, owner }` and
    stashes it in `file->private_data`. Capturing the owner at open
    time (not at release time) is critical — by the time
    `__fput()` runs, `current` may be a different thread group
    member or a forked child, while `map->owner` was set to the
    process that issued the `NVM_MAP_*` ioctl.
  - `.release` walks `host_list / device_queue_list` once to compute
    the rollback deltas for `ctrl->ioq_map_num` and `ctrl->cq_num`,
    then calls `map_purge_by_owner(list, owner)` against all three
    map lists. Use **checked subtraction** for the counter rollback
    (a buggy userspace path can leave counters in a state where
    `rb_*` exceeds the current value; clamp to zero rather than
    underflow into UINT_MAX, which would then disable the `use_sreg`
    branch on the next bind).
  - The split `pass-1 count / pass-2 free` is mandatory because
    `unmap_and_release()` does `list_remove()` on the descriptor;
    saving a `next` pointer across the call would dereference a
    freed node. `map_purge_by_owner` re-fetches `list_next(&head)`
    after every free for the same reason.

  Re-audit rule: any uplift that touches `snvm_dev_fops`,
  `struct map`, or the `ioq_map_num` / `cq_num` accounting MUST
  re-verify that these two hooks still fire — a single
  `kill -9 <pid>` against the smoke test, immediately followed by
  `cat /sys/module/snvme/refcnt` and `lsof /dev/ssnvme0`, is the
  fastest manual probe.

- **`NVM_GET_DEV_INFO` MUST wait for `nvme_scan_work` to finish
  before returning `snvme_find_get_ns(nsid=1) failed`.** `pci.c`
  `snvme_start_ctrl()` -> `nvme_queue_scan()` -> `queue_work(s_nvme_wq,
  &ctrl->scan_work)` is asynchronous: the worker is the only code
  path that calls `nvme_alloc_ns()` and `list_add_tail(&ns->list,
  &ctrl->namespaces)`. `snvm_rebind_driver` finishes (and userspace
  gets back from `SNVM_DEVICE_BIND` -> `ioctl()`) at the moment
  `device_attach` returns, which is **before** `scan_work` has even
  started in many cases. Userspace then immediately issues
  `NVM_GET_DEV_INFO` and `snvme_find_get_ns` walks an empty
  `namespaces` list, returning NULL.

  Symptom (TencentOS 5.4.241, `/var/log/messages` 2026-05-18
  16:11:12 and 19:21:46): every BIND logs **exactly three**
  consecutive `snvme: snvme_find_get_ns(nsid=1) failed` lines —
  the "three" comes from libnvm's caller-side retry loop in
  `device.cpp`. The 3x retries finish well within the
  `scan_work` window, so all three observe an empty list.

  Fix (`snvme-5.4.241-1-tlinux4-0017/pci.c` `NVM_GET_DEV_INFO`
  case): on first lookup failure, call `flush_work(&ndev->ctrl.scan_work)`
  (no-op if the work was never queued — `flush_work` documents this)
  and retry; then if still NULL, poll with `msleep(50)` +
  `flush_work` for up to 5 s before returning `-EFAULT`. The bound
  preserves caller EFAULT semantics if the controller is actually
  broken (admin queue dead, state never reached `NVME_CTRL_LIVE`,
  etc.).

  Pitfall to avoid: do NOT "fix" this in userspace by adding more
  retry layers in libnvm. The kernel side has the synchronisation
  primitive (`flush_work`) and the access to `ctrl->scan_work`;
  userspace can only sleep blindly and hope, which is what created
  the 3-retries-but-all-too-fast pattern visible in the logs.

- **`snvm_rebind_driver` MUST use `driver_attach(&snvme_driver.driver)`,
  NOT `device_attach(&pdev->dev)`.** This is a 5.4-specific landmine.
  `device_driver_attach()` (used by snvme-5.15.0) does not exist on
  5.4, and the obvious substitute `device_attach()` has subtly wrong
  semantics for our use case: `device_attach` walks the device's bus
  callback `__device_attach`, which iterates *all matching drivers*
  and picks the **first** registered one. On a TencentOS host the
  in-tree `nvme.ko` is loaded at boot, so it is always the first
  match, and `device_attach()` silently rebinds the BDF to the
  in-tree driver. dmesg signature (with the old code):

  ```
  snvme: binding nvme device to snvme: pci 0:8:0.0
  nvme nvme0: pci function 0000:08:00.0          <-- nvme, not snvme!
  nvme nvme0: 135/0/0 default/read/poll queues   <-- 3-tuple
  ```

  Note the absence of the `snvme: ctrl exist, ioq_num=...` line that
  a successful snvme bind emits, and the absence of the
  `snvme: device driver name: snvme` confirmation line. The follow-up
  `SNVM_DEVICE_UNBIND` then trips the "device's driver is not snvme"
  branch (-EFAULT on the old code; now -EINVAL after a separate
  errno-cleanup fix).

  The reproducer that catches this every time:

  ```
  ./run_snvme_smoke.sh 0000:08:00.0 --gpu              # leaves BDF "loose"
  ./run_snvme_smoke.sh 0000:08:00.0 --gpu --bind       # fails at step 15
  ```

  Fix: replace `device_attach` with a bounded retry loop calling
  `driver_attach(&snvme_driver.driver)`. `driver_attach` walks the
  bus's device list and invokes the SPECIFIC driver's probe on every
  unbound matching device. The `nvme_probe()` per-BDF gate
  (`ctrl_find_by_pci_dev(&ctrl_list, pdev) != NULL`) ensures the
  effect is scoped to the BDF the user already CHRDEV_CREATEd; every
  other NVMe on the bus short-circuits to `-ENODEV` at the top of
  probe. The retry is needed because, between `device_release_driver`
  and `driver_attach`, udev's drivers_autoprobe rule may rebind the
  device to the in-tree nvme — three attempts is enough in practice;
  if udev wins three times in a row the host has a misconfigured
  autoprobe rule and `-EBUSY` is the honest answer.

  Verification: after a successful BIND the dmesg block should
  contain `snvme: device driver name: snvme` and `snvme snvme0: pci
  function ...` (note the doubled `s` in the device name); the
  follow-up `default/read/poll/user queues` line MUST be the
  4-tuple variant (`135/0/0/0`), not 3-tuple.

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

### 8.1 Build/run troubleshooting cheat sheet

| Symptom (where it surfaces) | Root cause | Fix |
|---|---|---|
| `nvcc fatal : Unsupported gpu architecture 'sm_XX'` at `make` time | The Makefile auto-detects CUDA_ARCH from the running GPU's compute capability via `nvidia-smi --query-gpu=compute_cap`.  Auto-detect fails (no GPU visible / driver not loaded) or the toolkit is too old/new for the detected arch (e.g. CUDA 13 dropped `sm_70`). | Pass `CUDA_ARCH` explicitly: `make CUDA_ARCH=sm_80` (A100, accepted by CUDA 11.0–13.x) or `make CUDA_ARCH=sm_90` (H100/H200/H20). |
| `[FAIL] step=1  cudaGetDeviceCount -> system not yet initialized` at runtime, with `nvidia-smi -L` listing GPUs just fine | NVSwitch-equipped multi-GPU host (HGX H100/H200/H20 boards expose `/dev/nvidia-nvswitch*`).  CUDA runtime refuses `cuInit()` until `nvidia-fabricmanager` finishes the NVLink topology bring-up.  `nvidia-smi -L` does NOT need fabricmanager and so does not catch this. | `sudo systemctl enable --now nvidia-fabricmanager`.  If the service is missing, install the package matching your driver exactly: `sudo dnf install nvidia-fabric-manager-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \| head -n1)`. |
| `[FAIL] step=1 cudaGetDeviceCount -> system not yet initialized` even after fabricmanager is active | NVLink Inband mode (H20 / H100 8-GPU NVL3 hosts): the GPU half of the NVLink handshake hasn't completed.  **Authoritative success signal** (verified on HGX H20 / driver 580.65.06): every GPU's "Fabric" block in `nvidia-smi -q` shows `State : Completed` and `Status : Success`.  Notes: (1) `GPU Fabric GUID : N/A` is **not** a failure indicator on this hardware -- some firmware/driver combos legitimately leave the GUID field N/A even on a healthy fabric.  (2) `Persistence-Mode = Disabled` is **also not** a failure indicator on a freshly rebooted host -- verified 2026-05-19 on HGX H20: PM Disabled across all 8 GPUs, Fabric all Completed, CUDA programs run fine.  PM only matters as a **recovery knob** when nvidia-uvm has been poisoned by a previous failed-cuInit / killed-CUDA-process refcount leak; in that case `nvidia-smi -pm 1` keeps the driver context resident long enough for fabricmanager's retry to complete.  Detect with: `nvidia-smi -q \| awk '/^    Fabric$/,/^$/' \| grep State` (every line should read "Completed"); `nvidia-smi --query-gpu=persistence_mode --format=csv,noheader` is informational only. | If Fabric is incomplete: `sudo nvidia-smi -pm 1 && sudo systemctl restart nvidia-fabricmanager`.  If that still fails, the nvidia kernel modules are likely in a poisoned half-state (typically caused by a previous CUDA process that crashed mid-cuInit and leaked a refcount into nvidia-uvm); the only known recovery is to kill every CUDA-touching process on the host, `rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia` (in that order), then reinstall the NVIDIA driver from its `.run` payload.  To make PM persistent across reboot (only useful as a defensive measure on hosts that are known to crash CUDA processes), install an `nvidia-persistenced.service` systemd unit -- the driver `.run` does not install one on TencentOS by default. |
| **Misleading symptom note**: `strings /lib64/libcuda.so.1 \| grep -E '^[0-9]+\\.[0-9]+\\.[0-9]+$'` is NOT a valid way to verify libcuda's own version on driver >= 575.x.  The integers it returns are the *compatibility table* (what older client drivers this libcuda accepts), not the library's own version.  A 580.65.06 libcuda legitimately shows `575.57.07` as its highest match.  Use the file's hash against the matching `.run` payload, or trust `nvidia-smi --query-gpu=driver_version`, which queries the kernel module directly. | -- | -- |
| `insmod: Required key not available` / `Loading of unsigned module is rejected` | Deployment kernel built with `CONFIG_MODULE_SIG_FORCE=y` (TencentOS 5.4.241-1-tlinux4-0017).  See section 6.1 above. | Sign through the deployment signing service, or develop on a kernel without `CONFIG_MODULE_SIG_FORCE=y`. |
| `kobject_add_internal failed for nvme-wq with -EEXIST` at `insmod snvme-core.ko` time | Leftover `"nvme-wq"` / `"nvme-reset-wq"` / `"nvme-delete-wq"` string literal not renamed to `"snvme-*"`; collides with in-tree `nvme-core.ko`.  Section 2 string-literal rename rule missed.  Re-audit using the checklist in section 2 item 4. | Apply the rename to every workqueue / chrdev region / class literal listed in section 2 item 4.  Re-`insmod`. |
| `map_find_by_pci_dev_and_idx cq error!` in dmesg during probe | Two distinct off-by-one bugs share this dmesg line: (a) user `ioq_idx` starts at 1 instead of 0 (section 7.3.1 trap #8); (b) userspace called `NVM_MAP_HOST_MEMORY` but passed `is_cq != 1` to `NVM_SET_IOQ_NUM` so the kernel searches the wrong queue list (trap #9). | Re-read PORTING.md section 7.3.1 traps #8 and #9; verify libnvm / smoke-test caller matches the on_host vs device_queue split. |
| `snvme: snvme_find_get_ns(nsid=1) failed` (exactly 3x per BIND) | `NVM_GET_DEV_INFO` ioctl races `nvme_scan_work`. Userspace gets back from `SNVM_DEVICE_BIND` at the moment `device_attach` returns, but `nvme_alloc_ns()` (the only path that puts `nsid=1` on `ctrl->namespaces`) runs asynchronously on `s_nvme_wq` after `snvme_start_ctrl()`. libnvm's 3-retry loop in `device.cpp` finishes inside the race window. **Independent of** any dirty-rebind / `.release` issue — reproduces on a fresh module load. Often *correlated* with a `snvme: ctrl exist, ioq_num=N cq_num=M map_num=K` line just above it (which is the separate dirty-rebind symptom of the `.release` bug). | The fix is on the **kernel side**, not userspace: `NVM_GET_DEV_INFO` must `flush_work(&ndev->ctrl.scan_work)` + bounded poll before declaring failure. See §7.3.1 trap "`NVM_GET_DEV_INFO` MUST wait for `nvme_scan_work`". Verify by grepping the BIND-time dmesg block for `NVM_GET_DEV_INFO: nsid=1 ready after N ms scan wait` (info log emitted on slow-path success). |
| `rmmod snvme` says `module is in use` long after every `/dev/ssnvme*` user has exited, with `lsmod` showing `Used by 0` but the refcount in `/sys/module/snvme/refcnt` non-zero | A process died holding pinned p2p / host pages, the original `snvm_dev_fops` had no `.release` hook, so the refs leaked into `nvidia.ko`. See §7.3.1 trap "`snvm_dev_fops` MUST have `.open` + `.release` hooks". | Reboot is the only safe recovery on a host without the `.release` fix applied. With the fix in place this should be impossible — open a bug if it recurs. |
| `[FAIL] step=15 SNVM_DEVICE_UNBIND ... errno=14 (Bad address)` (or with the errno fix: `errno=22 (Invalid argument)`), dmesg shows `snvme: device's driver is '...nvme', not 'snvme'` and the BIND-time block contains `nvme nvme0: pci function ...` (in-tree, not snvme) | `snvm_rebind_driver` called the 5.4 helper `device_attach()` which picks the **first** registered matching driver. Since in-tree `nvme.ko` is loaded at boot it always wins, and the BIND silently rebinds to nvme.ko — the next UNBIND then refuses because the driver isn't snvme. Reproducible by running `--gpu` (no bind) immediately followed by `--gpu --bind` on the same BDF. See §7.3.1 trap "`snvm_rebind_driver` MUST use `driver_attach`". | Confirm `snvm_rebind_driver` uses `driver_attach(&snvme_driver.driver)` + bounded retry, not `device_attach(&pdev->dev)`. After the fix the BIND-time dmesg block must contain `snvme: device driver name: snvme` and the queue-summary line must be the 4-tuple `135/0/0/0 default/read/poll/user` variant. |
