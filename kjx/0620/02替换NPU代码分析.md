# snvme NVIDIA → 昇腾 NPU 函数替换分析与开关设计

> 日期：2026-06-20
> 目标：把 snvme 内核模块里所有 NVIDIA 专有的 GPUDirect（`nvidia_p2p_*`）调用，替换成在华为昇腾 NPU 服务器上可用的等价机制，并做成**编译期开关**（默认走 NVIDIA 路线，`SNVME_NPU=1` 时走 NPU 路线）。
> 配套代码改动见本文第六节，已落地到 `snvme-5.10-npu/`。

---

## 一、先回答你的核心问题：这是 CANN 的事，还是 NPU driver 的事？

**结论：是 NPU 内核 driver 框架的事，和 CANN 编程框架基本无关（在内核侧）。CANN/ACL 只出现在"用户态测试程序"那一端。**

要分清两个层次：

| 层次 | 运行位置 | NVIDIA 的东西 | 昇腾的对应物 | snvme 里涉及吗 |
|---|---|---|---|---|
| **用户态显存分配** | userspace | CUDA：`cudaMalloc` / `cuMemAlloc` | **CANN / ACL**：`aclrtMalloc` / `aclrtMallocHost` | ❌ 只在测试 `snvme_smoke_gpu.cu` 里，**不在内核模块里** |
| **内核态显存 pin + 拿 DMA 地址** | kernel | **`nvidia_p2p_get_pages` / `nvidia_p2p_dma_map_pages`**（nvidia.ko 导出的 nv-p2p API） | 昇腾驱动的 **dma-buf 导出** + 标准内核 **dma-buf 导入 API**（`devmm`/`devmm_svm` 子系统） | ✅ **就是 `map.c` 里要替换的这一层** |

**为什么 snvme 要替换的是 driver 层、不是 CANN 层：**

`map.c` 的 GPU 路径干的事是——把"加速卡显存"这块地址 **pin 住**，再针对"这块 NVMe SSD 的 PCI 设备"拿到一组 **NVMe 控制器能直接 DMA 的总线地址**，填进 `map->addrs[]`，让 SSD 直接 DMA 进/出显存（绕开 CPU bounce buffer）。这就是 **GPUDirect Storage 的内核机制**。它发生在内核态，是设备驱动之间的 peer-to-peer DMA，**和"用什么语言写 kernel、怎么 malloc 显存"（CANN 的范畴）无关**。

- NVIDIA 在内核里用**私有的 nv-p2p API**（`nvidia_p2p_get_pages` 家族，由 nvidia.ko 导出符号）来做这件事。snvme 现在就是动态查找这些符号（`__symbol_get`）来用。
- 昇腾**没有**公开的、和 `nvidia_p2p_get_pages` 同名/同形的内核符号。昇腾显存的内核接口是 **`devmm` / `devmm_svm`（device memory management，对应设备节点 `/dev/devmm_svm`）**；要把显存给第三方设备（NVMe/RNIC）做 P2P DMA，业界与内核社区收敛到的标准做法是 **dma-buf 导出/导入 + PCI P2PDMA**（Linux 5.12 起作为 RDMA P2P 的标准机制，6.2 起 NVMe P2PDMA 支持更完整）。NVIDIA 自己新版 GDS（CUDA 12.8+）也在往上游 dma-buf / PCI P2PDMA 迁移、不再依赖 `nvidia-fs.ko`。

**一句话**：替换的目标 = **把 NVIDIA 私有 nv-p2p API，换成"昇腾驱动 dma-buf 导出 + 内核标准 dma-buf 导入 API"**。CANN（`aclrtMalloc`）只在第二件事——把测试程序 `snvme_smoke_gpu.cu` 从 CUDA 迁到 Ascend C——时才登场，那是用户态、是另一条线。

**一个关键前提（必须在目标机实测确认）**：上述方案成立的**硬前提**是「昇腾的内核驱动（CANN 配套的 driver 包）支持把 NPU HBM 显存导出成 dma-buf fd」。如果当前昇腾驱动版本不支持 dma-buf 导出，则内核 P2P 直通这条路本身走不通，需要找华为要 driver 层的 peer-memory 能力，或退回"显存↔host bounce buffer↔SSD"的非直通路径。**本文档的代码开关把这一未知点收敛到了 2 个明确的 TODO 触点（见第六节）。**

---

## 二、snvme 当前用到的全部 NVIDIA 内核符号

`map.c` 通过 `nvfs-p2p.c` 这层 wrapper 间接调用 6 个 nvidia.ko 导出符号。wrapper 的作用是"找得到符号就调，找不到就返回 `-ENOMEM`"。

| # | NVIDIA 内核符号 | 在 nvfs wrapper 里的名字 | 作用 | map.c 调用点 |
|---|---|---|---|---|
| 1 | `nvidia_p2p_get_pages` | `nvfs_nvidia_p2p_get_pages` | **pin 显存**：把 GPU VA 对应的显存页钉住，输出 `page_table`（含 `entries`），注册强制回收回调 | `map_gpu_memory` L485 / `map_gpu_ioqueue_memory` L568 |
| 2 | `nvidia_p2p_dma_map_pages` | `nvfs_nvidia_p2p_dma_map_pages` | **映射给某块 NVMe**：针对 `peer`(NVMe PCI dev) 把 `page_table` 映射成可 DMA 地址，输出 `dma_mapping->dma_addresses[]` | L501 / L576 |
| 3 | `nvidia_p2p_dma_unmap_pages` | `nvfs_nvidia_p2p_dma_unmap_pages` | 解除某块 NVMe 的 DMA 映射 | L340/L370/L402/L431 |
| 4 | `nvidia_p2p_put_pages` | `nvfs_nvidia_p2p_put_pages` | **unpin 显存**（正常释放路径） | L411/L437 |
| 5 | `nvidia_p2p_free_dma_mapping` | `nvfs_nvidia_p2p_free_dma_mapping` | 释放 dma_mapping 结构 | （wrapper 提供，强制回收路径用 unmap 代替） |
| 6 | `nvidia_p2p_free_page_table` | `nvfs_nvidia_p2p_free_page_table` | 释放 page_table 结构（强制回收路径） | L349/L375 |

涉及的数据结构（`nv-p2p.h` 桩头里）：
- `struct nvidia_p2p_page_table { u32 entries; ... }` — map.c 读 `->entries`（页数）
- `struct nvidia_p2p_dma_mapping { u64 *dma_addresses; ... }` — map.c 读 `->dma_addresses[i]`（每页总线地址）

模块加载时 `pci.c nvme_init()` 调 `nvfs_nvidia_p2p_init()` 一次性查这 6 个符号（迁移修改 #1 已改成"查不到只告警、继续 HOST-ONLY"）。

---

## 三、替换映射表（NVIDIA nv-p2p ↔ 内核 dma-buf 导入 API）

**好消息：nv-p2p 的语义和 Linux 标准 dma-buf 导入 API 几乎 1:1 对齐**，因此 `map.c` 的调用结构可以完全不动，只换 wrapper 实现。

| nv-p2p（NVIDIA 私有） | 语义 | **昇腾 NPU 替换（内核标准 dma-buf 导入）** | 说明 / 数据落点 |
|---|---|---|---|
| `nvidia_p2p_get_pages(0,0, vaddr, len, &pt, cb, data)` | 按 GPU VA pin 显存，得 page_table | `dma_buf_get(fd)` → 得 `struct dma_buf *` | **输入模型变了**：nv 用 GPU VA；dma-buf 用 **fd**。昇腾侧 `aclrtMalloc` 的显存要先由昇腾驱动**导出成 dma-buf fd**，再经 ioctl 传进来。pin 语义由 dma-buf 的"映射期间 backing storage 被 pin 住"保证。 |
| `nvidia_p2p_dma_map_pages(peer_pdev, pt, &m)` | 针对某 NVMe 拿可 DMA 地址 | `dma_buf_attach(dmabuf, &peer_pdev->dev)` + `dma_buf_map_attachment(attach, DMA_BIDIRECTIONAL)` → `struct sg_table *` | **per-(显存, NVMe设备) 映射，完全对应 dma-buf 的 attach/map**。把返回 `sg_table` 里的 `sg_dma_address()` 按 64KB 页展开，填进 `dma_mapping->dma_addresses[]` → 最终进 `map->addrs[]`。 |
| `nvidia_p2p_dma_unmap_pages(peer, pt, m)` | 解除映射 | `dma_buf_unmap_attachment(attach, sgt, DMA_BIDIRECTIONAL)` + `dma_buf_detach(dmabuf, attach)` | 一一对应 |
| `nvidia_p2p_put_pages(0,0, vaddr, pt)` | unpin | `dma_buf_put(dmabuf)` | 释放对 dma_buf 的引用 |
| `nvidia_p2p_free_dma_mapping(m)` | 释放映射结构 | 释放我们自管的 `dma_mapping` 容器（含 dma_addresses 数组） | 容器由 npu wrapper 自己 kmalloc/kfree |
| `nvidia_p2p_free_page_table(pt)` | 释放页表结构 | 释放我们自管的 `page_table` 容器（含 dmabuf 引用） | 同上 |
| `free_callback`（强制回收回调） | nv 驱动回收显存时反向通知 | dma-buf **动态 attach 的 `move_notify`** 回调（可选） | v1 用**静态 attach**（映射期间一直 pin），**暂不接 move_notify**；强制回收语义弱化，记为 TODO/限制。 |
| `nvfs_nvidia_p2p_init()`（查 nvidia 符号） | 探测 GPU p2p 可用性 | 直接返回 0（dma-buf 是 per-mapping 行为，无需全局符号探测）；可选探测 `/dev/devmm_svm` 是否存在 | 模块加载不再依赖 nvidia.ko |

### 涉及的内核 API（5.10 均已存在、GPL 导出，可直接用）

`<linux/dma-buf.h>`：`dma_buf_get` / `dma_buf_put` / `dma_buf_attach` / `dma_buf_detach` / `dma_buf_map_attachment` / `dma_buf_unmap_attachment`；`<linux/scatterlist.h>`：`sg_dma_address` / `sg_dma_len` / `for_each_sgtable_dma_sg`。

### 结构体对应

| nv-p2p 结构 | NPU 替换后承载的内容 |
|---|---|
| `struct nvidia_p2p_page_table` | `{ u32 entries; struct dma_buf *dmabuf; }` —— pin 阶段拿到的 dma_buf 引用 + 页数 |
| `struct nvidia_p2p_dma_mapping` | `{ u64 *dma_addresses; struct dma_buf_attachment *attach; struct sg_table *sgt; ... }` —— 映射阶段的 attach/sgt + 展开后的逐页总线地址 |

> 因为 map.c 只读 `page_table->entries` 和 `dma_mapping->dma_addresses[]`，把这两个结构在 `#ifdef SNVME_NPU` 下扩展几个字段即可，**map.c 源码一字不改**。

---

## 四、为什么不用"昇腾私有符号"或"CANN 接口"直接替换

1. **没有公开的昇腾 nv-p2p 等价符号**：搜索（含华为开发者社区、CANN/ACL 文档）未发现昇腾导出过和 `nvidia_p2p_get_pages` 同形的内核 peer-memory 符号。昇腾显存内核接口是 `devmm`/`devmm_svm`，但其第三方 P2P 暴露方式官方未公开命名 API。
2. **CANN/ACL 是用户态**：`aclrtMalloc` 等是 userspace C/C++ 接口，内核模块不能调，也不暴露"把显存 pin 给外设 DMA"的回调。
3. **dma-buf 是厂商中立、内核标准、且是大势所趋**：Linux 社区在评估过 Peer Memory Client 等子系统专用方案后，统一到 dma-buf（5.12+）；NVIDIA 新版 GDS 也在弃用 `nvidia-fs.ko` 转向上游 PCI P2PDMA。对昇腾，只要其驱动支持 dma-buf 导出，snvme 用标准导入 API 即可，**无需任何华为私有头文件**。

---

## 五、对当前阶段（host-only / snvme_smoke.c）的影响 = 零

- `snvme_smoke.c` 是**纯 host 内存路径**，根本不调 GPU/NPU 显存路径（详见 `01迁移分析.md`）。
- 本次替换只动 GPU/NPU 显存路径（`map.c` 的 `map_gpu_*` + wrapper 层），且开关**默认关闭**（不定义 `SNVME_NPU`，走原 NVIDIA 路线）。
- 因此**第一阶段编译与 smoke 测试不受任何影响**；NPU 路径是为第二阶段（NPU 显存直通 + `snvme_smoke_gpu` 迁到 Ascend C）预留的、可独立开启的编译路线。

---

## 六、开关设计与已落地的代码改动

沿用仓库既有的"wrapper 层 + 编译期开关"风格（就像 `#if LINUX_VERSION_CODE` 那样）。**默认 NVIDIA，`make SNVME_NPU=1` 切到 NPU。**

### 6.1 开关入口：`Makefile`

```make
make                # 默认：NVIDIA 路线，链接 nvfs-p2p.o
make SNVME_NPU=1    # NPU 路线：-DSNVME_NPU，改链接 npu-p2p.o（dma-buf 实现）
```

实现：`SNVME_NPU=1` 时给 `ccflags-y` 加 `-DSNVME_NPU`，并把 `snvme-objs` 里的 `nvfs-p2p.o` 换成 `npu-p2p.o`。两份 wrapper 提供**完全相同的函数名/签名**（`nvfs_nvidia_p2p_*`），所以 `map.c`、`pci.c` 一字不改、二选一链接。

### 6.2 新增文件：`npu-p2p.c` / `npu-p2p.h`

- 提供和 `nvfs-p2p.c` 同名同签名的 6 个 `nvfs_nvidia_p2p_*` 函数 + `init`/`exit`，但实现改为 **dma-buf 导入序列**（见第三节映射表）。
- `nvfs_nvidia_p2p_init()` 返回 0（不再查 nvidia 符号）。
- **2 个明确 TODO 触点**（昇腾环境相关、需在目标机用 CANN driver SDK 确认）：
  - **TODO-A**：`get_pages` 里 `dma_buf_get(fd)` 的 `fd` 从哪来——v1 约定**把 ioctl 传入的 `vaddr` 字段复用为 dma-buf fd**（`(int)virtual_address`），并加显式注释；正式方案应在 `struct nvm_ioctl_map` 增一个 `dmabuf_fd` 字段，由昇腾用户态导出显存 fd 后传入。
  - **TODO-B**：确认昇腾驱动**确实支持 dma-buf 导出** HBM 显存；若不支持，`dma_buf_attach`/`map_attachment` 会失败返回错误码，snvme 安全报错（不 oops），但 P2P 直通不可用。

### 6.3 修改文件：`nv-p2p.h`

在 `#ifdef SNVME_NPU` 下，把 `struct nvidia_p2p_page_table` / `struct nvidia_p2p_dma_mapping` 扩展出承载 dma-buf 状态的字段（`dmabuf` / `attach` / `sgt`），`#else` 保持原 NVIDIA 桩定义。`map.c` 仍只用 `->entries` 与 `->dma_addresses`，源码不变。

### 6.4 不需要改的文件

`map.c`、`pci.c`、`ioctl.c`、`map.h` —— **零改动**。开关完全收敛在 wrapper 层 + `nv-p2p.h` 结构定义 + Makefile。

### 6.5 当前状态与验证边界（如实说明）

- ✅ NPU 路径用的 `dma_buf_*` / `sg_dma_address` 都是 5.10 内核标准导出符号，**能编译能链接**。
- ⚠️ **未在昇腾硬件上实测**：能否真正跑通取决于 TODO-A（fd 来源约定）与 TODO-B（昇腾驱动是否支持 dma-buf 导出）。在这两点确认前，NPU 路径是**结构正确、可编译、运行时遇到不支持会安全报错**的骨架，不是已验证可用的功能。
- ✅ 默认（不开 `SNVME_NPU`）= 原 NVIDIA 行为，第一阶段 host-only smoke 完全不受影响。

---

## 七、第二阶段后续待办

1. **确认昇腾 dma-buf 导出能力**（TODO-B）：查目标机 CANN driver 版本是否提供 HBM dma-buf 导出；不支持则需向华为索取 driver 层 peer-memory 方案。
2. **ioctl 增 `dmabuf_fd` 字段**（TODO-A）：把"复用 vaddr 当 fd"的临时约定改为正式接口。
3. **`snvme_smoke_gpu.cu` → Ascend C**：`cudaMalloc`→`aclrtMalloc`、kernel→AIV、并在分配后取得 dma-buf fd 传入 ioctl。
4. **强制回收**（move_notify）：如需昇腾侧回收显存时的反向通知，改用 dma-buf 动态 attach 并实现 `move_notify`。
5. **PCI P2PDMA 拓扑校验**：确认 NVMe 与 NPU 在同一 PCIe 域可 P2P（`pci_p2pdma_distance` 等），多路径/RAID0 在上游内核的 P2PDMA 限制需注意。

---

## 八、参考链接

**内核 dma-buf / P2PDMA（替换所依据的标准 API）**
- Buffer Sharing and Synchronization (dma-buf) — https://docs.kernel.org/driver-api/dma-buf.html
- PCI Peer-to-Peer DMA Support — https://docs.kernel.org/driver-api/pci/p2pdma.html
- `include/linux/dma-buf.h`（含 `<linux/pci-p2pdma.h>`）— https://github.com/torvalds/linux/blob/master/include/linux/dma-buf.h

**NVIDIA nv-p2p / GPUDirect（被替换的原机制）**
- GPUDirect RDMA（nv-p2p `get_pages/put_pages` 家族）— https://docs.nvidia.com/cuda/gpudirect-rdma/
- GPUDirect Storage Overview（GDS 内核 P2P，CUDA 12.8 转上游 P2PDMA）— https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
- Mellanox `nv_peer_memory` 的 `compat_nv-p2p.h`（nv-p2p 接口形态参考）— https://github.com/Mellanox/nv_peer_memory/blob/master/compat_nv-p2p.h
- Peer Memory Client `get_pages` 回调（nv-p2p 的结构对应物）/ GPUDirect RDMA 演进 — https://medium.com/@datenlord/the-evolution-and-implementation-of-gpudirect-rdma-19751f7b9413

**昇腾 NPU 侧（CANN/ACL 用户态 + devmm 内核显存子系统）**
- CANN ACL 内存管理（`aclrtMalloc`，用户态显存，对应 `cudaMalloc`）— https://support.huawei.com/enterprise/en/doc/EDOC1100192513/74f8188d/memory-management-ascend-310-ai-processor
- 昇腾容器设备节点（`/dev/devmm_svm`、`/dev/davinci*`、`/dev/hisi_hdc`，内核显存管理入口）— https://support.huaweicloud.com/intl/en-us/usermanual-cce/cce_10_0239.html
- RDMA with GPU Memory via DMA-Buf（dma-buf 做 P2P 的工程实践）— https://www.toolify.ai/hardware/boosting-performance-rdma-with-gpu-memory-via-dmabuf-2861912
