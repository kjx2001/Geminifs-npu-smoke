# snvme-5.10-npu 源码分析（写给新手的循序渐进版）

> 分析对象：`/home/kjx/nds/Geminifs-npu-smoke/backends/local/kernel_modules/snvme-5.10-npu/`
> 对照基准：原生内核 NVMe 主机驱动 `/home/kjx/nds/kernel-openEuler-22.03-LTS-drivers-nvme/drivers/nvme/host/`
> 目标：搞清楚**每个文件干什么、哪些文件配合实现什么效果、相比原生改了什么、NPU 迁移做到哪一步了**。

---

## 第 0 章　先把基础概念对齐（看懂后面的前提）

你已经知道 NVMe 的三件套，这里再把它们和"内存里到底发生了什么"对上号，因为这整个项目的灵魂就是围绕这三件套做文章：

| 概念 | 物理上是什么 | 谁写谁读 |
|------|--------------|----------|
| **SQ（提交队列）** | 一块内存，里面是一条条 64 字节的命令（SQE） | 主机写命令进去；SSD 通过 DMA 把命令读走 |
| **CQ（完成队列）** | 一块内存，里面是一条条 16 字节的完成项（CQE） | SSD 写完成项进去；主机读出来判断命令做完没 |
| **Doorbell（门铃）** | SSD 上的一个寄存器（在 PCIe BAR0 地址空间里） | 主机写它来"按门铃"，告诉 SSD「我往 SQ 放新命令了，快来取」 |

**一个关键事实**：SQ/CQ 这两块"内存"既可以放在内核里（普通驱动的做法），**也可以放在用户态进程的内存里、甚至放在 GPU/NPU 的显存里**。只要 SSD 能通过 DMA 访问到这块内存的**物理（总线）地址**，SSD 就不在乎它姓"内核"还是姓"用户"。

> **这就是整个 snvme / GeminiFS / BaM / GPUDirect-Storage 的核心思想：**
> 把 SQ/CQ 队列内存和 doorbell 寄存器**直接交给用户态（或 GPU/NPU）**，让用户态程序自己往 SQ 写命令、自己按 doorbell、自己轮询 CQ，**全程不进内核、不走系统调用**，从而获得极低延迟和极高 IOPS。

而本项目要解决的"内核侧难题"就是：
1. 怎么把用户态/显存的那块内存 **pin 住并拿到它的 DMA 物理地址**？（→ `map.c`）
2. 怎么用这个地址**让 SSD 认得这块内存是它的 SQ/CQ**？（→ 发 `Create I/O SQ/CQ` admin 命令，`pci.c`）
3. 怎么把 **doorbell 寄存器映射进用户态**让它能直接写？（→ `mmap` BAR0，`pci.c`）
4. 怎么在不和内核自带 `nvme` 驱动打架的前提下，**复用整套 NVMe 协议栈**？（→ `snvme-core.ko` 改名 + 导出符号）

把这 4 点记在心里，下面所有代码都是在回答这 4 个问题。

---

## 第 1 章　全局鸟瞰：这个目录编译出两个内核模块

打开 `Makefile`（迁移者新写的树外编译入口，注释标了 `[SNVME-NPU 迁移新增文件 #3]`），最关键的是这两行对象集定义：

```make
# snvme-core.ko = 通用 NVMe 协议栈（fork 自原生驱动，改名 + 导出符号）
snvme-core-y := core.o ioctl.o
snvme-core-$(CONFIG_NVME_MULTIPATH) += multipath.o
snvme-core-$(CONFIG_BLK_DEV_ZONED)  += zns.o
snvme-core-$(CONFIG_NVME_HWMON)     += hwmon.o

# snvme.ko = 自带 PCIe probe 的 NVMe 驱动 + "把队列暴露给用户态"的字符设备层
snvme-objs := nvfs-pci.o nvfs-p2p.o list.o ctrl.o map.o pci.o
```

所以这一堆文件其实分成**两个模块、四类角色**：

```
snvme-5.10-npu/
├── 【模块 A：snvme-core.ko】通用 NVMe 协议栈（改名版原生驱动）
│   ├── core.c        ← 控制器生命周期、命令构造、namespace 扫描…（改名 + 导出符号 主战场）
│   ├── ioctl.c       ← 标准块设备 ioctl（从原生 core.c 拆出来的，无私有命令）
│   ├── nvme.h        ← 共享头：把一批函数原型公开给 snvme.ko 用
│   ├── multipath.c   ← 多路径（改名 + 适配）
│   ├── zns.c         ← 分区命名空间 ZNS（改名 + 适配）
│   └── hwmon.c       ← 温度监控（改名）
│
├── 【模块 B：snvme.ko】把 NVMe 队列暴露给用户态
│   ├── pci.c         ← 【主体】自带 probe 的 NVMe 驱动 + 控制面/数据面 ioctl + mmap
│   ├── ctrl.c/.h     ← 一个控制器 ↔ 一个 /dev/ssnvmeN 字符设备
│   ├── map.c/.h      ← 【DMA 映射引擎】pin 用户内存、取总线地址
│   └── list.c/.h     ← 最底层双向链表原语（被 ctrl/map 复用）
│
├── 【角色 C：P2P / GPUDirect 子系统】（GPU/NPU 显存直连，当前是桩）
│   ├── nvfs-p2p.c/.h ← 封装 NVIDIA P2P 6 个 API（运行时动态绑定符号）
│   ├── nvfs-pci.c/.h ← PCIe 拓扑距离矩阵（为 P2P 选最近设备）
│   ├── nvfs-core.h   ← P2P 公共定义、IOCTL 结构、GPU 页常量
│   └── nv-p2p.h      ← 【桩头 stub】顶替 NVIDIA 官方头，让无 NVIDIA SDK 的机器能编译
│
└── 【角色 D：构建】Makefile / Makefile.in / Kconfig
```

> **一句话记住分工**：
> `snvme-core.ko` 提供"会说 NVMe 协议"的通用能力；`snvme.ko` 借用这个能力去驱动一块盘、建队列，再把队列和 doorbell 交给用户态；P2P 那一套是为了让队列内存还能放进 GPU/NPU 显存（目前在 NPU 上还是桩）。

---

## 第 2 章　对照基准：原生 NVMe 驱动长什么样

原生 `drivers/nvme/host/` 目录里：
- `core.c` —— 与具体传输无关的**通用层**：控制器状态机、命令构造（`nvme_setup_cmd`）、namespace 扫描、sysfs、keep-alive……
- `pci.c` —— **PCIe 传输层**：`pci_driver` 的 probe/remove、用真实硬件建 admin/IO 队列、写 doorbell、中断处理。它**调用** core.c 的通用函数。
- `fabrics.c / rdma.c / tcp.c / fc.c` —— 其它传输层（网络/光纤），和本项目关系不大。
- `nvme.h` —— 内部共享头。
- `multipath.c / zns.c / hwmon.c / ioctl.c(5.15起) / lightnvm.c / trace.c / fault_inject.c` —— 各种附加功能。

原生的分层关系是：

```
   应用 → 块设备层(blk-mq) → core.c(通用NVMe) → pci.c(PCIe硬件) → SSD
```

注意：**原生驱动里，SQ/CQ 队列内存全在内核里，doorbell 也只有内核能写**，用户态完全碰不到。snvme 要做的就是"凿开一个口子"把这些交给用户态。

snvme 目录里**缺了** `fc.c/fc.h`（光纤通道）、`lightnvm.c`、`trace.c/h`、`fault_inject.c`——因为用不到，删掉减负。

---

## 第 3 章　snvme 的整体架构（一张图看懂数据流）

```
┌─────────────────────────────────────────────────────────────┐
│ 用户态程序 (libnvm / GeminiFS)                                │
│  ① open(/dev/snvm_control)  做 bind / 创建字符设备            │
│  ② open(/dev/ssnvmeN)       针对某块盘                        │
│  ③ ioctl 注册队列内存、建队列、拿 doorbell 偏移               │
│  ④ mmap(BAR0) 拿到 doorbell 寄存器的可写映射                  │
│  ⑤ 自己写 SQ → 写 doorbell → 轮询 CQ   （全程不进内核！）     │
└───────────────┬─────────────────────────────────────────────┘
                │ ioctl / mmap
┌───────────────▼─────────────────────────────────────────────┐
│ snvme.ko  (pci.c 为主体)                                      │
│   控制面字符设备 /dev/snvm_control : bind/unbind/建删设备     │
│   数据面字符设备 /dev/ssnvmeN      : 映射内存/建队列/doorbell  │
│   ├─ ctrl.c : 管理"控制器↔字符设备"对象                       │
│   ├─ map.c  : pin 内存 + dma_map_page → 拿到总线地址 ★        │
│   ├─ list.c : 链表，挂全局对象表                              │
│   └─ nvfs-* : (可选)把队列/数据放进 GPU/NPU 显存 (当前是桩)   │
│         │ 调用 snvme_* 导出符号                               │
└─────────┼────────────────────────────────────────────────────┘
          │ EXPORT_SYMBOL_GPL(snvme_setup_cmd / snvme_enable_ctrl …)
┌─────────▼────────────────────────────────────────────────────┐
│ snvme-core.ko  (core.c + ioctl.c)                             │
│   通用 NVMe 协议栈（fork 原生驱动，全部改名 nvme_*→snvme_*）   │
└──────────────────────────────────────────────────────────────┘
                │ PCIe / DMA
            ┌───▼───┐
            │  SSD  │
            └───────┘
```

★ 标注的 `map.c` 是连接"用户内存"和"SSD 看到的物理地址"的桥，是整个机制的关键。

---

## 第 4 章　模块 A：snvme-core.ko 相比原生改了什么

> 一句话：**它不是"在原生上加功能"，而是把原生驱动整体 fork 出来、全部改名、再把一批内部函数导出**，好让它能和内核自带 `nvme` 共存、并被 `snvme.ko` 复用。

⚠️ 注意一个细节：snvme 的代码**基线其实是 Linux 5.15**，而要跑的目标内核是 **openEuler 5.10（aarch64）**。所以你在 diff 里看到的差异是两股叠加的：
- **(A) snvme 化**：改名 + 导出符号 + 拆分 ioctl。
- **(B) 5.15 → 5.10 backport**：把 5.15 才有的内核 API 用条件编译降级回 5.10 能用的写法。

### 4.1 改动一：全量改名（为了和内核自带 nvme 共存）

几乎所有对外函数 `nvme_xxx` → `snvme_xxx`，所有模块参数和工作队列加 `s_` 前缀：

| 原生 | snvme | 类型 |
|------|-------|------|
| `nvme_complete_rq` | `snvme_complete_rq` | 函数 |
| `nvme_setup_cmd` | `snvme_setup_cmd` | 函数 |
| `nvme_submit_sync_cmd` | `snvme_submit_sync_cmd` | 函数 |
| `nvme_enable_ctrl` | `snvme_enable_ctrl` | 函数 |
| `nvme_wq` | `s_nvme_wq` | 工作队列 |
| `io_timeout`(模块参数) | `s_nvme_io_timeout` | 参数 |

**为什么必须改名**：内核里已经有一个同名的 in-tree `nvme` 驱动。如果树外模块还叫 `nvme_complete_rq`、还注册 `io_timeout` 这个参数、还创建 `nvme_wq` 工作队列，就会和内核**符号表 / sysfs 参数 / workqueue 命名全面冲突**，模块根本加载不进去。加上 `s`/`snvme` 前缀后，两套驱动就能在同一个系统里并存、互不干扰。

### 4.2 改动二（心脏）：把一批 static 函数导出成符号

原生 core.c 里很多函数是 `static`（只给本文件用）。snvme 把它们改成 `EXPORT_SYMBOL_GPL(snvme_xxx)` 导出，并在 `nvme.h`（约 657–916 行）公开它们的原型，**专门给另一个模块 `snvme.ko` 调用**。

被导出的关键符号分几类：
- **控制器生命周期**：`snvme_init_ctrl` / `snvme_start_ctrl` / `snvme_enable_ctrl` / `snvme_disable_ctrl` / `snvme_reset_ctrl` / `snvme_change_ctrl_state` …
- **队列管理（关键）**：`snvme_start_queues` / `snvme_stop_queues` / `snvme_kill_queues` / `snvme_set_queue_count` / `snvme_start_freeze` / `snvme_unfreeze` …
- **命令路径**：`snvme_alloc_request` / `snvme_setup_cmd` / `snvme_complete_rq` / `snvme_submit_sync_cmd` …
- **namespace / features**：`snvme_find_get_ns` / `snvme_get_features` / `snvme_set_features` …

**为什么要导出**：`snvme.ko`（即 pci.c）是一份**自带 PCIe probe 的 NVMe 驱动**——它自己定义 `struct nvme_dev`、自己建 `blk_mq` tagset 和提交队列。但"通用 NVMe 协议逻辑"（怎么构造一条命令、怎么使能控制器、怎么协商队列数）它不想重写，于是直接调 snvme-core 导出的函数。例如 pci.c 在自己的 `nvme_queue_rq()` 里调 `snvme_setup_cmd()`，admin 命令走 `snvme_submit_sync_cmd()`，使能控制器走 `snvme_enable_ctrl()`。

> **因果链**：要把队列暴露给用户态 → snvme.ko 必须自己掌控控制器和队列的创建/freeze/start/stop → 这些动作的实现都在 core.c → 所以必须把它们导出。**"导出符号" = "让 snvme.ko 拿到操控队列的能力"。**

（顺带：有一批原生导出的符号被**故意注释掉**了，比如 `nvme_delete_ctrl`、`nvme_command_effects` 等，因为 snvme.ko 用不到，关掉能减小符号暴露面。）

### 4.3 改动三：把 ioctl 从 core.c 拆成 ioctl.c

原生 5.10 把块设备 ioctl 逻辑写在 core.c 里；upstream 5.15 把它拆成独立的 `ioctl.c`。snvme 跟随了 5.15 的做法。

- **拆出来的函数**：`nvme_submit_user_cmd` / `nvme_submit_io` / `nvme_user_cmd(64)` / `nvme_ioctl` / `nvme_dev_ioctl` 等。
- **处理的命令全是标准 UAPI**：`NVME_IOCTL_ADMIN_CMD` / `NVME_IOCTL_IO_CMD` / `NVME_IOCTL_SUBMIT_IO` / `NVME_IOCTL_RESET` …
- **有没有新增 snvme 私有 ioctl？没有！** snvme 的私有 ioctl（`NVM_MAP_HOST_MEMORY` 那一堆）全在**另一个模块 `snvme.ko` 的 pci.c 里**，不在这个 ioctl.c。**别搞混了**：这个 ioctl.c 只是普通的"把盘当块设备用"的标准接口。

### 4.4 multipath.c / zns.c / hwmon.c

这三个都是"改名 + 5.10 适配"，功能没变：
- `multipath.c`：多路径。主要适配 `bio->bi_disk`（5.10）vs `bi_bdev`（5.14+）、`blk_alloc_disk`（5.14+）不存在等差异。
- `zns.c`：分区命名空间上报，改名 `snvme_submit_sync_cmd`。
- `hwmon.c`：SSD 温度监控，纯改名。

---

---

## 第 4+ 章　snvme-core.ko 深入：五条核心调用链全景

> 上一章讲的是"改了什么"，这一章讲的是"它本来怎么工作"——把 core.c + ioctl.c 里**到底是哪些函数互相调用**完成了控制器生命周期、命令构造、完成处理、namespace 扫描，以及 ioctl.c 每条命令干什么，全部拆开。
>
> 先记住一个总览：core.c 里的函数大致分 5 组——①控制器生命周期 ②命令构造（下行）③命令完成（上行）④namespace 扫描 ⑤被 ioctl.c 复用的 passthrough。下面逐条画调用链。`snvme_` 前缀 = 导出给 snvme.ko 用的；`nvme_` 前缀 = 文件内部 static 辅助。

### 4+.1　控制器生命周期：从插上盘到能用，再到拔走

`snvme.ko`（pci.c）的 `nvme_probe()` 是总指挥，它按顺序调用 core.c 导出的这几个函数，把一个控制器从"刚发现"推到"LIVE 可用"：

```
nvme_probe() [pci.c]
  │
  ├─① snvme_init_ctrl(ctrl, dev, ops, quirks)            core.c:4662
  │     ├─ ctrl->state = NVME_CTRL_NEW                    （状态机起点，还没碰硬件）
  │     ├─ ida_simple_get 分配 instance → 取名 nvme%d
  │     ├─ cdev_device_add 建字符设备 /dev/nvmeX
  │     └─ INIT_WORK 把 4 个工作绑定到处理函数：
  │          scan_work        → nvme_scan_work        （namespace 扫描，见 4+.4）
  │          async_event_work → nvme_async_event_work （异步事件 AEN）
  │          fw_act_work      → nvme_fw_act_work      （固件激活）
  │          delete_work      → nvme_delete_ctrl_work （删除控制器）
  │          ka_work          → nvme_keep_alive_work  （心跳保活）
  │
  ├─② snvme_enable_ctrl(ctrl)                            core.c:2305
  │     ├─ 算页大小、CSS、IOSQES/IOCQES，组好 CC 寄存器
  │     ├─ ops->reg_write32(NVME_REG_CC, ...)            （写硬件 CC，拉高 EN 位 = 开机）
  │     └─ nvme_wait_ready(ctrl, cap, true)              （轮询 CSTS.RDY=1，等控制器就绪）
  │
  ├─③ snvme_init_ctrl_finish(ctrl)                       core.c:3228  ★identify 阶段★
  │     ├─ reg_read32(NVME_REG_VS)                       （读版本）
  │     ├─ nvme_init_identify(ctrl)                      core.c:3056
  │     │    ├─ nvme_identify_ctrl()  → 发 Identify Controller(CNS=0x01) admin 命令
  │     │    │     读回 oacs/oncs/mdts/sgls/kas/npss 等控制器能力，存进 ctrl
  │     │    ├─ nvme_init_subsystem() → 建/找 subsystem，匹配 quirks 表
  │     │    └─ nvme_get_effects_log() → 命令支持矩阵
  │     ├─ nvme_init_non_mdts_limits()                   （非 MDTS 限制）
  │     ├─ nvme_configure_apst()      （自动功耗状态切换）
  │     ├─ nvme_configure_timestamp() （对时）
  │     ├─ nvme_configure_directives()（Streams）
  │     ├─ nvme_configure_acre()      （命令重试延迟）
  │     ├─ nvme_hwmon_init()          （温度传感器，hwmon.c）
  │     └─ ctrl->identified = true
  │
  ├─④ snvme_change_ctrl_state(ctrl, NVME_CTRL_LIVE)      core.c:458
  │     └─ 状态机校验：只有 NEW/RESETTING/CONNECTING → LIVE 合法；非法转移返回 false
  │
  └─⑤ snvme_start_ctrl(ctrl)                             core.c:4590
        ├─ nvme_start_keep_alive()  → 启动 ka_work 周期心跳
        ├─ nvme_enable_aen()        → 打开异步事件通知
        └─ if (queue_count > 1):
             ├─ nvme_queue_scan()   → queue_work(scan_work)  触发 4+.4 的扫描
             └─ snvme_start_queues()→ 解冻所有 namespace 队列，开始收 IO
```

**配套的"状态机"**：`snvme_change_ctrl_state()`（core.c:458）是所有状态转移的唯一闸口。NVMe 控制器有这几个状态，转移必须合法：

```
NEW ──► CONNECTING ──► LIVE ──► RESETTING ──► LIVE        （重置后恢复）
                         │           │
                         └─────► DELETING ──► DELETING_NOIO ──► DEAD
```

**重置链**（盘出问题时自愈）：
```
snvme_reset_ctrl(ctrl) [core.c:204]
  → snvme_change_ctrl_state(RESETTING)
  → queue_work(reset_work)  → nvme_reset_work() [pci.c]  重新跑 disable→enable→finish→LIVE
nvme_reset_ctrl_sync() 是同步版（发起后 flush_work 等它跑完）
```

**删除链**（拔盘/卸载）：
```
nvme_delete_ctrl(ctrl) [core.c:249]
  → snvme_change_ctrl_state(DELETING) → queue_work(delete_work)
  → nvme_delete_ctrl_work → nvme_do_delete_ctrl():
       snvme_stop_ctrl()        停心跳/AEN/fw_act
       snvme_remove_namespaces() 删所有 ns（见 4+.4）
       ops->delete_ctrl()       传输层清理（pci.c）
       snvme_uninit_ctrl()      删字符设备、put_ctrl
```

**关机/下电**：`snvme_disable_ctrl()`（清 CC.EN）、`snvme_shutdown_ctrl()`（置 CC.SHN 优雅关机，等 CSTS.SHST=完成）。

**心跳保活**：`nvme_keep_alive_work()` 周期触发 → 发 Keep-Alive admin 命令 → `nvme_keep_alive_end_io()` 回调里重新排下一次；超时没回应就 `snvme_reset_ctrl()`。

> 这一组里 **snvme_init_ctrl / enable_ctrl / init_ctrl_finish / change_ctrl_state / start_ctrl / reset_ctrl / disable_ctrl / shutdown_ctrl / kill_queues / start_queues / stop_queues** 全部 `EXPORT_SYMBOL`，因为 pci.c 的 probe/reset/remove 流程要一个个调它们。这就是第 4.2 节说的"导出符号 = 让 snvme.ko 能操控控制器和队列"的具体兑现。

### 4+.2　命令构造（下行）：一个读写请求怎么变成 NVMe 命令

普通块设备 IO 走 blk-mq，最终落到 pci.c 的 `nvme_queue_rq()`，它调 core.c 的 `snvme_setup_cmd()` 把"块层请求 `request`"翻译成"64 字节 NVMe 命令 `nvme_command`"：

```
nvme_queue_rq(req) [pci.c]
  └─ snvme_setup_cmd(ns, req) [core.c:1034]   按 req 类型分派：
       ├─ REQ_OP_READ   → nvme_setup_rw(..., nvme_cmd_read)
       ├─ REQ_OP_WRITE  → nvme_setup_rw(..., nvme_cmd_write)   core.c:953
       │      填 cmd->rw：opcode / nsid / slba(起始LBA) / length(块数)
       │      / control(FUA/PI) / dsmgmt(Streams) / reftag(数据保护)
       ├─ REQ_OP_FLUSH        → nvme_setup_flush      （刷 cache）
       ├─ REQ_OP_DISCARD      → nvme_setup_discard    （DSM 释放，组 range 数组）
       ├─ REQ_OP_WRITE_ZEROES → nvme_setup_write_zeroes
       ├─ REQ_OP_ZONE_*       → nvme_setup_zone_mgmt_send  [zns.c]
       └─ REQ_OP_DRV_IN/OUT   → （passthrough，命令早在 nvme_init_request 备好）
     最后：cmd->common.command_id = nvme_cid(req)
            （nvme_cid 把 genctr 计数编进 cid，防止过期完成项被误认）
  └─ 回到 pci.c：把这条 cmd 拷进 SQ 环、写 doorbell 通知 SSD
```

**同步 admin 命令的构造与下发**（identify、set/get features、create queue 等都走这条）：

```
snvme_submit_sync_cmd(q, cmd, buf, len)  [core.c, 导出]
  └─ __snvme_submit_sync_cmd(...)
       ├─ snvme_alloc_request(q, cmd)        core.c → blk_mq_alloc_request 拿一个 request
       ├─ blk_rq_map_kern(...)               （若带数据缓冲，映射进 request）
       ├─ blk_execute_rq(...)                （丢给块层执行，阻塞等完成）
       └─ 返回 nvme_req(req)->status          （>0 是控制器状态码，<0 是内核错误）
```
辅助函数：`snvme_set_features` / `snvme_get_features`（core.c:1598/1607，包了一层发 Set/Get Features）、`snvme_set_queue_count`（协商 IO 队列数，pci.c 建队列前必调）。

### 4+.3　命令完成（上行）：SSD 写完 CQ 之后

pci.c 的中断处理（或轮询）发现 CQ 里有新完成项后，调 core.c 的 `snvme_complete_rq()` 收尾：

```
snvme_complete_rq(req) [core.c:374, 导出]
  ├─ snvme_cleanup_cmd(req)          释放 discard 等特殊 payload
  ├─ ctrl->comp_seen = true          （给心跳逻辑用）
  └─ nvme_decide_disposition(req)    core.c:333  决定怎么处理：
       ├─ status==0 或不可重试/超次数        → COMPLETE
       │     → nvme_end_req(req)
       │          ├─ nvme_error_status()  把 NVMe 状态码翻译成 blk_status_t
       │          └─ blk_mq_end_request() 通知块层"这个 IO done 了"
       ├─ 可重试的错误                        → RETRY
       │     → nvme_retry_req(req)  按 CRD 延迟后 blk_mq_requeue_request 重排
       └─ 多路径且是路径错误                  → FAILOVER
             → nvme_failover_req(req) [multipath.c]  切到另一条路径重试
```
错误码翻译表在 `nvme_error_status()`（core.c:271）：例如 `NVME_SC_CAP_EXCEEDED→BLK_STS_NOSPC`、`NVME_SC_WRITE_FAULT→BLK_STS_MEDIUM`、保护信息错→`BLK_STS_PROTECTION`。

取消/清理：`snvme_cancel_request()`（把在途请求标记 ABORTED 并完成）、`snvme_cancel_tagset` / `snvme_cancel_admin_tagset`（重置/删除时批量取消整张 tagset 的在途命令）。

### 4+.4　namespace 扫描：盘上有哪些"分区"，建成 /dev/nvmeXnY

控制器 LIVE 后，`nvme_queue_scan()` 触发 `scan_work`，把盘上的每个 namespace 发现出来、建成块设备：

```
nvme_queue_scan() → queue_work(scan_work)
  └─ nvme_scan_work(work) [core.c:4337]
       ├─ 先决条件：state==LIVE 且 tagset 存在（IO 队列已建好）
       ├─ nvme_scan_ns_list(ctrl)   core.c:4247
       │     发 Identify Active NS List(CNS=0x07) 拿到所有活跃 nsid
       │     对每个 nsid → nvme_validate_or_alloc_ns(ctrl, nsid)
       │     （老控制器不支持该命令时，回退 nvme_scan_ns_sequential：从 1 到 nn 挨个试）
       │
       └─ nvme_validate_or_alloc_ns(ctrl, nsid)  core.c:4184
            ├─ nvme_identify_ns_descs() 取标识符（uuid/nguid/eui64/csi 命令集）
            ├─ snvme_find_get_ns(nsid)  已存在？
            │     是 → nvme_validate_ns()   （校验/更新，标识符变了就移除）
            │     否 → 按 csi 分派：
            │            NVME_CSI_NVM / NVME_CSI_ZNS → nvme_alloc_ns()
            │
            └─ nvme_alloc_ns(ctrl, nsid, ids)  core.c:4001  ★把 ns 变成块设备★
                 ├─ nvme_identify_ns()  发 Identify Namespace(CNS=0x00)，读容量/LBA 格式
                 ├─ 分配 struct nvme_ns + gendisk + blk-mq queue
                 ├─ nvme_init_ns_head()  core.c:3912
                 │     nvme_find_ns_head / nvme_alloc_ns_head
                 │     （处理"多个控制器共享同一 namespace"和 multipath）
                 ├─ nvme_update_ns_info() → nvme_set_queue_limits()
                 │     设 lba_shift、容量、队列上限、discard/write-zeroes 能力
                 ├─ device_add_disk()    向系统注册 → 出现 /dev/nvmeXnY
                 └─ nvme_mpath_add_disk() [multipath.c]  挂多路径头盘
```

删除方向：`snvme_remove_namespaces()`（删控制器时清空所有 ns）、`nvme_ns_remove()`（单个 ns：`del_gendisk` + 回收）、`nvme_remove_invalid_namespaces()`（扫描后清理掉已不存在的 nsid）。

### 4+.5　ioctl.c 逐条详解：把盘当普通块设备用的标准接口

⚠️**先纠正一个最容易混淆的点**：ioctl.c 这一组 ioctl 是**标准的"把 NVMe 盘当普通块设备/做命令透传"**用的——`nvme-cli` 工具、`smartctl` 等就靠它们。它**和 snvme 把队列暴露给用户态那条私有路径完全是两回事**（后者的 `NVM_MAP_HOST_MEMORY` 等私有 ioctl 在 `snvme.ko` 的 pci.c 里，见第 5 章）。这个文件没有任何 snvme 私有命令，纯粹是 fork 自 upstream 5.15 的标准实现。

**四个入口（对应四种设备节点），最后都汇流到 `__nvme_ioctl`：**

```
nvme_ioctl(bdev,...)        /dev/nvme0n1   块设备           ┐
nvme_ns_chr_ioctl(file,...) /dev/ng0n1     per-ns 字符设备  ├─► __nvme_ioctl(ns, cmd, arg)
nvme_ns_head_ioctl(...)     multipath 头盘                  ┘        │
nvme_dev_ioctl(file,...)    /dev/nvme0     控制器字符设备 ──────────┘（单独分派，见下）

__nvme_ioctl(ns, cmd, arg)  [ioctl.c:365]
  ├─ is_ctrl_ioctl(cmd)?  → nvme_ctrl_ioctl()   （控制器级命令）
  └─ 否                    → nvme_ns_ioctl()     （namespace 级命令）
```

**命令字逐条作用：**

| ioctl 命令 | 入口/处理函数 | 作用 |
|------------|---------------|------|
| `NVME_IOCTL_ID` | `nvme_ns_ioctl` | 直接返回该 namespace 的 ns_id（最简单的查询） |
| `NVME_IOCTL_SUBMIT_IO` | `nvme_submit_io` | 提交一条简单读/写/compare。把用户的 `nvme_user_io` 组成 `rw` 命令下发 |
| `NVME_IOCTL_IO_CMD` | `nvme_user_cmd(ns)` | **IO 命令透传**：用户给完整命令，在某个 ns 上执行 |
| `NVME_IOCTL_IO64_CMD` | `nvme_user_cmd64(ns)` | 同上，64 位 result 版本 |
| `NVME_IOCTL_ADMIN_CMD` | `nvme_user_cmd(NULL)` | **admin 命令透传**（走 admin_q，需 `CAP_SYS_ADMIN`）。nvme-cli 的 identify/get-log 等都走它 |
| `NVME_IOCTL_ADMIN64_CMD` | `nvme_user_cmd64(NULL)` | 同上，64 位 result |
| （SED/Opal 系列）| `sed_ioctl` | 自加密盘 TCG Opal 安全命令 |
| `NVME_IOCTL_RESET` | `nvme_reset_ctrl_sync` | 重置控制器（仅 `/dev/nvmeX` 控制器节点 `nvme_dev_ioctl`） |
| `NVME_IOCTL_SUBSYS_RESET` | `nvme_reset_subsystem` | 子系统级重置 |
| `NVME_IOCTL_RESCAN` | `nvme_queue_scan` | 触发重新扫描 namespace（接 4+.4） |

**透传命令的内部流水（最值得理解的一条）**：

```
nvme_user_cmd(ctrl, ns, ucmd) [ioctl.c:206]
  ├─ capable(CAP_SYS_ADMIN) 权限检查
  ├─ copy_from_user 拷入 nvme_passthru_cmd
  ├─ nvme_validate_passthru_nsid()  校验 nsid 与 ns 一致
  ├─ 把用户的 opcode/cdw10..15/nsid 填进 struct nvme_command c
  └─ nvme_submit_user_cmd(q, &c, 用户数据buf, len, 元数据buf, ...) [ioctl.c:57]
       ├─ snvme_alloc_request()        ← 复用 core.c 导出符号
       ├─ blk_rq_map_user()            把用户态数据缓冲映射进 request（DMA）
       ├─ nvme_add_user_metadata()     若带保护信息(PI/metadata)
       ├─ nvme_execute_passthru_rq()   下发并等完成
       └─ copy_to_user 把 result / 元数据 回拷给用户态
```

> 注意这里也出现了 `[SNVME-NPU]` 的 5.15→5.10 适配：`nvme_submit_user_cmd` 里 5.15 用 `ns->disk->part0`（`block_device*`）关联 bio，5.10 没有这个字段，改用 `bio->bi_disk = ns->disk`（ioctl.c:64-101）。功能不变，只是 API 形态差异。

### 4+.6　本章小结（一句话串起 5 条链）

- **生命周期**：`init_ctrl → enable_ctrl → init_ctrl_finish(identify) → change_ctrl_state(LIVE) → start_ctrl`，出错走 `reset_ctrl`，拔盘走 `delete_ctrl`。
- **下行**：blk-mq → `snvme_setup_cmd` → `nvme_setup_rw/flush/discard...` → 写 SQ。
- **上行**：SSD 写 CQ → `snvme_complete_rq` → `decide_disposition` → 完成/重试/切路径。
- **扫描**：`scan_work → scan_ns_list → validate_or_alloc_ns → alloc_ns →` 出现 `/dev/nvmeXnY`。
- **ioctl.c**：四个入口 → `__nvme_ioctl` → 透传/重置/重扫，是"把盘当普通块设备用"的标准接口，**不是** snvme 私有路径。

这 5 条链里凡是 `snvme_` 开头的都被导出，正是为了让 `snvme.ko`（下一章）能借用它们去驱动盘、建队列。

---

## 第 5 章　模块 B：snvme.ko —— 把队列暴露给用户态（重点）

这是整个项目的灵魂。按"从底层到上层"的顺序讲。

### 5.1 list.c / list.h —— 最底层的双向链表原语

**作用**：一个带哨兵头节点 + 自旋锁保护的"侵入式双向循环链表"，供 ctrl/map 对象挂到全局表上。

- `struct list_node { list *list; next; prev; }`：被**内嵌**进 `struct ctrl` 和 `struct map` 的第一个字段，靠 `container_of` 从节点反查出宿主对象。
- `struct list { list_node head; spinlock_t lock; }`：永远有一个空 head 作哨兵。
- 函数：`list_init`（head 自环）、`list_insert`（加锁尾插）、`list_remove`（加锁摘除并清 NULL）、`list_next`（遍历，回到 head 即返回 NULL 终止）。

pci.c 用它维护 **4 个全局表**：`ctrl_list`（所有字符设备控制器）、`host_list`（pin 住的主机内存映射）、`device_list`（GPU 数据显存映射）、`device_queue_list`（GPU 队列环映射）。

> 类比：这就是 Linux 内核 `list_head` 的一个简化自制版。理解为"把对象串成链表，方便遍历查找和卸载时回收"即可。

### 5.2 ctrl.c / ctrl.h —— "一块盘 ↔ 一个 /dev/ssnvmeN 字符设备"

**作用**：把一个 PCI NVMe 控制器封装成一个能被用户态 open/ioctl/mmap 的字符设备节点 `/dev/ssnvme<N>`。

核心结构 `struct ctrl`（ctrl.h）关键字段：
- 设备身份：`pdev`（PCI 设备）、`name`（"ssnvme%d"）、`number`（minor 号）、`cdev`（字符设备）、`dev`（指向底层 `nvme_dev`）。
- 用户队列预算：`ioq_num` / `cq_num` / `ioq_map_num`（已注册的 DMA 映射数）。
- 用户 QID 池：`user_qid_bitmap`（位图，1=占用）、`user_qid_first/last`。**QID 空间布局**：`0`=admin 队列，`1..online_q-1`=内核 IOQ，`online_q..`=留给用户态的 IOQ 池。

主要函数：
- `ctrl_get` / `ctrl_put`：分配/释放 ctrl 对象，挂入/摘出 `ctrl_list`。
- `ctrl_find_by_inode`：**所有 ioctl/mmap 入口都靠它**——从用户态打开的 `file → inode → i_cdev` 反查回对应的 ctrl 对象。
- `ctrl_find_by_pci_dev`：按 PCI 设备查 ctrl。
- `ctrl_chrdev_create` / `_remove`：`cdev_add` + `device_create` 生成 `/dev/ssnvme<N>` 节点。

### 5.3 map.c / map.h —— DMA 映射引擎（★最关键★）

**作用**：把用户态传入的虚拟地址区间**pin 住**并做 **DMA 映射**，得到一组**总线（DMA）地址**回传给用户态。这些地址正是后续要填进 `Create I/O SQ/CQ` 命令的"队列环物理地址"。

核心结构 `struct map`（map.h）关键字段：
- `vaddr`：起始用户虚拟地址；`page_size`：页大小（主机 PAGE_SIZE 或 GPU 64KB）。
- `kind`：枚举 `RING_SQ / RING_CQ / DATA / UNSPECIFIED`——标明这块内存是 SQ 环、CQ 环还是数据缓冲。
- `n_addrs` + 柔性数组 `addrs[]`：**每一页的总线地址**，就是回传给用户态的 `ioaddrs`。

主要函数（按内存来源分两条路）：
- **主机内存路径（第一阶段在用的）**：
  - `map_userspace` → `map_user_pages`：用 `get_user_pages(FOLL_WRITE)` 把用户页**pin 在物理内存里不让换出**，再逐页 `dma_map_page(DMA_BIDIRECTIONAL)` 得到总线地址填进 `addrs[]`。
  - release 回调 `release_user_pages`：`dma_unmap_page` + `put_page` 解除。
- **GPU 显存路径（当前是桩，NPU 上走不到）**：
  - `map_gpu_memory` / `map_gpu_ioqueue_memory`：调 `nvfs_nvidia_p2p_get_pages` + `nvfs_nvidia_p2p_dma_map_pages`（见第 7 章）。
- 查找/回收：`map_find`（按 owner+vaddr）、`unmap_and_release`、`map_purge_by_owner`（进程退出时回收它 pin 的页）。

> **理解要点**：`map->addrs[]` 这个数组身兼两职——① `copy_to_user` 回传给用户态（让用户态知道队列的物理地址）；② 被 pci.c 的 `adapter_alloc_*_user` 当作 PRP1 填进 NVMe 命令（让 SSD 知道去哪 DMA）。它就是"用户内存"和"SSD 物理视角"之间的那座桥。

### 5.4 pci.c —— 主体（自带 probe 的驱动 + 字符设备 + ioctl + mmap）

这是个 6000+ 行的大文件，分两半：
- **前半**：fork 自原生 `pci.c` 的 NVMe PCIe 驱动（probe、建队列、doorbell、中断），但调用的是 `snvme_*` 导出符号。
- **后半（约 3800 行起，灵魂所在）**：把控制器资源暴露给用户态的字符设备层。

#### 两个字符设备

| 设备节点 | fops | 作用 |
|----------|------|------|
| `/dev/snvm_control` | `snvm_fops` | **控制面**：全局唯一，做 BDF 级别的 bind/unbind、创建/删除字符设备 |
| `/dev/ssnvme<N>` | `snvm_dev_fops` | **数据面**：每块被接管的盘一个，做映射内存/建队列/doorbell/mmap |

#### 控制面 ioctl `snvm_ioctl`（载荷是 PCI 地址 `{domain,bus,slot,func}`）

| 命令 | 作用 |
|------|------|
| `SNVM_DEVICE_BIND` | 把内核自带 `nvme` 从目标盘上解绑，再用 `driver_override="snvme"` 精准把 snvme 绑上去（绝不误绑回原生 nvme） |
| `SNVM_DEVICE_UNBIND` | 校验当前驱动是 snvme 后解绑 |
| `SNVM_CHRDEV_CREATE` | 为该盘创建 `/dev/ssnvme<N>`（`ctrl_get` + `ctrl_chrdev_create`），回传 minor 号 |
| `SNVM_CHRDEV_REMOVE` | 删除字符设备 |

#### 数据面 ioctl `snvm_dev_map_ioctl`（每个入口先 `ctrl_find_by_inode` 找回 ctrl）

| 命令 | 作用 |
|------|------|
| `NVM_MAP_HOST_MEMORY` | 调 `map_userspace` pin 主机页 → `copy_to_user` 回传 `ioaddrs[]`（**注册 SQ/CQ 环或数据缓冲**） |
| `NVM_MAP_DEVICE_MEMORY` | 注册 GPU 数据显存（走 nvfs p2p，当前是桩） |
| `NVM_MAP_DEVICE_QUEUE_MEMORY` | 注册 GPU 队列环显存（当前是桩） |
| `NVM_UNMAP_*` | 解除映射 + 回滚计数 |
| `NVM_GET_DEV_INFO` | **回传 `bar0_size / dstrd(doorbell 步长) / q_depth / max_user_qid` 等**，用户态据此算 doorbell 偏移、环大小、QID 范围 |
| `NVM_CREATE_QUEUE_GROUP` | 分配一个 group_id（per-fd 队列组） |
| `NVM_ADD_USER_QUEUE` | **★核心★**：建好 SQ/CQ 并回传 doorbell 偏移（见下） |
| `NVM_SET_IOQ_NUM` / `NVM_SET_SHARE_REG` / … | 传统 legacy 流程的预算声明 |

#### `NVM_ADD_USER_QUEUE`：内核侧"建好 SQ/CQ 并交出 doorbell 偏移"

这是把前面所有铺垫串起来的关键，流程：
1. 校验设备确实绑在 snvme、admin 队列 live。
2. 按用户传入的队列环虚拟地址，在队列组里找回对应的 `struct map`（并校验 kind 是 RING_SQ / RING_CQ）。
3. 从 QID 池分配一个 QID。
4. **按 NVMe 规范顺序：先 `adapter_alloc_cq_user`（建 CQ）再 `adapter_alloc_sq_user`（建 SQ）**——把 `map->addrs[0]`（用户队列环的 DMA 地址）填进 `Create I/O CQ/SQ` 命令的 PRP1 字段，经 `snvme_submit_sync_cmd` 走 admin 队列发给 SSD。**此刻 SSD 就认得"用户的这块内存是它的 SQ/CQ 环"了。**
5. 成功后**回填 doorbell 偏移**给用户态：
   - `sq_doorbell_offset = NVME_REG_DBS + qid*2*db_stride*4`
   - `cq_doorbell_offset = NVME_REG_DBS + (qid*2+1)*db_stride*4`
6. 任一步失败就反向 `Delete I/O SQ/CQ` + 还 QID，保证 all-or-nothing 原子性。

#### mmap 实现 `svm_mmap_registers`

极简但关键：找回 ctrl → 校验长度 ≤ BAR0 大小 → **`pgprot_noncached`（寄存器映射必须关 cache）** → `vm_iomap_memory` 把整个 BAR0 物理寄存器区映射进用户态。**用户态拿到这个映射 + 上面的 doorbell 偏移，就能直接写 doorbell 寄存器了。**

#### 崩溃自动清理 `snvm_dev_release`

`open` 时会建一个 `snvm_dev_owner` 挂在 `file->private_data` 上，记录这个 fd 创建的所有队列组、DMA 映射、QID。进程一旦退出（哪怕是崩溃），`release` 会分多趟自动：销毁队列（Delete I/O SQ/CQ）→ 释放数据映射 → 回收 pin 的页 → 还 QID → 回滚计数。**这解决了"进程崩溃漏掉 pin 页/计数变脏导致 rmmod 卸载失败"的问题。**

---

---

## 第 5+ 章　snvme.ko 深入：结合代码的逐函数解读

> 上一章（第 5 章）给了 snvme.ko 的"地图"，这一章按 4+ 章同样的粒度，**结合真实代码 + 行号**把 list.c / ctrl.c / map.c / pci.c 四个文件的关键函数逐一读透。行号对应 `snvme-5.10-npu/` 下的源码。

### 5+.1　list.c：侵入式双向链表（54 行，全看懂）

整个文件就 3 个函数 + 1 个头文件宏，但它撑起了 pci.c 里 4 张全局表，必须先懂它。

**核心技巧：侵入式 + container_of 反查。** `struct list_node`（list.h:16）被**内嵌**进 `struct ctrl` 和 `struct map` 的**第一个字段**，所以拿到 node 指针就能反推出宿主：

```c
struct ctrl* ctrl = container_of(element, struct ctrl, list);   // ctrl.c:75
struct map*  map  = container_of(element, struct map, list);    // map.c:116
```

**三个函数（list.c）：**
```c
list_init(list)       // :10  head 自环：head.prev = head.next = &head；初始化自旋锁
list_insert(list, e)  // :38  尾插：插到 head.prev 之后（加 spinlock 保护）
list_remove(e)        // :21  摘除：prev->next=next; next->prev=prev; 然后把 e 的指针清 NULL
```
`list_remove` 有三重保护（list.c:23）：`element != NULL && element->list != NULL && element != &head`——保证空节点、已摘节点、哨兵 head 都不会被误删。

**遍历宏（list.h:51）**——遍历终止的精髓：
```c
#define list_next(current)  \
    ( ((current)->next != &(current)->list->head) ? (current)->next : NULL )
// next 绕回 head 就返回 NULL，所以遍历写成 while(element != NULL)
```

**配合**：`ctrl_find_by_inode`、`map_find`、`map_purge_by_owner` 全是"`list_next` 循环 + `container_of` 反查 + 比字段"这一个套路。

### 5+.2　ctrl.c：控制器对象 ↔ 字符设备（167 行）

**`struct ctrl`（ctrl.h:53）关键字段**——注意它内嵌的 QID 池布局注释（ctrl.h:86）非常关键：
```c
struct ctrl {
    struct list_node list;          // 内嵌链表节点（挂 ctrl_list），必须放第一个
    struct pci_dev*  pdev;          // 对应的物理 PCI 设备
    char   name[64];                // "ssnvme%d"
    int    number;                  // minor 号
    struct cdev cdev;               // 字符设备
    struct nvme_dev *dev;
    unsigned int ioq_num, cq_num, ioq_map_num;   // legacy 模式的用户队列计数
    struct snvm_queue_setup setup;  // NVM_SET_IOQ_NUM 的快照（ctrl.h:35）
    /* B3 用户 QID 池： */
    unsigned long *user_qid_bitmap; // 位图，1=占用
    unsigned int   user_qid_first;  // = online_queues（首次 ADD 时确定）
    unsigned int   user_qid_last;   // = nr_allocated_queues - 1
    struct mutex   user_qid_lock;
};
// QID 空间：0=admin，1..online_q-1=内核IOQ，online_q..=用户IOQ池（ctrl.h:86 注释）
```

**主要函数：**
- `ctrl_get`（ctrl.c:13）：`kmalloc` 一个 ctrl，`list_node_init`，清零所有计数/QID 池，`snprintf(name, "ssnvme%d")`，`list_insert(list, &ctrl->list)` 挂表。
- `ctrl_find_by_inode`（ctrl.c:90）★：所有数据面 ioctl/mmap 的入口都靠它——遍历 ctrl_list，比 `&ctrl->cdev == inode->i_cdev`，把"用户打开的 fd"反查回 ctrl。
- `ctrl_find_by_pci_dev`（ctrl.c:68）：按 `ctrl->pdev == pdev` 查，bind/创建字符设备时用。
- `ctrl_chrdev_create`（ctrl.c:112）：`MKDEV(major, number)` → `cdev_init(&ctrl->cdev, fops)` + `cdev_add` + `device_create(cls,..., name)`，于是 `/dev/ssnvme<N>` 出现。
- `ctrl_put`（ctrl.c:52）：`list_remove` → `ctrl_chrdev_remove`（device_destroy + cdev_del）→ 释放 QID 位图 → `kfree`。

### 5+.3　map.c：DMA 映射引擎（679 行，★最关键★）

**`struct map`（map.h:21）—— 注意结尾的柔性数组：**
```c
struct map {
    struct list_node list;       // 挂全局表（host_list/device_list/...）
    struct task_struct* owner;   // = current，崩溃清理时按它回收
    u64    vaddr;                // 起始用户虚拟地址（已页对齐）
    unsigned long page_size;     // PAGE_SIZE(主机4K) 或 GPU_PAGE_SIZE(64K)
    void*  data;                 // 主机路径=pages[]数组；GPU路径=gpu_region
    release release;             // 回调：release_user_pages / release_gpu_*
    struct list_head group_link; // 挂 per-fd 队列组 g->maps 或 own->data_maps（B2/B6）
    uint32_t group_id;           // 0=legacy 全局表；!=0=属于某队列组
    uint8_t  kind;               // RING_SQ / RING_CQ / DATA / UNSPECIFIED（B6）
    unsigned long n_addrs;       // 页数
    uint64_t addrs[1];           // ★柔性数组：每页的总线(DMA)地址★
};
```
`addrs[]` 用"结构体尾部柔性数组"实现——`create_descriptor`（map.c:37）按 `sizeof(struct map) + (n_pages-1)*sizeof(uint64_t)` 一次 `kvmalloc` 出来。

**主机内存映射的两步魔法（最该理解的函数）`map_user_pages`（map.c:236）：**
```c
// 第1步：把用户页 pin 在物理内存里、不让换出
retval = get_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE, pages, NULL);  // :257
map->data    = pages;                  // 记下 page 数组，release 时 put_page 用
map->release = release_user_pages;
// 第2步：逐页做 DMA 映射，得到 SSD 能用的总线地址
for (i = 0; i < map->n_addrs; ++i) {
    map->addrs[i] = dma_map_page(dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);  // :278
    dma_mapping_error(dev, map->addrs[i]); // 校验
}
```
> 这两行就是"用户内存 → SSD 可 DMA 的物理地址"的全部秘密：`get_user_pages` 锁页，`dma_map_page` 过 IOMMU 拿总线地址填进 `addrs[]`。`dev = &map->pdev->dev`（map.c:275）——注意映射是针对**这块 NVMe 盘的 PCI 设备**做的。

**对外入口 `map_userspace`（map.c:294）：**
```c
md = create_descriptor(ctrl, vaddr & PAGE_MASK, n_pages);  // :304 页对齐 + 分配描述符
md->page_size = PAGE_SIZE;
err = map_user_pages(md);                                  // :312 上面两步
if (err) { unmap_and_release(md); return ERR_PTR(err); }
list_insert(list, &md->list);                              // :319 挂进全局 host_list
return md;
```

**反向释放 `release_user_pages`（map.c:210）：** 逐页 `dma_unmap_page` + `put_page`，再 `kvfree(pages)`。由 `unmap_and_release`（map.c:82）通过 `map->release` 回调触发。

**`unmap_and_release`（map.c:82）：** `list_remove(&map->list)`（摘全局表）→ `list_del(&map->group_link)`（摘队列组表，没挂也安全因为 create_descriptor 里 `INIT_LIST_HEAD` 过）→ 调 release 回调 → `kvfree(map)`。

**查找 `map_find`（map.c:109）：** 按 `owner==current` + `vaddr` 命中（同时试 `PAGE_MASK` 和 `GPU_PAGE_MASK` 两种对齐，map.c:120），因为主机页和 GPU 页对齐粒度不同。

**崩溃回收 `map_purge_by_owner`（map.c:164）—— 有个精妙的坑：**
```c
if (map->owner == owner && map->group_id == 0) {   // :194 只回收 legacy(非组) map！
    unmap_and_release(map);
    element = list_next(&list->head);  // head 变了，从新头重新开始（:199）
    continue;
}
```
为什么**只回收 group_id==0**？注释（map.c:177）说得很清楚：组内 map 是 per-fd 拥有的，由 `destroy_qgroup_locked` 在 fd 关闭时先行处理。如果这里也回收组内 map，一个进程开了多个 `/dev/ssnvme` fd 时，关一个 fd 会误删另一个 fd 注册的 map（因为它们 `map->owner` 都是同一个 task）——这正是 B2 冒烟测试发现的 bug。

**GPU 路径（当前是桩）`map_gpu_memory`（map.c:461）：** 调 `nvfs_nvidia_p2p_get_pages`（map.c:496）pin 显存 + 对每个 ctrl 调 `nvfs_nvidia_p2p_dma_map_pages`（map.c:512），把 `gd->mappings[0]->dma_addresses[i]` 填进 `addrs[]`（map.c:527）。NPU 机器上这些 nvfs 函数返回 -ENOMEM，走不到。

### 5+.4　pci.c 字符设备层：结合代码

#### 两个 per-fd 运行态结构

**`struct snvm_dev_owner`（pci.c:3840）—— 挂在 `file->private_data`，崩溃清理的总账本：**
```c
struct snvm_dev_owner {
    struct ctrl *ctrl;
    struct task_struct *owner;      // open 时的 current
    struct list_head groups;        // 本 fd 的队列组链表（snvm_qgroup）
    struct mutex groups_lock;
    unsigned int nr_groups;
    struct list_head data_maps;     // 本 fd 的 DATA 类映射（B6）
    struct mutex data_maps_lock;
    unsigned int nr_data_maps;
};
```

**`struct snvm_qgroup`（pci.c:3917）：** `group_id`（IDA 分配）、`max_queues`(=16)、`maps`（组内 map 链）、内联 `queues[16]`（每个 `snvm_user_queue{qid, alive, sq_vaddr, cq_vaddr}`，pci.c:3957）+ `cur_queues`。

#### 控制面 `snvm_ioctl`（pci.c:6047）

```c
copy_from_user(&dev_addr, ...);     // 载荷是 pci_device_addr{domain,bus,slot,func}
switch (cmd) {
  case SNVM_DEVICE_BIND:   return snvm_rebind_driver(dev_addr);   // 抢盘
  case SNVM_DEVICE_UNBIND: return snvm_unbind_driver(dev_addr);
  case SNVM_CHRDEV_CREATE: snvm_chrdev_helper(&dev_addr, 1); + copy_to_user 回传 minor;
  case SNVM_CHRDEV_REMOVE: return snvm_chrdev_helper(&dev_addr, 0);
}
```

**`snvm_rebind_driver`（pci.c:5855）—— 抢盘 + NPU 适配点：**
```c
device_release_driver(&pdev->dev);   // :5874 先把 stock nvme 从这个 BDF 解绑
register_driver();                   // :5887 注册 snvme 这个 pci_driver
#if LINUX_VERSION_CODE < KERNEL_VERSION(5,15,0)   // [SNVME-NPU batch8c] openEuler 5.10 没导出 device_driver_attach
    pdev->driver_override = kstrdup("snvme", GFP_KERNEL);  // :5926 钉死只认 snvme
    ret = device_attach(&pdev->dev);                       // :5929 触发匹配+probe，绝不误绑回 nvme
#else
    ret = device_driver_attach(&snvme_driver.driver, &pdev->dev);  // 5.15 路径
#endif
```
> 这就是第 8 章说的 batch8c：5.10 内核没导出 `device_driver_attach`，改用 PCI 的 `driver_override` 机制——把 override 钉成 "snvme"，PCI 总线匹配就只认 snvme，连同样匹配 `PCI_CLASS_STORAGE_EXPRESS` 的 stock nvme 也被排除。

**`snvm_chrdev_helper`（pci.c:5979）：** `ctrl_find_by_pci_dev` 查是否已有；create 且没有 → `snvm_chrdev_create` 建 `/dev/ssnvme<N>`，把 minor 经 `dev_addr->domain` 回传（**幂等**：已存在直接回传现有 minor）。remove 时**先 `ctrl_put` 再还 IDA minor**（pci.c:6030，顺序很重要，防 race）。

#### 数据面 `snvm_dev_map_ioctl`（pci.c:4314）

入口先 `ctrl_find_by_inode` 找回 ctrl，再 `switch(cmd)` 分发。逐条看最重要的几个：

**① `NVM_MAP_HOST_MEMORY`（pci.c:4333）—— pin 内存 + 三路由 + 回传地址：**
```c
copy_from_user(&request, arg);                              // 校验 reserved/map_kind
map = map_userspace(&host_list, ctrl, request.vaddr_start, request.n_pages);  // :4382 → 5+.3
map->kind = request.map_kind;
// 三种归属路由（决定生命周期）：
if (map_kind == DATA)            list_add_tail(&map->group_link, &own->data_maps);  // :4400 fd级
else if (group_id != 0)          list_add_tail(&map->group_link, &g->maps);          // :4419 组级
else if (ioq_idx >= 0)           ctrl->ioq_map_num++; (is_cq? ctrl->cq_num++);       // :4443 legacy
// 回传每页 DMA 地址给用户态：
copy_to_user(request.ioaddrs, map->addrs, map->n_addrs * sizeof(uint64_t));  // :4453
// 若 copy_to_user 失败：回滚刚加的计数 + unmap_and_release（:4458 起）
```
> 三路由就是 map.h 注释里说的 DATA（挂 `own->data_maps`，fd 关闭才回收）/ RING（挂 `g->maps`，组销毁时回收）/ legacy（记 ctrl 计数）。

**② `NVM_CREATE_QUEUE_GROUP`（pci.c:5016）：** `ida_simple_get(&snvm_queue_group_ida, 1, 0)` 分配 group_id（从 1 开始，0 是"无组"哨兵），建 `snvm_qgroup` 挂 `own->groups`，`copy_to_user` 回传 group_id + max_queues。

**③ `NVM_GET_DEV_INFO`（pci.c:4882）—— 回传用户态算队列/doorbell 所需的全部参数：**
```c
// 关键 race fix：scan_work 是异步的，nsid=1 可能还没扫出来
ns = snvme_find_get_ns(&ndev->ctrl, 1);
if (!ns) { flush_work(&ndev->ctrl.scan_work); ... 最多等 5s (msleep 轮询) }  // :4912
// 回传字段：
drequest.start_cq_idx = ndev->user_start_qid ?: ndev->online_queues;  // 用户 QID 起点
drequest.dstrd        = ndev->db_stride;                 // doorbell 步长 ★算偏移用
drequest.q_depth      = ndev->q_depth;                   // 每个队列深度 ★算环大小用
drequest.bar0_size    = pci_resource_len(ctrl->pdev, 0); // ★mmap 长度
drequest.max_user_qid = ndev->ctrl_max_io_queues;        // QID 池上界
drequest.sgl_supported= ndev->ctrl.sgls;
copy_to_user(arg, &drequest, ...);
```

**④ `NVM_ADD_USER_QUEUE`（pci.c:5156）★★ 整个机制的高潮 ★★** —— 把已注册的队列环交给 SSD、回传 doorbell 偏移。完整流程：

```c
// (a) 校验：flags/reserved MBZ、1<=nr_pairs<=16、batch 内 vaddr 去重（:5217-5245）
// (b) 确认设备真绑在 snvme 且 admin_q live：
uq_ndev = snvm_ctrl_get_live_ndev(ctrl);   // :5253 → 见下方说明
// (c) 找回队列组，校验 cur_queues + nr_pairs <= max_queues（:5263-5275）
g = find_qgroup_locked(own, req->group_id);
// (d) 按 vaddr 在 g->maps 里解析出每对 SQ/CQ 的 struct map（:5293-5343）：
//     用 map 自己的 page_size 算 mask；B6 还校验 kind：RING_SQ↔sq、RING_CQ↔cq，DATA 不许匹配
sq_maps[i] = m_sq;  cq_maps[i] = m_cq;
// (e) 分配 QID（首次惰性建池）：
snvm_user_qid_pool_init_locked(ctrl, uq_ndev);          // :5348
snvm_user_qid_alloc_locked(ctrl, req->nr_pairs, qids);  // :5354 从位图找空位
// (f) 驱动 SSD 建队列：NVMe 规范要求先 CQ 后 SQ（:5371）
for (i ...) {
    adapter_alloc_cq_user(uq_ndev, cq_maps[i], qids[i]); // 先建 CQ
    adapter_alloc_sq_user(uq_ndev, sq_maps[i], qids[i]); // 再建 SQ
}
// (g) 提交到组描述符 + 回填 doorbell 偏移（:5392-5414）★
uq->qid = qid; uq->alive = 1; uq->sq_vaddr/cq_vaddr = ...;
req->out_pairs[i].sq_doorbell_offset = NVME_REG_DBS + qid*2*db_stride*4;       // :5407
req->out_pairs[i].cq_doorbell_offset = NVME_REG_DBS + (qid*2+1)*db_stride*4;   // :5409
g->cur_queues += req->nr_pairs;
copy_to_user(arg, req, ...);   // 回传 doorbell 偏移给用户态
// 任一步失败 → rollback_unlocked：Delete 已建的 SQ/CQ + 还 QID（all-or-nothing）
```

**`adapter_alloc_cq_user`（pci.c:1303）/ `adapter_alloc_sq_user`（pci.c:1323）—— 把 DMA 地址变成 SSD 的队列：**
```c
c.create_cq.opcode = nvme_admin_create_cq;
c.create_cq.prp1   = cpu_to_le64(q_map->addrs[0]);  // ★用户队列环的 DMA 地址当 PRP1★
c.create_cq.cqid   = cpu_to_le16(qid);
c.create_cq.qsize  = cpu_to_le16(dev->q_depth - 1);
return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);  // 走 admin 队列发给 SSD
```
> 这就是闭环的关键一笔：`q_map->addrs[0]`（map.c 里 `dma_map_page` 得到的总线地址）被填进 Create I/O CQ/SQ 命令的 PRP1，发给 SSD 后，**SSD 就认得"用户那块内存是它的 SQ/CQ 环"了**，以后会 DMA 读 SQ、DMA 写 CQ。注意它复用了 core.c 导出的 `snvme_submit_sync_cmd`（呼应第 4+.2 节）。

**`snvm_ctrl_get_live_ndev`（pci.c:4281）—— 为什么不能直接用 pci_get_drvdata：** 因为 in-tree 的 `nvme` 驱动也把 nvme_dev 存在 drvdata 里。这个函数先查 `pdev->dev.driver` 名字是不是 "snvme"，确认是 snvme 接管了才返回 ndev，否则返回 NULL（让 ioctl 报 -ENODEV）——避免对一块其实还归 stock nvme 管的盘乱发 admin 命令。

#### mmap：把 doorbell 寄存器映射进用户态

**`svm_mmap_registers`（pci.c:5509）—— 极简但关键：**
```c
ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
if (vma->vm_end - vma->vm_start > pci_resource_len(ctrl->pdev, 0)) return -EINVAL; // 不超 BAR0
vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);    // ★寄存器必须关 cache★
return vm_iomap_memory(vma, pci_resource_start(ctrl->pdev, 0), len);  // 映射整个 BAR0 物理区
```
> 用户态拿到这个映射后，`mmap基址 + sq_doorbell_offset`（来自 ADD_USER_QUEUE）就是那个队列的 doorbell 寄存器地址，直接写就能"按门铃"。

#### open / release：建账本 + 崩溃自动清理

**`snvm_dev_open`（pci.c:5529）：** `ctrl_find_by_inode` 找 ctrl → `kzalloc` 一个 `snvm_dev_owner`，记 `current`、初始化 groups/data_maps 两个链表 → 挂 `file->private_data`。

**`snvm_dev_release`（pci.c:5556）—— 进程退出时分多趟回收（注释里强调顺序）：**
```
Pass 0  (:5634)  级联销毁所有 qgroup：destroy_qgroup_locked（含 Delete I/O SQ/CQ + 还 QID）
Pass 0.5(:5610)  释放所有 DATA maps（own->data_maps）
Pass 1  (:5630)  遍历 host_list 统计 legacy map 要回滚的 ioq/cq 计数
Pass 2           map_purge_by_owner 释放 legacy map（只 group_id==0 的）+ 回滚 ctrl 计数
```
**为什么 Pass 0 必须最先**（pci.c:5566 注释）：队列组里停着用户 IO 队列和 SSD 正在 DMA 的环 map；若先把全局 map 表释放了，`destroy_qgroup_locked` 发 Delete I/O SQ/CQ 时，SSD 可能还在 DMA 已经被解映射的页 → use-after-free。

**`destroy_qgroup_locked`（pci.c:4159）—— 销毁一个队列组的标准三步：**
```
Step 1 (:4226) 排空用户队列：对每个 alive 的 qid，按 NVMe 规范发 Delete I/O SQ 再 Delete I/O CQ，
               然后 snvm_user_qid_free_locked 还 QID。（ndev 为 NULL 时跳过 admin 命令——
               应对 unbind/rebind race，让清理幂等）
Step 2 (:4254) 排空 maps：对 g->maps 每个 unmap_and_release（释放 pin 页 / p2p 引用）
Step 3 (:4270) ida_simple_remove 还 group_id + kfree(g)
```

### 5+.5　把第 6 章 8 步对应到具体函数（收口）

| 第 6 章步骤 | 实际函数（文件:行号） |
|---|---|
| ① CHRDEV_CREATE | `snvm_ioctl` → `snvm_chrdev_helper`(pci.c:5979) → `ctrl_chrdev_create`(ctrl.c:112) |
| ② DEVICE_BIND | `snvm_ioctl` → `snvm_rebind_driver`(pci.c:5855) → `device_release_driver` + `driver_override`/`device_attach` |
| ③ open | `snvm_dev_open`(pci.c:5529) 建 `snvm_dev_owner` |
| ④ GET_DEV_INFO | `snvm_dev_map_ioctl` case(pci.c:4882) 回传 dstrd/q_depth/bar0_size/max_user_qid |
| ⑤ CREATE_QUEUE_GROUP | case(pci.c:5016) `ida_simple_get` |
| ⑥ MAP_HOST_MEMORY | case(pci.c:4333) → `map_userspace`(map.c:294) → `get_user_pages`+`dma_map_page`(map.c:257/278) |
| ⑦ ADD_USER_QUEUE | case(pci.c:5156) → `adapter_alloc_cq_user`/`sq_user`(pci.c:1303/1323) 用 `addrs[0]` 当 PRP1，回传 doorbell 偏移 |
| ⑧ mmap | `svm_mmap_registers`(pci.c:5509) `pgprot_noncached` + `vm_iomap_memory` |

整条链就是：`ctrl.c` 管"盘↔字符设备"对象 → `map.c` 把用户内存变成 SSD 能 DMA 的地址 → `pci.c` 用这个地址发 admin 命令让 SSD 建队列、再把 doorbell mmap 出去 → `list.c` 在底下把所有对象串成可遍历、可回收的链表。崩溃了也不漏，靠 `snvm_dev_owner` 这本账 + `destroy_qgroup_locked` / `map_purge_by_owner` 兜底。

---

## 第 6 章　把它串起来：用户态拿到 SQ/CQ + doorbell 的完整 8 步

这是把第 0 章那 4 个问题全部回答完的**端到端通路**：

```
① open(/dev/snvm_control)
   → ioctl(SNVM_CHRDEV_CREATE, BDF)   内核建 /dev/ssnvmeN，回传 N

② ioctl(SNVM_DEVICE_BIND, BDF)
   → 把内核 nvme 解绑、精准绑上 snvme；probe 跑完，admin 队列 live、内核 IOQ 就位

③ open(/dev/ssnvmeN)                  建 snvm_dev_owner（崩溃清理靠它）

④ ioctl(NVM_GET_DEV_INFO)
   → 拿 bar0_size / db_stride / q_depth / max_user_qid（据此算环大小和 doorbell 偏移）

⑤ ioctl(NVM_CREATE_QUEUE_GROUP)       拿 group_id

⑥ 用户态自己 malloc 两块页对齐内存（SQ 环 ≥ q_depth×64B，CQ 环 ≥ q_depth×16B）
   ioctl(NVM_MAP_HOST_MEMORY, kind=RING_SQ/RING_CQ, group_id)
   → 内核 get_user_pages pin 住 + dma_map_page → 回传 DMA 地址到 ioaddrs[]   【map.c】

⑦ ioctl(NVM_ADD_USER_QUEUE, group_id, pairs[{sq_vaddr, cq_vaddr}])
   → 内核用 addrs[0] 当 PRP1 发 Create I/O CQ + Create I/O SQ 给 SSD
   → 回传 sq_doorbell_offset / cq_doorbell_offset                          【pci.c】

⑧ mmap(/dev/ssnvmeN, len=bar0_size)
   → svm_mmap_registers 把 BAR0(non-cached) 映进用户态                      【pci.c】
```

**至此闭环（全程不进内核）**：
1. 把 64B 的 SQE 写进自己的 SQ 环内存（SSD 能通过步骤⑦注册的 DMA 地址读到）；
2. 把新的 SQ tail 写进 `mmap基址 + sq_doorbell_offset`（直接按门铃）；
3. SSD 收到通知，DMA 取命令、执行、把完成项写进 CQ 环；
4. 用户态轮询 CQ 环、读完成项，再把 CQ head 写 `cq_doorbell_offset`。

这正是你提到的"主机写 SQ → 写 doorbell → SSD 取命令"那条路，只不过**主机 = 用户态进程**，队列内存 = 用户态自己的内存，doorbell = mmap 进来的 BAR0 寄存器。

---

## 第 7 章　P2P / GPUDirect 子系统（nvfs-* 与 nv-p2p.h）

第 6 章的队列内存放在**主机内存**。如果想把它（或数据缓冲）放进 **GPU/NPU 显存**，让 SSD 直接 DMA 读写显存、绕开主机内存，就需要这一套 **peer-to-peer DMA**。这套代码原样搬自 NVIDIA 的 **nvidia-fs（GPUDirect Storage）**。

| 文件 | 作用 |
|------|------|
| `nv-p2p.h` | **桩头(stub)**。顶替 NVIDIA 官方头，让没装 NVIDIA SDK 的华为机器能编译通过。只裁剪保留了 2 个结构体字段（`page_table.entries`、`dma_mapping.dma_addresses`），**没有任何函数**。 |
| `nvfs-p2p.h` | 声明 NVIDIA P2P 的 6 个核心 API 的函数指针类型 + `nvfs_*` 封装函数 |
| `nvfs-p2p.c` | 封装实现：运行时用 `__symbol_get("nvidia_p2p_*")` **动态绑定** NVIDIA 驱动导出的符号；找不到就返回 `-ENOMEM` |
| `nvfs-core.h` | P2P 公共定义：IOCTL 结构、日志宏、GPU 页常量（`GPU_PAGE_SIZE = 64KB`） |
| `nvfs-pci.c/.h` | PCIe 拓扑距离矩阵：扫描 GPU 与 NVMe 的 PCIe 路径，算"距离"，为 P2P 选最近的设备 |

**6 个 P2P 核心 API**（`nvfs-p2p.c` 转发的对象）：
- `nvidia_p2p_get_pages`：pin 住 GPU 显存、取得页表（核心 pin/map）
- `nvidia_p2p_dma_map_pages`：把 GPU 页映射成某个 PCI 设备（NVMe）能访问的 P2P DMA 地址
- `put_pages` / `dma_unmap_pages` / `free_page_table` / `free_dma_mapping`：对应的释放

**配合关系**：`map.c` 的显存路径 → 调 `nvfs-p2p.c` 的封装 → `__symbol_get` 找 NVIDIA 真符号；结构体字段类型由 `nv-p2p.h` 桩头供给；目标 NVMe 的亲和性由 `nvfs-pci.c` 的距离矩阵决定。

---

## 第 8 章　NPU 迁移：做了什么 / 还没做什么（最关心的部分）

> **结论先行**：当前处于**第一阶段——只让代码在 aarch64 / openEuler 5.10 / 无 NVIDIA SDK 的华为机器上能编译并跑通"纯主机内存"的 NVMe 直连通路**。GPU/NPU 显存的 peer-DMA **还完全没有接入**，只是用桩占位让代码编译通过、运行时安全降级。代码里**没有一行真正调用昇腾接口**，也搜不到 `ascend`/`910` 的实质实现，全是注释里的"将来要做"。

### 8.1 已完成的迁移改动（逐条）

**(1) 去 NVIDIA 依赖：HOST-ONLY 模式（最重要，`pci.c:6152–6163`，标 `[迁移修改 #1]`）**
原代码在 `nvme_init` 里找不到 `nvidia_p2p_*` 符号就 `return -EOPNOTSUPP`，导致整个模块加载失败。华为机器上必然没有 NVIDIA 驱动，所以改成：**P2P 初始化失败只打 warning、继续加载，进入 HOST-ONLY 模式**。主机内存映射照常工作；显存路径在函数指针为 NULL 时由 nvfs 封装自动返回 `-ENOMEM`，不崩。
> 注释明确写：**"将来接华为 NPU 的 peer-DMA 后端时，再把这里换成『初始化华为 backend』"** —— 这就是下一阶段的接入点。

**(2) host-only 桩头（`nv-p2p.h`，标 `[迁移新增文件 #2]`）**
华为机器缺 NVIDIA 官方 `nv-p2p.h` 会导致编译报错。用裁剪版桩头顶替，只保留 snvme 会读的 2 个字段；并用双层 include guard（主动占用 `_NV_P2P_H_`）防止和真 NVIDIA 头共存时冲突。

**(3) 去 x86 化（`pci.c:33`，标 `[batch7]`）**
原 `#include <asm/string_64.h>`（x86_64 专属）→ `<linux/string.h>`（架构中立）。因为 NPU 服务器是 **aarch64**，没有 string_64.h。

**(4) openEuler 5.10 内核 API 适配**
- `pci.c:265 [batch8]`：5.10 无 `param_set_uint_minmax` 导出符号，改用 `kstrtou32` 手动校验。
- `pci.c:5898 [batch8c]`：5.10 未导出 `device_driver_attach`，改用 PCI 的 `driver_override` 机制 + 已导出的 `device_attach()` 来绑驱动。
- `nvfs-p2p.h:29 [batch6]`：补一句文件作用域前置声明 `struct pci_dev;`，解决 5.10 上的"幽灵类型"冲突。
- core.c / ioctl.c / multipath.c 里大量 `LINUX_VERSION_CODE < KERNEL_VERSION(5,15,0)` 的条件编译，把 5.15 API（`set_capacity_and_notify`、`bvec_virt`、`blk_execute_rq` 参数变化、`bi_bdev` vs `bi_disk` 等）降级回 5.10 写法。

### 8.2 还没做的（下一阶段）

要真正接入昇腾 910C 的显存 peer-DMA，至少要改三处：
1. **`nv-p2p.h`**：把 NVIDIA 的页表/DMA 映射类型，换成华为 NPU 对应的类型。
2. **`nvfs-p2p.c`**：把 `__symbol_get("nvidia_p2p_*")` 换成华为 NPU 驱动导出的 pin/map 符号（昇腾的 peer-DMA 接口）。
3. **`nvfs-pci.c/.h`**：设备发现现在硬编码 `PCI_VENDOR_ID_NVIDIA` + 显卡 class，**昇腾 NPU 根本不会被识别为"GPU"**，需要改成华为 vendor id / NPU device class。

这与你 MEMORY 里记录的"phase 1 = kernel_modules only / 纯主机内存通路"完全一致。

---

## 第 9 章　文件配合速查表

| 文件 | 所属 ko | 一句话作用 | 配合谁 |
|------|---------|------------|--------|
| `core.c` | snvme-core | 通用 NVMe 协议栈（改名+导出符号） | 被 pci.c 调用 |
| `ioctl.c` | snvme-core | 标准块设备 ioctl（无私有命令） | 调 core.c |
| `nvme.h` | snvme-core | 公开 `snvme_*` 原型给 snvme.ko | 桥接两个 ko |
| `multipath/zns/hwmon.c` | snvme-core | 多路径/ZNS/温控（改名适配） | core.c |
| `pci.c` | snvme | 自带 probe 的驱动 + 字符设备 + ioctl + mmap | 调 core.c；用 ctrl/map/list/nvfs |
| `ctrl.c/.h` | snvme | 盘 ↔ /dev/ssnvmeN 字符设备对象 | 被 pci.c 调；内嵌 list_node |
| `map.c/.h` | snvme | ★pin 内存取 DMA 地址★ | 被 pci.c 调；显存路径调 nvfs-p2p |
| `list.c/.h` | snvme | 双向链表原语 | 被 ctrl/map/pci 复用 |
| `nvfs-p2p.c/.h` | snvme | 封装 NVIDIA P2P 6 API（当前桩） | 被 map.c 调；类型来自 nv-p2p.h |
| `nvfs-pci.c/.h` | snvme | PCIe 拓扑距离矩阵 | 给 P2P 选设备 |
| `nvfs-core.h` | snvme | P2P 公共定义/常量 | 被 nvfs-* 共享 |
| `nv-p2p.h` | snvme | host-only 桩头 | 供 nvfs-p2p / map 编译 |

**"哪几个文件配合实现什么效果"总结**：
- **想把一块盘从内核 nvme 抢过来 + 复用协议栈** = `core.c`(导出) + `nvme.h`(原型) + `pci.c`(probe & bind)。
- **想把 SQ/CQ 队列内存交给用户态** = `pci.c`(ioctl) + `map.c`(pin+DMA 地址) + `list.c`(挂表) + `ctrl.c`(字符设备)。
- **想让用户态直接按 doorbell** = `pci.c` 的 `svm_mmap_registers`(mmap BAR0) + `NVM_ADD_USER_QUEUE`(回传偏移)。
- **想把队列/数据放进 GPU/NPU 显存** = `nvfs-*` + `nv-p2p.h`（当前是桩，NPU 未接入）。

---

## 第 10 章　给新手的下一步建议

1. **先只读 host-only 这条路**（第 6 章 8 步），把 `nvfs-*` 那一套先当黑盒——反正第一阶段也走不到它。
2. **按调用顺序读 pci.c 后半段**：`snvm_ioctl`（控制面）→ `snvm_dev_open` → `snvm_dev_map_ioctl`（数据面）→ 重点啃 `NVM_ADD_USER_QUEUE` 和 `svm_mmap_registers`。
3. **对照读 `map.c` 的 `map_user_pages`**：理解 `get_user_pages` + `dma_map_page` 这两步——这是"用户内存变成 SSD 能 DMA 的物理地址"的全部魔法。
4. **想验证迁移现状**：在目标 NPU 机器上 `insmod` 两个 ko，跑 smoke 测试（snvme/qgroup/recycle/addq/io），观察 `dmesg` 里是否打印 "HOST-ONLY 模式" 的 warning——出现即说明降级成功、且确实没接 GPU/NPU 显存路径。
5. **将来要接昇腾**：从第 8.2 节列的三个文件入手，先搞清楚昇腾 NPU 驱动**有没有导出类似 `nvidia_p2p_get_pages` 的 peer-DMA pin/map 符号**——这是能不能接的前提。

---

---

## 第 11 章　snvme.ko 按 probe 流程从头走一遍（谁先谁后、每一步和谁配合）

> **为什么再开这一章。** 第 4+ 章是站在 `snvme-core.ko`（core.c）的角度，把"控制器生命周期 / 命令构造 / 完成处理 / namespace 扫描 / ioctl"五条链拆开了。第 5/5+/6 章则是站在"用户态怎么拿到队列"的角度，把 snvme.ko 的**字符设备层**（后半段）讲透了。
>
> 但这两组都**没有按 `snvme.ko` 自己的执行时间线走一遍**：盘是怎么从"插在总线上没人管"一步步被 `snvme.ko` 接管、带起硬件、建好内核队列、最后才轮到把队列交给用户态的？这一章就补这条主线——**严格按 probe 的发生顺序**，每到一步就指出"此刻 pci.c 调了哪个文件的哪个函数、为了实现什么"。读完你应该能在脑子里放一部"从插盘到能用"的连续电影，而不是一堆零散的函数。

### 11.0 一句话主线

```
装载模块(只开控制面，故意不抢盘) → 用户 CHRDEV_CREATE 埋 ctrl 记录 → 用户 DEVICE_BIND 才注册 pci_driver
   → 内核回调 nvme_probe(只搭骨架) → snvme_reset_ctrl 把活儿丢进 reset_work
   → nvme_reset_work(真正带硬件)：开 PCI → 建 admin 队列 → identify → 建内核 IO 队列 → 建 tagset → LIVE → 扫 namespace
   → 盘可用；此后用户态再走第 6 章那 8 步把"额外的"队列要到用户态手里
```

下面把每一段拆开。所有行号对应 `snvme-5.10-npu/pci.c`（除非另注文件）。

---

### 11.1 起点其实不是 probe，是"装载模块" `nvme_init`（pci.c:6147）

`insmod snvme.ko` 第一刻执行的是 `nvme_init`，它**只做三件事，而且故意"不抢盘"**：

```
nvme_init() [pci.c:6147]
  ├─① nvfs_nvidia_p2p_init()  失败也只 warning → 进入 HOST-ONLY 模式   [迁移修改#1, :6172]
  ├─② list_init 四张全局表：ctrl_list / host_list / device_list / device_queue_list  [:6178-6181]
  └─③ snvm_cdev_init()        建控制面字符设备 /dev/snvm_control       [:6087]
        ├─ class_create(DRIVER_NAME)                              [:6094]
        ├─ alloc_chrdev_region(..., max_num_ctrls, ...)           [:6102]
        ├─ cdev_init(&snvm_cdev, &snvm_fops) + cdev_add           [:6109]
        └─ device_create(..., "snvm_control")                    [:6118]  → /dev/snvm_control 出现
```

**关键："这里没有 `pci_register_driver`！"** 普通 NVMe 驱动在 `module_init` 里就 `pci_register_driver`，于是模块一装载，内核就对**总线上每一块** NVMe 盘回调 probe、全部抢走。snvme **故意不这样做**——`snvm_registered = 0`（:6150），`pci_driver` 留到用户明确点名某块盘时才注册。

> **配合关系**：`nvme_init` 只跟 `list.c`（建四张表）和"控制面 cdev"打交道，**完全不碰任何一块盘**。这就是 snvme "按需接管、绝不误伤其它盘"设计的第一道闸。

---

### 11.2 是谁把 `nvme_probe` 叫起来的：先 `CHRDEV_CREATE` 埋记录，再 `DEVICE_BIND` 才注册驱动

probe 不会自己发生，要用户态通过 `/dev/snvm_control` 连发两条控制面 ioctl（第 5+.4 节讲过函数，这里强调**顺序与因果**）：

```
① ioctl(/dev/snvm_control, SNVM_CHRDEV_CREATE, BDF)
     → snvm_chrdev_helper(pci.c:5979)
         → ctrl_get(ctrl.c:13)         ★在 ctrl_list 里为这个 BDF 建一条 ctrl 记录★
         → ctrl_chrdev_create(ctrl.c:112) → /dev/ssnvme<N> 出现
   （此刻盘还归内核 nvme 管，snvme 只是先"挂了号"）

② ioctl(/dev/snvm_control, SNVM_DEVICE_BIND, BDF)
     → snvm_rebind_driver(pci.c:5855)
         → device_release_driver(&pdev->dev)        把 stock nvme 从这个 BDF 解绑  [:5874]
         → register_driver()(pci.c:5807) → pci_register_driver(&snvme_driver)      [:5798]
         → driver_override="snvme" + device_attach() 触发 PCI 总线重新匹配         [:5926/5929]
              └─►►► 内核对这个 BDF 回调 snvme_driver.probe == nvme_probe(pci.c:3320)
```

> **因果链（重点理解）**：是 `DEVICE_BIND` 里的 `device_attach()` 触发了 `nvme_probe`，而**不是** insmod。而 `CHRDEV_CREATE` 必须先于 `DEVICE_BIND`——因为它在 `ctrl_list` 里埋下的那条记录，正是下一节 probe **第一行**要查的"准入凭证"。两条 ioctl 的先后顺序，直接决定了 probe 会不会真的接管这块盘。

---

### 11.3 `nvme_probe`（pci.c:3320）：它只"搭骨架"，按顺序干 7 件事

注意 probe **本身不碰硬件寄存器**，它只是把数据结构、BAR 映射、工作项准备好，最后把"真正带硬件"的活儿丢给 `reset_work`。逐件看，并标出和哪个文件配合：

```
nvme_probe(pdev, id) [pci.c:3320]
 ┌① 准入闸：ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev)   [ctrl.c, 调用在 pci.c:3349]
 │     若 == NULL → 打印 "user must call SNVM_CHRDEV_CREATE first" → return -ENODEV  [:3350-3355]
 │     ★这一步把 11.2① 埋的记录取出来；没埋过就放手，让内核 nvme 去接（绝不误伤）★
 │     配合文件：ctrl.c
 │
 ├② 读"用户队列预算"快照：把 ctrl->ioq_num / cq_num / use_sreg / setup.* 拷进 dev   [:3369-3410]
 │     决定 dev->use_user_allocated、nr_write_queues、cap_kernel_ioq 等
 │     配合文件：ctrl.c（这些字段是 NVM_SET_IOQ_NUM 等 ioctl 早先写进 ctrl 的）
 │
 ├③ 分配 dev：kzalloc_node(nvme_dev) + kcalloc(queues[nr_allocated_queues])   [:3363/3412]
 │     pci_set_drvdata(pdev, dev)                                            [:3418]
 │
 ├④ nvme_dev_map(dev)  [pci.c:3252]   ★把 BAR0 映射进内核★
 │     ├─ pci_request_mem_regions(pdev, "nvme")          [:3256]
 │     └─ nvme_remap_bar(dev, NVME_REG_DBS + 4096*3)     [:3259]  → dev->bar 可读写寄存器
 │     （doorbell 区就在这块 BAR 里，第 11.5 节细说）
 │
 ├⑤ INIT_WORK(&dev->ctrl.reset_work, nvme_reset_work)    [:3424]  ★把"硬件带起来"绑成工作项★
 │     INIT_WORK(&dev->remove_work, nvme_remove_dead_ctrl_work) + nvme_setup_prp_pools  [:3425/3428]
 │
 ├⑥ snvme_init_ctrl(&dev->ctrl, &pdev->dev, &nvme_pci_ctrl_ops, quirks)  [core.c, 调用在 pci.c:3460]
 │     ★第一次进 core.c★：建控制器状态机(state=NEW)、分配 instance、绑 5 个 work（见 4+.1）
 │     传进去的 nvme_pci_ctrl_ops(pci.c:3239) = {reg_read32/write32/read64, free_ctrl, ...}
 │     —— 这张 ops 表就是 core.c 反过来回调 pci.c 读写硬件寄存器的"回拨电话"
 │     配合文件：core.c（提供通用协议栈）+ pci.c（提供 ops 实现）
 │
 └⑦ snvme_reset_ctrl(&dev->ctrl)  [core.c, 调用在 pci.c:3467]
       → change_ctrl_state(RESETTING) → queue_work(reset_work)   ★把活儿丢出去★
    async_schedule(nvme_async_probe, dev)  [:3468]  → 异步 flush_work(reset_work)+flush_work(scan_work)
    return 0    （probe 到此返回，硬件还没真正起来！真正的活儿在 reset_work 里）
```

> **一句话**：probe = ①查凭证(ctrl.c) ②读预算(ctrl.c) ③④搭内存/映射 BAR ⑤绑工作项 ⑥进 core.c 建控制器对象 ⑦把硬件带起来的活儿丢进 reset_work。**probe 自己不开机、不建队列**——它只是导演喊"预备"，真正"开拍"在下一节。

---

### 11.4 `nvme_reset_work`（pci.c:3061）：真正把盘带到能用，按 11 步走

这是 snvme.ko 前半段的**主战场**，也是和 `snvme-core.ko` 配合最密集的地方。它跑在内核工作队列里（异步），顺序如下：

```
nvme_reset_work(work) [pci.c:3061]
 ┌① nvme_pci_enable(dev)  [pci.c:2830]   ★开 PCI、读能力、算出 doorbell 基址★
 │    ├─ pci_enable_device_mem + pci_set_master            [:2836/2839]  开内存空间 + 允许 DMA 主控
 │    ├─ pci_alloc_irq_vectors(pdev, 1, 1, ...)            [:2856]       先要 1 个中断向量过渡
 │    ├─ dev->ctrl.cap = readq(NVME_REG_CAP)               [:2860]       读控制器能力寄存器
 │    ├─ dev->q_depth   = min(MQES+1, io_queue_depth)      [:2862]  ★每队列深度
 │    ├─ dev->db_stride = 1 << CAP_STRIDE(cap)             [:2866]  ★doorbell 步长(算偏移要它)
 │    └─ dev->dbs       = dev->bar + 4096                  [:2868]  ★doorbell 区起点 = BAR0+4KB
 │
 ├② nvme_pci_configure_admin_queue(dev)  [pci.c:1958]   ★建 0 号(admin)队列、给控制器上电★
 │    ├─ snvme_disable_ctrl(&dev->ctrl)                   [core.c, :1975]  先清 CC.EN 关机
 │    ├─ nvme_alloc_queue(dev, 0, NVME_AQ_DEPTH)          [pci.c:1979→1667]
 │    │     dma_alloc_coherent 出 admin 的 CQ 环 + SQ 环   [:1676]  ★内核自己的队列环(对比用户队列)
 │    │     nvmeq->q_db = &dev->dbs[0*2*db_stride]         [:1690]  ★admin 的 doorbell 地址
 │    ├─ 写 AQA / ASQ / ACQ 三个寄存器                     [:1989-1991]  把环的 DMA 地址告诉 SSD
 │    ├─ snvme_enable_ctrl(&dev->ctrl)                    [core.c, :1993]  拉高 CC.EN，等 CSTS.RDY
 │    ├─ nvme_init_queue(nvmeq, 0)                        [pci.c:1998→1717]  online_queues++
 │    └─ queue_request_irq(nvmeq)                         [:1999]  挂 admin 中断处理 nvme_irq
 │
 ├③ nvme_alloc_admin_tags(dev)                           [pci.c:3092]  建 admin_q 的 blk-mq tagset
 │
 ├④ change_ctrl_state(CONNECTING)                        [core.c, :3116]  状态机推进
 │
 ├⑤ snvme_init_ctrl_finish(&dev->ctrl)                   [core.c, :3129]  ★identify 阶段(见 4+.1)★
 │     发 Identify Controller、读 oacs/oncs/mdts、建 subsystem…全在 core.c
 │
 ├⑥ (可选) opal/dbbuf/host-mem 初始化                     [:3133-3155]
 │
 ├⑦ s_nvme_setup_io_queues(dev)  [pci.c:2512]   ★建"IO 队列"——内核的 + 给用户留的★
 │     ├─ snvme_set_queue_count(&dev->ctrl, ...)         [core.c]  和 SSD 协商能开几个 IO 队列
 │     ├─ pci_alloc_irq_vectors(...)                     重新按队列数要 MSI-X 向量
 │     ├─ nvme_create_io_queues(dev)      [pci.c:2040]   建**内核** IO 队列(adapter_alloc_cq/sq)
 │     └─ nvme_create_io_queues_mix(dev)  [pci.c:2009]   建**预留给用户**的 IO 队列(见 11.6)
 │
 ├⑧ if (online_queues < 2)  没建成任何 IO 队列 → kill_queues + remove_namespaces  [:3165-3169]
 │   else:  snvme_start_queues → wait_freeze → nvme_dev_add → unfreeze              [:3171-3174]
 │            nvme_dev_add(dev)  [pci.c:2787]  ★blk_mq_alloc_tag_set 建 IO tagset★
 │            —— 有了 tagset，namespace 才能挂上块设备队列，/dev/snvmeXnY 才可能出现
 │
 ├⑨ change_ctrl_state(LIVE)                              [core.c, :3181]  ★盘正式可用★
 │
 ├⑩ sysfs_create_group(...)                              [:3188]  挂 sysfs 属性
 │
 └⑪ snvme_start_ctrl(&dev->ctrl)                         [core.c, :3192]
       → 启动 keep-alive / AEN / 触发 scan_work
       → scan_work 跑 namespace 扫描(见 4+.4) → device_add_disk → /dev/snvmeXnY 出现
```

> **谁和谁配合（这一节的核心）**：
> - **pci.c 负责"硬件那一半"**：开 PCI、读寄存器、`dma_alloc_coherent` 建队列环、写 AQA/ASQ/ACQ、挂中断、建 blk-mq tagset。
> - **core.c 负责"协议那一半"**：disable/enable 控制器、identify、协商队列数、状态机推进、启动扫描。
> - 两者通过**两个方向**对接：pci.c **主动调** `snvme_*` 导出符号（disable/enable/init_ctrl_finish/set_queue_count/start_ctrl…）；core.c **回调** pci.c 提供的 `nvme_pci_ctrl_ops`（reg_read/write）去碰真实寄存器。这正是第 4.2 节"导出符号 = 让 snvme.ko 操控控制器"的运行时兑现。

---

### 11.5 顺手把"doorbell 是哪一步算出来的"钉死（串起内核队列与用户队列）

很多人对 doorbell 一直是模糊的，其实它在 11.4① 就被算死了，且**内核队列和用户队列用的是同一套公式**：

| 量 | 在哪算出 | 值 |
|----|----------|----|
| `dev->dbs`（doorbell 区起点） | `nvme_pci_enable` pci.c:2868 | `dev->bar + 4096`（BAR0 偏移 4KB 处） |
| `dev->db_stride`（步长） | `nvme_pci_enable` pci.c:2866 | `1 << CAP_STRIDE(cap)` |
| **内核**队列 q 的 SQ doorbell | `nvme_alloc_queue` pci.c:1690 | `&dev->dbs[qid*2*db_stride]` |
| **用户**队列 qid 的 SQ doorbell 偏移 | `NVM_ADD_USER_QUEUE` pci.c:5407 | `NVME_REG_DBS + qid*2*db_stride*4` |

- **内核自己按门铃**：`nvme_queue_rq`（pci.c:1030）下发 IO 时 → `nvme_submit_cmd`（pci.c:621）把 SQE memcpy 进 SQ 环 → `nvme_write_sq_db`（pci.c:598）执行 `writel(sq_tail, nvmeq->q_db)`（:611）——`q_db` 就是上表那个内核地址。
- **用户态按门铃**：`NVM_ADD_USER_QUEUE` 回传的 `sq_doorbell_offset`，加上 `mmap(BAR0)` 的基址，就是同一个寄存器——只是这次是用户进程自己 `writel`，不进内核。

> **一句话**：doorbell 区在 probe 的 `nvme_pci_enable` 一步就定位好了（`bar+4KB`），`db_stride` 也在那一步读出来。之后无论是内核 `nvme_write_sq_db` 还是用户态 `mmap+offset`，**算的是同一个寄存器、用的是同一条公式**——区别只是"谁来 `writel`"。

---

### 11.6 探测期 vs 探测后：两条"建用户队列"的路（都落到 map.c）

snvme 有**两种**把队列给用户态的时机，别混：

**路 A（探测期、legacy）**——`reset_work` 第⑦步里就建：
当 11.3② 读到的快照满足 `ctrl->use_sreg && ioq_num==ioq_map_num` 时，`nvme_create_io_queues_mix`（pci.c:2009）会在 probe 过程中顺便建好用户队列：

```
nvme_create_io_queues_mix [pci.c:2009]
  └─ for each → nvme_create_user_queue(dev, count, qid)  [pci.c:1805]
        ├─ list = queue_on_host ? &host_list : &device_queue_list      [:1815-1818]
        ├─ q_map = map_find_by_pci_dev_and_idx(list, pdev, uqid, is_cq) [map.c, :1823/1833]
        │     ★从 map.c 注册过的环里，按"第几个用户队列"找回那块 map★
        ├─ adapter_alloc_cq_user(dev, q_map, qid)  [pci.c:1829]  用 q_map->addrs[0] 当 PRP1 建 CQ
        └─ adapter_alloc_sq_user(dev, q_map, qid)  [pci.c:1841]  再建 SQ
```

**路 B（探测后、主流）**——probe 早已结束，盘 LIVE 了，用户态再走第 6 章那 8 步，靠 `NVM_ADD_USER_QUEUE`（pci.c:5156）按需建。**第一阶段冒烟测试走的就是路 B。**

> **共同点（配合 map.c 的本质）**：两条路最后都调 `adapter_alloc_cq_user/sq_user`，都把 **map.c 通过 `get_user_pages`+`dma_map_page` 得到的 `map->addrs[0]`** 当作 Create I/O CQ/SQ 命令的 PRP1，经 `snvme_submit_sync_cmd`（core.c）发给 SSD。区别只是"什么时候建、map 从哪张表找回来"。**map.c 始终是那座把'用户内存'翻译成'SSD 能 DMA 的物理地址'的桥**（见 5+.3）。

---

### 11.7 卸载/拔盘：把上面整条链严格逆序拆掉

`nvme_remove`（pci.c:3526，由 `UNBIND` 或 `rmmod` 触发）几乎是 11.4 的镜像倒放：

```
nvme_remove(pdev) [pci.c:3526]
  ├─ change_ctrl_state(DELETING)             [core.c, :3530]
  ├─ flush_work(reset_work)                  [:3538]  等任何在跑的 reset 收尾
  ├─ snvme_stop_ctrl(&dev->ctrl)             [core.c, :3539]  停 keep-alive/AEN
  ├─ snvme_remove_namespaces(&dev->ctrl)     [core.c, :3540]  删所有 /dev/snvmeXnY
  ├─ nvme_dev_disable(dev, true)             [pci.c:2939, :3541]  关 CC.EN、删队列、收中断
  ├─ nvme_dev_remove_admin + nvme_free_queues[:3544/3545]  拆 admin tagset、释放队列环
  ├─ nvme_release_prp_pools + nvme_dev_unmap [:3546/3547]  释放 PRP 池、解 BAR 映射
  └─ snvme_uninit_ctrl(&dev->ctrl)           [core.c, :3548]  删控制器对象
```

> 注意这里**只拆"硬件 + 控制器"这一半**。用户态那一半（队列组、pin 的页、QID）是由**进程退出时** `snvm_dev_release`（pci.c:5556）那本"账本"独立兜底的（见 5+.4）——两套清理路径互不依赖，所以进程崩了也不会让 `rmmod` 卡住。

---

### 11.8 一页总表：probe 时间线 → 函数(文件:行) → 配合谁

| 时刻 | 发生了什么 | 入口函数(文件:行) | 主要配合 |
|------|------------|-------------------|----------|
| insmod | 开控制面、建 4 张表、**不抢盘** | `nvme_init`(pci.c:6147) | list.c、控制面 cdev |
| 用户 CHRDEV_CREATE | 在 ctrl_list 埋记录 + 建 /dev/ssnvmeN | `snvm_chrdev_helper`(pci.c:5979)→`ctrl_get`(ctrl.c:13) | ctrl.c |
| 用户 DEVICE_BIND | 解绑 nvme、注册并 attach snvme | `snvm_rebind_driver`(pci.c:5855) | PCI 子系统 |
| **probe ①** | 准入闸：查 ctrl 记录，没有就放手 | `ctrl_find_by_pci_dev`(ctrl.c)@pci.c:3349 | ctrl.c |
| probe ②③ | 读用户队列预算、分配 dev | pci.c:3369-3418 | ctrl.c |
| probe ④ | 映射 BAR0（doorbell 区所在） | `nvme_dev_map`(pci.c:3252) | PCI 子系统 |
| probe ⑥ | 建控制器对象、传 ops 表 | `snvme_init_ctrl`(core.c)@pci.c:3460 | core.c |
| probe ⑦ | 把硬件带起来的活儿丢进 reset_work | `snvme_reset_ctrl`(core.c)@pci.c:3467 | core.c |
| reset① | 开 PCI、读 CAP、**算 dbs/db_stride/q_depth** | `nvme_pci_enable`(pci.c:2830) | PCI 子系统 |
| reset② | 建 admin 队列、给控制器上电 | `nvme_pci_configure_admin_queue`(pci.c:1958) | core.c(disable/enable) |
| reset⑤ | identify 阶段 | `snvme_init_ctrl_finish`(core.c)@pci.c:3129 | core.c |
| reset⑦ | 建内核 IO 队列 + 预留用户 IO 队列 | `s_nvme_setup_io_queues`(pci.c:2512) | core.c(set_queue_count)、map.c(路A) |
| reset⑧ | 建 IO tagset | `nvme_dev_add`(pci.c:2787) | blk-mq |
| reset⑨⑪ | 置 LIVE、启动扫描 | `snvme_start_ctrl`(core.c)@pci.c:3192 | core.c |
| LIVE 之后 | 扫出 namespace → /dev/snvmeXnY | `scan_work`→…(core.c, 见 4+.4) | core.c |
| **此后(路B)** | 用户态按需建队列、交 doorbell | `NVM_ADD_USER_QUEUE`(pci.c:5156) | map.c、core.c |
| UNBIND/rmmod | 逆序拆控制器与硬件 | `nvme_remove`(pci.c:3526) | core.c |
| 进程退出 | 独立兜底拆用户态资源 | `snvm_dev_release`(pci.c:5556) | map.c、ctrl.c |

### 11.9 本章一句话收口

`snvme.ko` 的前半段（pci.c 的 probe/reset/remove）是一条**严格有序的硬件带机线**：`nvme_init` 先只开门不抢盘 → 两条控制面 ioctl 先埋 `ctrl.c` 记录再注册驱动触发 `nvme_probe` → probe 只搭骨架并把活儿丢给 `nvme_reset_work` → reset_work 一步步**调 `core.c` 把控制器从 disable 推到 LIVE、自己用 `dma_alloc_coherent` 建内核队列环、用 `map.c` 的 DMA 地址建用户队列环、再建 blk-mq tagset** → LIVE 后 `core.c` 扫出 namespace。**pci.c 出"硬件与 BAR/中断/tagset"，core.c 出"协议与状态机与扫描"，map.c 出"用户内存的 DMA 地址"，ctrl.c 出"盘↔字符设备的身份与准入"，list.c 在底下把这些对象串起来**——这就是第 4+/5+ 章那些零散函数，按真实执行顺序拼成的一条完整流水线。

---

*（本文档由对 snvme-5.10-npu 全目录 + 原生 NVMe 驱动的逐文件分析归纳而成，可作为后续 NPU 迁移工作的入门索引。）*
