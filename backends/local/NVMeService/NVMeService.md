# NVMeService

## 概述

NVMeService 是一个本地守护进程。它负责：

1. **独占**地挂载并初始化 NVMe 控制器（libnvm 全量 admin 队列 + IO 队列创建）
2. 维护一个**共享队列池**：队列所在的 SQ / CQ 内存是 `cudaMalloc` 出来的，可以通过
   CUDA IPC 跨进程共享
3. 通过 gRPC 将队列**按连续区间**分配给客户端 GPU 进程（per-process 独占，互不冲突）
4. 基于心跳 + PID 存活 + `/proc/<pid>/stat` starttime 做租约回收，崩溃进程自动释放

设计前提：同一台机器内可信合作进程。不做加密鉴权。

客户端拿到队列后，可以直接把得到的 `std::shared_ptr<Controller>` 交给
`BlockDeviceManager`，和本地独占模式下使用 `Controller` 完全相同，**无感知共享**。

---

## 为什么需要 NVMeService

当前 libnvm 的 `Controller` 构造一次就独占一整个 NVMe 盘（admin 队列由单一进程管理、
IO 队列创建归一个进程所有）。但 NVMe 硬件支持 >100 个 IO 队列，而一个典型训练/推理
场景里同一块盘只会被少数几个 GPU 进程用。把队列在进程间分配，就能：

- 多个 GPU 进程**共用一块 NVMe**（不需要一对一绑定）
- 每个进程**独占自己的队列范围**（无跨进程 SQ/CQ 同步开销）
- 崩溃一个进程不影响其它进程（队列池隔离 + 自动回收）

同时保留原有独占模式（直接 `new Controller(pci_addr, ...)`）以供单进程使用。

---

## 跨进程共享的技术要点

### 三类内存的共享策略

| 类型 | 例子 | 策略 |
|------|------|------|
| `cudaMalloc` 普通 GPU 内存 | `d_qps` 数组、SQ ring、CQ ring、PRP pool | **CUDA IPC 共享**（`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`） |
| Host-mapped GPU 地址 | Doorbell 寄存器（BAR0 区通过 `cudaHostGetDevicePointer` 得到 GPU VA） | **每进程独立**。daemon 的 VA 在 client 里无效。client 自己 `mmap(/dev/snvm_*)` + `cudaHostRegister` + `cudaHostGetDevicePointer` 得到本地 GPU VA |
| 进程内 GPU 线程协调状态 | `sq_tickets`, `sq_tail_mark`, `sq_cid`, `cq_head_mark`, `cq_pos_locks` | **每 client 独立** `cudaMalloc`。跨进程不需要同步（每个队列只由一个进程拥有） |

关键观察：

- **CUDA IPC 是真正共享物理内存**。daemon 和 client 各自得到 GPU VA（地址不同），
  但读写的是同一块 device memory。lock-free atomic（`cuda::atomic`）语义保持。
- **Doorbell 不能 IPC**。`cudaHostGetDevicePointer` 返回的是 host-mapped 内存的
  GPU VA，对 host 的用户态 mmap 而言是进程私有。必须每个进程自己 mmap 一遍 BAR0。
- **SNVMe 允许多进程 mmap 同一个 BAR0**（`svm_mmap_registers` 无 exclusive-open），
  所以 client 自己开设备节点和 mmap 完全可行。硬件层面不同 doorbell 寄存器是
  posted write，互不干扰。

### Doorbell 重建公式

参见 `backends/local/nvme/libnvm/src/regs.h`：

```c
SQ_DBL(base, y, dstrd) = base + 0x1000 + (2*y)     * (4 << dstrd)
CQ_DBL(base, y, dstrd) = base + 0x1000 + (2*y + 1) * (4 << dstrd)
```

其中 `base` = 本进程 BAR0 mmap 的 GPU VA，`y` = 队列号（`qspec.queue_id`），
`dstrd` = 控制器 doorbell stride（来自 AllocResponse）。

---

## 进程间流程

```
┌─────────────────────────────────────────────────┐
│  NVMeService daemon (1 per host)                 │
│  ──────────────────────────────────────────────  │
│  sys_config.yaml → ServiceState                  │
│      ├── 每个 nvme: 全量 libnvm Controller       │
│      ├── 队列池 total_queues (配置)               │
│      ├── 预计算所有队列的 cudaIpcMemHandle        │
│      └── allocations_ map                        │
│                                                  │
│  grpc::Server listens                            │
│                                                  │
│  reaper thread (后台):                           │
│      每 heartbeat_interval/2 秒扫一次             │
│      超时 allocation → kill(pid, 0) + starttime  │
│      客户端已死 → release_range + erase          │
└─────────────────────────────────────────────────┘
                   │
                   │ gRPC
                   ▼
┌─────────────────────────────────────────────────┐
│  GPU 进程 (N 个，同 host)                          │
│  ──────────────────────────────────────────────  │
│  NvmeServiceClient client("127.0.0.1:50051");    │
│  auto alloc = client.allocate(device_id=0, 32);  │
│                                                  │
│  libnvm::build_shared_controller(spec):          │
│      1. open("/dev/snvm_xxx") + mmap(BAR0)       │
│      2. cudaHostRegister(IoMemory)               │
│         cudaHostGetDevicePointer → bar0_gpu_va   │
│      3. 每个队列:                                 │
│         - cudaIpcOpenMemHandle(SQ/CQ/PRP)        │
│         - db = SQ_DBL(bar0_gpu_va, qid, dstrd)   │
│         - cudaMalloc 本地 tickets/marks/cid      │
│         - new QueuePair(...)                     │
│      4. 组装完整 Controller (is_shared = true)   │
│                                                  │
│  BlockDeviceManager dm(alloc->controller, path); │
│  dm.device_file_create_managed(...);             │
│                                                  │
│  客户端内心跳线程自动维持租约。                    │
│  alloc 析构 → ReleaseQueues RPC + 关 IPC handle   │
└─────────────────────────────────────────────────┘
```

---

## 代码结构

### `src/nvmeservice.proto`
gRPC 接口定义。4 个方法：
- `ListDevices(Empty) → DeviceListResponse`
- `AllocateQueues(AllocRequest) → AllocResponse`
- `ReleaseQueues(ReleaseRequest) → ReleaseResponse`
- `Heartbeat(stream HeartbeatMsg) → stream HeartbeatMsg`

`AllocResponse` 携带：
- 设备信息（`pci_addr`、`snvme_dev_path`、`bar0_size`、`dstrd` 供 client 自己 mmap）
- 连续队列范围 `[queue_start_idx, queue_start_idx + queue_count)`
- 每队列 `QueueSharedMem`（64 字节 `cudaIpcMemHandle_t` for SQ/CQ/optional PRP + entries + ioaddrs）
- 控制器元数据（`namespace_id`、`page_size`、`blk_size`、`blk_size_log`、`queue_depth`）
- 租约参数（`heartbeat_interval_sec`、`lease_timeout_sec`）

### `src/nvmeservice_config.{h,cpp}`
`sys_config.yaml` 解析（依赖 yaml-cpp）。YAML schema：

```yaml
grpc:
  endpoint: "127.0.0.1:50051"

gpus:
  - id: 0
    mount_path: "/mnt/gpu0"     # GPU view dir; daemon symlinks NVMe subdirs here
  - id: 1
    mount_path: "/mnt/gpu1"

nvmes:
  - pci_addr: "0000:50:00.0"
    mount_path: "/mnt/nvme0"     # real NVMe block-device mount, unique per device
    namespace_id: 1
    queue_depth: 1024
    total_queues: 128            # 队列池大小
    # Required: per-GPU partition. Sum of count <= total_queues.
    # Each gpu_id must reference a gpus[].id above.
    queue_groups:
      - { gpu_id: 0, count: 64 }
      - { gpu_id: 1, count: 64 }

queue_pool:
  default_per_client: 32
  max_per_client: 128

lease:
  heartbeat_interval_sec: 10
  timeout_sec: 30
```

`validate_config` 做跨引用检查：`gpus[].id` 唯一且非负、`gpus[].mount_path`
非空、`nvmes[].pci_addr` 唯一、`nvmes[].mount_path` 唯一非空、`queue_groups`
非空且每条 `count > 0`、`gpu_id` 必须在 `gpus[]` 里、单 NVMe 内 `gpu_id`
不重复、`Σ count ≤ total_queues`。

**挂载与符号链接**：daemon 启动时 `Host_file_system_int` 把 NVMe mount 到
`nvme.mount_path`；随后对每个出现在该 NVMe 的 `queue_groups[].gpu_id`：

1. 在 NVMe 上 mkdir `<nvme.mount_path>/GPU<gpu_id>`（per-GPU 子目录隔离
   不同 GPU 在同一块盘上的数据）。
2. 在 GPU 视图 dir 里创建 symlink
   `<gpu.mount_path>/<basename(snvme_dev_path)>` → `<nvme.mount_path>/GPU<gpu_id>`。

退出（`~ServiceState`）时反序：`unlink` symlink → `rmdir` 子目录（best-effort）→
libnvm Controller dtor 走 umount。

### `src/nvmeservice_state.{h,cu}`
守护进程核心状态，**无 protobuf 依赖**。关键类型：

- `DeviceState` — 每盘一个：`shared_ptr<Controller>` + `queue_handles[total_queues]`
  （预计算的 IPC handles）+ `queue_allocated[total_queues]`（bit vector）
- `Allocation` — `allocation_id`、`device_id`、`queue_range`、`client_pid`、
  `client_pid_starttime`、`last_heartbeat`
- `ServiceState` — 内部互斥锁 + reaper 线程

**`init_queue_handles`**: 每个 queue 调 `cudaIpcGetMemHandle` for
`qp->sq_mem->vaddr`、`qp->cq_mem->vaddr`、`qp->prp_mem->vaddr`（如果有）。

**`allocate`**: first-fit 在 `queue_allocated` 里找连续空闲区间。生成 128-bit
hex allocation_id。记录 pid starttime。

**`reaper_loop`**: 每 `heartbeat_interval / 2` 扫一次。对超时 allocation:
- `kill(pid, 0)` == ESRCH → 死了，回收
- 存活但 starttime 变了 → PID 重用了，原进程已死，安全回收
- 存活且 starttime 一致 → 协议违约，**保守不回收**（记日志即可）

### `src/nvmeservice_server.{h,cpp}`
gRPC handler 薄壳，负责在 proto 和 `ServiceState` 之间翻译。
- `cudaIpcMemHandle_t` ↔ proto bytes：64 字节原始拷贝
- 业务错误用 `error_message` 字段返回，gRPC `Status` 保持 `OK`（避免混淆传输错和业务错）
- Heartbeat 双向 stream：`while(stream->Read)` 循环 + 回写 echo，发现 allocation
  被 reap 就发 `LEASE_REVOKED` 通知后关闭

### `src/nvmeservice_client.{h,cpp}`
客户端库。对外只暴露三个方法：
- `list_devices() → vector<ClientDeviceInfo>`
- `allocate(device_id, num_queues) → unique_ptr<Allocation>`
- `Allocation` 析构时自动 `ReleaseQueues` RPC + 清理本地资源

内部：
- `allocate()` 先调 `list_devices` 查出 `cuda_device`，再发 `AllocateQueues`
- 翻译 `AllocResponse` → `SharedControllerSpec`，调 `build_shared_controller()`
- 第一次成功 allocate 时启动心跳线程；最后一个 allocation 释放后线程退出
- 心跳线程每 tick 开一个新 bidi stream，发完所有 allocation 的心跳后关流
  （低频 10s 间隔下简单正确）

### `backends/local/nvme/libnvm/include/shared_ctrl.h`
libnvm 对 NVMeService client 的新公开 API。定义 `SharedQueueSpec`（单队列 IPC 描述）
和 `SharedControllerSpec`（整个 Controller 级描述），以及 `build_shared_controller()` 工厂函数。

### `backends/local/nvme/libnvm/src/shared_ctrl.cu`
`build_shared_controller()` 的实现，以及两个 shared-mode 构造函数。流程：

```
build_shared_controller(spec):
  cudaSetDevice(spec.cuda_device)

  res = SharedResources{}
  res.bar0_fd   = open(snvme_dev_path, O_RDWR)
  res.bar0_mmap = mmap(NULL, bar0_size, RW, SHARED, fd, 0)
  cudaHostRegister(res.bar0_mmap, bar0_size, cudaHostRegisterIoMemory)
  res.bar0_gpu_va = cudaHostGetDevicePointer(res.bar0_mmap)

  ctrl = new Controller(spec)              ← shared-mode 构造器，is_shared=true
  ctrl->h_qps = malloc(sizeof(QP*) * N)
  cudaMalloc(&ctrl->d_qps, sizeof(QP) * N)

  for i in [0, N):
      ctrl->h_qps[i] = new QueuePair(spec.queues[i], cuda_device, dstrd,
                                     nvmNamespace, page_size, blk_size,
                                     blk_size_log, res.bar0_gpu_va)
          ├── cudaIpcOpenMemHandle(sq_handle) → shared_sq_ptr
          ├── cudaIpcOpenMemHandle(cq_handle) → shared_cq_ptr
          ├── [if has_prp] cudaIpcOpenMemHandle(prp_handle) → shared_prp_ptr
          ├── nvm_queue_t sq/cq 元数据填充
          ├── sq.db = SQ_DBL(bar0_gpu_va, queue_id, dstrd)
          ├── cq.db = CQ_DBL(bar0_gpu_va, queue_id, dstrd)
          └── init_gpu_specific_struct(cudaDevice)  ← 本地 tickets/marks/cid
      cudaMemcpy(&ctrl->d_qps[i], ctrl->h_qps[i], sizeof(QP), H2D)

  ctrl->d_ctrl_buff = createBuffer(sizeof(Controller))
  cudaMemcpy(ctrl->d_ctrl_ptr, ctrl, sizeof(Controller), H2D)

  return shared_ptr<Controller>(ctrl, [res](Controller* p) {
      delete p;                           ← ~Controller → ~QueuePair → cudaIpcCloseMemHandle
      cudaHostUnregister(res.bar0_mmap)
      munmap(res.bar0_mmap, res.bar0_size)
      close(res.bar0_fd)
  })
```

清理顺序严格保证：IPC handle 先关（QueuePair dtor），再解除 BAR0 映射。

---

## libnvm 的修改面

对现有 `ctrl.h` 和 `queue.h` **纯增量修改**（不破坏原独占模式）：

### `include/ctrl.h`
- `struct Controller` 新增 `bool is_shared = false`
- 新增 `explicit Controller(const SharedControllerSpec&)` 构造函数声明
- 修改 `~Controller()`：`is_shared == true` 时跳过 `Host_file_system_exit`
  和 `nvm_ctrl_free`（daemon 拥有这些资源），只清理 `h_qps`/`d_qps`

### `include/queue.h`
- `struct QueuePair` 新增 `bool is_shared` + `shared_{sq,cq,prp}_ptr` 三个 void*
  用于记录 IPC 导入的 raw pointer（配合 dtor 做 `cudaIpcCloseMemHandle`）
- 新增 `QueuePair(const SharedQueueSpec&, ...)` 构造函数声明（定义在 shared_ctrl.cu）
- 新增显式 `~QueuePair()`：shared 模式下关 IPC handle；DmaPtr/BufferPtr 成员
  自己的 dtor 不受影响（shared 模式下 DmaPtr 字段为空 shared_ptr，dtor 是 no-op）

---

## 构建目标

- `libnvm` — 现有库扩展（新增 `shared_ctrl.cu`）
- `nvmeservice` 库（`nvmeservice_config.cpp` + `nvmeservice_state.cu` +
  `nvmeservice_server.cpp` + `nvmeservice_client.cpp` + 自动生成的
  `nvmeservice.pb.cc` / `nvmeservice.grpc.pb.cc`）
- `nvmeservice_daemon_example` 可执行文件
- `nvmeservice_client_example` 可执行文件

---

## 编译流程

### 环境前置要求

| 依赖 | 说明 |
|------|------|
| CUDA Toolkit | ≥ 12.6（根 `CMakeLists.txt` 的硬性要求） |
| gRPC + Protobuf | 已在根 CMake 被 `find_package` 检测。优先使用 `/usr/local` 下的版本，回退到系统包 |
| yaml-cpp | `find_package(yaml-cpp REQUIRED)` |
| libtorch | 位于 `third_pkgs/libtorch`（libgeminifs 需要，NVMeService 本身不依赖） |
| libunwind | `pkg_check_modules(UNWIND REQUIRED libunwind)` |
| NVIDIA driver 源码 | 为构建 snvme 内核模块（`nv-p2p.h`） |
| GPU Compute Capability | 默认 `sm_90`（`CMAKE_CUDA_ARCHITECTURES 90`），按实机修改 |

### 第一次配置 + 全量构建

```bash
cd /home/qs/CompanionFS/Geminifs

# 1. 配置
mkdir -p build
cd build
cmake ..

# 可选覆盖：
#   cmake -DCMAKE_BUILD_TYPE=Debug ..
#   cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..   # A100
#   cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..   # L40/4090

# 2. 全量构建（第一次推荐分层，便于定位问题）
make -j$(nproc)
```

### 分层构建（推荐用于调试）

从最底层往上，每层失败就停下修。二进制产物都在 `build/lib/`（库）或 `build/bin/`（可执行）。

```bash
cd build

# 1) libnvm：先扩展库（含新增 shared_ctrl.cu）
make -j libnvm

# 2) NVMeService 库
make -j nvmeservice

# 3) daemon + client 两个例子
make -j nvmeservice_daemon_example nvmeservice_client_example
```

构建产物：
```
build/lib/libnvm.so
build/lib/libnvmeservice.so
build/bin/nvmeservice_daemon       # 由 OUTPUT_NAME 决定
build/bin/nvmeservice_client
build/bin/sys_config.yaml          # examples/CMakeLists.txt 自动 copy
```

### 增量构建

常规代码修改后：

```bash
cd build && make -j
```

如果修改了 `nvmeservice.proto`：protoc 重跑由 `add_custom_command` 的
`DEPENDS` 触发，make 会自动重生成 `.pb.cc` / `.grpc.pb.cc` 再链接。

如果修改了 `ctrl.h` / `queue.h` / `shared_ctrl.h`：由于是 header，所有包含它的
`.cpp` / `.cu` 都会重编译。整体重编相对快（几十秒）。

### 加载 SNVMe 内核模块

第一次运行 daemon 前必须：

```bash
# 构建（根 CMakeLists 里的 `modules` target，make all 时已自动构建）
cd build
make modules

# 插入（需要 root）
make insmod     # 实际是 sudo insmod module/snvme.ko

# 验证
ls -l /dev/snvm_control           # 控制节点
ls -l /dev/snvm_*                  # 每块 NVMe 一个节点
lsmod | grep snvme

# 卸载
make rmmod     # 在 daemon 已停止的前提下
```

驱动正常工作时，`/dev/snvm_nvme0n1` 之类节点应存在。daemon 里 Controller 构造会
ioctl 这些节点。

### 常见构建问题

| 症状 | 原因 | 处理 |
|------|------|------|
| `Python.h: No such file` | libtorch 需要 Python 头（CUDA 头包含链带进来） | `sudo apt install python3-dev` |
| `cuda/atomic: No such file` | CUDA Toolkit 版本过旧 | 升级到 ≥ 12.6 |
| `nvmeservice.pb.h: No such file` | protoc 未运行 | 删 `build/` 重新 cmake（custom_command DEPENDS 可能断了） |
| `undefined reference to cudaIpcGetMemHandle` | 链接阶段缺 `CUDA::cudart` | 已在相关 target 里 `target_link_libraries(... CUDA::cudart)`，若仍报错检查是否自己加了新 target 没链接 cudart |
| `static_cast` / `reinterpret_cast` 从 `volatile void*` 出错 | nvcc 严格 | 参考 queue.h / shared_ctrl.cu 里现成的 cast 风格 |
| `fatal error: nv-p2p.h: No such file` | 内核模块构建路径找不到 NVIDIA 驱动源 | `cmake -DNVIDIA=/usr/src/nvidia-xxx ..` |
| `libnvm.so: cannot open shared object file` | 运行时找不到动态库 | 已配置 `RPATH=$ORIGIN`；若修改构建路径可能失效，`LD_LIBRARY_PATH=build/lib` 临时规避 |

---

## 测试流程

### 最小冒烟测试：daemon + client

**前置**：内核模块已 insmod，`/dev/snvm_*` 存在，`sys_config.yaml` 填好实机 PCI 地址。

#### 终端 1 — 启动 daemon

```bash
cd build/bin

# 改 sys_config.yaml 里的 nvmes[0].pci_addr 为你实机的地址，例如 0000:01:00.0
vim sys_config.yaml

./nvmeservice_daemon --config sys_config.yaml
```

预期输出：
```
NVMeService daemon listening on 127.0.0.1:50051 (port 50051)
Registered devices:
  device_id=0 pci=0000:01:00.0 snvme=/dev/snvm_nvme0n1 gpu=0 ns=1
  page=4096 blk=512 qdepth=1024 dstrd=0 bar0=16384 queues=128/128
lease: heartbeat=10s timeout=30s
queue_pool: default=32 max=128
```

如果卡在 `ServiceState init failed`：
- 看具体错误，通常是 `nvm_controller_init` 失败 → 内核模块、PCI 地址、权限问题
- 也可能是 `cudaIpcGetMemHandle failed` → 该 GPU 上 DmaPtr 的内存不是 cudaMalloc 的 → 需要排查 libnvm 里 `create_queue_Dma` 的分配路径

daemon 保持前台运行。`Ctrl+C` 触发 SIGINT → 优雅关闭。

#### 终端 2 — 测试 client

```bash
cd build/bin

# 只查询设备
./nvmeservice_client --list-only

# 申请 32 个队列，hold 15 秒（会触发 1~2 次心跳）
./nvmeservice_client --device 0 --count 32 --hold 15
```

预期输出：
```
=== Listing devices ===
  device_id=0 pci=0000:01:00.0 ... avail=128/128

=== Allocating 32 queues on device 0 ===
  allocation_id : <32 字节 hex>
  queue range   : [0, 32) count=32
  controller    : 0x7f...
  heartbeat     : 10s interval
  ...

=== Holding allocation for 15s (heartbeat thread running in background) ===
  5s elapsed
  10s elapsed
  15s elapsed

=== Releasing (via Allocation dtor) ===
Done.
```

daemon 侧应无异常日志。再跑 `--list-only` 应看到 `avail=128/128`（释放干净）。

### 租约回收测试（崩溃恢复）

模拟进程崩溃，验证 reaper 自动回收：

```bash
# 1) client 申请后 SIGKILL 自己（跳过 dtor 里的 ReleaseQueues RPC）
./nvmeservice_client --device 0 --count 16 --hold 60 &
CLIENT_PID=$!
sleep 3
./nvmeservice_client --list-only      # 应看到 avail=112/128
kill -9 $CLIENT_PID

# 2) 等超时时长（默认 30s）+ 心跳间隔
sleep 45

# 3) daemon 应已检测到 PID 死亡，回收了 16 个队列
./nvmeservice_client --list-only      # 应回到 avail=128/128
```

如果没回收成功，检查：
- daemon 日志有没有 reaper 相关输出
- 查配置 `lease.timeout_sec`
- 若 PID 被快速重用（极少见），daemon 用 `/proc/<pid>/stat` starttime 做二次校验，
  依然应能识别为死亡

### 并发多 client 测试

```bash
# 两个 client 同时申请不重叠的队列
./nvmeservice_client --device 0 --count 32 --hold 30 &
./nvmeservice_client --device 0 --count 32 --hold 30 &
wait

./nvmeservice_client --list-only      # 期间 avail=64/128，结束后 128/128
```

期间 daemon 侧不应有 error 日志；两个 client 都应正常完成。

### 端到端 IO 测试（需要 BlockDeviceManager 改造）

目前只能测 **控制面**（allocate / heartbeat / release）。要验证客户端真能通过
共享队列发 NVMe IO，还需要：
- `BlockDeviceManager` 的 shared-mode 构造函数（见 `Todolist.md`）
- 一个走 `build_shared_controller` 拿到 Controller 后发实际 IO 的例子

这是下一步工作，不在当前 rewrite 范围内。

### 调试技巧

- `cmake -DCMAKE_BUILD_TYPE=Debug ..` 重新配置，`DEBUG` 宏会打开 `geminifs_debug`
- daemon 加 `gdb --args ./nvmeservice_daemon --config sys_config.yaml`，崩溃时
  `thread apply all bt`
- `strace -f -e trace=openat,mmap,ioctl ./nvmeservice_daemon ...` 看 SNVMe
  设备访问序列
- `cuda-memcheck ./nvmeservice_client ...` 检查 GPU 内存越界
- gRPC 层面出错可加 `GRPC_VERBOSITY=DEBUG GRPC_TRACE=api ./nvmeservice_client ...`

---

## 使用方法

### 启动守护进程

```bash
nvmeservice_daemon_example --config /path/to/sys_config.yaml
```

守护进程会：
1. 解析 YAML，对每个 `nvmes[]` 项建完整 `Controller`（libnvm 标准独占路径）
2. 为每个队列的 SQ / CQ / PRP 内存调 `cudaIpcGetMemHandle` 预计算 handle
3. 启动 gRPC server 监听 `grpc.endpoint`
4. 启动 reaper 后台线程
5. 等待 SIGINT/SIGTERM 清理退出

### 客户端使用

```cpp
#include "nvmeservice_client.h"
#include "block_device_manager.cuh"

nvmeservice::NvmeServiceClient svc("127.0.0.1:50051");

// 查询设备
auto devs = svc.list_devices();
for (const auto& d : devs) {
    std::cout << "device " << d.device_id
              << " pci=" << d.pci_addr
              << " gpu=" << d.cuda_device
              << " avail=" << d.available_queues << "\n";
}

// 申请 32 个队列
auto alloc = svc.allocate(/*device_id=*/0, /*num_queues=*/32);
if (!alloc) { /* ... */ }

// alloc->controller 是 shared_ptr<Controller>，和标准 Controller 完全一致
BlockDeviceManager dm(alloc->controller, "/mnt/gpu0");
dm.device_file_create_managed(/* ... */);

// alloc 析构时自动 Release + 清理本地资源
```

### 心跳与崩溃恢复

- 心跳由 `NvmeServiceClient` 内部线程自动维护，无需手动管理
- 客户端进程崩溃：daemon reaper 在租约超时后检测 `kill(pid, 0) + starttime`，
  确认死亡后自动 `release_range`
- PID 重用场景：starttime 不匹配 → 视为死亡，安全回收
- 恶意进程不释放：**接受的限制**。daemon 会保留该 allocation 直到进程真的死，
  将来可以通过 SNVMe 内核模块增加 revoke ioctl 根治

---

## Docker

- 建议监听本机回环地址：`127.0.0.1:50051`
- 容器中使用时需保证容器可访问 `/dev/snvm_*` 和 `/dev/nvidia*` 设备
- CUDA IPC 跨容器：**同一个 cgroup 或 shared IPC namespace** 才工作

---

## 已知限制

- 不支持跨主机（CUDA IPC 只在同机同 GPU）
- 队列数上限受硬件（多数 NVMe ≤ 128 IO 队列）和配置 `total_queues` 双重限制
- daemon 重启后所有 allocation 作废（客户端的 IPC 导入会失效，需要重新 allocate）
- 恶意客户端不释放资源的场景不做激进处理（记日志，拒绝分配新资源）

---

## 相关代码路径

- 协议：[backends/local/NVMeService/src/nvmeservice.proto](src/nvmeservice.proto)
- 配置：[backends/local/NVMeService/src/nvmeservice_config.h](src/nvmeservice_config.h)
- 状态：[backends/local/NVMeService/src/nvmeservice_state.h](src/nvmeservice_state.h)
- Server：[backends/local/NVMeService/src/nvmeservice_server.h](src/nvmeservice_server.h)
- Client：[backends/local/NVMeService/src/nvmeservice_client.h](src/nvmeservice_client.h)
- libnvm shared 接口：[backends/local/nvme/libnvm/include/shared_ctrl.h](../nvme/libnvm/include/shared_ctrl.h)
- libnvm shared 实现：[backends/local/nvme/libnvm/src/shared_ctrl.cu](../nvme/libnvm/src/shared_ctrl.cu)
