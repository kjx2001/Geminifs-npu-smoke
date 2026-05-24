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
build/bin/sys_config.yaml          # examples/CMakeLists.txt 从仓库根 sys_config.yaml 自动 copy
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

### SNVMe 设备节点生命周期（先看这个，避免误判"卡住"）

| 节点 | 创建时机 | 销毁时机 |
|---|---|---|
| `/dev/snvm_control` | `insmod snvme.ko` 成功后由模块全局创建 | `rmmod snvme` |
| `/dev/ssnvme<N>` | 某进程对该 BDF 发 `SNVM_CHRDEV_CREATE` ioctl 时由 kernel `ida_alloc` 分配 minor 后创建 | 该进程发 `SNVM_CHRDEV_REMOVE` 或退出时 |
| `/dev/snvme<N>n1`（block） | `SNVM_DEVICE_BIND` 完成、snvme 从 in-tree nvme 抢绑、`nvme_alloc_ns` 跑完后 | `SNVM_DEVICE_UNBIND` 或模块卸载 |

**关键点**：刚 `insmod` 完只会看到 `/dev/snvm_control` 一个节点。`/dev/ssnvme*` 是 daemon 第一次走 `nvm_controller_init` 才会出现的，**不存在不代表配置有问题**——它就该等到 daemon 起来才有。

`daemon` 启动序列（`Controller(... &setup)` 内部）：

```
nvm_controller_init(snvme_control_path, pci_addr)
  ├─ open("/dev/snvm_control")              ← 必须已存在
  ├─ ioctl(SNVM_CHRDEV_CREATE, &pci_bdf)    ← 这一步创建 /dev/ssnvme<N>
  ├─ open("/dev/ssnvme<N>", O_RDWR)
  └─ mmap(BAR0) + cudaHostRegister(IoMemory)
nvm_queue_setup(ctrl, &setup)                ← B-2/B-3：NVM_SET_IOQ_NUM (groups[])
为每个 QueuePair 做 NVM_MAP_DEVICE_QUEUE_MEMORY
nvm_queue_share                              ← NVM_SET_SHARE_REG
SNVM_DEVICE_BIND                             ← 从 in-tree nvme.ko 抢绑到 snvme
NVM_GET_DEV_INFO
mount(<snvme block dev>, nvme.mount_path)
对每个 queue_groups[].gpu_id:
  mkdir <nvme.mount_path>/GPU<gpu_id>
  symlink <gpu.mount_path>/ssnvme<N> -> <nvme.mount_path>/GPU<gpu_id>
```

调试时按"启动到哪一步就有/还没有什么"反推卡点。

---

### 前置环境检查（每次新机或新 shell）

```bash
# 1. 编译器：必须 GCC >= 10（项目要求 CMAKE_CXX_STANDARD=20）
source /opt/rh/gcc-toolset-13/enable      # TencentOS 上的 toolset-13 是 13.1
gcc --version                              # 期望 13.x；8.x 会在 cmake LibTorch 探测时炸

# 2. SNVMe 内核模块已加载
lsmod | grep snvme                         # 期望看到 snvme + snvme_core
ls -la /dev/snvm_control                   # 必须存在（唯一全局节点）
# 不要在这一步检查 /dev/ssnvme*；它是 daemon 启动后才会出现

# 3. NUMA 拓扑（找一对 NUMA 0 上的 NVMe + GPU 做单卡 smoke）
sudo /data/home/ryeqiu/Geminifs/scripts/pci_topology_check.sh
# 看矩阵，挑 distance=0 (同 PCIe switch) 或 1 (同 NUMA) 的 (GPU, NVMe) 对
# 记下 NVMe BDF (如 0000:50:00.0) 和 GPU index (如 0)

# 4. 挂载点目录（一次性创建）
sudo mkdir -p /mnt/gpu0 /mnt/gpu1 /mnt/nvme0
sudo chown $USER:$USER /mnt/gpu0 /mnt/gpu1   # daemon 要在里面 symlink，需要写权限
# /mnt/nvme0 保持 root 即可，libnvm 用 mount(2)，不需要事先 chown
# 但 /mnt/nvme0 必须是空目录，否则 mount 失败：
mount | grep /mnt/nvme0                       # 有挂载？sudo umount /mnt/nvme0
ls -A /mnt/nvme0                              # 有残留？sudo rm -rf /mnt/nvme0/* 后重建
```

---

### 改 sys_config.yaml

仓库根 `/data/home/ryeqiu/Geminifs/sys_config.yaml` 是唯一来源（build 时由 examples/CMakeLists.txt 拷到 `build/bin/sys_config.yaml`）。两处 TODO 标记必改：

```yaml
nvmes:
  - pci_addr: "0000:50:00.0"        # ← 改成上面 0.3 找到的 NVMe BDF
    ...
    queue_groups:
      - { gpu_id: 0, count: 32 }    # ← gpu_id 改成上面 0.3 找到的 GPU index
```

---

### 编译

```bash
cd /data/home/ryeqiu/Geminifs
rm -rf build                       # 先清干净，避免老 cache
mkdir build && cd build

cmake ..
# 关键期望行：
#   -- CUDA 13+ detected; adding CCCL include dir: /usr/local/cuda-13.0/targets/x86_64-linux/include/cccl
#   -- Using snvme kernel baseline: 5.4.241-1-tlinux4-0017 (...)

# 分层 build，便于定位
make -j$(nproc) libnvm
make -j$(nproc) nvmeservice
make -j$(nproc) nvmeservice_daemon_example nvmeservice_client_example
```

产物：
```
build/lib/libnvm.so
build/lib/libnvmeservice.so
build/bin/nvmeservice_daemon
build/bin/nvmeservice_client
build/bin/sys_config.yaml      ← 从仓库根 yaml configure_file 拷过来的副本
```

注意：daemon 默认读 `build/bin/sys_config.yaml`（CMake copy 时的版本）。如果你想现场改配置且不重新 cmake，**直接改根 yaml 然后 `make` 触发 reconfigure**，或者直接编辑 `build/bin/sys_config.yaml`（这次绕过来源）。

---

### 终端 1：启动 daemon

```bash
cd /data/home/ryeqiu/Geminifs/build/bin

# 顺手开实时 dmesg 在另一窗口
# sudo dmesg -wH

sudo ./nvmeservice_daemon --config sys_config.yaml
```

**daemon 启动期 4 段关键输出**（按时序）：

**(A) YAML 解析摘要**（stdout，daemon main 打的）：
```
Parsed NVMe config: pci=0000:XX:00.0 mount=/mnt/nvme0 ns=1 qdepth=1024 total_queues=64 queue_groups=1 queue_setup={kernel_ioq_cap=32 on_host=false nr_write=0 nr_poll=0}
```
没看到 = YAML parse / validate 失败，前面 stderr 会有具体 emit() 消息。

**(B) ioctl_setup 翻译摘要**（stderr，`init_device` 在 Controller 构造前打）：
```
nvmeservice: device=0 pci=0000:XX:00.0 nvm_ioctl_setup{ioq_num=64 (SQ+CQ) cap_kernel_ioq=32 (pairs) nr_write=0 nr_poll=0 nr_groups=1}
nvmeservice:   group[0] owner_id=0 count=64 SQ+CQ (= 32 pairs)
```
单位换算回去：`32 pair × 2 = 64 SQ+CQ entries`。`Σgroups = ioq_num` 自洽。

**(C) kernel 侧 ack**（`dmesg -wH` 实时窗口）：
```
snvme: NVM_SET_IOQ_NUM: ioq_num=64 on_host=0 cap_kernel=32 groups=1
snvme: ... queue split: kernel=K user=M (controller granted ...)
```
**这就是 B 阶段的核心成功信号**。如果看到的是 `queue squeeze:`（不是 split:），说明 `kernel_ioq_cap` 还是太大、被控制器 MSI-X 挤掉了 user 份额——把 `kernel_ioq_cap` 调小到 16 或 8 重试。

**(D) daemon banner**（stdout）：
```
NVMeService daemon listening on 127.0.0.1:50051 (port 50051)
Registered devices:
  device_id=0 pci=0000:XX:00.0 snvme=/dev/ssnvme<N> gpu=0 ns=1 page=4096 blk=512 qdepth=1024 dstrd=0 bar0=16384 queues=32/32
      group: cuda_device=0 range=[0, 32) avail=32/32
lease: heartbeat=10s timeout=30s
queue_pool: default=16 max=32
```
`snvme=/dev/ssnvme<N>` 里的 `<N>` 是 kernel 分配的 minor（一般第一次是 0）。**只有这一刻起，`/dev/ssnvme<N>` 才存在**。

daemon 保持前台运行；`Ctrl+C` 走优雅退出（SIGINT → server->Shutdown → reaper 停 → 析构 → unmount + SNVM_CHRDEV_REMOVE）。

**如果 daemon 启动期就崩了，按这张表诊断：**

| 卡在 | dmesg / stderr 上能看到的 | 可能原因 | 检查 |
|---|---|---|---|
| (A) 之前 | `Config parse failed:` / `validation failed:` | YAML 语法或单位约束 | 看 emit() 消息 |
| (A) 之后 (B) 之前 | `validation failed: ... has no GPU entry` 或 `init_queues_multi_gpu: setup.nr_groups must be > 0` / `groups[].count must be a non-zero even number` | YAML 中 `queue_groups` 全是 `gpu_id < 0` 占位，或某行 count <= 0 | 检查 `nvme.queue_groups`，至少留一项 `gpu_id >= 0` 的 group |
| (B) 之后 (C) 之前 | `Failed to nvm_controller_init` | `nvm_controller_init` ioctl 失败 | `/dev/snvm_control` 权限？BDF 拼写？模块 `lsmod` 还在吗？ |
| dmesg `snvme: NVM_SET_IOQ_NUM: ... -EINVAL` | kernel 拒了 setup | `Σgroups[].count != ioq_num`，单位算错；或 `reserved` 字段非 0 | 看 (B) 段 stderr，对比 dmesg 里 kernel 报的 sum/ioq_num |
| dmesg `queue squeeze:` | kernel 谈判后 user 份额被挤光 | MSI-X 太少 + `kernel_ioq_cap` 太大 | 把 `kernel_ioq_cap` 调小 |
| dmesg `device's driver is '...nvme', not 'snvme'` | `SNVM_DEVICE_BIND` 抢不到设备 | in-tree `nvme.ko` 仍持有该 BDF；可能 udev 在 race | `lsblk` 看 BDF 当前归属；先 `sudo sh -c 'echo 0000:XX:00.0 > /sys/bus/pci/drivers/nvme/unbind'` 再启动 daemon |
| (D) banner 出来后 client 连不上 | `Failed to start gRPC server` | 端口被占 / 防火墙 / endpoint 拼错 | `ss -tlnp \| grep 50051` |

---

### 终端 2：跑 client（daemon 已 ready 后）

daemon banner 出现 `Registered devices` 一行**之后**再跑：

```bash
cd /data/home/ryeqiu/Geminifs/build/bin

# 4.1 先纯查询（不需要 sudo，但需要 CUDA 能用）
./nvmeservice_client --list-only

# 4.2 申请 16 个 pair，hold 15s
./nvmeservice_client --device 0 --cuda 0 --count 16 --hold 15
```

**client 预期输出**（B-4 加的 hand-off 验证段是关键）：

```
=== Listing devices ===
  device_id=0 pci=0000:XX:00.0 snvme=/dev/ssnvme<N> ns=1 page=4096 ... avail=32/32
      group: cuda_device=0 range=[0, 32) avail=32/32

=== Allocating 16 queues on device 0 (cuda_device=0) ===
  allocation_id : <32 hex chars>
  device_id     : 0
  queue range   : [0, 16) count=16
  controller    : 0x7fXXX
  mount_path    : /mnt/gpu0/ssnvme<N>
  heartbeat     : 10s interval
  lease timeout : 30s
  client_pid    : <pid>

=== Hand-off validation ===
  mount_path  : /mnt/gpu0/ssnvme<N> -> /mnt/nvme0/GPU0
  ls          : 0 entries reachable from this process
  queues      : n_qps=16 (probing first 4)
    qp[0] qp_id=0 is_shared=true sq_gpu=0x7fXXX cq_gpu=0x7fXXX prp_gpu=0x... sq.db=0x... cq.db=0x...
    qp[1] ...
    qp[2] ...
    qp[3] ...

=== Holding allocation for 15s (heartbeat thread running in background) ===
  5s elapsed
  10s elapsed
  15s elapsed

=== Releasing (via Allocation dtor) ===
Done.
```

**两个验收点**（你最初的目标）：

1. **GPU-view 工作目录权限**：`mount_path` 解析到 `/mnt/nvme0/GPU0`，`ls` 不报错（无内容是正常的，里面还没文件）
2. **GPU 队列地址**：`sq_gpu`、`cq_gpu`、`sq.db`、`cq.db` 都**不是 nullptr/0**——证明 CUDA IPC 导入和 BAR0 doorbell GPU VA 派生都成功

**client 失败诊断：**

| 症状 | 可能原因 | 检查 |
|---|---|---|
| `list_devices()` 返回空 | gRPC 通了但 daemon 没注册设备 | daemon 是否启动失败、`/dev/ssnvme<N>` 是否存在 |
| `cudaHostRegister(BAR0) failed` | BAR0 已被 daemon mmap 过、本进程再 register 失败 | 这是 Todolist 上 unchecked 那条；如果反复出现，看是否 same-host-different-process 的 cudaHostRegister 兼容性问题 |
| `cudaIpcOpenMemHandle failed` | CUDA IPC 跨进程不通 | 同一 IPC namespace？docker 容器要 `--ipc=host`；GPU MIG 模式不支持 IPC |
| client 退出后 daemon 没释放 | dtor 没跑（SIGKILL / abort）| 这是 reaper 测试，见下方"租约回收测试" |
| `sq_gpu` / `cq_gpu` 为 0 | IPC handle 导入失败但被吞 | 看 daemon stderr 是否打 `cudaIpcGetMemHandle failed`、shared_ctrl.cu 里有没有 silent fallback |

---

### 释放清洁性 + 反复测试

client 退出后回 daemon 看 reaper / release 是否干净：

```bash
./nvmeservice_client --list-only       # 应回到 avail=32/32 和初始一致
```

反复多跑几次 4.2 验证生命周期：

```bash
for i in 1 2 3; do
  ./nvmeservice_client --device 0 --cuda 0 --count 16 --hold 5
  ./nvmeservice_client --list-only
done
```

---

### 租约回收测试（崩溃恢复 reaper）

```bash
# 1) client 申请后 SIGKILL 自己，跳过 dtor 里的 ReleaseQueues RPC
./nvmeservice_client --device 0 --cuda 0 --count 8 --hold 60 &
CLIENT_PID=$!
sleep 3
./nvmeservice_client --list-only        # 应看到 avail=24/32
kill -9 $CLIENT_PID

# 2) 等超时（默认 lease.timeout_sec=30 + reaper tick=heartbeat/2=5）
sleep 45

# 3) daemon 应检测到 PID 死亡后 release_range
./nvmeservice_client --list-only        # 应回到 avail=32/32
```

如果没回收，看 daemon stderr 有没有 reaper 输出。`/proc/<pid>/stat` 读不到 = 进程死了，daemon `is_pid_dead` 应返回 true 触发回收。

---

### 并发多 client 测试（双 GPU 切换后再做）

当前 NUMA 0 单卡 profile 只有一个 group、cuda_device=0；并发测试需要切换到 yaml 末尾注释里的双 GPU profile。先把 NUMA 0 单卡跑通再说。

---

### 端到端 IO 测试（暂未启用）

只能测控制面（allocate / heartbeat / release / hand-off 探针）。要让 client 真正过共享队列发 NVMe 读写：

- BlockDeviceManager 的第二 ctor 已经存在（`device_manager/include/block_device_manager.cuh::BlockDeviceManager(const ControllerPtr&, std::unique_ptr<FileManager>, const std::string&)`），构造时跳过自己 init Controller
- 还需要一个 client examples 走 `alloc->controller` + `alloc->mount_path` 构造 BlockDeviceManager 并发 IO 的最小程序

这条在 Todolist "End-to-end smoke" 那条 unchecked 项里，下一轮做。

---

### 调试技巧

```bash
# Debug build
cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j

# daemon 崩溃栈
sudo gdb --args ./nvmeservice_daemon --config sys_config.yaml
#   (gdb) run
#   (gdb) thread apply all bt        # 崩溃时

# daemon ioctl 序列
sudo strace -f -e trace=openat,mmap,ioctl ./nvmeservice_daemon --config sys_config.yaml 2>&1 | head -300

# client GPU 内存检查（CUDA 12+ 用 compute-sanitizer 代替老的 cuda-memcheck）
sudo compute-sanitizer ./nvmeservice_client --device 0 --cuda 0 --count 16 --hold 5

# gRPC RPC 详情
GRPC_VERBOSITY=DEBUG GRPC_TRACE=api ./nvmeservice_client --device 0 --cuda 0 --count 16 --hold 5

# 实时 kernel 日志（强烈推荐 daemon 启动时开一窗）
sudo dmesg -wH
```

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
