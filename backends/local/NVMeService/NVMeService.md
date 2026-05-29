# NVMeService — Session Broker for SNVMe (L1 Commit 4b)

## 角色

NVMeService 是 SNVMe 上的 *会话代理 + chrdev owner*，**不**是配额账本，也不是队列宿主。

具体职责：

1. **chrdev / bind 的 owner**：每个配置的 NVMe 在启动时通过
   `nvm_controller_init_b3` 完成 `SNVM_CHRDEV_CREATE` +
   `NVM_SET_KERNEL_IOQ_CAP` + `SNVM_DEVICE_BIND` + `NVM_GET_DEV_INFO`，
   把 `/dev/ssnvme<N>` 拉起来后一直 hold 着。
2. **NUMA / PCIe-switch ACL**：通过 `nvmes[].allowed_gpus` 限制哪些
   `cuda_device` 可以 Connect 到这台 NVMe，避免跨 NUMA。
3. **元数据透传**：把 `snvme_dev_path` / `bar0_size` / `dstrd` /
   `ns_id` / `blk_size` / `queue_depth` 等内核 `NVM_GET_DEV_INFO`
   返回的字段，加上 daemon 装好的 GPU-view symlink，交给 client。
4. **per-client 策略**：把 `queue_pool.default_per_client` /
   `max_per_client` 作为 `granted_queues` 上限给到 client（client 可以
   不超过该值地调 `nvm_add_user_queue`）。这是 daemon 唯一对 queue
   计数施加的 *guidance*；真账本在内核。
5. **心跳 + 租约清理**：PID + `/proc/<pid>/stat` starttime；过期 →
   把 `Allocation` 记录从 daemon 内存表里清掉，仅此而已。Client 崩
   了的话 fd 自动 close → 内核 `snvm_dev_release` cascade 释放
   group / RING_* / DATA — daemon **不需要** refund 任何配额。

## Daemon 内部数据结构

```
ServiceState
├── DeviceState[N]                   // 每个 NVMe 一个
│   ├── nvm_ctrl_t* ctrl             // owner-side 句柄，daemon 生命周期内 hold
│   ├── snvme_dev_path / bar0_size / dstrd / namespace_id
│   ├── max_user_qid / max_queues_per_group / queue_depth   // 都来自 NVM_GET_DEV_INFO
│   ├── allowed_gpus: set<int>       // 来自 YAML（空时展开为所有 gpus[].id）
│   ├── gpu_view_paths: map<gpu_id, "/mnt/gpu0/ssnvme0">
│   └── created_symlinks/created_nvme_subdirs
└── unordered_map<allocation_id, Allocation>
    └── { device_id, cuda_device, granted_queues, client_pid, client_pid_starttime, last_heartbeat }
```

注意：**没有 `DeviceQuota`**。`granted_queues` 是 daemon 给 client 的
建议值，不入账，没有 refund 路径。

## 客户端做什么

收到 `Connect` 响应后由 **client 自己** 走完 libnvm 的 B3/B6 路径：

```cpp
nvmeservice::NvmeServiceClient client("127.0.0.1:50051");
auto sess = client.connect(device_id, cuda_device, num_queues);

cudaSetDevice(sess->cuda_device);

nvm_ctrl_t* ctrl = nullptr;
nvm_ctrl_attach_client(&ctrl,
                       sess->snvme_dev_path.c_str(),
                       (uint32_t)sess->bar0_size);

uint32_t group_id = 0, max_q = 0;
nvm_create_group(ctrl, &group_id, &max_q);

// cudaMalloc + nvm_dma_map_ring_device(SQ/CQ)
// cudaMalloc + nvm_dma_map_data_device(wbuf/rbuf)
// nvm_add_user_queue(...)  // 至多 sess->granted_queues 个

// drive IO ...

nvm_destroy_group(ctrl, group_id);   // RING_* 自动 cascade，DATA 仍存活
nvm_ctrl_free_client(ctrl);          // fd close → kernel 兜底回收
sess.reset();                        // → Disconnect RPC
```

约束：

- `sess->granted_queues` 是 daemon 政策上限，超过它会让 daemon 觉得
  client 在违约（没有强制；后续可以加）。**真正的硬上限**是内核的
  `NVM_MAX_QUEUES_PER_GROUP=16`。
- DATA 类型的 vaddr-map 在 client fd 上挂 `data_maps` 链表，跨
  `nvm_destroy_group` 存活；只在 `nvm_ctrl_free_client` (= fd close)
  时 cascade 释放（参见 PORTING.md §4.3.1 / §5.1）。
- Client 进程崩 → fd 自动 close → 内核 `snvm_dev_release` cascade
  释放 group + RING + DATA。Daemon 心跳超时后只是把 lease 记录擦掉。

## gRPC 接口

见 `nvmeservice.proto`。简化版：

| RPC          | 入参                                       | 关键出参 |
|--------------|--------------------------------------------|----------|
| ListDevices  | (Empty)                                    | 每 NVMe 元数据 + `allowed_gpus[]`（含 mount_path symlink） |
| Connect      | device_id / cuda_device / num_queues / pid | allocation_id / snvme_dev_path / bar0_size / granted_queues / 心跳参数 |
| Disconnect   | allocation_id / pid                        | success / error |
| Heartbeat    | bidi-stream (allocation_id, ts, notice)    | echo + 可选 LEASE_REVOKED 通知 |

## 进程崩溃下的清理

1. 客户端进程 SIGKILL → fd 自动 close。
2. 内核 `snvm_dev_release`：
   - 该 fd 上所有 queue group 走 `destroy_qgroup`（cascade
     `Delete I/O SQ/CQ` + RING_* maps）。
   - 该 fd 上所有 DATA maps 一起释放。
3. 内核 user QID 回到 pool，下个 client 拿到新 qid。
4. 几秒后 daemon 心跳超时 + PID dead → daemon 把内存里的
   `Allocation` 记录擦掉。**没有内核侧动作要做**——那部分内核已经
   自动完成。

## 为什么不需要 daemon 维护配额

- 内核 user QID pool（`[start_cq_idx, max_user_qid]`，典型 99 个）
  是真实配额账本。每次 `nvm_create_group` + `nvm_add_user_queue`
  从那里扣，destroy / fd-close 时还回去。
- B3 group 是 *fd-scoped*，跨 fd 看不见——daemon 没法预先创建好
  ring 让 client 用，所以"daemon 持有的 IO 资源"这个概念在新 ABI
  下根本不存在。
- 内核每个 group 的硬上限 `NVM_MAX_QUEUES_PER_GROUP=16` 已经把
  "单 client 一次性占太多"这个攻击面堵死了。daemon 再加一层 client
  policy 是策略层的最后一道闸（`max_per_client`），不需要也不应
  该做实际记账。

## YAML schema

参见 `sys_config.yaml`。最小版：

```yaml
grpc:
  endpoint: "127.0.0.1:50051"

gpus:
  - { id: 0, mount_path: "/mnt/gpu0" }

nvmes:
  - pci_addr: "0000:08:00.0"
    mount_path: "/mnt/nvme0"
    namespace_id: 1
    kernel_ioq_cap: 32        # OPTIONAL
    allowed_gpus: [0]         # OPTIONAL；缺省 = 所有 gpus[].id

queue_pool:
  default_per_client: 4
  max_per_client: 16

lease:
  heartbeat_interval_sec: 10
  timeout_sec: 30
```

不再有 `total_queues` / `queue_depth` / `queue_groups[].count` /
`queue_setup` 块——这些是 pre-B3 设计的产物，已废弃。
