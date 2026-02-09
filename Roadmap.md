# Roadmap



## v2.0
### 现状
现在的GeminiFS::init按照下面的流程进行初始化：
1. 初始化链路
	- GeminiFS::init -> parse_and_setup_controllers -> geminifs_create_gpu_controller -> geminifs_add_nvme_to_gpu -> NVMeController::NVMeController
2. 关键资源在进程内创建
	- NVMeController 在构造中调用 open_single_controller，构造 Controller 并触发 Controller::init_queues
	- Controller::init_queues 内部创建 QueuePair，并为每个队列分配 GPU 侧 DMA 队列内存（create_queue_Dma）、GPU 侧队列元数据（cudaMalloc d_qps）及门铃映射等资源
	- NVMeController 同时为每个控制器分配 d_queue_acquire_helper（cudaMalloc + kernel init）
	- Controller 还会创建并持有 GPU 侧 d_ctrl_ptr（createBuffer + cudaMemcpy）
3. 结果与问题关联
	- 队列内存、d_ctrl_ptr、d_queue_acquire_helper 以及 doorbell 映射都在每个进程内独立创建
	- 进程退出后资源释放，导致多进程无法共享同一块 NVMe 盘的队列内存

### 问题
这样子把GPU SNVMe 初始化融合在Geminifs初始化过程中，SNVMe和GPU进程绑定。每次开一个进程都得重新初始化，多个进程没办法共享同一块NVMe盘。

### 解决方案
思路：通过一个GPU守护进程分配好所有的SNVMe队列内存，然后初始化SNVMe，然后通过Cuda获得GPU内存地址句柄。其他GPU进程通过IPC获得必要的SNVMe的指针，然后初始化Gemnifs。守护进程用于NMVe盘队列的抽象、分配、（NVMe）容错以及管理。

#### 详细解决思路（按“资源与控制权拆分”落地）
1. 把“控制器创建 + 队列内存分配 + GPU 侧指针生成”从 GeminiFS 初始化中剥离
	- 这些动作由常驻 GPU 守护进程完成并长期持有，避免每个进程重复 init
   - 守护进程启动时读取 sys_config.ini，一次性初始化多个 GPU 与 NVMe 控制器
2. 引入“可共享的控制器资源包”
	- 资源包包含：d_ctrl_ptr、d_qps、队列 DMA 内存、d_queue_acquire_helper 以及必要的门铃映射信息
	- 对 GPU 侧指针使用 CUDA IPC（cudaIpcGetMemHandle / cudaIpcOpenMemHandle）共享
	- 对 host 侧门铃/控制平面资源使用共享内存或 FD 传递（基于 UNIX socket）
3. NVMeController 支持两种模式
	- 独立模式（现有）：当前进程自建 Controller 与队列
	- 共享模式（新）：从 IPC 句柄 attach 既有资源，跳过 open_single_controller 与 init_queues
4. 让守护进程做“队列池管理”
	- 维护每块 NVMe 的队列池，支持按进程分配/回收队列
	- 可引入配额与故障隔离（如 max_queues_per_process）
5. GeminiFS 初始化改造
	- 优先尝试连接守护进程获取共享控制器资源
	- 连接失败时回退到现有的独立初始化逻辑（保持兼容）

### Roadmap（v2.0 详细步骤）
1. NVMeService 守护进程落地
	- 守护进程启动时读取 sys_config.yaml，完成多 GPU / 多 NVMe 初始化
	- 守护进程启动时完成 Controller 初始化与队列池预分配
2. 队列池分配与回收
	- 实现 FsAllocQueues / FsReleaseQueues
	- 增加 max_queues_per_process 等配额限制
3. IPC 句柄协议
	- 定义资源包传输结构（设备指针 IPC handle + 共享内存/FD）
	- 增加“获取控制器资源包”的 RPC
4. GeminiFS 资源抽象
	- 抽出 NVMeControllerResource（或类似结构体），封装 Controller + 队列资源句柄
	- 为 NVMeController 增加“从资源包构造/attach”的构造函数
5. GeminiFS 接入
	- parse_and_setup_controllers 中增加“通过守护进程 attach 控制器”的分支
	- 配置层面增加开关：enable_nvme_daemon / daemon_socket_path
6. 生命周期与容错
	- 进程退出时归还队列（显式 release 或心跳超时回收）
	- 守护进程异常重启时的清理与重新初始化策略
7. 验证与回归
	- 多进程并发读写、共享队列稳定性、崩溃恢复
	- 与现有单进程路径保持性能一致或可控损耗

### 记录
