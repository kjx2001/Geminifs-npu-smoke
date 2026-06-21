# snvme.ko 详细分析（按文件 → 函数 → 代码逐步拆解）

> 配套文档：`01snvme源码分析.md`（讲"是什么、为什么、整体架构"）。
> **本文档定位**：讲"**怎么做的**"——把 `snvme.ko` 四个核心文件里**每个关键函数内部的代码操作逐行拆开**，并**严格按推荐阅读顺序（6 个阶段）**排列，让你建立"插盘 → 接管 → 带硬件 → 建内核队列 → pin 用户内存 → 建用户队列"这条连续的、可落到代码行的心智模型。
>
> 行号全部对照当前 `snvme-5.10-npu/pci.c`、`map.c`、`ctrl.c`（注意：这份代码一直在改，行号可能随时漂；漂了就按函数名搜）。

---

## 0. 怎么用这份文档

- **不要从头到尾读源码**。按下面第 2 部分的 6 个阶段顺序读，每个阶段先看"这一阶段要回答什么问题"，再看每个函数的逐步拆解。
- 每个函数我都按统一模板写：**签名(文件:行) → 它解决什么问题 → 关键代码逐步 → 和谁配合**。
- 看到 `★` 的地方是"机制的关键一笔"，务必停下来理解。

---

## 1. 先建立"文件职责"地图（一句话 + 核心数据结构）

| 文件 | 一句话职责 | 核心数据结构 | 维护的全局对象 |
|------|------------|--------------|----------------|
| `pci.c` | 控制 SNVMe 如何**接管 NVMe PCI 设备**、初始化 controller、建 Admin/内核 IO/用户 IO 队列、把 BAR doorbell mmap 给用户态 | `struct nvme_dev`、`struct snvm_dev_owner`、`struct snvm_qgroup` | `ctrl_list` + 4 张 map 表（间接） |
| `map.c` | 把用户态 host/GPU buffer 或 SQ/CQ ring 的**虚拟地址 pin 住**、转成 NVMe 控制器可 DMA 的地址、维护 map 描述符供建队列/释放用 | `struct map`（尾部柔性数组 `addrs[]`）、`struct gpu_region` | `host_list` / `device_list` / `device_queue_list` |
| `ctrl.c` | 维护 **BDF → ctrl → /dev/ssnvmeN** 的映射关系，让 probe / open / ioctl 都能找到同一个 controller 对象 | `struct ctrl` | `ctrl_list` |
| `list.c` | 最底层**侵入式双向链表**原语，被 ctrl/map 复用 | `struct list` / `struct list_node` | —— |

**配合关系总纲**（记住这三句，后面所有代码都在兑现它）：
- `ctrl.c` 回答：**这个 BDF 对应哪个 `/dev/ssnvmeN`、哪个 controller 对象？**
- `map.c` 回答：**这个用户虚拟地址，对应哪些 NVMe 可 DMA 的物理地址？**
- `pci.c` 回答：**把这块盘绑到 SNVMe，并用这些 DMA 地址创建 NVMe 队列、把 doorbell 交出去。**

---

# 2. 按 6 个阶段逐函数详解（主体）

## 阶段一：控制入口——`/dev/snvm_control` 与 `/dev/ssnvmeN` 是怎么来的

> **本阶段要回答**：① 模块装载时发生了什么？② `/dev/snvm_control` 怎么来的？③ `/dev/ssnvmeN` 怎么来的？④ `ctrl_list` 什么时候插入 ctrl？⑤ 为什么 probe 前**必须先** `CHRDEV_CREATE`？

### 1.1 `nvme_init`（pci.c:6516）—— 模块装载，**只开门、不抢盘**

**解决什么问题**：`insmod snvme.ko` 的第一刻执行它。关键点是它**故意不注册 pci_driver**，所以装载模块本身不会抢任何一块盘。

**关键代码逐步**：
```c
static int __init nvme_init(void) {           // pci.c:6516
    snvm_registered = 0;                       // ★ 标记"还没注册 pci_driver"
    if (nvfs_nvidia_p2p_init())                // 找不到 NVIDIA p2p 符号
        printk(KERN_WARNING "... HOST-ONLY mode ...");  // ★[迁移#1] 只 warning，不 return，继续加载
    list_init(&ctrl_list);                     // 4 张全局表初始化（list.c）
    list_init(&host_list);
    list_init(&device_list);
    list_init(&device_queue_list);
    return snvm_cdev_init();                    // 建控制面字符设备 /dev/snvm_control
}
```
- **`snvm_registered=0`** 是整个"按需接管"设计的总开关：pci_driver 留到第一次 `DEVICE_BIND` 才注册（见阶段二 `register_driver`）。
- **HOST-ONLY 降级** 就在这一行：原版找不到 NVIDIA 符号会 `return -EOPNOTSUPP` 整个模块加载失败；迁移后改成只 warning 继续走。

**配合**：`list.c`（建表）+ `snvm_cdev_init`（建控制面 cdev）。**完全不碰任何盘。**

### 1.2 `snvm_cdev_init`（pci.c:6453）—— 造出 `/dev/snvm_control`

**关键代码逐步**：
```c
static int snvm_cdev_init(void) {                              // pci.c:6453
    mutex_init(&snvm_control_lock);
    dev_class = class_create(THIS_MODULE, DRIVER_NAME);        // /sys/class/snvme
    alloc_chrdev_region(&dev_first, 0, max_num_ctrls, DRIVER_NAME); // ★一次申请 max_num_ctrls(8) 个 minor
    cdev_init(&snvm_cdev, &snvm_fops);                         // 绑定控制面 fops
    cdev_add(&snvm_cdev, snvm_devno, 1);
    device = device_create(dev_class, NULL, snvm_devno, NULL, "snvm_control"); // /dev/snvm_control 出现
}
```
- **关键设计**：`alloc_chrdev_region(..., max_num_ctrls, ...)` 一次性申请了一个 **major + 一段 minor**。控制面 `/dev/snvm_control` 用其中一个号；后面每块盘的 `/dev/ssnvmeN` 复用同一个 major、用不同 minor（见 `ctrl_chrdev_create`）。
- `snvm_fops`（pci.c:6441）= `{ .unlocked_ioctl = snvm_ioctl }`，即控制面只认 ioctl。

**配合**：是 `nvme_init` 唯一会对外暴露的入口。

### 1.3 `snvm_ioctl`（pci.c:6408）—— 控制面总分派

**解决什么问题**：用户态对 `/dev/snvm_control` 发的所有控制面命令都从这里进，**载荷统一是 PCI 地址 `{domain,bus,slot,func}`**。

**关键代码逐步**：
```c
static long snvm_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {  // pci.c:6408
    struct pci_device_addr dev_addr;
    copy_from_user(&dev_addr, argp, sizeof(dev_addr));    // 载荷 = BDF
    switch (cmd) {
    case SNVM_DEVICE_BIND:   return snvm_rebind_driver(dev_addr);   // 阶段二：抢盘
    case SNVM_DEVICE_UNBIND: return snvm_unbind_driver(dev_addr);
    case SNVM_CHRDEV_CREATE:                                        // 本阶段：建 /dev/ssnvmeN
        ret = snvm_chrdev_helper(&dev_addr, 1);
        if (!ret) ret = copy_to_user(argp, &dev_addr, sizeof(dev_addr)); // 回传 minor(放在 .domain)
        return ret;
    case SNVM_CHRDEV_REMOVE: return snvm_chrdev_helper(&dev_addr, 0);
    }
}
```
**配合**：`snvm_chrdev_helper`（建/删字符设备）、`snvm_rebind_driver`（抢盘）。

### 1.4 `snvm_chrdev_helper`（pci.c:6332）→ `snvm_chrdev_create`（pci.c:6028）—— ★在 `ctrl_list` 埋记录★

这是回答"⑤为什么 probe 前必须先 CHRDEV_CREATE"的关键。

**`snvm_chrdev_helper` 逐步**（幂等设计）：
```c
ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev);     // 这个 BDF 之前建过吗？
if (create && !ctrl) {                              // 没建过 → 真正创建
    ret = snvm_chrdev_create(pdev, PCI_CLASS_STORAGE_EXPRESS);  // pci.c:6360
    ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev);  // 取回刚建的 ctrl
    dev_addr->domain = ctrl->number;                // ★用 .domain 字段回传 minor 给用户态
} else if (create && ctrl) {                        // 已建过 → 幂等，直接回传现有 minor
    dev_addr->domain = ctrl->number;
} else if (!create && ctrl) {                       // remove → 先 ctrl_put 再还 minor(防 race)
    int released_minor = ctrl->number;
    ctrl_put(ctrl);
    ida_simple_remove(&snvm_chrdev_minor_ida, released_minor);
}
```

**`snvm_chrdev_create` 逐步**：
```c
static int snvm_chrdev_create(struct pci_dev *pdev, unsigned int class) {  // pci.c:6028
    if (pdev->class != class) return -1;                        // 必须是 NVMe(PCI_CLASS_STORAGE_EXPRESS)
    minor = ida_simple_get(&snvm_chrdev_minor_ida, 0, 0, ...);  // 分配一个 minor
    ctrl  = ctrl_get(&ctrl_list, dev_class, pdev, minor);       // ★ctrl.c:12 建 ctrl 对象并挂 ctrl_list★
    err   = ctrl_chrdev_create(ctrl, dev_first, &snvm_dev_fops);// ★ctrl.c:111 造 /dev/ssnvmeN★
}
```

> **★为什么 probe 前必须先 CHRDEV_CREATE★**：`snvm_chrdev_create` 这一步把 ctrl **挂进了 `ctrl_list`**。而阶段二里 `nvme_probe` 的**第一行**就是 `ctrl_find_by_pci_dev(&ctrl_list, pdev)`——查不到这条记录就 `return -ENODEV` 放手。所以"先 CHRDEV_CREATE 埋记录"是"probe 愿意接管这块盘"的**准入凭证**。这道闸保证 snvme 只接管用户**显式点名**的盘，绝不误伤总线上其它 NVMe。

### 1.5 `ctrl_get`（ctrl.c:12）/ `ctrl_chrdev_create`（ctrl.c:111）—— ctrl.c 的两块基石

**`ctrl_get` 逐步**：
```c
struct ctrl *ctrl_get(struct list *list, struct class *cls, struct pci_dev *pdev, int number) { // ctrl.c:12
    ctrl = kmalloc(sizeof(struct ctrl), ...);
    list_node_init(&ctrl->list);              // 内嵌链表节点初始化
    ctrl->pdev = pdev; ctrl->number = number; // ★记住"这个 ctrl ↔ 这个 PCI 设备 ↔ 这个 minor"★
    ctrl->use_sreg = 0; ctrl->ioq_num = 0; ... // 清零所有用户队列预算字段
    memset(&ctrl->setup, 0, sizeof(ctrl->setup));   // B3 预算快照清零
    ctrl->user_qid_bitmap = NULL;             // B3 用户 QID 池(惰性建)
    mutex_init(&ctrl->user_qid_lock);
    snprintf(ctrl->name, ..., "ssnvme%d", number);  // 名字 = "ssnvme<minor>"
    list_insert(list, &ctrl->list);           // ★挂进 ctrl_list★
    return ctrl;
}
```
**`ctrl_chrdev_create` 逐步**：
```c
int ctrl_chrdev_create(struct ctrl *ctrl, dev_t first, const struct file_operations *fops) { // ctrl.c:111
    ctrl->rdev = MKDEV(MAJOR(first), ctrl->number);  // 复用控制面的 major + 本 ctrl 的 minor
    cdev_init(&ctrl->cdev, fops);                    // fops = snvm_dev_fops(数据面)
    cdev_add(&ctrl->cdev, ctrl->rdev, 1);
    chrdev = device_create(ctrl->cls, NULL, ctrl->rdev, NULL, ctrl->name); // /dev/ssnvme<N> 出现
    ctrl->chrdev = chrdev;
}
```
> **`ctrl` 是贯穿全程的"锚点"**：probe 靠它（`ctrl_find_by_pci_dev`）判断要不要接管；数据面 ioctl/mmap 靠它（`ctrl_find_by_inode`）从打开的 fd 反查回控制器。它内嵌的 `cdev` 地址 `&ctrl->cdev` 就是 `inode->i_cdev`，这是 `ctrl_find_by_inode` 能反查的根据。

**配合**：`list.c`（`list_insert`/`list_node_init`）。

---

## 阶段二：bind 到 probe——怎么把盘从内核 nvme 抢过来、精准绑 snvme

> **本阶段要回答**：① `SNVM_DEVICE_BIND` 怎么把原生 nvme 解绑？② 怎么**指定**绑 snvme（不误绑回 nvme）？③ 为什么 `nvme_probe` 开头要 `ctrl_find_by_pci_dev`？

### 2.1 `snvm_rebind_driver`（pci.c:6188）—— ★抢盘 + NPU 适配点★

**解决什么问题**：把目标 BDF 从当前驱动（通常是 in-tree `nvme`）解绑，再**精准**绑到 `snvme`。

**关键代码逐步**：
```c
static int snvm_rebind_driver(struct pci_device_addr dev_addr) {       // pci.c:6188
    pdev = TO_PCI_DEV(dev_addr);                  // BDF → struct pci_dev
    dev_drv = pdev->dev.driver;
    if (dev_drv && dev_drv->name) {               // 当前有驱动(stock nvme)
        if (pci_is_enabled(pdev)) pci_disable_device(pdev);
        device_release_driver(&pdev->dev);        // ★第一步：把 stock nvme 解绑★
    }
    if (register_driver())                         // ★第二步：注册 snvme 这个 pci_driver(只首次真注册)★
        return -EFAULT;
    dev_drv = pdev->dev.driver;
    if (!dev_drv) {                                // 解绑后暂时无驱动 → 主动绑 snvme
#if LINUX_VERSION_CODE < KERNEL_VERSION(5,15,0)   // ★[迁移 batch8c] openEuler 5.10 没导出 device_driver_attach★
        pdev->driver_override = kstrdup(PCI_DRIVER_NAME, GFP_KERNEL); // 钉死"只认 snvme"
        ret = device_attach(&pdev->dev);          // 触发匹配+probe；override 排除了 stock nvme
        ret = (ret > 0) ? 0 : (ret == 0 ? -ENODEV : ret);
#else
        ret = device_driver_attach(&snvme_driver.driver, &pdev->dev); // 5.15 直接指定
#endif
    }
}
```
> **★为什么不能用裸 `device_attach`★**：stock `nvme` 也匹配 `PCI_CLASS_STORAGE_EXPRESS`，裸 `device_attach` 可能挑中它绑回去。`driver_override="snvme"` 让 PCI 总线匹配（`pci_match_device`）**只认 name=="snvme" 的驱动**，从而精准绑 snvme。这是 5.10 内核没导出 `device_driver_attach` 时的等效替代，也是一个迁移点。

### 2.2 `register_driver`（pci.c:6130）→ `snvm_register_driver`（pci.c:6111）—— 延迟注册 pci_driver

```c
static int register_driver(void) {                        // pci.c:6130
    mutex_lock(&snvm_control_lock);
    dev_drv = driver_find(PCI_DRIVER_NAME, &pci_bus_type);
    if (!dev_drv && !snvm_registered) {                   // ★只在首次 BIND 时真正注册★
        ret = snvm_register_driver();                     // → pci_register_driver(&snvme_driver)
        if (!ret) snvm_registered = 1;
    }
    mutex_unlock(&snvm_control_lock);
}
```
- `snvme_driver`（pci.c:6096）= `{ .name="snvme", .id_table=nvme_id_table, .probe=nvme_probe, .remove=nvme_remove, ... }`。
- **`pci_register_driver` 的副作用**：它会对总线上**每一块**匹配的 NVMe 回调 `nvme_probe`——这正是为什么 `nvme_probe` 内部必须有准入闸（下一节），否则会把别的盘也抢了。

> **时间线**：`nvme_init` 装载时 `snvm_registered=0` 不注册 → 第一次 `DEVICE_BIND` 的 `register_driver` 才注册 → `pci_register_driver` 触发 `nvme_probe`。

### 2.3 `nvme_probe` 开头的准入闸（pci.c:3349）—— 回答"③为什么 probe 开头要 ctrl_find_by_pci_dev"

```c
static int nvme_probe(struct pci_dev *pdev, const struct pci_device_id *id) {  // pci.c:3320
    ctrl = ctrl_find_by_pci_dev(&ctrl_list, pdev);      // ★pci.c:3349★
    if (ctrl == NULL) {
        dev_info(&pdev->dev, "snvme: no ctrl registered ... skipping probe");
        return -ENODEV;     // ★没埋过记录 → 放手，PCI 核心去找下一个匹配驱动(通常是 in-tree nvme)★
    }
    ...
}
```
> 这就是阶段一 1.4 那条 `ctrl_list` 记录的用处：`pci_register_driver` 把 probe 撒给了总线上**每块** NVMe，但只有"用户用 `CHRDEV_CREATE` 显式点过名"的盘才有 ctrl 记录、才会被真正接管。这是 snvme "按需接管、绝不误伤"的第二道闸（第一道是 `driver_override`）。

**配合**：`ctrl_find_by_pci_dev`（ctrl.c:67，遍历 `ctrl_list` 比 `pdev`）。

---

## 阶段三：probe 到 reset——probe 只做资源准备，真正初始化在 reset_work

> **本阶段要回答**：① probe 具体准备了哪些资源？② 为什么真正的控制器初始化放在 `reset_work` 而不是 probe 里？

### 3.1 `nvme_probe`（pci.c:3320）—— 只"搭骨架"

**关键代码逐步**（接 2.3 的准入闸之后）：
```c
    // ① 读"用户队列预算"快照（从 ctrl 拷到 dev）—— 这些字段是阶段六老路径的 ioctl 早先写进 ctrl 的
    if (ctrl && ctrl->ioq_num == ctrl->ioq_map_num && ctrl->use_sreg) {   // pci.c:3369
        dev->use_user_allocated = 1;
        dev->nr_user_allocated_cq = ctrl->cq_num;
        dev->nr_user_allocated_sq = ctrl->ioq_num - ctrl->cq_num;
        dev->queue_on_host = ctrl->on_host;
    }
    if (ctrl && ctrl->setup.valid) { ... dev->cap_kernel_ioq = ctrl->setup.cap_kernel_ioq; }  // B3 快照

    // ② 分配 dev 与 queues[] 数组
    dev->nr_allocated_queues = nvme_max_io_queues(dev) + 1;              // pci.c:3411
    dev->queues = kcalloc_node(dev->nr_allocated_queues, sizeof(struct nvme_queue), ...);
    pci_set_drvdata(pdev, dev);                                         // pci.c:3418

    // ③ 映射 BAR0（doorbell 区就在里面）
    result = nvme_dev_map(dev);                                         // pci.c:3420 → 3252

    // ④ 把"真正带硬件"的活儿绑成工作项
    INIT_WORK(&dev->ctrl.reset_work, nvme_reset_work);                  // pci.c:3424 ★关键★
    nvme_setup_prp_pools(dev);

    // ⑤ 第一次进 core.c：建控制器对象（state=NEW、绑 5 个 work、建 /dev/snvmeX）
    result = snvme_init_ctrl(&dev->ctrl, &pdev->dev, &nvme_pci_ctrl_ops, quirks); // pci.c:3460

    // ⑥ 把硬件带起来的活儿丢出去（异步）
    snvme_reset_ctrl(&dev->ctrl);                                       // pci.c:3467 → queue_work(reset_work)
    async_schedule(nvme_async_probe, dev);                             // pci.c:3468 → 异步 flush 等它跑完
    return 0;     // ★probe 到此返回，硬件还没真正起来★
```

> **★为什么初始化放 reset_work 不放 probe★**：① 复用同一条"初始化路径"——盘出错自愈走 `reset` 时，要重跑的步骤和首次带机**完全一样**，写成一个 work 两边都能用；② probe 上下文有锁/时序限制，把耗时的 identify、建队列异步出去更稳。`snvme_reset_ctrl`（core.c）干的事就是 `change_ctrl_state(RESETTING)` + `queue_work(reset_work)`。

**`nvme_dev_map`（pci.c:3252）逐步**：
```c
static int nvme_dev_map(struct nvme_dev *dev) {       // pci.c:3252
    pci_request_mem_regions(pdev, "nvme");            // 占住 BAR 内存区
    nvme_remap_bar(dev, NVME_REG_DBS + 4096*3);       // ★ioremap BAR0 → dev->bar 可读写寄存器★
}
```
- 这一步之后 `dev->bar` 指向 BAR0 的内核虚拟映射；doorbell、CAP、CC、CSTS 等寄存器都在这块里。**注意它和阶段五给用户态的 mmap 是同一块物理 BAR0**，只是一个映进内核、一个映进用户态。

**配合**：ctrl.c（准入闸、读预算）、core.c（`snvme_init_ctrl`/`snvme_reset_ctrl`）、PCI 子系统（`nvme_dev_map`）。

---

## 阶段四：Admin Queue 和 kernel IO Queue——`nvme_reset_work` 真正带起硬件

> **本阶段要回答**：① Admin Queue 怎么建？② kernel IO Queue 怎么建？③ `queue_count` 怎么和 controller 协商？④ 内核用多少队列、给用户留多少？

### 4.1 `nvme_reset_work`（pci.c:3061）—— 总流程（11 步，已在 01 文档第 11.4 节给过全景，这里只钉关键调用）

```c
nvme_pci_enable(dev)                       // pci.c:3084 → 开 PCI、读 CAP、算 dbs/db_stride/q_depth
nvme_pci_configure_admin_queue(dev)        // pci.c:3088 → 建 admin 队列、给控制器上电
nvme_alloc_admin_tags(dev)                 // pci.c:3092 → admin_q 的 blk-mq tagset
change_ctrl_state(CONNECTING)              // pci.c:3116 (core.c)
snvme_init_ctrl_finish(&dev->ctrl)         // pci.c:3129 (core.c) → identify 阶段
s_nvme_setup_io_queues(dev)                // pci.c:3157 → ★建内核 IO 队列 + 预留用户 IO 队列★
nvme_dev_add(dev)                          // pci.c:3173 → blk_mq tagset（让 namespace 能挂块设备）
change_ctrl_state(LIVE)                    // pci.c:3181 (core.c)
snvme_start_ctrl(&dev->ctrl)               // pci.c:3192 (core.c) → keep-alive/AEN/触发扫描
```

### 4.2 `nvme_pci_enable`（pci.c:2830）—— ★算出 doorbell 基址★

```c
static int nvme_pci_enable(struct nvme_dev *dev) {        // pci.c:2830
    pci_enable_device_mem(pdev);                          // 开内存空间
    pci_set_master(pdev);                                 // ★允许设备做 DMA 主控（NVMe 要 DMA 读 SQ/写 CQ）★
    dma_set_mask_and_coherent(dev->dev, DMA_BIT_MASK(64));
    pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_ALL_TYPES); // 先要 1 个中断向量过渡
    dev->ctrl.cap   = lo_hi_readq(dev->bar + NVME_REG_CAP);     // 读控制器能力寄存器
    dev->q_depth    = min(NVME_CAP_MQES(cap)+1, io_queue_depth);// pci.c:2862 ★每队列深度
    dev->db_stride  = 1 << NVME_CAP_STRIDE(dev->ctrl.cap);     // pci.c:2866 ★doorbell 步长
    dev->dbs        = dev->bar + 4096;                         // pci.c:2868 ★doorbell 区 = BAR0 + 4KB
}
```
> 这三行（`q_depth`/`db_stride`/`dbs`）是**整个 doorbell 机制的源头**。内核队列的 doorbell 地址 `&dev->dbs[qid*2*db_stride]`（4.3）、用户队列回传的 `sq_doorbell_offset = NVME_REG_DBS + qid*2*db_stride*4`（阶段六）用的都是这里算出的 `db_stride`。`NVM_GET_DEV_INFO` 也把 `db_stride`/`q_depth`/`bar0_size` 回传给用户态（阶段五）。

### 4.3 `nvme_pci_configure_admin_queue`（pci.c:1958）—— 建 0 号队列 + 给控制器上电

```c
static int nvme_pci_configure_admin_queue(struct nvme_dev *dev) {   // pci.c:1958
    nvme_remap_bar(dev, db_bar_size(dev, 0));
    snvme_disable_ctrl(&dev->ctrl);                  // pci.c:1975 (core.c) ★先清 CC.EN 关机，从干净态开始★
    nvme_alloc_queue(dev, 0, NVME_AQ_DEPTH);         // pci.c:1979 → 建 admin 的 SQ/CQ 环
    nvmeq = &dev->queues[0];
    aqa = nvmeq->q_depth - 1; aqa |= aqa << 16;
    writel(aqa, dev->bar + NVME_REG_AQA);            // ★把 admin 队列深度写进 AQA 寄存器
    lo_hi_writeq(nvmeq->sq_dma_addr, dev->bar + NVME_REG_ASQ);  // ★把 admin SQ 环的 DMA 地址告诉 SSD
    lo_hi_writeq(nvmeq->cq_dma_addr, dev->bar + NVME_REG_ACQ);  // ★把 admin CQ 环的 DMA 地址告诉 SSD
    snvme_enable_ctrl(&dev->ctrl);                   // pci.c:1993 (core.c) ★拉高 CC.EN，等 CSTS.RDY=1★
    nvme_init_queue(nvmeq, 0);                       // pci.c:1998
    queue_request_irq(nvmeq);                        // pci.c:1999 → 挂 admin 中断 nvme_irq
}
```
> **理解要点**：admin 队列是"用来发其它命令的命令通道"（identify、create I/O queue 全走它）。它和内核 IO 队列一样，**环内存是内核自己用 `dma_alloc_coherent` 分配的**（见 4.4），地址通过 ASQ/ACQ 寄存器直接告诉 SSD——这和阶段六"用户队列把环地址通过 admin 命令 PRP1 告诉 SSD"是对照的两种方式。

### 4.4 `nvme_alloc_queue`（pci.c:1667）/ `nvme_init_queue`（pci.c:1717）—— 内核队列环的诞生

```c
static int nvme_alloc_queue(struct nvme_dev *dev, int qid, int depth) {  // pci.c:1667
    nvmeq->sqes = qid ? dev->io_sqes : NVME_ADM_SQES;
    nvmeq->cqes = dma_alloc_coherent(dev->dev, CQ_SIZE(nvmeq),
                                     &nvmeq->cq_dma_addr, GFP_KERNEL);    // ★内核分配 CQ 环 + 拿到 DMA 地址★
    nvme_alloc_sq_cmds(dev, nvmeq, qid);                                  // 分配 SQ 环(+sq_dma_addr)
    nvmeq->q_db = &dev->dbs[qid * 2 * dev->db_stride];                    // pci.c:1690 ★本队列 doorbell 地址★
    dev->ctrl.queue_count++;
}
static void nvme_init_queue(struct nvme_queue *nvmeq, u16 qid) {          // pci.c:1717
    nvmeq->sq_tail = 0; nvmeq->cq_head = 0; nvmeq->cq_phase = 1;
    nvmeq->q_db = &dev->dbs[qid * 2 * dev->db_stride];
    dev->online_queues++;                                                 // pci.c:1729 ★关键计数★
}
```
> **`dma_alloc_coherent` vs `dma_map_page`（呼应 map.c）**：内核队列用 `dma_alloc_coherent`——内核**自己分配**一致性 DMA 内存；用户队列用 map.c 的 `dma_map_page`——把**用户已有的**内存 pin 住再映射。两者最终都得到"SSD 能 DMA 的物理地址"，区别只是内存从哪来。
> **`online_queues`** 这个计数是阶段六用户 QID 池的起点（`user_qid_first = online_queues`），务必记住它在 `nvme_init_queue` 里 ++。

### 4.5 `s_nvme_setup_io_queues`（pci.c:2610）—— ③④ 协商队列数 + 内核/用户分配

**解决什么问题**：和 controller 协商"能开几个 IO 队列"，再决定内核用多少、给用户留多少。

```c
static int s_nvme_setup_io_queues(struct nvme_dev *dev) {       // pci.c:2610
    nr_io_queues = dev->nr_allocated_queues - 1;
    if (dev->use_user_allocated)                                // 老路径：把用户要的 cq 数也算进总数
        nr_io_queues += dev->nr_user_allocated_cq;             // pci.c:2639

    result = snvme_set_queue_count(&dev->ctrl, &nr_io_queues);  // pci.c:2642 ★发 Set Features(0x07) 和 SSD 协商★
    if (result == 0)
        dev->ctrl_max_io_queues = nr_io_queues;                // pci.c:2655 ★记录 controller 授予的上限★
                                                                //   —— 这是阶段六用户 QID 池的合法上界

    if (dev->use_user_allocated) {                              // 老路径：按授予值切分内核/用户
        dev->nr_allocated_queues = nr_io_queues - dev->nr_user_allocated_cq;
        nr_io_queues = dev->nr_allocated_queues - 1;
        dev->nr_user_use_cq = dev->nr_user_allocated_cq;       // 给用户留这么多
    }
    ...
    nvme_setup_irqs(dev, nr_io_queues);                         // 按内核队列数要 MSI-X 向量
    result = nvme_create_io_queues(dev);                       // pci.c:2684 → 建内核 IO 队列
    if (dev->use_user_allocated)
        result = nvme_create_io_queues_mix(dev);               // pci.c:2687 → 建预留给用户的 IO 队列(老路径)
}
```
> **★`ctrl_max_io_queues` 为什么重要★**：它是 `snvme_set_queue_count` 后 controller **真正授予**的 IOQ 上限。阶段六用户 QID 池的合法范围就是 `[online_queues, ctrl_max_io_queues]`。注释里特别强调：不能用 `nr_allocated_queues-1`，因为在 vCPU 数 > 控制器 MSI-X 授予数的机器上，那个值偏大，会把 controller 会拒绝的 QID 放进池子（Create I/O CQ 时报 SC=0x4101）。

### 4.6 `nvme_create_io_queues`（pci.c:2040）—— 建内核 IO 队列

```c
static int nvme_create_io_queues(struct nvme_dev *dev) {       // pci.c:2040
    for (i = dev->ctrl.queue_count; i <= dev->max_qid; i++)
        nvme_alloc_queue(dev, i, dev->q_depth);                // 先分配每个队列的环(4.4)
    for (i = dev->online_queues; i <= max; i++)
        nvme_create_queue(&dev->queues[i], i, polled);         // 再逐个让 SSD 建队列
}
```
- `nvme_create_queue`（pci.c:1755）内部：`adapter_alloc_cq`（pci.c:1292）先建 CQ、`adapter_alloc_sq`（pci.c:1315）再建 SQ、`nvme_init_queue` 上线、`queue_request_irq` 挂中断。
- **`adapter_alloc_cq` 的 PRP1**：`c.create_cq.prp1 = cpu_to_le64(nvmeq->cq_dma_addr)`（pci.c:1306）——填的是**内核** `dma_alloc_coherent` 的地址。对照阶段六 `adapter_alloc_cq_user` 填的是 `q_map->addrs[0]`（**用户** map.c 的地址），**这是内核队列和用户队列唯一的实质区别**。

**配合**：core.c（`snvme_set_queue_count`/`snvme_submit_sync_cmd`）。

---

## 阶段五：map.c——把用户/GPU 内存 pin 成 NVMe 可 DMA 的地址

> **本阶段要回答**：① host memory 路径怎么 pin（`get_user_pages`+`dma_map_page`）？② GPU memory 路径怎么 pin（`nvfs_nvidia_p2p_*`，当前桩）？③ queue ring memory 最后怎么被 `adapter_alloc_*_user` 当 SQ/CQ 用？

### 5.0 `struct map` 与 `create_descriptor`（map.c:27）—— 一切的载体

```c
struct map {                       // map.h
    struct list_node list;         // 挂全局表(host_list/...)
    struct task_struct* owner;     // = current，崩溃清理按它回收
    u64 vaddr;                     // 起始用户虚拟地址(已对齐)
    struct pci_dev* pdev;          // 针对哪块 NVMe 盘做的映射
    unsigned long page_size;       // PAGE_SIZE(4K) 或 GPU_PAGE_SIZE(64K)
    void* data;                    // host 路径=pages[]数组；GPU 路径=gpu_region
    release release;               // 回调：release_user_pages / release_gpu_*
    struct list_head group_link;   // 挂 per-fd 队列组 或 data_maps
    uint32_t group_id; uint8_t kind; int ioq_idx, is_cq;  // 归属/类型路由
    unsigned long n_addrs;         // 页数
    uint64_t addrs[1];             // ★尾部柔性数组：每页的 DMA 总线地址★
};
```
`create_descriptor`（map.c:27）用 `kvmalloc(sizeof(struct map) + (n_pages-1)*sizeof(uint64_t))` 一次性把 `addrs[]` 连同结构体分配出来，并 `INIT_LIST_HEAD(&map->group_link)`（让未挂组时 `list_del` 也安全）、`kind=UNSPECIFIED`、`owner=current`、`ioq_idx=-1`。

### 5.1 host 路径：`map_userspace`（map.c:280）→ `map_user_pages`（map.c:223）★最关键★

**`map_userspace` 逐步**：
```c
struct map *map_userspace(struct list *list, const struct ctrl *ctrl, u64 vaddr, unsigned long n_pages) { // map.c:280
    md = create_descriptor(ctrl, vaddr & PAGE_MASK, n_pages);   // 页对齐 + 分配描述符
    md->page_size = PAGE_SIZE;
    err = map_user_pages(md);                                   // ★两步魔法★
    if (err) { unmap_and_release(md); return ERR_PTR(err); }
    list_insert(list, &md->list);                               // 挂进 host_list
    return md;
}
```
**`map_user_pages` 逐步（"用户内存 → SSD 可 DMA 地址"的全部秘密）**：
```c
static long map_user_pages(struct map *map) {                   // map.c:223
    pages = kvcalloc(map->n_addrs, sizeof(struct page *), ...);
    // 第1步：把用户页 pin 在物理内存里、不让换出/迁移
    retval = get_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE, pages, NULL); // map.c:244 ★
    map->data    = pages;                  // 记下 page 数组，release 时 put_page 用
    map->release = release_user_pages;
    dev = &map->pdev->dev;                 // ★针对"这块 NVMe 盘"的 PCI 设备做映射★
    // 第2步：逐页 DMA 映射，过 IOMMU 拿到 SSD 能用的总线地址
    for (i = 0; i < map->n_addrs; ++i) {
        map->addrs[i] = dma_map_page(dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL); // map.c:266 ★
        dma_mapping_error(dev, map->addrs[i]);
    }
}
```
> **★两行就是核心★**：`get_user_pages` 锁页（防止内核把这些用户页换出/迁移，否则 SSD DMA 会踩到错的物理页）；`dma_map_page` 过 IOMMU 得到总线地址填进 `addrs[]`。`addrs[]` 身兼两职：① `copy_to_user` 回传给用户态（让用户知道队列环物理地址）；② 被 `adapter_alloc_*_user` 当 PRP1 填进 NVMe 命令（让 SSD 知道去哪 DMA）。

**反向 `release_user_pages`（map.c:199）**：逐页 `dma_unmap_page` + `put_page`，再 `kvfree(pages)`。由 `unmap_and_release`（map.c:75）通过 `map->release` 回调触发。

### 5.2 GPU 路径（当前是桩，NPU 上走不到）：`map_device_memory` / `map_gpu_memory`

**`map_gpu_memory`（map.c:440）逐步**：
```c
int map_gpu_memory(struct map *map, struct list *list) {        // map.c:440
    gd = kmalloc(sizeof(struct gpu_region), ...);               // GPU 区描述符
    gd->mappings = kmalloc(sizeof(...) * max_num_ctrls, ...);   // 每块 NVMe 一份 p2p 映射
    map->page_size = GPU_PAGE_SIZE;       // 64KB
    map->release   = release_gpu_memory;
    err = nvfs_nvidia_p2p_get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE*map->n_addrs,
                                    &gd->pages, force_release_gpu_memory, map);  // map.c:475 ★pin 显存
    // 对每块 NVMe 盘做 p2p DMA 映射
    while (element ...) {                  // 遍历 ctrl_list
        nvfs_nvidia_p2p_dma_map_pages(ctrl->pdev, gd->pages, gd->mappings + j);  // map.c:491
        if (j == 1)
            for (i...) map->addrs[i] = gd->mappings[0]->dma_addresses[i];        // map.c:507 ★取 p2p DMA 地址
    }
}
```
> **和 host 路径的对应关系**：`get_user_pages` ↔ `nvfs_nvidia_p2p_get_pages`（pin），`dma_map_page` ↔ `nvfs_nvidia_p2p_dma_map_pages`（映射），结果都落进 `map->addrs[]`。**NPU 迁移就是把这两个 nvfs 调用换成昇腾 HBM 的 pin/map 能力**（见第 3 部分）。在没有 NVIDIA 驱动的华为机器上，这些 nvfs 函数返回 `-ENOMEM`，走不到，所以第一阶段只用 host 路径。
> `map_gpu_ioqueue_memory`（map.c:528）是同一套，但只映射**当前这块** NVMe（队列环只服务于本盘），所以 `mappings` 只分配 1 个。

### 5.3 数据面入口 `NVM_MAP_HOST_MEMORY`（pci.c:4562）—— pin + 三路由 + 回传地址

map.c 不是用户直接调的，入口在 `snvm_dev_map_ioctl`（pci.c:4543，先 `ctrl_find_by_inode` 找回 ctrl）的这个 case：
```c
case NVM_MAP_HOST_MEMORY: {                                     // pci.c:4562
    copy_from_user(&request, arg, ...);                        // 校验 reserved/map_kind
    map = map_userspace(&host_list, ctrl, request.vaddr_start, request.n_pages); // pci.c:4608 → 5.1
    map->kind = request.map_kind;
    // ★三种归属路由(决定生命周期)：
    if (map_kind == DATA)        list_add_tail(&map->group_link, &own->data_maps); // pci.c:4635 fd 级
    else if (group_id != 0)      list_add_tail(&map->group_link, &g->maps);        // pci.c:4660 组级(RING)
    else if (ioq_idx >= 0)     { ctrl->ioq_map_num++; if(is_cq) ctrl->cq_num++; }  // pci.c:4664 legacy
    // ★回传每页 DMA 地址给用户态：
    copy_to_user(request.ioaddrs, map->addrs, map->n_addrs * sizeof(uint64_t));    // pci.c:4683
    // 若 copy_to_user 失败：回滚刚加的计数 + unmap_and_release
}
```
> **三路由对应三种生命周期**：DATA（挂 `own->data_maps`，fd 关闭才回收）/ RING_SQ、RING_CQ（挂 `g->maps`，队列组销毁时回收）/ legacy（只记 ctrl 计数，老路径用）。新主流路径建 SQ/CQ 环时用 `map_kind=RING_SQ/RING_CQ` + `group_id`，让环挂到队列组上。

---

## 阶段六：用户队列创建——把队列环交给 SSD、把 doorbell 交给用户态

> **本阶段要回答**：老路径和新路径分别怎么建用户队列？两者最后都落到哪？

### 6.0 两个 per-fd 运行态结构

- **`struct snvm_dev_owner`**（挂 `file->private_data`，崩溃清理总账本）：`groups`（队列组链表）、`data_maps`（DATA 映射链表）、`owner=current`。在 `snvm_dev_open`（pci.c:5833）里 `kzalloc` 并初始化。
- **`struct snvm_qgroup`**（一个队列组）：`group_id`（IDA 分配）、`max_queues`、`maps`（组内 ring map 链）、内联 `queues[]`（每个 `snvm_user_queue{qid,alive,sq_vaddr,cq_vaddr}`）、`cur_queues`。

### 6.1 老路径（探测期建）：`nvme_create_io_queues_mix`（pci.c:2009）→ `nvme_create_user_queue`（pci.c:1805）

触发条件：probe 时 `ctrl->use_sreg && ioq_num==ioq_map_num`（阶段三 ①读到的快照），于是 `s_nvme_setup_io_queues` 末尾顺带建。
```c
static int nvme_create_user_queue(struct nvme_dev *dev, int uqid, int qid) {  // pci.c:1805
    list = dev->queue_on_host ? &host_list : &device_queue_list;
    q_map = map_find_by_pci_dev_and_idx(list, pdev, uqid, 1);   // ★map.c:123 找回 CQ 环 map(is_cq=1)★
    adapter_alloc_cq_user(dev, q_map, qid);                     // 先建 CQ
    q_map = map_find_by_pci_dev_and_idx(list, pdev, uqid, 0);   // 找回 SQ 环 map
    adapter_alloc_sq_user(dev, q_map, qid);                     // 再建 SQ
    dev->online_user_queues++;
}
```
- **老路径靠 `ioq_idx`/`is_cq`** 这两个字段（在 `NVM_MAP_HOST_MEMORY` 的 legacy 路由里写进 map）来配对 SQ/CQ。配套 ioctl：`NVM_SET_IOQ_NUM`/`NVM_MAP_*_QUEUE_MEMORY`/`NVM_SET_SHARE_REG`，建队列在 probe 里发生。

### 6.2 新路径（探测后按需建，主流）：`NVM_ADD_USER_QUEUE`（pci.c:5438）★整个机制的高潮★

**完整逐步**：
```c
case NVM_ADD_USER_QUEUE: {                                      // pci.c:5438
    // (a) 校验：flags/reserved MBZ、1<=nr_pairs<=16、batch 内 vaddr 去重(0/相等也拒)  pci.c:5497-5536
    // (b) 确认设备真绑 snvme 且 admin_q live：
    uq_ndev = snvm_ctrl_get_live_ndev(ctrl);                   // pci.c:5546 → 见 6.3
    if (!uq_ndev) return -ENODEV;
    // (c) 找回队列组 + 校验配额：
    g = find_qgroup_locked(own, req->group_id);                // pci.c:5555
    if (g->cur_queues + req->nr_pairs > g->max_queues) return -EBUSY;
    // (d) 按 vaddr 在 g->maps 里解析出每对 SQ/CQ 的 struct map：
    for (i...) {
        list_for_each_entry(cursor, &g->maps, group_link) {    // pci.c:5592
            mask = ~((cursor->page_size ?: PAGE_SIZE) - 1);    // 用 map 自己的页大小算 mask(host 4K / GPU 64K)
            if (cursor->kind == NVM_MAP_KIND_DATA) continue;   // ★DATA 不许当队列环(防把数据缓冲当 SQ 用→静默损坏)
            if (cursor->vaddr==(sq_vaddr & mask) && kind∈{0,RING_SQ}) m_sq=cursor;  // pci.c:5619
            if (cursor->vaddr==(cq_vaddr & mask) && kind∈{0,RING_CQ}) m_cq=cursor;  // pci.c:5623
        }
        sq_maps[i]=m_sq; cq_maps[i]=m_cq;
    }
    // (e) 分配 QID（首次惰性建池）：
    snvm_user_qid_pool_init_locked(ctrl, uq_ndev);             // pci.c:5646 → 6.4
    snvm_user_qid_alloc_locked(ctrl, req->nr_pairs, qids);     // pci.c:5654 → 从位图找空位
    // (f) 驱动 SSD 建队列：NVMe 规范要求先 CQ 后 SQ
    for (i...) {
        adapter_alloc_cq_user(uq_ndev, cq_maps[i], qids[i]);   // pci.c:5673 ★用 cq_map->addrs[0] 当 PRP1
        adapter_alloc_sq_user(uq_ndev, sq_maps[i], qids[i]);   // pci.c:5680 ★用 sq_map->addrs[0] 当 PRP1
        created++;
    }
    // (g) 提交到组描述符 + ★回填 doorbell 偏移★：
    for (i...) {
        uq = &g->queues[g->cur_queues + i];
        uq->qid = qids[i]; uq->alive = 1; uq->sq_vaddr/cq_vaddr = ...;
        req->out_pairs[i].sq_doorbell_offset = NVME_REG_DBS + qid*2*db_stride*4;       // pci.c:5709 ★
        req->out_pairs[i].cq_doorbell_offset = NVME_REG_DBS + (qid*2+1)*db_stride*4;   // pci.c:5711 ★
        req->out_pairs[i].qid = qid;
    }
    g->cur_queues += req->nr_pairs;
    copy_to_user(arg, req, ...);                               // 回传 doorbell 偏移给用户态
    // 任一步失败 → rollback_unlocked：逆序 Delete 已建的 SQ/CQ + 还所有 QID (all-or-nothing)  pci.c:5754
}
```

**`adapter_alloc_cq_user`（pci.c:1345）/ `adapter_alloc_sq_user`（pci.c:1365）—— 闭环的关键一笔**：
```c
static int adapter_alloc_cq_user(struct nvme_dev *dev, struct map *q_map, int qid) { // pci.c:1345
    c.create_cq.opcode = nvme_admin_create_cq;
    c.create_cq.prp1   = cpu_to_le64(q_map->addrs[0]);  // ★用户队列环的 DMA 地址(map.c 得来的)当 PRP1★
    c.create_cq.cqid   = cpu_to_le16(qid);
    c.create_cq.qsize  = cpu_to_le16(dev->q_depth - 1);
    return snvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);  // 走 admin 队列发给 SSD
}
```
> **★此刻闭环★**：`q_map->addrs[0]`（map.c 在 5.1 用 `dma_map_page` 得到的总线地址）被填进 `Create I/O CQ/SQ` 命令的 PRP1，经 `snvme_submit_sync_cmd`（core.c）走 admin 队列发给 SSD。SSD 收到后，**就认得"用户那块内存是它的 SQ/CQ 环"**了，以后会 DMA 读 SQ、DMA 写 CQ。这正是 map.c → pci.c → core.c → SSD 的完整配合。

### 6.3 `snvm_ctrl_get_live_ndev`（pci.c:4509）—— 为什么不能直接 `pci_get_drvdata`

```c
static struct nvme_dev *snvm_ctrl_get_live_ndev(const struct ctrl *ctrl) {  // pci.c:4509
    drv = ctrl->pdev->dev.driver;
    if (!drv || strcmp(drv->name, PCI_DRIVER_NAME) != 0)   // ★先确认驱动名是 "snvme"★
        return NULL;
    ndev = pci_get_drvdata(ctrl->pdev);
    if (!ndev || !ndev->ctrl.admin_q) return NULL;          // 且 admin_q 已 live(probe 跑完)
    return ndev;
}
```
> **坑**：in-tree 的 `nvme` 驱动**也**把它的 `nvme_dev` 存在 `pci_get_drvdata` 里。若只用 `pci_get_drvdata + admin_q` 判活，对一块其实还归 stock nvme 管的盘也会"看起来 live"，于是 snvme 会对它乱发 admin 命令、和内核 nvme 抢资源。所以必须先查 `pdev->dev.driver` 名字 == "snvme"。

### 6.4 `snvm_user_qid_pool_init_locked`（pci.c:4201）/ `..._alloc_locked`（pci.c:4282）—— 用户 QID 池

```c
static int snvm_user_qid_pool_init_locked(struct ctrl *ctrl, struct nvme_dev *ndev) {  // pci.c:4201
    if (ctrl->user_qid_bitmap) return 0;            // 惰性：已建过直接返回
    if (!ndev->ctrl_max_io_queues) return -ENODEV;  // ★没协商过队列数 → 拒绝(不回退到坏估计)★
    first = ndev->online_queues;                    // pci.c:4246 ★池起点 = admin + 内核 IOQ 数
    last  = ndev->ctrl_max_io_queues;               // pci.c:4247 ★池上界 = controller 授予值(阶段四 4.5)
    count = last - first + 1;
    bm = kcalloc(BITS_TO_LONGS(count), ...);        // 位图，1=占用
    ctrl->user_qid_first = first; ctrl->user_qid_last = last; ctrl->user_qid_bitmap = bm;
}
static int snvm_user_qid_alloc_locked(struct ctrl *ctrl, unsigned int nr, uint16_t *qids_out) { // pci.c:4282
    for (i = 0; i < nr; i++) {
        bit = find_first_zero_bit(ctrl->user_qid_bitmap, pool_size);  // 找空位
        if (bit >= pool_size) { 回滚已设的位; return -EAGAIN; }       // 池满
        set_bit(bit, ctrl->user_qid_bitmap);
        qids_out[i] = ctrl->user_qid_first + bit;   // 位 → 真实 QID
    }
}
```
> **QID 空间布局**（钉死）：`0`=admin；`1..online_queues-1`=内核 IOQ；`online_queues..ctrl_max_io_queues`=用户 IOQ 池。`find_first_zero_bit` 在位图里找空位，加 `user_qid_first` 还原成真实 QID。`snvm_user_qid_free_locked`（pci.c:4317）反向 `test_and_clear_bit`。

### 6.5 mmap：`svm_mmap_registers`（pci.c:5814）—— 把 doorbell 寄存器交给用户态

```c
static int svm_mmap_registers(struct file *file, struct vm_area_struct *vma) {  // pci.c:5814
    ctrl = ctrl_find_by_inode(&ctrl_list, file->f_inode);
    if (vma->vm_end - vma->vm_start > pci_resource_len(ctrl->pdev, 0)) return -EINVAL; // 不超 BAR0
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);    // ★寄存器映射必须关 cache★
    return vm_iomap_memory(vma, pci_resource_start(ctrl->pdev, 0), len);  // 映射整个 BAR0 物理区
}
```
> 用户态拿到这个映射后，`mmap基址 + sq_doorbell_offset`（6.2 回传的）就是那个队列的 doorbell 寄存器地址，直接 `writel` 就能"按门铃"。**这块 BAR0 物理区和阶段三 `nvme_dev_map` 映进内核的是同一块**，所以内核（`nvme_write_sq_db`，pci.c:611）和用户态写的是同一组寄存器，只是各自有各自的虚拟映射。

### 6.6 崩溃清理：`snvm_dev_release`（pci.c:5861）/ `destroy_qgroup_locked`（pci.c:4383）

进程退出（哪怕崩溃）时，`release` 按**严格顺序**回收：
```
Pass 0   (pci.c:5894) 级联销毁所有 qgroup：destroy_qgroup_locked
Pass 0.5 (pci.c:5920) 释放所有 DATA maps(own->data_maps)
Pass 1   (pci.c:5945) 遍历 host_list/device_queue_list 统计 legacy map 要回滚的 ioq/cq 计数
Pass 2   (pci.c:5969) map_purge_by_owner 释放 legacy map(只 group_id==0) + 回滚 ctrl 计数(带下溢保护)
```
**`destroy_qgroup_locked` 三步**：
```c
ndev = snvm_ctrl_get_live_ndev(ctrl);          // pci.c:4406 (可能已 unbind → NULL，则跳过 admin 命令保持幂等)
for (每个 alive 的 uq)                          // Step1 排空用户队列
    adapter_delete_sq(ndev, uq->qid);          // pci.c:4423 NVMe 规范：先 Delete SQ
    adapter_delete_cq(ndev, uq->qid);          // pci.c:4428 再 Delete CQ
    snvm_user_qid_free_locked(ctrl, uq->qid);  // 还 QID
list_for_each_entry_safe(m, ..., &g->maps)     // Step2 排空 maps
    unmap_and_release(m);                      // pci.c:4460 释放 pin 页 / p2p 引用
ida_simple_remove(&snvm_queue_group_ida, g->group_id); kfree(g);  // Step3 还 group_id
```
> **★为什么 Pass 0 必须最先★**（pci.c:5886 注释）：队列组里停着用户 IO 队列和 SSD 正在 DMA 的环 map。若先把全局 map 表释放，`destroy_qgroup_locked` 发 Delete I/O SQ/CQ 时，SSD 可能还在 DMA 已被解映射的页 → use-after-free。**这两套清理路径（probe 侧的 `nvme_remove` vs 进程侧的 `snvm_dev_release`）互相独立**，所以进程崩了不会卡住 `rmmod`。

---

## 2.7 收口：IO 真正发生时怎么按门铃（把内核路径和用户路径对上）

内核自己发 IO（用户态走块设备时）：
```c
nvme_queue_rq(req) [pci.c:1065]
  → snvme_setup_cmd(ns, req)        // core.c：request → 64B NVMe 命令
  → nvme_submit_cmd(nvmeq, cmnd, ...) [pci.c:1109→621]
       memcpy(SQ环, cmd); sq_tail++;
       nvme_write_sq_db(nvmeq, ...)  [pci.c:598]
         → writel(sq_tail, nvmeq->q_db);   // ★q_db = &dev->dbs[qid*2*db_stride]（阶段四 4.4）★
```
用户态发 IO（走私有路径时，全程不进内核）：
```
自己把 64B SQE 写进自己的 SQ 环内存(SSD 通过 6.2 注册的 addrs[0] 能 DMA 读到)
→ writel(new_sq_tail, mmap基址 + sq_doorbell_offset)   // ★同一组寄存器，只是用户态自己写★
→ SSD DMA 取命令、执行、把 CQE 写进 CQ 环
→ 用户态轮询 CQ 环、写 cq_doorbell_offset
```
> **同一个 doorbell、同一条公式（`db_stride`），区别只是"谁来 `writel`"**——这是理解整个 snvme 的题眼。

---

# 3. NPU direct storage 迁移：最关键的替换点

> 对 NPU direct storage 来说，**最关键的迁移点不是普通 `probe()`，而是 map.c 的显存路径**。`probe`/Admin Queue/Set Queue Count/Create I/O SQ-CQ/doorbell mmap 这套**总体框架可以原样沿用**——它们只关心"一个 SSD 能 DMA 的物理地址"，不关心这地址来自主机内存还是 NPU HBM。

要替换的函数（全在 map.c + nvfs-*）：

| 当前（NVIDIA） | 作用 | 迁移成（昇腾 NPU HBM） |
|----------------|------|------------------------|
| `map_device_memory`(map.c:592) / `map_gpu_memory`(map.c:440) | pin 数据显存 + 映射成 NVMe 可 DMA | 改成调昇腾 HBM 的 pin/map |
| `map_device_ioqueue_memory`(map.c:624) / `map_gpu_ioqueue_memory`(map.c:528) | pin 队列环显存 | 同上 |
| `nvfs_nvidia_p2p_get_pages`(map.c:475/557) | pin 显存、取页表 | 昇腾 peer-DMA 的 pin 接口 |
| `nvfs_nvidia_p2p_dma_map_pages`(map.c:491/565) | 把显存页映射成某 NVMe 可访问的 P2P DMA 地址 | 昇腾 peer-DMA 的 map 接口 |
| `release_gpu_memory`(map.c:378) / `release_gpu_ioqueue_memory`(map.c:414) | 反向释放 | 对应昇腾的 unmap/put |

**为什么框架能复用**：只要昇腾 HBM 能 pin、能得到"目标 NVMe 盘可 DMA 的总线地址"并填进 `map->addrs[]`，那么 6.2 的 `adapter_alloc_cq_user/sq_user` 把 `addrs[0]` 当 PRP1 发给 SSD 的逻辑、doorbell mmap 的逻辑就**一字不用改**。迁移本质 = **换掉 `map->addrs[]` 的来源**。

**第一阶段现状**：`nvme_init` 进 HOST-ONLY 模式（1.1），GPU/NPU 显存路径的 nvfs 函数返回 `-ENOMEM` 走不到，只验证 host 内存通路（5.1）。详见 `01snvme源码分析.md` 第 8 章。

---

# 4. 最核心一句话总结

**`pci.c` 负责**：控制 SNVMe 如何接管 NVMe PCI 设备；初始化 NVMe controller；创建 Admin Queue 和 kernel IO Queue；根据用户态提供的 ring DMA 地址创建 user IO Queue；把 BAR doorbell mmap 给用户态。

**`map.c` 负责**：把用户态传入的 host/GPU buffer 或 SQ/CQ ring 虚拟地址 pin 住，转换成 NVMe 控制器可 DMA 的地址，并维护 map 描述符用于后续创建 queue 或释放资源。

**`ctrl.c` 负责**：维护 BDF → ctrl → /dev/ssnvmeN 的映射关系，让 pci.c 的 probe、open、ioctl 都能找到同一个 controller 对象。

完整配合：
- **`ctrl.c`**：这个 BDF 对应哪个 `/dev/ssnvmeN`、哪个 controller 对象？
- **`map.c`**：这个用户虚拟地址对应哪些 NVMe DMA 地址？
- **`pci.c`**：把这块盘绑定到 SNVMe，并用这些 DMA 地址创建 NVMe 队列、交出 doorbell。

---

# 5. 一页函数索引（按阶段，含当前行号）

| 阶段 | 函数 | 文件:行 | 一句话 |
|------|------|---------|--------|
| 一 | `nvme_init` | pci.c:6516 | 装载模块，只开门不抢盘 |
| 一 | `snvm_cdev_init` | pci.c:6453 | 造 /dev/snvm_control |
| 一 | `snvm_ioctl` | pci.c:6408 | 控制面总分派 |
| 一 | `snvm_chrdev_helper` | pci.c:6332 | 建/删 /dev/ssnvmeN(幂等) |
| 一 | `snvm_chrdev_create` | pci.c:6028 | ctrl_get + ctrl_chrdev_create |
| 一 | `ctrl_get` | ctrl.c:12 | 建 ctrl 对象挂 ctrl_list |
| 一 | `ctrl_chrdev_create` | ctrl.c:111 | cdev_add + device_create |
| 二 | `snvm_rebind_driver` | pci.c:6188 | 抢盘 + driver_override 精准绑 |
| 二 | `register_driver` | pci.c:6130 | 延迟注册 pci_driver |
| 二 | `nvme_probe` 准入闸 | pci.c:3349 | ctrl_find_by_pci_dev 没有就放手 |
| 三 | `nvme_probe` | pci.c:3320 | 只搭骨架，活儿丢 reset_work |
| 三 | `nvme_dev_map` | pci.c:3252 | ioremap BAR0 |
| 四 | `nvme_reset_work` | pci.c:3061 | 真正带硬件(11 步) |
| 四 | `nvme_pci_enable` | pci.c:2830 | 算 dbs/db_stride/q_depth |
| 四 | `nvme_pci_configure_admin_queue` | pci.c:1958 | 建 admin 队列+上电 |
| 四 | `nvme_alloc_queue` | pci.c:1667 | dma_alloc_coherent 建环 |
| 四 | `s_nvme_setup_io_queues` | pci.c:2610 | 协商队列数+内核/用户分配 |
| 四 | `nvme_create_io_queues` | pci.c:2040 | 建内核 IO 队列 |
| 四 | `adapter_alloc_cq/sq` | pci.c:1292/1315 | 内核队列：PRP1=cq_dma_addr |
| 五 | `map_userspace` | map.c:280 | host 路径入口 |
| 五 | `map_user_pages` | map.c:223 | get_user_pages+dma_map_page★ |
| 五 | `map_gpu_memory` | map.c:440 | GPU 路径(桩) |
| 五 | `NVM_MAP_HOST_MEMORY` | pci.c:4562 | pin+三路由+回传地址 |
| 六 | `nvme_create_user_queue` | pci.c:1805 | 老路径建用户队列 |
| 六 | `NVM_ADD_USER_QUEUE` | pci.c:5438 | 新路径★机制高潮★ |
| 六 | `adapter_alloc_cq/sq_user` | pci.c:1345/1365 | 用户队列：PRP1=addrs[0]★ |
| 六 | `snvm_ctrl_get_live_ndev` | pci.c:4509 | 确认真绑 snvme 才发命令 |
| 六 | `snvm_user_qid_pool_init_locked` | pci.c:4201 | 用户 QID 池[online,ctrl_max] |
| 六 | `svm_mmap_registers` | pci.c:5814 | mmap BAR0(non-cached) |
| 六 | `snvm_dev_release` | pci.c:5861 | 崩溃清理总账本(4 趟) |
| 六 | `destroy_qgroup_locked` | pci.c:4383 | 销毁队列组(Delete SQ/CQ) |

---

*（本文档逐函数对照 `snvme-5.10-npu/{pci,map,ctrl}.c` 当前源码写成，与 `01snvme源码分析.md` 互补：01 讲"是什么/为什么/架构"，02 讲"每个函数内部怎么做"。）*
