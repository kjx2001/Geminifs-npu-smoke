/*
 * nv-p2p.h  --  [SNVME-NPU 迁移新增文件 #2]  host-only 桩头（stub）
 *
 * 原因
 * ----
 * 原始 snvme 的 nvfs-p2p.c / nvfs-p2p.h / map.c 都 #include "nv-p2p.h"，
 * 这本是 NVIDIA GPU 驱动（open-gpu-kernel-modules）提供的头，定义了
 * GPUDirect p2p 用到的类型。华为 NPU 服务器上没有 NVIDIA SDK，缺这个头
 * 会导致编译直接失败（fatal error: nv-p2p.h: No such file or directory）。
 *
 * 本次迁移第一阶段只验证「纯 CPU + host 内存」通路，根本不会用到 GPU/
 * device 内存路径。但代码在编译期仍然引用了下面两个类型，所以提供一份
 * 最小桩头让它「能编过」即可：
 *
 *   - struct nvidia_p2p_page_table   : map.c 访问了 ->entries（见 map.c 约
 *                                      537/602/607 行，device 路径里用来核对
 *                                      拿到的页数），所以需要 entries 字段。
 *                                      其余字段 snvme 不碰。
 *   - struct nvidia_p2p_dma_mapping  : map.c 在 device 路径里访问了
 *                                      ->dma_addresses[i]（见 map.c 约 527/
 *                                      597 行），所以这里必须给出 dma_addresses
 *                                      字段，否则编译报「没有该成员」。
 *                                      注意：HOST-ONLY 运行时根本走不到那段
 *                                      代码——nvfs_nvidia_p2p_get_pages() 在
 *                                      没有 NVIDIA 时先返回 -ENOMEM，函数提前
 *                                      退出。所以这个字段只为「编译通过」存在，
 *                                      运行时不会被解引用。
 *
 * 重要
 * ----
 * 这份桩头**只用于 host-only 构建**。将来如果要在真有 NVIDIA 的机器上跑
 * GPU 路径，必须换成 NVIDIA 官方的 nv-p2p.h（字段更完整）。等接入华为 NPU
 * 的 peer-DMA 后端时，这套类型会被华为对应的「页表 / DMA 映射」类型替换。
 */

#ifndef SNVME_NPU_STUB_NV_P2P_H
#define SNVME_NPU_STUB_NV_P2P_H

#include <linux/types.h>

/*
 * [SNVME-NPU 迁移修改 batch5] 与「真实 NVIDIA nv-p2p.h」共存防护
 * --------------------------------------------------------------
 * 现象（服务器上 make 报错）：
 *   error: redefinition of 'struct nvidia_p2p_page_table'
 *   error: conflicting types for 'nvidia_p2p_page_table_t' / 'nvidia_p2p_dma_mapping_t'
 *   →级联到 error: conflicting types for 'nvfs_nvidia_p2p_dma_unmap_pages'
 *     （它的形参引用了上面这两个被重复定义、彼此不兼容的结构体）
 *     以及 nvfs-p2p.c:147 那行调用的实参类型报错。
 *
 * 根因：本桩头的 include guard 是 SNVME_NPU_STUB_NV_P2P_H，而 NVIDIA 官方
 *       nv-p2p.h 的 guard 是 _NV_P2P_H_，两者**不同**。一旦机器上装了
 *       NVIDIA 驱动/GDS（其头位于 /usr/src/nvidia-<ver>/nvidia/nv-p2p.h 等），
 *       且它出现在内核模块的 -I 搜索路径里，两份头会**同时**被预处理器展开
 *       → 同名结构体 nvidia_p2p_page_table / nvidia_p2p_dma_mapping 被定义两次
 *       （字段还不一样：真头有 version/page_size/pages/gpu_uuid… 桩头只有 entries），
 *       于是 redefinition / conflicting types。本机 5.15 之所以不报，是因为
 *       `#include "nv-p2p.h"` 的引号查找优先命中本目录桩头、真头不在内核构建
 *       搜索路径里，纯属运气；换台装了 NVIDIA 头的机器（或路径不同）就会撞车。
 *
 * 修法（顺手 + 反向 双向防护，与 include 顺序无关）：
 *   下面把类型定义再套一层 `_NV_P2P_H_` 守卫并主动 #define 它。
 *   - 桩头先被包含：占用 _NV_P2P_H_ → 之后任何真头再被 #include 时整段被跳过，
 *     不会重复定义（host-only 只需要这两个结构体，真头其余声明用不到）。
 *   - 真头先被包含：_NV_P2P_H_ 已定义 → 跳过桩头的类型定义，直接用真头那份
 *     （真结构体同样有 entries / dma_addresses 字段，snvme 读取兼容）。
 * 两种顺序都只剩**一份**定义，彻底消除 redefinition / conflicting types。
 */
#ifndef _NV_P2P_H_
#define _NV_P2P_H_

#ifdef SNVME_NPU
/*
 * ============================ NPU（dma-buf）后端的结构体定义 ============================
 * [SNVME-NPU 替换 #2，见 kjx/0620/02替换NPU代码分析.md 第三/六节]
 *
 * 开启 SNVME_NPU 时，map.c 仍只读 page_table->entries 与 dma_mapping->dma_addresses[]，
 * 但这两个结构体额外承载 dma-buf 的状态，供 npu-p2p.c 的 wrapper 在 pin/map/unmap/unpin
 * 之间传递（语义对应见 02 文档映射表）：
 *   - nvidia_p2p_page_table  ←→ pin 阶段拿到的 dma_buf 引用（dma_buf_get 的产物）
 *   - nvidia_p2p_dma_mapping ←→ map 阶段对某块 NVMe 的 attach + sg_table
 *
 * 这里用前置声明而非 #include <linux/dma-buf.h>，避免本桩头被多处包含时拖入重头；
 * 真正用到完整类型的是 npu-p2p.c（它会包含 <linux/dma-buf.h>）。
 */
struct dma_buf;
struct dma_buf_attachment;
struct sg_table;

struct nvidia_p2p_page_table {
	u32 entries;            /* map.c 读取：GPU/NPU 页数（按 GPU_PAGE_SIZE 计） */
	struct dma_buf *dmabuf; /* dma_buf_get(fd) 的产物；put_pages/free_page_table 释放 */
};
typedef struct nvidia_p2p_page_table nvidia_p2p_page_table_t;

struct nvidia_p2p_dma_mapping {
	u64 *dma_addresses;                 /* map.c 读取：逐页 NVMe 可 DMA 总线地址 */
	struct dma_buf_attachment *attach;  /* dma_buf_attach(dmabuf, &nvme_pdev->dev) */
	struct sg_table *sgt;               /* dma_buf_map_attachment 的产物 */
	unsigned int n_addrs;               /* dma_addresses[] 的长度（释放时用） */
};
typedef struct nvidia_p2p_dma_mapping nvidia_p2p_dma_mapping_t;

#else  /* !SNVME_NPU —— 原 NVIDIA host-only 桩 */

/*
 * snvme 把它当句柄在 get_pages / put_pages / dma_map_pages 之间传递，
 * 并在 device 路径里读 ->entries（拿到的 GPU 页数）。真实 NVIDIA 结构体
 * 里还有 version/page_size/pages[] 等，host-only 用不到，省略。
 */
struct nvidia_p2p_page_table {
	u32 entries;
};
typedef struct nvidia_p2p_page_table nvidia_p2p_page_table_t;

/*
 * map.c 会读 mapping->dma_addresses[i]，所以这里必须给出该字段。
 * 真实 NVIDIA 结构体里还有 version/entries/page_size_type 等，host-only
 * 用不到，省略。dma_addresses 用 u64* 与 map.c 里 map->addrs[i]（uint64_t）
 * 的赋值兼容。
 */
struct nvidia_p2p_dma_mapping {
	u64 *dma_addresses;
};
typedef struct nvidia_p2p_dma_mapping nvidia_p2p_dma_mapping_t;

#endif /* SNVME_NPU */

#endif /* _NV_P2P_H_  —— 与真实 NVIDIA nv-p2p.h 共存防护，见顶部 batch5 说明 */
#endif /* SNVME_NPU_STUB_NV_P2P_H */
