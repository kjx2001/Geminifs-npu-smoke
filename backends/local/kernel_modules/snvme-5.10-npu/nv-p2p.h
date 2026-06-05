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

#endif /* SNVME_NPU_STUB_NV_P2P_H */
