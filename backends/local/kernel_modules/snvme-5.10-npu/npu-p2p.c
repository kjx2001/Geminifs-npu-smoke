/*
 * npu-p2p.c  --  [SNVME-NPU 替换新增文件]  昇腾 NPU 显存 P2P 后端（dma-buf 实现）
 *
 * 作用
 * ----
 * 这是 nvfs-p2p.c 的「NPU 替身」。开启编译开关 SNVME_NPU 时，Makefile 用本文件
 * 的 npu-p2p.o 取代 nvfs-p2p.o 参与链接。本文件提供与 nvfs-p2p.c **完全相同的
 * 函数名与签名**（nvfs_nvidia_p2p_*），因此 map.c / pci.c 一行都不用改，只是底层
 * 实现从「NVIDIA 私有 nv-p2p API」换成「Linux 内核标准 dma-buf 导入 API」。
 *
 * 语义对应（详见 kjx/0620/02替换NPU代码分析.md 第三节映射表）
 * ------------------------------------------------------------
 *   nvidia_p2p_get_pages(vaddr,len,&pt,cb,data)  ->  dma_buf_get(fd)              （pin/取引用）
 *   nvidia_p2p_dma_map_pages(peer_pdev,pt,&m)    ->  dma_buf_attach + map_attachment（按 NVMe 设备映射）
 *   nvidia_p2p_dma_unmap_pages(peer,pt,m)        ->  dma_buf_unmap_attachment + detach
 *   nvidia_p2p_put_pages / free_page_table(pt)   ->  dma_buf_put
 *   sg_table 里的 sg_dma_address() 按 64KB 页展开 -> dma_mapping->dma_addresses[] -> map->addrs[]
 *
 * 两个必须在昇腾目标机确认的 TODO（本骨架可编译、可链接，遇不支持时安全报错不 oops）
 * ------------------------------------------------------------------------------------
 *   TODO-A: get_pages 需要一个 dma-buf fd。v1 临时约定「把 ioctl 传入的 vaddr 复用为
 *           dma-buf fd」（见下方 (int)virtual_address）。正式方案应在 struct
 *           nvm_ioctl_map 增一个 dmabuf_fd 字段，由昇腾用户态（aclrtMalloc 后导出
 *           显存 fd）传入。
 *   TODO-B: 需要昇腾内核驱动支持把 HBM 显存导出成 dma-buf。若不支持，dma_buf_attach/
 *           map_attachment 会返回错误码，本层把错误透传给 map.c，安全失败。
 *
 * 依赖：内核需开启 CONFIG_DMA_SHARED_BUFFER（dma_buf_* 符号来源）。这些符号在
 * openEuler 5.10 内核中均为标准 GPL 导出，无需任何华为私有头文件。
 */

#include <linux/module.h>
#include <linux/version.h>
#include <linux/slab.h>
#include <linux/err.h>
#include <linux/types.h>
#include <linux/pci.h>          /* struct pci_dev 完整定义（peer->dev） */
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/scatterlist.h>

#include "nvfs-p2p.h"   /* 复用同一套函数原型 + nvidia_p2p_* 结构体（SNVME_NPU 变体） */

/*
 * nvfs-pci.c 仍参与链接，并通过 nvfs-core.h 的 nvfs_dbg() 宏引用 nvfs_dbg_enabled。
 * 原本该符号定义在 nvfs-p2p.c；本文件取代了它，所以这里补一份定义，避免链接缺符号。
 */
int nvfs_dbg_enabled = 0;

/* 与 map.c 保持一致的「GPU/NPU 页」粒度（64KB）。 */
#define NPU_PAGE_SHIFT 16
#define NPU_PAGE_SIZE  (1UL << NPU_PAGE_SHIFT)

/*
 * 模块加载探测。NVIDIA 版在这里查 6 个 nvidia.ko 符号；dma-buf 是 per-mapping 行为，
 * 没有需要预先解析的全局符号，直接返回 0（成功）。pci.c nvme_init() 据此进入正常模式。
 */
int nvfs_nvidia_p2p_init(void)
{
	printk(KERN_INFO "snvme(npu): dma-buf P2P backend active (NVIDIA nv-p2p replaced)\n");
	return 0;
}

void nvfs_nvidia_p2p_exit(void)
{
}

/*
 * pin/取引用：按 dma-buf fd 拿到 struct dma_buf 引用。
 * length = NPU_PAGE_SIZE * n_pages（map.c 传入），据此换算页数 entries。
 */
int nvfs_nvidia_p2p_get_pages(uint64_t p2p_token, uint32_t va_space,
		uint64_t virtual_address,
		uint64_t length,
		struct nvidia_p2p_page_table **page_table,
		void (*free_callback)(void *data),
		void *data)
{
	struct nvidia_p2p_page_table *pt;
	struct dma_buf *dmabuf;
	int fd;

	(void)p2p_token;
	(void)va_space;
	(void)free_callback; /* TODO-B: 如需昇腾侧强制回收，改用动态 attach 的 move_notify */
	(void)data;

	if (page_table == NULL)
		return -EINVAL;

	/* TODO-A: v1 临时约定——把 vaddr 复用为 dma-buf fd。正式应走 ioctl 新增 dmabuf_fd 字段。 */
	fd = (int)virtual_address;

	pt = kzalloc(sizeof(*pt), GFP_KERNEL);
	if (pt == NULL)
		return -ENOMEM;

	dmabuf = dma_buf_get(fd);
	if (IS_ERR(dmabuf)) {
		printk(KERN_ERR "snvme(npu): dma_buf_get(fd=%d) failed: %ld\n",
		       fd, PTR_ERR(dmabuf));
		kfree(pt);
		return (int)PTR_ERR(dmabuf);
	}

	pt->dmabuf = dmabuf;
	pt->entries = (u32)(length >> NPU_PAGE_SHIFT);
	if (pt->entries == 0)
		pt->entries = 1;

	*page_table = pt;
	return 0;
}

/*
 * 按某块 NVMe（peer）做 DMA 映射：attach + map_attachment 得 sg_table，
 * 把每段的总线地址按 64KB 页展开，填进 dma_mapping->dma_addresses[]。
 */
int nvfs_nvidia_p2p_dma_map_pages(struct pci_dev *peer,
		struct nvidia_p2p_page_table *page_table,
		struct nvidia_p2p_dma_mapping **dma_mapping)
{
	struct nvidia_p2p_dma_mapping *m;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	struct scatterlist *sg;
	u64 *addrs;
	unsigned int i, idx = 0, n;
	int rc;

	if (peer == NULL || page_table == NULL || dma_mapping == NULL)
		return -EINVAL;
	if (page_table->dmabuf == NULL)
		return -EINVAL;

	n = page_table->entries;

	m = kzalloc(sizeof(*m), GFP_KERNEL);
	if (m == NULL)
		return -ENOMEM;

	addrs = kcalloc(n, sizeof(u64), GFP_KERNEL);
	if (addrs == NULL) {
		kfree(m);
		return -ENOMEM;
	}

	attach = dma_buf_attach(page_table->dmabuf, &peer->dev);
	if (IS_ERR(attach)) {
		rc = (int)PTR_ERR(attach);
		printk(KERN_ERR "snvme(npu): dma_buf_attach failed: %d\n", rc);
		goto err_free;
	}

	sgt = dma_buf_map_attachment(attach, DMA_BIDIRECTIONAL);
	if (IS_ERR(sgt)) {
		rc = (int)PTR_ERR(sgt);
		printk(KERN_ERR "snvme(npu): dma_buf_map_attachment failed: %d\n", rc);
		goto err_detach;
	}

	/* 把 dma 映射后的 scatterlist 段，按 64KB 页展开成逐页总线地址。 */
	for_each_sg(sgt->sgl, sg, sgt->nents, i) {
		dma_addr_t base = sg_dma_address(sg);
		unsigned int len = sg_dma_len(sg);
		unsigned int off;

		for (off = 0; off < len && idx < n; off += NPU_PAGE_SIZE)
			addrs[idx++] = (u64)base + off;

		if (idx >= n)
			break;
	}

	if (idx < n)
		printk(KERN_WARNING "snvme(npu): dma-buf only yielded %u of %u pages\n",
		       idx, n);

	m->dma_addresses = addrs;
	m->attach = attach;
	m->sgt = sgt;
	m->n_addrs = idx;
	*dma_mapping = m;
	return 0;

err_detach:
	dma_buf_detach(page_table->dmabuf, attach);
err_free:
	kfree(addrs);
	kfree(m);
	return rc;
}

/*
 * 解除某块 NVMe 的 DMA 映射，并释放 dma_mapping 容器本身。
 * （对应 NVIDIA 语义：nvidia_p2p_dma_unmap_pages 自身会释放 dma_mapping，
 *   map.c 据此只 kfree(mappings 指针数组)、不再单独 free_dma_mapping。）
 */
int nvfs_nvidia_p2p_dma_unmap_pages(struct pci_dev *peer,
		struct nvidia_p2p_page_table *page_table,
		struct nvidia_p2p_dma_mapping *dma_mapping)
{
	(void)peer;

	if (dma_mapping == NULL)
		return -EINVAL;

	if (dma_mapping->attach != NULL && dma_mapping->sgt != NULL)
		dma_buf_unmap_attachment(dma_mapping->attach, dma_mapping->sgt,
					 DMA_BIDIRECTIONAL);

	if (dma_mapping->attach != NULL &&
	    page_table != NULL && page_table->dmabuf != NULL)
		dma_buf_detach(page_table->dmabuf, dma_mapping->attach);

	kfree(dma_mapping->dma_addresses);
	kfree(dma_mapping);
	return 0;
}

/* 正常释放路径：unpin（dma_buf_put）并释放 page_table 容器。 */
int nvfs_nvidia_p2p_put_pages(uint64_t p2p_token, uint32_t va_space,
		uint64_t virtual_address,
		struct nvidia_p2p_page_table *page_table)
{
	(void)p2p_token;
	(void)va_space;
	(void)virtual_address;

	if (page_table == NULL)
		return -EINVAL;

	if (page_table->dmabuf != NULL)
		dma_buf_put(page_table->dmabuf);
	kfree(page_table);
	return 0;
}

/* 强制回收路径：与 put_pages 等价（与 put_pages 互斥调用，无双重释放）。 */
int nvfs_nvidia_p2p_free_page_table(struct nvidia_p2p_page_table *page_table)
{
	if (page_table == NULL)
		return -EINVAL;

	if (page_table->dmabuf != NULL)
		dma_buf_put(page_table->dmabuf);
	kfree(page_table);
	return 0;
}

/* 备用：单独释放 dma_mapping 容器（当前 map.c 各路径未调用，提供以保持 API 完整）。 */
int nvfs_nvidia_p2p_free_dma_mapping(struct nvidia_p2p_dma_mapping *dma_mapping)
{
	if (dma_mapping == NULL)
		return -EINVAL;

	kfree(dma_mapping->dma_addresses);
	kfree(dma_mapping);
	return 0;
}
