/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef NVFS_P2P_H
#define NVFS_P2P_H

#include "nv-p2p.h"

/*
 * [SNVME-NPU 迁移修改 batch6] 文件作用域前置声明 struct pci_dev
 * --------------------------------------------------------------
 * 现象（openEuler 5.10 上 make 报错；5.15 不报）：
 *   warning: 'struct pci_dev' declared inside parameter list will not be
 *            visible outside of this definition or declaration
 *   error: conflicting types for 'nvfs_nvidia_p2p_dma_unmap_pages'
 *   error: passing argument 1 ... incompatible pointer type  (struct pci_dev *)
 *
 * 根因：本头在下面的 typedef / 函数声明里**首次**用到 struct pci_dev，但此前
 *       没有任何地方声明过它（本头只 #include "nv-p2p.h" → linux/types.h，不含
 *       pci.h，也没有前置声明）。按 C 规则，"在函数原型的参数列表里第一次出现
 *       的 struct 标签"被限定在**该原型的作用域**内（function-prototype scope），
 *       成为一个与真正 linux/pci.h 里 struct pci_dev **不同**的"幽灵类型"。
 *       于是：
 *         - typedef 里的 struct pci_dev*  = 幽灵类型；
 *         - 本头里 nvfs_..._dma_unmap_pages 声明的形参 = 幽灵类型；
 *         - 而 nvfs-p2p.c 定义该函数时，pci_dev 已被 module.h/nvfs-core.h 补成
 *           真实类型 → 声明(幽灵) vs 定义(真实) → conflicting types；
 *         - 函数体里 nvidia_p2p_dma_unmap_pages_p(peer, ...) 调用，peer 是真实
 *           pci_dev*、而函数指针首参是幽灵 pci_dev* → incompatible pointer type。
 *
 *       为什么本机 5.15 一直不报：5.15 的 <linux/module.h>（在 nvfs-p2p.c 里先于
 *       本头被包含）会**间接**先声明出 struct pci_dev，于是本头的 typedef 命中的
 *       是真实类型、不产生幽灵；openEuler 5.10 的 module.h 包含链不同，到本头时
 *       struct pci_dev 尚未声明 → 触发幽灵作用域。属内核版本相关的"暴露差异"。
 *
 * 修法：在用到它之前，于**文件作用域**前置声明一次。这样 typedef、本头的函数
 *       声明、以及 .c 里的定义引用的都是同一个文件作用域 struct pci_dev 标签，
 *       之后 linux/pci.h 把它补成完整类型也兼容（这里只用到指针，前置声明足够）。
 *       纯编译期、与内核版本无关，不改任何运行逻辑。
 */
struct pci_dev;

typedef int (*nvidia_p2p_dma_unmap_pages_fptr) (struct pci_dev*,
		struct nvidia_p2p_page_table*,
		struct nvidia_p2p_dma_mapping*);
typedef int (*nvidia_p2p_get_pages_fptr) (uint64_t, uint32_t,
		uint64_t,
		uint64_t ,
		struct nvidia_p2p_page_table **,
		void (*free_callback)(void *data),
		void *);
typedef int (*nvidia_p2p_put_pages_fptr)(uint64_t, uint32_t,
		uint64_t,
		struct nvidia_p2p_page_table *);
typedef int (*nvidia_p2p_dma_map_pages_fptr)(struct pci_dev *,
		        struct nvidia_p2p_page_table *,
			struct nvidia_p2p_dma_mapping **);
typedef int (*nvidia_p2p_free_dma_mapping_fptr)(struct nvidia_p2p_dma_mapping *);
typedef int (*nvidia_p2p_free_page_table_fptr)(struct nvidia_p2p_page_table *);


int nvfs_nvidia_p2p_dma_unmap_pages(struct pci_dev *peer,
		struct nvidia_p2p_page_table *page_table,
		struct nvidia_p2p_dma_mapping *dma_mapping);
int nvfs_nvidia_p2p_get_pages(uint64_t p2p_token, uint32_t va_space,
		uint64_t virtual_address,
		uint64_t length,
		struct nvidia_p2p_page_table **page_table,
		void (*free_callback)(void *data),
		void *data);
int nvfs_nvidia_p2p_put_pages(uint64_t p2p_token, uint32_t va_space,
		uint64_t virtual_address,
		struct nvidia_p2p_page_table *page_table);
int nvfs_nvidia_p2p_dma_map_pages(struct pci_dev *peer,
		        struct nvidia_p2p_page_table *page_table,
			        struct nvidia_p2p_dma_mapping **dma_mapping);
int nvfs_nvidia_p2p_free_dma_mapping(struct nvidia_p2p_dma_mapping *dma_mapping);
int nvfs_nvidia_p2p_free_page_table(struct nvidia_p2p_page_table *page_table);

int nvfs_nvidia_p2p_init(void);
void nvfs_nvidia_p2p_exit(void);

#endif
