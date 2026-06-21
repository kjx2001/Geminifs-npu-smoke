#include "map.h"
#include "list.h"
#include "ctrl.h"
#include <linux/version.h>
#include <linux/sched.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/slab.h>
#include <linux/mm_types.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/err.h>
#include "nvfs-p2p.h"

// **本阶段要回答**：① host memory 路径怎么 pin（`get_user_pages`+`dma_map_page`）？② GPU memory 路径怎么 pin（`nvfs_nvidia_p2p_*`，当前桩）？③ queue ring memory 最后怎么被 `adapter_alloc_*_user` 当 SQ/CQ 用？

struct gpu_region
{
    nvidia_p2p_page_table_t *pages;
    nvidia_p2p_dma_mapping_t **mappings;
};

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK ~(GPU_PAGE_SIZE - 1)

uint32_t max_num_ctrls = 8;

static struct map *create_descriptor(const struct ctrl *ctrl, u64 vaddr, unsigned long n_pages)
{
    unsigned long i;
    struct map *map = NULL;

    map = kvmalloc(sizeof(struct map) + (n_pages - 1) * sizeof(uint64_t), GFP_KERNEL);
    if (map == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return ERR_PTR(-ENOMEM);
    }

    list_node_init(&map->list);
    /*
     * Initialise per-fd queue-group link as an empty self-loop so
     * that list_empty(&map->group_link) is true and list_del()
     * remains safe even when the map is never attached to any
     * group (legacy mode, group_id == 0).
     */
    INIT_LIST_HEAD(&map->group_link);
    map->group_id = 0;
    /*
     * Default the B6 map-kind tag to UNSPECIFIED.  Pre-B6 callers
     * (and every internal helper that builds a map without
     * touching map_kind) keep their existing semantics this way.
     * NVM_MAP_* dispatch in pci.c overrides ->kind based on the
     * caller-supplied request.map_kind.
     */
    map->kind = 0; /* NVM_MAP_KIND_UNSPECIFIED */
    memset(map->reserved_pad, 0, sizeof(map->reserved_pad));

    map->owner = current;
    map->vaddr = vaddr;
    map->pdev = ctrl->pdev;
    map->page_size = 0;
    map->data = NULL;
    map->release = NULL;
    map->n_addrs = n_pages;
    map->ioq_idx = -1;
    map->is_cq = -1;
    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = 0;
    }

    return map;
}

void unmap_and_release(struct map *map)
{
    list_remove(&map->list);

    /*
     * If this map is attached to a per-fd queue group (group_id !=
     * 0, group_link non-empty), splice it out of the group's
     * maps list.  Done unconditionally via list_del because
     * group_link was INIT_LIST_HEAD'd in create_descriptor: a
     * never-attached map's list_del is a no-op (next/prev point
     * at itself, list_del rewires them and that's that).  The
     * caller is expected to be holding own->groups_lock if the
     * map is on a group list -- pci.c snvm_dev_release / the
     * destroy ioctl handler / NVM_UNMAP_* paths all do.
     */
    list_del(&map->group_link);

    if (map->release != NULL && map->data != NULL)
    {
        map->release(map);
    }

    kvfree(map);
}

struct map *map_find(const struct list *list, u64 vaddr)
{
    const struct list_node *element = list_next(&list->head);
    struct map *map = NULL;

    while (element != NULL)
    {
        map = container_of(element, struct map, list);

        if (map->owner == current)
        {
            if (map->vaddr == (vaddr & PAGE_MASK) || map->vaddr == (vaddr & GPU_PAGE_MASK))
            {
                return map;
            }
        }

        element = list_next(element);
    }

    return NULL;
}

struct map *map_find_by_pci_dev_and_idx(const struct list *list, const struct pci_dev *pdev, int idx, int is_cq)
{
    const struct list_node *element = list_next(&list->head);
    struct map *map = NULL;

    while (element != NULL)
    {
        map = container_of(element, struct map, list);

        if (map->pdev == pdev && map->ioq_idx == idx && map->is_cq == is_cq)
        {
            return map;
        }

        element = list_next(element);
    }

    return NULL;
}
EXPORT_SYMBOL_GPL(map_find_by_pci_dev_and_idx);

/*
 * snvme: walk `list` and unmap_and_release every descriptor whose
 * ->owner pointer equals `owner`.  Designed for snvm_dev_fops.release
 * cleanup when a userspace process dies without issuing NVM_UNMAP_*.
 *
 * Implementation note: unmap_and_release() does list_remove() on the
 * descriptor, so we must re-fetch list_next() from list->head on every
 * iteration -- saving a "next" pointer up front would dereference a
 * freed node on the next loop.
 */
unsigned long map_purge_by_owner(struct list *list, struct task_struct *owner)
{
    struct list_node *element;
    struct map *map;
    unsigned long freed = 0;

    if (list == NULL || owner == NULL)
        return 0;

    element = list_next(&list->head);
    while (element != NULL)
    {
        map = container_of(element, struct map, list);
        /*
         * Only reap legacy (non-group) maps here.
         *
         * Group-attached maps (group_id != 0) are owned per-fd
         * via the snvm_qgroup descriptor on file->private_data,
         * not per-task.  They are drained by
         * destroy_qgroup_locked() in pci.c during the fd's
         * release Pass 0 (which runs before this purge).
         *
         * If we walked group-attached maps here too, a process
         * holding multiple /dev/ssnvme<N> fds would have one
         * fd's release accidentally tear down maps registered
         * via a sibling fd, because all those fds share the same
         * task_struct as their map->owner.  That was the
         * symptom seen in the B2 smoke test (fd_d's release
         * reclaiming fd_a's group-attached map).
         */
        if (map->owner == owner && map->group_id == 0)
        {
            unmap_and_release(map);
            ++freed;
            /* head changed; restart from the new front */
            element = list_next(&list->head);
            continue;
        }
        element = list_next(element);
    }

    return freed;
}
EXPORT_SYMBOL_GPL(map_purge_by_owner);

static void release_user_pages(struct map *map)
{
    unsigned long i;
    struct page **pages;
    struct device *dev;

    dev = &map->pdev->dev;
    for (i = 0; i < map->n_addrs; ++i)
    {
        dma_unmap_page(dev, map->addrs[i], PAGE_SIZE, DMA_BIDIRECTIONAL);
    }

    pages = (struct page **)map->data;
    for (i = 0; i < map->n_addrs; ++i)
    {
        put_page(pages[i]);
    }

    kvfree(map->data);
    map->data = NULL;

    // printk(KERN_DEBUG "Released %lu host pages\n", map->n_addrs);
}

static long map_user_pages(struct map *map)
{
    unsigned long i;
    long retval;
    struct page **pages;
    struct device *dev;

    //  第1步：把用户页 pin 在物理内存里、不让换出/迁移。`get_user_pages` 的参数里 `map->vaddr` 是用户虚拟地址，`map->n_addrs` 是页数，`pages` 是输出的 struct page ** 数组。成功时返回实际 pin 住的页数（可能小于请求的页数），失败时返回负错误码。
    pages = (struct page **)kvcalloc(map->n_addrs, sizeof(struct page *), GFP_KERNEL);
    if (pages == NULL)
    {
        printk(KERN_CRIT "Failed to allocate page array\n");
        return -ENOMEM;
    }

#if LINUX_VERSION_CODE <= KERNEL_VERSION(4, 5, 7)
#warning "Building for older kernel, not properly tested"
    retval = get_user_pages(current, current->mm, map->vaddr, map->n_addrs, 1, 0, pages, NULL);
#elif LINUX_VERSION_CODE <= KERNEL_VERSION(4, 8, 17)
#warning "Building for older kernel, not properly tested"
    retval = get_user_pages(map->vaddr, map->n_addrs, 1, 0, pages, NULL);
#else
    retval = get_user_pages(map->vaddr, map->n_addrs, FOLL_WRITE, pages, NULL);
#endif
    if (retval <= 0)
    {
        kfree(pages);
        printk(KERN_ERR "get_user_pages() failed: %ld\n", retval);
        return retval;
    }

    if (map->n_addrs != retval)
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %ld\n", map->n_addrs, retval);
    }
    map->n_addrs = retval;
    map->page_size = PAGE_SIZE;
    // // 记下 page 数组，release 时 put_page 用
    map->data = (void *)pages;
    map->release = release_user_pages;

    // ★针对"这块 NVMe 盘"的 PCI 设备做映射★：把这些 struct page ** 映射成 NVMe 控制器可 DMA 的地址，填到 map->addrs[] 数组里。`dma_map_page` 的参数里 `dev` 是针对哪个设备做 DMA mapping（这里是 map->pdev，也就是 ctrl->pdev），`pages[i]` 是要映射的页，`PAGE_SIZE` 是映射的长度，`DMA_BIDIRECTIONAL` 是映射的方向（读写）。成功时返回 DMA 地址，失败时返回负错误码。
    dev = &map->pdev->dev;

    // 第2步：逐页 DMA 映射，过 IOMMU 拿到 SSD 能用的总线地址
    for (i = 0; i < map->n_addrs; ++i)
    {
        // 真正做 pin：然后对每个 page 做：把 CPU 用户态虚拟地址对应的物理页 pin 住，然后针对这个 NVMe PCI device 做 DMA mapping，得到 NVMe 控制器可以访问的 DMA 地址。
        map->addrs[i] = dma_map_page(dev, pages[i], 0, PAGE_SIZE, DMA_BIDIRECTIONAL);

        retval = dma_mapping_error(dev, map->addrs[i]);
        if (retval != 0)
        {
            printk(KERN_ERR "Failed to map page for some reason\n");
            return retval;
        }
        // printk("map_user_page: device: %02x:%02x.%1x\tvaddr: %llx\ti: %lu\tdma_addr: %llx\n", map->pdev->bus->number, PCI_SLOT(map->pdev->devfn), PCI_FUNC(map->pdev->devfn), (uint64_t) map->vaddr, i, map->addrs[i]);
    }

    return 0;
}

struct map *map_userspace(struct list *list, const struct ctrl *ctrl, u64 vaddr, unsigned long n_pages)
{
    long err;
    struct map *md;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    // 创建 struct map 描述符；// 页对齐 + 分配描述符
    md = create_descriptor(ctrl, vaddr & PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    // 把用户虚拟地址按 PAGE_MASK 对齐；
    md->page_size = PAGE_SIZE;

    // 调用 map_user_pages() pin 页面；// ★两步魔法★：① `get_user_pages` 把用户虚拟地址对应的物理页 pin 住，并返回 struct page **；② `dma_map_page` 把这些 struct page ** 映射成 NVMe 控制器可 DMA 的地址，填到 map->addrs[] 数组里。
    err = map_user_pages(md);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    // 把 map 挂到 host_list。 / // 挂进 host_list（全局链表，map_find 就在这个链表里找）。注意：map_userspace 只负责把 map 挂到 host_list，**不负责把 map 挂到 per-fd 的 snvm_qgroup 里**（这是后续 ioctl handler 的事了）。所以这里 group_id 保持 0，group_link 保持空。
    list_insert(list, &md->list);

    // printk(KERN_DEBUG "Mapped %lu host pages starting at address %llx\n",
    //         md->n_addrs, md->vaddr);
    return md;
}

static void force_release_gpu_memory(struct map *map)
{
    struct gpu_region *gd = (struct gpu_region *)map->data;
    struct list *list = map->ctrl_list;

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {
            const struct list_node *element = list_next(&list->head);
            struct ctrl *ctrl;

            uint32_t j = 0;
            while (element != NULL)
            {
                ctrl = container_of(element, struct ctrl, list);
                if (gd->mappings[j] != NULL)
                    nvfs_nvidia_p2p_dma_unmap_pages(ctrl->pdev, gd->pages, gd->mappings[j++]);

                element = list_next(element);
            }
            kfree(gd->mappings);
        }

        if (gd->pages != NULL)
        {
            nvfs_nvidia_p2p_free_page_table(gd->pages);
        }

        kfree(gd);
        map->data = NULL;

        printk(KERN_DEBUG "Nvidia driver forcefully reclaimed %lu GPU pages\n", map->n_addrs);
    }

    unmap_and_release(map);
}

static void force_release_gpu_ioqueue_memory(struct map *map)
{
    struct gpu_region *gd = (struct gpu_region *)map->data;

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {
            if (gd->mappings[0] != NULL)
                nvfs_nvidia_p2p_dma_unmap_pages(map->pdev, gd->pages, gd->mappings[0]);
            kfree(gd->mappings);
        }
        if (gd->pages != NULL)
        {
            nvfs_nvidia_p2p_free_page_table(gd->pages);
        }
        kfree(gd);
        map->data = NULL;
        printk(KERN_DEBUG "Nvidia driver forcefully reclaimed %lu GPU pages\n", map->n_addrs);
    }

    unmap_and_release(map);
}

void release_gpu_memory(struct map *map)
{
    struct gpu_region *gd = (struct gpu_region *)map->data;
    struct list *list = map->ctrl_list;

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {
            const struct list_node *element = list_next(&list->head);
            struct ctrl *ctrl;

            uint32_t j = 0;
            while (element != NULL)
            {
                ctrl = container_of(element, struct ctrl, list);
                if (gd->mappings[j] != NULL)
                    nvfs_nvidia_p2p_dma_unmap_pages(ctrl->pdev, gd->pages, gd->mappings[j++]);

                element = list_next(element);
            }
            kfree(gd->mappings);
        }

        if (gd->pages != NULL)
        {
            nvfs_nvidia_p2p_put_pages(0, 0, map->vaddr, gd->pages);
        }

        kfree(gd);
        map->data = NULL;

        // printk(KERN_DEBUG "Released %lu GPU pages\n", map->n_addrs);
    }
}

void release_gpu_ioqueue_memory(struct map *map)
{
    struct gpu_region *gd = (struct gpu_region *)map->data;

    if (gd != NULL)
    {
        if (gd->mappings != NULL)
        {

            if (gd->mappings[0] != NULL)
                nvfs_nvidia_p2p_dma_unmap_pages(map->pdev, gd->pages, gd->mappings[0]);

            kfree(gd->mappings);
        }
        if (gd->pages != NULL)
        {
            nvfs_nvidia_p2p_put_pages(0, 0, map->vaddr, gd->pages);
        }

        kfree(gd);
        map->data = NULL;
        // printk(KERN_DEBUG "Released %lu GPU pages\n", map->n_addrs);
    }
}

// 把 GPU 虚拟地址对应的 GPU pages pin 住，并把这些 GPU pages 映射成 NVMe 控制器可 DMA 的地址。
// **和 host 路径的对应关系**：`get_user_pages` ↔ `nvfs_nvidia_p2p_get_pages`（pin），`dma_map_page` ↔ `nvfs_nvidia_p2p_dma_map_pages`（映射），结果都落进 `map->addrs[]`。**NPU 迁移就是把这两个 nvfs 调用换成昇腾 HBM 的 pin/map 能力**（见第 3 部分）。在没有 NVIDIA 驱动的华为机器上，这些 nvfs 函数返回 `-ENOMEM`，走不到，所以第一阶段只用 host 路径。
// `map_gpu_ioqueue_memory`（map.c : 528）是同一套，但只映射 * *当前这块 **NVMe（队列环只服务于本盘），所以 `mappings` 只分配 1 个。
int map_gpu_memory(struct map *map, struct list *list)
{
    unsigned long i;
    uint32_t j;
    int err;
    struct gpu_region *gd;
    const struct list_node *element;
    struct ctrl *ctrl;

    // // GPU 区描述符
    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return -ENOMEM;
    }
    // 每块 NVMe 一份 p2p 映射表；如果有多块 NVMe，就有多份 p2p 映射表（每份表里都是同一批 GPU pages 的不同 DMA 地址）。所以这里分配 max_num_ctrls 份映射表的空间，后续根据实际 NVMe 数量来用。
    gd->mappings = (nvidia_p2p_dma_mapping_t **)kmalloc(sizeof(nvidia_p2p_dma_mapping_t *) * max_num_ctrls, GFP_KERNEL);

    if (gd->mappings == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        kfree(gd);
        return -ENOMEM;
    }
    for (j = 0; j < max_num_ctrls; j++)
        gd->mappings[j] = NULL;

    gd->pages = NULL;
    // gd->mappings = NULL;

    map->page_size = GPU_PAGE_SIZE; // 64KB
    map->data = gd;
    map->release = release_gpu_memory;

    // get the io addr  // map.c:475 ★pin 显存：把 GPU 虚拟地址对应的 GPU pages pin 住，并把这些 GPU pages 映射成 NVMe 控制器可 DMA 的地址。`nvfs_nvidia_p2p_get_pages` 的参数里 `map->vaddr` 是 GPU 虚拟地址，`GPU_PAGE_SIZE * map->n_addrs` 是映射的长度，`gd->pages` 是输出的 p2p page table，最后两个参数是当 GPU 内存被强制回收时的回调函数和参数（这里传 map 自身）。成功时返回 0，失败时返回负错误码。
    err = nvfs_nvidia_p2p_get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages,
                                    (void (*)(void *))force_release_gpu_memory, map);
    if (err != 0)
    {
        printk(KERN_ERR "nvfs_nvidia_p2p_get_pages() failed: %d\n", err);
        return err;
    }

    element = list_next(&list->head);

    // create the map between each nvme device an GPU 对每块 NVMe 盘做 p2p DMA 映射
    j = 0;
    while (element != NULL) // 遍历 ctrl_list
    {
        ctrl = container_of(element, struct ctrl, list);

        err = nvfs_nvidia_p2p_dma_map_pages(ctrl->pdev, gd->pages, gd->mappings + j);
        if (err != 0)
        {
            // printk(KERN_ERR "nvfs_nvidia_p2p_dma_map_pages() failed for nvme%u: %d\n", j-1, err);
            return err;
        }
        j++;
        // for (i = 0; i < map->n_addrs; ++i)
        //{

        //   printk("device: %u\ti: %lu\tpaddr: %llx\n", (j-1), i, (uint64_t)  gd->mappings[j-1]->dma_addresses[i]);
        //}
        // ★取 p2p DMA 地址：把这批 GPU pages 的 NVMe 可 DMA 地址填到 map->addrs[] 数组里。注意：每块 NVMe 盘的 DMA 地址都可能不同，所以这里取第 j 块 NVMe 盘的 DMA 地址（gd->mappings[j]），填到 map->addrs[] 里。最终 map->addrs[] 里存的，是针对第 j 块 NVMe 盘的 DMA 地址。
        if (j == 1)
        {
            for (i = 0; i < map->n_addrs; ++i)
            {
                map->addrs[i] = gd->mappings[0]->dma_addresses[i];
                // printk("++paddr: %llx\n", (uint64_t) map->addrs[i]);
            }
        }
        element = list_next(element);
    }

    if (map->n_addrs != gd->pages->entries)
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %u\n", map->n_addrs, gd->pages->entries);
    }

    map->n_addrs = gd->pages->entries;

    // printk("vaddr: %llx\n", (uint64_t) map->vaddr);
    //    for (j = 0; j < map->n_addrs; j++)
    //        printk("\tpaddr: %llx\n", (uint64_t) map->addrs[j]);

    return 0;
}

int map_gpu_ioqueue_memory(struct map *map)
{
    unsigned long i;
    int err;
    struct gpu_region *gd;
    gd = kmalloc(sizeof(struct gpu_region), GFP_KERNEL);
    if (gd == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        return -ENOMEM;
    }

    gd->mappings = (nvidia_p2p_dma_mapping_t **)kmalloc(sizeof(nvidia_p2p_dma_mapping_t *) * 1, GFP_KERNEL);

    if (gd->mappings == NULL)
    {
        printk(KERN_CRIT "Failed to allocate mapping descriptor\n");
        kfree(gd);
        return -ENOMEM;
    }

    gd->pages = NULL;
    // gd->mappings = NULL;

    map->page_size = GPU_PAGE_SIZE;
    map->data = gd;
    map->release = release_gpu_ioqueue_memory;

    // get the io addr
    err = nvfs_nvidia_p2p_get_pages(0, 0, map->vaddr, GPU_PAGE_SIZE * map->n_addrs, &gd->pages,
                                    (void (*)(void *))force_release_gpu_ioqueue_memory, map);
    if (err != 0)
    {
        printk(KERN_ERR "nvfs_nvidia_p2p_get_pages() failed: %d\n", err);
        return err;
    }

    err = nvfs_nvidia_p2p_dma_map_pages(map->pdev, gd->pages, &gd->mappings[0]);
    if (err != 0)
    {
        // printk(KERN_ERR "nvfs_nvidia_p2p_dma_map_pages() failed for nvme%u: %d\n", j-1, err);
        return err;
    }

    for (i = 0; i < map->n_addrs; ++i)
    {
        map->addrs[i] = gd->mappings[0]->dma_addresses[i];
        // printk("++paddr: %llx\n", (uint64_t) map->addrs[i]);
    }

    if (map->n_addrs != gd->pages->entries)
    {
        printk(KERN_WARNING "Requested %lu GPU pages, but only got %u\n", map->n_addrs, gd->pages->entries);
    }

    map->n_addrs = gd->pages->entries;

    // printk("vaddr: %llx\n", (uint64_t) map->vaddr);
    //    for (j = 0; j < map->n_addrs; j++)
    //        printk("\tpaddr: %llx\n", (uint64_t) map->addrs[j]);

    return 0;
}

struct map *map_device_memory(struct list *list, const struct ctrl *ctrl, u64 vaddr, unsigned long n_pages, struct list *ctrl_list)
{
    int err;
    struct map *md = NULL;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & GPU_PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }

    md->page_size = GPU_PAGE_SIZE;
    md->ctrl_list = ctrl_list;
    err = map_gpu_memory(md, ctrl_list);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    // printk(KERN_DEBUG "Mapped %lu GPU pages starting at address %llx\n",
    //         md->n_addrs, md->vaddr);
    return md;
}

struct map *map_device_ioqueue_memory(struct list *list, const struct ctrl *ctrl, u64 vaddr, unsigned long n_pages)
{
    int err;
    struct map *md = NULL;

    if (n_pages < 1)
    {
        return ERR_PTR(-EINVAL);
    }

    md = create_descriptor(ctrl, vaddr & GPU_PAGE_MASK, n_pages);
    if (IS_ERR(md))
    {
        return md;
    }
    md->page_size = GPU_PAGE_SIZE;
    err = map_gpu_ioqueue_memory(md);
    if (err != 0)
    {
        unmap_and_release(md);
        return ERR_PTR(err);
    }

    list_insert(list, &md->list);

    // printk(KERN_DEBUG "Mapped %lu GPU pages starting at address %llx\n",
    //         md->n_addrs, md->vaddr);
    return md;
}
