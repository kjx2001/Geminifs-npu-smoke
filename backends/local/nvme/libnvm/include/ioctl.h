#ifndef __NVM_INTERNAL_LINUX_IOCTL_H__
#define __NVM_INTERNAL_LINUX_IOCTL_H__
#include <asm-generic/ioctl.h>
#ifdef __linux__

#include <linux/types.h>
#include <asm/ioctl.h>

#ifndef __KERNEL__ //the header shared between kernel and user space
#include <stddef.h>
#include <stdint.h>
#endif

#define NVM_IOCTL_TYPE          0x80
#define NVM_CTRL_IOCTL_TYOE     0x90

#define DISK_NAME_LEN       32

#define DISK_NAME_COPY(dest, src) \
    memcpy(dest, src, DISK_NAME_LEN)


/* Memory map request */
struct nvm_ioctl_map
{
    uint64_t    vaddr_start;
    size_t      n_pages;
    uint64_t*   ioaddrs;
    int ioq_idx; // if the ioq_idx > 0, indicate the map is a IOQ
    int is_cq; // cq = 1 sq = 0
};

struct nvm_ioctl_dev
{
    uint32_t    nr_user_q;
    uint32_t    start_cq_idx;
    uint8_t     dstrd; 
    size_t      max_data_size; //get the ctrl->max_hw_sectors from kernel
    size_t      block_size;    // ns->lba_shift
    char        disk_name[DISK_NAME_LEN];
};

/*
 * Queue budget descriptor passed to NVM_SET_IOQ_NUM.
 *
 * Replaces the historical (and undersized) "pack ioq count into
 * nvm_ioctl_map.ioq_idx" pattern.  Lets userspace declare, in one
 * shot:
 *
 *   - how many IOQs it intends to register for itself (ioq_num /
 *     on_host -- same semantics as the legacy two fields),
 *   - what cap to put on the kernel side so the controller has room
 *     to grant both sets (cap_kernel_ioq, nr_write, nr_poll),
 *   - how to slice the user share across logical owners (typically
 *     GPUs) via the groups array.
 *
 * Layout is fixed-size (groups[NVM_MAX_QUEUE_GROUPS]) so the ioctl
 * is single-copy and _IOC_SIZE is stable.  Eight groups matches the
 * 8-GPU H20 / H100 SXM topology and is the upper bound exposed via
 * sys_config.yaml's queue_groups list.
 */
#define NVM_MAX_QUEUE_GROUPS  8

struct nvm_queue_group {
    uint32_t    owner_id;    /* opaque tag, typically a GPU id.        */
    uint32_t    count;       /* number of (SQ+CQ) pairs in this group. */
    int32_t     numa_node;   /* doc-only hint; kernel does NOT enforce */
                             /* this.  -1 = "don't care".              */
    uint32_t    reserved;    /* MBZ.                                   */
};

#define NVM_QUEUE_SETUP_F_ON_HOST   (1U << 0)
                                     /* Set => user IOQ ring memory   */
                                     /* lives in host RAM (pinned via */
                                     /* NVM_MAP_HOST_MEMORY).  Clear  */
                                     /* => GPU device memory (pinned  */
                                     /* via NVM_MAP_DEVICE_QUEUE_     */
                                     /* MEMORY).                      */

struct nvm_ioctl_setup
{
    uint32_t    ioq_num;        /* total user-side IOQ count (sum of  */
                                /* SQ+CQ pair counts; legacy field).  */
    uint32_t    flags;          /* NVM_QUEUE_SETUP_F_*.               */
    uint32_t    cap_kernel_ioq; /* upper bound on kernel-side IOQ     */
                                /* count requested from the           */
                                /* controller.  0 = use the upstream  */
                                /* default (num_possible_cpus()).     */
                                /* Set this when the controller's     */
                                /* MSI-X grant is smaller than the    */
                                /* host CPU count and you want the    */
                                /* user share path to actually        */
                                /* activate instead of falling back   */
                                /* to dma_alloc_coherent.             */
    uint32_t    nr_write;       /* per-BDF override of the            */
                                /* write_queues module parameter;     */
                                /* 0 = leave at module default.       */
    uint32_t    nr_poll;        /* ditto for poll_queues.             */
    uint32_t    nr_groups;      /* <= NVM_MAX_QUEUE_GROUPS.  0 means  */
                                /* "no per-owner partitioning";       */
                                /* groups[] is then ignored.          */
    uint32_t    reserved[2];    /* MBZ; future extension.             */
    struct nvm_queue_group  groups[NVM_MAX_QUEUE_GROUPS];
};

struct pci_device_addr{ // Removed redundant definition
    int domain;
    int bus;
    int slot;
    int func;
};

/* Supported operations */
enum nvm_ioctl_type{
    NVM_MAP_HOST_MEMORY             = _IOW(NVM_IOCTL_TYPE, 1, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_MEMORY           = _IOW(NVM_IOCTL_TYPE, 2, struct nvm_ioctl_map),
    NVM_MAP_DEVICE_QUEUE_MEMORY     = _IOW(NVM_IOCTL_TYPE, 3, struct nvm_ioctl_map),
    NVM_UNMAP_HOST_MEMORY           = _IOW(NVM_IOCTL_TYPE, 4, uint64_t),
    NVM_UNMAP_DEVICE_MEMORY         = _IOW(NVM_IOCTL_TYPE, 5, uint64_t),
    NVM_UNMAP_DEVICE_QUEUE_MEMORY   = _IOW(NVM_IOCTL_TYPE, 6, uint64_t),
    /*
     * NVM_SET_IOQ_NUM now carries a struct nvm_ioctl_setup, NOT the
     * historical nvm_ioctl_map.  This is an ABI break vs.
     * pre-Geminifs snvme-5.15-public binaries -- the project is
     * open-source and we deliberately do not keep the legacy
     * payload.  Any new userspace MUST populate nvm_ioctl_setup;
     * the ioctl number has the new _IOC_SIZE baked in, so a stale
     * binary trying the old layout will be rejected with -ENOTTY
     * at ioctl entry rather than silently mis-decoding the request.
     */
    NVM_SET_IOQ_NUM                 = _IOW(NVM_IOCTL_TYPE, 7, struct nvm_ioctl_setup),
    NVM_SET_SHARE_REG               = _IOW(NVM_IOCTL_TYPE, 8, struct nvm_ioctl_dev),
    NVM_GET_DEV_INFO                = _IOR(NVM_IOCTL_TYPE, 9, struct nvm_ioctl_dev),   
    NVM_CLEAR_IOQ_NUM               = _IOW(NVM_IOCTL_TYPE, 10, struct nvm_ioctl_dev),
};

// snvm_ctrl_ioctl_type
//
// Note: opcode 5 (formerly SNVM_CACULATE_PCIDISTANCE) was removed. Do NOT
// reuse it for a new command for at least one release cycle, otherwise old
// userspace binaries will silently get a different result.
enum snvm_ctrl_ioctl_type{
    SNVM_DEVICE_BIND                = _IOW(NVM_CTRL_IOCTL_TYOE, 1, struct pci_device_addr),
    SNVM_DEVICE_UNBIND              = _IOW(NVM_CTRL_IOCTL_TYOE, 2, struct pci_device_addr),
    SNVM_CHRDEV_CREATE              = _IOWR(NVM_CTRL_IOCTL_TYOE, 3, struct pci_device_addr),
    SNVM_CHRDEV_REMOVE              = _IOW(NVM_CTRL_IOCTL_TYOE, 4, struct pci_device_addr),
};


/* SNVME initiazation process*/
/*
1. Use NVM_SET_IOQ_NUM to declare both the kernel-side IO-queue
   budget AND the user-side IO-queue partition.  The argument is a
   struct nvm_ioctl_setup; key fields:
     ioq_num         total user IOQ count (kernel reserves room for
                     ioq_num CQs on top of cap_kernel_ioq)
     cap_kernel_ioq  upper bound on kernel-side IOQ count requested
                     from the controller.  Set this to less than
                     num_possible_cpus() when the controller's
                     MSI-X grant is smaller than the host CPU count
                     (e.g. Intel DC SSD: MSI-X=136 vs 192 vCPUs),
                     otherwise the user share path falls back to
                     dma_alloc_coherent.
     nr_groups       optional per-owner (typically per-GPU) split.
2. Use NVM_MAP_HOST_MEMORY/NVM_MAP_DEVICE_MEMORY/
   NVM_MAP_DEVICE_QUEUE_MEMORY to register DMA addresses for each
   declared user IOQ; the total must match ioq_num.
3. Use NVM_SET_SHARE_REG to flip the use_sreg gate to 1; the next
   SNVM_DEVICE_BIND will then commit the user share at probe time
   instead of letting upstream nvme allocate the queue pages.
4. Use SNVM_DEVICE_BIND to detach the in-tree nvme driver from the
   target BDF and bind snvme; probe consumes the use_sreg gate and
   commits the kernel/user IOQ split that NVM_SET_IOQ_NUM declared.
*/
#endif /* __linux__ */
#endif /* __NVM_INTERNAL_LINUX_IOCTL_H__ */
