#ifndef __NVM_CTRL_H__
#define __NVM_CTRL_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif




/* 
 * Minimum size of mapped controller memory.
 */
#define NVM_CTRL_MEM_MINSIZE                        0x2000



#if defined (__unix__)
/*
 * Initialize NVM controller handle.
 *
 * Read from controller registers and initialize controller handle. 
 * This function should be used when using the kernel module or to manually
 * read from sysfs.
 *
 * Note: fd must be opened with O_RDWR and O_NONBLOCK
 */
int nvm_ctrl_init(nvm_ctrl_t** ctrl, int snvme_c_fd, int snvme_d_fd);
#endif



/* 
 * Initialize NVM controller handle.
 *
 * Read from controller registers and initialize the controller handle using
 * a memory-mapped pointer to the PCI device BAR.
 *
 * This function should be used when neither SmartIO nor the disnvme kernel
 * module are used.
 *
 * Note: ctrl_mem must be at least NVM_CTRL_MEM_MINSIZE large and mapped
 *       as IO memory. See arguments for mmap() for more info.
 */
int nvm_raw_ctrl_init(nvm_ctrl_t** ctrl);

int ioctl_get_dev_info(nvm_ctrl_t* ctrl, struct disk* d);
/*
 * Release controller handle.
 */
void nvm_ctrl_free(nvm_ctrl_t* ctrl);


int nvm_queue_set(nvm_ctrl_t* ctrl, int q_num);
/*
 * Full-fidelity queue-budget configuration entry point.  Wraps the
 * NVM_SET_IOQ_NUM ioctl with a struct nvm_ioctl_setup payload that
 * lets the caller pin both the kernel-side IOQ cap (cap_kernel_ioq)
 * and the per-owner partition of the user share (groups[]).
 *
 * struct nvm_ioctl_setup is declared in <ioctl.h>; callers that
 * only need the legacy single-arg behaviour should keep using
 * nvm_queue_set() above, which is a thin wrapper around this.
 */
struct nvm_ioctl_setup;
int nvm_queue_setup(nvm_ctrl_t* ctrl, struct nvm_ioctl_setup* setup);
int nvm_queue_clear(nvm_ctrl_t* ctrl);
int nvm_queue_share(nvm_ctrl_t *ctrl);
int nvm_device_bind(nvm_ctrl_t* ctrl);
int nvm_device_unbind(nvm_ctrl_t* ctrl);
int nvm_chrdev_create(int fd_control, struct pci_device_addr *device_addr);
int nvm_chrdev_remove(int fd_control, struct pci_device_addr *device_addr);

int nvm_controller_init(nvm_ctrl_t** ctrl, const char *snvme_control_path, const char *pci_addr);
int nvm_device_init(nvm_ctrl_t* ctrl);

struct controller* ctrl_to_controller(nvm_ctrl_t* ctrl);

/* ===================================================================
 * B3 controller bring-up + queue-group API.
 *
 * All declared as plain functions; the Controller C++ class is built
 * on top of these.  Smoke tests and any future runtime can drive the
 * sequence directly without going through Controller.
 *
 * Sequence to bring a controller fully up:
 *
 *   nvm_controller_init_b3(&ctrl, "/dev/snvm_control", "0000:08:00.0",
 *                          36 /-* kernel_ioq_cap *-/, &disk);
 *   nvm_create_group(ctrl, &group_id, &max_q);
 *
 * Then for each queue:
 *
 *   nvm_dma_map_ring_{host,device}(... group_id ..., is_cq=0);  // SQ ring
 *   nvm_dma_map_ring_{host,device}(... group_id ..., is_cq=1);  // CQ ring
 *
 * Then submit all (SQ, CQ) pairs in one batch:
 *
 *   struct nvm_ioctl_add_user_queue req = { ... };
 *   nvm_add_user_queue(ctrl, &req);
 *
 * Tear-down (kernel cascades through Delete I/O SQ + CQ + map purge):
 *
 *   nvm_destroy_group(ctrl, group_id);
 * =================================================================== */

int nvm_controller_init_b3(nvm_ctrl_t** ctrl,
                           const char* snvme_control_path,
                           const char* pci_addr,
                           uint32_t kernel_ioq_cap,
                           struct disk* out_disk);

int nvm_set_kernel_ioq_cap(nvm_ctrl_t* ctrl, uint32_t cap);
int nvm_set_kernel_ioq_cap_fd(int fd_dev, uint32_t cap);

int nvm_create_group(nvm_ctrl_t* ctrl,
                     uint32_t* out_group_id,
                     uint32_t* out_max_queues);
int nvm_destroy_group(nvm_ctrl_t* ctrl, uint32_t group_id);

struct nvm_ioctl_add_user_queue;
int nvm_add_user_queue(nvm_ctrl_t* ctrl,
                       struct nvm_ioctl_add_user_queue* req);

int nvm_wait_dev_info(nvm_ctrl_t* ctrl,
                      struct nvm_ioctl_dev* out_info,
                      uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* __NVM_CTRL_H__ */
