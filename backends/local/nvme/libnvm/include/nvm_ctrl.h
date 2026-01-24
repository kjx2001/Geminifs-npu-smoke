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
int nvm_queue_clear(nvm_ctrl_t* ctrl);
int nvm_queue_share(nvm_ctrl_t *ctrl);
int nvm_device_bind(nvm_ctrl_t* ctrl);
int nvm_device_unbind(nvm_ctrl_t* ctrl);
int nvm_chrdev_create(int fd_control, struct pci_device_addr *device_addr);
int nvm_chrdev_remove(int fd_control, struct pci_device_addr *device_addr);

int nvm_controller_init(nvm_ctrl_t** ctrl, const char *snvme_control_path, const char *pci_addr);
int nvm_device_init(nvm_ctrl_t* ctrl);

struct controller* ctrl_to_controller(nvm_ctrl_t* ctrl);

#endif /* __NVM_CTRL_H__ */
