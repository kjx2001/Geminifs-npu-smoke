#ifndef __linux__
#error "Must compile for Linux"
#endif

#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_ctrl.h>
#include <nvm_util.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#include "map.h"
#include "ioctl.h"
#include "lib_ctrl.h"
#include "dprintf.h"
#include <nvm_queue.h>
#include <nvm_error.h>






/*
 * Unmap controller memory and close file descriptor.
 */
static void release_device(struct device* dev)
{

    close(dev->fd_control);
    close(dev->fd_dev);
    free(dev);
}



/*
 * Call kernel module ioctl and map memory for DMA.
 */
static int ioctl_map(const struct device* dev, const struct va_range* va, uint64_t* ioaddrs)
{
    const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
    enum nvm_ioctl_type type;

    switch (m->type)
    {
        case MAP_TYPE_API:
        case MAP_TYPE_HOST:
            type = NVM_MAP_HOST_MEMORY;
            break;
        case MAP_TYPE_CUDA:
            type = NVM_MAP_DEVICE_MEMORY;
            break;
        case MAP_TYPE_CUDA_QUEUE:
            type = NVM_MAP_DEVICE_QUEUE_MEMORY;
            break;
        default:
            dprintf("Unknown memory type in map for device");
            return EINVAL;
    }

    struct nvm_ioctl_map request = {
        .vaddr_start = (uintptr_t) m->buffer,
        .n_pages = va->n_pages,
        .ioaddrs = ioaddrs,
        .ioq_idx = m->ioq_idx,
        .is_cq = m->is_cq
    };

    int err = ioctl(dev->fd_dev, type, &request);
    if (err < 0)
    {
        dprintf("Page mapping kernel request failed (ptr=%p, n_pages=%zu): %s\n", 
                m->buffer, va->n_pages, strerror(errno));
        return errno;
    }
    
    return 0;
}
struct controller* ctrl_to_controller(nvm_ctrl_t* ctrl)
{
    struct controller* controll;
    controll = _nvm_container_of(ctrl, struct controller, handle);
    if(controll)
        return controll;
    else
        return NULL;
}



/*
 * Call kernel module ioctl and unmap memory.
 */
static void ioctl_unmap(const struct device* dev, const struct va_range* va)
{
    const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
    uint64_t addr = (uintptr_t) m->buffer;
    unsigned int unmap_type;
    if(m->type == MAP_TYPE_HOST)
        unmap_type = NVM_UNMAP_HOST_MEMORY;
    else if(m->type == MAP_TYPE_CUDA)
        unmap_type = NVM_UNMAP_DEVICE_MEMORY;
    else
    {
        printf("Page unmapping kernel type error,type is %u\n",m->type);
        return;
    }
    int err = ioctl(dev->fd_dev, unmap_type, &addr);
    if (err < 0)
    {
        dprintf("Page unmapping kernel request failed: %s\n", strerror(errno));
    }
}
static void ioctl_queue_unmap(const struct device* dev, const struct va_range* va)
{
    const struct ioctl_mapping* m = _nvm_container_of(va, struct ioctl_mapping, range);
    uint64_t addr = (uintptr_t) m->buffer;
    

    int err = ioctl(dev->fd_dev, NVM_UNMAP_DEVICE_QUEUE_MEMORY, &addr);
    if (err < 0)
    {
        dprintf("Page unmapping kernel request failed: %s\n", strerror(errno));
    }
}

int ioctl_get_dev_info(nvm_ctrl_t* ctrl, struct disk* d)
{
    struct nvm_ioctl_dev dev_info;
    struct controller* container;
    int err;
    container  = ctrl_to_controller(ctrl);
    if (container==NULL){
        printf("container error!\n");
        return -1;
    }
    
    err = ioctl(container->device->fd_dev, NVM_GET_DEV_INFO, &dev_info);
    if (err < 0){
        printf("ioctl_get_dev_info err is %d\n",err);
        return errno;
    }
    ctrl->start_cq_idx = dev_info.start_cq_idx;
    // ctrl->dstrd = dev_info.dstrd;
    ctrl->nr_user_q = dev_info.nr_user_q;

    d->max_data_size = dev_info.max_data_size * 512; //get the ctrl->max_hw_sectors from kernel    
    d->block_size = dev_info.block_size; // ns->lba_shift
    memcpy(d->disk_name, dev_info.disk_name, DISK_NAME_LEN * sizeof(char));
    nvm_debug("Disk info: start cq index is %u, max data size is %lu, block size is %lu", ctrl->start_cq_idx, d->max_data_size, d->block_size);
    return 0;
}

static inline int ioctl_chrdev_helper(int fd_control, struct pci_device_addr *device_addr, enum snvm_ctrl_ioctl_type type){
    int err;

    if ((type != SNVM_CHRDEV_CREATE) && (type != SNVM_CHRDEV_REMOVE)){
        printf("ioctl_chrdev_helper type error!\n");
        return -1;
    }

    err = ioctl(fd_control, type, device_addr);
    if (err < 0){
        printf("ioctl_chrdev_helper err is %d\n",err);
        return errno;
    }
    return 0;
}

int nvm_chrdev_create(int fd_control, struct pci_device_addr *device_addr){
    return ioctl_chrdev_helper(fd_control, device_addr, SNVM_CHRDEV_CREATE);
}

int nvm_chrdev_remove(int fd_control, struct pci_device_addr *device_addr){
    return ioctl_chrdev_helper(fd_control, device_addr, SNVM_CHRDEV_REMOVE);
}


static inline int ioctl_device_helper(nvm_ctrl_t* ctrl, enum snvm_ctrl_ioctl_type type){
    struct controller* container;
    int err;

    container = ctrl_to_controller(ctrl);
    if(container==NULL)
    {
        printf("container error!\n");
        return -1;
    }

    if ((type != SNVM_DEVICE_BIND) && (type != SNVM_DEVICE_UNBIND)){
        printf("ioctl_device_helper type error!\n");
        return -1;
    }

    err = ioctl(container->device->fd_control, type, &(ctrl->pdev_addr));
    if (err < 0){
        printf("ioctl_device_helper err is %d\n",err);
        return errno;
    }
    return 0;
}

int nvm_device_bind(nvm_ctrl_t* ctrl){
    return ioctl_device_helper(ctrl, SNVM_DEVICE_BIND);
}

int nvm_device_unbind(nvm_ctrl_t* ctrl){
    return ioctl_device_helper(ctrl, SNVM_DEVICE_UNBIND);
}


static inline int ioctl_queue_helper(nvm_ctrl_t* ctrl, int arg, enum nvm_ioctl_type type)
{
    struct controller* container;
    struct nvm_ioctl_map request;
    int err;

    container = ctrl_to_controller(ctrl);
    if (container == NULL){
        nvm_error("container error!");
        return -1;
    }
    memset(&request, 0, sizeof(request));
    switch (type) {
        case NVM_SET_IOQ_NUM:
            request.ioq_idx = arg;
            request.is_cq = (int)ctrl->on_host;
            break;
        case NVM_SET_SHARE_REG:
            request.ioq_idx = arg;
            break;
        case NVM_CLEAR_IOQ_NUM:
            break;
        default:
            return EINVAL;
    }

    err = ioctl(container->device->fd_dev, type, &request);
    if (err < 0){
        printf("ioctl_queue_helper err is %d\n",err);
        return errno;
    }
    
    return 0;
}

int nvm_queue_set(nvm_ctrl_t* ctrl, int q_num){
    return ioctl_queue_helper(ctrl, q_num, NVM_SET_IOQ_NUM);
}

int nvm_queue_clear(nvm_ctrl_t* ctrl){
    return ioctl_queue_helper(ctrl, 0, NVM_CLEAR_IOQ_NUM);
}
int nvm_queue_share(nvm_ctrl_t *ctrl){
    return ioctl_queue_helper(ctrl, 1, NVM_SET_SHARE_REG);
}

int nvm_ctrl_init(nvm_ctrl_t** ctrl, int snvme_c_fd, int snvme_d_fd)
{
    int err;
    struct device* dev;

    const struct device_ops ops = {
        .release_device = &release_device,
        .map_range = &ioctl_map,
        .unmap_range = &ioctl_unmap,
        .unmap_queue_range = &ioctl_queue_unmap,
    };

    *ctrl = NULL;

    dev = (struct device*) malloc(sizeof(struct device));
    if (dev == NULL)
    {
        dprintf("Failed to allocate device handle: %s\n", strerror(errno));
        return ENOMEM;
    }

    dev->fd_control = dup(snvme_c_fd);
    if (dev->fd_control < 0)
    {
        free(dev);
        dprintf("Could not duplicate file descriptor: %s\n", strerror(errno));
        return errno;
    }
    
    dev->fd_dev = dup(snvme_d_fd);
    if (dev->fd_control < 0)
    {
        close(dev->fd_control);
        free(dev);
        dprintf("Could not duplicate file descriptor: %s\n", strerror(errno));
        return errno;
    }
    

    err = fcntl(dev->fd_control, F_SETFD, O_RDWR);
    if (err == -1)
    {
        close(dev->fd_control);
        close(dev->fd_dev);
        free(dev);
        dprintf("Failed to set file descriptor control: %s\n", strerror(errno));
        return errno;
    }
    err = fcntl(dev->fd_dev, F_SETFD, O_RDWR);
    if (err == -1)
    {
        close(dev->fd_control);
        close(dev->fd_dev);
        free(dev);
        dprintf("Failed to set file descriptor control: %s\n", strerror(errno));
        return errno;
    }

    void* mm_ptr = mmap(NULL, NVM_CTRL_MEM_MINSIZE, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FILE|MAP_LOCKED, dev->fd_dev, 0);
    if (mm_ptr == NULL)
    {
        close(dev->fd_control);
        close(dev->fd_dev);
        free(dev);
        dprintf("Failed to map device memory: %s\n", strerror(errno));
        return err;
    }    

    err = _nvm_ctrl_init(ctrl, dev, &ops, DEVICE_TYPE_IOCTL,mm_ptr,NVM_CTRL_MEM_MINSIZE);
    if (err != 0)
    {
        release_device(dev);
        return err;
    }
    return 0;
}

int nvm_device_init(nvm_ctrl_t* ctrl){
    int status;

    status = nvm_queue_share(ctrl);
    if (status != 0){
        nvm_error("Failed to set reg snvme");
        return EFAULT;
    }

    status = nvm_device_bind(ctrl);
    if (status != 0){
        nvm_error("Failed to bind device");
        return EFAULT;
    }
    sleep(3);
    return 0;
}

int nvm_controller_init(nvm_ctrl_t** ctrl, const char *snvme_control_path, const char *pci_addr){
    struct pci_device_addr device_addr, pdev_addr;
    cudaError_t err;
    char snvme_path[256];
    int snvme_d_fd, snvme_c_fd;
    int status;

    snvme_c_fd = open(snvme_control_path, O_RDWR | O_NONBLOCK);
    if (snvme_c_fd < 0){
        nvm_error("Failed to open control descriptor");
        return EFAULT;
    }
    
    status = sscanf(pci_addr, "%x:%x:%x.%x", &device_addr.domain, 
                                                       &device_addr.bus, 
                                                       &device_addr.slot, 
                                                       &device_addr.func);
    if (status != 4){
        nvm_error("Failed to parse PCI address");
        return EFAULT;
    }
    // to avoid pci device address changes
    pdev_addr = device_addr;
    // create chrdev 
    status = nvm_chrdev_create(snvme_c_fd, &device_addr);
    if (status != 0){
        nvm_error("Failed to open device descriptor");
        return EFAULT;
    }
    
    snprintf(snvme_path, sizeof(snvme_path), "/dev/ssnvme%d", device_addr.domain);
    nvm_info("Create chrdev: %s", snvme_path);

    snvme_d_fd = open(snvme_path, O_RDWR | O_NONBLOCK);
    if (snvme_d_fd < 0){
        nvm_error("Failed to open device descriptor");
        return EFAULT;
    }

    status = nvm_ctrl_init(ctrl, snvme_c_fd, snvme_d_fd);
    if (status != 0){
        nvm_error("Failed to initialize controller");
        return EFAULT;
    }
    (*ctrl)->on_host = 0;
    (*ctrl)->pdev_addr = pdev_addr;

    close(snvme_c_fd);
    close(snvme_d_fd);

    
    err = cudaHostRegister((void*) (*ctrl)->mm_ptr, NVM_CTRL_MEM_MINSIZE, cudaHostRegisterIoMemory); //UVM
    if (err != cudaSuccess){
        nvm_error("Failed to register device memory");
        return EFAULT;
    }

    return 0;
}

