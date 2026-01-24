#ifndef __BENCHMARK_CTRL_H__
#define __BENCHMARK_CTRL_H__

// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <cstdint>
#include "buffer.h"
#include "ioctl.h"
#include "nvm_types.h"
#include "nvm_ctrl.h"
#include "nvm_error.h"
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <algorithm>
#include <cuda/atomic>
#include "file.h"
#include "queue.h"

// Forward declarations

#define MAX_QUEUES 1024
#define NVM_CTRL_IOQ_MINNUM    64

struct Controller
{
    cuda::atomic<uint64_t, cuda::thread_scope_device> access_counter;
    nvm_ctrl_t*             ctrl;
    struct nvm_ctrl_info    info;
    struct nvm_ns_info      ns;
    struct disk             disk;
    uint16_t                n_sqs;
    uint16_t                n_cqs;
    uint16_t                n_qps;
    uint16_t                n_user_qps;

    uint32_t                deviceId;
    QueuePair**             h_qps;
    QueuePair*              d_qps;


    cuda::atomic<uint64_t, cuda::thread_scope_device> queue_counter;

    uint32_t page_size;
    uint32_t blk_size;
    uint32_t blk_size_log;
    char* dev_path;
    // char* dev_mount_path;
    std::string dev_mount_path;
    void* d_ctrl_ptr;
    BufferPtr d_ctrl_buff;

    Controller(const char* snvme_control_path,  
                const char* pci_addr,
                std::string mount_path,
                uint32_t ns_id, 
                uint32_t cudaDevice, 
                uint64_t queueDepth, 
                uint64_t numQueues);


    void print_reset_stats(void);
    int init_queues(uint32_t ns_id,  uint32_t cudaDevice, 
                     uint64_t numQueues, uint64_t queueDepth);

    ~Controller();
};



using error = std::runtime_error;
using std::string;


inline void Controller::print_reset_stats(void) {
    cuda_err_chk(cudaMemcpy(&access_counter, d_ctrl_ptr, sizeof(cuda::atomic<uint64_t, cuda::thread_scope_device>), cudaMemcpyDeviceToHost));
    std::cout << "------------------------------------" << std::endl;
    std::cout << std::dec << "#SSDAccesses:\t" << access_counter << std::endl;

    cuda_err_chk(cudaMemset(d_ctrl_ptr, 0, sizeof(cuda::atomic<uint64_t, cuda::thread_scope_device>)));
}





inline Controller::Controller(const char* snvme_control_path, 
                                const char* pci_addr,
                                std::string mount_path,
                                uint32_t ns_id, 
                                uint32_t cudaDevice, 
                                uint64_t queueDepth, 
                                uint64_t numQueues)
    : ctrl(nullptr), deviceId(cudaDevice)
{
    
    int status;
    
    status = nvm_controller_init(&ctrl, snvme_control_path, pci_addr);
    if (status != 0){
        nvm_throw_error("Failed to nvm_controller_init", status);
    }

    status = init_queues(ns_id, cudaDevice, numQueues, queueDepth);
    if (status != 0){
        nvm_throw_error("Failed to init_queues", status);
    }
    
    page_size = ctrl->page_size;
    blk_size = disk.block_size;
    blk_size_log = std::log2(blk_size);

    dev_path = (char *)malloc(40 * sizeof(char));
    snprintf(dev_path, 40, "/dev/%s", disk.disk_name);
    
    dev_mount_path = mount_path;

    d_ctrl_buff = createBuffer(sizeof(Controller), cudaDevice);
    d_ctrl_ptr = d_ctrl_buff.get();
    cuda_err_chk(cudaMemcpy(d_ctrl_ptr, this, sizeof(Controller), cudaMemcpyHostToDevice));

    Host_file_system_int(dev_path, dev_mount_path.c_str());
}


inline int Controller::init_queues(uint32_t ns_id,  uint32_t cudaDevice, 
                                    uint64_t numQueues, uint64_t queueDepth){
    
    uint16_t max_queue = 75;
    void* devicePtr = nullptr;
    size_t i;
    int status;

    this->n_qps = std::min(max_queue, (uint16_t)numQueues);
    this->n_sqs = n_qps;
    this->n_cqs = n_qps;

    this->ctrl->cq_num = this->n_cqs;
    this->ctrl->sq_num = this->n_sqs;

    status = nvm_queue_set(ctrl, this->n_sqs + this->n_cqs);
    if (status != 0){
        printf("Failed to nvm_queue_set : %s\n", nvm_strerror(status));
        return EFAULT;
    }

    this->h_qps = (QueuePair**) malloc(sizeof(QueuePair) * n_qps);
    cuda_err_chk(cudaMalloc((void**)&this->d_qps, sizeof(QueuePair) * this->n_qps));

    for (i = 0; i < n_qps; i++) {
        h_qps[i] = new QueuePair(ctrl, deviceId, i, queueDepth);
    }

    status = nvm_device_init(ctrl);
    if (status != 0){
        printf("Failed to nvm_device_init : %s\n", nvm_strerror(status));
        return EFAULT;
    }

    this->disk.ns_id = ns_id;
    this->disk.page_size = ctrl->page_size;
    status = init_userioq_device(ctrl, this->h_qps, &this->disk);
    if (status != 0){
        printf("Failed to init userioq : %s\n", nvm_strerror(status));
    }

    this->n_qps = ctrl->nr_user_q;

    for (i = 0; i < this->n_qps; i++) {
        devicePtr = nullptr;
        cudaError_t err = cudaHostGetDevicePointer(&devicePtr, (void*) h_qps[i]->cq.db, 0);
        if (err != cudaSuccess) {
            printf("Failed to get device pointer %s\n", cudaGetErrorString(err));
            return -1;
        }
        h_qps[i]->cq.db = (volatile uint32_t*) devicePtr;

        // Get a valid device pointer for SQ doorbell
        err = cudaHostGetDevicePointer(&devicePtr, (void*) h_qps[i]->sq.db, 0);
        if (err != cudaSuccess) {
            printf("Failed to get device pointer %s\n", cudaGetErrorString(err));
            return -1;
        }
        h_qps[i]->sq.db = (volatile uint32_t*) devicePtr;
        
        cuda_err_chk(cudaMemcpy(d_qps + i, h_qps[i], sizeof(QueuePair), cudaMemcpyHostToDevice));
    }
    return 0;
}

inline Controller::~Controller()
{
    
    cudaFree(d_qps);
    for (size_t i = 0; i < n_qps; i++) {
        delete h_qps[i];
    }
    
    free(h_qps);
    int ret = Host_file_system_exit(dev_path);
    if(ret < 0)
        exit(-1);
    printf("Controller realease\n");
    nvm_ctrl_free(ctrl);

}


#endif
