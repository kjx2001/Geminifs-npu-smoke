#ifndef __GEMINIFS_CUH__
#define __GEMINIFS_CUH__

// #include "geminifs.h"
// #include "ctrl.h"
// #include "buffer.h"
#include "file.cuh"
#include <cstdint>
#include <torch/all.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <memory>
#include <concepts>
#include "geminifs_nvme_file.h"

using ControllerPtr = std::shared_ptr<Controller>;





class NVMeController {
public:
    ControllerPtr controller;      // NVMe controller smart pointer
    std::unique_ptr<FileManager> file_manager; // File manager for this mount
    std::string mount_path;        // Mount path
 
    NVMeController(const ControllerPtr& ctrl, std::unique_ptr<FileManager> fm, const std::string& path)
        : controller(ctrl), file_manager(std::move(fm)), mount_path(path), is_initialized_(false) {}

    NVMeController(const nvme_ctrl_param& params);
    
    // Destructor
    ~NVMeController();

    void* g_open(std::string filename, size_t file_size, uint32_t o_flag);

    // Managed file operations - automatically handle file descriptor tracking
    host_fd_t host_file_create_managed(int block_size, size_t file_size, const std::string& filename);
    host_fd_t host_file_open_managed(const std::string& filepath, uint32_t o_flag);
    void host_file_close_managed(host_fd_t fd);

    // Check if controller is properly initialized
    bool is_initialized() const { return is_initialized_; }

private:
    // Private helper methods
    bool check_sys_config_exists();
    bool check_snvme_control_exists();
    ControllerPtr open_single_controller(const std::string& pci_addr, const nvme_ctrl_param& params);
    
    // Initialization state
    bool is_initialized_;
};
// Smart pointer for MountController
using NVMeControllerPtr = std::shared_ptr<NVMeController>;


struct geminifs_metadata{
    std::vector<ControllerPtr> ctrls;
    std::atomic<bool> is_init{false};
    thrust::device_ptr<GPUFilePool> global_pool; //device_ptr
    GPUPoolId pool_id;
    uint64_t file_size;
    uint64_t file_block_size;
    std::vector<cudaStream_t> streams;
};

struct geminifs_dma{
    uint64_t *ioaddrs;
    DmaPtr dma_ptr;
};


__host__ GPUFilePool* geminifs_init_fds(size_t nr_files, size_t file_size);
__host__ struct geminifs_metadata* geminifs_get_metadata(int device_id);
__host__ struct geminifs_dma* geminifs_get_dma(const torch::Tensor &tensor);
__host__ bool geminifs_create_dma(const torch::Tensor& tensor);


__host__ std::vector<ControllerPtr> 
host_open_ctrls(struct geminifs_ctrl_params *ctrl_params);

__host__ std::vector<ControllerPtr> 
geminifs_nvme_host_open_ctrls(struct geminifs_ctrl_params *ctrl_params);


__host__ void 
geminifs_nvme_host_close_ctrls(std::vector<ControllerPtr> &ctrls); 


// geminifs_batch_create(int nr_device, int nr_files, size_t block_size, size_t file_size, int  cudaDevice);

__host__ GPUFile* 
geminifs_file_batch_create(std::vector<ControllerPtr> &ctrls, GPUPoolId pool_id, int nr_files, 
                        size_t block_size, size_t file_size, int cudaDevice);

__host__ DmaPtr createDmaTest(int idx, size_t block_size, int device);

__host__
DmaPtr getdeviceDmaTest(int idx, void* buffer, size_t size, int cudaDevice);




/*------------------------------xfer function-----------------------------------*/
__host__ void 
geminifs_device_xfer(GPUFile *file, uint64_t *ioaddr, 
                        size_t file_offset, size_t nbytes,
                        enum FileXferType type);
                        
__host__ void
geminifs_device_batch_xfer(GPUFile *files, uint64_t **ioaddr, 
                            size_t file_offset, size_t nbytes,
                            enum FileXferType type, size_t batch_size);

__global__ void 
__geminifs_device_batch_xfer_once(GPUFilePool *global_pool, 
                            GPUFileId file_id, uint64_t ioaddr,
                            size_t file_offset, size_t nbytes, 
                            enum FileXferType type);


__global__ void 
__geminifs_device_batch_xfer_once2(GPUFilePool *global_pool, 
                            GPUFileId file_id, cuda::std::span<uint64_t> ioaddr,
                            size_t file_offset, size_t nbytes, 
                            enum FileXferType type);




/*------------------------------xfer test-----------------------------------*/
bool geminifs_init_fds_wrapper_cuda_test(int64_t nr_files, int64_t file_size, int64_t device_id, 
                                        const std::string& mount_path, const std::string& pcie_addr);

bool geminifs_device_xfer_wrapper_test(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    const torch::Tensor &key_cache,         // shape = [max_num_block, block_size, num_heads, head_size]
    const torch::Tensor &value_cache,       // shape = [max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, enum FileXferType type);

bool geminifs_get_dma_wrapper_cuda_test(const torch::Tensor& tensor);

bool geminifs_device_xfer_wrapper_test2(
    const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
    const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
    std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
    int64_t start_layer_idx, int64_t num_layers, enum FileXferType type);

#endif