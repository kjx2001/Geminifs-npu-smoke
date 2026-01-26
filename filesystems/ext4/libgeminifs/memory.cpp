#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <torch/all.h>
#include "buffer.h"
#include "memory.h"
/*----------------------ops wrapper-------------------*/

// __host__ struct geminifs_metadata* geminifs_get_metadata(int device_id) {
//     auto it = global_metadata.find(device_id);
//     if (it != global_metadata.end()) {
//         return it->second;
//     }
//     return nullptr;
// }

// __host__ struct geminifs_dma* geminifs_get_dma(const torch::Tensor &tensor) {
//     auto it = global_dam_ctx.find((uintptr_t)tensor.data_ptr());
//     if (it != global_dam_ctx.end()) {
//         return it->second;
//     }
//     return nullptr;
// }

// bool geminifs_get_dma_wrapper_cuda(const torch::Tensor& tensor) {
//     if (!is_device_pointer(tensor.data_ptr())) {
//         geminifs_error("geminifs_pin_memory_wrapper_cuda: tensor data pointer is not a device pointer\n");
//         return false;
//     }

//     if (!is_ptr_aligned(tensor.data_ptr())) {
//         geminifs_error("geminifs_pin_memory_wrapper_cuda: tensor data pointer %p is not aligned to GPU_PAGE_SIZE\n", tensor.data_ptr());
//         return false;
//     }

//     auto tensor_size = tensor.numel() * tensor.element_size();
//     if (!is_aligned(tensor_size)) {
//         geminifs_error("geminifs_pin_memory_wrapper_cuda: tensor size %ld is not aligned to GPU_PAGE_SIZE\n", tensor.numel() * tensor.element_size());
//         return false;
//     }
    
//     struct geminifs_metadata *metadata;
//     if ((metadata = geminifs_get_metadata(tensor.device().index())) == nullptr) {
//         geminifs_error("geminifs_pin_memory_wrapper_cuda: device %d has not been initialized\n", tensor.device().index());
//         return false;
//     }

//     DmaPtr dma_ptr = getDeviceDma(metadata->ctrls[0].get()->ctrl, 
//                                     tensor.data_ptr(), tensor_size, tensor.device().index());
//     if (dma_ptr == nullptr) {
//         geminifs_error("geminifs_pin_memory_wrapper_cuda: failed to get DMA pointer for tensor\n");
//         return false;
//     }

//     uint64_t *ioaddrs = nullptr;
//     if (!dma_ptr->contiguous){ // if the ioaddr of dma is not contiguous, we need to allocate a device buffer
//         cuda_check_error(cudaMalloc(&ioaddrs, sizeof(uint64_t) * dma_ptr->n_ioaddrs));
//         cuda_check_error(cudaMemcpy(ioaddrs, dma_ptr->ioaddrs, sizeof(uint64_t) * dma_ptr->n_ioaddrs, cudaMemcpyHostToDevice));    
//     }

//     geminifs_debug("geminifs_get_dma_wrapper_cuda: tensor data pointer %p, size %ld, ioaddr %lx, n_ioaddrs %ld, contiguous %d\n", 
//                     tensor.data_ptr(), tensor_size, dma_ptr->ioaddrs[0], dma_ptr->n_ioaddrs, dma_ptr->contiguous);

//     global_dam_ctx[(uint64_t)tensor.data_ptr()] = new geminifs_dma{
//         .ioaddrs = ioaddrs,
//         .dma_ptr = dma_ptr
//     };
//     return true;
// }

// bool geminifs_create_dma(const torch::Tensor& tensor) {
//     return geminifs_get_dma_wrapper_cuda(tensor);
// }