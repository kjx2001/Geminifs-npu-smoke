#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <sys/types.h> 
#include <sys/stat.h>
#include <sys/resource.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include <ctrl.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#include <sys/file.h>
#include <fcntl.h>

#include <torch/library.h>
#include <torch/torch.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>          // 包含 device_of 和其他张量工具
#include <thrust/device_ptr.h>
#include <cuda/std/span>


#include "buffer.h"
#include "geminifs.h"
#include "geminifs_helper.h"
#include "torch/types.h"
#include "nvm_error.h"
#include "file.cuh"
#include "utils.cuh"
#include "geminifs.cuh"
#include "nvme_controller.cuh"
#include "geminifs_helper.h"
#include "gpu_controller.cuh"
static char snvme_control_path[] = "/dev/snvm_control";
static char sys_config_path[] = "/mnt/sys_GPU_NVMe_topology.json";

    
// Global helper functions for checking system components
static inline bool check_snvme_control_exists() {
    if (access(snvme_control_path, F_OK) != 0) {
        geminifs_error("SNVM control device '%s' does not exist. Please ensure the kernel module is properly installed.\n", snvme_control_path);
        return false;
    }
    return true;
}

static inline bool check_sys_config_exists() {
    if (access(sys_config_path, F_OK) != 0) {
        geminifs_error("Sys GPU-NVMe topology '%s' does not exist. Please ensure the kernel module is properly installed.\n", sys_config_path);
        return false;
    }
    return true;
}

// static inline void host_close_ctrls(struct geminifs_metadata *metadata){
//     for (auto &ctrl : metadata->ctrls) {
//         ctrl.reset();
//     }
//     metadata->ctrls.clear();
// }

// void host_close_all(){
//     for (auto &kv : global_metadata) {
//         auto metadata = kv.second;
//         if (metadata->is_init) {
//             host_close_ctrls(metadata);
//             metadata->is_init = false;
//         }
//     }
//     global_metadata.clear();
// }

// force close all controllers when the program exits
// __attribute__((destructor))
// static void clean_geminifs() {
//     host_close_all();
// }







struct DMAInfo{
    uint64_t *vaddr;
    uint64_t ioaddr_base;
    DmaPtr dma_ptr;
};










// static inline geminifs_metadata* __geminifs_init(struct geminifs_ctrl_params &ctrl_params, 
//                                                 size_t nr_files, size_t file_size, size_t file_block_size) {
//     assert(nr_files > 0);

//     // check current device 
//     int current_device;
//     cuda_check_error(cuda_getDevice(&current_device));
//     if (current_device != ctrl_params.cudaDevice) {
//         geminifs_warn("geminifs_init_fds_wrapper_cuda: current device %d is not the same as ctrl_params.cudaDevice %d\n", 
//                         current_device, ctrl_params.cudaDevice);
//         // cuda_check_error(cudaSetDevice(ctrl_params.cudaDevice));
//     }

//     geminifs_debug("geminifs_init_fds_wrapper_cuda: current device %d, ctrl_params.cudaDevice %d\n", 
//                         current_device, ctrl_params.cudaDevice);
//     file_size = ROUND_UP(file_size, file_block_size);
//     GPUPoolId this_pool_id = (GPUPoolId)time(NULL); //unique pool id
//     std::vector<ControllerPtr> ctrls = host_open_ctrls(&ctrl_params);
//     auto files = geminifs_batch_create(ctrls, this_pool_id, nr_files, 
//         file_block_size, file_size, ctrl_params.cudaDevice);

//     GPUFilePool *pool;
//     uint16_t *is_allocated;
//     cuda_check_error(cudaMalloc(&pool, sizeof(GPUFilePool)));
//     cuda_check_error(cudaMalloc(&is_allocated, nr_files  * sizeof(uint16_t)));
//     cuda_check_error(cudaMemset(is_allocated, 0x0, nr_files * sizeof(uint16_t)));
    
    
//     RUN_ON_DEVICE({
//         new (pool) GPUFilePool(files, is_allocated, file_size, file_block_size, nr_files);
//         pool->set_pool_id(this_pool_id);
//     });

//     std::vector<cudaStream_t> streams(32);
//     for (size_t i = 0; i < 32; i++) {
//         cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
//     }

//     return new geminifs_metadata{
//         .ctrls = std::move(ctrls),
//         .is_init = true,
//         .global_pool = thrust::device_pointer_cast(pool),
//         .pool_id = this_pool_id,
//         .file_size = file_size,
//         .file_block_size = file_block_size,
//         .streams = std::move(streams)
//     };
// }


// /*----------------------Xfer-------------------*/
// __global__ void 
// __geminifs_device_batch_xfer(GPUFilePool *global_pool, 
//                             cuda::std::span<GPUFileId> file_ids,
//                             cuda::std::span<uint64_t> ioaddr,
//                             size_t file_offset, 
//                             size_t nbytes, enum FileXferType type){
//     size_t nr_block = gridDim.x;
//     size_t nr_thread_per_block = blockDim.x;
//     assert(nr_block == file_ids.size());
//     assert(nr_thread_per_block == 32);
//     assert(nbytes % GPU_PAGE_SIZE == 0);
    
//     size_t nbytes__per_thread = std::max(nbytes / 32, GPU_PAGE_SIZE);

//     int lane = my_lane_id();
//     geminifs_debug("file_ids[blockIdx.x] %ld\n", file_ids[blockIdx.x]);
//     auto file = global_pool->get_file(file_ids[blockIdx.x]);
//     if (lane == 0) {
//         file->scatter_ioaddrs(ioaddr, file_offset, nbytes);
//     }

//     __syncwarp();
//     size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
//     if (lane * nbytes__per_thread  + nbytes__per_thread > nbytes) {
//         return;
//     }

//     if (type == FILE_XFER_READ) {
//         file->read_in(this_thread_file_offset, nbytes__per_thread);
//     } else {
//         file->write_out(this_thread_file_offset, nbytes__per_thread);
//     }
// }

// __global__ void 
// __geminifs_device_batch_xfer_once(GPUFilePool *global_pool, 
//                             GPUFileId file_id, uint64_t ioaddr,
//                             size_t file_offset, size_t nbytes, 
//                             enum FileXferType type){
//     size_t nr_block = gridDim.x;
//     size_t nr_thread_per_block = blockDim.x;
//     assert(nr_block == 1);
//     assert(nr_thread_per_block == 32);
//     assert(nbytes % GPU_PAGE_SIZE == 0);
    
//     size_t nbytes__per_thread = std::max(nbytes / 32, GPU_PAGE_SIZE);

//     int lane = my_lane_id();
//     auto file = global_pool->get_file(file_id);
//     if (lane == 0) {
//         file->scatter_ioaddrs(ioaddr, file_offset, nbytes);
//     }

//     __syncwarp();
//     size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
//     if (lane * nbytes__per_thread  + nbytes__per_thread > nbytes) {
//         return;
//     }

//     if (type == FILE_XFER_READ) {
//         file->read_in(this_thread_file_offset, nbytes__per_thread);
//     } else {
//         file->write_out(this_thread_file_offset, nbytes__per_thread);
//     }
// }

// __global__ void 
// __geminifs_device_batch_xfer_once2(GPUFilePool *global_pool, 
//                             GPUFileId file_id, cuda::std::span<uint64_t> ioaddr,
//                             size_t file_offset, size_t nbytes, 
//                             enum FileXferType type){
//     size_t nr_block = gridDim.x;
//     size_t nr_thread_per_block = blockDim.x;
//     assert(nr_block == 1);
//     assert(nr_thread_per_block == 32);
//     assert(nbytes % GPU_PAGE_SIZE == 0);
    
//     size_t nbytes__per_thread = std::max(nbytes / 32, GPU_PAGE_SIZE);

//     int lane = my_lane_id();
//     auto file = global_pool->get_file(file_id);
//     if (lane == 0) {
//         file->scatter_ioaddrs(ioaddr, file_offset, nbytes);
//     }

//     __syncwarp();
//     size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
//     if (lane * nbytes__per_thread  + nbytes__per_thread > nbytes) {
//         return;
//     }

//     if (type == FILE_XFER_READ) {
//         file->read_in(this_thread_file_offset, nbytes__per_thread);
//     } else {
//         file->write_out(this_thread_file_offset, nbytes__per_thread);
//     }
// }

// __global__ void 
// __geminifs_device_batch_xfer(GPUFilePool *global_pool, 
//                             cuda::std::span<GPUFileId> file_ids,
//                             cuda::std::span<uint64_t> block_ids,
//                             cuda::std::span<uint64_t> ioaddr, 
//                             size_t per_chuck_size, // per_chuck_size = chuck_size * block_size
//                             size_t file_offset, enum FileXferType type){
//     size_t nr_block = gridDim.x;
//     size_t nr_thread_per_block = blockDim.x;
//     assert(nr_block == file_ids.size());
//     assert(nr_thread_per_block == 32);
//     assert(per_chuck_size % global_pool->file_block_size == 0);
    
//     size_t nbytes__per_thread = std::max(per_chuck_size / 32, global_pool->file_block_size);

//     int lane = my_lane_id();
//     auto file = global_pool->get_file(file_ids[blockIdx.x]);
//     if (lane == 0) {
//         // fixme
//         file->scatter_ioaddrs(ioaddr, file_offset, per_chuck_size);
//         // file
//     }

//     __syncwarp();
//     size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
//     if (lane * nbytes__per_thread  + nbytes__per_thread > per_chuck_size) {
//         return;
//     }

//     if (type == FILE_XFER_READ) {
//         file->read_in(this_thread_file_offset, nbytes__per_thread);
//     } else {
//         file->write_out(this_thread_file_offset, nbytes__per_thread);
//     }
// }


// __global__ void 
// __geminifs_device_batch_xfer(GPUFilePool *global_pool, 
//                             cuda::std::span<GPUFileId> file_ids,
//                             cuda::std::span<uint64_t> block_ids,
//                             uint64_t ioaddr, size_t per_chuck_size, // per_chuck_size = chuck_size * block_size
//                             size_t file_offset, enum FileXferType type){
//     size_t nr_block = gridDim.x;
//     size_t nr_thread_per_block = blockDim.x;
//     assert(nr_block == file_ids.size());
//     assert(nr_thread_per_block == 32);
//     assert(per_chuck_size % global_pool->file_block_size == 0);
    
//     size_t nbytes__per_thread = std::max(per_chuck_size / 32, global_pool->file_block_size);
//     int lane = my_lane_id();
//     auto file = global_pool->get_file(file_ids[blockIdx.x]);
    
//     if (lane == 0) {
//         auto this_block_ioaddr = ioaddr + block_ids[blockIdx.x] * per_chuck_size;
//         file->scatter_ioaddrs(this_block_ioaddr, file_offset, per_chuck_size);
//     }
//     __syncwarp();


//     size_t this_thread_file_offset = file_offset + lane * nbytes__per_thread;
//     if (lane * nbytes__per_thread  + nbytes__per_thread > per_chuck_size) {
//         return;
//     }
//     if (type == FILE_XFER_READ) {
//         // geminifs_debug("read_in, file[%p]:this_thread_file_offset %ld, nbytes__per_thread %ld\n", file, this_thread_file_offset, nbytes__per_thread);
//         file->read_in(this_thread_file_offset, nbytes__per_thread);
//         // geminifs_debug("read_in done\n");
//     } else {
//         file->write_out(this_thread_file_offset, nbytes__per_thread);
//     }
//     global_pool->put_file(file_ids[blockIdx.x]);
// }







// static inline bool __geimifs_device_one_layer_xfer(
//     const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
//     const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
//     const torch::Tensor &key_cache,         // shape = [max_num_block, block_size, num_heads, head_size]
//     const torch::Tensor &value_cache,       // shape = [max_num_block, block_size, num_heads, head_size]
//     int64_t start_layer_idx, enum FileXferType type,
//     struct geminifs_metadata *metadata, 
//     cudaStream_t stream) {
    
//     int max_num_block = key_cache.size(0);
//     int block_size = key_cache.size(1);
//     int num_heads = key_cache.size(2);
//     int head_size = key_cache.size(3);
    
//     uint64_t block_nbytes = block_size * num_heads * head_size * key_cache.element_size();
//     uint64_t layer_stride = 2 * block_nbytes;
//     uint64_t key_file_offset = start_layer_idx * layer_stride;
//     uint64_t value_file_offset = key_file_offset + block_nbytes;

//     if (block_nbytes & (metadata->file_block_size - 1)) { // to avoid xfer to other page
//         geminifs_error("block_nbytes %ld is not aligned to file block size\n", block_nbytes);
//         return false;
//     }

//     struct geminifs_dma *key_dma_ctx, *value_dma_ctx;
//     if ((key_dma_ctx = geminifs_get_dma(key_cache)) == nullptr) {
//         geminifs_error("geminifs_device_xfer_wrapper_cuda: key_cache.data_ptr() %p has not been initialized\n", key_cache.data_ptr());
//         return false;
//     }

//     if ((value_dma_ctx = geminifs_get_dma(value_cache)) == nullptr) {
//         geminifs_error("geminifs_device_xfer_wrapper_cuda: value_cache.data_ptr() %p has not been initialized\n", value_cache.data_ptr());
//         return false;
//     }

//     dim3 grid(cached_file_ids.numel());
//     dim3 block(32);
//     const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));

//     cuda::std::span<GPUFileId> file_ids = {(GPUFileId *)cached_file_ids.data_ptr(), (size_t)cached_file_ids.numel()};
//     cuda::std::span<uint64_t> block_ids = {(uint64_t *)inner_block_ids.data_ptr(), (size_t)inner_block_ids.numel()};

//     auto * pool = metadata->global_pool.get();
//     if (key_dma_ctx->dma_ptr->contiguous) {
//         __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
//                         (pool, file_ids, block_ids, key_dma_ctx->dma_ptr->ioaddrs[0], 
//                             block_nbytes, key_file_offset, type);
//     }
    
//     if (value_dma_ctx->dma_ptr->contiguous) {
//         __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
//                         (pool, file_ids, block_ids, value_dma_ctx->dma_ptr->ioaddrs[0], 
//                             block_nbytes, value_file_offset, type);
//     }
    
//     if (!key_dma_ctx->dma_ptr->contiguous) { // assert that ioaddrs is not null
//         __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
//                         (pool, file_ids, block_ids, 
//                         {key_dma_ctx->ioaddrs, key_dma_ctx->dma_ptr->n_ioaddrs}, 
//                         block_nbytes, key_file_offset, type);
//     }   
    
//     if (!value_dma_ctx->dma_ptr->contiguous) { // assert that ioaddrs is not null
//         __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
//                         (pool, file_ids, block_ids, 
//                         {value_dma_ctx->ioaddrs, value_dma_ctx->dma_ptr->n_ioaddrs}, 
//                         block_nbytes, value_file_offset, type);
//     }

//     return true;
// }

// static inline bool geminifs_device_mutiple_layer_xfer(
//     const torch::Tensor& cached_file_ids,  // shape = [num_cached_files,]
//     const torch::Tensor& inner_block_ids,  // shape = [num_cached_files,]
//     const std::vector<torch::Tensor>& key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
//     const std::vector<torch::Tensor>& value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
//     int64_t start_layer_idx, int64_t num_layers,
//     enum FileXferType type) {

//     // Input validation
//     if (num_layers <= 0) {
//         geminifs_error("geminifs_device_mutiple_layer_xfer: num_layers must be positive, got %lld\n", num_layers);
//         return false;
//     }
//     // printf("num_layers %lld, key_caches.size() %zu, value_caches.size() %zu\n", num_layers, key_caches.size(), value_caches.size());
//     if (key_caches.size() != num_layers || value_caches.size() != num_layers) {
//         geminifs_error("geminifs_device_mutiple_layer_xfer: key_caches (%zu) or value_caches (%zu) size mismatch with num_layers (%lld)\n", key_caches.size(), value_caches.size(), num_layers);
//         return false;
//     }
//     if (cached_file_ids.numel() != inner_block_ids.numel()) {
//         geminifs_error("geminifs_device_mutiple_layer_xfer: cached_file_ids and inner_block_ids must have same number of elements (%lld vs %lld)\n", cached_file_ids.numel(), inner_block_ids.numel());
//         return false;
//     }
//     if (key_caches.empty() || !key_caches[0].is_cuda()) {
//         geminifs_error("geminifs_device_mutiple_layer_xfer: key_caches are empty or not on a CUDA device.\n");
//         return false;
//     }

//     auto device = key_caches[0].device().index();
//     struct geminifs_metadata* metadata = geminifs_get_metadata(device);

//     if (metadata == nullptr) {
//         geminifs_error("geminifs_device_mutiple_layer_xfer: device %d has not been initialized.\n", device);
//         return false;
//     }

//     const size_t num_available_streams = metadata->streams.size();
//     if (num_available_streams == 0) {
//         geminifs_error("geminifs_device_mutiple_layer_xfer: No CUDA streams available in metadata for device %d.\n", device);
//         return false;
//     }

//     // Create a vector to hold CUDA events for each layer's transfer
//     std::vector<cudaEvent_t> completion_events(num_layers);
//     for (int i = 0; i < num_layers; ++i) {
//         // Ensure event creation is successful
//         cudaError_t err = cudaEventCreate(&completion_events[i]);
//         if (err != cudaSuccess) {
//             geminifs_error("geminifs_device_mutiple_layer_xfer: Failed to create CUDA event %d: %s\n", i, cudaGetErrorString(err));
//             // Clean up already created events before returning
//             for (int j = 0; j < i; ++j) {
//                 cudaEventDestroy(completion_events[j]);
//             }
//             return false;
//         }
//     }

//     // Launch transfers on different streams
//     for (int i = 0; i < num_layers; ++i) {
//         cudaStream_t stream = metadata->streams[i % num_available_streams];

//         // Perform the transfer for the current layer
//         __geimifs_device_one_layer_xfer(
//             cached_file_ids,
//             inner_block_ids,
//             key_caches[i],
//             value_caches[i],
//             i + start_layer_idx, // Absolute layer index
//             type,
//             metadata,
//             stream
//         );

//         // Record an event in the stream after the transfer is submitted
//         cudaError_t err = cudaEventRecord(completion_events[i], stream);
//         if (err != cudaSuccess) {
//             geminifs_error("geminifs_device_mutiple_layer_xfer: Failed to record CUDA event %d for stream %p: %s\n", i, (void*)stream, cudaGetErrorString(err));
//             // This is a critical error, might need more robust cleanup
//             for (auto& event : completion_events) { // Clean up all events
//                 if (event) cudaEventDestroy(event);
//             }
//             return false;
//         }
//     }


//     return true; // All transfers are guaranteed to be complete on the GPU
// }


// static inline bool geminifs_device_xfer_wrapper_cuda(
//     const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
//     const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
//     const torch::Tensor &key_cache,         // shape = [max_num_block, block_size, num_heads, head_size]
//     const torch::Tensor &value_cache,       // shape = [max_num_block, block_size, num_heads, head_size]
//     int64_t start_layer_idx, enum FileXferType type) {
    
//     int max_num_block = key_cache.size(0);
//     int block_size = key_cache.size(1);
//     int num_heads = key_cache.size(2);
//     int head_size = key_cache.size(3);
//     int device = key_cache.device().index();

//     // layer -> file offset, block id -> key/value cache offset
//     uint64_t block_nbytes = block_size * num_heads * head_size * key_cache.element_size();
//     uint64_t layer_stride = 2 * block_nbytes;
//     uint64_t key_file_offset = start_layer_idx * layer_stride;
//     uint64_t value_file_offset = key_file_offset + block_nbytes;

//     struct geminifs_metadata *metadata;
//     if ((metadata = geminifs_get_metadata(device)) == nullptr) {
//         geminifs_error("geminifs_device_xfer_wrapper_cuda: device %d has not been initialized\n", device);
//         return false;
//     }
    
//     if (block_nbytes & (metadata->file_block_size - 1)) { // to avoid xfer to other page
//         geminifs_error("block_nbytes %ld is not aligned to file block size\n", block_nbytes);
//         return false;
//     }
    
//     auto * pool = metadata->global_pool.get();
//     is_device_pointer(pool, "global_pool must be device ptr");

//     if (!is_ptr_aligned(key_cache.data_ptr())) {
//         geminifs_error("key_cache.data_ptr() %p, block_nbytes %ld\n", key_cache.data_ptr(), block_nbytes);
//         return false;
//     }

//     if (!is_ptr_aligned(value_cache.data_ptr())) {
//         geminifs_error("value_cache.data_ptr() %p, block_nbytes %ld\n", value_cache.data_ptr(), block_nbytes);
//         return false;
//     }

//     struct geminifs_dma *key_dma_ctx, *value_dma_ctx;
//     if ((key_dma_ctx = geminifs_get_dma(key_cache)) == nullptr) {
//         geminifs_error("geminifs_device_xfer_wrapper_cuda: key_cache.data_ptr() %p has not been initialized\n", key_cache.data_ptr());
//         return false;
//     }

//     if ((value_dma_ctx = geminifs_get_dma(value_cache)) == nullptr) {
//         geminifs_error("geminifs_device_xfer_wrapper_cuda: value_cache.data_ptr() %p has not been initialized\n", value_cache.data_ptr());
//         return false;
//     }

//     dim3 grid(cached_file_ids.numel());
//     dim3 block(32);
//     const at::cuda::OptionalCUDAGuard device_guard(device_of(key_cache));
//     // const at::cuda::OptionalCUDAGuard device_guard(device_of(value_cache));
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     // geminifs_debug("tensor info: key_cache.data_ptr() %p, value_cache.data_ptr() %p, block_nbytes %ld, key_file_offset %ld, value_file_offset %ld\n", 
//     //                 key_cache.data_ptr(), value_cache.data_ptr(), block_nbytes, key_file_offset, value_file_offset);

//     cuda::std::span<GPUFileId> file_ids = {(GPUFileId *)cached_file_ids.data_ptr(), (size_t)cached_file_ids.numel()};
//     cuda::std::span<uint64_t> block_ids = {(uint64_t *)inner_block_ids.data_ptr(), (size_t)inner_block_ids.numel()};
//     if (key_dma_ctx->dma_ptr->contiguous) {
//         // geminifs_debug("key dma is contiguous, ready to transfer key\n");
//         __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
//                         (pool, file_ids, block_ids, key_dma_ctx->dma_ptr->ioaddrs[0], 
//                             block_nbytes, key_file_offset, type);
//         // geminifs_debug("key transfer done\n");
//     }

//     if (value_dma_ctx->dma_ptr->contiguous) {
//         // geminifs_debug("value dma is contiguous, ready to transfer value\n");
//         __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
//                         (pool, file_ids, block_ids, value_dma_ctx->dma_ptr->ioaddrs[0], 
//                             block_nbytes, value_file_offset, type);
//         // geminifs_debug("value transfer done\n");
//     }

//     if (!key_dma_ctx->dma_ptr->contiguous) { // assert that ioaddrs is not null
//         __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
//                         (pool, file_ids, block_ids, 
//                         {key_dma_ctx->ioaddrs, key_dma_ctx->dma_ptr->n_ioaddrs}, 
//                         block_nbytes, key_file_offset, type);
//     }

//     if (!value_dma_ctx->dma_ptr->contiguous) { // assert that ioaddrs is not null
//         __geminifs_device_batch_xfer<<<grid, block, 0, stream>>>
//                         (pool, file_ids, block_ids, 
//                         {value_dma_ctx->ioaddrs, value_dma_ctx->dma_ptr->n_ioaddrs}, 
//                         block_nbytes, value_file_offset, type);
//     }
//     cudaDeviceSynchronize();
//     // geminifs_debug("xfer done\n");
//     return true;
// }

// bool batch_write_direct(const torch::Tensor& cached_file_ids,   //shape = [num_cached_files,]
//                         const torch::Tensor& inner_block_ids,  // shape = [num_cached_files,]
//                         const torch::Tensor& key_cache,               // shape = [max_num_block, block_size, num_heads, head_size]
//                         const torch::Tensor& value_cache,             // shape = [max_num_block, block_size, num_heads, head_size]
//                         int64_t start_layer_idx) {

//     return geminifs_device_xfer_wrapper_cuda(cached_file_ids, inner_block_ids, 
//                                             key_cache, value_cache, 
//                                             start_layer_idx, FILE_XFER_WRITE);
// }

// bool batch_read_direct(const torch::Tensor& cached_file_ids,   //shape = [num_cached_files,]
//                         const torch::Tensor& inner_block_ids,  // shape = [num_cached_files,]
//                         const torch::Tensor& key_cache,               // shape = [max_num_block, block_size, num_heads, head_size]
//                         const torch::Tensor& value_cache,             // shape = [max_num_block, block_size, num_heads, head_size]
//                         int64_t start_layer_idx) {

//     return geminifs_device_xfer_wrapper_cuda(cached_file_ids, inner_block_ids, 
//                                             key_cache, value_cache, 
//                                             start_layer_idx, FILE_XFER_READ);
// }

// bool geminifs_device_mutiple_layer_read_wrapper_cuda(
//                                         const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
//                                         const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
//                                         const std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
//                                         const std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
//                                         int64_t start_layer_idx, int64_t num_layers) {
//     return geminifs_device_mutiple_layer_xfer(cached_file_ids, inner_block_ids, 
//                                             key_caches, value_caches, 
//                                             start_layer_idx, num_layers, FILE_XFER_READ);
// }

// bool geminifs_device_mutiple_layer_write_wrapper_cuda(
//                                         const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
//                                         const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
//                                         const std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
//                                         const std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
//                                         int64_t start_layer_idx, int64_t num_layers) {
//     return geminifs_device_mutiple_layer_xfer(cached_file_ids, inner_block_ids, 
//                                             key_caches, value_caches,   
//                                             start_layer_idx, num_layers, FILE_XFER_WRITE);
// }

// bool geminifs_init_fds_wrapper_cuda(const torch::Tensor& file_meta, 
//                                     const std::string& mount_path, 
//                                     const std::string& pcie_addr) {
    

//     TORCH_CHECK(file_meta.dim() == 1 && file_meta.numel() >= 2, "file_meta should be a 1D tensor with at least 2 elements");
//     TORCH_CHECK(file_meta.scalar_type() == torch::kUInt64, "file_meta should have kUInt64 type");
    
    
//     torch::Tensor file_meta_host = file_meta.cpu();
//     const uint64_t* file_meta_ptr = file_meta_host.data_ptr<uint64_t>();
//     uint64_t nr_files = file_meta_ptr[0];
//     uint64_t file_size = file_meta_ptr[1];
//     int64_t device_id = file_meta.device().index();

//     geminifs_debug("geminifs_init_fds_wrapper_cuda: nr_files %ld, file_size %ld, device_id %ld\n", 
//                     nr_files, file_size, device_id);

//     assert(nr_files > 0 && file_size > 0 && device_id >= 0);

//     std::vector<string> pcie_addr_vec = split(pcie_addr, ',');
//     auto size = pcie_addr_vec.size();
//     if (size == 0 || (size & (size - 1)) != 0) {
//         geminifs_error("geminifs_init_fds_wrapper_cuda: pcie_addr %s is not a power of 2\n", pcie_addr.c_str());
//         return false;
//     }
    
//     struct geminifs_ctrl_params ctrl_params = {
//         .mount_path = mount_path,
//         .snvme_control_path = "/dev/snvm_control",
//         .pci_addr = pcie_addr_vec,
//         .cudaDevice = (int)device_id,
//         .ns_id = 1,
//         .queueDepth = 1024,
//         .numQueues = 64
//     };

//     if (geminifs_get_metadata(ctrl_params.cudaDevice) != nullptr) {
//         geminifs_error("geminifs_init_fds_wrapper_cuda: device %d has been initialized\n", ctrl_params.cudaDevice);
//         return false;
//     } else {
//         global_metadata[ctrl_params.cudaDevice] = __geminifs_init(ctrl_params, nr_files,
//                                                                     file_size,  __4KB__);
//     }

//     return true;
// }




// bool geminifs_init_fds_wrapper_cuda_test(int64_t nr_files, int64_t file_size, int64_t device_id, 
//                                             const std::string& mount_path, const std::string& pcie_addr){
//     // 指定 TensorOptions 来设置设备和数据类型为 int64
//     torch::Tensor file_meta = torch::tensor({nr_files, file_size}, 
//                                             torch::TensorOptions().device(torch::kCUDA, device_id).dtype(torch::kUInt64));
//     return geminifs_init_fds_wrapper_cuda(file_meta, mount_path, pcie_addr);
// }

// bool geminifs_device_xfer_wrapper_test(
//     const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
//     const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
//     const torch::Tensor &key_cache,         // shape = [max_num_block, block_size, num_heads, head_size]
//     const torch::Tensor &value_cache,       // shape = [max_num_block, block_size, num_heads, head_size]
//     int64_t start_layer_idx, enum FileXferType type) {
    
//     return geminifs_device_xfer_wrapper_cuda(cached_file_ids, inner_block_ids, 
//                                             key_cache, value_cache,
//                                             start_layer_idx, type);
// }

// bool geminifs_device_xfer_wrapper_test2(
//     const torch::Tensor &cached_file_ids,   //shape = [num_cached_files,]
//     const torch::Tensor &inner_block_ids,  // shape = [num_cached_files,]
//     std::vector<torch::Tensor> &key_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
//     std::vector<torch::Tensor> &value_caches, // shape = List[num_layers, max_num_block, block_size, num_heads, head_size]
//     int64_t start_layer_idx, int64_t num_layers, enum FileXferType type) {

//     return geminifs_device_mutiple_layer_xfer(cached_file_ids, inner_block_ids, 
//                                             key_caches, value_caches,
//                                             start_layer_idx, num_layers, type);
// }
// 保持兼容性，之后可以去除
__host__ GPUControllerPtr geminifs_create_gpu_controller(int device_id, const std::string& mount_base_path) {
    auto gpu_controller = std::make_shared<GPUController>(device_id, mount_base_path);
    
    if (!gpu_controller->isInitialized()) {
        geminifs_error("geminifs_create_gpu_controller: Failed to initialize GPU controller for device %d\n", device_id);
        return nullptr;
    }
    
    // Register with the global registry
    auto& registry = GPUControllerRegistry::getInstance();
    if (!registry.registerGPUController(device_id, gpu_controller)) {
        geminifs_error("geminifs_create_gpu_controller: Failed to register GPU controller for device %d\n", device_id);
        return nullptr;
    }
    
    geminifs_debug("geminifs_create_gpu_controller: Successfully created and registered GPU controller for device %d\n", device_id);
    return gpu_controller;
}

/**
 * Create and register a GPU controller for a specific device
 */
__host__ GPUControllerPtr GeminiFS::geminifs_create_gpu_controller(int device_id, const std::string& mount_base_path) {
    auto gpu_controller = std::make_shared<GPUController>(device_id, mount_base_path);
    
    if (!gpu_controller->isInitialized()) {
        geminifs_error("geminifs_create_gpu_controller: Failed to initialize GPU controller for device %d\n", device_id);
        return nullptr;
    }
    
    // Register with the global registry
    auto& registry = GPUControllerRegistry::getInstance();
    if (!registry.registerGPUController(device_id, gpu_controller)) {
        geminifs_error("geminifs_create_gpu_controller: Failed to register GPU controller for device %d\n", device_id);
        return nullptr;
    }
    
    geminifs_debug("geminifs_create_gpu_controller: Successfully created and registered GPU controller for device %d\n", device_id);
    return gpu_controller;
}

__host__ GPUControllerPtr geminifs_get_gpu_controller(int device_id) {
    auto& registry = GPUControllerRegistry::getInstance();
    return registry.getGPUController(device_id);
}

/**
 * Get GPU controller for a specific device
 */
__host__ GPUControllerPtr GeminiFS::geminifs_get_gpu_controller(int device_id) {
    auto& registry = GPUControllerRegistry::getInstance();
    return registry.getGPUController(device_id);
}

/**
 * Add an NVMe controller to a GPU controller
 */
__host__ bool geminifs_add_nvme_to_gpu(int device_id, const nvme_ctrl_param& params) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_add_nvme_to_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    // Create modified parameters with mount path under GPU controller
    nvme_ctrl_param modified_params = params;
    std::filesystem::path gpu_mount_path = gpu_controller->getMountBasePath();
    std::filesystem::path nvme_mount_path = gpu_mount_path / ("nvme-" + params.pci_addr);
    modified_params.mount_path = nvme_mount_path.string();
    
    geminifs_debug("geminifs_add_nvme_to_gpu: Creating NVMe controller with mount path '%s' under GPU path '%s'\n", 
                   modified_params.mount_path.c_str(), gpu_mount_path.c_str());
    
    // Create NVMe controller with modified mount path
    auto nvme_controller = std::make_shared<NVMeController>(modified_params);
    if (!nvme_controller->is_initialized()) {
        geminifs_error("geminifs_add_nvme_to_gpu: Failed to initialize NVMe controller\n");
        return false;
    }
    
    // Add to GPU controller
    if (!gpu_controller->addNVMeController(nvme_controller)) {
        geminifs_error("geminifs_add_nvme_to_gpu: Failed to add NVMe controller to GPU %d\n", device_id);
        return false;
    }
    
    geminifs_debug("geminifs_add_nvme_to_gpu: Successfully added NVMe controller to GPU %d with mount path '%s'\n", 
                   device_id, modified_params.mount_path.c_str());
    return true;
}

/**
 * Add an NVMe controller to a GPU controller
 */
__host__ bool GeminiFS::geminifs_add_nvme_to_gpu(int device_id, const nvme_ctrl_param& params) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_add_nvme_to_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    // Create modified parameters with mount path under GPU controller
    nvme_ctrl_param modified_params = params;
    std::filesystem::path gpu_mount_path = gpu_controller->getMountBasePath();
    std::filesystem::path nvme_mount_path = gpu_mount_path / ("nvme-" + params.pci_addr);
    modified_params.mount_path = nvme_mount_path.string();
    
    geminifs_debug("geminifs_add_nvme_to_gpu: Creating NVMe controller with mount path '%s' under GPU path '%s'\n", 
                   modified_params.mount_path.c_str(), gpu_mount_path.c_str());
    
    // Create NVMe controller with modified mount path
    auto nvme_controller = std::make_shared<NVMeController>(modified_params);
    if (!nvme_controller->is_initialized()) {
        geminifs_error("geminifs_add_nvme_to_gpu: Failed to initialize NVMe controller\n");
        return false;
    }
    
    // Add to GPU controller
    if (!gpu_controller->addNVMeController(nvme_controller)) {
        geminifs_error("geminifs_add_nvme_to_gpu: Failed to add NVMe controller to GPU %d\n", device_id);
        return false;
    }
    
    geminifs_debug("geminifs_add_nvme_to_gpu: Successfully added NVMe controller to GPU %d with mount path '%s'\n", 
                   device_id, modified_params.mount_path.c_str());
    return true;
}

__host__ bool geminifs_register_tensor_with_gpu(const torch::Tensor& tensor, uint64_t granularity) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_register_tensor_with_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    return gpu_controller->registerTensorMemory(tensor, granularity);
}

/**
 * Register tensor memory with GPU controller
 */
__host__ bool GeminiFS::geminifs_register_tensor_with_gpu(const torch::Tensor& tensor, uint64_t granularity) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_register_tensor_with_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    return gpu_controller->registerTensorMemory(tensor, granularity);
}

__host__ bool geminifs_unregister_tensor_from_gpu(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_unregister_tensor_from_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    return gpu_controller->unregisterTensorMemory(tensor.data_ptr());
}

/**
 * Unregister tensor memory from GPU controller
 */
__host__ bool GeminiFS::geminifs_unregister_tensor_from_gpu(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_unregister_tensor_from_gpu: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    return gpu_controller->unregisterTensorMemory(tensor.data_ptr());
}

__host__ struct geminifs_dma* geminifs_get_tensor_dma_from_gpu(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_get_tensor_dma_from_gpu: No GPU controller found for device %d\n", device_id);
        return nullptr;
    }
    
    return gpu_controller->getDMAContext(tensor.data_ptr());
}

/**
 * Get DMA context from GPU controller
 */
__host__ struct geminifs_dma* GeminiFS::geminifs_get_tensor_dma_from_gpu(const torch::Tensor& tensor) {
    int device_id = tensor.device().index();
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    
    if (!gpu_controller) {
        geminifs_error("geminifs_get_tensor_dma_from_gpu: No GPU controller found for device %d\n", device_id);
        return nullptr;
    }
    
    return gpu_controller->getDMAContext(tensor.data_ptr());
}

/**
 * Open file using GPU controller
 */
__host__ void* GeminiFS::geminifs_gpu_open_file(int device_id, GPUFileId gpu_file_id, size_t file_size, uint32_t o_flag) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_gpu_open_file: No GPU controller found for device %d\n", device_id);
        return nullptr;
    }
    
    return gpu_controller->openFile(gpu_file_id, file_size, o_flag, nvme_params_, gpu_file_manager_);
}

// --- NVMe_Link LRU cache helpers ---
bool GeminiFS::get_links_cached(GPUFileId file_id, std::vector<NVMe_Link>& out_links) {
    auto it = links_cache_.find(file_id);
    if (it == links_cache_.end()) return false;
    out_links = it->second.links;
    touch_links_cache(file_id);
    return true;
}

void GeminiFS::put_links_cache(GPUFileId file_id, const std::vector<NVMe_Link>& links) {
    size_t bytes = links.size() * sizeof(NVMe_Link);
    auto it = links_cache_.find(file_id);
    if (it != links_cache_.end()) {
        // update existing
        links_cache_bytes_ -= it->second.bytes;
        it->second.links = links;
        it->second.bytes = bytes;
        touch_links_cache(file_id);
    } else {
        links_cache_lru_.push_front(file_id);
        links_cache_[file_id] = {links, bytes, links_cache_lru_.begin()};
        links_cache_bytes_ += bytes;
    }
    evict_links_cache_if_needed();
}

void GeminiFS::touch_links_cache(GPUFileId file_id) {
    auto it = links_cache_.find(file_id);
    if (it == links_cache_.end()) return;
    links_cache_lru_.erase(it->second.lru_it);
    links_cache_lru_.push_front(file_id);
    it->second.lru_it = links_cache_lru_.begin();
}

void GeminiFS::evict_links_cache_if_needed() {
    while (links_cache_bytes_ > LINKS_CACHE_MAX_BYTES && !links_cache_lru_.empty()) {
        GPUFileId victim = links_cache_lru_.back();
        links_cache_lru_.pop_back();
        auto it = links_cache_.find(victim);
        if (it != links_cache_.end()) {
            links_cache_bytes_ -= it->second.bytes;
            links_cache_.erase(it);
        }
    }
}

__host__ bool GeminiFS::geminifs_GPU_read_kernel(torch::Tensor& tensor, GPUFileId gpu_file_id, GPUControllerPtr gpu_controller) {
    if (!gpu_controller || !gpu_controller->isInitialized()) {
         geminifs_error("GPU_read_kernel: GPU controller is not initialized\n");
          return false;
    }

    if (!tensor.is_cuda()) { 
        geminifs_error("GPU_read_kernel: tensor must be on CUDA device\n");
         return false;
    }

    GPUFileDesc desc;
    if (!gpu_file_manager_.getGPUFileDesc(gpu_file_id, desc)) { 
        geminifs_error("GPU_read_kernel: GPUFileId %lu not found\n", gpu_file_id); 
        return false;
    }

    // Do not auto-register here; only validate tensor already registered
    auto dma_ctx = gpu_controller->getDMAContext(tensor.data_ptr());
    if (!dma_ctx) {
        geminifs_error("GPU_read_kernel: tensor not registered; call geminifs_register_tensor_with_gpu before IO\n");
        return false;
    }

    // 优先从缓存取链接，未命中则用 getLinksForFile 一次性拉取并缓存
    std::vector<NVMe_Link> links;
    if (!get_links_cached(gpu_file_id, links)) {
        if (!gpu_file_manager_.getLinksForFile(gpu_file_id, links)) {
            geminifs_error("GPU_read_kernel: failed to get links for file %lu\n", gpu_file_id);
            return false;
        }
        put_links_cache(gpu_file_id, links);
    }

    auto* device_view = gpu_controller->getMemoryMapper()->getDeviceViewPtr();
    if (!device_view) {
        geminifs_error("GPU_read_kernel: device view is null\n");
        return false;
    }

    // 准备 NVMe_File* 列表（与 links 对齐）
    std::vector<NVMe_File*> nvme_fds;
    nvme_fds.reserve(links.size());
    for (const auto& link : links) {
        auto nvme_ctrl = gpu_controller->getNVMeController(link.controller_index);
        if (!nvme_ctrl) {
            geminifs_error("GPU_read_kernel: NVMe controller %zu not found\n", link.controller_index);
            return false;
        }
        void* device_fd = nvme_ctrl->g_open(std::string(link.name), link.file_size, O_DEVICE);
        if (!device_fd) {
            geminifs_error("GPU_read_kernel: g_open failed for %s\n", link.name);
            return false;
        }
        nvme_fds.push_back(reinterpret_cast<NVMe_File*>(device_fd));
    }

    // 拷贝 NVMe_File** 到设备
    NVMe_File** d_fds = nullptr;
    size_t num_fds = nvme_fds.size();
    cudaError_t err = cudaMalloc(&d_fds, num_fds * sizeof(NVMe_File*));
    if (err != cudaSuccess) {
        geminifs_error("GPU_read_kernel: cudaMalloc d_fds failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(d_fds, nvme_fds.data(), num_fds * sizeof(NVMe_File*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_fds);
        geminifs_error("GPU_read_kernel: cudaMemcpy d_fds failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    // 启动多FD内核（内核内部根据 tid%num_fds 分发并触发 v3 批量）
    uint64_t tensor_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());
    size_t offset = 0;
    size_t len = static_cast<size_t>(tensor.numel()) * static_cast<size_t>(tensor.element_size());
    GPU_Read_kernel_multi<<<1,1>>>(d_fds, static_cast<uint32_t>(num_fds), tensor_ptr, offset, len, device_view);
    err = cudaDeviceSynchronize();

    cudaFree(d_fds);
    if (err != cudaSuccess) {
        geminifs_error("GPU_read_kernel: GPU_Read_kernel_multi failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

__host__ bool GeminiFS::geminifs_GPU_write_kernel(const torch::Tensor& tensor, GPUFileId gpu_file_id, GPUControllerPtr gpu_controller) {
    if (!gpu_controller || !gpu_controller->isInitialized()) {
        geminifs_error("GPU_write_kernel: GPU controller is not initialized\n");
        return false;
    }
    if (!tensor.is_cuda()) {
        geminifs_error("GPU_write_kernel: tensor must be on CUDA device\n");
        return false;
    }
    GPUFileDesc desc;
    if (!gpu_file_manager_.getGPUFileDesc(gpu_file_id, desc)) {
        geminifs_error("GPU_write_kernel: GPUFileId %lu not found\n", gpu_file_id);
        return false;
    }

    // Do not auto-register here; only validate tensor already registered
    auto dma_ctx = gpu_controller->getDMAContext(const_cast<void*>(tensor.data_ptr()));
    if (!dma_ctx) {
        geminifs_error("GPU_write_kernel: tensor not registered; call geminifs_register_tensor_with_gpu before IO\n");
        return false;
    }

    std::vector<NVMe_Link> links;
    if (!get_links_cached(gpu_file_id, links)) {
        if (!gpu_file_manager_.getLinksForFile(gpu_file_id, links)) {
            geminifs_error("GPU_write_kernel: failed to get links for file %lu\n", gpu_file_id);
            return false;
        }
        put_links_cache(gpu_file_id, links);
    }

    auto* device_view = gpu_controller->getMemoryMapper()->getDeviceViewPtr();
    if (!device_view) {
        geminifs_error("GPU_write_kernel: device view is null\n");
        return false;
    }

    // 准备 NVMe_File* 列表（与 links 对齐）
    std::vector<NVMe_File*> nvme_fds;
    nvme_fds.reserve(links.size());
    for (const auto& link : links) {
        auto nvme_ctrl = gpu_controller->getNVMeController(link.controller_index);
        if (!nvme_ctrl) {
            geminifs_error("GPU_write_kernel: NVMe controller %zu not found\n", link.controller_index);
            return false;
        }
        void* device_fd = nvme_ctrl->g_open(std::string(link.name), link.file_size, O_DEVICE);
        if (!device_fd) {
            geminifs_error("GPU_write_kernel: g_open failed for %s\n", link.name);
            return false;
        }
        nvme_fds.push_back(reinterpret_cast<NVMe_File*>(device_fd));
    }

    // 拷贝 NVMe_File** 到设备
    NVMe_File** d_fds = nullptr;
    size_t num_fds = nvme_fds.size();
    cudaError_t err = cudaMalloc(&d_fds, num_fds * sizeof(NVMe_File*));
    if (err != cudaSuccess) {
        geminifs_error("GPU_write_kernel: cudaMalloc d_fds failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(d_fds, nvme_fds.data(), num_fds * sizeof(NVMe_File*), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_fds);
        geminifs_error("GPU_write_kernel: cudaMemcpy d_fds failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    // 启动多FD内核
    uint64_t tensor_ptr = reinterpret_cast<uint64_t>(tensor.data_ptr());
    size_t offset = 0;
    size_t len = static_cast<size_t>(tensor.numel()) * static_cast<size_t>(tensor.element_size());
    GPU_Write_kernel_multi<<<1,1>>>(d_fds, static_cast<uint32_t>(num_fds), tensor_ptr, offset, len, device_view);
    err = cudaDeviceSynchronize();

    cudaFree(d_fds);
    if (err != cudaSuccess) {
        geminifs_error("GPU_write_kernel: GPU_Write_kernel_multi failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

__host__ void geminifs_cleanup_all_gpu_controllers() {
    auto& registry = GPUControllerRegistry::getInstance();
    registry.clearAll();
    geminifs_debug("geminifs_cleanup_all_gpu_controllers: Cleaned up all GPU controllers\n");
}

/**
 * Cleanup all GPU controllers
 */
__host__ void GeminiFS::geminifs_cleanup_all_gpu_controllers() {
    auto& registry = GPUControllerRegistry::getInstance();
    registry.clearAll();
    geminifs_debug("geminifs_cleanup_all_gpu_controllers: Cleaned up all GPU controllers\n");
}

__host__ bool geminifs_nvme_delete_all_files(int device_id, size_t controller_index) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_nvme_delete_all_files: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    auto nvme_controller = gpu_controller->getNVMeController(controller_index);
    if (!nvme_controller) {
        geminifs_error("geminifs_nvme_delete_all_files: No NVMe controller found at index %zu for device %d\n", 
                       controller_index, device_id);
        return false;
    }
    
    bool success = nvme_controller->device_file_delete_all_files_managed();
    if (success) {
        geminifs_debug("geminifs_nvme_delete_all_files: Successfully cleaned all files for device %d controller %zu\n", 
                       device_id, controller_index);
    } else {
        geminifs_error("geminifs_nvme_delete_all_files: Failed to clean all files for device %d controller %zu\n", 
                       device_id, controller_index);
    }
    
    return success;
}

/**
 * Clean all files managed by a specific NVMe controller
 */
__host__ bool GeminiFS::geminifs_nvme_delete_all_files(int device_id, size_t controller_index) {
    auto gpu_controller = geminifs_get_gpu_controller(device_id);
    if (!gpu_controller) {
        geminifs_error("geminifs_nvme_delete_all_files: No GPU controller found for device %d\n", device_id);
        return false;
    }
    
    auto nvme_controller = gpu_controller->getNVMeController(controller_index);
    if (!nvme_controller) {
        geminifs_error("geminifs_nvme_delete_all_files: No NVMe controller found at index %zu for device %d\n", 
                       controller_index, device_id);
        return false;
    }
    
    bool success = nvme_controller->device_file_delete_all_files_managed();
    if (success) {
        geminifs_debug("geminifs_nvme_delete_all_files: Successfully cleaned all files for device %d controller %zu\n", 
                       device_id, controller_index);
    } else {
        geminifs_error("geminifs_nvme_delete_all_files: Failed to clean all files for device %d controller %zu\n", 
                       device_id, controller_index);
    }
    
    return success;
}

__host__ void GeminiFS::init(const std::string& config_file_path) {
    auto& registry = GPUControllerRegistry::getInstance();
    registry.clearAll();
    auto_configure_fd_limits(NUM_FILES);
    geminifs_debug("geminifs_cleanup_all_gpu_controllers: Cleaned up all GPU controllers\n");

    ParsedSystemConfig config = parse_system_config(config_file_path);
    if (config.valid) {
        // 转换为nvme_ctrl_param格式
        nvme_params_ = convert_to_nvme_ctrl_params(config);

        for (const auto& group : config.groups) {
            geminifs_debug("geminifs_init: Creating GPU controller for device %d with mount path %s\n", group.gpu.cudaDevice, group.gpu.mount_path.c_str());
            geminifs_create_gpu_controller(group.gpu.cudaDevice, group.gpu.mount_path);
            for (const auto& nvme_ctrl_param : nvme_params_) {
                if (!nvme_ctrl_param.pci_addr.empty()) {
                    geminifs_debug("geminifs_init: Adding NVMe controller for device %d with mount path %s\n", group.gpu.cudaDevice, nvme_ctrl_param.mount_path.c_str());
                    geminifs_add_nvme_to_gpu(group.gpu.cudaDevice, nvme_ctrl_param);
                }
            }
        }
    } else {
        std::cerr << "Config parsing failed: " << config.error_message << std::endl;
        return ;
    }

    geminifs_debug("geminifs_init: Successfully initialized GeminiFS\n");
}

__host__ void GeminiFS::cleanup() {
    geminifs_cleanup_all_gpu_controllers();
}
