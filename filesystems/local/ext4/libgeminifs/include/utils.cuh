#ifndef __UTILS_CUH__
#define __UTILS_CUH__

#include <cassert>
#include <cstdint>
#include <cuda/semaphore>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


#if defined( __clang__ ) || defined( __GNUC__ )
#define class_container_of(ptr, type, member) ({                 \
        const typeof( ((type *) 0)->member )* __mptr = (ptr);   \
        (type *) (((unsigned char*) __mptr) - offsetof(type, member)); })
#else
#define class_container_of(ptr, type, member) \
    ((type *) (((unsigned char*) (ptr)) - ((unsigned char*) (&((type *) 0)->member))))
#endif

__forceinline__ __device__ int my_lane_id() {
    int lane_id = threadIdx.x & 0x1f;
    return lane_id;
}

template <typename T>
__global__ void
__run_device_lambda(T lambda) {
    lambda();
}

#define RUN_ON_DEVICE__MULTI_THREAD(CODE_BLOCK, nr_thread) { \
    __run_device_lambda<<<1, nr_thread>>>([=] __device__ () { \
        CODE_BLOCK \
    }); \
    cudaDeviceSynchronize(); \
}

#define RUN_ON_DEVICE(CODE_BLOCK) RUN_ON_DEVICE__MULTI_THREAD(CODE_BLOCK, 1)

static inline void cuda_assert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
}

static __device__ void cuda_device_assert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
       printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) assert(0);
    }
}

#define ROUND_UP(x, align)(((uint64_t) (x) + ((uint64_t)align - 1)) & ~((uint64_t)align - 1))
#define cuda_check_error(ans) { cuda_assert((ans), __FILE__, __LINE__); }
#define cuda_device_check_error(ans) { cuda_device_assert((ans), __FILE__, __LINE__); }
// page macros
#define __128KB__ 128 * (1ull << 10)
#define __64KB__  64 * (1ull << 10)
#define __4KB__  4096ul
#define __WORD__ 8
#define MAX_BLOCK_SIZE __128KB__



using cuda_device_lock = cuda::binary_semaphore<cuda::thread_scope_device>;
using cuda_device_ref = cuda::atomic<uint32_t, cuda::thread_scope_device>;
using FileGroupId = int32_t;
using GPUFileId = uint32_t;



#endif