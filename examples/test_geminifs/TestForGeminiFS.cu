#include "geminifs.cuh"
#include "gpu_controller.cuh"
#include "geminifs_helper.h"
#include "gpu_controller.cuh"

constexpr size_t file_size = 512ull * 1024;

int main(int argc, char** argv) {
    GeminiFS geminifs("/home/zwh/Geminifs/sys_config.ini");
    geminifs.geminifs_gpu_open_file(0, 1, file_size, O_DEVICE);
    printf("finish open file\n");
    auto key_cache = torch::rand({4, 1024, 1024, 2}, // 512kb
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, 0)
            .pinned_memory(false)
        );


    auto key_cache2 = torch::rand({4, 1024, 1024, 2}, // 512kb
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, 0)
            .pinned_memory(false)
        );
    geminifs.geminifs_register_tensor_with_gpu(key_cache);
    printf("finish register tensor keycache\n");
    geminifs.geminifs_register_tensor_with_gpu(key_cache2);
    printf("finish register tensor keycache2\n");
    geminifs.geminifs_GPU_write_kernel(key_cache, 1, geminifs.geminifs_get_gpu_controller(0));
    printf("finish write kernel\n");
    geminifs.geminifs_GPU_read_kernel(key_cache2, 1, geminifs.geminifs_get_gpu_controller(0));
    printf("finish read kernel\n");
    torch::all(key_cache == key_cache2);
    return 0;
}