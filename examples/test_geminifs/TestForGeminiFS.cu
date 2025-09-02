#include "geminifs.cuh"
#include "gpu_controller.cuh"
#include "geminifs_helper.h"
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

//constexpr size_t GPU_file_size = 32ull * 1024 * 1024; // 32MB
constexpr int GPU_file_nums = 100;
constexpr int device_id = 1;

torch::Tensor create_tensor() {
    return torch::rand({4, 1024, 1024, 2},
        torch::TensorOptions()
            .dtype(torch::kFloat16)
            .device(torch::kCUDA, device_id)
            .pinned_memory(false));
}

void check_tensor_equal(const torch::Tensor& t1, const torch::Tensor& t2, const std::string& name) {
    if (!torch::all(t1 == t2).item<bool>()) {
        std::cout << "Tensors " << name << " are not equal!" << std::endl;
        auto t1_flat = t1.flatten().slice(0, 0, 20).cpu();
        auto t2_flat = t2.flatten().slice(0, 0, 20).cpu();
        std::cout << "First 20 elements of " << name << "1: " << t1_flat << std::endl;
        std::cout << "First 20 elements of " << name << "2: " << t2_flat << std::endl;
        TORCH_CHECK(false, "Tensors " + name + " are not equal!");
    }
}

int main(int argc, char **argv)
{
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    vector<size_t> GPU_file_shape = {2, 32, 524288}; // 32MB
    GeminiFS geminifs("/home/qs/CompanionFS/Geminifs/sys_config.ini", GPU_file_nums, GPU_file_shape, 1);

    GPUFileId file_id, file_id2;
    assert(geminifs.geminifs_gpu_open_file(device_id, file_id));
    assert(geminifs.geminifs_gpu_open_file(device_id, file_id2));
    printf("finish open file, file id %u, file id2 %u\n", file_id, file_id2);

    auto key = create_tensor(), value = create_tensor();
    auto key2 = create_tensor(), value2 = create_tensor();
    auto key3 = create_tensor(), value3 = create_tensor();
    auto key4 = create_tensor(), value4 = create_tensor();

    for (auto& t : {&key, &key2, &key3, &key4}) geminifs.geminifs_register_tensor_with_gpu(*t);
    printf("finish register tensor key\n");
    for (auto& t : {&value, &value2, &value3, &value4}) geminifs.geminifs_register_tensor_with_gpu(*t);
    printf("finish register tensor value\n");

    geminifs.geminifs_GPU_write_kernel(key, value, file_id, geminifs.geminifs_get_gpu_controller(device_id));
    printf("finish write kernel\n");
    geminifs.geminifs_GPU_read_kernel(key2, value2, file_id, geminifs.geminifs_get_gpu_controller(device_id));
    printf("finish read kernel\n");

    check_tensor_equal(key, key2, "key");
    check_tensor_equal(value, value2, "value");

    vector<torch::Tensor> key_list = {key, key2};
    vector<torch::Tensor> value_list = {value, value2};
    vector<GPUFileId> file_id_list = {file_id, file_id2};
    vector<torch::Tensor> key_list2 = {key3, key4};
    vector<torch::Tensor> value_list2 = {value3, value4};
    vector<int> layer_id_list = {0, 1};

    geminifs.geminifs_batched_write(key_list, value_list, file_id_list, layer_id_list, geminifs.geminifs_get_gpu_controller(device_id));
    geminifs.geminifs_batched_read(key_list2, value_list2, file_id_list, layer_id_list, geminifs.geminifs_get_gpu_controller(device_id));

    check_tensor_equal(key, key3, "key_batch");
    check_tensor_equal(key2, key4, "key_batch");
    check_tensor_equal(value, value3, "value_batch");
    check_tensor_equal(value2, value4, "value_batch");

    geminifs.geminifs_batched_write(key_list, layer_id_list, value_list, layer_id_list, file_id_list, geminifs.geminifs_get_gpu_controller(device_id));
    geminifs.geminifs_batched_read(key_list2, layer_id_list, value_list2, layer_id_list, file_id_list, geminifs.geminifs_get_gpu_controller(device_id));

    check_tensor_equal(key, key3, "key_batch");
    check_tensor_equal(key2, key4, "key_batch");
    check_tensor_equal(value, value3, "value_batch");
    check_tensor_equal(value2, value4, "value_batch");

    return 0;
}
