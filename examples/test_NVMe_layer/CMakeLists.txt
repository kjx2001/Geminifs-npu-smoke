cmake_minimum_required (VERSION 3.18)

set (CMAKE_CUDA_ARCHITECTURES 90)
set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CUDA_STANDARD 20)

project(geminifs-test LANGUAGES CUDA CXX)


# setup CUDA 
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")
find_package(CUDAToolkit 12.6 REQUIRED)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Find CUDA !")
else()
    message(FATAL_ERROR "Not found CUDA!")
endif()

if(CUDAToolkit_VERSION VERSION_LESS 12.6)
    message(FATAL_ERROR "CUDA version must be at least 12.6 (found: ${CUDAToolkit_VERSION})")
endif()

set (root "${CMAKE_CURRENT_SOURCE_DIR}/../../")

set(LIBTORCH_LOCAL_PATH "${root}/third_pkgs/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_LOCAL_PATH}")
message(STATUS "Project-local LibTorch path added: ${LIBTORCH_LOCAL_PATH}")
message(STATUS "Current CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${Torch_VERSION}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



include_directories(
    ${root}/libgeminifs/include/
    ${root}/libnvm/include
    ${root}/build/
)

link_directories(
    ${root}/build/lib
)
                                                                                                                                  

add_executable(TestForNVMeLayer_rw TestForNVMeLayer_rw.cu)

# Link libraries
target_link_libraries(TestForNVMeLayer_rw PRIVATE
    geminifs
    nvm
    CUDA::cudart
    "${TORCH_LIBRARIES}"
)

# Set CUDA properties
set_target_properties(TestForNVMeLayer_rw PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)