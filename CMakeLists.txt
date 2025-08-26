cmake_minimum_required (VERSION 3.18)

set (CMAKE_CUDA_ARCHITECTURES 90)
set (CMAKE_C_STANDARD 99)
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS on CACHE INTERNAL "" FORCE)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
# set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

project (libnvm LANGUAGES CUDA C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
endif()


set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")

find_package (Threads REQUIRED)
find_package(CUDAToolkit 12.6 REQUIRED)
find_package(PkgConfig)

pkg_check_modules(UNWIND REQUIRED libunwind)

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

message(STATUS "Using CUDA ${CMAKE_CUDA_COMPILER_VERSION}")

set(LIBTORCH_LOCAL_PATH "${CMAKE_SOURCE_DIR}/third_pkgs/libtorch")
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_LOCAL_PATH}")
message(STATUS "Project-local LibTorch path added: ${LIBTORCH_LOCAL_PATH}")
message(STATUS "Current CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${Torch_VERSION}")

set (NVIDIA "" CACHE PATH "Path to Nvidia driver source")
set (KERNEL "/lib/modules/${CMAKE_SYSTEM_VERSION}/build" CACHE PATH "Path to kernel source, or module directory")

# set (nvidia_archs "90" CACHE STRING "NVIDIA compute architectures")
set (no_module false CACHE BOOL "Do not build kernel module")
set (no_cuda false CACHE BOOL "Do not build library with CUDA support")
set (no_cc false CACHE BOOL "Do not assume cache-coherent architecture")
set (BUILD_SHARED_LIBS true CACHE BOOL "Build shared library, set to false for static")


# Set global compiler flags
set (GPU_ARCHS "" CACHE INTERNAL "" FORCE)
foreach (arch ${nvidia_archs})
    string (CONCAT GPU_ARCHS ${GPU_ARCHS} " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach ()

set (convert_warnings implicit-function-declaration incompatible-pointer-types int-conversion return-type unused-function unused-variable)
set (flags "")
foreach (flag ${convert_warnings})
    string (APPEND flags " -Werror=${flag}")
endforeach()


# C_FLAGS
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra ${flags}" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG} -g -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
# CXX_FLAGS
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++20" CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG " CACHE INTERNAL "" FORCE)
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -std=c++20" CACHE INTERNAL "" FORCE)
# CUDA_FLAGS
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -D__CUDA__ -D__CUDACC__ -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -rdc=true ${GPU_ARCHS}")
set (CUDA_NVCC_FLAGS_DEBUG "${CMAKE_NVCC_FLAGS_DEBUG} -DDEBUG" CACHE INTERNAL "" FORCE)
set (CUDA_NVCC_FLAGS_RELEASE "${CMAKE_NVCC_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)

# set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker -rpath=${CMAKE_BINARY_DIR}/lib --expt-relaxed-constexpr --expt-extended-lambda  -lcudart ${GPU_ARCHS}")
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker -rpath=${CMAKE_BINARY_DIR}/lib -rdc=true --expt-relaxed-constexpr --expt-extended-lambda -lineinfo ${GPU_ARCHS}")
set (CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -DDEBUG" CACHE INTERNAL "" FORCE)
set (CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2" CACHE INTERNAL "" FORCE)
set (CMAKE_CUDA_GEMINIFS_LIBS "${CMAKE_CUDA_GEMINIFS_LIBS} -L${PROJECT_BINARY_DIR}/libgeminifs -L${PROJECT_SOURCE_DIR}/include/geminifs -L${CMAKE_BINARY_DIR}/lib -lgeminifs -lnvm")

if (no_cc)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
    set(CCUDA_NVCC_FLAGS "${CMAKE_C_FLAGS} -D__NO_COHERENCE__")
endif ()

include_directories ("${PROJECT_SOURCE_DIR}/libnvm/src"
                     "${PROJECT_SOURCE_DIR}/libnvm/include" 
                     "${PROJECT_SOURCE_DIR}/libgeminifs/include")

# Try to find Nvidia driver
if (CUDA_FOUND AND NOT no_cuda)

    # Try to find Nvidia headers (will choose the newest version)
    file (GLOB_RECURSE driver_paths "/usr/src/nvidia-*/nv-p2p.h")
    string (REPLACE "nv-p2p.h" "" driver_paths "${driver_paths}")
    if (driver_paths)
        list (REVERSE driver_paths)
    endif ()

    find_path (driver_include NAMES "nv-p2p.h" PATHS "${NVIDIA}" ${driver_paths})
    set (driver_dir "${driver_include}")

    # Find Nvidia driver symbols
    if (NOT EXISTS "${driver_dir}/Module.symvers")
        get_filename_component (driver_dir "${driver_dir}" DIRECTORY)
    endif ()

    if ("${driver_dir}" STREQUAL "")
        message(STATUS "No NVIDIA driver symbols found")
    else ()
        message(STATUS "Using NVIDIA driver found in ${driver_dir}")
    endif ()
endif ()


# Library source files
set (libnvm_root "${PROJECT_SOURCE_DIR}/libnvm/")
file (GLOB libnvm_source "${libnvm_root}/src/*.cpp")
file (GLOB libnvm_linux_source "${libnvm_root}/src/linux/*.cpp")
file (GLOB libnvm_include "${libnvm_root}/include/*.h")


# # Module source files
set (module_root "${PROJECT_SOURCE_DIR}/module")
set (module_output "${CMAKE_BINARY_DIR}/module")
# for Makefile.in, do not delete
set (module_ccflags "-I${PROJECT_SOURCE_DIR}/libnvm/include -I${driver_include}")
configure_file ("${module_root}/Makefile.in" "${module_output}/Makefile" @ONLY)
message (STATUS "Configuring module build in ${module_output}")

add_custom_target(
    modules
    ALL
    COMMAND make
    WORKING_DIRECTORY ${module_output}
    COMMENT "Building kernel module"
)

add_custom_target(
    clean_modules
    COMMAND make clean
    WORKING_DIRECTORY ${module_output}
    COMMENT "Cleaning kernel module"
)

add_custom_target(
    insmod
    COMMAND sudo make insmod
    WORKING_DIRECTORY ${module_output}
    COMMENT "Inserting kernel module"
)

add_custom_target(
    rmmod
    COMMAND sudo make rmmod
    WORKING_DIRECTORY ${module_output}
    COMMENT "Removing kernel module"
)

# Benchmarks
set (benchmarks_root "${PROJECT_SOURCE_DIR}/benchmarks")
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


# Create NVME library target
# add_library(libnvm ${libnvm_source} ${libnvm_linux_source})
# CUDA_ADD_LIBRARY (libnvm ${libnvm_source} ${libnvm_linux_source}) # TODO: add check for OS
add_library(libnvm ${libnvm_source} ${libnvm_linux_source})
set_target_properties(libnvm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories (libnvm PRIVATE "${libnvm_root}")
target_link_libraries(libnvm CUDA::cudart)
target_link_libraries (libnvm Threads::Threads)
message (STATUS "Configuring libnvm without SmartIO")


if (CUDA_FOUND AND NOT no_cuda)
    target_compile_definitions(libnvm PRIVATE _CUDA __CUDA__)
endif ()

# Set name of library
set_target_properties (libnvm PROPERTIES OUTPUT_NAME "nvm")
# Set install locations
install (FILES ${libnvm_include} DESTINATION "include")
install (TARGETS libnvm DESTINATION "lib")

# libgeminifs
set (libgeminifs_root "${PROJECT_SOURCE_DIR}/libgeminifs")
file (GLOB libgeminifs_cpp_source "${libgeminifs_root}/*.cpp")
file (GLOB libgeminifs_cuda_source "${libgeminifs_root}/*.cu")
file (GLOB libgeminifs_include "${libgeminifs_root}/include/geminifs/*.h" "${libgeminifs_root}/include/geminifs/*.cuh")

message(STATUS "Configuring libgeminifs: ${libgeminifs_cuda_source} ${libgeminifs_cpp_source} ${libgeminifs_include}")

add_library(libgeminifs ${libgeminifs_cuda_source} ${libgeminifs_cpp_source} ${libgeminifs_include})

# Add Torch related configurations
target_link_libraries(libgeminifs PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(libgeminifs PRIVATE "${TORCH_INCLUDE_DIRS}")

target_link_libraries(libgeminifs PRIVATE rocksdb)

target_include_directories(libgeminifs PRIVATE "${libgeminifs_root}")
target_link_libraries(libgeminifs PRIVATE libnvm CUDA::cudart "${TORCH_LIBRARIES}")
# target_include_directories(libgeminifs PRIVATE "${libgeminifs_root}")
# target_link_libraries(libgeminifs ${CUDA_LIBRARIES} libnvm )
target_link_libraries(libgeminifs
    PRIVATE
    unwind
    unwind-x86_64
    ${CMAKE_DL_LIBS}
)

set_target_properties(libgeminifs PROPERTIES OUTPUT_NAME "geminifs")
set_target_properties(libgeminifs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(libgeminifs PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
install (FILES ${libgeminifs_include} DESTINATION "include/geminifs")
install (TARGETS libgeminifs DESTINATION "lib")


# Macro for creating sample target
macro (make_sample target binary_name files)
    cuda_add_executable (${target} EXCLUDE_FROM_ALL ${files})
    
    add_dependencies (${target} libnvm)
    target_link_libraries (${target} PRIVATE libnvm)

    # TODO: Do we want to install samples or not?
    #set_target_properties (${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/samples")
    set_target_properties (${target} PROPERTIES OUTPUT_NAME "nvm-${binary_name}")

    list (APPEND sample_targets "${target}")
    set (sample_targets "${sample_targets}" PARENT_SCOPE)

    #install (TARGETS ${target} DESTINATION "bin")
endmacro ()

# Make target depend on pthreads
macro (set_multithread target)
    target_link_libraries (${target} PRIVATE Threads::Threads)
endmacro ()

# Set preferred ordering for sample targets
# Make benchmark target
# We assume all benchmarks are CUDA programs
macro (make_benchmark target binary_name files)
    if (CUDA_FOUND AND NOT no_cuda)
        cuda_add_executable (${target}) # Ugly bugly
        target_sources (${target} PRIVATE ${files})

        add_dependencies (${target} libgeminifs)
        target_link_libraries (${target}  libnvm libgeminifs CUDA::cudart)
        set_target_properties (${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties (${target} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        set_target_properties (${target} PROPERTIES OUTPUT_NAME "nvm-${binary_name}")

        list (APPEND benchmark_targets "${target}")
        set (benchmark_targets "${benchmark_targets}" PARENT_SCOPE)
    endif ()
endmacro ()

macro (make_samples_choice ctarget sisci_target other_target)
    if (NOT no_smartio_samples AND sisci_include AND sisci_lib)
        add_custom_target (${ctarget} DEPENDS ${sisci_target})
        list (REMOVE_ITEM sample_targets "${other_target}")
        set (sample_targets "${sample_targets}" PARENT_SCOPE)
    else ()
        add_custom_target (${ctarget} DEPENDS ${other_target})
        list (REMOVE_ITEM sample_targets "${sisci_target}")
        set (sample_targets "${sample_targets}" PARENT_SCOPE)
    endif ()
endmacro ()

#  add_subdirectory ("${PROJECT_SOURCE_DIR}/examples/test")

 add_custom_target (benchmarks DEPENDS ${benchmark_targets})