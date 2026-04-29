# device_manager — GeminiFS device management layer
#
# Builds: libdevice_manager.so
#   - BlockDeviceManager  (host-side NVMe controller + file lifecycle)
#   - BlockAddressTranslator (GPU-side header-only, extracted here)
#   - init_block_translator_kernel
#
# Depends on:
#   libnvm      — Controller, QueuePair
#   libgeminifs — FileManager, utils, host_refine_nvmeofst (legacy utilities)

set(DEVICE_MANAGER_SOURCES
    src/block_device_manager.cu
)

add_library(device_manager ${DEVICE_MANAGER_SOURCES})

target_include_directories(device_manager
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${PROJECT_SOURCE_DIR}/backends/local/nvme/libnvm/include"
        "${PROJECT_SOURCE_DIR}/filesystems/ext4/libgeminifs/include"
)

target_link_libraries(device_manager
    PRIVATE
        libnvm
        libgeminifs
        CUDA::cudart
)

set_target_properties(device_manager PROPERTIES
    OUTPUT_NAME "device_manager"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS device_manager DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include/device_manager"
        FILES_MATCHING PATTERN "*.cuh" PATTERN "*.h")
