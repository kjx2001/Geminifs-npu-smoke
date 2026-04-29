# io_engine — GeminiFS IO submission layer
#
# Builds: libio_engine.so
#   - BlockIoEngine       (host-side scheduler owner, channel factory)
#   - BlockIoChannel      (GPU-side per-file IO handle, header-only)
#   - NvmeQueueScheduler  (GPU-side queue selection + NVMe cmd submit, header-only)
#   - init / one-shot kernels
#
# Depends on:
#   libnvm          — QueuePair, nvm_cmd_t, sq_enqueue / cq_poll
#   device_manager  — BlockAddressTranslator (for BlockIoChannel)
#   libgeminifs     — utils (geminifs_debug / geminifs_error)

set(IO_ENGINE_SOURCES
    src/block_io_engine.cu
)

add_library(io_engine ${IO_ENGINE_SOURCES})

target_include_directories(io_engine
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${PROJECT_SOURCE_DIR}/backends/local/nvme/libnvm/include"
        "${PROJECT_SOURCE_DIR}/filesystems/ext4/libgeminifs/include"
)

target_link_libraries(io_engine
    PRIVATE
        libnvm
        device_manager
        libgeminifs
        CUDA::cudart
)

set_target_properties(io_engine PROPERTIES
    OUTPUT_NAME "io_engine"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

install(TARGETS io_engine DESTINATION "lib")
install(DIRECTORY include/ DESTINATION "include/io_engine"
        FILES_MATCHING PATTERN "*.cuh" PATTERN "*.h")