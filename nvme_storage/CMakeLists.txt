# nvme_storage/ -- Tutti named-LBA-range storage layer (R5a)
#
# Builds: libtutti_nvme_storage.so + nvme_storage_smoke
#
# - libtutti_nvme_storage: HostFsBackedNvmeStorage (mounts an ext4
#   filesystem on /dev/snvmeNn1, manages a flat directory of named
#   files via PersistentFileLog, exposes host-side blocking IO).
# - nvme_storage_smoke    : end-to-end (1 + multi NVMe).
#
# Depends on:
#   tutti_device_manager  -- for LocalNvmeDevice payload + the two
#                            registry impls used in smokes.
#   stdc++fs              -- std::filesystem (older toolchains).
#   pthread               -- std::mutex.

set(TUTTI_NVME_STORAGE_SOURCES
    src/fiemap_helper.cpp
    src/persistent_file_log.cpp
    src/host_fs_backed_nvme_storage.cpp
)

add_library(tutti_nvme_storage ${TUTTI_NVME_STORAGE_SOURCES})

target_include_directories(tutti_nvme_storage
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(tutti_nvme_storage
    PUBLIC
        tutti_device_manager
)

set_target_properties(tutti_nvme_storage PROPERTIES
    OUTPUT_NAME    "tutti_nvme_storage"
    BUILD_RPATH    "$ORIGIN"
    INSTALL_RPATH  "$ORIGIN"
    POSITION_INDEPENDENT_CODE ON
)

# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

add_executable(nvme_storage_smoke
    test/nvme_storage_smoke.cu
)

target_link_libraries(nvme_storage_smoke
    PRIVATE
        tutti_nvme_storage
        tutti_device_manager
        libnvm
        CUDA::cudart
)

set_target_properties(nvme_storage_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH   "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

install(TARGETS tutti_nvme_storage DESTINATION "lib")
install(TARGETS nvme_storage_smoke DESTINATION "bin")
install(DIRECTORY include/ DESTINATION "include/tutti/nvme_storage"
        FILES_MATCHING PATTERN "*.h")
