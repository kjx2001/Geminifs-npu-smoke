# NVMeService examples: daemon + client smoke test.
#
# Both link against the `nvmeservice` library (built by the root CMakeLists).
# The daemon needs libnvm for Controller construction; the client needs it
# only for shared_ctrl.h types (the function is inside libnvm too once
# shared_ctrl.cu is added).

add_executable(nvmeservice_daemon_example
    nvmeservice_daemon.cpp
)

target_link_libraries(nvmeservice_daemon_example PRIVATE
    nvmeservice
    libnvm
    ${GEMINIFS_YAML_CPP_TARGET}
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
    CUDA::cudart
)

set_target_properties(nvmeservice_daemon_example PROPERTIES
    OUTPUT_NAME   "nvmeservice_daemon"
    BUILD_RPATH   "$ORIGIN:$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(nvmeservice_client_example
    nvmeservice_client.cpp
)

target_link_libraries(nvmeservice_client_example PRIVATE
    nvmeservice
    libnvm
    gRPC::grpc++
    protobuf::libprotobuf
    Threads::Threads
    CUDA::cudart
)

set_target_properties(nvmeservice_client_example PROPERTIES
    OUTPUT_NAME   "nvmeservice_client"
    BUILD_RPATH   "$ORIGIN:$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
    CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS nvmeservice_daemon_example nvmeservice_client_example
        DESTINATION "bin")

# Copy the canonical sys_config.yaml from the repo root next to the
# binaries so users can run them without figuring out a path. The repo
# only keeps ONE sys_config.yaml (at the project root) -- the previous
# per-examples copy was removed to avoid drift.
configure_file("${CMAKE_SOURCE_DIR}/sys_config.yaml"
               "${CMAKE_BINARY_DIR}/bin/sys_config.yaml"
               COPYONLY)
