cmake_minimum_required(VERSION 3.16)
project(NVMeService LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(NVMESERVICE_BUILD_TESTS "Build NVMeService tests" ON)

set(NVMESERVICE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIBNVM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../nvme/libnvm/include)

add_library(nvmeservice STATIC
    src/nvmeservice_state.cpp
    src/nvmeservice_server.cpp
    src/nvmeservice_client.cpp
    src/nvmeservice_nvme_controller.cpp
)

target_include_directories(nvmeservice
    PUBLIC
        ${NVMESERVICE_INCLUDE_DIR}
        ${LIBNVM_INCLUDE_DIR}
)

target_compile_definitions(nvmeservice PRIVATE NVMESERVICE_USE_LIBNVM=1)

add_executable(nvmeservice_daemon src/main.cpp)

target_link_libraries(nvmeservice_daemon PRIVATE nvmeservice)

target_include_directories(nvmeservice_daemon PRIVATE
    ${NVMESERVICE_INCLUDE_DIR}
    ${LIBNVM_INCLUDE_DIR}
)

if (NVMESERVICE_BUILD_TESTS)
    add_executable(nvmeservice_admin_init tests/nvmeservice_admin_init.cpp)
    target_link_libraries(nvmeservice_admin_init PRIVATE nvmeservice)
    target_include_directories(nvmeservice_admin_init PRIVATE
        ${NVMESERVICE_INCLUDE_DIR}
        ${LIBNVM_INCLUDE_DIR}
    )

    add_executable(nvmeservice_fs_info tests/nvmeservice_fs_info.cpp)
    target_link_libraries(nvmeservice_fs_info PRIVATE nvmeservice)
    target_include_directories(nvmeservice_fs_info PRIVATE
        ${NVMESERVICE_INCLUDE_DIR}
        ${LIBNVM_INCLUDE_DIR}
    )
endif()
