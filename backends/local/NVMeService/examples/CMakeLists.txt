cmake_minimum_required(VERSION 3.18)

project(nvmeservice_examples LANGUAGES CXX)

message(STATUS "CMAKE_CURRENT_LIST_DIR=${CMAKE_CURRENT_LIST_DIR}")

set(NVMESERVICE_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../build/lib" CACHE PATH "Path to Geminifs build/lib")
get_filename_component(NVMESERVICE_LIB_DIR "${NVMESERVICE_LIB_DIR}" ABSOLUTE)
set(NVMESERVICE_LIB "${NVMESERVICE_LIB_DIR}/libnvmeservice.so")

if(NOT EXISTS "${NVMESERVICE_LIB}")
    message(FATAL_ERROR "libnvmeservice.so not found at ${NVMESERVICE_LIB}. Set NVMESERVICE_LIB_DIR to Geminifs/build/lib")
endif()

add_library(nvmeservice SHARED IMPORTED)
set_target_properties(nvmeservice PROPERTIES IMPORTED_LOCATION "${NVMESERVICE_LIB}")
target_include_directories(nvmeservice INTERFACE "${CMAKE_CURRENT_LIST_DIR}/../src")

add_executable(nvmeservice_daemon_example nvmeservice_daemon.cpp)
target_link_libraries(nvmeservice_daemon_example PRIVATE nvmeservice)
