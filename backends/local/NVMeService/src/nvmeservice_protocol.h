#pragma once

#include <cstddef>
#include <cstdint>

namespace nvmeservice {

constexpr uint32_t kMagic = 0x4E564D45; // "NVME"
constexpr uint16_t kVersion = 1;

enum class Role : uint16_t {
    Admin = 1,
    Filesystem = 2,
};

enum class Command : uint16_t {
    Ping = 1,
    FsGetInfo = 4,
    FsAllocQueues = 5,
    FsReleaseQueues = 6,
};

enum class Status : uint16_t {
    Ok = 0,
    Invalid = 1,
    Denied = 2,
    Error = 3,
};

struct MsgHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t role;
    uint16_t command;
    uint16_t reserved;
    uint32_t request_id;
    uint32_t payload_bytes;
};

struct RespHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    uint32_t request_id;
    uint32_t payload_bytes;
};

struct CtrlConfig {
    char mount_path[256];
    char pci_addr[32];
    uint32_t ns_id;
    uint32_t queue_depth;
    uint32_t num_queues;
    uint32_t cuda_device;
    uint32_t max_io_kb;
};

struct FsGetInfoResp {
    char mount_base_path[256];
    uint32_t ctrl_count;
    uint32_t max_queues_per_process;
};

struct FsAllocQueuesReq {
    uint32_t controller_index;
    uint32_t queue_count;
    int32_t pid;
};

struct FsAllocQueuesResp {
    uint32_t granted;
};

struct FsReleaseQueuesReq {
    uint32_t controller_index;
    uint32_t queue_count;
    int32_t pid;
};

} // namespace nvmeservice
