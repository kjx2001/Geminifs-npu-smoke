#ifndef NVMESERVICE_PROTOCOL_H
#define NVMESERVICE_PROTOCOL_H

#include <cstdint>

namespace nvmeservice {

constexpr uint32_t kMagic = 0x4E564D53; // "NVMS"
constexpr uint16_t kVersion = 1;

enum class Role : uint16_t {
    Admin = 1,
    Filesystem = 2,
};

enum class Command : uint16_t {
    Ping = 0,
    AdminInit = 1,
    AdminSetLimits = 2,
    FsGetInfo = 10,
    FsAllocQueues = 11,
    FsReleaseQueues = 12,
};

enum class Status : uint16_t {
    Ok = 0,
    Err = 1,
    NotFound = 2,
    Denied = 3,
    Invalid = 4,
    NoResources = 5,
};

struct MsgHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t role;
    uint16_t command;
    uint16_t reserved0;
    uint32_t request_id;
    uint32_t payload_bytes;
};

struct CtrlConfig {
    char mount_path[128];
    char pci_addr[32];
    uint32_t ns_id;
    uint32_t queue_depth;
    uint32_t num_queues;
    uint32_t cuda_device;
    uint32_t max_io_kb;
};

struct AdminInitReq {
    char mount_base_path[128];
    uint32_t ctrl_count;
    uint32_t reserved;
    // Followed by ctrl_count * CtrlConfig
};

struct AdminSetLimitsReq {
    uint32_t max_queues_per_process;
    uint32_t reserved;
};

struct FsGetInfoReq {
    uint32_t reserved;
};

struct FsAllocQueuesReq {
    uint32_t controller_index;
    uint32_t queue_count;
    int32_t pid;
    uint32_t reserved;
};

struct FsReleaseQueuesReq {
    uint32_t controller_index;
    uint32_t queue_count;
    int32_t pid;
    uint32_t reserved;
    // Followed by queue_count * uint32_t queue_ids
};

struct FsAllocQueuesResp {
    uint32_t granted;
    uint32_t reserved;
    // Followed by granted * uint32_t queue_ids
};

struct FsGetInfoResp {
    char mount_base_path[128];
    uint32_t ctrl_count;
    uint32_t max_queues_per_process;
    // Followed by ctrl_count * CtrlConfig
};

struct RespHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    uint32_t request_id;
    uint32_t payload_bytes;
};

} // namespace nvmeservice

#endif // NVMESERVICE_PROTOCOL_H
