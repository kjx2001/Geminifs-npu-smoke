#pragma once

#include <string>
#include <vector>

#include "nvmeservice_protocol.h"

namespace nvmeservice {

struct ParsedConfigAll {
    std::string mount_base_path;
    bool multi_mount = false;
    std::vector<CtrlConfig> ctrls;
    std::string grpc_endpoint;
    uint32_t max_queues_per_process = 64;
};

constexpr const char* kDefaultGrpcEndpoint = "127.0.0.1:50051";
constexpr uint32_t kDefaultMaxQueuesPerProcess = 64;

bool parseSysConfig(const std::string& path, ParsedConfigAll& out, std::string* error_message);

} // namespace nvmeservice
