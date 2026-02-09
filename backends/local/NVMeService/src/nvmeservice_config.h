#pragma once

#include <string>
#include <vector>

#include "nvmeservice_protocol.h"

namespace nvmeservice {

struct ParsedConfigAll {
    std::string mount_base_path;
    bool multi_mount = false;
    std::vector<CtrlConfig> ctrls;
    bool socket_enabled = false;
    std::string socket_path;
    uint32_t max_queues_per_process = 64;
};

constexpr const char* kDefaultSocketPath = "/var/run/nvmeservice.sock";
constexpr uint32_t kDefaultMaxQueuesPerProcess = 64;

bool parseSysConfig(const std::string& path, ParsedConfigAll& out, std::string* error_message);

} // namespace nvmeservice
