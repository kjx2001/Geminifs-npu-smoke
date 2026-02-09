#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nvmeservice_protocol.h"

namespace nvmeservice {

class NvmeServiceClient {
public:
    explicit NvmeServiceClient(std::string socket_path);

    bool ping();
    bool getInfo(std::string& mount_base_path, std::vector<CtrlConfig>& ctrls, uint32_t& max_queues_per_process);

private:
    bool request(const MsgHeader& hdr, const void* payload, uint32_t payload_bytes, RespHeader& resp, std::vector<uint8_t>& resp_payload);
    std::string socket_path_;
};

} // namespace nvmeservice
