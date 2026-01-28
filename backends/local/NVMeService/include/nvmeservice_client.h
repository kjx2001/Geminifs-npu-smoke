#ifndef NVMESERVICE_CLIENT_H
#define NVMESERVICE_CLIENT_H

#include "nvmeservice_protocol.h"
#include "nvmeservice_nvme_controller.h"

#include <memory>
#include <string>
#include <vector>

namespace nvmeservice {

class NvmeServiceClient {
public:
    explicit NvmeServiceClient(std::string socket_path);
    bool getInfo(FsGetInfoResp& out_header, std::vector<CtrlConfig>& out_ctrls);
    bool allocQueues(uint32_t controller_index, uint32_t count, int32_t pid, std::vector<uint32_t>& out_qids);
    bool releaseQueues(uint32_t controller_index, int32_t pid, const std::vector<uint32_t>& qids);
    bool adminInitFromConfig(const std::string& config_path, int gpu_id, bool start_module = true);
    bool adminInit(const std::string& mount_base_path, const std::vector<CtrlConfig>& ctrls, bool start_module = true);

private:
    bool request(const MsgHeader& hdr, const void* payload, uint32_t payload_bytes,
                 RespHeader& resp, std::vector<uint8_t>& resp_payload);

    std::string socket_path_;
    std::vector<NVMeControllerPtr> controllers_;
};

} // namespace nvmeservice

#endif // NVMESERVICE_CLIENT_H
