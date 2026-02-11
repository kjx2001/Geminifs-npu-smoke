#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>

#include "nvmeservice.grpc.pb.h"
#include "nvmeservice_protocol.h"

namespace nvmeservice {

class NvmeServiceClient {
public:
    explicit NvmeServiceClient(std::string grpc_endpoint);

    bool ping();
    bool getInfo(std::string& mount_base_path, std::vector<CtrlConfig>& ctrls, uint32_t& max_queues_per_process);
    bool allocQueues(uint32_t controller_index,
                     uint32_t requested,
                     int32_t pid,
                     uint64_t client_id,
                     std::vector<uint32_t>& qids,
                     std::string& d_qps_handle,
                     std::string& d_ctrl_handle,
                     uint64_t& lease_id,
                     uint32_t& ttl_ms);
    bool releaseQueues(uint32_t controller_index, int32_t pid, const std::vector<uint32_t>& qids);
    bool releaseLease(uint64_t lease_id);
    bool heartbeatLeases(uint64_t client_id,
                         const std::vector<uint64_t>& lease_ids,
                         uint32_t duration_ms,
                         uint32_t interval_ms);
    bool shutdown();

private:
    void applyDeadline(grpc::ClientContext& context) const;

    std::string grpc_endpoint_;
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<nvmeservice::rpc::NvmeService::Stub> stub_;
};

} // namespace nvmeservice
