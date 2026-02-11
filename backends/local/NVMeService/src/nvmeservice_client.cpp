#include "nvmeservice_client.h"

#include "nvmeservice.pb.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <chrono>
#include <cstdio>
#include <thread>

namespace nvmeservice {

namespace rpc = nvmeservice::rpc;

namespace {

constexpr int kDefaultRpcTimeoutMs = 2000;
constexpr int kMaxGrpcMessageBytes = 4 * 1024 * 1024;
constexpr uint32_t kQueueRequestAlign = 16;
constexpr uint32_t kDefaultQueueRequest = 32;

uint32_t normalizeQueueRequest(uint32_t requested) {
    uint32_t count = requested == 0 ? kDefaultQueueRequest : requested;
    if (count % kQueueRequestAlign != 0) {
        count = ((count + kQueueRequestAlign - 1) / kQueueRequestAlign) * kQueueRequestAlign;
    }
    return count;
}

} // namespace

NvmeServiceClient::NvmeServiceClient(std::string grpc_endpoint)
    : grpc_endpoint_(std::move(grpc_endpoint)) {
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(kMaxGrpcMessageBytes);
    args.SetMaxSendMessageSize(kMaxGrpcMessageBytes);
    channel_ = grpc::CreateCustomChannel(grpc_endpoint_, grpc::InsecureChannelCredentials(), args);
    stub_ = rpc::NvmeService::NewStub(channel_);
}

void NvmeServiceClient::applyDeadline(grpc::ClientContext& context) const {
    auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(kDefaultRpcTimeoutMs);
    context.set_deadline(deadline);
}

bool NvmeServiceClient::ping() {
    grpc::ClientContext context;
    applyDeadline(context);

    rpc::PingReq req;
    rpc::PingResp resp;
    grpc::Status status = stub_->Ping(&context, req, &resp);
    return status.ok() && resp.status() == rpc::Status::STATUS_OK;
}

bool NvmeServiceClient::getInfo(std::string& mount_base_path, std::vector<CtrlConfig>& ctrls, uint32_t& max_queues_per_process) {
    grpc::ClientContext context;
    applyDeadline(context);

    rpc::FsGetInfoReq req;
    rpc::FsGetInfoResp resp;
    grpc::Status status = stub_->FsGetInfo(&context, req, &resp);
    if (!status.ok() || resp.status() != rpc::Status::STATUS_OK) {
        return false;
    }

    mount_base_path = resp.mount_base_path();
    max_queues_per_process = resp.max_queues_per_process();
    ctrls.clear();
    ctrls.reserve(static_cast<size_t>(resp.ctrls_size()));
    for (const auto& ctrl : resp.ctrls()) {
        CtrlConfig cfg{};
        std::snprintf(cfg.mount_path, sizeof(cfg.mount_path), "%s", ctrl.mount_path().c_str());
        std::snprintf(cfg.pci_addr, sizeof(cfg.pci_addr), "%s", ctrl.pci_addr().c_str());
        cfg.ns_id = ctrl.ns_id();
        cfg.queue_depth = ctrl.queue_depth();
        cfg.num_queues = ctrl.num_queues();
        cfg.cuda_device = ctrl.cuda_device();
        cfg.max_io_kb = ctrl.max_io_kb();
        ctrls.push_back(cfg);
    }

    return true;
}

bool NvmeServiceClient::shutdown() {
    grpc::ClientContext context;
    applyDeadline(context);

    rpc::ShutdownReq req;
    rpc::ShutdownResp resp;
    grpc::Status status = stub_->Shutdown(&context, req, &resp);
    return status.ok() && resp.status() == rpc::Status::STATUS_OK;
}

bool NvmeServiceClient::allocQueues(uint32_t controller_index,
                                    uint32_t requested,
                                    int32_t pid,
                                    uint64_t client_id,
                                    std::vector<uint32_t>& qids,
                                    std::string& d_qps_handle,
                                    std::string& d_ctrl_handle,
                                    uint64_t& lease_id,
                                    uint32_t& ttl_ms) {
    grpc::ClientContext context;
    applyDeadline(context);

    rpc::FsAllocQueuesReq req;
    req.set_controller_index(controller_index);
    req.set_queue_count(normalizeQueueRequest(requested));
    req.set_pid(pid);
    req.set_client_id(client_id);

    rpc::FsAllocQueuesResp resp;
    grpc::Status status = stub_->FsAllocQueues(&context, req, &resp);
    if (!status.ok() || resp.status() != rpc::Status::STATUS_OK) {
        return false;
    }

    qids.clear();
    qids.reserve(static_cast<size_t>(resp.qids_size()));
    for (int i = 0; i < resp.qids_size(); ++i) {
        qids.push_back(resp.qids(i));
    }
    d_qps_handle = resp.d_qps_handle();
    d_ctrl_handle = resp.d_ctrl_handle();
    lease_id = resp.lease_id();
    ttl_ms = resp.ttl_ms();
    return true;
}

bool NvmeServiceClient::releaseQueues(uint32_t controller_index, int32_t pid, const std::vector<uint32_t>& qids) {
    grpc::ClientContext context;
    applyDeadline(context);

    rpc::FsReleaseQueuesReq req;
    req.set_controller_index(controller_index);
    req.set_pid(pid);
    for (uint32_t qid : qids) {
        req.add_qids(qid);
    }

    rpc::FsReleaseQueuesResp resp;
    grpc::Status status = stub_->FsReleaseQueues(&context, req, &resp);
    return status.ok() && resp.status() == rpc::Status::STATUS_OK;
}

bool NvmeServiceClient::releaseLease(uint64_t lease_id) {
    grpc::ClientContext context;
    applyDeadline(context);

    rpc::FsReleaseQueuesReq req;
    req.set_lease_id(lease_id);

    rpc::FsReleaseQueuesResp resp;
    grpc::Status status = stub_->FsReleaseQueues(&context, req, &resp);
    return status.ok() && resp.status() == rpc::Status::STATUS_OK;
}

bool NvmeServiceClient::heartbeatLeases(uint64_t client_id,
                                        const std::vector<uint64_t>& lease_ids,
                                        uint32_t duration_ms,
                                        uint32_t interval_ms) {
    grpc::ClientContext context;
    auto stream = stub_->LeaseHeartbeat(&context);
    if (!stream) {
        return false;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        for (uint64_t lease_id : lease_ids) {
            rpc::LeaseHeartbeatReq req;
            req.set_lease_id(lease_id);
            req.set_client_id(client_id);
            if (!stream->Write(req)) {
                break;
            }

            rpc::LeaseHeartbeatResp resp;
            if (!stream->Read(&resp)) {
                break;
            }
            if (resp.status() != rpc::Status::STATUS_OK) {
                return false;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    stream->WritesDone();
    grpc::Status status = stream->Finish();
    return status.ok();
}

} // namespace nvmeservice
