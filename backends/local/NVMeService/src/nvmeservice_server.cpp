#include "nvmeservice_server.h"

#include "nvmeservice.grpc.pb.h"

#include <grpcpp/grpcpp.h>

#include "ctrl.h"

#include <cuda_runtime.h>

#include <vector>

#include <chrono>
#include <string>
#include <thread>

namespace nvmeservice {

namespace {

namespace rpc = nvmeservice::rpc;

constexpr uint32_t kQueueRequestAlign = 16;
constexpr uint32_t kDefaultQueueRequest = 32;
constexpr int kMaxGrpcMessageBytes = 4 * 1024 * 1024;
constexpr uint32_t kDefaultLeaseTtlMs = 5000;

uint32_t normalizeQueueRequest(uint32_t requested, uint32_t max_allowed) {
    uint32_t count = requested == 0 ? kDefaultQueueRequest : requested;
    if (count % kQueueRequestAlign != 0) {
        count = ((count + kQueueRequestAlign - 1) / kQueueRequestAlign) * kQueueRequestAlign;
    }
    if (max_allowed > 0 && count > max_allowed) {
        count = (max_allowed / kQueueRequestAlign) * kQueueRequestAlign;
        if (count == 0) {
            count = kQueueRequestAlign;
        }
    }
    return count;
}

class NvmeServiceImpl final : public rpc::NvmeService::Service {
public:
    NvmeServiceImpl(ServiceState& state, NvmeServiceServer& owner)
        : state_(state), owner_(owner) {}

    grpc::Status Ping(grpc::ServerContext*, const rpc::PingReq*, rpc::PingResp* resp) override {
        resp->set_status(rpc::Status::STATUS_OK);
        return grpc::Status::OK;
    }

    grpc::Status FsGetInfo(grpc::ServerContext*, const rpc::FsGetInfoReq*, rpc::FsGetInfoResp* resp) override {
        resp->set_status(rpc::Status::STATUS_OK);
        resp->set_mount_base_path(state_.mount_base_path);
        resp->set_max_queues_per_process(state_.max_queues_per_process);

        for (const auto& entry : state_.controllers) {
            auto* ctrl = resp->add_ctrls();
            ctrl->set_mount_path(entry.config.mount_path);
            ctrl->set_pci_addr(entry.config.pci_addr);
            ctrl->set_ns_id(entry.config.ns_id);
            ctrl->set_queue_depth(entry.config.queue_depth);
            ctrl->set_num_queues(entry.config.num_queues);
            ctrl->set_cuda_device(entry.config.cuda_device);
            ctrl->set_max_io_kb(entry.config.max_io_kb);
        }

        return grpc::Status::OK;
    }

    grpc::Status FsAllocQueues(grpc::ServerContext*, const rpc::FsAllocQueuesReq* req, rpc::FsAllocQueuesResp* resp) override {
        const uint32_t controller_index = req->controller_index();
        const uint32_t requested = req->queue_count();
        const int32_t pid = req->pid();
        const uint64_t client_id = req->client_id();

        uint32_t count = normalizeQueueRequest(requested, state_.max_queues_per_process);
        std::vector<uint32_t> qids;
        if (!state_.allocQueues(controller_index, count, pid, qids)) {
            resp->set_status(rpc::Status::STATUS_DENIED);
            resp->set_granted(0);
            return grpc::Status::OK;
        }

        resp->set_status(rpc::Status::STATUS_OK);
        resp->set_granted(static_cast<uint32_t>(qids.size()));
        for (uint32_t qid : qids) {
            resp->add_qids(qid);
        }

        const uint64_t lease_id = owner_.createLease(controller_index, pid, client_id, qids);
        resp->set_lease_id(lease_id);
        resp->set_ttl_ms(owner_.leaseTtlMs());

        if (controller_index < state_.controllers.size()) {
            const auto& entry = state_.controllers[controller_index];
            if (!entry.controller) {
                return grpc::Status::OK;
            }
            cudaIpcMemHandle_t d_qps_handle{};
            cudaIpcMemHandle_t d_ctrl_handle{};

            cudaError_t qps_status = cudaIpcGetMemHandle(&d_qps_handle, entry.controller->d_qps);
            cudaError_t ctrl_status = cudaIpcGetMemHandle(&d_ctrl_handle, entry.controller->d_ctrl_ptr);

            if (qps_status == cudaSuccess) {
                resp->set_d_qps_handle(std::string(reinterpret_cast<const char*>(&d_qps_handle), sizeof(d_qps_handle)));
            }
            if (ctrl_status == cudaSuccess) {
                resp->set_d_ctrl_handle(std::string(reinterpret_cast<const char*>(&d_ctrl_handle), sizeof(d_ctrl_handle)));
            }
        }
        return grpc::Status::OK;
    }

    grpc::Status FsReleaseQueues(grpc::ServerContext*, const rpc::FsReleaseQueuesReq* req, rpc::FsReleaseQueuesResp* resp) override {
        const uint32_t controller_index = req->controller_index();
        const int32_t pid = req->pid();

        std::vector<uint32_t> qids;
        qids.reserve(static_cast<size_t>(req->qids_size()));
        for (int i = 0; i < req->qids_size(); ++i) {
            qids.push_back(req->qids(i));
        }

        if (!state_.releaseQueues(controller_index, pid, qids)) {
            resp->set_status(rpc::Status::STATUS_DENIED);
            return grpc::Status::OK;
        }

        resp->set_status(rpc::Status::STATUS_OK);
        return grpc::Status::OK;
    }

    grpc::Status Shutdown(grpc::ServerContext*, const rpc::ShutdownReq*, rpc::ShutdownResp* resp) override {
        resp->set_status(rpc::Status::STATUS_OK);
        owner_.requestStop();
        return grpc::Status::OK;
    }

    grpc::Status LeaseHeartbeat(grpc::ServerContext*,
                                grpc::ServerReaderWriter<rpc::LeaseHeartbeatResp, rpc::LeaseHeartbeatReq>* stream) override {
        rpc::LeaseHeartbeatReq req;
        uint64_t client_id = 0;

        while (stream->Read(&req)) {
            if (client_id == 0) {
                client_id = req.client_id();
            }
            rpc::LeaseHeartbeatResp resp;
            resp.set_lease_id(req.lease_id());
            resp.set_ttl_ms(owner_.leaseTtlMs());

            if (req.lease_id() == 0 || req.client_id() == 0) {
                resp.set_status(rpc::Status::STATUS_INVALID);
            } else if (owner_.renewLease(req.lease_id(), req.client_id())) {
                resp.set_status(rpc::Status::STATUS_OK);
            } else {
                resp.set_status(rpc::Status::STATUS_DENIED);
            }

            if (!stream->Write(resp)) {
                break;
            }
        }

        if (client_id != 0) {
            owner_.releaseLeasesByClient(client_id);
        }
        return grpc::Status::OK;
    }

private:
    ServiceState& state_;
    NvmeServiceServer& owner_;
};

} // namespace

NvmeServiceServer::NvmeServiceServer(std::string grpc_endpoint)
    : grpc_endpoint_(std::move(grpc_endpoint)), lease_ttl_(kDefaultLeaseTtlMs) {}

void NvmeServiceServer::requestStop() {
    running_.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lock(server_mutex_);
    if (server_) {
        server_->Shutdown();
    }
}

uint32_t NvmeServiceServer::leaseTtlMs() const {
    return static_cast<uint32_t>(lease_ttl_.count());
}

uint64_t NvmeServiceServer::createLease(uint32_t controller_index,
                                        int32_t pid,
                                        uint64_t client_id,
                                        const std::vector<uint32_t>& qids) {
    uint64_t lease_id = lease_counter_.fetch_add(1, std::memory_order_relaxed);
    LeaseInfo info;
    info.controller_index = controller_index;
    info.pid = pid;
    info.client_id = client_id;
    info.qids = qids;
    info.expires_at = std::chrono::steady_clock::now() + lease_ttl_;

    std::lock_guard<std::mutex> lock(leases_mutex_);
    leases_[lease_id] = std::move(info);
    return lease_id;
}

bool NvmeServiceServer::renewLease(uint64_t lease_id, uint64_t client_id) {
    std::lock_guard<std::mutex> lock(leases_mutex_);
    auto it = leases_.find(lease_id);
    if (it == leases_.end() || it->second.client_id != client_id) {
        return false;
    }
    it->second.expires_at = std::chrono::steady_clock::now() + lease_ttl_;
    return true;
}

bool NvmeServiceServer::releaseLease(uint64_t lease_id) {
    LeaseInfo info;
    {
        std::lock_guard<std::mutex> lock(leases_mutex_);
        auto it = leases_.find(lease_id);
        if (it == leases_.end()) {
            return false;
        }
        info = std::move(it->second);
        leases_.erase(it);
    }

    if (!state_) {
        return false;
    }
    return state_->releaseQueues(info.controller_index, info.pid, info.qids);
}

size_t NvmeServiceServer::releaseLeasesByClient(uint64_t client_id) {
    std::vector<uint64_t> to_release;
    {
        std::lock_guard<std::mutex> lock(leases_mutex_);
        for (const auto& kv : leases_) {
            if (kv.second.client_id == client_id) {
                to_release.push_back(kv.first);
            }
        }
    }

    for (uint64_t lease_id : to_release) {
        releaseLease(lease_id);
    }
    return to_release.size();
}

void NvmeServiceServer::releaseExpiredLeases() {
    std::vector<uint64_t> expired;
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(leases_mutex_);
        for (const auto& kv : leases_) {
            if (kv.second.expires_at <= now) {
                expired.push_back(kv.first);
            }
        }
    }

    for (uint64_t lease_id : expired) {
        releaseLease(lease_id);
    }
}

void NvmeServiceServer::leaseReaperLoop() {
    while (running_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        releaseExpiredLeases();
    }
}

bool NvmeServiceServer::serve(ServiceState& state) {
    NvmeServiceImpl service(state, *this);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(grpc_endpoint_, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    builder.SetMaxReceiveMessageSize(kMaxGrpcMessageBytes);
    builder.SetMaxSendMessageSize(kMaxGrpcMessageBytes);

    {
        std::lock_guard<std::mutex> lock(server_mutex_);
        server_ = builder.BuildAndStart();
    }

    if (!server_) {
        return false;
    }

    state_ = &state;
    running_.store(true, std::memory_order_release);
    lease_reaper_ = std::thread([this]() { leaseReaperLoop(); });
    server_->Wait();
    running_.store(false, std::memory_order_release);
    if (lease_reaper_.joinable()) {
        lease_reaper_.join();
    }

    std::lock_guard<std::mutex> lock(server_mutex_);
    server_.reset();
    state_ = nullptr;
    return true;
}

} // namespace nvmeservice
