#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <string>

#include <grpcpp/server.h>

#include "nvmeservice_state.h"

namespace nvmeservice {

class NvmeServiceServer {
public:
    explicit NvmeServiceServer(std::string grpc_endpoint);

    bool serve(ServiceState& state);
    void requestStop();
    uint32_t leaseTtlMs() const;
    uint64_t createLease(uint32_t controller_index,
                         int32_t pid,
                         uint64_t client_id,
                         const std::vector<uint32_t>& qids);
    bool renewLease(uint64_t lease_id, uint64_t client_id);
    bool releaseLease(uint64_t lease_id);
    size_t releaseLeasesByClient(uint64_t client_id);

private:
    struct LeaseInfo {
        uint32_t controller_index = 0;
        int32_t pid = 0;
        uint64_t client_id = 0;
        std::vector<uint32_t> qids;
        std::chrono::steady_clock::time_point expires_at;
    };

    void leaseReaperLoop();
    void releaseExpiredLeases();

    std::string grpc_endpoint_;
    std::atomic<bool> running_{false};
    std::mutex server_mutex_;
    std::unique_ptr<grpc::Server> server_;
    std::atomic<uint64_t> lease_counter_{1};
    std::chrono::milliseconds lease_ttl_{5000};
    std::mutex leases_mutex_;
    std::unordered_map<uint64_t, LeaseInfo> leases_;
    ServiceState* state_ = nullptr;
    std::thread lease_reaper_;
};

} // namespace nvmeservice
