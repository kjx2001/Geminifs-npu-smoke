#include "nvmeservice_client.h"

#include "shared_ctrl.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace nvmeservice {

namespace {

uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Inverse of server-side set_ipc_bytes: copy 64 raw bytes from proto bytes
// field into a cudaIpcMemHandle_t.
bool ipc_from_bytes(cudaIpcMemHandle_t* out, const std::string& src) {
    if (src.size() != sizeof(cudaIpcMemHandle_t)) return false;
    std::memcpy(out, src.data(), sizeof(cudaIpcMemHandle_t));
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

NvmeServiceClient::Allocation::~Allocation() {
    if (owner != nullptr) {
        owner->release_allocation(this);
    }
}

// ---------------------------------------------------------------------------
// NvmeServiceClient
// ---------------------------------------------------------------------------

NvmeServiceClient::NvmeServiceClient(const std::string& endpoint)
    : endpoint_(endpoint),
      channel_(grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials())),
      stub_(NvmeService::NewStub(channel_))
{}

NvmeServiceClient::~NvmeServiceClient() {
    stop_heartbeat();
}

std::vector<ClientDeviceInfo> NvmeServiceClient::list_devices() {
    std::vector<ClientDeviceInfo> out;

    grpc::ClientContext ctx;
    Empty req;
    DeviceListResponse resp;

    auto status = stub_->ListDevices(&ctx, req, &resp);
    if (!status.ok()) {
        std::fprintf(stderr, "list_devices RPC failed: %s\n",
                     status.error_message().c_str());
        return out;
    }

    out.reserve(resp.devices_size());
    for (const auto& d : resp.devices()) {
        ClientDeviceInfo info;
        info.device_id        = d.device_id();
        info.pci_addr         = d.pci_addr();
        info.snvme_dev_path   = d.snvme_dev_path();
        info.cuda_device      = d.cuda_device();
        info.namespace_id     = d.namespace_id();
        info.page_size        = d.page_size();
        info.blk_size         = d.blk_size();
        info.blk_size_log     = d.blk_size_log();
        info.queue_depth      = d.queue_depth();
        info.total_queues     = d.total_queues();
        info.available_queues = d.available_queues();
        info.queue_groups.reserve(d.queue_groups_size());
        for (const auto& g : d.queue_groups()) {
            ClientQueueGroup cg;
            cg.cuda_device     = g.cuda_device();
            cg.queue_start_idx = g.queue_start_idx();
            cg.queue_count     = g.queue_count();
            cg.available       = g.available();
            info.queue_groups.push_back(cg);
        }
        out.push_back(std::move(info));
    }
    return out;
}

std::unique_ptr<NvmeServiceClient::Allocation>
NvmeServiceClient::allocate(int32_t device_id, int32_t num_queues) {
    // 2-arg overload: pick the first queue_group's cuda_device for the
    // target device. Single-GPU pools see no behaviour change.
    auto devs = list_devices();
    int32_t cuda_dev = -1;
    for (const auto& d : devs) {
        if (d.device_id != device_id) continue;
        if (!d.queue_groups.empty()) {
            cuda_dev = d.queue_groups.front().cuda_device;
        } else {
            cuda_dev = d.cuda_device;  // legacy single-GPU server
        }
        break;
    }
    if (cuda_dev < 0) {
        std::fprintf(stderr, "allocate: device_id %d not found\n", device_id);
        return nullptr;
    }
    return allocate(device_id, cuda_dev, num_queues);
}

std::unique_ptr<NvmeServiceClient::Allocation>
NvmeServiceClient::allocate(int32_t device_id, int32_t cuda_device, int32_t num_queues) {
    grpc::ClientContext ctx;
    AllocRequest req;
    AllocResponse resp;

    req.set_device_id(device_id);
    req.set_cuda_device(cuda_device);
    req.set_num_queues(num_queues);
    req.set_client_pid(static_cast<uint32_t>(::getpid()));

    auto status = stub_->AllocateQueues(&ctx, req, &resp);
    if (!status.ok()) {
        std::fprintf(stderr, "AllocateQueues RPC failed: %s\n",
                     status.error_message().c_str());
        return nullptr;
    }
    if (!resp.error_message().empty()) {
        std::fprintf(stderr, "AllocateQueues rejected: %s\n",
                     resp.error_message().c_str());
        return nullptr;
    }

    // Translate AllocResponse -> SharedControllerSpec
    SharedControllerSpec spec;
    spec.snvme_dev_path = resp.snvme_dev_path();
    spec.bar0_size      = resp.bar0_size();
    spec.dstrd          = resp.dstrd();
    spec.page_size      = resp.page_size();
    spec.blk_size       = resp.blk_size();
    spec.blk_size_log   = resp.blk_size_log();
    spec.namespace_id   = resp.namespace_id();
    spec.cuda_device    = req.cuda_device();
    spec.queue_depth    = resp.queue_depth();
    // mount_path: callers know their own; leave empty here (daemon-bound path)

    spec.queues.reserve(resp.queue_shared_mem_size());
    for (const auto& q : resp.queue_shared_mem()) {
        SharedQueueSpec qs;
        qs.queue_id   = q.queue_id();
        qs.sq_entries = q.sq_entries();
        qs.cq_entries = q.cq_entries();
        qs.sq_ioaddr  = q.sq_ioaddr();
        qs.cq_ioaddr  = q.cq_ioaddr();

        if (!ipc_from_bytes(&qs.sq_handle, q.ipc_handle_sq())) {
            std::fprintf(stderr, "bad SQ IPC handle size for queue %d\n", q.queue_id());
            return nullptr;
        }
        if (!ipc_from_bytes(&qs.cq_handle, q.ipc_handle_cq())) {
            std::fprintf(stderr, "bad CQ IPC handle size for queue %d\n", q.queue_id());
            return nullptr;
        }
        if (!q.ipc_handle_prp().empty()) {
            if (!ipc_from_bytes(&qs.prp_handle, q.ipc_handle_prp())) {
                std::fprintf(stderr, "bad PRP IPC handle size for queue %d\n", q.queue_id());
                return nullptr;
            }
            qs.has_prp = true;
        }
        spec.queues.push_back(qs);
    }

    // Hand off to libnvm to build the local Controller
    std::shared_ptr<Controller> ctrl;
    try {
        ctrl = build_shared_controller(spec);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "build_shared_controller failed: %s\n", e.what());
        // Best-effort release
        grpc::ClientContext rctx;
        ReleaseRequest rreq;
        ReleaseResponse rresp;
        rreq.set_allocation_id(resp.allocation_id());
        rreq.set_client_pid(::getpid());
        stub_->ReleaseQueues(&rctx, rreq, &rresp);
        return nullptr;
    }

    auto alloc = std::make_unique<Allocation>();
    alloc->allocation_id          = resp.allocation_id();
    alloc->device_id              = device_id;
    alloc->queue_start_idx        = resp.queue_start_idx();
    alloc->queue_count            = resp.queue_count();
    alloc->controller             = std::move(ctrl);
    alloc->heartbeat_interval_sec = resp.heartbeat_interval_sec();
    alloc->lease_timeout_sec      = resp.lease_timeout_sec();
    alloc->client_pid             = static_cast<uint32_t>(::getpid());
    alloc->owner                  = this;

    // Register for heartbeat
    {
        std::lock_guard<std::mutex> lock(live_mtx_);
        LiveAlloc la;
        la.allocation_id          = alloc->allocation_id;
        la.heartbeat_interval_sec = alloc->heartbeat_interval_sec;
        live_allocs_.emplace(alloc->allocation_id, std::move(la));
    }
    ensure_heartbeat_started();

    return alloc;
}

void NvmeServiceClient::release_allocation(Allocation* alloc) {
    if (!alloc || alloc->allocation_id.empty()) return;

    {
        std::lock_guard<std::mutex> lock(live_mtx_);
        live_allocs_.erase(alloc->allocation_id);
    }

    grpc::ClientContext ctx;
    ReleaseRequest req;
    ReleaseResponse resp;
    req.set_allocation_id(alloc->allocation_id);
    req.set_client_pid(alloc->client_pid);

    auto status = stub_->ReleaseQueues(&ctx, req, &resp);
    if (!status.ok()) {
        std::fprintf(stderr, "ReleaseQueues RPC failed: %s\n",
                     status.error_message().c_str());
    } else if (!resp.success()) {
        std::fprintf(stderr, "ReleaseQueues rejected: %s\n",
                     resp.error_message().c_str());
    }

    // The local Controller (and all its imported handles, BAR0 mmap, etc.)
    // are released by the shared_ptr deleter in shared_ctrl.cu.
}

// ---------------------------------------------------------------------------
// Heartbeat
// ---------------------------------------------------------------------------

void NvmeServiceClient::ensure_heartbeat_started() {
    if (hb_running_.exchange(true)) return;
    hb_thread_ = std::thread(&NvmeServiceClient::heartbeat_loop, this);
}

void NvmeServiceClient::stop_heartbeat() {
    if (!hb_running_.exchange(false)) return;
    if (hb_thread_.joinable()) hb_thread_.join();
}

void NvmeServiceClient::heartbeat_loop() {
    while (hb_running_.load()) {
        // Snapshot live allocations
        std::vector<std::string> ids;
        uint32_t interval = 10;
        {
            std::lock_guard<std::mutex> lock(live_mtx_);
            if (live_allocs_.empty()) {
                hb_running_ = false;
                break;
            }
            ids.reserve(live_allocs_.size());
            for (const auto& kv : live_allocs_) {
                ids.push_back(kv.first);
                interval = std::min(interval, kv.second.heartbeat_interval_sec);
            }
        }

        // Open one bidi stream per tick. Send all allocations' heartbeats,
        // drain responses, close. Wasteful but trivially correct for low
        // frequencies (default 10s).
        {
            grpc::ClientContext ctx;
            auto stream = stub_->Heartbeat(&ctx);

            for (const auto& aid : ids) {
                HeartbeatMsg msg;
                msg.set_allocation_id(aid);
                msg.set_timestamp_ns(now_ns());
                if (!stream->Write(msg)) break;
            }
            stream->WritesDone();

            HeartbeatMsg resp;
            while (stream->Read(&resp)) {
                if (resp.has_notice() &&
                    resp.notice().kind() == AdminNotice::LEASE_REVOKED) {
                    std::fprintf(stderr,
                        "lease revoked by daemon for allocation %s: %s\n",
                        resp.allocation_id().c_str(),
                        resp.notice().message().c_str());
                    std::lock_guard<std::mutex> lock(live_mtx_);
                    live_allocs_.erase(resp.allocation_id());
                }
            }
            stream->Finish();
        }

        // Wait for next tick (interruptible)
        for (uint32_t i = 0; i < interval && hb_running_.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

} // namespace nvmeservice
