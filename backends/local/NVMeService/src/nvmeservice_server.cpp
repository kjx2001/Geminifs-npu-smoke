#include "nvmeservice_server.h"

#include <cstring>

namespace nvmeservice {

namespace {

// Copy a cudaIpcMemHandle_t (64 raw bytes) into a proto bytes field.
inline void set_ipc_bytes(std::string* dst, const cudaIpcMemHandle_t& h) {
    dst->assign(reinterpret_cast<const char*>(&h), sizeof(cudaIpcMemHandle_t));
}

} // namespace

NvmeServiceImpl::NvmeServiceImpl(std::shared_ptr<ServiceState> state)
    : state_(std::move(state)) {}

// ---------------------------------------------------------------------------
// ListDevices
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::ListDevices(grpc::ServerContext* /*ctx*/,
                                           const Empty* /*request*/,
                                           DeviceListResponse* response) {
    const auto snaps = state_->list_devices();
    for (const auto& s : snaps) {
        DeviceInfo* di = response->add_devices();
        di->set_device_id(s.device_id);
        di->set_pci_addr(s.pci_addr);
        di->set_snvme_dev_path(s.snvme_dev_path);
        di->set_cuda_device(s.cuda_device);
        di->set_namespace_id(s.namespace_id);
        di->set_page_size(s.page_size);
        di->set_blk_size(s.blk_size);
        di->set_blk_size_log(s.blk_size_log);
        di->set_queue_depth(s.queue_depth);
        di->set_dstrd(s.dstrd);
        di->set_bar0_size(s.bar0_size);
        di->set_total_queues(s.total_queues);
        di->set_available_queues(s.available_queues);
        for (const auto& g : s.groups) {
            QueueGroupInfo* qg = di->add_queue_groups();
            qg->set_cuda_device(g.cuda_device);
            qg->set_queue_start_idx(g.queue_start_idx);
            qg->set_queue_count(g.queue_count);
            qg->set_available(g.available);
        }
    }
    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// AllocateQueues
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::AllocateQueues(grpc::ServerContext* /*ctx*/,
                                              const AllocRequest* request,
                                              AllocResponse* response) {
    auto result = state_->allocate(request->device_id(),
                                    request->cuda_device(),
                                    request->num_queues(),
                                    request->client_pid());

    if (!result.success) {
        response->set_error_message(result.error);
        return grpc::Status::OK;  // report via error_message, not gRPC status
    }

    const auto& g = result.grant;
    response->set_allocation_id(g.allocation_id);
    response->set_pci_addr(g.pci_addr);
    response->set_snvme_dev_path(g.snvme_dev_path);
    response->set_bar0_size(g.bar0_size);
    response->set_dstrd(g.dstrd);
    response->set_queue_start_idx(g.queue_start_idx);
    response->set_queue_count(g.queue_count);
    response->set_namespace_id(g.namespace_id);
    response->set_page_size(g.page_size);
    response->set_blk_size(g.blk_size);
    response->set_blk_size_log(g.blk_size_log);
    response->set_queue_depth(g.queue_depth);
    response->set_heartbeat_interval_sec(g.heartbeat_interval_sec);
    response->set_lease_timeout_sec(g.lease_timeout_sec);

    for (const auto& qs : g.queue_shared) {
        QueueSharedMem* m = response->add_queue_shared_mem();
        m->set_queue_id(qs.queue_id);

        std::string sq_bytes;
        set_ipc_bytes(&sq_bytes, qs.ipc_sq);
        m->set_ipc_handle_sq(std::move(sq_bytes));

        std::string cq_bytes;
        set_ipc_bytes(&cq_bytes, qs.ipc_cq);
        m->set_ipc_handle_cq(std::move(cq_bytes));

        if (qs.has_prp) {
            std::string prp_bytes;
            set_ipc_bytes(&prp_bytes, qs.ipc_prp);
            m->set_ipc_handle_prp(std::move(prp_bytes));
        }  // else leave empty -> client allocates its own PRP

        m->set_sq_entries(qs.sq_entries);
        m->set_cq_entries(qs.cq_entries);
        m->set_sq_ioaddr(qs.sq_ioaddr);
        m->set_cq_ioaddr(qs.cq_ioaddr);
    }

    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// ReleaseQueues
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::ReleaseQueues(grpc::ServerContext* /*ctx*/,
                                             const ReleaseRequest* request,
                                             ReleaseResponse* response) {
    std::string err;
    bool ok = state_->release(request->allocation_id(),
                               request->client_pid(),
                               &err);
    response->set_success(ok);
    if (!ok) response->set_error_message(err);
    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// Heartbeat (bidi stream)
// ---------------------------------------------------------------------------

grpc::Status NvmeServiceImpl::Heartbeat(grpc::ServerContext* /*ctx*/,
                                         grpc::ServerReaderWriter<HeartbeatMsg, HeartbeatMsg>* stream) {
    HeartbeatMsg in;
    while (stream->Read(&in)) {
        std::string err;
        bool ok = state_->update_heartbeat(in.allocation_id(), &err);

        if (!ok) {
            // Allocation gone (probably reclaimed). Notify and close.
            HeartbeatMsg out;
            out.set_allocation_id(in.allocation_id());
            out.set_timestamp_ns(in.timestamp_ns());
            auto* notice = out.mutable_notice();
            notice->set_kind(AdminNotice::LEASE_REVOKED);
            notice->set_message(err);
            stream->Write(out);
            return grpc::Status::OK;
        }

        // Echo with current timestamp (client can estimate skew).
        HeartbeatMsg out;
        out.set_allocation_id(in.allocation_id());
        out.set_timestamp_ns(in.timestamp_ns());
        stream->Write(out);
    }
    return grpc::Status::OK;
}

} // namespace nvmeservice
