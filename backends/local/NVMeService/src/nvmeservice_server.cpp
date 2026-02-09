#include "nvmeservice_server.h"

#include "nvmeservice_protocol.h"

#include <cstdio>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace nvmeservice {

NvmeServiceServer::NvmeServiceServer(std::string socket_path)
    : socket_path_(std::move(socket_path)) {}

bool NvmeServiceServer::readExact(int fd, void* buf, size_t len) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t read_bytes = 0;
    while (read_bytes < len) {
        ssize_t r = ::read(fd, p + read_bytes, len - read_bytes);
        if (r <= 0) return false;
        read_bytes += static_cast<size_t>(r);
    }
    return true;
}

bool NvmeServiceServer::writeExact(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t written = 0;
    while (written < len) {
        ssize_t w = ::write(fd, p + written, len - written);
        if (w <= 0) return false;
        written += static_cast<size_t>(w);
    }
    return true;
}

bool NvmeServiceServer::handleClient(int client_fd, ServiceState& state) {
    MsgHeader hdr{};
    if (!readExact(client_fd, &hdr, sizeof(hdr))) return false;

    if (hdr.magic != kMagic || hdr.version != kVersion) return false;

    std::vector<uint8_t> payload(hdr.payload_bytes);
    if (hdr.payload_bytes > 0) {
        if (!readExact(client_fd, payload.data(), payload.size())) return false;
    }

    RespHeader resp{};
    resp.magic = kMagic;
    resp.version = kVersion;
    resp.request_id = hdr.request_id;
    resp.status = static_cast<uint16_t>(Status::Ok);
    std::vector<uint8_t> resp_payload;

    auto role = static_cast<Role>(hdr.role);
    auto cmd = static_cast<Command>(hdr.command);

    if (cmd == Command::Ping && role == Role::Filesystem) {
        // no payload
    } else if (cmd == Command::FsGetInfo && role == Role::Filesystem) {
        FsGetInfoResp info{};
        std::snprintf(info.mount_base_path, sizeof(info.mount_base_path), "%s", state.mount_base_path.c_str());
        info.ctrl_count = static_cast<uint32_t>(state.controllers.size());
        info.max_queues_per_process = state.max_queues_per_process;

        resp_payload.resize(sizeof(FsGetInfoResp) + info.ctrl_count * sizeof(CtrlConfig));
        std::memcpy(resp_payload.data(), &info, sizeof(info));
        for (uint32_t i = 0; i < info.ctrl_count; ++i) {
            std::memcpy(resp_payload.data() + sizeof(FsGetInfoResp) + i * sizeof(CtrlConfig),
                        &state.controllers[i].config,
                        sizeof(CtrlConfig));
        }
    } else if (cmd == Command::FsAllocQueues && role == Role::Filesystem) {
        resp.status = static_cast<uint16_t>(Status::Denied);
    } else if (cmd == Command::FsReleaseQueues && role == Role::Filesystem) {
        resp.status = static_cast<uint16_t>(Status::Denied);
    } else {
        resp.status = static_cast<uint16_t>(Status::Denied);
    }

    resp.payload_bytes = static_cast<uint32_t>(resp_payload.size());
    if (!writeExact(client_fd, &resp, sizeof(resp))) return false;
    if (!resp_payload.empty()) {
        if (!writeExact(client_fd, resp_payload.data(), resp_payload.size())) return false;
    }
    return true;
}

bool NvmeServiceServer::serve(ServiceState& state) {
    int server_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) return false;

    ::unlink(socket_path_.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path_.c_str());

    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(server_fd);
        return false;
    }

    if (::listen(server_fd, 16) < 0) {
        ::close(server_fd);
        return false;
    }

    while (true) {
        int client_fd = ::accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;
        handleClient(client_fd, state);
        ::close(client_fd);
    }

    ::close(server_fd);
    return true;
}

} // namespace nvmeservice
