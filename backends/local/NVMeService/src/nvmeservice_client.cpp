#include "nvmeservice_client.h"

#include <cstdio>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace nvmeservice {

namespace {

bool readExact(int fd, void* buf, size_t len) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t read_bytes = 0;
    while (read_bytes < len) {
        ssize_t r = ::read(fd, p + read_bytes, len - read_bytes);
        if (r <= 0) return false;
        read_bytes += static_cast<size_t>(r);
    }
    return true;
}

bool writeExact(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t written = 0;
    while (written < len) {
        ssize_t w = ::write(fd, p + written, len - written);
        if (w <= 0) return false;
        written += static_cast<size_t>(w);
    }
    return true;
}

} // namespace

NvmeServiceClient::NvmeServiceClient(std::string socket_path)
    : socket_path_(std::move(socket_path)) {}

bool NvmeServiceClient::request(const MsgHeader& hdr, const void* payload, uint32_t payload_bytes, RespHeader& resp, std::vector<uint8_t>& resp_payload) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path_.c_str());

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return false;
    }

    if (!writeExact(fd, &hdr, sizeof(hdr))) {
        ::close(fd);
        return false;
    }
    if (payload_bytes > 0 && payload != nullptr) {
        if (!writeExact(fd, payload, payload_bytes)) {
            ::close(fd);
            return false;
        }
    }

    if (!readExact(fd, &resp, sizeof(resp))) {
        ::close(fd);
        return false;
    }

    resp_payload.resize(resp.payload_bytes);
    if (resp.payload_bytes > 0) {
        if (!readExact(fd, resp_payload.data(), resp_payload.size())) {
            ::close(fd);
            return false;
        }
    }

    ::close(fd);
    return resp.magic == kMagic && resp.version == kVersion;
}

bool NvmeServiceClient::ping() {
    MsgHeader hdr{};
    hdr.magic = kMagic;
    hdr.version = kVersion;
    hdr.role = static_cast<uint16_t>(Role::Filesystem);
    hdr.command = static_cast<uint16_t>(Command::Ping);
    hdr.request_id = 1;
    hdr.payload_bytes = 0;

    RespHeader resp{};
    std::vector<uint8_t> payload;
    if (!request(hdr, nullptr, 0, resp, payload)) return false;
    return resp.status == static_cast<uint16_t>(Status::Ok);
}

bool NvmeServiceClient::getInfo(std::string& mount_base_path, std::vector<CtrlConfig>& ctrls, uint32_t& max_queues_per_process) {
    MsgHeader hdr{};
    hdr.magic = kMagic;
    hdr.version = kVersion;
    hdr.role = static_cast<uint16_t>(Role::Filesystem);
    hdr.command = static_cast<uint16_t>(Command::FsGetInfo);
    hdr.request_id = 2;
    hdr.payload_bytes = 0;

    RespHeader resp{};
    std::vector<uint8_t> payload;
    if (!request(hdr, nullptr, 0, resp, payload)) return false;
    if (resp.status != static_cast<uint16_t>(Status::Ok)) return false;
    if (payload.size() < sizeof(FsGetInfoResp)) return false;

    FsGetInfoResp info{};
    std::memcpy(&info, payload.data(), sizeof(info));
    mount_base_path = info.mount_base_path;
    max_queues_per_process = info.max_queues_per_process;

    size_t expected = sizeof(FsGetInfoResp) + info.ctrl_count * sizeof(CtrlConfig);
    if (payload.size() < expected) return false;

    ctrls.resize(info.ctrl_count);
    if (info.ctrl_count > 0) {
        std::memcpy(ctrls.data(), payload.data() + sizeof(FsGetInfoResp), info.ctrl_count * sizeof(CtrlConfig));
    }
    return true;
}

} // namespace nvmeservice
