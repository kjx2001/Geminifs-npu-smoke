#include "nvmeservice_client.h"

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace nvmeservice {

NvmeServiceClient::NvmeServiceClient(std::string socket_path)
    : socket_path_(std::move(socket_path)) {}

static std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(static_cast<unsigned char>(*start))) {
        ++start;
    }
    auto end = s.end();
    do {
        --end;
    } while (end >= start && std::isspace(static_cast<unsigned char>(*end)));
    return (start <= end) ? std::string(start, end + 1) : std::string();
}

static std::string stripQuotes(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

static bool ensureSnvmeModuleLoaded() {
    if (::access("/dev/snvm_control", F_OK) == 0) {
        return true;
    }
    int ret = std::system("modprobe snvme");
    if (ret == 0) {
        return ::access("/dev/snvm_control", F_OK) == 0;
    }
    return false;
}

struct ParsedConfig {
    std::string mount_base_path;
    int gpu_id = -1;
    std::vector<CtrlConfig> ctrls;
};

static bool parseSysConfig(const std::string& path, int target_gpu_id, ParsedConfig& out) {
    std::ifstream in(path);
    if (!in.is_open()) return false;

    std::string line;
    std::string section;
    CtrlConfig current_ctrl{};
    bool in_nvme = false;
    bool in_gpu = false;

    auto flush_nvme = [&]() {
        if (!in_nvme) return;
        if (current_ctrl.pci_addr[0] == '\0') return;
        if (current_ctrl.cuda_device != static_cast<uint32_t>(target_gpu_id)) return;
        if (current_ctrl.mount_path[0] == '\0') return;
        out.ctrls.push_back(current_ctrl);
    };

    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;

        if (line.front() == '<' && line.back() == '>') {
            flush_nvme();
            section = line.substr(1, line.size() - 2);
            in_gpu = section.rfind("GPU", 0) == 0;
            in_nvme = section.rfind("NVMe", 0) == 0;
            if (in_nvme) {
                current_ctrl = {};
            }
            continue;
        }

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string value = stripQuotes(trim(line.substr(eq + 1)));

        if (in_gpu) {
            if (key == "mount_path") {
                out.mount_base_path = value;
            } else if (key == "cudaDevice") {
                out.gpu_id = std::stoi(value);
            }
        } else if (in_nvme) {
            if (key == "mount_path") {
                std::string mp = value;
                if (!mp.empty() && mp.front() != '/') {
                    if (!out.mount_base_path.empty()) {
                        mp = out.mount_base_path + "/" + mp;
                    }
                }
                std::snprintf(current_ctrl.mount_path, sizeof(current_ctrl.mount_path), "%s", mp.c_str());
            } else if (key == "pci_addr") {
                std::snprintf(current_ctrl.pci_addr, sizeof(current_ctrl.pci_addr), "%s", value.c_str());
            } else if (key == "ns_id") {
                current_ctrl.ns_id = static_cast<uint32_t>(std::stoul(value));
            } else if (key == "queueDepth") {
                current_ctrl.queue_depth = static_cast<uint32_t>(std::stoul(value));
            } else if (key == "numQueues") {
                current_ctrl.num_queues = static_cast<uint32_t>(std::stoul(value));
            } else if (key == "cudaDevice") {
                current_ctrl.cuda_device = static_cast<uint32_t>(std::stoul(value));
            } else if (key == "maxIOsize") {
                current_ctrl.max_io_kb = static_cast<uint32_t>(std::stoul(value));
            }
        }
    }

    flush_nvme();

    if (out.gpu_id != target_gpu_id) return false;
    return !out.mount_base_path.empty() && !out.ctrls.empty();
}

static bool readExact(int fd, void* buf, size_t len) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t read_bytes = 0;
    while (read_bytes < len) {
        ssize_t r = ::read(fd, p + read_bytes, len - read_bytes);
        if (r <= 0) return false;
        read_bytes += static_cast<size_t>(r);
    }
    return true;
}

static bool writeExact(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t written = 0;
    while (written < len) {
        ssize_t w = ::write(fd, p + written, len - written);
        if (w <= 0) return false;
        written += static_cast<size_t>(w);
    }
    return true;
}

bool NvmeServiceClient::request(const MsgHeader& hdr, const void* payload, uint32_t payload_bytes,
                                RespHeader& resp, std::vector<uint8_t>& resp_payload) {
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
    if (payload_bytes > 0 && payload) {
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
    return true;
}

bool NvmeServiceClient::getInfo(FsGetInfoResp& out_header, std::vector<CtrlConfig>& out_ctrls) {
    MsgHeader hdr{};
    hdr.magic = kMagic;
    hdr.version = kVersion;
    hdr.role = static_cast<uint16_t>(Role::Filesystem);
    hdr.command = static_cast<uint16_t>(Command::FsGetInfo);
    hdr.request_id = 1;
    hdr.payload_bytes = 0;

    RespHeader resp{};
    std::vector<uint8_t> payload;
    if (!request(hdr, nullptr, 0, resp, payload)) return false;
    if (resp.status != static_cast<uint16_t>(Status::Ok)) return false;
    if (payload.size() < sizeof(FsGetInfoResp)) return false;

    std::memcpy(&out_header, payload.data(), sizeof(FsGetInfoResp));
    out_ctrls.resize(out_header.ctrl_count);
    size_t expected = sizeof(FsGetInfoResp) + out_header.ctrl_count * sizeof(CtrlConfig);
    if (payload.size() < expected) return false;

    for (uint32_t i = 0; i < out_header.ctrl_count; ++i) {
        std::memcpy(&out_ctrls[i],
                    payload.data() + sizeof(FsGetInfoResp) + i * sizeof(CtrlConfig),
                    sizeof(CtrlConfig));
    }
    return true;
}

bool NvmeServiceClient::allocQueues(uint32_t controller_index, uint32_t count, int32_t pid, std::vector<uint32_t>& out_qids) {
    FsAllocQueuesReq req{};
    req.controller_index = controller_index;
    req.queue_count = count;
    req.pid = pid;

    MsgHeader hdr{};
    hdr.magic = kMagic;
    hdr.version = kVersion;
    hdr.role = static_cast<uint16_t>(Role::Filesystem);
    hdr.command = static_cast<uint16_t>(Command::FsAllocQueues);
    hdr.request_id = 2;
    hdr.payload_bytes = sizeof(req);

    RespHeader resp{};
    std::vector<uint8_t> payload;
    if (!request(hdr, &req, sizeof(req), resp, payload)) return false;
    if (resp.status != static_cast<uint16_t>(Status::Ok)) return false;
    if (payload.size() < sizeof(FsAllocQueuesResp)) return false;

    FsAllocQueuesResp ar{};
    std::memcpy(&ar, payload.data(), sizeof(ar));
    if (payload.size() < sizeof(FsAllocQueuesResp) + ar.granted * sizeof(uint32_t)) return false;

    out_qids.resize(ar.granted);
    std::memcpy(out_qids.data(), payload.data() + sizeof(FsAllocQueuesResp), ar.granted * sizeof(uint32_t));
    return true;
}

bool NvmeServiceClient::releaseQueues(uint32_t controller_index, int32_t pid, const std::vector<uint32_t>& qids) {
    FsReleaseQueuesReq req{};
    req.controller_index = controller_index;
    req.queue_count = static_cast<uint32_t>(qids.size());
    req.pid = pid;

    std::vector<uint8_t> payload(sizeof(FsReleaseQueuesReq) + qids.size() * sizeof(uint32_t));
    std::memcpy(payload.data(), &req, sizeof(req));
    std::memcpy(payload.data() + sizeof(FsReleaseQueuesReq), qids.data(), qids.size() * sizeof(uint32_t));

    MsgHeader hdr{};
    hdr.magic = kMagic;
    hdr.version = kVersion;
    hdr.role = static_cast<uint16_t>(Role::Filesystem);
    hdr.command = static_cast<uint16_t>(Command::FsReleaseQueues);
    hdr.request_id = 3;
    hdr.payload_bytes = static_cast<uint32_t>(payload.size());

    RespHeader resp{};
    std::vector<uint8_t> resp_payload;
    if (!request(hdr, payload.data(), hdr.payload_bytes, resp, resp_payload)) return false;
    return resp.status == static_cast<uint16_t>(Status::Ok);
}

bool NvmeServiceClient::adminInitFromConfig(const std::string& config_path, int gpu_id, bool start_module) {
    ParsedConfig parsed;
    if (!parseSysConfig(config_path, gpu_id, parsed)) return false;
    return adminInit(parsed.mount_base_path, parsed.ctrls, start_module);
}

bool NvmeServiceClient::adminInit(const std::string& mount_base_path, const std::vector<CtrlConfig>& ctrls, bool start_module) {
    if (ctrls.empty()) return false;
    if (start_module && !ensureSnvmeModuleLoaded()) return false;

    AdminInitReq req{};
    std::snprintf(req.mount_base_path, sizeof(req.mount_base_path), "%s", mount_base_path.c_str());
    req.ctrl_count = static_cast<uint32_t>(ctrls.size());

    std::vector<uint8_t> payload(sizeof(AdminInitReq) + ctrls.size() * sizeof(CtrlConfig));
    std::memcpy(payload.data(), &req, sizeof(req));
    std::memcpy(payload.data() + sizeof(AdminInitReq), ctrls.data(), ctrls.size() * sizeof(CtrlConfig));

    MsgHeader hdr{};
    hdr.magic = kMagic;
    hdr.version = kVersion;
    hdr.role = static_cast<uint16_t>(Role::Admin);
    hdr.command = static_cast<uint16_t>(Command::AdminInit);
    hdr.request_id = 100;
    hdr.payload_bytes = static_cast<uint32_t>(payload.size());

    RespHeader resp{};
    std::vector<uint8_t> resp_payload;
    if (!request(hdr, payload.data(), hdr.payload_bytes, resp, resp_payload)) return false;
    if (resp.status != static_cast<uint16_t>(Status::Ok)) return false;

    controllers_.clear();
    controllers_.reserve(ctrls.size());
    for (const auto& c : ctrls) {
        nvme_ctrl_param p{};
        p.mount_path = c.mount_path;
        p.pci_addr = c.pci_addr;
        p.cudaDevice = static_cast<int>(c.cuda_device);
        p.ns_id = c.ns_id;
        p.queueDepth = c.queue_depth;
        p.numQueues = c.num_queues;
        p.maxIOsize = c.max_io_kb;

        try {
            controllers_.push_back(std::make_shared<NVMeController>(p));
        } catch (...) {
            return false;
        }
    }

    return true;
}

} // namespace nvmeservice
