#include "nvmeservice_client.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

struct Options {
    std::string endpoint = "127.0.0.1:50051";
    std::string command = "ping";
    uint32_t controller_index = 0;
    uint32_t queue_count = 0;
    int32_t pid = 0;
    uint64_t client_id = 0;
    uint64_t lease_id = 0;
    uint32_t duration_ms = 5000;
    uint32_t interval_ms = 1000;
};

bool parseArgs(int argc, char** argv, Options& out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--endpoint" && i + 1 < argc) {
            out.endpoint = argv[++i];
        } else if (arg == "--cmd" && i + 1 < argc) {
            out.command = argv[++i];
        } else if (arg == "--controller" && i + 1 < argc) {
            out.controller_index = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--count" && i + 1 < argc) {
            out.queue_count = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--pid" && i + 1 < argc) {
            out.pid = static_cast<int32_t>(std::stoi(argv[++i]));
        } else if (arg == "--client" && i + 1 < argc) {
            out.client_id = static_cast<uint64_t>(std::stoull(argv[++i]));
        } else if (arg == "--lease" && i + 1 < argc) {
            out.lease_id = static_cast<uint64_t>(std::stoull(argv[++i]));
        } else if (arg == "--duration" && i + 1 < argc) {
            out.duration_ms = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--interval" && i + 1 < argc) {
            out.interval_ms = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--help") {
            return false;
        } else {
            return false;
        }
    }
    return true;
}

void printUsage(const char* exe) {
    std::cerr << "Usage: " << exe << " [--endpoint HOST:PORT] [--cmd ping|info|alloc|release|heartbeat|shutdown]"
              << " [--controller N] [--count N] [--pid PID] [--client ID] [--lease ID]"
              << " [--duration MS] [--interval MS]\n";
}

void printInfo(const std::string& mount_base_path,
               const std::vector<nvmeservice::CtrlConfig>& ctrls,
               uint32_t max_queues_per_process) {
    std::cout << "mount_base_path=" << mount_base_path
              << ", max_queues_per_process=" << max_queues_per_process
              << ", ctrl_count=" << ctrls.size() << "\n";

    for (size_t i = 0; i < ctrls.size(); ++i) {
        const auto& cfg = ctrls[i];
        std::cout << "  [" << i << "] mount_path=" << cfg.mount_path
                  << ", pci_addr=" << cfg.pci_addr
                  << ", ns_id=" << cfg.ns_id
                  << ", queue_depth=" << cfg.queue_depth
                  << ", num_queues=" << cfg.num_queues
                  << ", cuda_device=" << cfg.cuda_device
                  << ", max_io_kb=" << cfg.max_io_kb
                  << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseArgs(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    nvmeservice::NvmeServiceClient client(options.endpoint);

    if (options.command == "ping") {
        if (!client.ping()) {
            std::cerr << "Ping failed\n";
            return 1;
        }
        std::cout << "Ping ok\n";
        return 0;
    }

    if (options.command == "info") {
        std::string mount_base_path;
        std::vector<nvmeservice::CtrlConfig> ctrls;
        uint32_t max_queues_per_process = 0;
        if (!client.getInfo(mount_base_path, ctrls, max_queues_per_process)) {
            std::cerr << "GetInfo failed\n";
            return 1;
        }
        printInfo(mount_base_path, ctrls, max_queues_per_process);
        return 0;
    }

    if (options.command == "alloc") {
        std::vector<uint32_t> qids;
        std::string d_qps_handle;
        std::string d_ctrl_handle;
        uint64_t lease_id = 0;
        uint32_t ttl_ms = 0;
        uint64_t client_id = options.client_id == 0
            ? (static_cast<uint64_t>(options.pid) << 32) | static_cast<uint32_t>(::getpid())
            : options.client_id;
        if (!client.allocQueues(options.controller_index,
                                options.queue_count,
                                options.pid,
                                client_id,
                                qids,
                                d_qps_handle,
                                d_ctrl_handle,
                                lease_id,
                                ttl_ms)) {
            std::cerr << "AllocQueues failed\n";
            return 1;
        }
        std::cout << "AllocQueues ok: controller=" << options.controller_index
                  << ", granted=" << qids.size() << "\n";
        std::cout << "client_id=" << client_id << ", lease_id=" << lease_id
                  << ", ttl_ms=" << ttl_ms << "\n";
        std::cout << "Hint: use --cmd heartbeat --client " << client_id
              << " --lease " << lease_id << " to renew, or --cmd release --lease "
              << lease_id << " to release.\n";
        std::cout << "qids:";
        for (uint32_t qid : qids) {
            std::cout << " " << qid;
        }
        std::cout << "\n";
        std::cout << "d_qps_handle bytes=" << d_qps_handle.size() << "\n";
        std::cout << "d_ctrl_handle bytes=" << d_ctrl_handle.size() << "\n";

        if (d_qps_handle.size() == sizeof(cudaIpcMemHandle_t)) {
            cudaIpcMemHandle_t handle{};
            std::memcpy(&handle, d_qps_handle.data(), sizeof(handle));
            void* d_qps_ptr = nullptr;
            cudaError_t status = cudaIpcOpenMemHandle(&d_qps_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
            if (status == cudaSuccess) {
                std::cout << "Opened d_qps handle: ptr=" << d_qps_ptr << "\n";
                cudaIpcCloseMemHandle(d_qps_ptr);
            } else {
                std::cerr << "cudaIpcOpenMemHandle(d_qps) failed: " << cudaGetErrorString(status) << "\n";
            }
        }

        if (d_ctrl_handle.size() == sizeof(cudaIpcMemHandle_t)) {
            cudaIpcMemHandle_t handle{};
            std::memcpy(&handle, d_ctrl_handle.data(), sizeof(handle));
            void* d_ctrl_ptr = nullptr;
            cudaError_t status = cudaIpcOpenMemHandle(&d_ctrl_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
            if (status == cudaSuccess) {
                std::cout << "Opened d_ctrl handle: ptr=" << d_ctrl_ptr << "\n";
                cudaIpcCloseMemHandle(d_ctrl_ptr);
            } else {
                std::cerr << "cudaIpcOpenMemHandle(d_ctrl) failed: " << cudaGetErrorString(status) << "\n";
            }
        }
        return 0;
    }

    if (options.command == "release") {
        if (options.lease_id != 0) {
            if (!client.releaseLease(options.lease_id)) {
                std::cerr << "ReleaseLease failed\n";
                return 1;
            }
            std::cout << "ReleaseLease ok\n";
            return 0;
        }

        std::vector<uint32_t> qids;
        if (options.queue_count > 0) {
            for (uint32_t i = 0; i < options.queue_count; ++i) {
                qids.push_back(i);
            }
        }
        if (!client.releaseQueues(options.controller_index, options.pid, qids)) {
            std::cerr << "ReleaseQueues failed\n";
            return 1;
        }
        std::cout << "ReleaseQueues ok\n";
        return 0;
    }

    if (options.command == "heartbeat") {
        uint64_t client_id = options.client_id == 0
            ? (static_cast<uint64_t>(options.pid) << 32) | static_cast<uint32_t>(::getpid())
            : options.client_id;
        std::vector<uint64_t> lease_ids;
        if (options.lease_id != 0) {
            lease_ids.push_back(options.lease_id);
        }
        if (!client.heartbeatLeases(client_id, lease_ids, options.duration_ms, options.interval_ms)) {
            std::cerr << "Heartbeat failed\n";
            return 1;
        }
        std::cout << "Heartbeat ok\n";
        return 0;
    }

    if (options.command == "shutdown") {
        if (!client.shutdown()) {
            std::cerr << "Shutdown failed\n";
            return 1;
        }
        std::cout << "Shutdown ok\n";
        return 0;
    }

    std::cerr << "Unknown command: " << options.command << "\n";
    printUsage(argv[0]);
    return 1;
}
