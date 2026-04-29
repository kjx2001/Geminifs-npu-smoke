/**
 * nvmeservice_daemon.cpp -- NVMeService daemon entry point.
 *
 * Reads sys_config.yaml, initialises every NVMe controller described there
 * (standalone libnvm path, full admin + queue pool), pre-computes
 * cudaIpcMemHandle_t for every queue's SQ/CQ/PRP memory, starts the gRPC
 * server, and runs until SIGINT/SIGTERM.
 */

#include "nvmeservice_config.h"
#include "nvmeservice_server.h"
#include "nvmeservice_state.h"

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

static std::atomic<grpc::Server*> g_server{nullptr};

static void on_signal(int /*sig*/) {
    auto* s = g_server.load();
    if (s != nullptr) {
        s->Shutdown();
    }
}

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --config <sys_config.yaml>\n"
        "\n"
        "Reads the given config and runs the NVMeService daemon until\n"
        "a signal is received (SIGINT/SIGTERM).\n",
        prog);
}

int main(int argc, char** argv) {
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (config_path.empty()) {
        std::fprintf(stderr, "Missing --config\n");
        print_usage(argv[0]);
        return 1;
    }

    // --- Parse config ---
    std::string parse_err;
    auto cfg_opt = nvmeservice::parse_config_file(config_path, &parse_err);
    if (!cfg_opt.has_value()) {
        std::fprintf(stderr, "Config parse failed: %s\n", parse_err.c_str());
        return 1;
    }
    const auto& cfg = cfg_opt.value();

    for(const auto& n : cfg.nvmes) {
        std::cout << "Parsed NVMe config: pci=" << n.pci_addr
                  << " mount=" << n.mount_path
                  << " ns=" << n.namespace_id
                  << " qdepth=" << n.queue_depth
                  << " total_queues=" << n.total_queues
                  << " queue_groups=" << n.queue_groups.size()
                  << "\n";
    }
    // --- Build service state (opens Controllers, pre-computes IPC handles) ---
    std::shared_ptr<nvmeservice::ServiceState> state;
    try {
        state = std::make_shared<nvmeservice::ServiceState>(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ServiceState init failed: %s\n", e.what());
        return 1;
    }

    // state->start_reaper();

    // // --- gRPC server ---
    // nvmeservice::NvmeServiceImpl svc(state);

    // grpc::ServerBuilder builder;
    // int bound_port = 0;
    // builder.AddListeningPort(cfg.grpc.endpoint,
    //                           grpc::InsecureServerCredentials(),
    //                           &bound_port);
    // builder.RegisterService(&svc);

    // std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    // if (!server) {
    //     std::fprintf(stderr, "Failed to start gRPC server on %s\n",
    //                  cfg.grpc.endpoint.c_str());
    //     state->stop_reaper();
    //     return 1;
    // }
    // g_server.store(server.get());

    // std::signal(SIGINT,  on_signal);
    // std::signal(SIGTERM, on_signal);
    // std::signal(SIGPIPE, SIG_IGN);

    // std::cout << "NVMeService daemon listening on "
    //           << cfg.grpc.endpoint << " (port " << bound_port << ")\n";
    // std::cout << "Registered devices:\n";
    // for (const auto& d : state->list_devices()) {
    //     std::cout << "  device_id=" << d.device_id
    //               << " pci=" << d.pci_addr
    //               << " snvme=" << d.snvme_dev_path
    //               << " gpu=" << d.cuda_device
    //               << " ns=" << d.namespace_id
    //               << " page=" << d.page_size
    //               << " blk=" << d.blk_size
    //               << " qdepth=" << d.queue_depth
    //               << " dstrd=" << d.dstrd
    //               << " bar0=" << d.bar0_size
    //               << " queues=" << d.available_queues
    //               << "/" << d.total_queues
    //               << "\n";
    // }
    // std::cout << "lease: heartbeat=" << cfg.lease.heartbeat_interval_sec
    //           << "s timeout=" << cfg.lease.timeout_sec << "s\n";
    // std::cout << "queue_pool: default=" << cfg.queue_pool.default_per_client
    //           << " max=" << cfg.queue_pool.max_per_client << "\n";
    // std::cout.flush();

    // // Block until signal -> server->Shutdown()
    // server->Wait();

    // std::cout << "Shutting down...\n";
    // state->stop_reaper();
    // g_server.store(nullptr);
    // std::cout << "Daemon exited cleanly.\n";
    return 0;
}
