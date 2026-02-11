#include "nvmeservice_config.h"
#include "nvmeservice_server.h"
#include "nvmeservice_state.h"

#include <iostream>
#include <string>

namespace {

struct Options {
    std::string config_path = "sys_config.yaml";
};

bool parseArgs(int argc, char** argv, Options& out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            out.config_path = argv[++i];
        } else if (arg == "--help") {
            return false;
        } else {
            return false;
        }
    }
    return true;
}

void printUsage(const char* exe) {
    std::cerr << "Usage: " << exe << " [--config PATH]\n";
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseArgs(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    nvmeservice::ParsedConfigAll parsed;
    std::string error_message;
    if (!nvmeservice::parseSysConfig(options.config_path, parsed, &error_message)) {
        std::cerr << "Failed to parse config: " << error_message << "\n";
        return 1;
    }

    std::cerr << "Parsed config: grpc_endpoint=" << parsed.grpc_endpoint
              << ", max_queues_per_process=" << parsed.max_queues_per_process
              << ", ctrl_count=" << parsed.ctrls.size()
              << ", mount_base_path=" << parsed.mount_base_path
              << "\n";

    nvmeservice::ServiceState state;
    state.max_queues_per_process = parsed.max_queues_per_process;
    if (!state.applyAdminInitAll(parsed.ctrls, parsed.mount_base_path)) {
        std::cerr << "Failed to initialize controllers from config\n";
        return 1;
    }

    nvmeservice::NvmeServiceServer server(parsed.grpc_endpoint);
    if (!server.serve(state)) {
        std::cerr << "NVMeService daemon failed to start on endpoint: " << parsed.grpc_endpoint << "\n";
        return 1;
    }

    return 0;
}
