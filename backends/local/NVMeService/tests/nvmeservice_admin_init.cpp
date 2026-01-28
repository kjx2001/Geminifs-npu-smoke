#include "nvmeservice_client.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: nvmeservice_admin_init <socket_path> <sys_config.ini> <gpu_id>\n";
        return 1;
    }

    std::string socket_path = argv[1];
    std::string config_path = argv[2];
    int gpu_id = std::stoi(argv[3]);

    nvmeservice::NvmeServiceClient client(socket_path);
    if (!client.adminInitFromConfig(config_path, gpu_id, true)) {
        std::cerr << "Admin init failed\n";
        return 1;
    }

    std::cout << "Admin init succeeded\n";
    return 0;
}
