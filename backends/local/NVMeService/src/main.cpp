#include "nvmeservice_server.h"
#include "nvmeservice_state.h"

#include <iostream>

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    nvmeservice::ServiceState state;
    nvmeservice::NvmeServiceServer server("/tmp/nvmeservice.sock");

    if (!server.serve(state)) {
        std::cerr << "NVMeService failed to start\n";
        return 1;
    }
    return 0;
}
