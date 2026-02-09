#pragma once

#include <string>

#include "nvmeservice_state.h"

namespace nvmeservice {

class NvmeServiceServer {
public:
    explicit NvmeServiceServer(std::string socket_path);

    bool serve(ServiceState& state);

private:
    bool handleClient(int client_fd, ServiceState& state);
    bool readExact(int fd, void* buf, size_t len);
    bool writeExact(int fd, const void* buf, size_t len);

    std::string socket_path_;
};

} // namespace nvmeservice
