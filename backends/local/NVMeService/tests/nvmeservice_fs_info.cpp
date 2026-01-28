#include "nvmeservice_client.h"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: nvmeservice_fs_info <socket_path>\n";
        return 1;
    }

    nvmeservice::NvmeServiceClient client(argv[1]);
    nvmeservice::FsGetInfoResp header{};
    std::vector<nvmeservice::CtrlConfig> ctrls;

    if (!client.getInfo(header, ctrls)) {
        std::cerr << "Get info failed\n";
        return 1;
    }

    std::cout << "Mount base: " << header.mount_base_path << "\n";
    std::cout << "Controllers: " << header.ctrl_count << "\n";
    std::cout << "Max queues/process: " << header.max_queues_per_process << "\n";

    for (size_t i = 0; i < ctrls.size(); ++i) {
        std::cout << "[" << i << "] pci=" << ctrls[i].pci_addr
                  << " mount=" << ctrls[i].mount_path
                  << " ns=" << ctrls[i].ns_id
                  << " qd=" << ctrls[i].queue_depth
                  << " nq=" << ctrls[i].num_queues
                  << " gpu=" << ctrls[i].cuda_device
                  << " maxIOKB=" << ctrls[i].max_io_kb
                  << "\n";
    }

    return 0;
}
