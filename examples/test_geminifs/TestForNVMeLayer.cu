#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <ctime>
#include <string>
#include <unistd.h>
#include <vector>
#include "buffer.h"
#include "geminifs.h"
#include "linux/ioctl.h"
#include "geminifs.cuh"
#include "nvm_error.h"
#include "utils.cuh"
// 定义常量


int main(int argc, char** argv) {


    try {
        // Create nvme_ctrl_param for a single NVMe controller
        nvme_ctrl_param params = {
            .mount_path = "/mnt/nvme_layer",
            .pci_addr = "0000:50:00.0",  // Single PCI address
            .cudaDevice = 0,
            .ns_id = 1,
            .queueDepth = 1024,
            .numQueues = 64
        };

        // Create NVMeController instance
        auto nvme_controller = std::make_shared<NVMeController>(params);
        
        std::cout << "NVMeController initialized successfully!" << std::endl;
        std::cout << "Mount path: " << nvme_controller->mount_path << std::endl;
        std::cout << "Controller initialized for PCI: " << params.pci_addr << std::endl;

        // Use the controller and file manager
        if (nvme_controller->controller) {
            std::cout << "Controller is ready for use" << std::endl;
        }

        if (nvme_controller->file_manager) {
            std::cout << "File manager is ready for file operations" << std::endl;
            
            // // Example: Create a file record
            // NVMeFileDesc file_desc;
            // if (nvme_controller->file_manager->createFile("test_file.dat", file_desc)) {
            //     std::cout << "Created file record with slot index: " << file_desc.slot_index << std::endl;
            // }
        }

       if (nvme_controller->is_initialized()) {
            std::cout << "✓ NVMeController is properly initialized" << std::endl;
        } else {
            std::cerr << "✗ NVMeController failed to initialize" << std::endl;
            return 1;
        }
        
        // NVMeController will be automatically cleaned up when going out of scope
        void* file_handle = nvme_controller->g_open("test_file", 16384, O_HOST);
        if (file_handle == nullptr) {
            std::cout << "g_open returned nullptr (expected if no actual device)" << std::endl;
        } else {
            std::cout << "✓ g_open succeeded" << std::endl;
        }
        
        // Wait for user to type "end" to terminate the program
        std::cout << "\nProgram is running. Type 'end' to terminate and release resources:" << std::endl;
        std::string user_input;
        while (true) {
            std::cout << "> ";
            std::getline(std::cin, user_input);
            
            if (user_input == "end") {
                std::cout << "Terminating program and releasing resources..." << std::endl;
                break;
            } else {
                std::cout << "Invalid input. Please type 'end' to terminate the program." << std::endl;
            }
        }
        nvme_controller.reset(); // Explicitly reset the controller to release resources
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    
    std::cout << "Program terminated successfully." << std::endl;
    return 0;
}