/**
 * nvmeservice_client.cpp -- NVMeService client smoke test.
 *
 * Connects to the daemon, lists devices, allocates a queue range, holds it
 * while the built-in heartbeat thread keeps the lease alive, then releases.
 *
 * Useful to verify:
 *   - gRPC connectivity
 *   - AllocateQueues end-to-end
 *   - build_shared_controller (BAR0 mmap + IPC import) on the client side
 *   - GPU-view mount_path symlink is reachable from this process
 *   - Heartbeat stream stays stable for the hold duration
 *   - Release on Allocation dtor
 */

#include "nvmeservice_client.h"

// Include libnvm Controller / QueuePair definitions for the post-allocate
// hand-off probe (we walk a handful of imported queues to confirm IPC
// + BAR0 hand-off succeeded). The client library itself only needs the
// forward declaration.
#include "ctrl.h"
#include "queue.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <thread>

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --endpoint <host:port>   gRPC endpoint (default 127.0.0.1:50051)\n"
        "  --device   <id>          device_id to allocate on (default 0)\n"
        "  --cuda     <id>          target cuda_device (default: first queue group)\n"
        "  --count    <n>           number of queues to request (0 = daemon default)\n"
        "  --hold     <sec>         how long to hold the allocation (default 30)\n"
        "  --list-only              list devices and exit (no allocate)\n"
        "  -h, --help               show this message\n",
        prog);
}

int main(int argc, char** argv) {
    std::string endpoint = "127.0.0.1:50051";
    int32_t device_id    = 0;
    int32_t cuda_device  = -1;  // -1 => auto-pick first queue group
    int32_t num_queues   = 0;   // 0 => use daemon default
    int     hold_seconds = 30;
    bool    list_only    = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s requires an argument\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--endpoint")      endpoint   = next("--endpoint");
        else if (a == "--device")   device_id  = std::atoi(next("--device"));
        else if (a == "--cuda")     cuda_device = std::atoi(next("--cuda"));
        else if (a == "--count")    num_queues = std::atoi(next("--count"));
        else if (a == "--hold")     hold_seconds = std::atoi(next("--hold"));
        else if (a == "--list-only") list_only = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "Connecting to " << endpoint << " ...\n";
    nvmeservice::NvmeServiceClient client(endpoint);

    // --- List devices ---
    std::cout << "\n=== Listing devices ===\n";
    auto devs = client.list_devices();
    if (devs.empty()) {
        std::fprintf(stderr, "No devices returned. Is the daemon running?\n");
        return 1;
    }
    for (const auto& d : devs) {
        std::cout << "  device_id=" << d.device_id
                  << " pci=" << d.pci_addr
                  << " snvme=" << d.snvme_dev_path
                  << " ns=" << d.namespace_id
                  << " page=" << d.page_size
                  << " blk=" << d.blk_size
                  << " qdepth=" << d.queue_depth
                  << " avail=" << d.available_queues
                  << "/" << d.total_queues
                  << "\n";
        for (const auto& g : d.queue_groups) {
            std::cout << "      group: cuda_device=" << g.cuda_device
                      << " range=[" << g.queue_start_idx
                      << ", " << (g.queue_start_idx + g.queue_count) << ")"
                      << " avail=" << g.available
                      << "/" << g.queue_count
                      << "\n";
        }
    }

    if (list_only) return 0;

    // --- Allocate ---
    std::cout << "\n=== Allocating "
              << (num_queues == 0 ? "(daemon default)" : std::to_string(num_queues))
              << " queues on device " << device_id;
    if (cuda_device >= 0) {
        std::cout << " (cuda_device=" << cuda_device << ")";
    }
    std::cout << " ===\n";
    auto alloc = (cuda_device >= 0)
        ? client.allocate(device_id, cuda_device, num_queues)
        : client.allocate(device_id, num_queues);
    if (!alloc) {
        std::fprintf(stderr, "allocate() failed\n");
        return 1;
    }

    std::cout << "  allocation_id : " << alloc->allocation_id << "\n";
    std::cout << "  device_id     : " << alloc->device_id << "\n";
    std::cout << "  queue range   : [" << alloc->queue_start_idx
              << ", " << (alloc->queue_start_idx + alloc->queue_count) << ")"
              << " count=" << alloc->queue_count << "\n";
    std::cout << "  controller    : " << alloc->controller.get() << "\n";
    std::cout << "  mount_path    : "
              << (alloc->mount_path.empty() ? "(empty)" : alloc->mount_path)
              << "\n";
    std::cout << "  heartbeat     : " << alloc->heartbeat_interval_sec
              << "s interval\n";
    std::cout << "  lease timeout : " << alloc->lease_timeout_sec << "s\n";
    std::cout << "  client_pid    : " << alloc->client_pid << "\n";

    // --- Hand-off validation: prove the client-side bring-up actually
    //     gives us (1) a usable working directory and (2) live GPU
    //     queue addresses that came from the daemon's IPC handles. ---
    std::cout << "\n=== Hand-off validation ===\n";

    // (1) GPU-view filesystem path. The daemon pre-installed a symlink
    //     under the consuming GPU's mount_path; verify it resolves and
    //     is enumerable from this process. Empty string means symlink
    //     install failed at daemon init -- callers can fall back to
    //     `alloc->controller->dev_mount_path`.
    if (alloc->mount_path.empty()) {
        std::cout << "  mount_path  : EMPTY -- daemon symlink install "
                     "failed; falling back to controller->dev_mount_path='"
                  << alloc->controller->dev_mount_path << "'\n";
    } else {
        std::error_code ec;
        const auto resolved = std::filesystem::read_symlink(alloc->mount_path, ec);
        if (ec) {
            std::cout << "  mount_path  : " << alloc->mount_path
                      << " (read_symlink failed: " << ec.message()
                      << ", trying as plain dir)\n";
        } else {
            std::cout << "  mount_path  : " << alloc->mount_path
                      << " -> " << resolved.string() << "\n";
        }

        size_t entries = 0;
        for (const auto& it : std::filesystem::directory_iterator(
                 alloc->mount_path, std::filesystem::directory_options::skip_permission_denied, ec)) {
            (void)it;
            ++entries;
        }
        if (ec) {
            std::cout << "  ls          : FAILED (" << ec.message() << ")\n";
        } else {
            std::cout << "  ls          : " << entries
                      << " entries reachable from this process\n";
        }
    }

    // (2) Per-queue address sanity. Walk the first few QueuePairs and
    //     print the GPU pointers the client-side build_shared_controller
    //     just imported. If any of these are zero we know the IPC handle
    //     import path went wrong.
    {
        Controller* ctrl = alloc->controller.get();
        const uint16_t n = (ctrl != nullptr) ? ctrl->n_qps : 0;
        const uint16_t probe = std::min<uint16_t>(n, 4);
        std::cout << "  queues      : n_qps=" << n
                  << " (probing first " << probe << ")\n";
        for (uint16_t i = 0; i < probe; ++i) {
            const QueuePair* qp = ctrl->h_qps[i];
            if (qp == nullptr) {
                std::cout << "    qp[" << i << "] : NULL\n";
                continue;
            }
            std::cout << "    qp[" << i << "] qp_id=" << qp->qp_id
                      << " is_shared=" << (qp->is_shared ? "true" : "false")
                      << " sq_gpu=" << qp->shared_sq_ptr
                      << " cq_gpu=" << qp->shared_cq_ptr
                      << " prp_gpu=" << qp->shared_prp_ptr
                      // sq.db / cq.db are `volatile uint32_t*` (BAR0
                      // doorbell GPU VAs). We only want to print the
                      // numeric pointer for the hand-off probe; strip
                      // volatile via const_cast and let the resulting
                      // uint32_t* decay to void* in operator<<.
                      << " sq.db=" << static_cast<void*>(const_cast<uint32_t*>(qp->sq.db))
                      << " cq.db=" << static_cast<void*>(const_cast<uint32_t*>(qp->cq.db))
                      << "\n";
        }
    }
    std::cout.flush();

    // --- Hold the allocation so the heartbeat thread has time to run ---
    if (hold_seconds > 0) {
        std::cout << "\n=== Holding allocation for " << hold_seconds
                  << "s (heartbeat thread running in background) ===\n";
        for (int i = 0; i < hold_seconds; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if ((i + 1) % 5 == 0 || i + 1 == hold_seconds) {
                std::cout << "  " << (i + 1) << "s elapsed\n";
                std::cout.flush();
            }
        }
    }

    // --- Release (automatic via Allocation dtor -> ReleaseQueues RPC) ---
    std::cout << "\n=== Releasing (via Allocation dtor) ===\n";
    alloc.reset();

    std::cout << "\nDone.\n";
    return 0;
}
