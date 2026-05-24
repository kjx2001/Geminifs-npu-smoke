#include "nvmeservice_state.h"

#include "ctrl.h"
#include "queue.h"
#include "ioctl.h"          // struct nvm_ioctl_setup, NVM_QUEUE_SETUP_F_*

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace nvmeservice {

namespace {

constexpr char kSnvmeControlPath[] = "/dev/snvm_control";

/**
 * Parse /proc/<pid>/stat field 22 (starttime, clock ticks since boot).
 *
 * Field layout: "pid (comm) state ppid ..."
 * The comm field may contain spaces and parens, so scan from the last ')'
 * and then walk space-separated fields.
 */
std::optional<uint64_t> read_starttime_impl(uint32_t pid) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/%u/stat", pid);
    std::ifstream in(path);
    if (!in.is_open()) return std::nullopt;

    std::string line;
    if (!std::getline(in, line)) return std::nullopt;

    auto rparen = line.rfind(')');
    if (rparen == std::string::npos) return std::nullopt;

    std::istringstream iss(line.substr(rparen + 1));
    // After ")" we're on field 3 (state). starttime is field 22 -> 20 more fields to skip.
    std::string tok;
    for (int i = 0; i < 19; ++i) {
        if (!(iss >> tok)) return std::nullopt;
    }
    uint64_t starttime;
    if (!(iss >> starttime)) return std::nullopt;
    return starttime;
}

} // namespace

// ---------------------------------------------------------------------------
// Helpers (static)
// ---------------------------------------------------------------------------

std::string ServiceState::generate_allocation_id() {
    // 16 random hex bytes -> 32-char id. Not cryptographically secure; only
    // serves as a unique identifier among cooperating clients.
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t hi = dist(rng);
    uint64_t lo = dist(rng);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hi
        << std::setw(16) << lo;
    return oss.str();
}

std::optional<uint64_t> ServiceState::read_pid_starttime(uint32_t pid) {
    return read_starttime_impl(pid);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

ServiceState::ServiceState(const ServiceConfig& cfg) : cfg_(cfg) {
    devices_.reserve(cfg_.nvmes.size());

    for (size_t i = 0; i < cfg_.nvmes.size(); ++i) {
        const auto& nvme = cfg_.nvmes[i];
        // Cross-references inside queue_groups (gpu_id known, count > 0,
        // sums valid, no all-CPU lists) are already enforced by
        // validate_config(); the YAML parser also did the QP -> SQ+CQ
        // unit translation, so nvme.queue_setup is ready to forward to
        // libnvm verbatim. We pass the gpus vector here so init_device
        // can look up each YAML group's mount_path for symlink install.
        init_device(cfg_.gpus, nvme, static_cast<int32_t>(i));
    }
}

ServiceState::~ServiceState() {
    stop_reaper();
    // Tear down GPU-view symlinks before Controller dtors run umount.
    for (auto& dev : devices_) {
        remove_gpu_symlinks(dev);
    }
    // devices_'s shared_ptr<Controller> will release libnvm on dtor
}

// ---------------------------------------------------------------------------
// Device initialisation
// ---------------------------------------------------------------------------

void ServiceState::init_device(const std::vector<GpuEntry>& gpus,
                                const NvmeEntry& nvme,
                                int32_t device_id) {
    DeviceState dev;
    dev.device_id    = device_id;
    dev.pci_addr     = nvme.pci_addr;
    dev.mount_path   = nvme.mount_path;
    dev.namespace_id = nvme.namespace_id;
    dev.queue_depth  = static_cast<uint32_t>(nvme.queue_depth);

    // The YAML parser already populated nvme.queue_setup in kernel ABI
    // units (groups[].count = 2 * QueuePair count, owner_id = gpu_id,
    // CPU placeholders dropped). validate_config() guarantees
    // nr_groups > 0 and the local invariant
    //   sum(yaml count) + cap_kernel_ioq <= total_queues  (QP units).
    // libnvm fills setup.ioq_num and setup.flags' on_host bit
    // authoritatively from groups[] inside
    // Controller::init_queues_multi_gpu, so we forward this struct
    // as-is. The multi-GPU ctor (chosen by overload resolution on
    // the nvm_ioctl_setup argument) deliberately leaves
    // h_qps[i]->sq.db / cq.db as BAR0 host VAs -- NVMeService clients
    // re-resolve doorbell GPU VAs locally inside
    // build_shared_controller().
    dev.controller = std::make_shared<Controller>(
        kSnvmeControlPath,
        nvme.pci_addr.c_str(),
        nvme.mount_path,
        nvme.namespace_id,
        nvme.queue_depth,
        nvme.queue_setup);

    std::fprintf(stderr,
        "nvmeservice: device=%d pci=%s nvm_ioctl_setup{ioq_num=%u (SQ+CQ) "
        "cap_kernel_ioq=%u (pairs) nr_write=%u nr_poll=%u nr_groups=%u}\n",
        device_id, nvme.pci_addr.c_str(),
        nvme.queue_setup.ioq_num,
        nvme.queue_setup.cap_kernel_ioq,
        nvme.queue_setup.nr_write,
        nvme.queue_setup.nr_poll,
        nvme.queue_setup.nr_groups);
    for (uint32_t i = 0; i < nvme.queue_setup.nr_groups; ++i) {
        std::fprintf(stderr,
            "nvmeservice:   group[%u] owner_id=%u count=%u SQ+CQ (= %u pairs)\n",
            i, nvme.queue_setup.groups[i].owner_id,
            nvme.queue_setup.groups[i].count,
            nvme.queue_setup.groups[i].count / 2u);
    }

    // Populate fields that come from libnvm Controller/ctrl
    dev.page_size      = dev.controller->page_size;
    dev.blk_size       = dev.controller->blk_size;
    dev.blk_size_log   = dev.controller->blk_size_log;
    dev.dstrd          = dev.controller->ctrl->dstrd;
    dev.bar0_size      = dev.controller->ctrl->mm_size;
    dev.snvme_dev_path = dev.controller->dev_path ? dev.controller->dev_path : "";
    dev.total_queues   = static_cast<int32_t>(dev.controller->n_qps);

    // Build dev.groups directly from the kernel-facing groups[]: the
    // i-th group claims the next groups[i].count/2 QueuePairs (kernel
    // unit -> QP), pinned to groups[i].owner_id. CPU placeholders were
    // already filtered by parse_queue_groups, so every entry here is
    // GPU-resident. Defensive cap against total_queues handles the
    // edge case where the controller granted fewer queues than
    // requested (snvme returns the actual count via NVM_GET_DEV_INFO).
    int32_t cursor = 0;
    dev.groups.reserve(nvme.queue_setup.nr_groups);
    for (uint32_t gi = 0; gi < nvme.queue_setup.nr_groups; ++gi) {
        const int32_t qp = static_cast<int32_t>(
            nvme.queue_setup.groups[gi].count / 2u);
        DeviceQueueGroup g;
        g.cuda_device     = static_cast<int32_t>(
            nvme.queue_setup.groups[gi].owner_id);
        g.queue_start_idx = cursor;
        g.count           = std::min(qp, dev.total_queues - cursor);
        if (g.count <= 0) break;
        g.queue_allocated.assign(g.count, false);
        dev.groups.push_back(std::move(g));
        cursor += qp;
    }

    init_queue_handles(dev);
    install_gpu_symlinks(dev, gpus, nvme);

    devices_.push_back(std::move(dev));
}

// ---------------------------------------------------------------------------
// GPU-view symlink lifecycle
// ---------------------------------------------------------------------------

void ServiceState::install_gpu_symlinks(DeviceState& dev,
                                         const std::vector<GpuEntry>& gpus,
                                         const NvmeEntry& nvme) {
    namespace fs = std::filesystem;

    // Symlink name under each consuming GPU's mount_path is the basename
    // of the snvme device node (e.g. "snvm_nvme0n1"), so a GPU client
    // can find "its" NVMe at <gpu.mount_path>/<basename>.
    std::string link_name;
    if (!dev.snvme_dev_path.empty()) {
        link_name = fs::path(dev.snvme_dev_path).filename().string();
    }
    if (link_name.empty()) {
        // Fallback: sanitise the PCI addr.
        link_name = nvme.pci_addr;
        for (auto& c : link_name) {
            if (c == ':' || c == '.' || c == '/') c = '_';
        }
    }

    std::unordered_set<int> gpu_ids_seen;
    for (const auto& qg : nvme.yaml_queue_groups) {
        if (qg.gpu_id < 0) continue;                   // CPU placeholder
        if (!gpu_ids_seen.insert(qg.gpu_id).second) continue;  // already linked

        const GpuEntry* matched = nullptr;
        for (const auto& g : gpus) {
            if (g.id == qg.gpu_id) { matched = &g; break; }
        }
        if (matched == nullptr || matched->mount_path.empty()) continue;

        const fs::path nvme_subdir = fs::path(nvme.mount_path) /
            ("GPU" + std::to_string(qg.gpu_id));
        const fs::path link_path   = fs::path(matched->mount_path) / link_name;

        // 1) Create the per-GPU subdirectory on the NVMe (idempotent).
        std::error_code ec;
        fs::create_directories(nvme_subdir, ec);
        if (ec) {
            std::fprintf(stderr,
                "warning: mkdir %s failed: %s\n",
                nvme_subdir.c_str(), ec.message().c_str());
            // Continue trying other links rather than aborting init.
            continue;
        }
        dev.created_nvme_subdirs.push_back(nvme_subdir.string());

        // 2) Ensure the GPU view directory exists.
        fs::create_directories(matched->mount_path, ec);
        // Ignore ec: operator may have set up the dir already with
        // specific permissions; if it really doesn't exist the symlink
        // create below will surface the error.

        // 3) Symlink. If a symlink already exists pointing at the right
        // target, leave it alone; if it points elsewhere, replace.
        std::error_code link_ec;
        const fs::file_status st = fs::symlink_status(link_path, link_ec);
        if (st.type() == fs::file_type::symlink) {
            const fs::path existing = fs::read_symlink(link_path, link_ec);
            if (!link_ec && existing == nvme_subdir) {
                dev.created_symlinks.push_back(link_path.string());
                continue;  // already correct
            }
            fs::remove(link_path, link_ec);  // drop stale link
        } else if (fs::exists(st)) {
            std::fprintf(stderr,
                "warning: %s exists and is not a symlink; refusing to overwrite\n",
                link_path.c_str());
            continue;
        }

        fs::create_symlink(nvme_subdir, link_path, link_ec);
        if (link_ec) {
            std::fprintf(stderr,
                "warning: symlink %s -> %s failed: %s\n",
                link_path.c_str(), nvme_subdir.c_str(), link_ec.message().c_str());
            continue;
        }
        dev.created_symlinks.push_back(link_path.string());

        // Surface the symlink path to the matching DeviceQueueGroup so
        // allocate() can hand it to clients via AllocResponse.mount_path.
        for (auto& dg : dev.groups) {
            if (dg.cuda_device == qg.gpu_id) {
                dg.gpu_view_path = link_path.string();
                break;
            }
        }
    }
}

void ServiceState::remove_gpu_symlinks(DeviceState& dev) {
    namespace fs = std::filesystem;
    std::error_code ec;

    // Remove symlinks first so the GPU subdirs underneath the NVMe
    // mount become rmdir-able if empty.
    for (const auto& link : dev.created_symlinks) {
        fs::remove(link, ec);  // best effort; ignore errors
    }
    dev.created_symlinks.clear();

    // rmdir the GPU subdirs on the NVMe; ENOTEMPTY means user data
    // lives there and we leave it intact.
    for (const auto& sub : dev.created_nvme_subdirs) {
        ::rmdir(sub.c_str());  // best effort
    }
    dev.created_nvme_subdirs.clear();
}

void ServiceState::init_queue_handles(DeviceState& dev) {
    dev.queue_handles.clear();
    dev.queue_handles.resize(dev.total_queues);

    // Helper: locate the owning group for an absolute queue index.
    // Returns nullptr if the queue isn't covered by any allocatable
    // group (e.g. tail padding when sum(group.count) < total_queues,
    // or CPU-placeholder regions).
    auto find_group_for_queue = [&dev](int32_t qi) -> const DeviceQueueGroup* {
        for (const auto& g : dev.groups) {
            if (qi >= g.queue_start_idx && qi < g.queue_start_idx + g.count) {
                return &g;
            }
        }
        return nullptr;
    };

    for (int32_t i = 0; i < dev.total_queues; ++i) {
        QueuePair* qp = dev.controller->h_qps[i];

        // For multi-GPU pools, set the device matching this queue's
        // memory before producing IPC handles so the CUDA context is
        // consistent with the source allocation.
        if (const DeviceQueueGroup* owner = find_group_for_queue(i)) {
            cudaSetDevice(owner->cuda_device);
        }

        QueueShared qs;
        qs.queue_id   = i;
        qs.sq_entries = qp->sq.qs;
        qs.cq_entries = qp->cq.qs;
        qs.sq_ioaddr  = qp->sq.ioaddr;
        qs.cq_ioaddr  = qp->cq.ioaddr;

        // SQ memory
        if (qp->sq_mem && qp->sq_mem->vaddr) {
            cudaError_t err = cudaIpcGetMemHandle(&qs.ipc_sq,
                const_cast<void*>(qp->sq_mem->vaddr));
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "cudaIpcGetMemHandle(sq) failed on device " +
                    std::to_string(dev.device_id) + " queue " + std::to_string(i) +
                    ": " + cudaGetErrorString(err));
            }
        }

        // CQ memory
        if (qp->cq_mem && qp->cq_mem->vaddr) {
            cudaError_t err = cudaIpcGetMemHandle(&qs.ipc_cq,
                const_cast<void*>(qp->cq_mem->vaddr));
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "cudaIpcGetMemHandle(cq) failed on device " +
                    std::to_string(dev.device_id) + " queue " + std::to_string(i) +
                    ": " + cudaGetErrorString(err));
            }
        }

        // PRP memory (optional — if daemon pre-allocates PRP pool)
        if (qp->prp_mem && qp->prp_mem->vaddr) {
            cudaError_t err = cudaIpcGetMemHandle(&qs.ipc_prp,
                const_cast<void*>(qp->prp_mem->vaddr));
            if (err == cudaSuccess) {
                qs.has_prp = true;
            }
            // If PRP IPC fails, fall back to client-allocated PRP
        }

        dev.queue_handles[i] = qs;
    }
}

// ---------------------------------------------------------------------------
// Queue range reservation (group-local)
// ---------------------------------------------------------------------------

bool ServiceState::reserve_range(DeviceQueueGroup& group, int32_t count,
                                  int32_t* out_start, int32_t* out_count) {
    if (count <= 0) return false;

    int32_t run_start = -1;
    int32_t run_len   = 0;

    for (int32_t i = 0; i < group.count; ++i) {
        if (!group.queue_allocated[i]) {
            if (run_start == -1) run_start = i;
            ++run_len;
            if (run_len >= count) {
                // Reserve [run_start, run_start+count) inside this group.
                for (int32_t j = run_start; j < run_start + count; ++j) {
                    group.queue_allocated[j] = true;
                }
                // Hand back the *absolute* queue index so callers index
                // into dev.queue_handles[] without thinking about groups.
                *out_start = group.queue_start_idx + run_start;
                *out_count = count;
                return true;
            }
        } else {
            run_start = -1;
            run_len   = 0;
        }
    }
    return false;
}

void ServiceState::release_range(DeviceState& dev, int32_t start, int32_t count) {
    if (count <= 0) return;
    for (auto& g : dev.groups) {
        const int32_t g_end = g.queue_start_idx + g.count;
        if (start < g.queue_start_idx || start >= g_end) continue;

        const int32_t local_start = start - g.queue_start_idx;
        const int32_t local_end   = std::min(local_start + count, g.count);
        for (int32_t i = std::max<int32_t>(0, local_start); i < local_end; ++i) {
            g.queue_allocated[i] = false;
        }
        return;
    }
    // start fell outside every group -- treat as a no-op rather than
    // throw, since reaper / release pathways already guard against
    // unknown allocation_ids upstream.
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

std::vector<DeviceSnapshot> ServiceState::list_devices() const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    std::vector<DeviceSnapshot> out;
    out.reserve(devices_.size());
    for (const auto& d : devices_) {
        DeviceSnapshot s;
        s.device_id        = d.device_id;
        s.pci_addr         = d.pci_addr;
        s.snvme_dev_path   = d.snvme_dev_path;
        // Legacy single cuda_device field: surface the first group's
        // GPU id so a single-GPU client that ignores `groups` still
        // gets a sensible value.
        s.cuda_device      = d.groups.empty() ? -1 : d.groups.front().cuda_device;
        s.namespace_id     = d.namespace_id;
        s.page_size        = d.page_size;
        s.blk_size         = d.blk_size;
        s.blk_size_log     = d.blk_size_log;
        s.queue_depth      = d.queue_depth;
        s.dstrd            = d.dstrd;
        s.bar0_size        = d.bar0_size;
        s.total_queues     = d.total_queues;

        s.groups.reserve(d.groups.size());
        int32_t total_avail = 0;
        for (const auto& g : d.groups) {
            QueueGroupSnapshot gs;
            gs.cuda_device     = g.cuda_device;
            gs.queue_start_idx = g.queue_start_idx;
            gs.queue_count     = g.count;
            gs.available       = static_cast<int32_t>(
                std::count(g.queue_allocated.begin(),
                           g.queue_allocated.end(), false));
            total_avail += gs.available;
            s.groups.push_back(gs);
        }
        s.available_queues = total_avail;
        out.push_back(std::move(s));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Allocate / Release / Heartbeat
// ---------------------------------------------------------------------------

ServiceState::AllocResult ServiceState::allocate(int32_t device_id,
                                                  int32_t cuda_device,
                                                  int32_t num_queues,
                                                  uint32_t client_pid) {
    AllocResult result;

    std::lock_guard<std::mutex> lock(state_mtx_);

    if (device_id < 0 || device_id >= static_cast<int32_t>(devices_.size())) {
        result.error = "invalid device_id";
        return result;
    }

    DeviceState& dev = devices_[device_id];

    // Find the queue group that lives on the requested cuda_device.
    // With multi-GPU pools there can be more than one group per device,
    // each on a different GPU.
    DeviceQueueGroup* matching_group = nullptr;
    for (auto& g : dev.groups) {
        if (g.cuda_device == cuda_device) {
            matching_group = &g;
            break;
        }
    }
    if (matching_group == nullptr) {
        result.error = "no queue group on device_id=" + std::to_string(device_id) +
                       " for cuda_device=" + std::to_string(cuda_device);
        return result;
    }

    int32_t want = (num_queues > 0) ? num_queues : cfg_.queue_pool.default_per_client;
    want = std::min(want, cfg_.queue_pool.max_per_client);
    if (want <= 0) {
        result.error = "num_queues clamp -> 0";
        return result;
    }

    int32_t start = 0, count = 0;
    if (!reserve_range(*matching_group, want, &start, &count)) {
        result.error = "no contiguous range of size " + std::to_string(want) +
                       " available on cuda_device=" + std::to_string(cuda_device);
        return result;
    }

    // Record allocation. queue_start_idx is *absolute* (already offset
    // by the owning group's start), so release_range can find the group
    // again from the start index alone.
    Allocation alloc;
    alloc.allocation_id        = generate_allocation_id();
    alloc.device_id            = device_id;
    alloc.cuda_device          = cuda_device;
    alloc.queue_start_idx      = start;
    alloc.queue_count          = count;
    alloc.client_pid           = client_pid;
    alloc.client_pid_starttime = read_pid_starttime(client_pid).value_or(0);
    alloc.last_heartbeat       = std::chrono::steady_clock::now();

    const std::string aid = alloc.allocation_id;
    allocations_.emplace(aid, std::move(alloc));

    // Fill grant
    AllocationGrant& g = result.grant;
    g.allocation_id          = aid;
    g.device_id              = device_id;
    g.pci_addr               = dev.pci_addr;
    g.snvme_dev_path         = dev.snvme_dev_path;
    // GPU-view symlink path the daemon installed for this group's GPU.
    // Empty if symlink install failed -- client falls back to no
    // mount_path on the local Controller.
    g.mount_path             = matching_group->gpu_view_path;
    g.bar0_size              = dev.bar0_size;
    g.dstrd                  = dev.dstrd;
    g.queue_start_idx        = start;
    g.queue_count            = count;
    g.namespace_id           = dev.namespace_id;
    g.page_size              = dev.page_size;
    g.blk_size               = dev.blk_size;
    g.blk_size_log           = dev.blk_size_log;
    g.queue_depth            = dev.queue_depth;
    g.heartbeat_interval_sec = cfg_.lease.heartbeat_interval_sec;
    g.lease_timeout_sec      = cfg_.lease.timeout_sec;

    g.queue_shared.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
        g.queue_shared.push_back(dev.queue_handles[start + i]);
    }

    result.success = true;
    return result;
}

bool ServiceState::release(const std::string& allocation_id,
                            uint32_t client_pid,
                            std::string* error) {
    std::lock_guard<std::mutex> lock(state_mtx_);

    auto it = allocations_.find(allocation_id);
    if (it == allocations_.end()) {
        if (error) *error = "unknown allocation_id";
        return false;
    }

    const Allocation& alloc = it->second;
    if (alloc.client_pid != client_pid) {
        if (error) *error = "pid mismatch (recorded=" +
            std::to_string(alloc.client_pid) + " got=" + std::to_string(client_pid) + ")";
        return false;
    }

    DeviceState& dev = devices_[alloc.device_id];
    release_range(dev, alloc.queue_start_idx, alloc.queue_count);
    allocations_.erase(it);
    return true;
}

bool ServiceState::update_heartbeat(const std::string& allocation_id, std::string* error) {
    std::lock_guard<std::mutex> lock(state_mtx_);

    auto it = allocations_.find(allocation_id);
    if (it == allocations_.end()) {
        if (error) *error = "unknown allocation_id";
        return false;
    }
    it->second.last_heartbeat = std::chrono::steady_clock::now();
    return true;
}

bool ServiceState::has_allocation(const std::string& allocation_id) const {
    std::lock_guard<std::mutex> lock(state_mtx_);
    return allocations_.find(allocation_id) != allocations_.end();
}

// ---------------------------------------------------------------------------
// Reaper
// ---------------------------------------------------------------------------

void ServiceState::start_reaper() {
    if (reaper_running_.exchange(true)) return;
    reaper_thread_ = std::thread(&ServiceState::reaper_loop, this);
}

void ServiceState::stop_reaper() {
    if (!reaper_running_.exchange(false)) return;
    if (reaper_thread_.joinable()) {
        reaper_thread_.join();
    }
}

bool ServiceState::is_pid_dead(uint32_t pid, uint64_t recorded_starttime) const {
    if (::kill(static_cast<pid_t>(pid), 0) != 0) {
        // ESRCH or similar -> definitely dead
        return true;
    }
    // PID is alive, but might be a reused slot. Compare starttime.
    auto st = read_pid_starttime(pid);
    if (!st.has_value()) {
        // Can't read /proc -> assume dead to err on the side of reclaim
        return true;
    }
    return st.value() != recorded_starttime;
}

void ServiceState::reaper_loop() {
    const auto tick = std::chrono::seconds(std::max<uint32_t>(
        1, cfg_.lease.heartbeat_interval_sec / 2));
    const auto timeout = std::chrono::seconds(cfg_.lease.timeout_sec);

    while (reaper_running_.load()) {
        std::this_thread::sleep_for(tick);
        if (!reaper_running_.load()) break;

        const auto now = std::chrono::steady_clock::now();
        std::vector<std::string> to_reap;

        {
            std::lock_guard<std::mutex> lock(state_mtx_);
            for (const auto& kv : allocations_) {
                const Allocation& alloc = kv.second;
                if (now - alloc.last_heartbeat <= timeout) continue;

                if (is_pid_dead(alloc.client_pid, alloc.client_pid_starttime)) {
                    to_reap.push_back(alloc.allocation_id);
                }
                // If PID is alive but silent, we deliberately leave the
                // allocation in place and log (TODO: log) -- safer than
                // racing the client on queue ownership.
            }

            for (const auto& aid : to_reap) {
                auto it = allocations_.find(aid);
                if (it == allocations_.end()) continue;
                const Allocation& alloc = it->second;
                DeviceState& dev = devices_[alloc.device_id];
                release_range(dev, alloc.queue_start_idx, alloc.queue_count);
                allocations_.erase(it);
            }
        }
    }
}

} // namespace nvmeservice
