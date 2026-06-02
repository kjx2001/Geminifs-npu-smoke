/**
 * host_fs_backed_nvme_storage.cpp -- mount-the-snvme-block-device
 * implementation of INvmeStorage.
 */

#include "host_fs_backed_nvme_storage.h"
#include "fiemap_helper.h"
#include "nvme_file_header.h"
#include "persistent_file_log.h"

#include "../../device_manager/include/local_nvme_device.h"
#include "../../runtime/include/device.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <linux/magic.h>
#include <sys/ioctl.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

namespace tutti {

namespace {

constexpr uint32_t kNvmeBlockSize = 4096;   // matches all NVMe deployments
                                            // we currently support; the
                                            // bootstrap path validates against
                                            // LocalNvmeDevice::blk_size.

bool path_exists(const std::string& p) {
    struct stat st{};
    return ::stat(p.c_str(), &st) == 0;
}

bool is_mounted(const std::string& mount_point) {
    // "Is something mounted at mount_point?" -- compare statfs of
    // mount_point and its parent.  If they differ we know there's
    // a mount boundary here.
    struct stat st_self{}, st_parent{};
    if (::stat(mount_point.c_str(), &st_self) != 0) return false;
    std::string parent = mount_point + "/..";
    if (::stat(parent.c_str(),       &st_parent) != 0) return false;
    return st_self.st_dev != st_parent.st_dev;
}

// Run a system command, log the command line, return true on rc=0.
bool run_cmd(const std::string& cmd) {
    std::fprintf(stderr, "[nvme_storage] $ %s\n", cmd.c_str());
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr,
            "[nvme_storage]   command failed (rc=%d)\n", rc);
        return false;
    }
    return true;
}

// "/dev/ssnvme7" -> 7
int minor_from_chrdev_impl(const std::string& chrdev) {
    static const std::string prefix = "/dev/ssnvme";
    if (chrdev.compare(0, prefix.size(), prefix) != 0) return -1;
    const std::string tail = chrdev.substr(prefix.size());
    if (tail.empty()) return -1;
    char* endp = nullptr;
    long v = std::strtol(tail.c_str(), &endp, 10);
    if (endp == tail.c_str() || v < 0) return -1;
    return (int)v;
}

} // namespace

// ---------------------------------------------------------------------------

std::string HostFsBackedNvmeStorage::blk_path_from_chrdev(
    const std::string& chrdev)
{
    int m = minor_from_chrdev_impl(chrdev);
    if (m < 0) return {};
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/dev/snvme%dn1", m);
    return std::string(buf);
}

int HostFsBackedNvmeStorage::minor_from_chrdev(const std::string& chrdev) {
    return minor_from_chrdev_impl(chrdev);
}

// ---------------------------------------------------------------------------

HostFsBackedNvmeStorage::HostFsBackedNvmeStorage(Config cfg)
    : cfg_(std::move(cfg))
{}

HostFsBackedNvmeStorage::HostFsBackedNvmeStorage()
    : cfg_(Config())
{}

HostFsBackedNvmeStorage::~HostFsBackedNvmeStorage() {
    // Best-effort shutdown so destructor doesn't leave mounts behind.
    if (booted_) {
        (void)shutdown();
    }
}

// ---------------------------------------------------------------------------
// State lookup
// ---------------------------------------------------------------------------

HostFsBackedNvmeStorage::PerDeviceState*
HostFsBackedNvmeStorage::find_state(const Device* dev) {
    for (auto& sp : states_) {
        if (sp->device == dev) return sp.get();
    }
    return nullptr;
}
const HostFsBackedNvmeStorage::PerDeviceState*
HostFsBackedNvmeStorage::find_state(const Device* dev) const {
    for (const auto& sp : states_) {
        if (sp->device == dev) return sp.get();
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

bool HostFsBackedNvmeStorage::mkfs_if_needed_locked(const std::string& blk_path) {
    // Use blkid to see if the block device already has a recognised fs.
    // We don't link against libblkid; just shell out and check rc.
    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "blkid -p -s TYPE -o value %s 2>/dev/null",
        blk_path.c_str());
    FILE* fp = ::popen(cmd, "r");
    if (fp == nullptr) {
        std::fprintf(stderr,
            "[nvme_storage] popen(blkid) failed: errno %d\n", errno);
        return false;
    }
    char line[64] = {0};
    if (::fgets(line, sizeof(line), fp) != nullptr) {
        // Got something -- already formatted.  Trim newline.
        for (char* c = line; *c; ++c) {
            if (*c == '\n' || *c == '\r') { *c = 0; break; }
        }
        ::pclose(fp);
        std::fprintf(stderr,
            "[nvme_storage] %s already formatted (TYPE=%s); reusing\n",
            blk_path.c_str(), line);
        return true;
    }
    ::pclose(fp);

    if (!cfg_.auto_mkfs) {
        std::fprintf(stderr,
            "[nvme_storage] %s has no fs and auto_mkfs=false; aborting\n",
            blk_path.c_str());
        return false;
    }

    // mkfs.ext4 -F (force, accept "device is in use" warnings)
    std::string mkfs_cmd = "mkfs.ext4 -F -q " + blk_path;
    return run_cmd(mkfs_cmd);
}

bool HostFsBackedNvmeStorage::mount_if_needed_locked(PerDeviceState& s) {
    // mkdir mount_path if it doesn't exist
    if (!path_exists(s.mount_path)) {
        std::error_code ec;
        std::filesystem::create_directories(s.mount_path, ec);
        if (ec) {
            std::fprintf(stderr,
                "[nvme_storage] mkdir(%s) failed: %s\n",
                s.mount_path.c_str(), ec.message().c_str());
            return false;
        }
    }

    if (is_mounted(s.mount_path)) {
        std::fprintf(stderr,
            "[nvme_storage] %s is already a mount point; reusing\n",
            s.mount_path.c_str());
        s.we_mounted = false;
        return true;
    }

    // mount(2) with type ext4, no special flags.
    if (::mount(s.snvme_blk_path.c_str(), s.mount_path.c_str(),
                "ext4", 0, nullptr) != 0) {
        std::fprintf(stderr,
            "[nvme_storage] mount(%s -> %s, ext4) failed: errno %d (%s)\n",
            s.snvme_blk_path.c_str(), s.mount_path.c_str(),
            errno, std::strerror(errno));
        return false;
    }
    s.we_mounted = true;
    std::fprintf(stderr,
        "[nvme_storage] mounted %s -> %s\n",
        s.snvme_blk_path.c_str(), s.mount_path.c_str());
    return true;
}

bool HostFsBackedNvmeStorage::umount_locked(PerDeviceState& s) {
    if (!s.we_mounted) return true;
    if (::umount(s.mount_path.c_str()) != 0) {
        std::fprintf(stderr,
            "[nvme_storage] umount(%s) failed: errno %d (%s)\n",
            s.mount_path.c_str(), errno, std::strerror(errno));
        return false;
    }
    s.we_mounted = false;
    std::fprintf(stderr,
        "[nvme_storage] unmounted %s\n", s.mount_path.c_str());
    return true;
}

bool HostFsBackedNvmeStorage::bootstrap(
    const std::vector<const Device*>& devices)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (booted_) {
        std::fprintf(stderr, "[nvme_storage] bootstrap: already booted\n");
        return true;
    }
    if (devices.empty()) {
        std::fprintf(stderr, "[nvme_storage] bootstrap: empty device list\n");
        return false;
    }

    // mkdir mount_root once
    {
        std::error_code ec;
        std::filesystem::create_directories(cfg_.mount_root, ec);
        if (ec) {
            std::fprintf(stderr,
                "[nvme_storage] mkdir(%s): %s\n",
                cfg_.mount_root.c_str(), ec.message().c_str());
            return false;
        }
    }

    states_.reserve(devices.size());
    for (const Device* dev : devices) {
        if (dev == nullptr || dev->backend_private == nullptr) {
            std::fprintf(stderr, "[nvme_storage] null Device\n");
            goto rollback;
        }
        auto* lnd = static_cast<LocalNvmeDevice*>(dev->backend_private);
        if (lnd->blk_size != kNvmeBlockSize) {
            std::fprintf(stderr,
                "[nvme_storage] device %d blk_size=%u != expected %u\n",
                dev->device_id, lnd->blk_size, kNvmeBlockSize);
            goto rollback;
        }

        auto sp = std::make_unique<PerDeviceState>();
        sp->device         = dev;
        sp->snvme_blk_path = blk_path_from_chrdev(lnd->snvme_dev_path);
        if (sp->snvme_blk_path.empty()) {
            std::fprintf(stderr,
                "[nvme_storage] cannot derive blk path from %s\n",
                lnd->snvme_dev_path.c_str());
            goto rollback;
        }
        // Mount point: <mount_root>/snvme<minor>
        int m = minor_from_chrdev(lnd->snvme_dev_path);
        char tail[64];
        std::snprintf(tail, sizeof(tail), "/snvme%d", m);
        sp->mount_path = cfg_.mount_root + tail;

        // 1. mkfs (if needed)
        if (!mkfs_if_needed_locked(sp->snvme_blk_path)) {
            goto rollback;
        }
        // 2. mount
        if (!mount_if_needed_locked(*sp)) {
            goto rollback;
        }
        // 3. mkdir <mount>/.tutti
        std::string tutti_dir = sp->mount_path + "/.tutti";
        {
            std::error_code ec;
            std::filesystem::create_directories(tutti_dir, ec);
            if (ec) {
                std::fprintf(stderr,
                    "[nvme_storage] mkdir(%s): %s\n",
                    tutti_dir.c_str(), ec.message().c_str());
                (void)umount_locked(*sp);
                goto rollback;
            }
        }
        // 4. load file_log.bin
        sp->log = std::make_unique<PersistentFileLog>();
        std::string log_path = tutti_dir + "/file_log.bin";
        if (!sp->log->load_or_init(log_path)) {
            std::fprintf(stderr,
                "[nvme_storage] failed to load %s\n", log_path.c_str());
            (void)umount_locked(*sp);
            goto rollback;
        }

        std::fprintf(stderr,
            "[nvme_storage] device %d ready: blk=%s mount=%s entries=%zu\n",
            dev->device_id, sp->snvme_blk_path.c_str(),
            sp->mount_path.c_str(), sp->log->size());

        states_.push_back(std::move(sp));
    }

    booted_ = true;
    return true;

rollback:
    for (auto it = states_.rbegin(); it != states_.rend(); ++it) {
        (void)umount_locked(**it);
    }
    states_.clear();
    return false;
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

bool HostFsBackedNvmeStorage::shutdown() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!booted_) return true;
    bool all_ok = true;

    // Reverse-order teardown.
    for (auto it = states_.rbegin(); it != states_.rend(); ++it) {
        PerDeviceState& s = **it;

        // 1. Close all open NvmeFiles.
        for (auto& [fid, fptr] : s.files) {
            if (fptr->host_fd >= 0) {
                ::fsync(fptr->host_fd);
                ::close(fptr->host_fd);
                fptr->host_fd = -1;
            }
        }

        // 2. Persist log one last time.
        if (s.log && !s.log->persist()) {
            all_ok = false;
        }

        // 3. umount.
        if (!umount_locked(s)) {
            all_ok = false;
        }
    }

    states_.clear();
    booted_ = false;
    return all_ok;
}

// ---------------------------------------------------------------------------
// Capacity
// ---------------------------------------------------------------------------

uint64_t HostFsBackedNvmeStorage::total_capacity(const Device* dev) const {
    std::lock_guard<std::mutex> lock(mtx_);
    const auto* s = find_state(dev);
    if (s == nullptr) return 0;
    struct statvfs sv{};
    if (::statvfs(s->mount_path.c_str(), &sv) != 0) return 0;
    return (uint64_t)sv.f_blocks * (uint64_t)sv.f_frsize;
}

uint64_t HostFsBackedNvmeStorage::available_capacity(const Device* dev) const {
    std::lock_guard<std::mutex> lock(mtx_);
    const auto* s = find_state(dev);
    if (s == nullptr) return 0;
    struct statvfs sv{};
    if (::statvfs(s->mount_path.c_str(), &sv) != 0) return 0;
    return (uint64_t)sv.f_bavail * (uint64_t)sv.f_frsize;
}

// ---------------------------------------------------------------------------
// Directory operations
// ---------------------------------------------------------------------------

bool HostFsBackedNvmeStorage::create_file_locked(
    PerDeviceState& s,
    std::string_view name,
    uint64_t       size_bytes,
    NvmeFile**     out)
{
    if (out == nullptr) return false;
    *out = nullptr;
    if (name.empty() || size_bytes == 0) return false;

    if (s.log->find_by_name(std::string(name)) != nullptr) {
        std::fprintf(stderr,
            "[nvme_storage] create_file: '%.*s' already exists\n",
            (int)name.size(), name.data());
        return false;
    }

    const uint64_t header_bytes = sizeof(NvmeFileHeader);
    const uint64_t total_bytes  = header_bytes + size_bytes;

    std::string host_path = s.mount_path + "/.tutti/" +
                            std::string(name) + ".bin";

    int fd = ::open(host_path.c_str(),
                     O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        std::fprintf(stderr,
            "[nvme_storage] open(%s) for create: errno %d (%s)\n",
            host_path.c_str(), errno, std::strerror(errno));
        return false;
    }

    if (::fallocate(fd, 0, 0, (off_t)total_bytes) != 0) {
        std::fprintf(stderr,
            "[nvme_storage] fallocate(%s, %llu): errno %d (%s)\n",
            host_path.c_str(), (unsigned long long)total_bytes,
            errno, std::strerror(errno));
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }

    if (::fsync(fd) != 0) {
        std::fprintf(stderr,
            "[nvme_storage] fsync(%s): errno %d\n", host_path.c_str(), errno);
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }

    auto fr = read_extents(fd, kNvmeBlockSize);
    if (!fr.ok) {
        std::fprintf(stderr,
            "[nvme_storage] read_extents(%s): %s\n",
            host_path.c_str(), fr.error.c_str());
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }

    // Build header.
    NvmeFileHeader hdr;
    std::memset(&hdr, 0, sizeof(hdr));
    hdr.magic           = kNvmeFileHeaderMagic;
    hdr.version         = kNvmeFileHeaderVersion;
    hdr.fs_block_size   = fr.fs_block_size;
    hdr.file_size_bytes = size_bytes;
    hdr.file_id         = s.log->next_file_id();
    hdr.num_extents     = (uint32_t)fr.extents.size();
    {
        size_t name_len = std::min(name.size(), sizeof(hdr.name) - 1);
        std::memcpy(hdr.name, name.data(), name_len);
    }
    for (size_t i = 0; i < fr.extents.size(); ++i) {
        hdr.extents[i] = fr.extents[i];
    }

    // Write header at byte 0.
    if (::pwrite(fd, &hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        std::fprintf(stderr,
            "[nvme_storage] pwrite(header) on %s: errno %d\n",
            host_path.c_str(), errno);
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }
    if (::fsync(fd) != 0) {
        std::fprintf(stderr,
            "[nvme_storage] fsync(after header) on %s: errno %d\n",
            host_path.c_str(), errno);
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }

    // Add to PersistentFileLog and persist.
    PersistentFileLog::Entry e{};
    e.file_id    = hdr.file_id;
    e.name       = std::string(name);
    e.size_bytes = size_bytes;
    e.extents    = fr.extents;
    if (!s.log->add(std::move(e))) {
        std::fprintf(stderr,
            "[nvme_storage] log.add returned false (race?)\n");
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }
    if (!s.log->persist()) {
        std::fprintf(stderr,
            "[nvme_storage] log.persist failed\n");
        // Best-effort rollback: remove from log + delete host file.
        s.log->remove(hdr.file_id);
        ::close(fd);
        ::unlink(host_path.c_str());
        return false;
    }

    auto nf = std::make_unique<NvmeFile>();
    nf->id          = hdr.file_id;
    nf->name        = std::string(name);
    nf->size_bytes  = size_bytes;
    nf->device      = s.device;
    nf->extents     = fr.extents;
    nf->host_fd     = fd;
    nf->host_path   = host_path;
    nf->data_offset = sizeof(NvmeFileHeader);
    NvmeFile* raw = nf.get();
    s.files[hdr.file_id] = std::move(nf);
    *out = raw;
    return true;
}

NvmeFile* HostFsBackedNvmeStorage::create_file(const Device* dev,
                                                std::string_view name,
                                                uint64_t size_bytes)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto* s = find_state(dev);
    if (s == nullptr) {
        std::fprintf(stderr,
            "[nvme_storage] create_file: device not bootstrapped\n");
        return nullptr;
    }
    NvmeFile* out = nullptr;
    if (!create_file_locked(*s, name, size_bytes, &out)) return nullptr;
    return out;
}

NvmeFile* HostFsBackedNvmeStorage::open_file(const Device* dev,
                                              std::string_view name)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto* s = find_state(dev);
    if (s == nullptr) return nullptr;

    std::string nm(name);
    const auto* e = s->log->find_by_name(nm);
    if (e == nullptr) return nullptr;

    // Already open?
    auto it = s->files.find(e->file_id);
    if (it != s->files.end()) {
        // Re-open host_fd if previously closed.
        if (it->second->host_fd < 0) {
            int fd = ::open(it->second->host_path.c_str(),
                             O_RDWR | O_CLOEXEC);
            if (fd < 0) {
                std::fprintf(stderr,
                    "[nvme_storage] open(%s) for reopen: errno %d\n",
                    it->second->host_path.c_str(), errno);
                return nullptr;
            }
            it->second->host_fd = fd;
        }
        return it->second.get();
    }

    // Not in memory yet; reconstruct from log + reopen fd.
    std::string host_path = s->mount_path + "/.tutti/" + nm + ".bin";
    int fd = ::open(host_path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::fprintf(stderr,
            "[nvme_storage] open(%s) for open_file: errno %d\n",
            host_path.c_str(), errno);
        return nullptr;
    }
    auto nf = std::make_unique<NvmeFile>();
    nf->id          = e->file_id;
    nf->name        = e->name;
    nf->size_bytes  = e->size_bytes;
    nf->device      = s->device;
    nf->extents     = e->extents;
    nf->host_fd     = fd;
    nf->host_path   = host_path;
    nf->data_offset = sizeof(NvmeFileHeader);
    NvmeFile* raw = nf.get();
    s->files[e->file_id] = std::move(nf);
    return raw;
}

bool HostFsBackedNvmeStorage::close_file(NvmeFile* file) {
    if (file == nullptr) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    auto* s = find_state(file->device);
    if (s == nullptr) return false;

    if (file->host_fd >= 0) {
        if (::fsync(file->host_fd) != 0) {
            std::fprintf(stderr,
                "[nvme_storage] close_file fsync(%s): errno %d\n",
                file->host_path.c_str(), errno);
            // continue anyway -- best effort
        }
        ::close(file->host_fd);
        file->host_fd = -1;
    }
    return s->log->persist();
}

bool HostFsBackedNvmeStorage::delete_file(NvmeFile* file) {
    if (file == nullptr) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    auto* s = find_state(file->device);
    if (s == nullptr) return false;

    uint64_t fid = file->id;
    std::string host_path = file->host_path;

    if (file->host_fd >= 0) {
        ::close(file->host_fd);
        file->host_fd = -1;
    }

    if (::unlink(host_path.c_str()) != 0 && errno != ENOENT) {
        std::fprintf(stderr,
            "[nvme_storage] unlink(%s): errno %d\n",
            host_path.c_str(), errno);
        return false;
    }

    s->files.erase(fid);
    if (!s->log->remove(fid)) {
        // Already gone from log; not fatal.
    }
    return s->log->persist();
}

std::vector<NvmeFile*>
HostFsBackedNvmeStorage::list_files(const Device* dev) const {
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<NvmeFile*> out;
    const auto* s = find_state(dev);
    if (s == nullptr) return out;
    out.reserve(s->files.size());
    for (const auto& [fid, ptr] : s->files) {
        out.push_back(ptr.get());
    }
    return out;
}

// ---------------------------------------------------------------------------
// Host-side IO
// ---------------------------------------------------------------------------

ssize_t HostFsBackedNvmeStorage::read_blocking(NvmeFile* file,
                                                uint64_t byte_offset,
                                                void* dst, size_t len)
{
    if (file == nullptr || file->host_fd < 0 || dst == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (byte_offset + len > file->size_bytes) {
        errno = EINVAL;
        return -1;
    }
    off_t real_off = (off_t)(file->data_offset + byte_offset);
    auto* p = static_cast<uint8_t*>(dst);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t r = ::pread(file->host_fd, p, remaining, real_off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) break;
        p += r;
        remaining -= (size_t)r;
        real_off += r;
    }
    return (ssize_t)(len - remaining);
}

ssize_t HostFsBackedNvmeStorage::write_blocking(NvmeFile* file,
                                                 uint64_t byte_offset,
                                                 const void* src, size_t len)
{
    if (file == nullptr || file->host_fd < 0 || src == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (byte_offset + len > file->size_bytes) {
        errno = EINVAL;
        return -1;
    }
    off_t real_off = (off_t)(file->data_offset + byte_offset);
    const auto* p = static_cast<const uint8_t*>(src);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t r = ::pwrite(file->host_fd, p, remaining, real_off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) break;
        p += r;
        remaining -= (size_t)r;
        real_off += r;
    }
    return (ssize_t)(len - remaining);
}

bool HostFsBackedNvmeStorage::sync(NvmeFile* file) {
    if (file == nullptr || file->host_fd < 0) return false;
    return ::fsync(file->host_fd) == 0;
}

} // namespace tutti
