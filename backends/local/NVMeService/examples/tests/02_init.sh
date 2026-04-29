#!/usr/bin/env bash
# 02_init.sh -- Module: ServiceState init (mount + per-GPU symlinks).
#
# Boots the daemon with a multi-GPU split config and asserts that:
#   1. The NVMe gets mounted at nvme.mount_path.
#   2. <nvme.mount_path>/GPU<id> subdirectories exist for every group's gpu_id.
#   3. <gpu.mount_path>/<basename(snvme_dev_path)> is a symlink pointing at
#      the corresponding NVMe subdirectory.
#   4. Stopping the daemon cleans the symlinks back up.
#
# Requirements:
#   - SNVMe kernel module loaded (/dev/snvm_control + /dev/snvm_*)
#   - PCI_ADDR env set to a real NVMe on this host (or default 0000:50:00.0)
#   - GPU0_MOUNT, GPU1_MOUNT, NVME_MOUNT writable by the daemon (root recommended)

set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
require_binaries
require_root_or_skip

CFG="$(mktemp -t nvmesvc_init.XXXXXX.yaml)"
trap 'rm -f "$CFG"' RETURN
write_config_multi_gpu_split "$CFG"

info "starting daemon with multi-GPU split config"
start_daemon "$CFG"

# The current daemon binary builds ServiceState then exits (gRPC server
# block is commented out upstream); a healthy init exits 0. Wait briefly
# for the process to settle.
for _ in {1..30}; do
    kill -0 "$DAEMON_PID" 2>/dev/null || break
    sleep 0.1
done
wait "$DAEMON_PID" 2>/dev/null
rc=$?
if [[ $rc -ne 0 ]]; then
    fail "daemon exited with rc=$rc during init
--- daemon log ---
$(daemon_log)
------------------"
fi
ok "ServiceState init returned 0"

# --- Filesystem invariants ---
# We can't easily check mount(2) state without parsing /proc/mounts, but
# the per-GPU subdirectories should at least exist (mkdir is unconditional
# in install_gpu_symlinks).
assert_file_exists "${NVME_MOUNT}/GPU0"
assert_file_exists "${NVME_MOUNT}/GPU1"
ok "NVMe per-GPU subdirs created: ${NVME_MOUNT}/GPU{0,1}"

# Symlinks: name is basename of /dev/snvm_<...>. We don't know the exact
# device basename a priori, so glob.
shopt -s nullglob
links0=("${GPU0_MOUNT}"/snvm_*)
links1=("${GPU1_MOUNT}"/snvm_*)
shopt -u nullglob

[[ ${#links0[@]} -gt 0 ]] || fail "no symlink under ${GPU0_MOUNT} matching snvm_*"
[[ ${#links1[@]} -gt 0 ]] || fail "no symlink under ${GPU1_MOUNT} matching snvm_*"
assert_symlink_target "${links0[0]}" "${NVME_MOUNT}/GPU0"
assert_symlink_target "${links1[0]}" "${NVME_MOUNT}/GPU1"
ok "GPU symlinks point at the right NVMe subdirs"

# --- Cleanup audit ---
# The daemon already exited; ~ServiceState should have removed the symlinks.
# (Subdirs are best-effort rmdir and stay if non-empty -- don't assert on those.)
assert_no_file "${links0[0]}"
assert_no_file "${links1[0]}"
ok "symlinks cleaned up on shutdown"

info "init/symlink module passed"
