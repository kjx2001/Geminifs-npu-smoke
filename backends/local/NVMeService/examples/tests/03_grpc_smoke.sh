#!/usr/bin/env bash
# 03_grpc_smoke.sh -- Module: gRPC ListDevices + AllocateQueues + Release.
#
# Single-GPU config. Verifies:
#   - daemon serves gRPC on cfg.grpc.endpoint
#   - --list-only returns one device with one group
#   - --device 0 (no --cuda) successfully allocates a contiguous range
#     and releases it on dtor
#   - after release, available count is restored
#
# Requirements: same as 02_init.sh, plus the daemon's gRPC server block
# in examples/nvmeservice_daemon.cpp must be active (uncommented).

set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
require_binaries
require_root_or_skip

CFG="$(mktemp -t nvmesvc_grpc.XXXXXX.yaml)"
trap 'rm -f "$CFG"' RETURN
write_config_single_gpu "$CFG"

start_daemon "$CFG"
if ! wait_daemon_ready 15; then
    fail "daemon did not become ready on $ENDPOINT within 15s
--- daemon log ---
$(daemon_log)
------------------
note: if the daemon binary exits immediately after init, the gRPC server
block in examples/nvmeservice_daemon.cpp is likely still commented out.
Uncomment it (around the start_reaper / BuildAndStart calls) and rebuild."
fi
ok "daemon listening on $ENDPOINT"

# --- list-only: expect exactly one device, exactly one group ---
list_out="$(run_client_list)"
echo "$list_out"
assert_contains "$list_out" "device_id=0"
assert_contains "$list_out" "pci=$PCI_ADDR"
assert_contains "$list_out" "group: cuda_device=0"
assert_contains "$list_out" "avail=128/128"
ok "ListDevices returned one device with one full group"

# --- allocate 32 queues, hold 3s, dtor releases ---
alloc_out="$(run_client_alloc 0 --default 32 3)"
echo "$alloc_out"
assert_contains "$alloc_out" "allocation_id"
assert_contains "$alloc_out" "queue range"
assert_contains "$alloc_out" "Done."
ok "allocate + release flow OK"

# --- post-release availability is restored ---
post_out="$(run_client_list)"
assert_contains "$post_out" "avail=128/128"
ok "post-release availability restored to 128/128"

info "gRPC smoke module passed"
