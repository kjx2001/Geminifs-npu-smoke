#!/usr/bin/env bash
# 05_reaper.sh -- Module: lease reaper reclaims dead-client allocations.
#
# Allocates from a client process, then SIGKILLs it (skipping the
# Allocation dtor that would otherwise send ReleaseQueues). Waits past
# lease.timeout_sec and checks that the daemon's reaper thread has
# returned the queues to the available pool.
#
# Uses the multi-GPU split config to additionally verify the reaper
# reclaims into the correct group (only the GPU that had the dead
# allocation should regain queues).
#
# Note: lease timing in lib.sh / config defaults: heartbeat=10s,
# timeout=30s. We override timeout to keep the test short.

set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
require_binaries
require_root_or_skip

# Override lease timing so the test wraps in well under a minute.
HEARTBEAT_INTERVAL_SEC=2
LEASE_TIMEOUT_SEC=6
WAIT_FOR_REAP_SEC=$(( LEASE_TIMEOUT_SEC + HEARTBEAT_INTERVAL_SEC + 4 ))

CFG="$(mktemp -t nvmesvc_reaper.XXXXXX.yaml)"
trap 'rm -f "$CFG"' RETURN
cat > "$CFG" <<EOF
grpc:
  endpoint: "${ENDPOINT}"

gpus:
  - id: 0
    mount_path: "${GPU0_MOUNT}"
  - id: 1
    mount_path: "${GPU1_MOUNT}"

nvmes:
  - pci_addr: "${PCI_ADDR}"
    mount_path: "${NVME_MOUNT}"
    namespace_id: 1
    queue_depth: 1024
    total_queues: 128
    queue_groups:
      - { gpu_id: 0, count: 64 }
      - { gpu_id: 1, count: 64 }

queue_pool:
  default_per_client: 32
  max_per_client: 128

lease:
  heartbeat_interval_sec: ${HEARTBEAT_INTERVAL_SEC}
  timeout_sec: ${LEASE_TIMEOUT_SEC}
EOF

start_daemon "$CFG"
if ! wait_daemon_ready 15; then
    fail "daemon did not become ready on $ENDPOINT
--- daemon log ---
$(daemon_log)
------------------"
fi
ok "daemon listening"

# --- Spawn a client that allocates 32 queues on GPU 1 and holds for a long
# time; then SIGKILL it so the dtor never runs (no Release RPC sent). ---
"$CLIENT_BIN" --endpoint "$ENDPOINT" \
    --device 0 --cuda 1 --count 32 --hold 600 \
    >/tmp/nvmesvc_reaper_client.log 2>&1 &
VICTIM_PID=$!
trap 'kill -9 $VICTIM_PID 2>/dev/null || true; rm -f "$CFG" /tmp/nvmesvc_reaper_client.log' RETURN

# Give the client a moment to register the allocation.
sleep 2

# --- During hold: GPU 1 should show 32/64 available; GPU 0 still 64/64. ---
busy="$(run_client_list)"
echo "$busy"
assert_contains "$busy" "cuda_device=1"
assert_contains "$busy" "avail=32/64"
assert_contains "$busy" "cuda_device=0"
assert_contains "$busy" "avail=64/64"
ok "during hold: GPU 1 = 32/64, GPU 0 = 64/64"

# --- SIGKILL the client to skip its dtor + Release RPC. ---
info "SIGKILL victim client (pid=$VICTIM_PID); waiting ${WAIT_FOR_REAP_SEC}s for reaper"
kill -9 "$VICTIM_PID" 2>/dev/null || true
wait "$VICTIM_PID" 2>/dev/null || true
sleep "$WAIT_FOR_REAP_SEC"

# --- After reap: both groups should be 64/64. ---
post="$(run_client_list)"
echo "$post"
assert_contains "$post" "cuda_device=0"
assert_contains "$post" "cuda_device=1"
# Both groups must show full availability now.
gpu0_avail="$(echo "$post" | grep -oE 'cuda_device=0 range=\[0, 64\) avail=[0-9]+/64' | grep -oE 'avail=[0-9]+/64' | tail -1)"
gpu1_avail="$(echo "$post" | grep -oE 'cuda_device=1 range=\[64, 128\) avail=[0-9]+/64' | grep -oE 'avail=[0-9]+/64' | tail -1)"
[[ "$gpu0_avail" == "avail=64/64" ]] || fail "GPU 0 not full after reap: $gpu0_avail"
[[ "$gpu1_avail" == "avail=64/64" ]] || fail "GPU 1 not reclaimed after reap: $gpu1_avail"
ok "reaper reclaimed dead client's allocation; both groups full again"

info "reaper module passed"
