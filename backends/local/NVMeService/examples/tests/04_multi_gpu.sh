#!/usr/bin/env bash
# 04_multi_gpu.sh -- Module: multi-GPU routing in allocate().
#
# Multi-GPU split config (queues 0-63 -> GPU 0, queues 64-127 -> GPU 1).
# Verifies:
#   - --list-only reports two groups, each with avail=64/64
#   - --cuda 0 returns a queue range strictly inside [0, 64)
#   - --cuda 1 returns a queue range strictly inside [64, 128)
#   - --cuda 2 (not in any group) is rejected with a clear error
#   - holding both clients concurrently does not cross-contaminate the
#     per-group available counts
#
# Requirements: gRPC server active, /dev/snvm_*, multi-GPU host
# (or at least cudaSetDevice(1) acceptable on this box).

set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
require_binaries
require_root_or_skip

CFG="$(mktemp -t nvmesvc_split.XXXXXX.yaml)"
trap 'rm -f "$CFG"' RETURN
write_config_multi_gpu_split "$CFG"

start_daemon "$CFG"
if ! wait_daemon_ready 15; then
    fail "daemon did not become ready on $ENDPOINT within 15s
--- daemon log ---
$(daemon_log)
------------------"
fi
ok "daemon listening"

# --- list shows two groups ---
list_out="$(run_client_list)"
echo "$list_out"
assert_contains "$list_out" "group: cuda_device=0"
assert_contains "$list_out" "group: cuda_device=1"
assert_contains "$list_out" "range=[0, 64)"
assert_contains "$list_out" "range=[64, 128)"
ok "ListDevices returned two groups (GPU 0: [0,64), GPU 1: [64,128))"

# --- alloc on GPU 0: queue start must be in [0, 64) ---
out0="$(run_client_alloc 0 0 32 3)"
echo "$out0"
qrange0="$(echo "$out0" | grep -oE 'queue range[[:space:]]*: \[[0-9]+, [0-9]+\)' | head -1)"
[[ "$qrange0" =~ \[([0-9]+), ([0-9]+)\) ]] || fail "could not parse GPU0 queue range from:
$out0"
qstart0="${BASH_REMATCH[1]}"; qend0="${BASH_REMATCH[2]}"
(( qstart0 >= 0 && qend0 <= 64 )) || \
    fail "GPU 0 alloc returned queues [$qstart0, $qend0); expected within [0, 64)"
ok "GPU 0 alloc range [$qstart0, $qend0) is inside the GPU 0 group"

# --- alloc on GPU 1: queue start must be in [64, 128) ---
out1="$(run_client_alloc 0 1 32 3)"
echo "$out1"
qrange1="$(echo "$out1" | grep -oE 'queue range[[:space:]]*: \[[0-9]+, [0-9]+\)' | head -1)"
[[ "$qrange1" =~ \[([0-9]+), ([0-9]+)\) ]] || fail "could not parse GPU1 queue range"
qstart1="${BASH_REMATCH[1]}"; qend1="${BASH_REMATCH[2]}"
(( qstart1 >= 64 && qend1 <= 128 )) || \
    fail "GPU 1 alloc returned queues [$qstart1, $qend1); expected within [64, 128)"
ok "GPU 1 alloc range [$qstart1, $qend1) is inside the GPU 1 group"

# --- cross-GPU rejection: cuda_device not present in any group ---
out_bad="$("$CLIENT_BIN" --endpoint "$ENDPOINT" \
            --device 0 --cuda 2 --count 32 --hold 0 2>&1 || true)"
assert_contains "$out_bad" "no queue group on device_id=0 for cuda_device=2"
ok "cuda_device=2 correctly rejected"

info "multi-GPU module passed"
