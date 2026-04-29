#!/usr/bin/env bash
# lib.sh -- shared helpers for NVMeService module tests.
#
# Usage from a test script:
#   source "$(dirname "$0")/lib.sh"
#
# Environment knobs (override at the call site):
#   BIN_DIR     where the built binaries live (default: build/bin under repo root)
#   PCI_ADDR    NVMe PCI address used in generated configs (default: 0000:50:00.0)
#   ENDPOINT    gRPC endpoint the daemon listens on (default: 127.0.0.1:50051)
#   GPU0_MOUNT  path used as gpus[0].mount_path                (default: /mnt/gpu0)
#   GPU1_MOUNT  path used as gpus[1].mount_path                (default: /mnt/gpu1)
#   NVME_MOUNT  path used as nvmes[0].mount_path               (default: /mnt/nvme0)

set -uo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
: "${BIN_DIR:=${REPO_ROOT}/build/bin}"
: "${PCI_ADDR:=0000:50:00.0}"
: "${ENDPOINT:=127.0.0.1:50051}"
: "${GPU0_MOUNT:=/mnt/gpu0}"
: "${GPU1_MOUNT:=/mnt/gpu1}"
: "${NVME_MOUNT:=/mnt/nvme0}"

DAEMON_BIN="${BIN_DIR}/nvmeservice_daemon"
CLIENT_BIN="${BIN_DIR}/nvmeservice_client"

DAEMON_PID=""
DAEMON_LOG=""

# ---------------------------------------------------------------------------
# Output / assertion helpers
# ---------------------------------------------------------------------------

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }
fail()  { printf '\033[1;31m[FAIL]\033[0m  %s\n' "$*" >&2; exit 1; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*" >&2; }

assert_file_exists() {
    [[ -e "$1" ]] || fail "expected file/dir to exist: $1"
}
assert_no_file() {
    [[ ! -e "$1" ]] || fail "expected file/dir NOT to exist: $1"
}
assert_symlink_target() {
    local link="$1" expected="$2"
    [[ -L "$link" ]] || fail "expected symlink at $link"
    local actual
    actual="$(readlink "$link")"
    [[ "$actual" == "$expected" ]] || \
        fail "symlink $link -> $actual, expected -> $expected"
}
assert_contains() {
    local haystack="$1" needle="$2"
    [[ "$haystack" == *"$needle"* ]] || \
        fail "expected output to contain: $needle"$'\n''---got---'$'\n'"$haystack"
}
assert_not_contains() {
    local haystack="$1" needle="$2"
    [[ "$haystack" != *"$needle"* ]] || \
        fail "expected output NOT to contain: $needle"$'\n''---got---'$'\n'"$haystack"
}

# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

# Write a single-GPU config (1 NVMe → 1 GPU, all queues bound).
# Args: out_path
write_config_single_gpu() {
    local out="$1"
    cat > "$out" <<EOF
grpc:
  endpoint: "${ENDPOINT}"

gpus:
  - id: 0
    mount_path: "${GPU0_MOUNT}"

nvmes:
  - pci_addr: "${PCI_ADDR}"
    mount_path: "${NVME_MOUNT}"
    namespace_id: 1
    queue_depth: 1024
    total_queues: 128
    queue_groups:
      - { gpu_id: 0, count: 128 }

queue_pool:
  default_per_client: 32
  max_per_client: 128

lease:
  heartbeat_interval_sec: 10
  timeout_sec: 30
EOF
}

# Write a multi-GPU split config (1 NVMe split between 2 GPUs).
# Args: out_path
write_config_multi_gpu_split() {
    local out="$1"
    cat > "$out" <<EOF
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
  heartbeat_interval_sec: 10
  timeout_sec: 30
EOF
}

# ---------------------------------------------------------------------------
# Daemon lifecycle
# ---------------------------------------------------------------------------

# Start the daemon in the background. Captures stdout+stderr to a temp log.
# Args: config_path
# Sets DAEMON_PID and DAEMON_LOG.
start_daemon() {
    local cfg="$1"
    DAEMON_LOG="$(mktemp -t nvmesvc_daemon.XXXXXX.log)"
    info "starting daemon (config=$cfg, log=$DAEMON_LOG)"
    "$DAEMON_BIN" --config "$cfg" >"$DAEMON_LOG" 2>&1 &
    DAEMON_PID=$!
}

# Wait until the daemon is accepting gRPC connections, OR has exited.
# Returns 0 if ready, 1 if it exited before becoming ready.
# Args: timeout_seconds (default 10)
wait_daemon_ready() {
    local timeout="${1:-10}"
    local host="${ENDPOINT%:*}" port="${ENDPOINT##*:}"
    for ((i = 0; i < timeout * 10; ++i)); do
        if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
            return 1   # exited early
        fi
        if (echo > /dev/tcp/"$host"/"$port") 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# Stop the daemon if running. Tries SIGINT first, escalates to SIGKILL.
stop_daemon() {
    [[ -z "$DAEMON_PID" ]] && return 0
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill -INT "$DAEMON_PID" 2>/dev/null || true
        for ((i = 0; i < 30; ++i)); do
            kill -0 "$DAEMON_PID" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL "$DAEMON_PID" 2>/dev/null || true
    fi
    wait "$DAEMON_PID" 2>/dev/null || true
    DAEMON_PID=""
}

# Read the daemon's captured log; useful inside test assertions.
daemon_log() { cat "$DAEMON_LOG" 2>/dev/null || true; }

# Run the daemon, collect stdout+stderr+exit code, do NOT background.
# Use this for tests that just want to exercise the init path with a
# possibly-bad config and inspect the message.
# Args: config_path
# Echoes captured combined output. Returns the daemon's exit code.
run_daemon_oneshot() {
    local cfg="$1"
    "$DAEMON_BIN" --config "$cfg" 2>&1
}

# Always cleanup on script exit.
trap 'stop_daemon' EXIT

# ---------------------------------------------------------------------------
# Client wrappers
# ---------------------------------------------------------------------------

run_client_list() {
    "$CLIENT_BIN" --endpoint "$ENDPOINT" --list-only
}

# Args: device_id [cuda_device|--default] [count] [hold_seconds]
run_client_alloc() {
    local dev="$1"
    local cuda="${2:---default}"
    local count="${3:-32}"
    local hold="${4:-5}"
    if [[ "$cuda" == "--default" ]]; then
        "$CLIENT_BIN" --endpoint "$ENDPOINT" \
            --device "$dev" --count "$count" --hold "$hold"
    else
        "$CLIENT_BIN" --endpoint "$ENDPOINT" \
            --device "$dev" --cuda "$cuda" --count "$count" --hold "$hold"
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

require_binaries() {
    [[ -x "$DAEMON_BIN" ]] || fail "daemon not found: $DAEMON_BIN
build with: cd build && make -j nvmeservice_daemon_example"
    [[ -x "$CLIENT_BIN" ]] || fail "client not found: $CLIENT_BIN
build with: cd build && make -j nvmeservice_client_example"
}

require_root_or_skip() {
    if [[ "$(id -u)" != "0" ]]; then
        warn "not running as root; mount/symlink ops may be denied"
    fi
}
