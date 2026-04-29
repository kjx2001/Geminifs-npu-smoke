#!/usr/bin/env bash
# 01_config.sh -- Module: config parser + validate_config.
#
# This script feeds the daemon a series of intentionally-bad YAML configs
# and asserts that each is rejected with a recognisable error message.
# It does NOT require real NVMe hardware -- the daemon exits during
# parse/validate before reaching ServiceState init.
#
# It also feeds one *good* config; that one is allowed to fail later
# (during ServiceState init) on hosts without real hardware -- we only
# require the parse-validate stage to succeed (i.e. no "validation
# failed:" prefix in the captured output).

set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
require_binaries

CFG_DIR="$HERE/configs"

run_bad_case() {
    local label="$1" cfg="$2" expected_msg="$3"

    info "case: $label  (config: $(basename "$cfg"))"
    local out
    out="$(run_daemon_oneshot "$cfg")"
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        fail "[$label] expected non-zero exit, got 0
$out"
    fi
    assert_contains "$out" "$expected_msg"
    ok "rejected with: $expected_msg"
}

# --- Bad configs ---
run_bad_case "missing nvme.mount_path" \
    "$CFG_DIR/bad_missing_mount.yaml" \
    "mount_path is empty"

run_bad_case "queue_groups gpu_id not in gpus[]" \
    "$CFG_DIR/bad_unknown_gpu.yaml" \
    "has no matching entry in gpus[]"

run_bad_case "queue_groups count sum > total_queues" \
    "$CFG_DIR/bad_count_overflow.yaml" \
    "exceeds total_queues"

run_bad_case "duplicate nvmes[].mount_path" \
    "$CFG_DIR/bad_dup_mount.yaml" \
    "duplicate nvmes[].mount_path"

# --- Good config: parse+validate must pass; init may fail on no-hw hosts ---
GOOD_CFG="$(mktemp -t nvmesvc_good.XXXXXX.yaml)"
trap 'rm -f "$GOOD_CFG"' EXIT
write_config_single_gpu "$GOOD_CFG"

info "case: well-formed single_gpu (parse/validate must pass)"
out="$(run_daemon_oneshot "$GOOD_CFG")"
# Whether the daemon went on to crash in init is fine here -- this
# script only audits the config layer. Forbid the validate-failed marker.
if [[ "$out" == *"validation failed:"* ]]; then
    fail "well-formed config was rejected by validate_config:
$out"
fi
if [[ "$out" == *"YAML parse error:"* ]]; then
    fail "well-formed config failed YAML parse:
$out"
fi
ok "parse + validate passed"

info "all config-layer cases passed"
