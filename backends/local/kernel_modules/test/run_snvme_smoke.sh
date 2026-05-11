#!/usr/bin/env bash
# run_snvme_smoke.sh -- one-shot wrapper around snvme_smoke / snvme_smoke_gpu.
#
# What it does, in order:
#   1. (Re)builds the test binaries via the local Makefile.
#   2. Verifies that /dev/snvm_control exists; if not, refuses to run.
#   3. Picks a PCI BDF: from $PCI_ADDR if set, else from the first NVMe
#      device the kernel knows about.
#   4. Runs the chosen smoke test:
#        --gpu       : ./snvme_smoke_gpu (additionally exercises GPU paths)
#        otherwise   : ./snvme_smoke      (UAPI-only, libc, no CUDA needed)
#      Add --bind to also exercise the destructive bring-up.
#
# Usage:
#   ./run_snvme_smoke.sh                          # host UAPI smoke (safe)
#   ./run_snvme_smoke.sh --gpu                    # + GPU map paths (safe)
#   ./run_snvme_smoke.sh --gpu --bind             # GPU + full bring-up (destructive)
#   ./run_snvme_smoke.sh --gpu --gpu-id 1         # use cuda device 1
#   PCI_ADDR=0000:50:00.0 ./run_snvme_smoke.sh
#   ./run_snvme_smoke.sh 0000:50:00.0 --gpu --bind

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- Parse args ---
BIND_FLAG=""
GPU_MODE=""
GPU_ID=""
ARG_BDF=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bind)    BIND_FLAG="--bind"; shift ;;
        --gpu)     GPU_MODE="1"; shift ;;
        --gpu-id)  GPU_ID="$2"; GPU_MODE="1"; shift 2 ;;
        --help|-h)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)         ARG_BDF="$1"; shift ;;
    esac
done

# --- Build ---
echo "[+] building snvme smoke tests ..."
make -s

# --- Pre-flight ---
if [[ ! -e /dev/snvm_control ]]; then
    echo "[!] /dev/snvm_control not found." >&2
    echo "    Load the kernel module first, e.g.:" >&2
    echo "      sudo insmod /path/to/snvme-core.ko" >&2
    echo "      sudo insmod /path/to/snvme.ko" >&2
    exit 1
fi

# --- Choose binary ---
if [[ -n "$GPU_MODE" ]]; then
    BIN="./snvme_smoke_gpu"
    if [[ ! -x "$BIN" ]]; then
        echo "[!] $BIN not built. Make sure 'nvcc' is on PATH and re-run." >&2
        exit 1
    fi
else
    BIN="./snvme_smoke"
fi

# --- Pick BDF ---
BDF="${ARG_BDF:-${PCI_ADDR:-}}"
if [[ -z "$BDF" ]]; then
    BDF="$(lspci -D -d ::0108 2>/dev/null | awk 'NR==1 {print $1}')"
fi
if [[ -z "$BDF" ]]; then
    echo "[!] No NVMe BDF found. Set PCI_ADDR=DDDD:BB:DD.F or pass it as arg." >&2
    exit 1
fi
echo "[+] using PCI BDF: $BDF"
if [[ -n "$BIND_FLAG" ]]; then
    echo "[!] --bind requested: this will detach the in-tree nvme driver"
    echo "    from $BDF for the duration of the test."
fi

# --- Run ---
EXTRA=()
[[ -n "$BIND_FLAG" ]] && EXTRA+=("$BIND_FLAG")
[[ -n "$GPU_ID"    ]] && EXTRA+=("--gpu" "$GPU_ID")
exec sudo "$BIN" "${EXTRA[@]}" "$BDF"
