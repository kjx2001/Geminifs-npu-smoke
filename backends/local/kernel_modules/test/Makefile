# Build the standalone SNVMe sanity tests.
#
# Two binaries:
#   snvme_smoke       -- UAPI smoke, libc-only. Always built.
#   snvme_smoke_gpu   -- adds the NVM_MAP_DEVICE_*  GPU paths via CUDA.
#                        Built automatically if `nvcc` is on PATH.
#
# Both depend on nothing but the SNVMe UAPI header
#   ../../nvme/libnvm/include/ioctl.h
#
# Usage:
#   make
#   sudo ./snvme_smoke 0000:50:00.0              # UAPI-smoke (safe, no GPU)
#   sudo ./snvme_smoke --bind 0000:50:00.0       # full bring-up (destructive)
#   sudo ./snvme_smoke_gpu 0000:50:00.0          # UAPI + GPU map paths (safe)
#   sudo ./snvme_smoke_gpu --bind 0000:50:00.0   # + run probe with GPU queues (destructive)
#
# Override knobs:
#   CC, CFLAGS, NVCC, NVCCFLAGS, CUDA_ARCH
#
# GPU arch selection (CUDA_ARCH):
#   - default: auto-detect via `nvidia-smi --query-gpu=compute_cap`
#     so the build follows the host GPU (e.g. H20 -> sm_90,
#     A100 -> sm_80, V100 -> sm_70).
#   - fallback: sm_80 (Ampere A100), still accepted by every
#     CUDA toolkit from 11.0 through 13.x.
#   - override: `make CUDA_ARCH=sm_90` to force one explicitly.
#   The previous baseline hard-coded sm_70, which CUDA 13 has
#   dropped -- nvcc reports "Unsupported gpu architecture 'sm_70'"
#   on any toolkit >= 13.0 (e.g. /usr/local/cuda-13.0/).

TARGET      := snvme_smoke
TARGET_GPU  := snvme_smoke_gpu

UAPI_INC := ../../nvme/libnvm/include

CFLAGS    ?= -O2 -Wall -Wextra -Wno-unused-parameter -std=c11 -I$(UAPI_INC)
LDFLAGS   ?=

NVCC       ?= nvcc

# Probe the host's primary GPU compute capability and translate
# "X.Y" (e.g. "9.0") into "sm_XY" (e.g. "sm_90").  Empty if
# nvidia-smi is missing, returns no GPUs, or returns a malformed
# value -- in which case CUDA_ARCH falls through to the static
# default below.
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap \
                       --format=csv,noheader 2>/dev/null \
                   | head -n1 \
                   | sed -e 's/[[:space:]]//g' -e 's/\.//' )
ifneq ($(DETECTED_CC),)
CUDA_ARCH  ?= sm_$(DETECTED_CC)
else
# Static fallback if auto-detect fails.  sm_80 is the lowest arch
# that every CUDA toolkit from 11.0 through 13.x accepts AND that
# covers the GPU families we actually ship on (A100, H100/H20).
CUDA_ARCH  ?= sm_80
endif

NVCCFLAGS  ?= -O2 -std=c++17 -arch=$(CUDA_ARCH) -I$(UAPI_INC) \
              -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter

SRC     := snvme_smoke.c
SRC_GPU := snvme_smoke_gpu.cu

# Build the GPU test only if nvcc is available.
HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes || echo no)

.PHONY: all clean run run-gpu

ifeq ($(HAVE_NVCC),yes)
all: $(TARGET) $(TARGET_GPU)
else
all: $(TARGET)
	@echo "[i] nvcc not found; skipping $(TARGET_GPU)."
	@echo "    Install CUDA Toolkit (or set NVCC=...) and re-run make to enable it."
endif

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TARGET_GPU): $(SRC_GPU)
	@echo "[+] nvcc: CUDA_ARCH=$(CUDA_ARCH) (override with 'make CUDA_ARCH=sm_XX')"
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) $(TARGET_GPU)

# Convenience: PCI=0000:50:00.0 make run [BIND=1]
run: $(TARGET)
	@if [ -z "$(PCI)" ]; then \
	    echo "usage: PCI=DDDD:BB:DD.F make run [BIND=1]"; exit 1; \
	fi; \
	if [ "$(BIND)" = "1" ]; then \
	    sudo ./$(TARGET) --bind $(PCI); \
	else \
	    sudo ./$(TARGET) $(PCI); \
	fi

# Convenience: PCI=0000:50:00.0 make run-gpu [BIND=1]
run-gpu: $(TARGET_GPU)
	@if [ -z "$(PCI)" ]; then \
	    echo "usage: PCI=DDDD:BB:DD.F make run-gpu [BIND=1]"; exit 1; \
	fi; \
	if [ "$(BIND)" = "1" ]; then \
	    sudo ./$(TARGET_GPU) --bind $(PCI); \
	else \
	    sudo ./$(TARGET_GPU) $(PCI); \
	fi
