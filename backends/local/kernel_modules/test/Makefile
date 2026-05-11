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

TARGET      := snvme_smoke
TARGET_GPU  := snvme_smoke_gpu

UAPI_INC := ../../nvme/libnvm/include

CFLAGS    ?= -O2 -Wall -Wextra -Wno-unused-parameter -std=c11 -I$(UAPI_INC)
LDFLAGS   ?=

NVCC       ?= nvcc
CUDA_ARCH  ?= sm_70
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
