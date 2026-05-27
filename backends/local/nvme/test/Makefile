# Build the L1 (libnvm-level) sanity test.
#
# This directory hosts smokes that live ONE layer above the raw-ioctl
# smokes in backends/local/kernel_modules/test/ -- i.e. they link
# against libnvm.so and exercise the C API surface (nvm_ctrl_init,
# nvm_dma_map_*, etc.) rather than poking the snvme ioctl interface
# directly.
#
# Layered test inventory (each layer isolated in its own directory):
#
#   backends/local/kernel_modules/test/   -- L0: raw ioctl smokes
#   backends/local/nvme/test/             -- L1: libnvm C API smokes  (this dir)
#   backends/local/NVMeService/test/      -- L2: NVMeService client/daemon
#                                                IPC smokes  (planned)
#   filesystems/.../test/                 -- L3: Controller / libgeminifs
#                                                smokes  (planned)
#
# Initial L1 binaries:
#
#   snvme_smoke_libnvm  -- Commit 1 of L1: validates the B3/B6 DMA API
#                          (nvm_dma_map_data_*, nvm_dma_map_ring_*) and
#                          confirms the legacy nvm_dma_map_host(-1,-1)
#                          path still travels the UNSPECIFIED fallback
#                          branch through the kernel.
#                          DESTRUCTIVE: binds the controller.
#
# Build artefacts depend on libnvm.so + its public headers.  We resolve
# both out of the CMake build tree at $(REPO_ROOT)/build/ by default;
# override BUILD_DIR= if you build out-of-tree elsewhere.

THIS_DIR  := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# THIS_DIR  = .../backends/local/nvme/test/
# REPO_ROOT = .../  (four levels up)
REPO_ROOT := $(abspath $(THIS_DIR)/../../../..)

BUILD_DIR ?= $(REPO_ROOT)/build

LIBNVM_INC := $(REPO_ROOT)/backends/local/nvme/libnvm/include
LIBNVM_LIB := $(BUILD_DIR)/lib

# Pre-flight: libnvm.so must exist.  If not, point the user at the
# right build command instead of producing a cryptic ld error.
LIBNVM_SO := $(LIBNVM_LIB)/libnvm.so

TARGET_LIBNVM := snvme_smoke_libnvm
TARGET_LIBNVM_B3 := snvme_smoke_libnvm_b3

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -Wno-unused-parameter -std=c11 \
           -I$(LIBNVM_INC)
LDFLAGS ?= -L$(LIBNVM_LIB) -Wl,-rpath,$(LIBNVM_LIB) -lnvm

.PHONY: all clean check-libnvm run

all: check-libnvm $(TARGET_LIBNVM) $(TARGET_LIBNVM_B3)

check-libnvm:
	@if [ ! -f "$(LIBNVM_SO)" ]; then \
	    echo "ERROR: $(LIBNVM_SO) not found."; \
	    echo "       Build libnvm first:"; \
	    echo "         cd $(BUILD_DIR) && cmake --build . --target libnvm -j"; \
	    echo "       Or override BUILD_DIR= if your build tree lives elsewhere."; \
	    exit 1; \
	fi

$(TARGET_LIBNVM): snvme_smoke_libnvm.c $(LIBNVM_SO)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TARGET_LIBNVM_B3): snvme_smoke_libnvm_b3.c $(LIBNVM_SO)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET_LIBNVM) $(TARGET_LIBNVM_B3)

# Convenience: PCI=0000:08:00.0 make run
run: $(TARGET_LIBNVM)
	@if [ -z "$(PCI)" ]; then \
	    echo "usage: PCI=DDDD:BB:DD.F make run"; exit 1; \
	fi
	sudo ./$(TARGET_LIBNVM) $(PCI)
