# Makefile  --  [SNVME-NPU 迁移新增文件 #3]  独立树外编译入口
#
# 原因
# ----
# 原始仓库用顶层 CMake 渲染 Makefile.in 里的 @module_root@ 等占位符再编译。
# 为了让你「拷贝整个目录到 NPU 服务器后直接 make」，这里提供一个不依赖
# CMake 的独立 Makefile。
#
# 用法
# ----
#   make                      # 对着当前运行内核编译，产出 snvme-core.ko + snvme.ko
#   make KDIR=/path/to/build  # 指定别的内核构建目录
#   make clean
#   sudo insmod snvme-core.ko && sudo insmod snvme.ko
#   sudo rmmod snvme && sudo rmmod snvme_core
#
# 对象集与原始 Makefile.in 完全一致：
#   snvme-core.ko = core.o + ioctl.o (+ multipath/zns/hwmon，按内核 CONFIG)
#   snvme.ko      = nvfs-pci.o + nvfs-p2p.o + list.o + ctrl.o + map.o + pci.o

KDIR ?= /lib/modules/$(shell uname -r)/build

# UAPI 头 ioctl.h 的位置。snvme-5.10-npu 与 nvme/ 的相对位置和原始
# snvme-5.15.0-public 一致（都在 kernel_modules/ 下），所以这个相对路径
# 在「整棵 Geminifs_smoke 一起拷过去」时依然成立。
UAPI_INC := $(src)/../../nvme/libnvm/include

obj-m += snvme-core.o
obj-m += snvme.o

snvme-core-y                         := core.o ioctl.o
snvme-core-$(CONFIG_NVME_MULTIPATH)  += multipath.o
snvme-core-$(CONFIG_BLK_DEV_ZONED)   += zns.o
snvme-core-$(CONFIG_NVME_HWMON)      += hwmon.o

# ---------------------------------------------------------------------------
# [SNVME-NPU 替换] GPU/NPU 显存 P2P 后端的编译期开关
#   make             -> 默认 NVIDIA 路线，链接 nvfs-p2p.o（nv-p2p 私有 API）
#   make SNVME_NPU=1 -> NPU 路线，-DSNVME_NPU，改链接 npu-p2p.o（内核 dma-buf 实现）
# 两份 wrapper 提供完全相同的 nvfs_nvidia_p2p_* 函数名/签名，map.c/pci.c 二选一链接、零改动。
# 详见 kjx/0620/02替换NPU代码分析.md。
# ---------------------------------------------------------------------------
ifeq ($(SNVME_NPU),1)
P2P_OBJ   := npu-p2p.o
ccflags-y += -DSNVME_NPU
else
P2P_OBJ   := nvfs-p2p.o
endif

snvme-objs := nvfs-pci.o $(P2P_OBJ) list.o ctrl.o map.o pci.o

# -I$(UAPI_INC): 让 pci.c/ioctl.c/core.c 能找到 "ioctl.h"（UAPI）
# -I$(src):      让各文件能找到本目录的 nv-p2p.h 桩头和 *.h
ccflags-y += -I$(UAPI_INC) -I$(src)

# 把 SNVME_NPU 显式透传给 kbuild 子 make（命令行变量虽会自动下传，显式更稳妥）。
all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) SNVME_NPU=$(SNVME_NPU) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
	rm -f *.o *.ko *.mod *.mod.c .*.cmd modules.order Module.symvers
