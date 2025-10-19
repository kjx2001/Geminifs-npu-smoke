# GeminiFS: A Companion File System for GPUs
This is the open-source repository for our paper: **GeminiFS: A Companion File System for GPUs** \
This code is still a "toy", and we will fix the bugs, make it better.

## Directory structure
```shell
GeminiFS
|--examples  # examples for how to use GeminiFS
|--libgeminifs       # lib for geminifs+
|--libnvm       # lib for GPU nvme driver(modified from BaM)
|--module    # modifield NVMe module for CPU/GPU (snvme)
|--scripts   # scripts for system setup
|--src       # bam src to construct the GPU-NVMe path
```
## How to build
### 1. System Configurations ###
* `Above 4G Decoding` needs to be ENABLED in the BIOS
* The system's IOMMU should be disabled for ease of debugging.
  * In Intel Systems, this requires disabling `Vt-d` in the BIOS
  * In AMD Systems, this requires disabling `IOMMU` in the BIOS
* The `iommu` support in Linux must be disabled too, which can be checked and disabled following the instructions [below](#disable-iommu-in-linux).
* In the system's BIOS, `ACS` must be disabled if the option is available
* Linux kernel (ie. native 5.15.0).  6.9 kernel and 5.15.0-100-generic in Ubuntu 20.04 may not work.
* CMake 4.0.3 or newer and the _FindCUDA_ package for CMake. Using CMake 3.22.1 will report errors.
* GCC version 5.4.0 or newer. Compiler must support C++11 and POSIX threads.
* CUDA 12.4 or newer



### 2. Disable IOMMU in Linux ###
If you are using CUDA or implementing support for your own custom devices, 
you need to explicitly disable IOMMU as IOMMU support for peer-to-peer on 
Linux is a bit flaky at the moment. If you are not relying on peer-to-peer,
we would in fact recommend you leaving the IOMMU _on_ for protecting memory 
from rogue writes.

To check if the IOMMU is on, you can do the following:

```
$ cat /proc/cmdline | grep iommu
```

If either `iommu=on` or `intel_iommu=on` is found by `grep`, the IOMMU
is enabled.

You can disable it by removing `iommu=on` and `intel_iommu=on` from the 
`CMDLINE` variable in `/etc/default/grub` and then reconfiguring GRUB.
The next time you reboot, the IOMMU will be disabled.

### 2. Download Torch
```
./scripts/prepare_env.sh 
```

### 3. Build the Project
From the project root directory, do the following:
```shell
$ mkdir  build && cd build
$ cmake ..
$ make libgeminifs            # builds geminifs
$ make modules                 # make snvme kernel module
$ make insmod                 # insmod snvme kernel module
```

## How to run
Here, we provide partial functional verification code.
> Please modify the device and corresponding path in the code as needed.

Add a sys_config.ini like the example. 
It should be noted that different disks have different configurations due to their respective physical limitations.
For example, using the following cmd to get the to get the max IO 
cat /sys/block/nvme0n1/queue/max_hw_sectors_kb

### Test
```shell
cd examples/test_geminifs 
mkdir build && cd build
cmake ..
make TestForPythonInterface 
sudo ./TestForNvmeBacking.exe
```

