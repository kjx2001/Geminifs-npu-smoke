<img src="doc/pics/tardis_logo.png" align="left" width="32" />

## Tardis: Companion File System for GPUs
Data-centric Storage Architecture for GPU Application 

## Directory structure
```shell
GeminiFS
|--examples  # examples for how to use GeminiFS
|--backends  # storage backends (e.g., local NVMe)
|   `--local/nvme/libnvm  # lib for GPU nvme driver(modified from BaM)
|--filesystems  # filesystem implementations
|   `--local/ext4/libgeminifs  # lib for geminifs+
|--kernel_modules  # Linux kernel modules
|   `--snvme  # modified NVMe module for CPU/GPU (for backends/local/nvme)
|--scripts   # scripts for system setup
```
## How to build and run
See [filesystems/local/ext4/README.md](filesystems/local/ext4/README.md).

