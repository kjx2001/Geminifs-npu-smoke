<img src="doc/pics/tardis_logo.png" align="left" width="32" />

## GeminiFS

GPU-oriented storage codebase under refactor toward a `Unified Storage Runtime`.

Note:

- `GeminiFS` is the current repository and legacy implementation name
- the long-term runtime name may change in a future version
- the active architecture baseline is tracked in [`Roadmap.md`](Roadmap.md)

## Current Status

This repository currently contains:

- the existing storage/runtime implementation
- a modified local NVMe stack, including kernel-module changes
- an `NVMeService` device-manager prototype
- architecture and refactor planning documents for `v0.1`

The repository is in a transition stage:

- current code layout reflects historical implementation boundaries
- target architecture is being redefined around `api`, `runtime`, `memory`, `device_manager`, `io_engine`, `backends`, and `adapters`
- interface and directory changes should be discussed before major code movement

## Start Here

If you are a new contributor or another AI agent, read these files first:

1. [`Roadmap.md`](Roadmap.md)
   Current `v0.1` architecture baseline, roadmap, feature snapshot, and known bugs.

2. [`Todolist.md`](Todolist.md)
   Current active work items, discussion items, and immediate priorities.

3. [`CONTRIBUTING.md`](CONTRIBUTING.md)
   Coding rules, interface discipline, collaboration rules, and commit naming requirements.

4. [`doc/history/README.md`](doc/history/README.md)
   Rules for versioned roadmap history and archive handling.

5. [`intergration/intergration.md`](intergration/intergration.md)
   Current upper-layer integration intent, especially for framework-facing APIs.

After that, read the implementation-specific documents relevant to your area:

- [`doc/design/backend-spi.md`](doc/design/backend-spi.md)
  Backend SPI contract — read this before implementing any new storage backend.

- [`doc/architecture/system-architecture.md`](doc/architecture/system-architecture.md)
  Full system architecture with component diagrams and data flow.

- [`backends/local/NVMeService/NVMeService.md`](backends/local/NVMeService/NVMeService.md)
- [`filesystems/ext4/README.md`](filesystems/ext4/README.md)

## Knowledge Keywords

These keywords help human contributors and AI agents search and understand the codebase faster.

### Domain Keywords

- `NVMe Specification`
- `PCIe`
- `NVMe queue`
- `submission queue`
- `completion queue`
- `queue pair`
- `doorbell`
- `PRP`
- `DMA`
- `pinned memory`
- `CUDA`
- `CUDA IPC`
- `GPU Direct Storage`
- `GDS`
- `RDMA`
- `Linux kernel module`
- `DKMS`
- `systemd`
- `EXT4 file system`
- `FIEMAP`
- `LMCache`
- `Mooncake`
- `device manager`
- `IO engine`
- `memory registration`
- `Unified Storage Runtime`

### Code Search Keywords

- `GeminiFS::init`
- `parse_and_setup_controllers`
- `GPUController`
- `NVMeController`
- `GPUFileManager`
- `GPUMemoryMapper`
- `FsAllocQueues`
- `FsReleaseQueues`
- `LeaseHeartbeat`
- `Controller::init_queues`
- `d_qps`
- `d_ctrl_ptr`
- `d_queue_acquire_helper`

### Search Strategy

When modifying or reviewing code, it is usually best to search in this order:

1. the architecture and roadmap terms in [`Roadmap.md`](Roadmap.md)
2. the subsystem documents
3. the exact code identifiers above
4. the local backend implementation under [`backends/local/`](backends/local/)
5. the current monolithic path under [`filesystems/ext4/libgeminifs`](filesystems/ext4/libgeminifs)

## Repository Map

This is the current repository structure as it exists today.

### Core Implementation Areas

- [`backends/local/nvme/libnvm`](backends/local/nvme/libnvm)
  User-space NVMe support library used by the local backend path.

- [`backends/local/kernel_modules/snvme`](backends/local/kernel_modules/snvme)
  Modified Linux NVMe kernel-module lineage used to support CPU/GPU access to NVMe queue resources.

- [`backends/local/NVMeService`](backends/local/NVMeService)
  Local device-manager prototype for controller initialization, queue leasing, and process attach flow.

- [`filesystems/ext4/libgeminifs`](filesystems/ext4/libgeminifs)
  Current monolithic implementation area. This is where GPU controller logic, NVMe controller logic, memory mapping, and GPU file management are currently mixed together.

### Supporting Areas

- [`examples/`](examples)
  Validation and example programs.

- [`scripts/`](scripts)
  Environment preparation, reset, binding, and operational scripts.

- [`doc/`](doc)
  Project documents, including roadmap history and future architecture material.

- [`build/`](build)
  Local generated build output. Treat as generated artifacts, not source-of-truth documentation.

## Current Collaboration Rules

When working in this repository:

- treat [`Roadmap.md`](Roadmap.md) as the active architecture source
- treat [`Todolist.md`](Todolist.md) as the active task board
- preserve version snapshots in [`doc/history/`](doc/history/) when the maintainer changes versions
- avoid assuming the current directory layout is the final layout
- avoid baking the `GeminiFS` name into new abstractions unless the maintainer explicitly wants it
- discuss top-level interface changes and directory reshaping before large code edits

## Current Known Problem

The currently tracked `v0.1` known bug is:

- `GPU file persistence` is not yet stable and must not be treated as a reliable persistence contract

See [`Roadmap.md`](Roadmap.md) for the version snapshot that tracks this.

## Build and Runtime Notes

The current build still reflects the existing implementation, not the final target architecture.

Relevant build/runtime entry points:

- root [`CMakeLists.txt`](CMakeLists.txt)
- [`filesystems/ext4/README.md`](filesystems/ext4/README.md)
- [`backends/local/NVMeService/examples/sys_config.yaml`](backends/local/NVMeService/examples/sys_config.yaml)
- root configuration samples:
  - [`sys_config.ini`](sys_config.ini)
  - [`sys_config.yaml`](sys_config.yaml)

Important operational constraint:

- the modified NVMe kernel module is part of the local backend baseline and must be considered in deployment, Linux-version compatibility, and startup sequencing
