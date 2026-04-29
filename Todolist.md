# Todolist

This file tracks the current active work for the repository.

It is not a long-term vision document.
Use [`Roadmap.md`](Roadmap.md) for versioned architecture and roadmap planning.

## Current Focus

- Refactor the project from a GPU-file-oriented implementation into a `Unified Storage Runtime`
- Freeze `v0.1` architecture boundaries before major code movement
- Keep interface and directory changes reviewable and explicit

## Active Tasks

- [ ] Finalize the `v0.1` target top-level directory structure
- [ ] Finalize the `v0.1` top-level public API boundaries
- [ ] Define the core runtime object model
- [ ] Define the standalone memory subsystem API
- [ ] Define the device manager and IO engine boundary
- [ ] Define the backend SPI for the first `local_nvme` backend
- [ ] Define how `LMCache` and `Mooncake` adapters will attach to the runtime
- [ ] Unify the configuration strategy and remove split config semantics over time
- [ ] Define deployment flow for the modified NVMe kernel module
- [ ] Define Linux version compatibility policy for the kernel module
- [ ] Define startup ordering for kernel module, service, and runtime attach flow
- [ ] Add AI-consumable architecture docs under `doc/`

## NVMeService Rewrite — Remaining Work

Control-plane code, state, server, client, and `libnvm` shared-resource
reconstruction are in place. Outstanding items to make it buildable and
runnable end-to-end:

- [ ] `backends/local/NVMeService/examples/nvmeservice_daemon.cpp` — daemon
      entry point (parse `sys_config.yaml`, construct `ServiceState`,
      start gRPC server, start reaper, wait for SIGINT)
- [ ] `backends/local/NVMeService/examples/nvmeservice_client.cpp` — smoke
      test that connects, lists devices, allocates, sleeps, releases
- [ ] `backends/local/NVMeService/examples/sys_config.yaml` — example config
      matching the new YAML schema (grpc / gpus / nvmes / queue_pool / lease)
- [ ] `backends/local/NVMeService/examples/CMakeLists.txt` — build the two
      example executables against the new `nvmeservice` library
- [ ] Update root `CMakeLists.txt` NVMeService section: compile new file
      layout (`nvmeservice_config.cpp`, `nvmeservice_state.cu`,
      `nvmeservice_server.cpp`, `nvmeservice_client.cpp`) and the new
      `backends/local/nvme/libnvm/src/shared_ctrl.cu`; add
      `add_subdirectory(backends/local/NVMeService/examples)`
- [ ] Add `BlockDeviceManager` second constructor that takes a
      `std::shared_ptr<Controller>` plus mount path (skips own controller
      init so it can consume a shared Controller from NVMeService)
- [ ] Verify CUDA IPC works for PRP memory; if not, wire client-side PRP
      allocation fallback (the server already honours `has_prp=false`)
- [ ] Verify `cudaHostRegister(BAR0, cudaHostRegisterIoMemory)` works in a
      non-daemon process (same SNVMe device was already mmapped by daemon)
- [ ] End-to-end smoke: daemon + one client, run a trivial read/write via
      `BlockDeviceManager` built on the shared `Controller`

## Discussion Required Before Major Refactor

- [ ] Decide the future runtime/product name that may replace `GeminiFS`
- [ ] Decide the final naming style for runtime-facing APIs
- [ ] Decide the initial target directory migration plan
- [ ] Decide what must remain temporarily compatible during the first refactor wave

## Known Bugs To Track

- [ ] `GPU file persistence` correctness/stability issue

## Current Non-Goals

- [ ] Do not introduce `cooperative submit` in `v0.1`
- [ ] Do not treat simultaneous CPU/GPU access semantics as a stable runtime contract
- [ ] Do not hardcode framework-specific types into the core runtime

## Collaboration Notes

- Update this file when current priorities change
- Keep completed items visible until the next version snapshot is archived
- If a new active version is created, move the old roadmap snapshot into [`doc/history/`](doc/history/)
