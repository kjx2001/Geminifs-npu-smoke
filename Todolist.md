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
