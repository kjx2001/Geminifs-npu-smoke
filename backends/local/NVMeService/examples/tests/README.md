# NVMeService module tests

A small set of bash scripts that exercise NVMeService one module at a
time. Each script focuses on one concern; pick the ones whose
prerequisites you can satisfy.

## Layout

```
tests/
├── lib.sh              # shared helpers + config templates
├── configs/            # YAML configs that should be rejected by validate_config
├── 01_config.sh        # config parser + validate_config (no hardware needed)
├── 02_init.sh          # ServiceState init: mount + per-GPU symlinks
├── 03_grpc_smoke.sh    # ListDevices + AllocateQueues + Release on single-GPU pool
├── 04_multi_gpu.sh     # multi-GPU split routing + cross-GPU rejection
└── 05_reaper.sh        # lease reclaim of dead-client allocations
```

## Build prerequisites

```bash
cd <repo>/build
make -j nvmeservice nvmeservice_daemon_example nvmeservice_client_example
```

The scripts default to `$REPO/build/bin` for the binaries; override via
`BIN_DIR=...`.

## Runtime prerequisites by script

| Script           | SNVMe mod | Real NVMe | Multi-GPU host | gRPC server uncommented |
| ---------------- | :-------: | :-------: | :------------: | :---------------------: |
| 01_config.sh     |     -     |     -     |       -        |            -            |
| 02_init.sh       |    yes    |    yes    |     yes¹       |          yes²           |
| 03_grpc_smoke.sh |    yes    |    yes    |       -        |           yes           |
| 04_multi_gpu.sh  |    yes    |    yes    |     yes¹       |           yes           |
| 05_reaper.sh     |    yes    |    yes    |     yes¹       |           yes           |

¹ The split tests configure `gpus: [0, 1]`. If the host has only one
visible CUDA device, set `GPU1_MOUNT=$GPU0_MOUNT` and adjust the script's
config to reuse `gpu_id: 0` twice (validate_config will reject duplicate
`gpu_id` within one nvme; you'd need to relax that or use a different
config).

² 02_init.sh works against the current daemon binary even if the gRPC
server block in `examples/nvmeservice_daemon.cpp` is commented out --
the script only needs ServiceState to construct successfully and exit.
The other scripts call gRPC and need the server active.

## Environment knobs

These can be set in the shell before running any script. Defaults are
defined at the top of `lib.sh`.

| Var          | Default              | Meaning                                       |
| ------------ | -------------------- | --------------------------------------------- |
| `BIN_DIR`    | `<repo>/build/bin`   | Where `nvmeservice_daemon` / `_client` live   |
| `PCI_ADDR`   | `0000:50:00.0`       | NVMe PCI address used in generated configs    |
| `ENDPOINT`   | `127.0.0.1:50051`    | gRPC endpoint                                 |
| `GPU0_MOUNT` | `/mnt/gpu0`          | `gpus[0].mount_path`                          |
| `GPU1_MOUNT` | `/mnt/gpu1`          | `gpus[1].mount_path` (multi-GPU tests only)   |
| `NVME_MOUNT` | `/mnt/nvme0`         | `nvmes[0].mount_path`                         |

Mount points must be writable by whoever runs the script (typically
root, since libnvm's `Host_file_system_int` runs `mount`).

## Running

```bash
# Adjust to your hardware before running anything that touches it.
export PCI_ADDR=0000:50:00.0

cd <repo>/backends/local/NVMeService/examples/tests

# 1) Cheap, no hardware:
bash 01_config.sh

# 2) Hardware-only checks (mount + symlinks; no gRPC needed):
sudo bash 02_init.sh

# 3..5) Require gRPC server uncommented in nvmeservice_daemon.cpp:
sudo bash 03_grpc_smoke.sh
sudo bash 04_multi_gpu.sh
sudo bash 05_reaper.sh
```

Each script prints `[ OK ]` for each invariant it asserts and `[FAIL]`
the moment one fails (with a short stack of the relevant output).
Exit code is non-zero on any failure. Daemons spawned by a script are
killed on script exit via a trap.

## When something fails

- **02 says `daemon exited with rc=N during init`** -- the daemon log is
  printed inline. Most common: missing `/dev/snvm_*` (kernel module not
  loaded), wrong `PCI_ADDR`, no permission to mount.

- **03/04/05 say "daemon did not become ready on ... within 15s"** --
  the gRPC server block in `examples/nvmeservice_daemon.cpp` is still
  commented out. Uncomment the
  `start_reaper` / `BuildAndStart` / `server->Wait()` block, rebuild,
  retry.

- **04 cross-GPU rejection failure** with message about no group --
  good, that's the correctness-positive case. The script greps for
  `no queue group on device_id=0 for cuda_device=2` exactly.

- **05 "GPU 1 not reclaimed after reap"** -- check
  `lease.heartbeat_interval_sec` / `timeout_sec` in the generated
  config and bump `WAIT_FOR_REAP_SEC` if your reaper tick is slow.
