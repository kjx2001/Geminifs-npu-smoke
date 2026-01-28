# NVMeService

Userspace daemon + IPC protocol for managing NVMe controllers.

## Roles
- **Admin**: initializes controllers (mount path, PCI addr, queue config).
- **Filesystem process**: queries controller info.

## IPC
- Unix domain socket: `/tmp/nvmeservice.sock`
- Protocol defined in `include/nvmeservice_protocol.h`

## Build
```
cmake -S . -B build
cmake --build build -j
```

## Run
### 1) Start daemon
```
./build/nvmeservice_daemon
```

### 2) Admin init (parse sys_config.ini and create NVMe controllers)
```
./build/nvmeservice_admin_init /tmp/nvmeservice.sock /home/qs/CompanionFS/Geminifs/sys_config.ini 1
```

### 3) Filesystem process query
```
./build/nvmeservice_fs_info /tmp/nvmeservice.sock
```

## Notes
- This module intentionally does **not** depend on libgeminifs.
- NVMeController uses libnvm (`ctrl.h`) and creates queues + mounts filesystem.
