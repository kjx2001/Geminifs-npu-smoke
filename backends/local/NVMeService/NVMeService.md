# NVMeService

## 概述
NVMeService 是一个本地守护进程，用于在每块 GPU 上只初始化一次 NVMe 控制器，并通过 IPC 接口为 GeminiFS 提供控制器信息（后续扩展队列资源分配）。

本文档记录代码结构、构建目标和基础使用方法。

## 代码结构
- src/nvmeservice.proto
  - gRPC/Protobuf 接口定义。
- src/nvmeservice_config.{h,cpp}
  - sys_config.yaml 解析（grpc/gpus/nvmes）。
- src/nvmeservice_state.{h,cu}
  - 服务状态、控制器创建、队列池跟踪。
- src/nvmeservice_server.{h,cpp}
  - gRPC 服务端与请求分发。
- src/nvmeservice_client.{h,cpp}
  - gRPC 客户端 API（信息查询）。
- examples/nvmeservice_daemon.cpp
  - 守护进程示例入口（读取 sys_config.yaml 后直接初始化并提供服务）。

## 构建目标（根目录 CMake）
- nvmeservice（库）

## 使用方法
1. 启动守护进程（直接读取 sys_config.yaml 并初始化）
  - nvmeservice_daemon_example --config sys_config.yaml
2. 客户端调用（只需 gRPC 端口）
  - nvmeservice_client_example --endpoint 127.0.0.1:50051 --cmd ping
  - nvmeservice_client_example --endpoint 127.0.0.1:50051 --cmd info
  - nvmeservice_client_example --endpoint 127.0.0.1:50051 --cmd alloc --controller 0 --count 32 --pid $$
  - nvmeservice_client_example --endpoint 127.0.0.1:50051 --cmd heartbeat --client <client_id> --lease <lease_id> --duration 5000 --interval 1000
  - nvmeservice_client_example --endpoint 127.0.0.1:50051 --cmd release --lease <lease_id>
  - nvmeservice_client_example --endpoint 127.0.0.1:50051 --cmd shutdown

## Docker 说明
- 建议监听本机回环地址：127.0.0.1:50051
- 示例：docker run ... -p 50051:50051

## 说明
- nvmes 使用 gpu_id 绑定到 gpus 列表中的 GPU。
- 守护进程启动即完成初始化，不再区分 AdminInit 阶段。
- 队列分配与资源共享将在下一阶段实现。
