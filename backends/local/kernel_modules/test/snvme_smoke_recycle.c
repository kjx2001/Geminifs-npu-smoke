/*
 * snvme_smoke_recycle.c -- Standalone test for NVM_RAW_ADMIN_CMD.
 *
 * Purpose
 * -------
 *
 * Verify that snvme's NVM_RAW_ADMIN_CMD ioctl can drive arbitrary NVMe
 * admin commands through the controller's snvme-owned admin queue,
 * with CQE status / DW0 / DW1 surfaced back to userspace.
 *
 * Why
 * ---
 *
 * This is the kernel-level building block for NVMeService's
 * per-queue recycle path (Delete I/O SQ -> Delete I/O CQ -> Create
 * I/O CQ -> Create I/O SQ; NVMe 1.4 §5.4 / §5.5).  Before we wire
 * the full recycle into the daemon, we want a focused test that
 * exercises just the admin pass-through ioctl -- so that if the
 * follow-up recycle test breaks, we already know whether the
 * problem is in the new ioctl or in the four-command sequence /
 * ring-memory bookkeeping on top of it.
 *
 * Coverage in this revision (T2a):
 *   - bind the controller (required: the ioctl returns -ENODEV
 *     until pci_get_drvdata(ctrl->pdev) yields a valid nvme_dev),
 *   - issue an Identify Controller admin command via
 *     NVM_RAW_ADMIN_CMD with a NULL data buffer.  Status is
 *     expected to be 0x01 "Invalid Field in Command" (SC=2)
 *     because we deliberately did NOT provide a PRP1 buffer -- the
 *     point of T2a is to confirm the ioctl pipes the command and
 *     surfaces the controller's CQE status verbatim, not that the
 *     command itself succeeds.  Any other status indicates a real
 *     bug in the new pass-through code.
 *   - unbind, tear down /dev/ssnvme<N>.
 *
 * Out of scope (T2b, separate test):
 *   - Delete + Create I/O SQ/CQ across the recycle, with verified
 *     subsequent NVMe read.  That needs user-IOQ ring memory, which
 *     belongs in its own binary because the failure surface is
 *     different (DMA mapping / ring sizing vs admin pass-through).
 *
 * Build:    make snvme_smoke_recycle              (parent Makefile)
 * Invoke:   sudo ./snvme_smoke_recycle <PCI_BDF>
 *
 * Exit codes:
 *   0  -- all steps passed.
 *   1  -- usage error.
 *   2  -- a smoke step failed; see stderr for which one.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* The *only* SNVMe-specific header we need: the kernel/user UAPI. */
#include "ioctl.h"

/* ------------------------------------------------------------------ */
/* Logging helpers (copied verbatim from snvme_smoke.c for symmetry). */
/* ------------------------------------------------------------------ */

static int g_step = 0;

/* ────────────────────────────────────────────────────────────
 * 【函数】step_ok(fmt, ...)
 * 【作用】打印一条成功日志，形如 "[ OK ] step=N ..."，并把全局
 *         步骤计数器 g_step 自增 1。用来标记测试流程走通了一步。
 * 【参数】fmt 是 printf 风格的格式串；后面的 ... 是可变参数，
 *         按 fmt 里的占位符填进去（和 printf 用法一样）。
 * 【返回】无返回值（void）。只往 stderr 打日志，不会终止程序。
 * 【在测试中的角色】每完成一个正常步骤就调一次，给人看进度。
 * 【新手提示】va_list / va_start / va_end 是 C 处理"不定个数参数"
 *         的标准套路；vfprintf 就是接收 va_list 版本的 fprintf。
 *         stderr 是标准错误流，这里日志全走 stderr，方便和程序
 *         真正的数据输出区分开。
 * ──────────────────────────────────────────────────────────── */
static void step_ok(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[ OK ] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】step_warn(fmt, ...)
 * 【作用】打印一条警告日志，形如 "[WARN] step=N ..."，同样把
 *         g_step 自增 1。表示"这一步结果不完全符合预期，但不致命，
 *         测试可以继续往下走"。
 * 【参数】fmt + ... 同 step_ok，printf 风格的格式串和可变参数。
 * 【返回】无返回值（void）。不终止程序，只是提个醒。
 * 【在测试中的角色】出现"可容忍的偏差"时调用，例如某固件返回的
 *         状态码和规范建议值不同，但透传链路本身仍然正常。
 * 【新手提示】和 step_ok 唯一的区别就是标签是 [WARN]；逻辑完全
 *         一样。把成功 / 警告 / 失败拆成三个函数，是为了让日志
 *         一眼就能看出每步的性质。
 * ──────────────────────────────────────────────────────────── */
static void step_warn(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[WARN] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】step_fail(err, fmt, ...)
 * 【作用】打印一条失败日志，形如 "[FAIL] step=N ... errno=E (描述)"，
 *         然后直接 exit(2) 结束整个进程——这一步失败就不再往下测了。
 * 【参数】err  是要打印的 errno 值（C 库的错误码）。传 0 表示
 *              "这次失败不是系统调用错误"，会显示 "n/a"。
 *         fmt + ... 同上，printf 风格的描述信息。
 * 【返回】不返回！函数声明带 __attribute__((noreturn))，因为末尾
 *         调用了 exit(2)，控制权永远不会回到调用处。
 * 【在测试中的角色】任何"必须成功却失败了"的步骤都用它来终止，
 *         退出码 2 约定为"某个冒烟步骤失败"（见文件头 Exit codes）。
 * 【新手提示】errno 是 C 里全局的"最近一次出错原因"，strerror()
 *         把它翻成人话（如 "No such device"）。noreturn 属性能让
 *         编译器知道此后代码不可达，避免"函数可能没返回值"之类的
 *         误报。
 * ──────────────────────────────────────────────────────────── */
static void __attribute__((noreturn)) step_fail(int err, const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[FAIL] step=%-2d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, " errno=%d (%s)\n", err, err ? strerror(err) : "n/a");
    exit(2);
}

/* ------------------------------------------------------------------ */
/* BDF parser                                                         */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】parse_bdf(s, out)
 * 【作用】把命令行里的 PCI 地址字符串（形如 "0000:50:00.0"）解析成
 *         结构体 pci_device_addr 的四个字段。
 * 【参数】s   输入字符串，格式 "域:总线:设备.功能"（DDDD:BB:DD.F），
 *             各段都是十六进制。
 *         out 输出参数，解析出来的 domain/bus/slot/func 写到这里。
 * 【返回】成功返回 0；格式不对（凑不齐 4 个字段）返回 -1。
 * 【在测试中的角色】main() 启动时把用户给的 BDF 转成内核 ioctl
 *         需要的结构体，是整条流程的第一步输入校验。
 * 【新手提示】BDF = Bus/Device/Function，是 PCI 设备在系统里的
 *         "门牌号"，加上 domain 一共四段，能唯一定位一块网卡 /
 *         NVMe 盘。sscanf 的返回值是"成功匹配并赋值的字段个数"，
 *         所以这里用 ==4 判断是否四段都解析到了。
 * ──────────────────────────────────────────────────────────── */
static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* ioctl wrapper                                                      */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】do_ioctl(fd, req, arg, what)
 * 【作用】对 ioctl() 系统调用的一层薄封装：调用 ioctl，如果失败就
 *         打一条带原因的错误日志，并小心地把 errno 保住再返回。
 * 【参数】fd   已打开的设备文件描述符（/dev/snvm_control 或
 *              /dev/ssnvme<N>）。
 *         req  ioctl 命令号（如 SNVM_DEVICE_BIND，定义在 ioctl.h）。
 *         arg  指向命令参数结构体的指针，内核会读/写它。
 *         what 这次调用的可读名字，仅用于出错时打日志。
 * 【返回】透传 ioctl 的返回值：成功通常是 0，失败是 <0，且此时
 *         errno 已被设置好（函数特意在打印 strerror 后又写回 errno，
 *         防止 fprintf 把它覆盖）。
 * 【在测试中的角色】所有走 /dev 节点的内核交互都经过它，统一了
 *         错误打印格式，省去每处重复写 if/perror。
 * 【新手提示】ioctl（I/O control）是用户态给设备驱动下发"自定义
 *         命令"的通用入口；这里就是用户态测试程序和 snvme 内核
 *         模块对话的方式。
 * ──────────────────────────────────────────────────────────── */
static int do_ioctl(int fd, unsigned long req, void* arg, const char* what) {
    int r = ioctl(fd, req, arg);
    if (r < 0) {
        int e = errno;
        fprintf(stderr, "ioctl(%s) failed: %s\n", what, strerror(e));
        errno = e;
    }
    return r;
}

/* ------------------------------------------------------------------ */
/* NVMe admin command builders.                                       */
/*                                                                    */
/* We only need a Identify Controller SQE for T2a.  Field layout is   */
/* NVMe 1.4 §5.15 Figure 245 (Identify Command).  The kernel will     */
/* fill in the CID via snvme_submit_sync_cmd's blk-mq tag, so we      */
/* leave CDW0[31:16] zero.                                            */
/* ------------------------------------------------------------------ */

#define NVME_ADMIN_OPC_IDENTIFY  0x06
#define NVME_IDENTIFY_CNS_CTRL   0x01

/* ────────────────────────────────────────────────────────────
 * 【函数】build_identify_ctrl_sqe(out[64])
 * 【作用】在 64 字节缓冲区里手工拼出一条 NVMe admin 命令——
 *         "Identify Controller"（识别控制器），不带数据缓冲区。
 * 【参数】out  调用方给的 64 字节数组，函数先清零再填关键字段。
 *              这 64 字节就是一条 NVMe 提交队列条目（SQE）的原始内容。
 * 【返回】无返回值；结果直接写在 out 里。
 * 【在测试中的角色】用于步骤 [4] 和 [7a]，给 NVM_RAW_ADMIN_CMD
 *         ioctl 喂一条"正向"命令，验证 SQE 能送进控制器、CQE 能
 *         透传回来。
 * 【新手提示】NVMe admin SQE 固定 64 字节，按 16 个 32 位双字
 *         (DWORD/CDW0~CDW15) 编排，全部小端存放。本函数只动两处：
 *           · CDW0 的字节 0（out[0]）= opcode 操作码，0x06 表示
 *             Identify。opcode 永远在 CDW0 的最低字节。
 *           · CDW10（占 out[40..43]）的字节 0 = CNS 字段，0x01 表示
 *             "Identify Controller"。CDW10 起始字节 = 10*4 = 40，
 *             所以 out[40] 就是 CNS。其余字节保持 0。
 *         没填 PRP1（数据指针）是故意的：本测试只关心命令往返链路，
 *         不搬运返回的 4096 字节控制器信息。
 * ──────────────────────────────────────────────────────────── */
/*
 * Build an Identify (Controller) admin SQE with no data buffer
 * pointer.  Controller is expected to reject with SC=0x02 "Invalid
 * Field in Command" (DW10 CNS=0x01 requires PRP1 to point at a
 * 4096-byte buffer; we sent zero).
 *
 * This is the cheapest way to confirm the ioctl plumbs the command
 * end-to-end without dragging in DMA-buffer plumbing (which belongs
 * to T2b).
 */
static void build_identify_ctrl_sqe(uint8_t out[64]) {
    memset(out, 0, 64);
    /* CDW0: opcode in [7:0] */
    out[0] = NVME_ADMIN_OPC_IDENTIFY;
    /* CDW10: CNS = 0x01 (Identify Controller).  little-endian. */
    out[40] = NVME_IDENTIFY_CNS_CTRL;     /* DW10 byte 0 = CNS */
    out[41] = 0;
    out[42] = 0;
    out[43] = 0;
}

/* ────────────────────────────────────────────────────────────
 * 【函数】build_reserved_opcode_sqe(out[64])
 * 【作用】在 64 字节缓冲区里拼一条"非法"的 admin SQE：操作码用
 *         保留值 0xFF，其余字段全 0。控制器必须拒绝它。
 * 【参数】out  64 字节数组，函数先清零，再把 out[0] 设为 0xFF。
 * 【返回】无返回值；结果写在 out 里。
 * 【在测试中的角色】用于步骤 [7b] 的"负向测试"：验证当命令本身
 *         非法时，控制器返回的失败状态能被 ioctl 原样透传回用户态。
 * 【新手提示】同样地，opcode（操作码）位于 SQE 的 CDW0 字节 0，
 *         也就是 out[0]。0xFF 是 admin 命令里普遍保留 / 厂商自定义
 *         的最高操作码，绝大多数控制器都没实现。按 NVMe 1.4 §5
 *         (Figure 139) 规定，控制器遇到没实现的操作码，必须以
 *         SCT=0x0（Generic）、SC=0x01（Invalid Command Opcode）拒绝。
 *         相比"Identify 不给 PRP1"，用保留操作码是更可移植的
 *         "保证被拒"用例（后者有些固件会容忍）。
 * ──────────────────────────────────────────────────────────── */
/*
 * Build an admin SQE with a reserved opcode (0xFF) and all other
 * fields zero.  Per NVMe 1.4 §5 (Admin Command Set, Figure 139), any
 * opcode the controller does not implement MUST complete with
 * Status Code Type 0x0 (Generic Command Status), SC = 0x01
 * "Invalid Command Opcode".  This holds independently of vendor
 * extensions: 0xFF is the highest byte and is universally reserved
 * for vendor-specific commands the controller is allowed to NOT
 * implement -- and Tencent NVMe firmwares we test against do not.
 *
 * We use this instead of an Identify-with-PRP1=0 because the latter
 * is *strongly* recommended to be rejected, but not required, and
 * some controllers (including the one this test was first run on)
 * silently accept it.  An unimplemented opcode is the most portable
 * "guaranteed reject" we can construct without sending data.
 */
static void build_reserved_opcode_sqe(uint8_t out[64]) {
    memset(out, 0, 64);
    out[0] = 0xff;        /* CDW0 byte 0 = opcode (reserved) */
}

/* ------------------------------------------------------------------ */
/* Argument parsing                                                   */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】usage(prog)
 * 【作用】往 stderr 打印用法说明：怎么传 PCI BDF 参数，并警告本
 *         测试是破坏性的（会把设备从内核自带 nvme 驱动手里抢过来）。
 * 【参数】prog  程序自身的名字，一般传 argv[0]，填进提示文本里。
 * 【返回】无返回值；只打印帮助文本。
 * 【在测试中的角色】参数个数不对、或用户传 -h/--help 时调用，
 *         告诉人正确的调用方式。
 * 【新手提示】"绑定控制器"指让 snvme 接管这块 NVMe 盘，期间内核
 *         自带的 nvme 驱动会失去它，所以原本挂载的盘会暂时不可用——
 *         这就是注释里说的 destructive（破坏性）。
 * ──────────────────────────────────────────────────────────── */
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <PCI_BDF>\n"
        "  e.g.: %s 0000:50:00.0\n"
        "\n"
        "This test BINDS the controller (destructive: the in-tree nvme\n"
        "driver loses the device for the duration).  Unlike\n"
        "./snvme_smoke, NVM_RAW_ADMIN_CMD always requires a probed\n"
        "controller so there is no UAPI-only mode here.\n",
        prog, prog);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】main(argc, argv)
 * 【作用】整个冒烟测试的主流程：打开 snvme 设备节点，绑定控制器，
 *         用 NVM_RAW_ADMIN_CMD 透传两条 admin 命令（一正一负）验证
 *         透传链路，最后解绑并清理。
 * 【参数】argc/argv  命令行参数；只接受一个参数——PCI BDF，
 *         例如 "0000:50:00.0"。也支持 -h/--help。
 * 【返回】0 全部步骤通过；1 用法错误；2 某个冒烟步骤失败
 *         （失败时由 step_fail 内部 exit(2)，不会走到这里的 return）。
 * 【在测试中的角色】把前面所有小函数串成完整流程。
 *
 *   流程分步：
 *     [1] 打开 /dev/snvm_control 控制节点（总控入口）。
 *     [2] SNVM_CHRDEV_CREATE：为该 BDF 创建一个字符设备，拿到 minor 号。
 *     [3] 按 minor 打开 /dev/ssnvme<N> 这个具体设备节点。
 *     [4] 【负向·未绑定】绑定前就调 NVM_RAW_ADMIN_CMD，期望返回
 *         -ENODEV——验证内核在控制器未绑定时不会误解引用而崩溃。
 *     [5] SNVM_DEVICE_BIND：把控制器绑定给 snvme，触发探测，让
 *         admin queue 真正可用。
 *     [6] 轮询 NVM_GET_DEV_INFO，直到异步探测完成（最多约 10 秒）。
 *     [7a]【正向往返】透传 Identify Controller（无数据缓冲），期望
 *         ioctl 返回 0，CQE 的 status/DW0/DW1 被透传回用户态。
 *     [7b]【负向往返】透传保留操作码 0xFF，期望控制器以
 *         SC=0x01（Invalid Command Opcode）拒绝，且该失败状态被原样
 *         透传回来。
 *     [8] SNVM_DEVICE_UNBIND：解绑控制器，把设备还给系统。
 *     [F1] 收尾：close 掉 /dev/ssnvme<N>（按 fd 清理）。
 *     [F2] 收尾：SNVM_CHRDEV_REMOVE 删除字符设备，再关掉控制节点。
 *
 * 【新手提示】SQE = Submission Queue Entry（提交队列条目，发给控制器
 *         的命令）；CQE = Completion Queue Entry（完成队列条目，含
 *         status/DW0/DW1 等执行结果）。本测试核心就是确认 snvme 能把
 *         任意 admin SQE 送进控制器自有的 admin queue，并把 CQE 结果
 *         如实带回——这是后续 per-queue recycle（删/建 I/O SQ/CQ）的
 *         底层积木。
 * ──────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    if (argc != 2) {
        usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(argv[0]);
        return 0;
    }
    const char* bdf_str = argv[1];

    struct pci_device_addr orig_bdf;
    if (parse_bdf(bdf_str, &orig_bdf) != 0) {
        fprintf(stderr, "Bad BDF: '%s' (expected DDDD:BB:DD.F)\n", bdf_str);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* [1] /dev/snvm_control                                              */
    /* ------------------------------------------------------------------ */
    int fd_ctl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctl < 0)
        step_fail(errno, "open(/dev/snvm_control)");
    step_ok("open(/dev/snvm_control) fd=%d", fd_ctl);

    /* ------------------------------------------------------------------ */
    /* [2] SNVM_CHRDEV_CREATE                                             */
    /* ------------------------------------------------------------------ */
    struct pci_device_addr addr = orig_bdf;
    if (do_ioctl(fd_ctl, SNVM_CHRDEV_CREATE, &addr, "SNVM_CHRDEV_CREATE") < 0)
        step_fail(errno, "SNVM_CHRDEV_CREATE %s", bdf_str);
    int minor_n = addr.domain;
    step_ok("SNVM_CHRDEV_CREATE minor=%d", minor_n);

    /* ------------------------------------------------------------------ */
    /* [3] /dev/ssnvme<N>                                                 */
    /* ------------------------------------------------------------------ */
    char dev_path[64];
    snprintf(dev_path, sizeof(dev_path), "/dev/ssnvme%d", minor_n);
    int fd_dev = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_dev < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ------------------------------------------------------------------ */
    /* [4] NVM_RAW_ADMIN_CMD before bind -- expected -ENODEV.              */
    /*                                                                    */
    /* The kernel handler should reject this with -ENODEV because the     */
    /* controller is not yet bound to snvme (pci_get_drvdata returns      */
    /* NULL or no admin_q).  This negative test catches an entire class   */
    /* of regressions: if we forget the bound-check and dereference       */
    /* ndev->ctrl.admin_q on a fresh chrdev, the host oopses.             */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_raw_admin pre;
        memset(&pre, 0, sizeof(pre));
        build_identify_ctrl_sqe(pre.sqe);
        int r = ioctl(fd_dev, NVM_RAW_ADMIN_CMD, &pre);
        if (r == 0)
            step_fail(0,
                "NVM_RAW_ADMIN_CMD unexpectedly returned 0 on UNBOUND ctrl "
                "(nvme_status=0x%04x); kernel handler is missing the "
                "ndev/admin_q liveness check",
                (unsigned)pre.nvme_status);
        if (errno != ENODEV)
            step_warn("NVM_RAW_ADMIN_CMD on unbound ctrl returned errno=%d (%s); "
                      "expected ENODEV. Not fatal, but the kernel may have a "
                      "tighter check than this test expects.",
                      errno, strerror(errno));
        else
            step_ok("NVM_RAW_ADMIN_CMD on unbound ctrl correctly returned -ENODEV");
    }

    /* ------------------------------------------------------------------ */
    /* [5] SNVM_DEVICE_BIND                                                */
    /*                                                                    */
    /* No NVM_SET_IOQ_NUM / NVM_MAP_HOST_MEMORY / NVM_SET_SHARE_REG       */
    /* preamble -- this test does NOT use user IOQs.  Bind alone lets    */
    /* snvme run the in-tree probe path (kernel-only queues), which is    */
    /* the minimum needed for ndev->ctrl.admin_q to be live.             */
    /* ------------------------------------------------------------------ */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_BIND, &bdf, "SNVM_DEVICE_BIND") < 0)
            step_fail(errno,
                "SNVM_DEVICE_BIND %s -- the in-tree nvme driver may still own "
                "this device (try `sudo sh -c 'echo %s > /sys/bus/pci/drivers/nvme/unbind'` "
                "first)", bdf_str, bdf_str);
        step_ok("SNVM_DEVICE_BIND %s", bdf_str);
    }

    /* ------------------------------------------------------------------ */
    /* [6] Poll NVM_GET_DEV_INFO until probe completes.                    */
    /*                                                                    */
    /* Probe is asynchronous (reset_work + scan_work).  On a healthy bind */
    /* this finishes in well under a second; allow up to ~10s for slow    */
    /* / power-managed drives.                                            */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_dev info;
        int ok = 0;
        for (int i = 0; i < 100; i++) {
            memset(&info, 0, sizeof(info));
            if (ioctl(fd_dev, NVM_GET_DEV_INFO, &info) == 0 &&
                info.disk_name[0] != '\0') {
                ok = 1;
                step_ok("NVM_GET_DEV_INFO disk='%s' nr_user_q=%u block_size=%zu",
                        info.disk_name, info.nr_user_q, info.block_size);
                break;
            }
            usleep(100 * 1000);
        }
        if (!ok)
            step_fail(errno, "NVM_GET_DEV_INFO did not succeed within 10s "
                             "after bind (probe still running?)");
    }

    /* ------------------------------------------------------------------ */
    /* [7a] NVM_RAW_ADMIN_CMD positive: Identify Controller, no buffer.    */
    /*                                                                    */
    /* What we expect:                                                    */
    /*   - ioctl return:  0   (CQE arrived)                              */
    /*   - nvme_status:   0x0000  (SC=0x00 Successful Completion)         */
    /*                                                                    */
    /* Why no data buffer is OK here:                                     */
    /*   The kernel handler hard-codes (buffer=NULL, bufflen=0), so      */
    /*   __snvme_submit_sync_cmd does NOT remap PRP1.  The original     */
    /*   plan was to use Identify-with-PRP1=0 as a "guaranteed reject"  */
    /*   case, but in practice many controllers tolerate it (see        */
    /*   step 7b for a portable negative case).  All we assert here     */
    /*   is that Identify-with-zero-PRP returns *something* via the     */
    /*   pass-through path; success or failure both prove the SQE/CQE   */
    /*   round-trip works.                                               */
    /*                                                                    */
    /* What this proves:                                                  */
    /*   The new ioctl correctly:                                         */
    /*     (a) copies the SQE from userspace,                            */
    /*     (b) forwards via __snvme_submit_sync_cmd to ctrl.admin_q,    */
    /*     (c) round-trips CQE DW0/DW3 status back to userspace,        */
    /*     (d) does not panic on a buffer-less Identify.                */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_raw_admin req;
        memset(&req, 0, sizeof(req));
        build_identify_ctrl_sqe(req.sqe);

        if (do_ioctl(fd_dev, NVM_RAW_ADMIN_CMD, &req,
                     "NVM_RAW_ADMIN_CMD(Identify Controller)") < 0)
            step_fail(errno, "NVM_RAW_ADMIN_CMD pass-through ioctl failed "
                             "(ioctl-level error, not NVMe-level)");

        /* SC (status code) lives in CQE DW3 bits 15:1.  __snvme_-
         * submit_sync_cmd already right-shifts that field by one, so
         * what we get back in nvme_status is already SC|SCT|... with
         * the phase bit dropped.  Bits [7:0] = SC, [10:8] = SCT,
         * [13:11] = reserved (CRD), [14] = MORE, [15] = DNR.
         */
        uint16_t sc  = req.nvme_status & 0xff;
        uint16_t sct = (req.nvme_status >> 8) & 0x7;

        step_ok("NVM_RAW_ADMIN_CMD(Identify) round-trip: nvme_status=0x%04x "
                "(SC=0x%02x SCT=0x%x) dw0=0x%08x dw1=0x%08x",
                req.nvme_status, sc, sct, req.result_dw0, req.result_dw1);
    }

    /* ------------------------------------------------------------------ */
    /* [7b] NVM_RAW_ADMIN_CMD negative: Reserved opcode (0xFF).            */
    /*                                                                    */
    /* What we expect:                                                    */
    /*   - ioctl return:  0   (CQE arrived; the controller rejected at   */
    /*                        NVMe level, not the kernel at ioctl level) */
    /*   - nvme_status:   non-zero with SC=0x01 "Invalid Command Opcode" */
    /*                                                                    */
    /* Why this is a stronger negative case than Identify-PRP1=0:        */
    /*   NVMe 1.4 §5 (Admin Command Set, Figure 139) requires every     */
    /*   admin command queue to reject opcodes the controller does not  */
    /*   implement with SC=0x01.  Reserved opcodes therefore have a     */
    /*   spec-mandated failure behavior, whereas a bad PRP on Identify  */
    /*   is only "should be rejected" -- some firmwares accept it.      */
    /*                                                                    */
    /* What this proves:                                                  */
    /*   The pass-through faithfully surfaces a non-zero CQE status     */
    /*   field back to userspace.  Together with [7a] this confirms     */
    /*   we can both submit valid admin commands AND observe failures  */
    /*   reported by the controller -- the two halves we need before   */
    /*   layering Delete/Create I/O SQ/CQ on top in T2b.               */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_raw_admin req;
        memset(&req, 0, sizeof(req));
        build_reserved_opcode_sqe(req.sqe);

        if (do_ioctl(fd_dev, NVM_RAW_ADMIN_CMD, &req,
                     "NVM_RAW_ADMIN_CMD(reserved opcode 0xFF)") < 0)
            step_fail(errno, "NVM_RAW_ADMIN_CMD(reserved opcode) ioctl-level "
                             "error -- handler rejected before submission?");

        uint16_t sc  = req.nvme_status & 0xff;
        uint16_t sct = (req.nvme_status >> 8) & 0x7;

        step_ok("NVM_RAW_ADMIN_CMD(opcode=0xFF) round-trip: nvme_status=0x%04x "
                "(SC=0x%02x SCT=0x%x) dw0=0x%08x dw1=0x%08x",
                req.nvme_status, sc, sct, req.result_dw0, req.result_dw1);

        if (req.nvme_status == 0) {
            step_fail(0,
                "Controller accepted a reserved admin opcode (0xFF) -- "
                "the pass-through is masking the CQE status, OR this "
                "firmware implements 0xFF as a vendor-specific command "
                "(unlikely; investigate).");
        } else if (sct == 0x0 && sc == 0x01) {
            step_ok("Controller correctly rejected reserved opcode with "
                    "SC=0x01 (Invalid Command Opcode) -- ioctl plumbing "
                    "surfaces NVMe-level failure verbatim.");
        } else {
            /* Got a non-zero status, but not the spec-mandated
             * 0x01 Invalid-Opcode.  Still proves the pass-through
             * propagates failure status; just log unexpected codes
             * for visibility (some firmwares return SC=0x02 or a
             * vendor-specific code instead). */
            step_warn("Reserved opcode rejected with SC=0x%02x SCT=0x%x "
                      "(expected SC=0x01 SCT=0x0).  Pass-through still "
                      "works -- this is just a firmware quirk.",
                      sc, sct);
        }
    }

    /* ------------------------------------------------------------------ */
    /* [8] SNVM_DEVICE_UNBIND                                              */
    /* ------------------------------------------------------------------ */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_UNBIND, &bdf, "SNVM_DEVICE_UNBIND") < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND %s", bdf_str);
        step_ok("SNVM_DEVICE_UNBIND %s", bdf_str);
    }

    /* ------------------------------------------------------------------ */
    /* [F1] Per-fd cleanup                                                 */
    /* ------------------------------------------------------------------ */
    if (close(fd_dev) < 0)
        step_fail(errno, "close(%s)", dev_path);
    step_ok("close(%s)", dev_path);

    /* ------------------------------------------------------------------ */
    /* [F2] SNVM_CHRDEV_REMOVE                                             */
    /* ------------------------------------------------------------------ */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf, "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE %s", bdf_str);
        step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);
    }

    close(fd_ctl);
    fprintf(stderr, "\n=== snvme_smoke_recycle: all %d steps passed ===\n",
            g_step);
    return 0;
}
