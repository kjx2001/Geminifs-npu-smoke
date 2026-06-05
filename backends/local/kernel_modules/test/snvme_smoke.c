/*
 * snvme_smoke.c -- Standalone end-to-end sanity test for the SNVMe kernel module.
 *
 * This program is intentionally self-contained:
 *   - no CUDA, no libnvm, no Geminifs filesystem dependencies,
 *   - it links against nothing but libc,
 *   - it only includes the SNVMe UAPI header
 *     (backends/local/nvme/libnvm/include/ioctl.h).
 *
 * Two test modes are supported:
 *
 *   default ("UAPI smoke")   exercise every kernel↔user entry point that
 *                            does NOT trigger an NVMe controller probe.
 *                            Safe to run on any host with snvme loaded:
 *                            it does not touch the NVMe data path.
 *
 *   --bind   ("full bring-up") additionally runs SNVM_DEVICE_BIND, waits
 *                            for /dev/snvme<X>n<Y> (note: leading 's' --
 *                            SNVMe block devices are namespaced away from
 *                            the in-tree nvme driver), does a 512 B pread()
 *                            and tears the controller down. DESTRUCTIVE --
 *                            the in-tree nvme driver loses the device for
 *                            the duration of the test.
 *
 * UAPI-smoke steps (always run):
 *   [ 1] open /dev/snvm_control                                      (UAPI: chrdev factory exists)
 *   [ 2] SNVM_CHRDEV_CREATE(BDF)                                     (UAPI: chrdev create + minor returned)
 *   [ 3] open /dev/ssnvme<minor>                                     (UAPI: per-controller chrdev usable)
 *   [ 4] mmap(BAR0)                                                  (UAPI: BAR0 mapped, register reachable)
 *   [ 5] read NVMe CAP from BAR0                                     (sanity: register decoder agrees with spec)
 *   [ 6] NVM_SET_IOQ_NUM(2)                                          (state: ioq_num set)
 *   [ 7] mmap host pages, NVM_MAP_HOST_MEMORY for SQ ring (ioq_idx=1, is_cq=0)
 *   [ 8] NVM_MAP_HOST_MEMORY for CQ ring (ioq_idx=1, is_cq=1)        (state: ioq_map_num == ioq_num)
 *   [ 9] NVM_SET_SHARE_REG(1)                                        (state: use_sreg = 1)
 *   [F1] NVM_UNMAP_HOST_MEMORY x 2 + NVM_CLEAR_IOQ_NUM               (state machine resets cleanly)
 *   [F2] munmap(BAR0) + close(/dev/ssnvme<N>)                        (chrdev release path)
 *   [F3] SNVM_CHRDEV_REMOVE(BDF)                                     (release minor)
 *
 * Additional steps when --bind is given (run BEFORE the cleanup tail):
 *   [B1] SNVM_DEVICE_BIND(BDF)                                       (kernel probe runs, /dev/snvme<X>n<Y> appears)
 *   [B2] NVM_GET_DEV_INFO                                            (returns disk_name, block_size...)
 *   [B3] open /dev/<disk_name>, pread 512 bytes                      (block device works)
 *   [B4] SNVM_DEVICE_UNBIND(BDF)
 *
 * Each step prints "[ OK ] step=..." on success or "[FAIL] step=... errno=N (...)"
 * and exits non-zero immediately on the first failure.
 *
 * Build:        make           (in this directory)
 * Invoke:       sudo ./snvme_smoke <PCI_BDF>
 *               sudo ./snvme_smoke --bind <PCI_BDF>
 *
 * Pre-conditions:
 *   - snvme-core.ko + snvme.ko loaded (insmod or via Makefile).
 *   - /dev/snvm_control exists (mode 0666).
 *   - The PCI device at <PCI_BDF> is a real NVMe SSD whose data you can lose
 *     when running with --bind. UAPI-smoke is safe even if the device is
 *     currently bound to the in-tree nvme driver.
 *
 * Exit codes:
 *   0  -- all steps passed; SNVMe is healthy on this kernel.
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
/* Logging helpers                                                    */
/* ------------------------------------------------------------------ */

static int g_step = 0;

/* ────────────────────────────────────────────────────────────
 * 【函数】step_ok(const char* fmt, ...)
 * 【作用】打印一条"这一步成功了"的日志，形如 "[ OK ] step=3 ..."。
 *         它是个可变参数函数（像 printf 一样可以带格式串和参数）。
 * 【参数】fmt 是 printf 风格的格式串，后面的 ... 是要填进去的值。
 * 【返回】无返回值；只往 stderr（标准错误）打印一行。
 * 【在测试中的角色】整个 smoke 测试由很多"步骤"组成，每做成一步就
 *         调一次它，把全局计数器 g_step 加一并打印出来，让人能看到
 *         测试走到了第几步、每一步在干什么。
 * 【新手提示】va_list / va_start / vfprintf 是 C 处理"不定个数参数"的
 *         标准套路：va_start 定位到第一个可变参数，vfprintf 把这些参数
 *         按 fmt 格式写到 stderr，va_end 收尾。stderr 是程序的诊断输出
 *         通道，和正常结果用的 stdout 分开，所以日志不会污染数据输出。
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
 * 【函数】step_warn(const char* fmt, ...)
 * 【作用】打印一条"警告"日志，形如 "[WARN] step=5 ..."。和 step_ok
 *         几乎一样，区别只是标记是 [WARN] 而不是 [ OK ]。
 * 【参数】fmt 加可变参数，printf 风格。
 * 【返回】无；只打印一行到 stderr。注意它【不会终止程序】，测试继续往下走。
 * 【在测试中的角色】用于那种"不算成功、也不算致命错误"的情况。最典型
 *         的例子：读 CAP 寄存器读出全 0xFF，说明 NVMe 控制器可能处于
 *         掉电/休眠状态——这不是 SNVMe 的 bug，所以只警告、不失败，
 *         让后续不依赖控制器上电的 UAPI 检查继续跑完。
 * 【新手提示】可变参数机制同 step_ok。把"警告"和"失败"分成两个函数，
 *         是为了让测试既能严格（真错就停），又能容忍环境差异（比如设备
 *         没被任何驱动唤醒过）。
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
 * 【函数】step_fail(int err, const char* fmt, ...)
 * 【作用】打印一条"失败"日志（形如 "[FAIL] step=2 ... errno=13 (...)"），
 *         然后【立刻退出整个程序】，退出码为 2。
 * 【参数】err 是失败时的 errno 值（系统调用失败后记录下来的错误码）；
 *         fmt 加可变参数描述是哪一步、做什么时失败的。
 * 【返回】不返回——函数带 __attribute__((noreturn))，因为内部调用了
 *         exit(2)，永远走不到调用它之后的代码。
 * 【在测试中的角色】这是"遇到真错误就立即停"的快速失败开关。smoke 测试
 *         里几乎每个系统调用/ioctl 失败后都会调它，保证一旦某步出错就
 *         马上停下并报清楚是哪一步、errno 是多少，方便定位问题。
 * 【新手提示】errno 是 C 标准的全局错误号，系统调用返回 -1 时它被设置；
 *         strerror(err) 把数字错误码翻译成人能读的英文描述（如
 *         "Permission denied"）。noreturn 是给编译器的提示，告诉它这个
 *         函数不会返回，从而避免"函数可能没返回值"之类的误报警告。
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
/* Minimal NVMe register access: we only read CAP for sanity.          */
/* ------------------------------------------------------------------ */

#define NVME_REG_CAP    0x0000   /* 64-bit Controller Capabilities */

/* ────────────────────────────────────────────────────────────
 * 【函数】mmio_read64(volatile void* base, size_t off)
 * 【作用】从一段已经 mmap 进来的设备内存里，按偏移读出一个 64 位的值。
 *         本测试只用它来读 NVMe 的 CAP 寄存器。
 * 【参数】base 是 BAR0 被 mmap 到用户空间后的起始地址；off 是要读的
 *         寄存器相对 BAR0 起点的字节偏移（这里 CAP 的偏移是 0x0000）。
 * 【返回】返回该偏移处的 64 位寄存器原始值。
 * 【在测试中的角色】对应流程 [5]：把 BAR0 起点加上偏移得到寄存器地址，
 *         直接解引用读出 CAP，用来验证"BAR0 确实映射到了真实的控制器
 *         寄存器空间"，并顺便解码出队列深度等字段做合理性检查。
 * 【新手提示】MMIO（Memory-Mapped I/O）指设备寄存器被映射成内存地址，
 *         读写这块内存就等于读写硬件寄存器。BAR0 是 PCIe 设备的第 0 号
 *         基址寄存器，NVMe 控制器的核心寄存器（CAP、版本、Admin 队列
 *         门铃等）都在这里。volatile 关键字告诉编译器"这块内存随时可能
 *         被硬件改变，不许优化掉读操作"，对设备寄存器是必须的。
 * ──────────────────────────────────────────────────────────── */
static uint64_t mmio_read64(volatile void* base, size_t off) {
    /*
     * x86 supports unaligned 64-bit MMIO loads; if you port this to
     * an arch that doesn't, switch to two readl()-equivalents.
     */
    volatile uint64_t* p = (volatile uint64_t*)((volatile char*)base + off);
    return *p;
}

/* ------------------------------------------------------------------ */
/* BDF parser                                                         */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】parse_bdf(const char* s, struct pci_device_addr* out)
 * 【作用】把命令行传进来的 PCI 地址字符串（如 "0000:50:00.0"）解析成
 *         结构体里的四个数字字段。
 * 【参数】s 是用户输入的地址字符串；out 是输出结构体，解析成功后它的
 *         domain/bus/slot/func 四个字段会被填好。
 * 【返回】成功返回 0，格式不对返回 -1。
 * 【在测试中的角色】程序一开始就要把用户给的设备地址变成内核 ioctl 能
 *         接受的二进制结构，之后所有"创建字符设备 / 绑定 / 解绑"的
 *         ioctl 都靠这个结构来指明操作的是哪一块 NVMe 卡。
 * 【新手提示】BDF 是 PCI 设备的标准定位法："域:总线:设备.功能"
 *         （Domain:Bus:Device.Function），唯一标识一块插在 PCIe 上的卡。
 *         代码里 slot 对应 Device 号。sscanf 用 "%x" 按十六进制读，正好
 *         匹配 BDF 各段用十六进制书写的惯例；返回值 4 表示四段都读到了。
 * ──────────────────────────────────────────────────────────── */
static int parse_bdf(const char* s, struct pci_device_addr* out) {
    /* Accept the canonical "DDDD:BB:DD.F" form, e.g. "0000:50:00.0". */
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Convenience: ioctl wrapper that turns -1 into errno-with-context.   */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】do_ioctl(int fd, unsigned long req, void* arg, const char* what)
 * 【作用】对 ioctl() 的一层薄封装：照常发出 ioctl，如果失败就打印一条
 *         带名字和错误描述的日志，再把 errno 恢复好交还给调用者。
 * 【参数】fd 是要操作的文件描述符（/dev/snvm_control 或 /dev/ssnvme<N>）；
 *         req 是 ioctl 命令号（如 SNVM_CHRDEV_CREATE）；arg 是指向命令参数
 *         结构体的指针；what 是给人看的命令名字符串，用于日志。
 * 【返回】成功返回 ioctl 的返回值（通常 0），失败返回 -1 并保证 errno
 *         仍是失败时的值（中途打印 strerror 不会把它冲掉）。
 * 【在测试中的角色】本测试和内核交互几乎全靠 ioctl，这个包装让每个调用点
 *         少写一遍"判负、取 errno、打印名字"的样板代码，失败信息也更统一。
 * 【新手提示】ioctl（I/O control）是 Linux 里"对设备文件下达特殊命令"的
 *         通用入口：普通 read/write 之外的设备专有操作都走它，命令号 + 参数
 *         结构体由驱动自己定义。这里特意先把 errno 存进局部变量 e，是因为
 *         fprintf/strerror 等调用可能顺手改动全局 errno，存一份再写回才稳妥。
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
/* Argument parsing                                                   */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】usage(const char* prog)
 * 【作用】把程序的用法说明打印到 stderr，包括两种运行模式的示例命令。
 * 【参数】prog 是程序名（一般传 argv[0]），用来把示例命令里的程序名
 *         替换成实际调用的名字。
 * 【返回】无返回值；只打印。
 * 【在测试中的角色】当用户没给设备地址、给了多余参数、或显式请求 --help
 *         时被调用，告诉用户该怎么正确运行（安全的 UAPI-smoke 还是
 *         破坏性的 --bind 全流程）。
 * 【新手提示】argv[0] 是命令行里程序自己的名字，argc 是参数个数。把用法
 *         打到 stderr 而不是 stdout，是 Unix 命令行工具的惯例，方便和
 *         正常输出区分、也方便脚本重定向。
 * ──────────────────────────────────────────────────────────── */
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--bind] <PCI_BDF>\n"
        "  e.g.: %s 0000:50:00.0           # UAPI-smoke only (safe)\n"
        "        %s --bind 0000:50:00.0    # full bring-up (destructive)\n",
        prog, prog, prog);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】main(int argc, char** argv)
 * 【作用】整个 smoke 测试的主体：按固定顺序把"用户态↔SNVMe 内核模块"
 *         的每一个入口都走一遍，任何一步出错就立刻失败退出。
 * 【参数】argc/argv 是命令行参数。识别两类输入：可选的 --bind 开关，
 *         以及必填的 PCI 设备地址 BDF（如 0000:50:00.0）。--help/-h 打印用法。
 * 【返回】全部步骤通过返回 0；用法错误返回 1；某步失败时由 step_fail 退出码 2。
 *
 * 【在测试中的角色 / 完整执行流程】
 *   先解析参数、用 parse_bdf 把 BDF 变成结构体，然后分三段执行：
 *
 *   ── UAPI-smoke 段（默认总会跑，安全、不碰 NVMe 数据通路）──
 *     [1] open /dev/snvm_control —— 打开"字符设备工厂"控制节点，
 *         证明 SNVMe 核心模块已加载、控制入口存在。
 *     [2] SNVM_CHRDEV_CREATE(BDF) —— 让内核为这块卡创建一个专属字符设备，
 *         内核把分配到的 minor（次设备号）写回结构体的 domain 字段。
 *     [3] open /dev/ssnvme<minor> —— 打开刚创建的 per-controller 字符设备，
 *         证明它真的可用（注意前缀是 ssnvme，刻意和系统自带 nvme 区分开）。
 *     [4] mmap(BAR0) —— 把控制器的 BAR0 寄存器空间映射进用户内存（8 KiB）。
 *     [5] 读 CAP 寄存器 —— 用 mmio_read64 读出 CAP 并解码 mqes/dstrd；
 *         全 0 说明 BAR0 没映射到真东西（真错，失败）；全 0xFF 说明控制器
 *         可能掉电（只警告、继续）。
 *     [6] NVM_SET_IOQ_NUM(2) —— 通过 struct nvm_ioctl_setup 告诉内核要用
 *         1 个 SQ + 1 个 CQ，队列放在主机内存里，设置好内核侧状态。
 *     [7] mmap 一页主机内存并 NVM_MAP_HOST_MEMORY 映射成 0 号队列的 SQ 环。
 *     [8] 同样映射一页作为 0 号队列的 CQ 环（注意队列号是 0 起算的）。
 *     [9] NVM_SET_SHARE_REG(1) —— 打开 use_sreg 门控，完成队列共享状态机配置。
 *
 *   ── --bind 段（只有给了 --bind 才跑，破坏性，会真正接管设备）──
 *     [B1] SNVM_DEVICE_BIND(BDF) —— 真正触发内核 s_nvme_probe，让 SNVMe
 *          接管控制器；之后轮询等待探测完成（最多约 10 秒）。
 *     [B2] NVM_GET_DEV_INFO —— 取回磁盘名、用户队列数、块大小等信息。
 *     [B3] open /dev/<disk_name> 并 pread 512 字节 —— 证明块设备真能读数据。
 *     [B4] SNVM_DEVICE_UNBIND(BDF) —— 解绑，把设备还回去。
 *
 *   ── 清理段（总会跑，把前面占用的东西按相反顺序释放）──
 *     [F1] NVM_UNMAP_HOST_MEMORY ×2 + NVM_CLEAR_IOQ_NUM —— 解映射 SQ/CQ 环、
 *          清空队列数，验证内核状态机能干净复位。
 *     [F2] munmap(BAR0) + close(/dev/ssnvme<N>) —— 释放 BAR0 映射、关字符设备。
 *     [F3] SNVM_CHRDEV_REMOVE(BDF) —— 让内核回收那个 minor，最后关掉控制节点。
 *
 *   整体在验证：SNVMe 模块从"创建字符设备 → 映射寄存器 → 配置队列 →
 *   （可选）真正驱动磁盘读数据 → 干净拆除"这一整条路径在当前内核上都正常。
 *
 * 【新手提示】minor（次设备号）是内核区分"同一类设备里的第几个实例"的编号；
 *   mmap 把内核/设备的一段内存映射到用户进程地址空间，之后像普通指针一样
 *   访问；SQ/CQ 是 NVMe 的提交队列/完成队列——主机把读写命令放进 SQ，
 *   控制器把完成结果写进 CQ，是 NVMe 数据通路的核心环形缓冲区。
 * ──────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    int do_bind = 0;
    const char* bdf_str = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bind") == 0) {
            do_bind = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (bdf_str == NULL) {
            bdf_str = argv[i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (bdf_str == NULL) {
        usage(argv[0]);
        return 1;
    }

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
    /*     The kernel writes the allocated minor back into addr.domain.   */
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
    /* [4] mmap BAR0                                                      */
    /*     8 KiB matches NVM_CTRL_MEM_MINSIZE used by libnvm; for any      */
    /*     stock NVMe controller BAR0 is at least 4 KiB (CAP + admin).     */
    /*                                                                    */
    /*     We intentionally do NOT pass MAP_LOCKED -- BAR0 is device      */
    /*     memory (vm_iomap_memory), not pageable, so MAP_LOCKED is       */
    /*     meaningless for it and only serves to fail under a tight       */
    /*     RLIMIT_MEMLOCK.                                                */
    /* ------------------------------------------------------------------ */
    const size_t bar0_size = 8192;
    void* bar0 = mmap(NULL, bar0_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_dev, 0);
    if (bar0 == MAP_FAILED)
        step_fail(errno, "mmap(BAR0, %zu)", bar0_size);
    step_ok("mmap(BAR0, %zu) -> %p", bar0_size, bar0);

    /* ------------------------------------------------------------------ */
    /* [5] Read CAP and decode dstrd / mqes for sanity.                    */
    /*                                                                    */
    /*     NOTE: if CAP reads as all-ones, the controller is most likely  */
    /*     in D3 / ASPM / disabled. That is NOT a SNVMe bug -- it just    */
    /*     means the device was never powered up by any driver. Warn and  */
    /*     continue; the rest of the UAPI-smoke path is still meaningful. */
    /*     all-zeros, on the other hand, means BAR0 is mapped to nothing  */
    /*     (pci_resource_start returns 0), which IS a real problem.        */
    /* ------------------------------------------------------------------ */
    uint64_t cap = mmio_read64(bar0, NVME_REG_CAP);
    uint32_t mqes = (uint32_t)(cap & 0xffff) + 1;     /* CAP.MQES is 0-based */
    uint32_t dstrd = (uint32_t)((cap >> 32) & 0xf);
    if (cap == 0)
        step_fail(EIO, "BAR0 CAP reads as all-zeros "
                       "(BAR not mapped or pci_resource_start==0)");
    if (cap == (uint64_t)-1)
        step_warn("BAR0 CAP=0xFFF..FF -- controller is probably powered down; "
                  "continuing UAPI-smoke. Bind to nvme/snvme first to test the full path.");
    else
        step_ok("BAR0 CAP=0x%016" PRIx64 " (mqes=%u, dstrd=%u)", cap, mqes, dstrd);

    /* ------------------------------------------------------------------ */
    /* [6] NVM_SET_IOQ_NUM                                                */
    /*                                                                    */
    /* Geminifs ABI: NVM_SET_IOQ_NUM now takes a struct nvm_ioctl_setup   */
    /* (NOT the legacy nvm_ioctl_map packing).  Fields used here:         */
    /*                                                                    */
    /*   .ioq_num = 2                                                     */
    /*       Total user-side IOQ count = 1 SQ + 1 CQ.  The kernel uses    */
    /*       this to size the user share at probe time.                   */
    /*                                                                    */
    /*   .flags  |= NVM_QUEUE_SETUP_F_ON_HOST                             */
    /*       Queue ring pages will live in host memory (we map them via   */
    /*       NVM_MAP_HOST_MEMORY below).  Clearing this flag tells the    */
    /*       probe path to look in device_queue_list instead -- see       */
    /*       PORTING.md §7.3.1 trap #9.                                   */
    /*                                                                    */
    /*   .cap_kernel_ioq = 32                                             */
    /*       Hard-coded smoke-test default.  Picked because:               */
    /*         (a) it is small enough that the controller-grant path      */
    /*             reliably exercises the new "queue squeeze" Case A2     */
    /*             branch in s_nvme_setup_io_queues even on NVMes with    */
    /*             generous MSI-X vector counts;                          */
    /*         (b) it is large enough that blk-mq has at least one IOQ   */
    /*             per ~6 CPUs on a 192-vCPU host, keeping the smoke     */
    /*             test's pread() responsive;                             */
    /*         (c) production callers should NOT hard-code this -- they  */
    /*             read it from sys_config.yaml's queue_setup section    */
    /*             via the NVMeService daemon.                            */
    /*                                                                    */
    /*   .nr_groups = 0                                                   */
    /*       No per-owner partitioning -- single-queue smoke test.       */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_setup setup;
        memset(&setup, 0, sizeof(setup));
        setup.ioq_num        = 2;
        setup.flags          = NVM_QUEUE_SETUP_F_ON_HOST;
        setup.cap_kernel_ioq = 32;
        if (do_ioctl(fd_dev, NVM_SET_IOQ_NUM, &setup, "NVM_SET_IOQ_NUM") < 0)
            step_fail(errno, "NVM_SET_IOQ_NUM nr=2");
    }
    step_ok("NVM_SET_IOQ_NUM nr=2 on_host=1 cap_kernel=32");

    /* ------------------------------------------------------------------ */
    /* [7] Allocate one page, map as SQ ring of queue #0                   */
    /*                                                                    */
    /* IMPORTANT: user-queue indices are **0-based**. The kernel side of  */
    /* s_nvme_probe looks up the first user CQ by calling                 */
    /*   map_find_by_pci_dev_and_idx(list, pdev, uqid=0, is_cq=1)         */
    /* inside nvme_create_io_queues_mix() (pci.c:1964-1969). Using        */
    /* ioq_idx=1 for the first queue will miss that lookup and the probe */
    /* aborts with "map_find_by_pci_dev_and_idx cq error!".               */
    /*                                                                    */
    /* No MAP_LOCKED: the kernel side pins these pages via               */
    /* get_user_pages_fast() inside NVM_MAP_HOST_MEMORY, so a userspace  */
    /* mlock()-equivalent is redundant (and fails under low              */
    /* RLIMIT_MEMLOCK inside containers).                                */
    /* ------------------------------------------------------------------ */
    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0)
        step_fail(errno, "sysconf(_SC_PAGESIZE)");

    void* sq_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (sq_buf == MAP_FAILED)
        step_fail(errno, "mmap(SQ ring)");
    memset(sq_buf, 0, (size_t)psz);

    uint64_t sq_ioaddr = 0;
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)sq_buf;
        req.n_pages     = 1;
        req.ioaddrs     = &sq_ioaddr;
        req.ioq_idx     = 0;        /* user queue #0 (0-based, see comment above) */
        req.is_cq       = 0;        /* SQ */
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(SQ)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY SQ");
    }
    step_ok("NVM_MAP_HOST_MEMORY(SQ ring) vaddr=%p ioaddr=0x%016" PRIx64,
            sq_buf, sq_ioaddr);

    /* ------------------------------------------------------------------ */
    /* [8] Allocate one page, map as CQ ring of queue #0                   */
    /* ------------------------------------------------------------------ */
    void* cq_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (cq_buf == MAP_FAILED)
        step_fail(errno, "mmap(CQ ring)");
    memset(cq_buf, 0, (size_t)psz);

    uint64_t cq_ioaddr = 0;
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)cq_buf;
        req.n_pages     = 1;
        req.ioaddrs     = &cq_ioaddr;
        req.ioq_idx     = 0;        /* user queue #0 (matches SQ above) */
        req.is_cq       = 1;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(CQ)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY CQ");
    }
    step_ok("NVM_MAP_HOST_MEMORY(CQ ring) vaddr=%p ioaddr=0x%016" PRIx64,
            cq_buf, cq_ioaddr);

    /* ------------------------------------------------------------------ */
    /* [9] NVM_SET_SHARE_REG -- arms the use_sreg gate.                    */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.ioq_idx = 1;            /* enable */
        if (do_ioctl(fd_dev, NVM_SET_SHARE_REG, &req, "NVM_SET_SHARE_REG") < 0)
            step_fail(errno, "NVM_SET_SHARE_REG(1)");
    }
    step_ok("NVM_SET_SHARE_REG(1)");

    /* ================================================================== */
    /*  --bind path: actually run s_nvme_probe and exercise the resulting */
    /*  /dev/snvme<X>n<Y> block device (note the leading 's' -- SNVMe     */
    /*  namespaces its disks away from the in-tree nvme driver; see       */
    /*  PORTING.md §2 and core.c:3806).                                    */
    /*                                                                    */
    /*  Note: 1 SQ + 1 CQ is below NVM_CTRL_IOQ_MINNUM=64 in libnvm. The   */
    /*  in-kernel probe is still happy (it adapts nr_user_q from ctrl),   */
    /*  but a higher queue count needs more user pages -- this UAPI smoke */
    /*  intentionally stays minimal.                                      */
    /* ================================================================== */
    char disk_name_buf[DISK_NAME_LEN + 1] = {0};

    if (do_bind) {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_BIND, &bdf, "SNVM_DEVICE_BIND") < 0)
            step_fail(errno,
                "SNVM_DEVICE_BIND %s -- the in-tree nvme driver may still own this device, "
                "or use_sreg/ioq_map_num invariants are off (see PORTING.md §5)",
                bdf_str);
        /*
         * s_nvme_probe() schedules an async worker for reset + scan.
         * Poll for NVM_GET_DEV_INFO to succeed instead of assuming a
         * hard-coded sleep is enough -- slow / power-managed drives
         * can easily take more than 3 s to come up.
         */
        struct nvm_ioctl_dev info;
        int ok = 0;
        for (int i = 0; i < 100; i++) {   /* up to ~10 s */
            memset(&info, 0, sizeof(info));
            if (ioctl(fd_dev, NVM_GET_DEV_INFO, &info) == 0 &&
                info.disk_name[0] != '\0') {
                ok = 1;
                break;
            }
            usleep(100 * 1000);
        }
        if (!ok)
            step_fail(errno, "NVM_GET_DEV_INFO did not succeed within 10s after bind "
                             "(probe may still be running; try --bind on a faster disk)");
        step_ok("SNVM_DEVICE_BIND %s (probe done)", bdf_str);

        /* `info` was already filled by the poll loop above. */
        memcpy(disk_name_buf, info.disk_name, DISK_NAME_LEN);
        step_ok("NVM_GET_DEV_INFO disk='%s' nr_user_q=%u block_size=%zu max_data_size=%zu",
                disk_name_buf, info.nr_user_q, info.block_size, info.max_data_size);

        char blk_path[DISK_NAME_LEN + 8];
        snprintf(blk_path, sizeof(blk_path), "/dev/%s", disk_name_buf);
        int fd_blk = open(blk_path, O_RDONLY);
        if (fd_blk < 0)
            step_fail(errno, "open(%s) -- block device did not appear", blk_path);
        char buf[512];
        ssize_t got = pread(fd_blk, buf, sizeof(buf), 0);
        if (got != (ssize_t)sizeof(buf))
            step_fail(errno, "pread(%s, 512) returned %zd", blk_path, got);
        close(fd_blk);
        step_ok("pread(%s, 512) ok", blk_path);

        bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_UNBIND, &bdf, "SNVM_DEVICE_UNBIND") < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND %s", bdf_str);
        step_ok("SNVM_DEVICE_UNBIND %s", bdf_str);
    }

    /* ------------------------------------------------------------------ */
    /* [F1] Cleanup: unmap rings, clear ioq state                          */
    /* ------------------------------------------------------------------ */
    {
        uint64_t v;
        v = (uint64_t)(uintptr_t)sq_buf;
        if (do_ioctl(fd_dev, NVM_UNMAP_HOST_MEMORY, &v,
                     "NVM_UNMAP_HOST_MEMORY(SQ)") < 0)
            step_fail(errno, "NVM_UNMAP_HOST_MEMORY SQ");
        v = (uint64_t)(uintptr_t)cq_buf;
        if (do_ioctl(fd_dev, NVM_UNMAP_HOST_MEMORY, &v,
                     "NVM_UNMAP_HOST_MEMORY(CQ)") < 0)
            step_fail(errno, "NVM_UNMAP_HOST_MEMORY CQ");
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_dev, NVM_CLEAR_IOQ_NUM, &req,
                     "NVM_CLEAR_IOQ_NUM") < 0)
            step_fail(errno, "NVM_CLEAR_IOQ_NUM");
    }
    step_ok("NVM_UNMAP_HOST_MEMORY x2 + NVM_CLEAR_IOQ_NUM");

    /* ------------------------------------------------------------------ */
    /* [F2] Release per-controller resources                               */
    /* ------------------------------------------------------------------ */
    munmap(sq_buf, (size_t)psz);
    munmap(cq_buf, (size_t)psz);
    if (munmap(bar0, bar0_size) < 0)
        step_fail(errno, "munmap(BAR0)");
    if (close(fd_dev) < 0)
        step_fail(errno, "close(%s)", dev_path);
    step_ok("munmap(BAR0) + close(%s)", dev_path);

    /* ------------------------------------------------------------------ */
    /* [F3] SNVM_CHRDEV_REMOVE -- release the per-controller minor.        */
    /* ------------------------------------------------------------------ */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf,
                     "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE %s", bdf_str);
    }
    step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);

    close(fd_ctl);
    fprintf(stderr, "\nAll %d steps passed. SNVMe is healthy%s.\n",
            g_step, do_bind ? " (full bring-up)" : " (UAPI-smoke)");
    return 0;
}
