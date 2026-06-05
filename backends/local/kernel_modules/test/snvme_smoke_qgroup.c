/*
 * snvme_smoke_qgroup.c -- Smoke test for NVM_CREATE_QUEUE_GROUP /
 * NVM_DESTROY_QUEUE_GROUP (queue-group plan, step B1).
 *
 * Purpose
 * -------
 *
 * B1 introduces the per-fd queue group container in snvme, but
 * still has *zero* NVMe-side resources attached to it -- no maps,
 * no user IO queues, no admin commands.  The point of this test
 * is therefore to exercise just the kernel container lifecycle:
 *
 *   - allocate a group and get an opaque group_id back,
 *   - destroy it explicitly,
 *   - allocate again and let the fd close cascade-clean it,
 *   - confirm cap enforcement (NVM_MAX_GROUPS_PER_FD),
 *   - confirm cross-fd isolation (group_id from fd A is not
 *     destroyable on fd B),
 *   - confirm error paths (group_id=0 is rejected; invalid id is
 *     -ENOENT).
 *
 * Crucially: this test does NOT call SNVM_DEVICE_BIND.  Group
 * lifecycle is bind-agnostic by design -- userspace can prepare
 * the group container before the controller is bound, and the
 * later NVM_ADD_USER_QUEUE call (B3) is the one that requires a
 * live admin_q.  Skipping bind here also keeps the test
 * non-destructive: it does not touch the in-tree nvme driver's
 * ownership of the BDF, so it is safe to run on a host where the
 * target NVMe namespace is mounted.
 *
 * After running, you can grep dmesg for:
 *
 *   "snvme: NVM_CREATE_QUEUE_GROUP id=N max_queues=16 pid=..."
 *   "snvme: NVM_DESTROY_QUEUE_GROUP id=N pid=..."
 *   "snvme: snvm_dev_release: cascade-destroyed K orphan group(s) ..."
 *
 * to confirm the kernel-side path executed.  These are pr_debug /
 * pr_info; you may need `dmesg -n 8` or similar.
 *
 * Build:    make snvme_smoke_qgroup        (parent Makefile)
 * Invoke:   sudo ./snvme_smoke_qgroup <PCI_BDF>
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
#include <unistd.h>

#include "ioctl.h"

/* ------------------------------------------------------------------ */
/* Logging helpers                                                    */
/* ------------------------------------------------------------------ */

static int g_step = 0;

/* ────────────────────────────────────────────────────────────
 * 【函数】step_ok(const char* fmt, ...) —— 可变参数（类似 printf）
 * 【作用】打印一行“某一步通过”的日志。每调用一次，全局步骤计数
 *         器 g_step 自增 1，然后输出形如 "[ OK ] step=3 ...." 的信息。
 * 【参数】fmt + 后续可变参数：和 printf 一样的格式串和参数，描述
 *         本步骤做了什么。
 * 【返回】无返回值（void）；不会让程序退出，测试继续往下走。
 * 【在测试中的角色】贯穿整个测试，每个验证点成功后都用它打一条
 *         绿色的“通过”记录，方便人眼/CI 数清楚走到了第几步。
 * 【新手提示】
 *   - va_list / va_start / va_end 是 C 处理“可变参数函数”的标准
 *     机制；vfprintf 就是把这一串可变参数转交给 fprintf 去格式化。
 *   - 这里全部写到 stderr（标准错误），是为了日志和真正的程序
 *     输出分流，CI 抓日志更方便。
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
 * 【函数】step_fail(int err, const char* fmt, ...) —— 标记 noreturn
 * 【作用】打印一行“某一步失败”的日志，然后立刻 exit(2) 结束整个
 *         测试程序。和 step_ok 一样会让 g_step 自增并打印步号，但
 *         额外打印 errno 及其文字含义，最后直接退出。
 * 【参数】
 *   - err：失败时的错误码（通常传入捕获到的 errno）。若传 0，表示
 *          “这次失败不是系统调用错误”（如逻辑断言不满足），日志里
 *          会显示 errno=0 (n/a)。
 *   - fmt + 可变参数：描述哪一步、为什么失败的格式串。
 * 【返回】不返回（__attribute__((noreturn))）。函数末尾调用
 *         exit(2)，进程以退出码 2 终止 —— 对应文件头注释里“某个
 *         冒烟步骤失败”的约定。
 * 【在测试中的角色】是整个测试的“硬失败出口”。任何一个验证点
 *         不满足预期，就调它打印原因并让整程序非零退出，CI 据此
 *         判定测试不通过。
 * 【新手提示】
 *   - errno 是 C 标准库里记录“上一次系统调用出错原因”的全局变量；
 *     strerror() 把它翻成人能读的字符串（如 "No such file"）。
 *   - noreturn 是给编译器的提示：调用它之后的代码不会再执行，可
 *     避免“函数可能没有返回值”之类的误报。
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
 * 【函数】parse_bdf(const char* s, struct pci_device_addr* out)
 * 【作用】把命令行里给的 PCI 设备地址字符串（形如
 *         "0000:50:00.0"）解析成结构体里的四个数字字段。
 * 【参数】
 *   - s：用户输入的 BDF 字符串。格式是 域:总线:槽位.功能
 *        （Domain : Bus : Device/slot . Function），全部按十六进制解释。
 *   - out：输出参数，解析得到的四个值分别写入 out->domain /
 *          ->bus / ->slot / ->func。
 * 【返回】成功返回 0；格式不对（没凑齐 4 个字段）返回 -1。
 * 【在测试中的角色】测试启动初期，把用户给的盘地址转成内核
 *         ioctl 需要的结构体，后面 SNVM_CHRDEV_CREATE 才能据此
 *         找到目标 NVMe 设备。
 * 【新手提示】
 *   - BDF 是 PCI 设备在系统里的“门牌号”，唯一定位一块板卡/功能。
 *   - sscanf 的返回值是“成功匹配并赋值的字段个数”，这里用
 *     "== 4" 来判断是不是四个字段都解析成功了。
 *   - "%x" 表示按十六进制读取，所以 BDF 里的数都当 16 进制看。
 * ──────────────────────────────────────────────────────────── */
static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* Tiny ioctl wrapper.  We deliberately do NOT step_fail() inside     */
/* the wrapper -- some tests below use ioctl() directly because they */
/* expect a specific errno.                                           */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】do_ioctl(int fd, unsigned long req, void* arg, const char* what)
 * 【作用】对内核字符设备发起一次 ioctl 调用的“薄封装”。如果
 *         调用失败（返回 < 0），它会顺手把错误打印出来，并且
 *         小心地把 errno 原样保留好（先存起来，打印完再写回），
 *         以免 fprintf 等操作把 errno 冲掉。
 * 【参数】
 *   - fd：已打开的设备文件描述符（如 /dev/ssnvmeN 或控制节点）。
 *   - req：ioctl 命令号（如 NVM_CREATE_QUEUE_GROUP 这些宏）。
 *   - arg：指向命令所需参数结构体的指针，内核会读/写它。
 *   - what：命令的可读名字，仅用于出错时打印，便于定位。
 * 【返回】直接返回 ioctl 的结果：0/正数表示成功，<0 表示失败
 *         （此时 errno 已被恢复成内核给的错误码）。
 * 【在测试中的角色】测试里“期望成功”的 ioctl 都走这个封装，省去
 *         重复的错误打印代码。注意：它【故意】不在内部调用
 *         step_fail —— 因为有些用例是【期望失败并校验具体 errno】，
 *         那些地方直接用裸 ioctl()，不能让封装替它们决定成败。
 * 【新手提示】
 *   - ioctl（input/output control）是用户态向设备驱动下达“自定义
 *     命令”的通用入口，命令号 + 参数结构体由驱动自行约定。
 *   - errno 会被后续库调用覆盖，所以这里用局部变量 e 临时保存，
 *     是写系统编程时很常见的防坑写法。
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
 * 【作用】向 stderr 打印本程序的用法说明：怎么传 BDF 参数、举个
 *         例子，并强调“本测试不绑定控制器、因此对已挂载的盘也
 *         安全”。
 * 【参数】prog：程序名（一般就是 argv[0]），插进提示里让示例更
 *         贴合实际调用方式。
 * 【返回】无（void），只负责打印。
 * 【在测试中的角色】参数个数不对、或用户传了 -h/--help 时调用，
 *         给出可读的帮助。
 * 【新手提示】把帮助信息打到 stderr 而不是 stdout，是命令行工具
 *         的惯例 —— 这样即使把正常输出重定向到文件，帮助/报错
 *         仍能显示在终端上。
 * ──────────────────────────────────────────────────────────── */
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <PCI_BDF>\n"
        "  e.g.: %s 0000:50:00.0\n"
        "\n"
        "This test exercises the per-fd queue-group container only.\n"
        "It does NOT bind the controller, so it is safe to run on a\n"
        "host where the target NVMe device is mounted.\n",
        prog, prog);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】main(int argc, char** argv)
 * 【作用】整个冒烟测试的主控流程。先解析命令行 BDF，创建并打开
 *         snvme 字符设备，然后按 [1]~[19] 一步步验证“队列组
 *         (queue group) 容器”的生命周期，以及 B2 阶段“带 group_id
 *         的 host 内存映射”的行为。全程【不绑定控制器】，所以
 *         不会破坏在用的盘，挂载着也能安全跑。
 *
 * 【整体在验证什么】队列组是 snvme 里“每个 fd 私有”的资源容器。
 *         本测试要证明内核把它的“分配 / 销毁 / 关 fd 自动清理 /
 *         上限 / 跨 fd 隔离 / 各种错误码”都做对了，并且销毁组时
 *         能级联清理挂在组上的 host 内存映射，不泄漏被 pin 住的页。
 *
 * 【完整执行流程】
 *   入口：argc 必须为 2（程序名 + BDF）；传 -h/--help 打印用法后
 *        退出 0；BDF 解析失败退出 1。
 *   [1] open(/dev/snvm_control) 控制节点，再用 SNVM_CHRDEV_CREATE
 *       为这个 BDF 创建出 /dev/ssnvme<N>，拿回分配到的 minor 号 N。
 *   [2] 打开 ssnvme 设备得到主 fd_a（后续大部分操作都在它上面做）。
 *   [3] NVM_CREATE_QUEUE_GROUP 正常路径：校验返回的 group_id≠0
 *       （0 是“无组”哨兵），max_queues 等于 NVM_MAX_QUEUES_PER_GROUP。
 *   [4] 同一个 fd 再创建一次：期望 -EBUSY，验证每 fd 组数上限
 *       (NVM_MAX_GROUPS_PER_FD，B1 阶段为 1) 被正确强制执行。
 *   [5] NVM_DESTROY_QUEUE_GROUP 正常销毁 fd_a 自己的组。
 *   [6] 再销毁同一个 id（已不存在）：期望 -ENOENT，防内核重复释放。
 *   [7] 销毁 group_id=0（哨兵值）：期望 -EINVAL（查表前就拒绝）。
 *   [8] 跨 fd 隔离：开 fd_b 并在其上建组，然后从 fd_a 去销毁
 *       fd_b 的组 —— 必须 -ENOENT。证明组只能被自己的 fd 访问，
 *       别的进程不能扫 group_id 去拆别人的组。（注释里也解释了
 *       IDA 回收 id 导致 group_b 可能等于刚释放的 group_a，属正常。）
 *   [9] 关 fd 级联清理：故意不销毁 fd_b 的组就 close(fd_b)，内核须
 *       自动回收该孤儿组（证据看 dmesg 的 "cascade-destroyed ..."）。
 *       随后开 fd_c 再建一个组、再 close(fd_c)，验证分配器仍健康、
 *       第二次级联清理也正常。
 *  [10] flags 必须为 0 (MBZ)：建组时 flags 设成非零 —— 期望 -EINVAL。
 *
 *   —— 以下为 B2：NVM_MAP_HOST_MEMORY 带 group_id 的新模式 ——
 *       先用 sysconf 取页大小 psz。
 *  [11] 在 fd_a 上新建一个组 group_d，专门给 B2 的映射实验用。
 *  [12] mmap 一页匿名内存当测试缓冲区（不加 MAP_LOCKED，因为内核
 *       会用 get_user_pages_fast 自己 pin 页）。
 *  [13] NVM_MAP_HOST_MEMORY 新模式正常路径：把这页注册到 group_d，
 *       校验返回的 DMA/IO 地址非 0，内核应把 map 挂到组的 maps 链上。
 *  [14] reserved 字段必须为 0：故意把保留字节写脏 —— 期望 -EINVAL
 *       （前向兼容保护，用独立缓冲区以免污染 [13] 的映射）。
 *  [15] 用一个几乎不可能被分配过的 group_id (0xdeadbeef) 注册：
 *       期望 -ENOENT，且内核在查不到组时不能泄漏已 pin 的页。
 *  [16] 跨 fd 映射隔离：开 fd_d，拿 fd_a 的 group_d 去注册 —— 必须
 *       -ENOENT，验证映射注册同样受每 fd 隔离约束。
 *  [17] 销毁 group_d：组上还挂着 [13] 注册的那个 map，销毁时内核必须
 *       级联 drain 掉它（无需显式 unmap），证据见 dmesg
 *       "destroy_qgroup id=N drained 1 map"。
 *  [18] 验证级联确实生效：对同一 vaddr 显式 NVM_UNMAP_HOST_MEMORY
 *       现在应返回 -EINVAL（说明 map 已从全局表里被清掉）。然后
 *       munmap 掉测试缓冲区。
 *  [19] 收尾，让测试可重复运行：close(fd_a)（此时它名下应无残留组，
 *       不应出现 cascade 日志）；SNVM_CHRDEV_REMOVE 删掉刚才创建的
 *       字符设备节点；关闭控制 fd。
 *   末尾：打印 "all N steps passed" 并 return 0。
 *
 * 【参数】argc/argv：标准命令行参数，argv[1] 必须是 PCI BDF 字符串。
 * 【返回】全部步骤通过返回 0；用法错误返回 1；任何冒烟步骤失败时
 *         会在 step_fail 内部 exit(2)，不会正常走到 return。
 * 【新手提示】
 *   - “group_id / IO 地址 / EBUSY / ENOENT / EINVAL” 这些是测试反复
 *     校验的核心：能拿到非 0 的 group_id、错误路径返回对的 errno、
 *     关 fd 自动清理、跨 fd 互不可见 —— 就是这个容器要保证的语义。
 *   - 很多“清理是否发生”的证据落在内核日志里（用 dmesg 看那几条
 *     pr_info/pr_debug），用户态只能通过“后续操作行为是否符合预期”
 *     来间接确认，注释里已逐处说明。
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
    /* [1] open /dev/snvm_control and create /dev/ssnvme<N> for our BDF.  */
    /*                                                                    */
    /* CHRDEV_CREATE returns the freshly allocated minor in addr.domain   */
    /* (mirroring SNVM_CHRDEV_REMOVE on the way out).                     */
    /* ------------------------------------------------------------------ */
    int fd_ctl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctl < 0)
        step_fail(errno, "open(/dev/snvm_control)");
    step_ok("open(/dev/snvm_control) fd=%d", fd_ctl);

    struct pci_device_addr addr = orig_bdf;
    if (do_ioctl(fd_ctl, SNVM_CHRDEV_CREATE, &addr, "SNVM_CHRDEV_CREATE") < 0)
        step_fail(errno, "SNVM_CHRDEV_CREATE %s", bdf_str);
    int minor_n = addr.domain;
    step_ok("SNVM_CHRDEV_CREATE minor=%d", minor_n);

    char dev_path[64];
    snprintf(dev_path, sizeof(dev_path), "/dev/ssnvme%d", minor_n);

    /* ------------------------------------------------------------------ */
    /* [2] Open ssnvme fd #A.  This is the primary fd; we'll do most of */
    /* the lifecycle work on it.                                          */
    /* ------------------------------------------------------------------ */
    int fd_a = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_a < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd_a=%d", dev_path, fd_a);

    /* ------------------------------------------------------------------ */
    /* [3] NVM_CREATE_QUEUE_GROUP -- happy path.                          */
    /*                                                                    */
    /* Verifies:                                                          */
    /*   - kernel returns group_id != 0 (0 is the "no group" sentinel)   */
    /*   - kernel echoes max_queues = NVM_MAX_QUEUES_PER_GROUP            */
    /*   - flags / reserved untouched                                    */
    /* ------------------------------------------------------------------ */
    uint32_t group_a;
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_a, NVM_CREATE_QUEUE_GROUP, &req, "NVM_CREATE_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP on fd_a");
        if (req.group_id == 0)
            step_fail(0, "NVM_CREATE_QUEUE_GROUP returned group_id=0 (sentinel)");
        if (req.max_queues != NVM_MAX_QUEUES_PER_GROUP)
            step_fail(0, "NVM_CREATE_QUEUE_GROUP max_queues=%u, expected %u",
                      req.max_queues, NVM_MAX_QUEUES_PER_GROUP);
        group_a = req.group_id;
        step_ok("NVM_CREATE_QUEUE_GROUP fd_a -> group_id=%u max_queues=%u",
                req.group_id, req.max_queues);
    }

    /* ------------------------------------------------------------------ */
    /* [4] NVM_CREATE_QUEUE_GROUP -- second call on same fd, expect      */
    /* -EBUSY (cap = NVM_MAX_GROUPS_PER_FD = 1 in B1).                   */
    /*                                                                    */
    /* This guards against a regression where the cap check is dropped   */
    /* or off-by-one'd.  If userspace ever needs >1 group/fd, the cap    */
    /* must be raised in the kernel AND in this assertion together.      */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        int r = ioctl(fd_a, NVM_CREATE_QUEUE_GROUP, &req);
        if (r == 0)
            step_fail(0,
                "NVM_CREATE_QUEUE_GROUP unexpectedly succeeded on second call "
                "(got group_id=%u); per-fd cap (%u) is not enforced",
                req.group_id, NVM_MAX_GROUPS_PER_FD);
        if (errno != EBUSY)
            step_fail(errno,
                "NVM_CREATE_QUEUE_GROUP second call returned errno=%d "
                "(expected EBUSY=%d)",
                errno, EBUSY);
        step_ok("per-fd group cap correctly returns -EBUSY on second create");
    }

    /* ------------------------------------------------------------------ */
    /* [5] NVM_DESTROY_QUEUE_GROUP -- happy path on fd_a's own group.    */
    /* ------------------------------------------------------------------ */
    {
        uint32_t gid = group_a;
        if (do_ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid, "NVM_DESTROY_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP id=%u on fd_a", group_a);
        step_ok("NVM_DESTROY_QUEUE_GROUP id=%u on fd_a", group_a);
    }

    /* ------------------------------------------------------------------ */
    /* [6] NVM_DESTROY_QUEUE_GROUP -- destroying the same id twice must */
    /* return -ENOENT (the descriptor is gone after step 5).             */
    /* ------------------------------------------------------------------ */
    {
        uint32_t gid = group_a;
        int r = ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid);
        if (r == 0)
            step_fail(0,
                "NVM_DESTROY_QUEUE_GROUP id=%u unexpectedly succeeded twice "
                "(double-free of group descriptor in kernel?)", group_a);
        if (errno != ENOENT)
            step_fail(errno, "second destroy returned errno=%d, expected ENOENT(%d)",
                      errno, ENOENT);
        step_ok("double NVM_DESTROY_QUEUE_GROUP correctly returns -ENOENT");
    }

    /* ------------------------------------------------------------------ */
    /* [7] NVM_DESTROY_QUEUE_GROUP -- group_id=0 is the sentinel and     */
    /* must be rejected with -EINVAL before any list lookup.             */
    /* ------------------------------------------------------------------ */
    {
        uint32_t gid = 0;
        int r = ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid);
        if (r == 0)
            step_fail(0, "NVM_DESTROY_QUEUE_GROUP id=0 unexpectedly succeeded");
        if (errno != EINVAL)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP id=0 returned errno=%d, "
                             "expected EINVAL(%d)", errno, EINVAL);
        step_ok("NVM_DESTROY_QUEUE_GROUP id=0 correctly returns -EINVAL");
    }

    /* ------------------------------------------------------------------ */
    /* [8] Cross-fd isolation: open a second fd, create a group on it,  */
    /* then try to destroy that group from fd_a -- must -ENOENT.         */
    /*                                                                    */
    /* This validates the design invariant that group descriptors are    */
    /* per-fd reachable only.  An adversarial process should not be     */
    /* able to scan group_ids and tear down a sibling's groups.         */
    /* ------------------------------------------------------------------ */
    int fd_b = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_b < 0)
        step_fail(errno, "open(%s) for fd_b", dev_path);
    step_ok("open(%s) fd_b=%d", dev_path, fd_b);

    uint32_t group_b;
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_b, NVM_CREATE_QUEUE_GROUP, &req, "NVM_CREATE_QUEUE_GROUP fd_b") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP on fd_b");
        if (req.group_id == 0)
            step_fail(0, "NVM_CREATE_QUEUE_GROUP returned group_id=0 on fd_b");
        /*
         * NB: req.group_id == group_a (i.e. the IDA recycled the
         * id we just freed in step 5) is *expected* and not a
         * bug.  Linux's IDA always allocates the lowest free id,
         * so after ida_simple_remove(1) the very next get() will
         * return 1 again.  Userspace MUST treat group_id as an
         * opaque cookie and never rely on it being unique across
         * the program's lifetime, only for the lifetime of the
         * fd that owns it.  Cross-fd isolation (step below) is
         * what actually guarantees safety.
         */
        group_b = req.group_id;
        step_ok("NVM_CREATE_QUEUE_GROUP fd_b -> group_id=%u%s",
                group_b,
                group_b == group_a ? " (IDA recycled freed id; expected)" : "");
    }
    {
        uint32_t gid = group_b;
        int r = ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid);
        if (r == 0)
            step_fail(0,
                "fd_a managed to destroy fd_b's group %u -- cross-fd "
                "isolation is broken in the kernel",
                group_b);
        if (errno != ENOENT)
            step_fail(errno,
                "fd_a destroy of fd_b's group returned errno=%d, "
                "expected ENOENT(%d)", errno, ENOENT);
        step_ok("cross-fd destroy correctly blocked: fd_a -> fd_b's group %u "
                "returns -ENOENT", group_b);
    }

    /* ------------------------------------------------------------------ */
    /* [9] Cascade cleanup on fd close: leave fd_b's group alive and    */
    /* close fd_b.  The kernel must reap group_b automatically.         */
    /*                                                                    */
    /* We can't directly observe fd_b's groups list from userspace --   */
    /* the proof point is in dmesg ("snvme: snvm_dev_release:           */
    /* cascade-destroyed N orphan group(s)...").  To make this testable */
    /* in CI even without dmesg access, we follow up by re-opening      */
    /* /dev/ssnvme<N> and creating yet another group; if the IDA leaked */
    /* the id, we'd see ever-growing group_ids across runs (best-effort */
    /* signal, not deterministic).                                      */
    /* ------------------------------------------------------------------ */
    if (close(fd_b) < 0)
        step_fail(errno, "close(fd_b)");
    step_ok("close(fd_b) -- kernel must cascade-destroy group_id=%u "
            "(grep dmesg for 'cascade-destroyed 1 orphan group(s)')",
            group_b);

    int fd_c = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_c < 0)
        step_fail(errno, "open(%s) for fd_c", dev_path);
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_c, NVM_CREATE_QUEUE_GROUP, &req, "NVM_CREATE_QUEUE_GROUP fd_c") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP on fd_c after cascade");
        step_ok("post-cascade NVM_CREATE_QUEUE_GROUP fd_c -> group_id=%u "
                "(allocator still healthy)",
                req.group_id);
        /* Leave this group attached -- close(fd_c) should clean it. */
    }
    if (close(fd_c) < 0)
        step_fail(errno, "close(fd_c)");
    step_ok("close(fd_c) -- second cascade for the post-recovery group");

    /* ------------------------------------------------------------------ */
    /* [10] flags MBZ rejection: non-zero flags must be -EINVAL.         */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        req.flags = 0xdeadbeef;
        int r = ioctl(fd_a, NVM_CREATE_QUEUE_GROUP, &req);
        if (r == 0)
            step_fail(0,
                "NVM_CREATE_QUEUE_GROUP with flags=0x%x unexpectedly succeeded "
                "(group_id=%u); MBZ check missing",
                0xdeadbeef, req.group_id);
        if (errno != EINVAL)
            step_fail(errno,
                "NVM_CREATE_QUEUE_GROUP with flags!=0 returned errno=%d, "
                "expected EINVAL(%d)", errno, EINVAL);
        step_ok("flags MBZ check rejects non-zero with -EINVAL");
    }

    /* ================================================================== */
    /* B2: NVM_MAP_HOST_MEMORY with group_id                              */
    /*                                                                    */
    /* The next block creates a fresh group on fd_a, registers a host    */
    /* page against it via NVM_MAP_HOST_MEMORY (group_id != 0, new       */
    /* mode), then validates:                                             */
    /*   - the map IO addresses are returned correctly,                  */
    /*   - registering against a foreign group_id returns -ENOENT,       */
    /*   - registering with reserved!=0 is rejected with -EINVAL,        */
    /*   - destroying the group cascades through the map and releases   */
    /*     the pinned page (dmesg "destroy_qgroup id=N drained 1 map"). */
    /* ================================================================== */

    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0)
        step_fail(errno, "sysconf(_SC_PAGESIZE)");

    /* ------------------------------------------------------------------ */
    /* [11] Create a new group on fd_a for the B2 map experiments.       */
    /* ------------------------------------------------------------------ */
    uint32_t group_d;
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_a, NVM_CREATE_QUEUE_GROUP, &req,
                     "NVM_CREATE_QUEUE_GROUP fd_a (B2 setup)") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP fd_a (B2 setup)");
        group_d = req.group_id;
        step_ok("B2 setup: NVM_CREATE_QUEUE_GROUP fd_a -> group_id=%u",
                group_d);
    }

    /* ------------------------------------------------------------------ */
    /* [12] mmap one page of host memory for the test buffer.            */
    /*                                                                    */
    /* No MAP_LOCKED: NVM_MAP_HOST_MEMORY pins the page kernel-side via  */
    /* get_user_pages_fast(), so a userspace mlock is redundant (and    */
    /* fails under low RLIMIT_MEMLOCK in containers anyway).             */
    /* ------------------------------------------------------------------ */
    void* host_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host_buf == MAP_FAILED)
        step_fail(errno, "mmap(host page)");
    memset(host_buf, 0, (size_t)psz);
    step_ok("mmap(host page) -> %p", host_buf);

    /* ------------------------------------------------------------------ */
    /* [13] NVM_MAP_HOST_MEMORY against group_d -- happy path.           */
    /*                                                                    */
    /* This validates the whole new-mode pipeline: kernel finds the      */
    /* group via find_qgroup_locked, attaches the map to g->maps,        */
    /* increments g->nr_maps, and returns the DMA addresses.             */
    /* ------------------------------------------------------------------ */
    uint64_t io_addr = 0;
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)host_buf;
        req.n_pages     = 1;
        req.ioaddrs     = &io_addr;
        req.ioq_idx     = -1;        /* new mode: ioq_idx ignored */
        req.is_cq       = -1;
        req.group_id    = group_d;
        if (do_ioctl(fd_a, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(group_d)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY new-mode against group %u",
                      group_d);
        if (io_addr == 0)
            step_fail(0, "NVM_MAP_HOST_MEMORY returned ioaddr=0 (DMA mapping failed?)");
        step_ok("NVM_MAP_HOST_MEMORY new-mode group=%u vaddr=%p ioaddr=0x%016" PRIx64,
                group_d, host_buf, io_addr);
    }

    /* ------------------------------------------------------------------ */
    /* [14] NVM_MAP_HOST_MEMORY with reserved!=0 must -EINVAL.           */
    /*                                                                    */
    /* Guards forward-compat: if a future kernel adds new flags via the  */
    /* reserved field, an old userspace that happens to pass garbage     */
    /* there must fail loudly rather than silently misinterpret.         */
    /* ------------------------------------------------------------------ */
    {
        /* Use a separate buffer so the failure path doesn't accidentally
         * shadow the [13] mapping above. */
        void* probe_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (probe_buf == MAP_FAILED)
            step_fail(errno, "mmap(probe page) for reserved-MBZ test");

        uint64_t throwaway_ioaddr = 0;
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)probe_buf;
        req.n_pages     = 1;
        req.ioaddrs     = &throwaway_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_d;
        /* B6: `reserved` was renamed to `map_kind` (1 B) +
         * `reserved0[3]`.  Probe the MBZ check by setting one of
         * the reserved padding bytes; the kernel must still
         * reject the request with -EINVAL.                       */
        req.reserved0[1] = 0xab;
        int r = ioctl(fd_a, NVM_MAP_HOST_MEMORY, &req);
        if (r == 0)
            step_fail(0, "NVM_MAP_HOST_MEMORY with reserved0!=0 unexpectedly "
                         "succeeded -- MBZ check missing");
        if (errno != EINVAL)
            step_fail(errno, "NVM_MAP_HOST_MEMORY with reserved0!=0 errno=%d, "
                             "expected EINVAL(%d)", errno, EINVAL);
        munmap(probe_buf, (size_t)psz);
        step_ok("NVM_MAP_HOST_MEMORY reserved-MBZ check returns -EINVAL");
    }

    /* ------------------------------------------------------------------ */
    /* [15] NVM_MAP_HOST_MEMORY with bogus group_id (never allocated)    */
    /* must -ENOENT.                                                      */
    /*                                                                    */
    /* This validates that find_qgroup_locked rejects unknown ids        */
    /* without leaking a half-pinned page (the kernel must               */
    /* unmap_and_release on the find-failure path).  We can't directly  */
    /* observe the leak from userspace; the proof is "test reruns        */
    /* without exhausting RLIMIT_MEMLOCK across many invocations".       */
    /* ------------------------------------------------------------------ */
    {
        void* probe_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (probe_buf == MAP_FAILED)
            step_fail(errno, "mmap(probe page) for bogus-group test");

        uint64_t throwaway_ioaddr = 0;
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)probe_buf;
        req.n_pages     = 1;
        req.ioaddrs     = &throwaway_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0xdeadbeefU;   /* almost certainly never assigned */
        int r = ioctl(fd_a, NVM_MAP_HOST_MEMORY, &req);
        if (r == 0)
            step_fail(0, "NVM_MAP_HOST_MEMORY against unknown group_id=0x%x "
                         "unexpectedly succeeded", 0xdeadbeefU);
        if (errno != ENOENT)
            step_fail(errno, "NVM_MAP_HOST_MEMORY against unknown group_id "
                             "errno=%d, expected ENOENT(%d)", errno, ENOENT);
        munmap(probe_buf, (size_t)psz);
        step_ok("NVM_MAP_HOST_MEMORY against bogus group_id returns -ENOENT");
    }

    /* ------------------------------------------------------------------ */
    /* [16] Cross-fd: open fd_d, try to register a host page under       */
    /* fd_a's group_d.  Must -ENOENT (per-fd isolation).                  */
    /* ------------------------------------------------------------------ */
    {
        int fd_d = open(dev_path, O_RDWR | O_NONBLOCK);
        if (fd_d < 0)
            step_fail(errno, "open(%s) for fd_d", dev_path);

        void* probe_buf = mmap(NULL, (size_t)psz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (probe_buf == MAP_FAILED)
            step_fail(errno, "mmap(probe page) for cross-fd test");

        uint64_t throwaway_ioaddr = 0;
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)probe_buf;
        req.n_pages     = 1;
        req.ioaddrs     = &throwaway_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_d;     /* fd_a's group, used from fd_d */
        int r = ioctl(fd_d, NVM_MAP_HOST_MEMORY, &req);
        if (r == 0)
            step_fail(0,
                "fd_d managed to register a map under fd_a's group %u -- "
                "per-fd isolation broken", group_d);
        if (errno != ENOENT)
            step_fail(errno, "cross-fd map registration errno=%d, "
                             "expected ENOENT(%d)", errno, ENOENT);
        munmap(probe_buf, (size_t)psz);
        close(fd_d);
        step_ok("cross-fd NVM_MAP_HOST_MEMORY against another fd's group "
                "correctly returns -ENOENT");
    }

    /* ------------------------------------------------------------------ */
    /* [17] Destroy group_d -- the registered map MUST be drained        */
    /* without an explicit NVM_UNMAP_HOST_MEMORY.                         */
    /*                                                                    */
    /* Proof points:                                                      */
    /*   - dmesg shows "destroy_qgroup id=N drained 1 map(s)"             */
    /*   - Re-issuing NVM_UNMAP_HOST_MEMORY for the same vaddr now       */
    /*     returns -EINVAL (the map is gone from the global host_list   */
    /*     too).                                                          */
    /* ------------------------------------------------------------------ */
    {
        uint32_t gid = group_d;
        if (do_ioctl(fd_a, NVM_DESTROY_QUEUE_GROUP, &gid,
                     "NVM_DESTROY_QUEUE_GROUP id=group_d") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP id=%u (B2 cascade)",
                      group_d);
        step_ok("NVM_DESTROY_QUEUE_GROUP id=%u cascades through 1 map "
                "(grep dmesg for 'destroy_qgroup id=%u drained 1 map')",
                group_d, group_d);
    }

    /* ------------------------------------------------------------------ */
    /* [18] Confirm the cascade actually freed the map: an explicit      */
    /* NVM_UNMAP_HOST_MEMORY for the same vaddr must now -EINVAL.        */
    /* ------------------------------------------------------------------ */
    {
        uint64_t vaddr = (uint64_t)(uintptr_t)host_buf;
        int r = ioctl(fd_a, NVM_UNMAP_HOST_MEMORY, &vaddr);
        if (r == 0)
            step_fail(0,
                "NVM_UNMAP_HOST_MEMORY succeeded after destroy_qgroup -- "
                "the cascade did NOT remove the map from the global list");
        if (errno != EINVAL)
            step_fail(errno, "expected EINVAL(%d) after cascade, got errno=%d",
                      EINVAL, errno);
        step_ok("post-cascade NVM_UNMAP_HOST_MEMORY -EINVAL confirms map "
                "was freed");
    }

    munmap(host_buf, (size_t)psz);

    /* ------------------------------------------------------------------ */
    /* [19] Cleanup chrdev so the test is rerunnable.                    */
    /*                                                                    */
    /* close(fd_a) must NOT see any leftover group (we destroyed group_a */
    /* in step 6 and group_d in step 17; steps 4/10 failed before        */
    /* allocation, so nothing was attached).  No "cascade-destroyed N"  */
    /* line should appear for fd_a in dmesg.                             */
    /* ------------------------------------------------------------------ */
    if (close(fd_a) < 0)
        step_fail(errno, "close(fd_a)");
    step_ok("close(fd_a)");

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf, "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE %s", bdf_str);
        step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);
    }
    close(fd_ctl);

    fprintf(stderr, "\n=== snvme_smoke_qgroup: all %d steps passed ===\n", g_step);
    return 0;
}
