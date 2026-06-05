/*
 * snvme_smoke_addq.c -- Smoke test for NVM_ADD_USER_QUEUE (queue-group
 * plan, step B3).
 *
 * Purpose
 * -------
 *
 * Validate the per-group user-queue creation path end-to-end on a
 * BOUND controller, without sending any NVMe IO commands.  We
 * deliberately do not even try to ring a doorbell here -- the goal
 * is to confirm that the kernel:
 *
 *   1. Rejects ADD_USER_QUEUE before bind with -ENODEV.
 *   2. Returns plausible q_depth / max_user_qid / bar0_size from
 *      the new NVM_GET_DEV_INFO ABI.
 *   3. Pipes the (group_id, sq_vaddr, cq_vaddr) batch through the
 *      Create I/O CQ + Create I/O SQ admin path on the controller,
 *      and returns BAR0 doorbell offsets back to userspace.
 *   4. Cascade-destroys all created user queues when the group is
 *      destroyed (NVM_DESTROY_QUEUE_GROUP), via the kernel's
 *      adapter_delete_sq + adapter_delete_cq path.
 *   5. Leaves the controller in a state where it can be unbound
 *      cleanly afterwards.
 *
 * Out of scope (deferred to B4):
 *   - Issuing actual NVMe Read / Write commands through these queues.
 *   - GPU / nvfs ring memory; B3 smoke uses host pages only.
 *
 * Why a separate binary from snvme_smoke_qgroup:
 *   snvme_smoke_qgroup is the no-bind smoke that runs even on hosts
 *   where /dev/nvmeN holds a mounted filesystem; it cannot bind.
 *   B3 fundamentally requires bind, so it lives in its own binary
 *   that the operator has to opt into running.  Same split as
 *   snvme_smoke (UAPI-only) vs snvme_smoke_recycle (bind required).
 *
 * Build:    make snvme_smoke_addq
 * Invoke:   sudo ./snvme_smoke_addq <PCI_BDF>
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
 * 【函数】step_ok(fmt, ...)
 * 【作用】打印一条 "[ OK ] step=N ..." 的成功日志，并把全局步骤
 *         计数器 g_step 自增 1。是个像 printf 一样的变参函数。
 * 【参数】fmt 是格式化字符串（同 printf）；后面跟可变参数，填进 fmt。
 * 【返回】无返回值（void）。只往 stderr 写日志，不会让程序退出。
 * 【在测试中的角色】每完成冒烟测试的一小步就调它一次，给操作员一条
 *         带编号的可读记录，方便对照哪一步过了。
 * 【新手提示】va_list / va_start / vfprintf 是 C 处理“参数个数不定”
 *         函数（变参函数）的标准三件套；vfprintf 就是接收 va_list 版
 *         的 fprintf。stderr 是标准错误流，这里所有日志都走它。
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
 * 【函数】step_fail(err, fmt, ...)
 * 【作用】打印一条 "[FAIL] step=N ... errno=E (描述)" 的失败日志，
 *         然后直接 exit(2) 终止整个进程——这一步出错就不再继续了。
 * 【参数】err 是要打印的 errno 值（0 表示“此处无 errno 可言”，会显示
 *         n/a）；fmt + 可变参数同 printf，描述失败原因。
 * 【返回】不返回。函数带 __attribute__((noreturn))，告诉编译器它绝不
 *         会执行到结尾（因为里面会 exit(2)）。退出码 2 = 某步冒烟失败。
 * 【在测试中的角色】测试里所有“断言失败 / ioctl 失败”的统一出口，
 *         保证一旦出错立刻停下并给出清晰原因。
 * 【新手提示】errno 是 C/Unix 里系统调用失败后设置的全局错误码；
 *         strerror(errno) 把它翻成人话（如 "No such device"）。
 *         noreturn 让编译器知道调用它之后的代码不可达，避免误报警告。
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

/* ────────────────────────────────────────────────────────────
 * 【函数】parse_bdf(s, out)
 * 【作用】把命令行传进来的 PCI 地址字符串（形如 "0000:50:00.0"）
 *         解析成结构体 pci_device_addr 的四个字段。
 * 【参数】s   输入字符串，格式 DDDD:BB:DD.F（域:总线:槽.功能，十六进制）；
 *         out 输出，解析出的 domain/bus/slot/func 写进它的四个成员。
 * 【返回】成功（恰好读到 4 个字段）返回 0；否则返回 -1。
 * 【在测试中的角色】程序启动时第一步，把用户给的 BDF 文本变成内核 ioctl
 *         能用的二进制地址；解析失败就直接报“Bad BDF”退出。
 * 【新手提示】BDF = Bus/Device(Slot)/Function，是 PCIe 设备在系统里的
 *         唯一定位；前面再加一个 4 位 domain 段。sscanf 的 "%x" 表示按
 *         十六进制读，返回值是“成功匹配并赋值的字段个数”，所以这里判 ==4。
 * ──────────────────────────────────────────────────────────── */
static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ────────────────────────────────────────────────────────────
 * 【函数】do_ioctl(fd, req, arg, what)
 * 【作用】对内核驱动发起一次 ioctl 系统调用的薄封装；如果失败，先把
 *         出错的命令名打到 stderr，再把 errno 原样恢复后返回。
 * 【参数】fd   已打开的设备文件描述符（如 /dev/ssnvmeN 或 /dev/snvm_control）；
 *         req  ioctl 命令号（如 NVM_ADD_USER_QUEUE 这些宏）；
 *         arg  指向与该命令配套的参数结构体的指针；
 *         what 命令的可读名字，仅用于出错日志。
 * 【返回】透传 ioctl 的返回值：成功通常为 0，失败为负数（同时 errno 有效）。
 * 【在测试中的角色】几乎所有跟内核交互都走它，统一了“失败先打印再返回”
 *         的行为，省去每个调用点都写一遍报错。
 * 【新手提示】ioctl 是 Unix 里“给设备下达自定义控制命令”的通用入口；
 *         它在内核态可能改写 errno，这里特意把 errno 暂存再恢复，避免
 *         中间的 fprintf 把 errno 覆盖掉，保证调用方拿到真正的错误码。
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

/* ────────────────────────────────────────────────────────────
 * 【函数】usage(prog)
 * 【作用】把本程序的用法说明打到 stderr：怎么调、参数是什么、并警告
 *         它会“绑定控制器（破坏性）”。
 * 【参数】prog 程序自己的名字（一般传 argv[0]），用来拼出示例命令行。
 * 【返回】无返回值。只打印帮助文本，不退出（退不退由 main 决定）。
 * 【在测试中的角色】参数个数不对、或用户传 --help 时显示帮助。
 * 【新手提示】“破坏性/destructive”指它会把目标 NVMe 控制器从原驱动夺过来
 *         绑到本测试驱动上，因此不能在挂着文件系统、正在使用的盘上跑。
 * ──────────────────────────────────────────────────────────── */
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <PCI_BDF>\n"
        "  e.g.: %s 0000:50:00.0\n"
        "\n"
        "BINDS the target controller (destructive).  Validates the\n"
        "B3 NVM_ADD_USER_QUEUE / NVM_DESTROY_QUEUE_GROUP cascade,\n"
        "without issuing any NVMe IO commands.\n",
        prog, prog);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】round_up_pages(n_bytes, page_size)
 * 【作用】把字节数 n_bytes 向上取整到 page_size 的整数倍（“凑整到整页”）。
 *         例如 page_size=4096 时，把 100 变成 4096，把 5000 变成 8192。
 * 【参数】n_bytes   原始字节数；page_size 一页的字节数（通常 4096）。
 * 【返回】>= n_bytes 的、能被 page_size 整除的最小值（字节数）。
 * 【在测试中的角色】给 SQ/CQ 环算实际要申请的、对齐到整页的大小；也用来
 *         算环占了几页（除以 page_size）。
 * 【新手提示】为什么 NVMe 的 SQ/CQ 环必须页对齐？因为创建 I/O 队列时，
 *         环的物理首地址通过 PRP1 这个字段交给控制器，而 PRP（物理区域页）
 *         寻址要求地址按页对齐——即低 12 位（4096=2^12）必须为 0。只有
 *         整页对齐、整页大小，控制器才能正确按页定位整个环。这里的取整就是
 *         为后面 posix_memalign 申请整页内存做准备。
 *         常用取整套路：(n + p - 1) / p * p。
 * ──────────────────────────────────────────────────────────── */
/*
 * Round n_bytes up to the nearest multiple of page_size.
 */
static size_t round_up_pages(size_t n_bytes, long page_size) {
    return ((n_bytes + page_size - 1) / page_size) * page_size;
}

/* ────────────────────────────────────────────────────────────
 * 【函数】alloc_ring(bytes, page_size)
 * 【作用】申请一块“页对齐 + 整页大小 + 已清零”的主机内存，专门用作
 *         一个 NVMe SQ 或 CQ 环的缓冲区。
 * 【参数】bytes     环需要的逻辑字节数（如 q_depth*64 或 q_depth*16）；
 *         page_size 一页字节数，用作对齐边界，并把 bytes 取整到整页。
 * 【返回】成功返回指向缓冲区的指针；失败（posix_memalign 出错）返回 NULL，
 *         调用方据此 step_fail。
 * 【在测试中的角色】Phase 4 给每个 SQ/CQ 环分配真实主机页，随后这块内存的
 *         虚拟地址会注册给内核、最终作为 I/O 队列的环交给控制器。
 * 【新手提示】为什么一定要 posix_memalign 而不能用普通 malloc？因为 NVMe
 *         创建 I/O SQ/CQ 时，环的物理首地址要填进 PRP1，而 PRP1 要求地址
 *         页对齐（低 12 位为 0）。malloc 不保证任何对齐，posix_memalign 能
 *         保证按 page_size 对齐，恰好满足“低 12 位为 0”。另外缓冲区清零很
 *         关键：CQ 环要靠 phase（相位）位判断条目是否是控制器新写的，首圈
 *         必须从 0 开始；清零也避免调试早期读到假的 NVMe 状态码。
 * ──────────────────────────────────────────────────────────── */
/*
 * Allocate a page-aligned host buffer suitable for use as an NVMe
 * SQ/CQ ring.  We MUST use posix_memalign (or aligned mmap) because
 * NVMe Create I/O SQ/CQ requires the ring's PRP1 to be page-aligned
 * (low 12 bits zero).  malloc() makes no such guarantee.
 *
 * The buffer is zeroed: the CQ ring's phase bit must start at 0
 * for the first lap, and the SQ ring is effectively don't-care but
 * zeroing it avoids false NVMe SC values during early debug.
 */
static void* alloc_ring(size_t bytes, long page_size) {
    void* p = NULL;
    size_t rounded = round_up_pages(bytes, page_size);
    if (posix_memalign(&p, page_size, rounded) != 0)
        return NULL;
    memset(p, 0, rounded);
    return p;
}

/* ────────────────────────────────────────────────────────────
 * 【函数】main(argc, argv)
 * 【作用】整个 B3 冒烟测试的主流程：从命令行拿到 PCI BDF，按 Phase 0~10
 *         一步步把“建组 → 绑定 → 取设备信息 → 分配/注册环 → 建用户队列 →
 *         做负面校验 → 销毁级联 → 解绑收尾”跑完一遍。中途任一步失败即
 *         step_fail 退出（码 2）。
 * 【参数】argc/argv：要求恰好一个参数，即目标控制器的 PCI BDF（如
 *         "0000:50:00.0"）；传 --help 或参数个数不对则打印用法。
 * 【返回】0=全部通过；1=用法错误；2=某步失败（由 step_fail 内部 exit）。
 *
 * 【完整流程（Phase 0~10）】
 *   Phase 0  打开控制面 /dev/snvm_control，用 SNVM_CHRDEV_CREATE 为该 BDF
 *            建出字符设备，再打开 /dev/ssnvmeN（N 是返回的 minor）。
 *   Phase 1  绑定之前先 NVM_CREATE_QUEUE_GROUP 建一个队列组，证明“组的
 *            生命周期与是否绑定无关”，并拿到 group_id。
 *   [4]      绑定之前对该组发 NVM_ADD_USER_QUEUE，必须返回 -ENODEV（控制器
 *            还没绑、没存活），验证内核的存活性检查在前。
 *   [4b]     NVM_SET_KERNEL_IOQ_CAP 把内核侧 IOQ 数量上限压到 36，给用户
 *            队列池留出足够的 QID，避免后面 Create I/O CQ 因 QID 用光而失败。
 *   Phase 2  SNVM_DEVICE_BIND 真正绑定控制器（破坏性操作）。
 *   Phase 3  轮询 NVM_GET_DEV_INFO 直到 probe 完成，读出 q_depth、bar0_size、
 *            max_user_qid、max_queues_per_group 等并做合理性断言。
 *   Phase 4  按 q_depth 算 SQ(=q_depth*64B)/CQ(=q_depth*16B) 大小，校验每个
 *            环不超过一页（单 PRP 限制），用 alloc_ring 分配 2 对 SQ+CQ 环。
 *   Phase 5  对每个环发 NVM_MAP_HOST_MEMORY，把它的虚拟地址登记到 group_id 下，
 *            供后续按 (group, vaddr) 反查。
 *   Phase 6  NVM_ADD_USER_QUEUE 批量提交 2 对 (sq_vaddr, cq_vaddr)，内核走
 *            Create I/O CQ + Create I/O SQ admin 路径真正建队列，返回每队列的
 *            qid 和 BAR0 上的 doorbell 偏移（断言非 0）。
 *   Phase 7  负面：再加“超过本组 max_queues_per_group”的队列，必须被拒，
 *            errno 为 EBUSY（超额）或 ENOENT（额度过了但 vaddr 查不到）。
 *   Phase 8  NVM_DESTROY_QUEUE_GROUP 销毁组，内核级联 Delete I/O SQ + CQ、
 *            释放 QID、清空所有 map。
 *   Phase 9  负面：对已销毁的 group_id 再 NVM_ADD_USER_QUEUE，必须 -ENOENT。
 *            随后 free 掉用户侧环内存（内核侧 map 已被销毁级联清掉）。
 *   Phase 10 收尾：SNVM_DEVICE_UNBIND 解绑、close 设备、SNVM_CHRDEV_REMOVE
 *            删字符设备、close 控制面，打印总通过数并返回 0。
 *
 * 【在测试中的角色】本文件的总驱动；本测试只验证“建/销用户队列”的控制路径，
 *         全程不发 NVMe 读写 IO、不敲 doorbell。
 * 【新手提示】doorbell（门铃）是 BAR0 寄存器空间里的一组寄存器，软件往里写
 *         队列尾/头索引来“通知”控制器有新命令或已消费完成项；本测试只取回
 *         偏移、不去写它。admin 路径指通过控制器的管理队列下发 Create/Delete
 *         I/O Queue 这类管理命令。
 * ──────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    if (argc != 2 || strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return argc == 2 ? 0 : 1;
    }
    const char* bdf_str = argv[1];

    struct pci_device_addr orig_bdf;
    if (parse_bdf(bdf_str, &orig_bdf) != 0) {
        fprintf(stderr, "Bad BDF: '%s' (expected DDDD:BB:DD.F)\n", bdf_str);
        return 1;
    }

    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0)
        step_fail(errno, "sysconf(_SC_PAGESIZE)");

    /* ============================================================== */
    /* Phase 0: bring up control plane + chrdev.                      */
    /* ============================================================== */
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
    int fd_dev = open(dev_path, O_RDWR | O_NONBLOCK);
    if (fd_dev < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ============================================================== */
    /* Phase 1: create a queue group BEFORE bind.                     */
    /*                                                                */
    /* This validates that group lifecycle is bind-agnostic AND       */
    /* that NVM_ADD_USER_QUEUE on a not-yet-bound controller          */
    /* returns -ENODEV (caught at controller liveness check).         */
    /* ============================================================== */
    uint32_t group_id;
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_dev, NVM_CREATE_QUEUE_GROUP, &req,
                     "NVM_CREATE_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_CREATE_QUEUE_GROUP");
        group_id = req.group_id;
        step_ok("NVM_CREATE_QUEUE_GROUP -> group_id=%u max_queues=%u",
                group_id, req.max_queues);
    }

    /* ------------------------------------------------------------------ */
    /* [4] Pre-bind: NVM_ADD_USER_QUEUE must -ENODEV.                    */
    /*                                                                    */
    /* We don't even bother allocating real ring buffers here -- the      */
    /* kernel checks bind status BEFORE looking up any maps, so a         */
    /* zeroed payload is fine for testing the rejection path.             */
    /* ------------------------------------------------------------------ */
    {
        struct nvm_ioctl_add_user_queue req;
        memset(&req, 0, sizeof(req));
        req.group_id = group_id;
        req.nr_pairs = 1;
        req.pairs[0].sq_vaddr = 0xfeed0000UL;  /* won't be inspected */
        req.pairs[0].cq_vaddr = 0xfeed4000UL;
        int r = ioctl(fd_dev, NVM_ADD_USER_QUEUE, &req);
        if (r == 0)
            step_fail(0,
                "NVM_ADD_USER_QUEUE unexpectedly succeeded on UNBOUND "
                "ctrl -- liveness check missing");
        if (errno != ENODEV)
            step_fail(errno,
                "NVM_ADD_USER_QUEUE on unbound ctrl: errno=%d, expected ENODEV(%d)",
                errno, ENODEV);
        step_ok("NVM_ADD_USER_QUEUE on unbound ctrl correctly returns -ENODEV");
    }

    /* ------------------------------------------------------------------ */
    /* [4b] Pre-bind: cap kernel-side IOQ count via NVM_SET_KERNEL_IOQ_CAP.*/
    /*                                                                    */
    /* Without this, probe defaults nvme_max_io_queues() to               */
    /* num_possible_cpus() + write_queues + poll_queues -- on a 192-vCPU  */
    /* host that asks the controller for ~192 IOQs.  Most NVMe SSDs grant */
    /* only as many as their MSI-X vector count allows (e.g. Intel DC SSD */
    /* MSI-X=136 -> grant=135), and the kernel then consumes every QID    */
    /* it was granted, leaving the [online_queues..ctrl_max_io_queues]    */
    /* user pool empty.  Create I/O CQ from NVM_ADD_USER_QUEUE later      */
    /* would fail with NVMe SC=0x4101 (Invalid Queue Identifier).         */
    /*                                                                    */
    /* Cap=36 is arbitrary but small: leaves ~99 QIDs in the user pool   */
    /* on a 135-grant controller, more than enough for any smoke run.    */
    /*                                                                    */
    /* NOTE: This is a B3 cap-only ioctl -- it sets                       */
    /* ctrl->setup.cap_kernel_ioq WITHOUT touching ctrl->ioq_num /        */
    /* use_sreg / on_host (which the legacy NVM_SET_IOQ_NUM would).  As   */
    /* such, probe still runs as plain in-tree-style nvme; only           */
    /* nvme_max_io_queues() is capped.                                    */
    /* ------------------------------------------------------------------ */
    {
        uint32_t cap = 36;
        if (ioctl(fd_dev, NVM_SET_KERNEL_IOQ_CAP, &cap) != 0)
            step_fail(errno, "NVM_SET_KERNEL_IOQ_CAP cap=%u failed", cap);
        step_ok("NVM_SET_KERNEL_IOQ_CAP cap=%u (kernel will get <=%u IOQs, "
                "rest of controller grant goes to user pool)", cap, cap);
    }

    /* ============================================================== */
    /* Phase 2: BIND.                                                  */
    /* ============================================================== */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_BIND, &bdf, "SNVM_DEVICE_BIND") < 0)
            step_fail(errno,
                "SNVM_DEVICE_BIND %s -- in-tree nvme may still own this BDF; "
                "try `sudo sh -c 'echo %s > /sys/bus/pci/drivers/nvme/unbind'` "
                "first", bdf_str, bdf_str);
        step_ok("SNVM_DEVICE_BIND %s", bdf_str);
    }

    /* ============================================================== */
    /* Phase 3: NVM_GET_DEV_INFO -- read the controller-derived       */
    /* ring sizing constraints.  Poll until probe completes.          */
    /* ============================================================== */
    struct nvm_ioctl_dev info;
    {
        int ok = 0;
        for (int i = 0; i < 100; i++) {
            memset(&info, 0, sizeof(info));
            if (ioctl(fd_dev, NVM_GET_DEV_INFO, &info) == 0 &&
                info.disk_name[0] != '\0') {
                ok = 1;
                break;
            }
            usleep(100 * 1000);
        }
        if (!ok)
            step_fail(errno, "NVM_GET_DEV_INFO did not complete within 10s");
        step_ok("NVM_GET_DEV_INFO disk='%s' block_size=%zu",
                info.disk_name, info.block_size);
    }

    /* Sanity-check the new B3 fields. */
    if (info.q_depth == 0)
        step_fail(0, "NVM_GET_DEV_INFO: q_depth == 0 (kernel bug)");
    if (info.bar0_size < 4096)
        step_fail(0, "NVM_GET_DEV_INFO: bar0_size=%u suspiciously small",
                  info.bar0_size);
    if (info.max_user_qid <= info.start_cq_idx && info.start_cq_idx != 0)
        /* It's OK if both are zero on a controller that gave us no
         * room for user queues at all -- the next add will -EBUSY. */
        step_fail(0, "NVM_GET_DEV_INFO: max_user_qid (%u) <= start_cq_idx (%u); "
                     "no room for user queues",
                  info.max_user_qid, info.start_cq_idx);
    if (info.max_queues_per_group == 0)
        step_fail(0, "NVM_GET_DEV_INFO: max_queues_per_group == 0");
    step_ok("dev_info q_depth=%u bar0_size=0x%x max_user_qid=%u "
            "max_queues_per_group=%u start_cq_idx=%u",
            info.q_depth, info.bar0_size, info.max_user_qid,
            info.max_queues_per_group, info.start_cq_idx);

    /* ============================================================== */
    /* Phase 4: Allocate two SQ+CQ ring pairs (we'll create 2 user    */
    /* queues in this test so the rollback / cur_queues math gets a   */
    /* multi-queue exercise).                                          */
    /*                                                                 */
    /* Sizing math:                                                    */
    /*   SQ bytes = q_depth * 64                                       */
    /*   CQ bytes = q_depth * 16                                       */
    /*   Both must be page-aligned (PRP1 has low 12 bits = 0).         */
    /*   Both must fit in one host page (snvme single-PRP limit).     */
    /*                                                                 */
    /*   For q_depth=1024:                                             */
    /*     SQ = 64K bytes = 16 pages -- DOES NOT FIT in one page!     */
    /*                                                                 */
    /*   To stay safe we cap our test to q_depth' = min(64,            */
    /*   info.q_depth) for the SQ (4096 / 64 = 64 entries) and        */
    /*   q_depth' = min(256, info.q_depth) for the CQ (4096 / 16 =    */
    /*   256 entries).  NVMe spec lets a CQ be deeper than its SQ;     */
    /*   we set both QSIZE in the kernel side to dev->q_depth-1, but  */
    /*   we don't have to USE all the entries.  The actual requirement */
    /*   is just that the buffer covers (q_depth * entry_size) bytes  */
    /*   -- the controller will cap to whatever buffer size implies.  */
    /*                                                                 */
    /* In practice the kernel uses dev->q_depth verbatim in           */
    /* adapter_alloc_*_user, so the test ring MUST satisfy the full   */
    /* q_depth * entry_size.  If that exceeds one page, we fail        */
    /* loudly here rather than on Create I/O CQ.                       */
    /* ============================================================== */
    const unsigned NR_PAIRS = 2;
    const size_t   sqe_size = 64;
    const size_t   cqe_size = 16;
    size_t sq_bytes = (size_t)info.q_depth * sqe_size;
    size_t cq_bytes = (size_t)info.q_depth * cqe_size;
    size_t sq_pages = round_up_pages(sq_bytes, psz) / psz;
    size_t cq_pages = round_up_pages(cq_bytes, psz) / psz;

    if (sq_pages > 1)
        step_fail(0,
            "B3 single-PRP limit: q_depth=%u * sqe=64 = %zu B requires "
            "%zu pages; the snvme NVM_MAP_HOST_MEMORY -> "
            "adapter_alloc_sq_user path only uses addrs[0], so a "
            "multi-page ring would not be physically contiguous from "
            "the controller's POV.  Lower io_queue_depth at insmod time "
            "(or extend snvme to chain PRP).",
            info.q_depth, sq_bytes, sq_pages);
    if (cq_pages > 1)
        step_fail(0,
            "B3 single-PRP limit: q_depth=%u * cqe=16 = %zu B requires "
            "%zu pages; same constraint as above.",
            info.q_depth, cq_bytes, cq_pages);

    void* sq_buf[NR_PAIRS];
    void* cq_buf[NR_PAIRS];
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        sq_buf[i] = alloc_ring(sq_bytes, psz);
        cq_buf[i] = alloc_ring(cq_bytes, psz);
        if (!sq_buf[i] || !cq_buf[i])
            step_fail(errno, "alloc_ring pair %u", i);
    }
    step_ok("allocated %u SQ+CQ ring pairs (sq=%zu B/page, cq=%zu B/page)",
            NR_PAIRS, sq_bytes, cq_bytes);

    /* ============================================================== */
    /* Phase 5: NVM_MAP_HOST_MEMORY each ring against group_id (new   */
    /* mode).  Throw-away ioaddr buffer just to satisfy the ABI; we   */
    /* don't need to remember it -- the kernel will look it up via    */
    /* (group, vaddr) in NVM_ADD_USER_QUEUE.                           */
    /* ============================================================== */
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        uint64_t throwaway[1];
        struct nvm_ioctl_map req;

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)sq_buf[i];
        req.n_pages     = 1;
        req.ioaddrs     = throwaway;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_id;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(SQ)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY pair %u SQ", i);

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)cq_buf[i];
        req.n_pages     = 1;
        req.ioaddrs     = throwaway;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_id;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(CQ)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY pair %u CQ", i);
    }
    step_ok("NVM_MAP_HOST_MEMORY x %u ring pairs registered against group=%u",
            NR_PAIRS * 2, group_id);

    /* ============================================================== */
    /* Phase 6: NVM_ADD_USER_QUEUE -- the actual Create I/O CQ + SQ. */
    /* ============================================================== */
    struct nvm_ioctl_add_user_queue add_req;
    memset(&add_req, 0, sizeof(add_req));
    add_req.group_id = group_id;
    add_req.nr_pairs = NR_PAIRS;
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        add_req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t)sq_buf[i];
        add_req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t)cq_buf[i];
    }

    if (do_ioctl(fd_dev, NVM_ADD_USER_QUEUE, &add_req,
                 "NVM_ADD_USER_QUEUE") < 0)
        step_fail(errno, "NVM_ADD_USER_QUEUE -- check dmesg for which "
                         "Create I/O CQ/SQ admin command failed");
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        if (add_req.out_pairs[i].sq_doorbell_offset == 0 ||
            add_req.out_pairs[i].cq_doorbell_offset == 0)
            step_fail(0,
                "NVM_ADD_USER_QUEUE pair %u: kernel returned zero doorbell "
                "offset (sq=0x%x cq=0x%x), expected non-zero",
                i, add_req.out_pairs[i].sq_doorbell_offset,
                add_req.out_pairs[i].cq_doorbell_offset);
    }
    step_ok("NVM_ADD_USER_QUEUE created %u user queue(s):", NR_PAIRS);
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        fprintf(stderr, "             pair[%u] qid=%u sq_db=0x%x cq_db=0x%x\n",
                i, add_req.out_pairs[i].qid,
                add_req.out_pairs[i].sq_doorbell_offset,
                add_req.out_pairs[i].cq_doorbell_offset);
    }

    /* ============================================================== */
    /* Phase 7: Negative -- ADD_USER_QUEUE filling the group beyond  */
    /* max_queues must -EBUSY.                                         */
    /* ============================================================== */
    {
        unsigned remaining = info.max_queues_per_group - NR_PAIRS;
        if (remaining > 0 && remaining < NVM_MAX_QUEUES_PER_GROUP) {
            /*
             * Try to add (remaining + 1) more pairs; the +1 must
             * push us over the cap.  We don't actually want them to
             * succeed, so we deliberately use stale vaddrs that
             * *would* otherwise -ENOENT on map lookup -- but the
             * cur_queues+nr_pairs > max_queues check happens
             * BEFORE map lookup, so we'll hit -EBUSY instead.
             *
             * The +1 also has to stay within
             * NVM_MAX_QUEUES_PER_GROUP itself (the per-call upper
             * bound), so we cap at that.
             */
            struct nvm_ioctl_add_user_queue req;
            unsigned over = remaining + 1;
            if (over > NVM_MAX_QUEUES_PER_GROUP)
                over = NVM_MAX_QUEUES_PER_GROUP;
            memset(&req, 0, sizeof(req));
            req.group_id = group_id;
            req.nr_pairs = over;
            for (unsigned i = 0; i < over; i++) {
                req.pairs[i].sq_vaddr = 0xdeadbeef0000UL + i * 0x2000;
                req.pairs[i].cq_vaddr = 0xdeadbeef0000UL + i * 0x2000 + 0x1000;
            }
            int r = ioctl(fd_dev, NVM_ADD_USER_QUEUE, &req);
            if (r == 0)
                step_fail(0,
                    "NVM_ADD_USER_QUEUE overflow (%u pairs into group with "
                    "max=%u, cur=%u) unexpectedly succeeded",
                    over, info.max_queues_per_group, NR_PAIRS);
            if (errno != EBUSY && errno != ENOENT)
                step_fail(errno,
                    "NVM_ADD_USER_QUEUE overflow: errno=%d, expected EBUSY "
                    "(cap exceeded) or ENOENT (cap check passed but vaddr "
                    "lookup failed)", errno);
            step_ok("NVM_ADD_USER_QUEUE overflow correctly rejected (errno=%s)",
                    errno == EBUSY ? "EBUSY" : "ENOENT");
        } else {
            step_ok("skip overflow check (max_queues_per_group=%u, NR_PAIRS=%u "
                    "leaves no room or full room)",
                    info.max_queues_per_group, NR_PAIRS);
        }
    }

    /* ============================================================== */
    /* Phase 8: Destroy group -- cascade must Delete I/O SQ + CQ for */
    /* every queue we created, free the QIDs, and drain all maps.    */
    /* ============================================================== */
    {
        uint32_t gid = group_id;
        if (do_ioctl(fd_dev, NVM_DESTROY_QUEUE_GROUP, &gid,
                     "NVM_DESTROY_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP");
        step_ok("NVM_DESTROY_QUEUE_GROUP id=%u cascades through %u user "
                "queue(s) + %u maps (grep dmesg for "
                "'destroy_qgroup id=%u drained ... user queue(s)')",
                group_id, NR_PAIRS, NR_PAIRS * 2, group_id);
    }

    /* ============================================================== */
    /* Phase 9: After destroy, ADD_USER_QUEUE against the same gid    */
    /* must -ENOENT (group is gone).                                   */
    /* ============================================================== */
    {
        struct nvm_ioctl_add_user_queue req;
        memset(&req, 0, sizeof(req));
        req.group_id = group_id;
        req.nr_pairs = 1;
        req.pairs[0].sq_vaddr = (uint64_t)(uintptr_t)sq_buf[0];
        req.pairs[0].cq_vaddr = (uint64_t)(uintptr_t)cq_buf[0];
        int r = ioctl(fd_dev, NVM_ADD_USER_QUEUE, &req);
        if (r == 0)
            step_fail(0,
                "NVM_ADD_USER_QUEUE against destroyed group %u "
                "unexpectedly succeeded", group_id);
        if (errno != ENOENT)
            step_fail(errno,
                "NVM_ADD_USER_QUEUE against destroyed group: errno=%d, "
                "expected ENOENT(%d)", errno, ENOENT);
        step_ok("NVM_ADD_USER_QUEUE against destroyed group correctly "
                "returns -ENOENT");
    }

    /* Free user-side ring buffers.  Note: the snvme-side maps were
     * already drained by NVM_DESTROY_QUEUE_GROUP above, so we don't
     * call NVM_UNMAP_HOST_MEMORY -- it would just -EINVAL. */
    for (unsigned i = 0; i < NR_PAIRS; i++) {
        free(sq_buf[i]);
        free(cq_buf[i]);
    }

    /* ============================================================== */
    /* Phase 10: Tear-down -- unbind, close, remove chrdev.            */
    /* ============================================================== */
    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_UNBIND, &bdf, "SNVM_DEVICE_UNBIND") < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND %s", bdf_str);
        step_ok("SNVM_DEVICE_UNBIND %s", bdf_str);
    }

    if (close(fd_dev) < 0)
        step_fail(errno, "close(%s)", dev_path);
    step_ok("close(%s)", dev_path);

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf, "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE %s", bdf_str);
        step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);
    }
    close(fd_ctl);

    fprintf(stderr, "\n=== snvme_smoke_addq: all %d steps passed ===\n", g_step);
    return 0;
}
