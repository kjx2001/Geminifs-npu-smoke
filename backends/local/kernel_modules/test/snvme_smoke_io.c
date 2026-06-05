/*
 * snvme_smoke_io.c -- End-to-end NVMe Read/Write IO smoke test on user
 * IO queues created via NVM_ADD_USER_QUEUE.
 *
 * Builds on snvme_smoke_addq's bring-up (B1+B2+B3 ioctl plumbing) by
 * actually issuing NVMe Read / Write commands on the resulting user
 * IOQs and verifying the round-trip data integrity.  Goal: prove that
 *
 *   1. The doorbell offset returned by NVM_ADD_USER_QUEUE is correct
 *      and writeable from user space (after BAR0 mmap).
 *   2. The user-side SQ ring placement (vaddr -> dma_addr from
 *      NVM_MAP_HOST_MEMORY) is what the controller reads SQEs from.
 *   3. The user-side CQ ring receives CQEs with the right phase bit
 *      and command_id echo.
 *   4. Data PRP1 (also via NVM_MAP_HOST_MEMORY) round-trips: a Write
 *      command lands the host-side bytes on the device, and a Read
 *      command on the same LBA brings them back unchanged.
 *   5. Multiple IOs per queue work (covers SQ tail wrap-around within
 *      q_depth, and CQ phase flips).
 *   6. Multiple queues coexist (qid=37 for writes, qid=38 for reads)
 *      without interfering with each other's CQE delivery.
 *
 * DESTRUCTIVE: writes to LBAs starting at TEST_LBA_BASE (default
 * 2621440 = 10 GiB / 4 KiB).  Caller has confirmed the target NVMe
 * is empty / disposable.
 *
 * Build:    make snvme_smoke_io
 * Invoke:   sudo ./snvme_smoke_io <PCI_BDF>
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
#include <sched.h>
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
/* NVMe spec constants (subset; we don't pull in the kernel's nvme.h) */
/* ------------------------------------------------------------------ */

#define NVME_OPC_FLUSH            0x00
#define NVME_OPC_WRITE            0x01
#define NVME_OPC_READ             0x02

#define NVME_SQE_SIZE             64u
#define NVME_CQE_SIZE             16u

/* Test parameters.  TEST_LBA_BASE = 10 GiB on a 4-KiB-LBA disk; well
 * past any partition table the user might add later.  Bumping this
 * up is fine; lower than ~1 GiB risks colliding with GPT/superblock
 * playgrounds that some platforms autocreate.                          */
#define TEST_LBA_BASE             2621440ULL   /* 10 GiB / 4 KiB */
#define TEST_NR_QUEUES            2u
#define TEST_NR_IO_PER_QUEUE      16u

/* Default opcode-specific dataword pattern.  Must be byte-stable so we
 * can also verify on big-endian, even though x86 is the only platform
 * we test on right now.                                                */
#define WRITE_PATTERN_BYTE(qid, ioidx) \
    ((uint8_t)(0xA5 ^ ((qid) & 0xff) ^ ((ioidx) & 0xff)))

/* ------------------------------------------------------------------ */
/* NVMe submission queue entry (Common Format, NVMe 1.4 figure 105).  */
/* We construct these by hand because libnvm's nvm_cmd_t is C++-       */
/* templated and would drag in CUDA headers; the smoke is libc-only.   */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【结构体】struct nvme_sqe —— NVMe 提交队列条目（Submission Queue Entry）
 * 【作用】用大白话说，这就是一条“硬盘命令单”。CPU 把要做的事（读/写哪个
 *         LBA、数据放在哪块内存）按 NVMe 规范规定的二进制格式填进这 64 字节，
 *         再拷进 SQ 环里，控制器（NVMe SSD 固件）就会读走并执行。
 * 【字段】NVMe 把命令拆成 16 个 32 位“命令双字”CDW0..CDW15：
 *   - opcode  ：操作码。0x01=Write，0x02=Read，0x00=Flush（见上方宏）。
 *   - flags   ：CDW0 高位里的标志字节。本测试用它存 PSDT 位（数据指针类型，
 *               PRP 还是 SGL，见下面 NVME_FLAG_PSDT_* 宏）。
 *   - cid     ：command_id，命令编号。完成时控制器会在 CQE 里原样回显它，
 *               让我们能把“哪条完成对应哪条提交”对上号。
 *   - nsid    ：namespace id，命名空间编号（snvme 暴露的是 ns 1）。
 *   - rsvd_2_3：CDW2-3 保留。
 *   - metadata：元数据指针（本测试不用，置 0）。
 *   - prp1    ：CDW6-7。数据缓冲指针 1。PRP 模式下是第一个数据页的 DMA 地址
 *               （可带页内偏移）；SGL 模式下这里塞 SGL 描述符的低 64 位。
 *   - prp2    ：CDW8-9。数据缓冲指针 2。PRP 模式下可能是第二个数据页的 DMA
 *               地址、或一张“PRP List”页的地址、或 0（数据只占一页时）；
 *               SGL 模式下塞 SGL 描述符的高 64 位。
 *   - cdw10..cdw15：操作码相关。对 Read/Write：CDW10-11=SLBA（起始 LBA，
 *               64 位），CDW12 低 16 位=NLB（块数，0 表示 1 块），其余是
 *               保护信息/DSM 等本测试不用的字段。
 * 【在测试中的角色】tq_submit_rw() 就是按这个布局手工填好一条 SQE 再发出去的。
 * 【新手提示】PRP=Physical Region Page，NVMe 描述数据缓冲位置的方式，按物理
 *   页一页一页地指。__attribute__((packed)) + 下面的 _Static_assert 保证它
 *   恰好 64 字节、字段紧凑无填充，跟硬件期望的字节布局逐字节一致。
 * ──────────────────────────────────────────────────────────── */
struct nvme_sqe {
    /* CDW0 */
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t cid;
    /* CDW1 */
    uint32_t nsid;
    /* CDW2-3 */
    uint64_t rsvd_2_3;
    /* CDW4-5 */
    uint64_t metadata;
    /* CDW6-7 */
    uint64_t prp1;
    /* CDW8-9 */
    uint64_t prp2;
    /* CDW10-15: opcode-specific.  For Read/Write:
     *   CDW10-11 = SLBA (64-bit)
     *   CDW12    = NLB (low 16) | reserved | PRINFO | FUA | LR
     *   CDW13    = DSM
     *   CDW14    = ELBST/ILBRT
     *   CDW15    = ELBATM/ELBAT
     */
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
} __attribute__((packed));

_Static_assert(sizeof(struct nvme_sqe) == NVME_SQE_SIZE,
               "nvme_sqe must be exactly 64 bytes");

/* ────────────────────────────────────────────────────────────
 * 【结构体】struct nvme_cqe —— NVMe 完成队列条目（Completion Queue Entry）
 * 【作用】这是控制器执行完一条命令后写回给我们的“回执”，固定 16 字节。
 *         控制器把它写进 CQ 环，我们在用户态轮询读取，从而知道命令做完了、
 *         做成功没有、对应的是哪条命令。
 * 【字段】
 *   - result  ：DW0，命令相关的返回值（多数 Read/Write 用不到）。
 *   - rsvd    ：DW1，保留。
 *   - sq_head ：DW2 低 16 位。控制器告诉我们它已经消费到 SQ 的哪个位置
 *               （SQ head 指针），用于流控（防止我们覆盖未读的 SQE）。
 *   - sq_id   ：DW2 高 16 位。这条完成属于哪个提交队列。
 *   - cid     ：DW3 低 16 位。原样回显当初 SQE 里的 command_id，我们用它
 *               核对“提交—完成”配对是否正确。
 *   - status  ：DW3 高 16 位。最关键的状态字：
 *               * bit[0]   = phase bit（相位位），用来判断这一格是不是本圈
 *                            刚写进来的“新”CQE（详见 struct test_queue）。
 *               * bit[8:1] = SC（Status Code，状态码），0 表示成功。
 *               * bit[11:9]= SCT（Status Code Type，状态码类型）。
 * 【在测试中的角色】tq_poll_one() 轮询读它的 phase 位判断是否完成，format_status()
 *   解析它的 SC/SCT 来报错，主流程比对它的 cid。
 * 【新手提示】“轮询 CQ ring 的 phase bit”就是反复读 status 的 bit0，直到它
 *   翻转成我们期待的值，说明控制器刚写了新回执——这是无中断模式下知道
 *   IO 完成的标准手段。
 * ──────────────────────────────────────────────────────────── */
/* NVMe completion queue entry (NVMe 1.4 figure 39). */
struct nvme_cqe {
    uint32_t result;        /* DW0: command-specific */
    uint32_t rsvd;          /* DW1: reserved */
    uint16_t sq_head;       /* DW2 lo */
    uint16_t sq_id;         /* DW2 hi */
    uint16_t cid;           /* DW3 lo */
    uint16_t status;        /* DW3 hi: phase bit in [0], SC in [8:1], SCT in [11:9] */
} __attribute__((packed));

_Static_assert(sizeof(struct nvme_cqe) == NVME_CQE_SIZE,
               "nvme_cqe must be exactly 16 bytes");

/* ------------------------------------------------------------------ */
/* Logging helpers                                                    */
/* ------------------------------------------------------------------ */

static int g_step = 0;

/* ────────────────────────────────────────────────────────────
 * 【函数】step_ok
 * 【作用】打印一条“某个测试步骤通过”的日志（[ OK ] step=N ...），并把全局
 *         步骤计数器 g_step 加 1。属于可变参数的格式化日志（像 printf）。
 * 【参数】fmt + 后续可变参数：跟 printf 一样的格式串和实参，输出到 stderr。
 * 【返回】无。
 * 【在测试中的角色】每完成一个阶段就调一次，给人看进度；step 编号方便定位。
 * 【新手提示】va_list/va_start/vfprintf 是 C 处理“…”可变参数的标准套路。
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
 * 【函数】step_fail
 * 【作用】打印一条“某步骤失败”的日志（[FAIL] step=N ...），附带 errno 和它
 *         对应的文字说明，然后直接 exit(2) 终止整个测试程序。
 * 【参数】
 *   - err ：错误码（通常是失败时记下来的 errno；传 0 表示“无 errno”）。
 *   - fmt + 可变参数：失败原因的格式化描述。
 * 【返回】不返回——带 __attribute__((noreturn))，调用后进程就退出了。
 * 【在测试中的角色】端到端测试的统一“失败即停”出口；任何一步对不上就在这里
 *   报清楚是哪一步、什么 errno，退出码固定 2（见文件头说明）。
 * 【新手提示】strerror(err) 把数字 errno 翻成人话（如 “Bad address”）。
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
 * 【函数】parse_bdf
 * 【作用】把命令行里形如 "0000:08:00.0" 的 PCI 地址字符串解析成结构体的四个
 *         数字字段（domain:bus:slot.func，即 PCI 设备的“身份证号”BDF）。
 * 【参数】
 *   - s   ：输入字符串，格式 DDDD:BB:DD.F（十六进制）。
 *   - out ：解析结果写到这里（domain/bus/slot/func 四个字段）。
 * 【返回】0=成功解析出全部 4 段；-1=格式不对。
 * 【在测试中的角色】main() 开头把用户给的 BDF 转成内核 ioctl 需要的结构体。
 * 【新手提示】BDF = Bus:Device(Slot).Function，唯一标识一块 PCIe 设备；
 *   sscanf 返回成功匹配的字段个数，这里要求正好 4 个。
 * ──────────────────────────────────────────────────────────── */
static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ────────────────────────────────────────────────────────────
 * 【函数】do_ioctl
 * 【作用】对 ioctl() 的一层薄封装：调用失败时自动打印“哪个 ioctl 失败 + 原因”，
 *         并小心地把 errno 原样保留下来给调用方用。
 * 【参数】
 *   - fd   ：要操作的文件描述符（/dev/snvm_control 或 /dev/ssnvmeN）。
 *   - req  ：ioctl 命令号（如 NVM_ADD_USER_QUEUE 等宏）。
 *   - arg  ：指向命令参数结构体的指针，内核会读/写它。
 *   - what ：人类可读的命令名字，仅用于出错时打日志。
 * 【返回】透传 ioctl 的返回值：0/正数=成功，<0=失败（errno 已被保留）。
 * 【在测试中的角色】贯穿全程，所有跟内核驱动打交道的 ioctl 都走它，省去
 *   每处重复写错误打印。
 * 【新手提示】ioctl 是用户态向设备驱动“下达带参数指令”的通用入口；fprintf
 *   可能会改写 errno，所以这里先存 e、打完日志再 errno=e 还原。
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
 * 【函数】usage
 * 【作用】打印命令行用法说明到 stderr，并强调这是“破坏性”操作（会真的往盘上
 *         写数据），告知会写多少个 LBA、从哪个 LBA 起。
 * 【参数】prog：程序名（argv[0]），用于拼出示例命令。
 * 【返回】无。
 * 【在测试中的角色】参数个数不对或用户加了 --help 时调用，提示正确用法。
 * 【新手提示】LBA = Logical Block Address，硬盘上的逻辑块编号；本测试从
 *   TEST_LBA_BASE（10 GiB 处）开始写，确认目标盘可随意覆盖才能跑。
 * ──────────────────────────────────────────────────────────── */
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s <PCI_BDF>\n"
        "  e.g.: %s 0000:08:00.0\n"
        "\n"
        "BINDS the target controller and writes/reads %llu LBAs starting at\n"
        "LBA %llu (=10 GiB on a 4 KiB block device).  DESTRUCTIVE.\n",
        prog, prog,
        (unsigned long long)(TEST_NR_QUEUES * TEST_NR_IO_PER_QUEUE),
        (unsigned long long)TEST_LBA_BASE);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】round_up_pages
 * 【作用】把字节数 n_bytes 向上取整到 page_size 的整数倍（向上对齐到整页）。
 * 【参数】
 *   - n_bytes  ：原始字节数。
 *   - page_size：页大小（通常 4096 字节）。
 * 【返回】对齐后的字节数（>= n_bytes 的最小整页倍数）。
 * 【在测试中的角色】分配 SQ/CQ 环、数据缓冲前算实际要 mmap/对齐的大小；也用来
 *   检查一个环是否会跨页（B3 单 PRP 限制要求环只占一页）。
 * 【新手提示】公式 (n + p - 1) / p * p 是整数向上取整到 p 倍的经典写法。
 * ──────────────────────────────────────────────────────────── */
/*
 * Round n_bytes up to the nearest multiple of page_size.
 */
static size_t round_up_pages(size_t n_bytes, long page_size) {
    return ((n_bytes + page_size - 1) / page_size) * page_size;
}

/* ────────────────────────────────────────────────────────────
 * 【函数】alloc_aligned
 * 【作用】分配一块“按页对齐”的主机内存（首地址是 page_size 的整数倍），并清零。
 *         适合当作 NVMe 的 SQ/CQ 环或数据页用。
 * 【参数】
 *   - bytes    ：需要的字节数（内部会先用 round_up_pages 向上取整到整页）。
 *   - page_size：对齐粒度（页大小）。
 * 【返回】成功返回对齐且清零的缓冲指针；失败返回 NULL。
 * 【在测试中的角色】所有要交给控制器 DMA 的内存（环、wbuf/rbuf、PRP List 页）
 *   都用它分配——必须页对齐，因为 PRP/环都是按物理页寻址的。
 * 【新手提示】posix_memalign 保证返回地址按指定边界对齐；NVMe 的 PRP 要求数据
 *   缓冲的 DMA 地址页对齐（PRP2 和 PRP List 尤其是低 12 位必须为 0）。
 * ──────────────────────────────────────────────────────────── */
/*
 * Allocate a page-aligned host buffer suitable for use as an NVMe
 * SQ/CQ ring or PRP1 data page.
 */
static void* alloc_aligned(size_t bytes, long page_size) {
    void* p = NULL;
    size_t rounded = round_up_pages(bytes, page_size);
    if (posix_memalign(&p, page_size, rounded) != 0)
        return NULL;
    memset(p, 0, rounded);
    return p;
}

/* ------------------------------------------------------------------ */
/* Doorbell writes.  BAR0 is mapped WC/UC (snvme uses pgprot_noncached)*/
/* so a plain volatile store reaches the controller without caching;   */
/* we still issue an sfence first to flush any prior WC stores into    */
/* the SQ ring before the controller sees the new tail.                */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】mmio_writel
 * 【作用】向一个 MMIO 地址（映射到 BAR0 的 doorbell 寄存器）写一个 32 位值，
 *         写之前先插一道内存屏障。这就是“敲门铃（ring doorbell）”的动作。
 * 【参数】
 *   - addr ：volatile uint32_t* 指针，指向 mmap 进来的 BAR0 里某个 doorbell。
 *   - value：要写入的值（SQ doorbell 写新的 tail；CQ doorbell 写新的 head）。
 * 【返回】无。
 * 【在测试中的角色】tq_submit_rw 写完 SQE 后用它敲 SQ doorbell 通知控制器“有
 *   新命令了”；tq_poll_one 消费完 CQE 后用它敲 CQ doorbell 通知“我读到这了”。
 * 【新手提示】
 *   - doorbell（门铃）：BAR0 里的寄存器，CPU 写它来告诉控制器队列指针动了。
 *   - MMIO：把设备寄存器映射进内存地址空间，用普通访存指令读写。BAR0 被
 *     映射成不可缓存（noncached）的，所以一个 volatile 写就能直达设备。
 *   - sfence：x86 的写屏障。先 sfence 再写 doorbell，保证我们刚拷进 SQ 环的
 *     SQE 内容“先于”doorbell 对设备可见——否则控制器可能被门铃唤醒去读
 *     一条还没写完的命令。非 x86 平台用 __atomic_thread_fence(RELEASE) 等效。
 * ──────────────────────────────────────────────────────────── */
static inline void mmio_writel(volatile uint32_t* addr, uint32_t value) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__ ("sfence" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
    *addr = value;
}

/* ------------------------------------------------------------------ */
/* Per-queue runtime state for the test driver.                       */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【结构体】struct test_queue —— 测试驱动里一对 SQ/CQ 队列的运行时状态
 * 【作用】把“操作一对 NVMe IO 队列所需的全部信息”打包：环的地址、当前指针、
 *         doorbell 地址、下一个命令号等。一个 test_queue 实例 = 一条可用队列。
 * 【字段】
 *   - qid     ：队列 ID（NVM_ADD_USER_QUEUE 建好后返回，如 37/38）。
 *   - q_depth ：队列深度，即环里有多少个槽位；指针在 [0, q_depth) 间回绕。
 *   - sq      ：SQ 环的主机虚拟地址。我们往这里写 SQE；控制器通过当初注册的
 *               DMA 地址来读它。
 *   - sq_tail ：SQ 尾指针。指向“下一条 SQE 要写入的槽位”。每提交一条就 +1
 *               （模 q_depth），并把新值写进 SQ doorbell。
 *   - cq      ：CQ 环的主机虚拟地址。控制器往这里写 CQE，我们轮询读。
 *   - cq_head ：CQ 头指针。指向“下一条要读取的 CQE 槽位”。每消费一条就 +1
 *               （模 q_depth），并把新值写进 CQ doorbell。
 *   - cq_phase：我们当前“期待”的 phase bit 值（0 或 1）。CQ 环初始清零，
 *               控制器第一圈把每格 phase 翻成 1；每当 cq_head 绕回 0（走完一
 *               整圈），cq_phase 取反。读到的 CQE.status.phase == cq_phase
 *               才算是本圈刚写入的新完成。这是无中断轮询判断“有没有新 CQE”
 *               的核心机制。
 *   - sq_db   ：SQ doorbell 寄存器指针，落在 mmap 进来的 BAR0 区域内。
 *   - cq_db   ：CQ doorbell 寄存器指针，同样在 BAR0 内。
 *   - next_cid：单调递增的命令号发号器，给每条新 SQE 分配 cid（不回收，
 *               q_depth 足够大用不完）。
 * 【在测试中的角色】tq_submit_rw / tq_poll_one 都围着它转：提交时动 sq_tail+敲
 *   sq_db，完成时读 cq_phase/动 cq_head+敲 cq_db。
 * 【新手提示】SQ tail（生产者写）与 CQ head（消费者读）是环形缓冲区的两个指针；
 *   doorbell 就是把这两个指针的新值“告知”硬件的途径。
 * ──────────────────────────────────────────────────────────── */
struct test_queue {
    uint16_t            qid;
    uint16_t            q_depth;

    /* SQ ring (host vaddr; controller reads via PRP1 dma_addr we
     * registered with NVM_MAP_HOST_MEMORY).                         */
    struct nvme_sqe*    sq;
    uint16_t            sq_tail;

    /* CQ ring (host vaddr; controller writes here, we poll). */
    struct nvme_cqe*    cq;
    uint16_t            cq_head;
    uint8_t             cq_phase;   /* expected next phase bit (0 or 1) */

    /* Doorbell pointers in mmap'd BAR0 region. */
    volatile uint32_t*  sq_db;
    volatile uint32_t*  cq_db;

    /* Monotonic command id source.  We don't bother recycling; q_depth
     * is plenty.                                                      */
    uint16_t            next_cid;
};

/* ────────────────────────────────────────────────────────────
 * 【函数】tq_submit_rw
 * 【作用】在指定队列上“提交”一条 Read 或 Write 命令。具体三步：①手工填好一条
 *         nvme_sqe；②把它拷进 SQ 环的 sq_tail 槽位；③推进 sq_tail 并敲 SQ
 *         doorbell 通知控制器。它是 CPU 亲手发 NVMe 命令的核心动作。
 * 【参数】
 *   - q             ：目标队列（提供环地址、sq_tail、sq_db、发号器）。
 *   - opcode        ：0x01=Write / 0x02=Read。
 *   - flags         ：CDW0 标志字节，主要承载 PSDT 位（PRP 还是 SGL）。
 *   - nsid          ：namespace id（本测试恒为 1）。
 *   - dptr0         ：填入 SQE.prp1 的值（PRP 模式=PRP1；SGL 模式=描述符低 64 位）。
 *   - dptr1         ：填入 SQE.prp2 的值（PRP 模式=PRP2/PRP List 页地址/0；
 *                     SGL 模式=描述符高 64 位）。调用方已按数据指针形式算好。
 *   - slba          ：起始 LBA（拆进 CDW10/11）。
 *   - nlb_zero_based：块数，0 表示 1 块（NVMe 的 NLB 是“个数减一”，填进 CDW12）。
 *   - cid_out       ：若非 NULL，回传本次分配的 command_id，供完成时核对。
 * 【返回】无（命令已写入环并敲过门铃；完成情况由 tq_poll_one 取）。
 * 【在测试中的角色】Phase 5~7 每发一条 IO 都调它；通过传不同的 dptr0/dptr1/flags
 *   覆盖 PRP1、PRP1+PRP2、PRP1+PRP List、SGL 各种数据指针形式。
 * 【新手提示】
 *   - “写 SQ 环”只是普通内存写；真正让控制器动起来的是随后的 doorbell。
 *   - doorbell 写的是“新的 tail”，语义是“到这个位置之前的都是新命令，请取走”。
 *   - mmio_writel 内含 sfence，确保 SQE 内容先于门铃对设备可见（顺序关键）。
 * ──────────────────────────────────────────────────────────── */
/*
 * Submit one Read/Write SQE on this queue.  Caller has already
 * computed the data pointer (PRP1/PRP2 or SGL1) and packed it into
 * dptr0/dptr1 along with the right CDW0 PSDT bits in `flags`.
 *
 * For PRP-style commands (PSDT=0):
 *   dptr0 = PRP1 (may have page offset)
 *   dptr1 = PRP2 (page-aligned data page, OR PRP List page address,
 *           OR 0 when transfer fits in one page)
 *   flags = 0
 *
 * For SGL-style data block (PSDT=01b):
 *   dptr0 = low 64 bits of the 16-byte SGL Data Block descriptor
 *           (i.e. the descriptor's `address` field)
 *   dptr1 = high 64 bits (length:32 | reserved:24 | type:8)
 *   flags = NVME_CMD_SGL_METABUF (0x40, PSDT=01b in CDW0)
 */
static void tq_submit_rw(struct test_queue* q,
                         uint8_t opcode,
                         uint8_t flags,
                         uint32_t nsid,
                         uint64_t dptr0,
                         uint64_t dptr1,
                         uint64_t slba,
                         uint16_t nlb_zero_based,
                         uint16_t* cid_out) {
    struct nvme_sqe sqe;
    memset(&sqe, 0, sizeof(sqe));

    uint16_t cid = q->next_cid++;
    sqe.opcode = opcode;
    sqe.flags  = flags;
    sqe.cid    = cid;
    sqe.nsid   = nsid;
    sqe.prp1   = dptr0;
    sqe.prp2   = dptr1;
    sqe.cdw10  = (uint32_t)(slba & 0xffffffffu);
    sqe.cdw11  = (uint32_t)(slba >> 32);
    sqe.cdw12  = nlb_zero_based & 0xffffu;

    /* Copy SQE into the ring at sq_tail. */
    q->sq[q->sq_tail] = sqe;

    /* Advance tail (modular).  Ring the doorbell with the NEW tail
     * value -- NVMe doorbell semantics is "this is where I have not
     * yet written".                                                  */
    uint16_t new_tail = (uint16_t)((q->sq_tail + 1) % q->q_depth);
    q->sq_tail = new_tail;
    mmio_writel(q->sq_db, new_tail);

    if (cid_out) *cid_out = cid;
}

/* NVMe spec PSDT bits, in CDW0[15:14] -- expressed as raw flags byte
 * (CDW0[7:0] = opcode, CDW0[15:8] = flags).  PSDT lives in [15:14] of
 * CDW0, i.e. bits [7:6] of the flags byte.
 *   00b = PRP, 01b = SGL data block, 10b = SGL with metadata SGL.   */
#define NVME_FLAG_PSDT_PRP     (0u << 6)
#define NVME_FLAG_PSDT_SGL     (1u << 6)

/* SGL descriptor type / subtype field (high byte of the 16-byte
 * descriptor).  Spec figure 105.  We only build the simplest form:
 *   type = 0 (Data Block), subtype = 0 (Address).                  */
#define NVME_SGL_TYPE_DATA_BLOCK   (0x0u << 4)
#define NVME_SGL_SUBTYPE_ADDR      (0x0u)
#define NVME_SGL_DESC_BYTE15       (NVME_SGL_TYPE_DATA_BLOCK | \
                                    NVME_SGL_SUBTYPE_ADDR)

/* ────────────────────────────────────────────────────────────
 * 【函数】tq_poll_one
 * 【作用】轮询等待这条队列上出现一条新的 CQE（即等一条命令完成）。读到后把它
 *         拷出来、推进 cq_head（必要时翻转 cq_phase）、敲 CQ doorbell 告知控制器
 *         “这一条我已消费”。
 * 【参数】
 *   - q          ：目标队列（提供 cq 环、cq_head、cq_phase、cq_db）。
 *   - cqe_out    ：输出参数，成功时把读到的 CQE 整条拷进来。
 *   - timeout_ms ：超时上限（毫秒），换算成内层自旋的最大迭代次数。
 * 【返回】0=成功并已填好 *cqe_out；-ETIMEDOUT=超时内控制器一直没写新 CQE。
 * 【在测试中的角色】每条 IO 提交后都调它等完成，是验证“CQE phase bit 正确、
 *   command_id 回显正确、数据已落盘/取回”的同步点。
 * 【新手提示】
 *   - 判定“新 CQE”的办法：读当前 cq_head 槽位的 status，取 bit0=phase，若它
 *     等于我们期待的 q->cq_phase，说明这格是本圈刚写入的新完成。
 *   - cq_head 绕回 0（走完一圈）时 cq_phase 取反——因为环复用同一块内存，
 *     靠 phase 翻转区分“上一圈的旧 CQE”和“这一圈的新 CQE”。
 *   - 用 volatile 读 slot，防编译器把对设备会改写的内存读优化掉。
 *   - 自旋时每 4096 次 sched_yield() 让出 CPU，避免空转独占一个核。
 * ──────────────────────────────────────────────────────────── */
/*
 * Wait for any CQE on this queue's CQ.  Returns 0 on success and fills
 * *cqe_out; returns -ETIMEDOUT if the controller never wrote one.
 *
 * Polls in a tight CPU loop with a 5-second wall-clock timeout (enough
 * for any sane controller; misbehaviour shows as timeout rather than
 * smoke hang).
 */
static int tq_poll_one(struct test_queue* q,
                       struct nvme_cqe* cqe_out,
                       unsigned timeout_ms) {
    /* Crude polling timer: we don't call clock_gettime in the inner
     * loop; instead we count iterations and back off after N misses
     * by yielding the CPU.  At ~100 ns/iter that's 5e7 iters per 5 s.   */
    unsigned long long max_iters = (unsigned long long)timeout_ms * 100000ULL;
    unsigned long long i = 0;

    for (;;) {
        volatile struct nvme_cqe* slot = &q->cq[q->cq_head];
        uint16_t status = slot->status;
        uint8_t phase = status & 0x1;
        if (phase == q->cq_phase) {
            /* CQE valid.  Copy out before advancing the head, then
             * ring the CQ doorbell so the controller knows we
             * consumed it.                                          */
            *cqe_out = *(const struct nvme_cqe*)slot;

            uint16_t new_head = (uint16_t)((q->cq_head + 1) % q->q_depth);
            if (new_head == 0)
                q->cq_phase ^= 1;     /* wrap -> phase flips */
            q->cq_head = new_head;
            mmio_writel(q->cq_db, new_head);
            return 0;
        }
        if (++i > max_iters) {
            return -ETIMEDOUT;
        }
        /* Release on every 4096th iter; avoids hogging a core if the
         * controller is slow.                                       */
        if ((i & 0xfff) == 0)
            sched_yield();
    }
}

/* ────────────────────────────────────────────────────────────
 * 【函数】format_status
 * 【作用】把 CQE 里的 16 位 status 状态字格式化成人类可读的字符串，拆出 SC
 *         （状态码）和 SCT（状态码类型）方便排错。
 * 【参数】
 *   - status ：CQE.status 原始值。
 *   - buf/cap：输出缓冲区及其容量。
 * 【返回】无（结果写进 buf）。
 * 【在测试中的角色】当某条 IO 返回非零 NVMe 状态（失败）时，用它生成可读信息
 *   再交给 step_fail 打印。
 * 【新手提示】先把 status 右移 1 位丢掉 bit0 的 phase 位（phase 在 tq_poll_one
 *   里已用过），剩下低 8 位是 SC、再上 3 位是 SCT；SC=0 即表示命令成功。
 * ──────────────────────────────────────────────────────────── */
/* Pretty-print an NVMe status word.  Phase bit (bit 0) is masked out
 * since we already consumed it in tq_poll_one.                        */
static void format_status(uint16_t status, char* buf, size_t cap) {
    uint16_t s   = status >> 1;            /* SF + DNR + M */
    uint8_t  sc  = s & 0xff;
    uint8_t  sct = (s >> 8) & 0x7;
    snprintf(buf, cap, "0x%04x (SC=0x%02x SCT=0x%x)", status, sc, sct);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】main
 * 【作用】整个端到端 NVMe IO 冒烟测试的总流程。从命令行拿一个 PCI BDF，把目标
 *         NVMe 设备接管过来，在用户态建好 SQ/CQ 环和数据缓冲、建用户 IO 队列、
 *         mmap BAR0 拿到 doorbell，然后亲手发一连串 Read/Write，校验数据往返
 *         完整无误，最后干净拆除。任何一步不对就 step_fail 退出（码 2）。
 * 【参数】argc/argv：argv[1] 是目标设备的 PCI 地址（DDDD:BB:DD.F）；--help 打用法。
 * 【返回】0=全部步骤通过；1=用法错误；2=某步失败（由 step_fail 退出）。
 * 【在测试中的角色】把下面所有函数/结构体串成完整剧本。各阶段：
 *   - Phase 0：打开 /dev/snvm_control，建并打开字符设备 /dev/ssnvmeN（控制面）。
 *   - Phase 1：建队列组（queue group）、设内核 IOQ 配额、绑定设备、取设备信息
 *              （盘名、块大小、q_depth、doorbell 布局等），断言是 4KiB 块盘。
 *   - Phase 2：分配并按页对齐 SQ/CQ 环 + wbuf/rbuf 数据页，全部用
 *              NVM_MAP_HOST_MEMORY 注册（pin + 建立 DMA 映射）。数据缓冲特意用
 *              map_kind=DATA + group_id=0 注册成“fd 作用域”，为 Phase 7(B6) 埋伏笔。
 *   - Phase 3：NVM_ADD_USER_QUEUE 真正在控制器上建用户 IO 队列，并拿回每条队列
 *              的 qid 与 SQ/CQ doorbell 在 BAR0 内的偏移。
 *   - Phase 4：mmap BAR0，把 doorbell 偏移换算成可写指针，初始化各 test_queue
 *              （cq_phase 初值=1，因为环已清零、控制器第一圈把 phase 翻成 1）。
 *   - Phase 5：基础往返。写队列(Q0)逐个 LBA 写 1 块、读队列(Q1)读回同一 LBA 并
 *              逐字节比对。验证 doorbell 可用、SQ/CQ 环放置正确、CQE phase 与
 *              command_id 正确、PRP1 单页数据往返无损、两条队列互不串扰。
 *   - Phase 5b：PRP1+PRP2 双 PRP，8KiB（2 页）IO。字节里混入页号，能抓到“控制器
 *              把 page0 当成 PRP2”之类的页错位 bug。
 *   - Phase 5c：PRP1 + PRP List，16KiB（4 页）IO。PRP2 指向一张页对齐的 PRP List
 *              页，表内依次放数据页 1/2/3 的 DMA 地址。验证 >2 页时的 PRP List 形式。
 *   - Phase 5d：SGL Data Block 描述符（PSDT=01b），仅当控制器在 Identify 里宣称
 *              支持 SGL 时才跑，否则跳过。验证 SGL 这条数据指针路径也能往返。
 *   - Phase 6：SQ tail 回绕压力测试。在 Q0 上连发 (q_depth+8) 条写，保证 sq_tail
 *              至少绕过 q_depth-1 一整圈、CQ phase 翻转一次，验证回绕逻辑正确。
 *   - Phase 7（B6）：销毁原队列组（应只回收 4 个 ring 映射，DATA 缓冲因挂在
 *              own->data_maps 上而存活）；再新建组、新分配并注册全新环、重发
 *              NVM_ADD_USER_QUEUE；用“跨组销毁后仍存活的同一对 wbuf/rbuf”做一次
 *              4KiB 写读校验。验证 fd 作用域的数据缓冲跨 group 销毁依然有效可用。
 *   - Phase 8：拆除。先 munmap BAR0（让 doorbell 指针失效，避免误写陈旧 tail），
 *              再销毁（Phase 7 新建的）队列组，free 所有用户态内存，解绑设备、
 *              关闭 fd、移除字符设备。剩余 DATA 映射在 close(fd_dev) 时统一回收。
 * 【新手提示】整条链路：CPU 填 SQE → 写 SQ 环 → 敲 SQ doorbell → 控制器 DMA 读
 *   SQE、按 PRP/SGL 搬数据、写 CQE → CPU 轮询 CQ 的 phase 位 → 敲 CQ doorbell。
 *   这就是 NVMe 一来一回的完整生命周期，本测试把每个环节都验了一遍。
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
    int fd_dev = open(dev_path, O_RDWR);
    if (fd_dev < 0)
        step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ============================================================== */
    /* Phase 1: queue group + cap + bind + dev info.                  */
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

    {
        uint32_t cap = 36;
        if (ioctl(fd_dev, NVM_SET_KERNEL_IOQ_CAP, &cap) != 0)
            step_fail(errno, "NVM_SET_KERNEL_IOQ_CAP cap=%u failed", cap);
        step_ok("NVM_SET_KERNEL_IOQ_CAP cap=%u (rest of grant -> user pool)",
                cap);
    }

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_BIND, &bdf, "SNVM_DEVICE_BIND") < 0)
            step_fail(errno, "SNVM_DEVICE_BIND %s", bdf_str);
        step_ok("SNVM_DEVICE_BIND %s", bdf_str);
    }

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
        step_ok("NVM_GET_DEV_INFO disk='%s' block_size=%zu q_depth=%u "
                "start_cq_idx=%u max_user_qid=%u",
                info.disk_name, info.block_size, info.q_depth,
                info.start_cq_idx, info.max_user_qid);
    }

    if (info.block_size != 4096)
        step_fail(0,
            "smoke assumes a 4 KiB-LBA controller; this disk reports "
            "block_size=%zu.  Adjust TEST_LBA_BASE / read-buffer size "
            "in the source if you really want to test a 512 B disk.",
            info.block_size);

    /* ============================================================== */
    /* Phase 2: allocate SQ+CQ rings + data buffers, register all     */
    /* against the queue group via NVM_MAP_HOST_MEMORY.               */
    /*                                                                 */
    /* We allocate one shared write-source buffer (filled with the    */
    /* per-IO pattern before each Write) and one shared read-target   */
    /* buffer (zeroed before each Read, then verified).  Same buffer  */
    /* reused across IOs because the controller serialises with the   */
    /* CQE (we only submit one IO at a time per queue and wait for it */
    /* to complete before reusing the buffer).                        */
    /* ============================================================== */

    const size_t   sqe_size = NVME_SQE_SIZE;
    const size_t   cqe_size = NVME_CQE_SIZE;
    size_t sq_bytes = (size_t)info.q_depth * sqe_size;
    size_t cq_bytes = (size_t)info.q_depth * cqe_size;

    if (round_up_pages(sq_bytes, psz) > (size_t)psz)
        step_fail(0,
            "B3 single-PRP limit: SQ ring (q_depth=%u * %zu = %zu B) "
            "spans more than one host page; lower io_queue_depth.",
            info.q_depth, sqe_size, sq_bytes);
    if (round_up_pages(cq_bytes, psz) > (size_t)psz)
        step_fail(0,
            "B3 single-PRP limit: CQ ring (q_depth=%u * %zu = %zu B) "
            "spans more than one host page.",
            info.q_depth, cqe_size, cq_bytes);

    void* sq_buf[TEST_NR_QUEUES];
    void* cq_buf[TEST_NR_QUEUES];
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        sq_buf[i] = alloc_aligned(sq_bytes, psz);
        cq_buf[i] = alloc_aligned(cq_bytes, psz);
        if (!sq_buf[i] || !cq_buf[i])
            step_fail(errno, "alloc_aligned ring pair %u", i);
    }

    /* One Write-source page and one Read-target page (both 4 KiB,
     * one LBA's worth on this disk).                                */
    void* wbuf = alloc_aligned(info.block_size, psz);
    void* rbuf = alloc_aligned(info.block_size, psz);
    if (!wbuf || !rbuf)
        step_fail(errno, "alloc_aligned data buffers");
    step_ok("allocated %u SQ+CQ ring pairs + 2 data buffers (block=%zu)",
            TEST_NR_QUEUES, info.block_size);

    /* Register every ring + data buffer against the group.  We need
     * the ioaddr write-back for the data buffers (PRP1 below) but
     * not for the rings (the kernel resolves those internally
     * during NVM_ADD_USER_QUEUE).                                   */
    uint64_t wbuf_ioaddr = 0;
    uint64_t rbuf_ioaddr = 0;

    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        uint64_t throwaway[1];
        struct nvm_ioctl_map req;

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)sq_buf[i];
        req.n_pages     = 1;
        req.ioaddrs     = throwaway;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = group_id;
        req.map_kind    = NVM_MAP_KIND_RING_SQ;
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
        req.map_kind    = NVM_MAP_KIND_RING_CQ;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(CQ)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY pair %u CQ", i);
    }

    /* Data buffers: capture ioaddr for PRP1 use later.  Registered
     * with kind=DATA + group_id=0 so they are fd-scoped (B6) -- they
     * survive NVM_DESTROY_QUEUE_GROUP and only get reaped on close().
     * This is what Phase 7 below actually verifies.                 */
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)wbuf;
        req.n_pages     = 1;
        req.ioaddrs     = &wbuf_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(wbuf)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY wbuf");
    }
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)rbuf;
        req.n_pages     = 1;
        req.ioaddrs     = &rbuf_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(rbuf)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY rbuf");
    }
    step_ok("NVM_MAP_HOST_MEMORY x %u rings + 2 data buffers (wbuf ioaddr=0x%llx, "
            "rbuf ioaddr=0x%llx)",
            TEST_NR_QUEUES * 2,
            (unsigned long long)wbuf_ioaddr,
            (unsigned long long)rbuf_ioaddr);

    /* ============================================================== */
    /* Phase 3: NVM_ADD_USER_QUEUE -- create queues and capture       */
    /* doorbell offsets.                                               */
    /* ============================================================== */
    struct nvm_ioctl_add_user_queue add_req;
    memset(&add_req, 0, sizeof(add_req));
    add_req.group_id = group_id;
    add_req.nr_pairs = TEST_NR_QUEUES;
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        add_req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t)sq_buf[i];
        add_req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t)cq_buf[i];
    }
    if (do_ioctl(fd_dev, NVM_ADD_USER_QUEUE, &add_req,
                 "NVM_ADD_USER_QUEUE") < 0)
        step_fail(errno, "NVM_ADD_USER_QUEUE -- check dmesg for which "
                         "Create I/O CQ/SQ admin command failed");

    step_ok("NVM_ADD_USER_QUEUE created %u user queue(s)", TEST_NR_QUEUES);
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        fprintf(stderr, "             pair[%u] qid=%u sq_db=0x%x cq_db=0x%x\n",
                i, add_req.out_pairs[i].qid,
                add_req.out_pairs[i].sq_doorbell_offset,
                add_req.out_pairs[i].cq_doorbell_offset);
    }

    /* ============================================================== */
    /* Phase 4: mmap BAR0 so we can ring the doorbells from user      */
    /* space.                                                          */
    /* ============================================================== */
    void* bar0 = mmap(NULL, info.bar0_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_dev, 0);
    if (bar0 == MAP_FAILED)
        step_fail(errno, "mmap BAR0 (%u bytes) on fd=%d", info.bar0_size, fd_dev);
    step_ok("mmap BAR0 size=0x%x at %p (snvme svm_mmap_registers)",
            info.bar0_size, bar0);

    /* Build per-queue runtime state. */
    struct test_queue Q[TEST_NR_QUEUES];
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        memset(&Q[i], 0, sizeof(Q[i]));
        Q[i].qid       = (uint16_t)add_req.out_pairs[i].qid;
        Q[i].q_depth   = info.q_depth;
        Q[i].sq        = (struct nvme_sqe*)sq_buf[i];
        Q[i].cq        = (struct nvme_cqe*)cq_buf[i];
        Q[i].sq_tail   = 0;
        Q[i].cq_head   = 0;
        Q[i].cq_phase  = 1;        /* CQ ring zeroed, controller XORs to 1 first lap */
        Q[i].sq_db     = (volatile uint32_t*)
                         ((char*)bar0 + add_req.out_pairs[i].sq_doorbell_offset);
        Q[i].cq_db     = (volatile uint32_t*)
                         ((char*)bar0 + add_req.out_pairs[i].cq_doorbell_offset);
        Q[i].next_cid  = 0;
    }

    /* ============================================================== */
    /* Phase 5: Sequential 1-LBA write+verify per queue, repeated     */
    /* TEST_NR_IO_PER_QUEUE times.  Each (queue, io) pair gets a      */
    /* unique LBA (no aliasing), and the byte pattern encodes both    */
    /* the qid and the iteration index so cross-queue/cross-io        */
    /* corruption is detectable from a single byte.                   */
    /*                                                                 */
    /* Layout:                                                         */
    /*   write queue (Q[0]): writes LBA = TEST_LBA_BASE + i            */
    /*                       for i in [0, TEST_NR_IO_PER_QUEUE)        */
    /*   read  queue (Q[1]): reads back the SAME LBA range             */
    /*                                                                 */
    /* This exercises:                                                 */
    /*   - sq_tail advancing 16 times (still within q_depth=64,        */
    /*     so no wrap-around -- separate test for that below)          */
    /*   - cq_phase staying at 1 the whole time                        */
    /*   - cross-queue isolation: Q[0]'s CQEs never appear in Q[1].   */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;       /* snvme exposes ns 1 */
        const uint16_t nlb_zero_based = 0;  /* 1 LBA per IO */

        struct test_queue* qw = &Q[0];
        struct test_queue* qr = &Q[1];
        char status_buf[64];

        for (unsigned i = 0; i < TEST_NR_IO_PER_QUEUE; i++) {
            uint64_t lba = TEST_LBA_BASE + i;
            uint8_t  pat = WRITE_PATTERN_BYTE(qw->qid, i);

            /* Stamp the write buffer with this iteration's pattern. */
            memset(wbuf, pat, info.block_size);

            uint16_t cid_w;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                         wbuf_ioaddr, /*dptr1=*/0,
                         lba, nlb_zero_based, &cid_w);

            struct nvme_cqe cqe_w;
            int rc = tq_poll_one(qw, &cqe_w, /*timeout_ms=*/5000);
            if (rc != 0)
                step_fail(-rc, "Write IO %u (qid=%u, lba=%" PRIu64 "): "
                               "CQE poll timed out",
                          i, qw->qid, lba);
            if ((cqe_w.status >> 1) != 0) {
                format_status(cqe_w.status, status_buf, sizeof(status_buf));
                step_fail(0, "Write IO %u (qid=%u, lba=%" PRIu64
                             ") returned non-zero NVMe status %s",
                          i, qw->qid, lba, status_buf);
            }
            if (cqe_w.cid != cid_w)
                step_fail(0, "Write IO %u CQE.cid=%u, expected %u "
                             "(SQ/CQ command_id mismatch)",
                          i, cqe_w.cid, cid_w);

            /* Now read it back from the OTHER queue. */
            memset(rbuf, 0, info.block_size);

            uint16_t cid_r;
            tq_submit_rw(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP, nsid,
                         rbuf_ioaddr, /*dptr1=*/0,
                         lba, nlb_zero_based, &cid_r);

            struct nvme_cqe cqe_r;
            rc = tq_poll_one(qr, &cqe_r, 5000);
            if (rc != 0)
                step_fail(-rc, "Read IO %u (qid=%u, lba=%" PRIu64 "): "
                               "CQE poll timed out",
                          i, qr->qid, lba);
            if ((cqe_r.status >> 1) != 0) {
                format_status(cqe_r.status, status_buf, sizeof(status_buf));
                step_fail(0, "Read IO %u (qid=%u, lba=%" PRIu64
                             ") returned non-zero NVMe status %s",
                          i, qr->qid, lba, status_buf);
            }
            if (cqe_r.cid != cid_r)
                step_fail(0, "Read IO %u CQE.cid=%u, expected %u",
                          i, cqe_r.cid, cid_r);

            /* Verify every byte. */
            uint8_t* rbytes = (uint8_t*)rbuf;
            for (size_t b = 0; b < info.block_size; b++) {
                if (rbytes[b] != pat) {
                    step_fail(0, "Read IO %u (lba=%" PRIu64 ") byte %zu = "
                                 "0x%02x, expected 0x%02x",
                              i, lba, b, rbytes[b], pat);
                }
            }
        }
        step_ok("write+verify x %u IOs across qid=%u (write) / qid=%u (read), "
                "LBA [%" PRIu64 "..%" PRIu64 "], 4 KiB each",
                TEST_NR_IO_PER_QUEUE, qw->qid, qr->qid,
                (uint64_t)TEST_LBA_BASE,
                (uint64_t)(TEST_LBA_BASE + TEST_NR_IO_PER_QUEUE - 1));
    }

    /* ============================================================== */
    /* Phase 5b: PRP1 + PRP2 (dual-PRP), 8 KiB IO = 2 host pages.     */
    /*                                                                 */
    /* NVMe PRP rules (1.4 spec, figure 11):                           */
    /*   PRP1 may have a page offset (low bits non-zero).              */
    /*   PRP2 MUST be page-aligned.                                    */
    /*   When the transfer crosses exactly one page boundary,          */
    /*   PRP1 = first-page-dma, PRP2 = second-page-dma.                */
    /*                                                                 */
    /* Buffer layout: 2 contiguous host pages, vaddr-aligned.  We      */
    /* register both pages in ONE NVM_MAP_HOST_MEMORY (n_pages=2),     */
    /* getting back ioaddrs[0] = page0_dma, ioaddrs[1] = page1_dma --  */
    /* the kernel pins each page individually so the dmas are NOT      */
    /* required to be physically contiguous.                           */
    /*                                                                 */
    /* Pattern: byte at offset b in the 8KiB write buffer =            */
    /*   0xA5 ^ qid ^ io_idx ^ (b >> 12)                               */
    /* so we can detect a page-swap bug (controller wrote page 0       */
    /* into PRP2 etc.) by spotting the wrong sub-byte at offset 0      */
    /* vs offset 4096.                                                 */
    /* ============================================================== */
    void* wbuf2 = alloc_aligned(2 * info.block_size, psz);
    void* rbuf2 = alloc_aligned(2 * info.block_size, psz);
    if (!wbuf2 || !rbuf2)
        step_fail(errno, "alloc_aligned 2-page buffers");

    uint64_t wbuf2_ioaddr[2] = {0, 0};
    uint64_t rbuf2_ioaddr[2] = {0, 0};
    {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)wbuf2;
        req.n_pages     = 2;
        req.ioaddrs     = wbuf2_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(wbuf2 x 2)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY wbuf2");

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)rbuf2;
        req.n_pages     = 2;
        req.ioaddrs     = rbuf2_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(rbuf2 x 2)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY rbuf2");
    }
    step_ok("Phase 5b: 2-page data buffers registered "
            "(wbuf2 ioaddrs=[0x%llx,0x%llx])",
            (unsigned long long)wbuf2_ioaddr[0],
            (unsigned long long)wbuf2_ioaddr[1]);

    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 1;     /* 2 LBAs per IO */
        const uint64_t LBA_PHASE_5B = TEST_LBA_BASE + 100;
        const unsigned NR_IO_5B = 8;
        const size_t io_bytes = 2 * info.block_size;

        struct test_queue* qw = &Q[0];
        struct test_queue* qr = &Q[1];
        char status_buf[64];

        for (unsigned i = 0; i < NR_IO_5B; i++) {
            uint64_t lba = LBA_PHASE_5B + 2u * i;       /* 2 LBA stride */
            uint8_t  pat = WRITE_PATTERN_BYTE(qw->qid, 100 + i);

            /* Stamp wbuf2 with sub-byte mixing per page. */
            uint8_t* wbytes = (uint8_t*)wbuf2;
            for (size_t b = 0; b < io_bytes; b++)
                wbytes[b] = (uint8_t)(pat ^ (uint8_t)(b >> 12));

            uint16_t cid_w;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                         wbuf2_ioaddr[0], wbuf2_ioaddr[1],
                         lba, nlb_zero_based, &cid_w);

            struct nvme_cqe cqe;
            int rc = tq_poll_one(qw, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5b Write %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qw->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5b Write %u: NVMe %s", i, status_buf);
            }
            if (cqe.cid != cid_w)
                step_fail(0, "5b Write %u CQE.cid=%u, expected %u",
                          i, cqe.cid, cid_w);

            /* Read back. */
            memset(rbuf2, 0, io_bytes);
            uint16_t cid_r;
            tq_submit_rw(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP, nsid,
                         rbuf2_ioaddr[0], rbuf2_ioaddr[1],
                         lba, nlb_zero_based, &cid_r);
            rc = tq_poll_one(qr, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5b Read %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qr->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5b Read %u: NVMe %s", i, status_buf);
            }
            if (cqe.cid != cid_r)
                step_fail(0, "5b Read %u CQE.cid=%u, expected %u",
                          i, cqe.cid, cid_r);

            /* Verify byte-for-byte against the same per-page pattern. */
            uint8_t* rbytes = (uint8_t*)rbuf2;
            for (size_t b = 0; b < io_bytes; b++) {
                uint8_t expect = (uint8_t)(pat ^ (uint8_t)(b >> 12));
                if (rbytes[b] != expect)
                    step_fail(0, "5b IO %u byte %zu: got 0x%02x, "
                                 "expected 0x%02x (page index %zu)",
                              i, b, rbytes[b], expect, b >> 12);
            }
        }
        step_ok("Phase 5b: PRP1+PRP2 dual-PRP x %u IOs, 8 KiB each, "
                "LBA [%" PRIu64 "..%" PRIu64 "]",
                NR_IO_5B, LBA_PHASE_5B, LBA_PHASE_5B + 2u * (NR_IO_5B - 1) + 1);
    }

    /* ============================================================== */
    /* Phase 5c: PRP1 + PRP List, 16 KiB IO = 4 host pages.           */
    /*                                                                 */
    /* When transfer > 2 pages, NVMe spec requires:                    */
    /*   PRP1 = first-page-dma (with optional offset)                  */
    /*   PRP2 = PRP_LIST_PAGE_DMA (ABSOLUTELY page-aligned, low 12     */
    /*          bits zero)                                             */
    /*   PRP_LIST_PAGE[i] = page (i+1) dma_addr                        */
    /*                                                                 */
    /* For 4 data pages we need PRP_LIST entries [page1, page2, page3].*/
    /* The PRP List itself is one host page allocated for this purpose,*/
    /* registered with NVM_MAP_HOST_MEMORY so the kernel pins it and   */
    /* hands us the dma_addr to put in PRP2.                           */
    /* ============================================================== */
    const unsigned PRP_LIST_NR_DATA_PAGES = 4;     /* one PRP1 + 3 list entries */
    const unsigned PRP_LIST_ENTRIES = PRP_LIST_NR_DATA_PAGES - 1;
    void* wbuf4 = alloc_aligned(PRP_LIST_NR_DATA_PAGES * info.block_size, psz);
    void* rbuf4 = alloc_aligned(PRP_LIST_NR_DATA_PAGES * info.block_size, psz);
    void* prp_list_w = alloc_aligned(info.block_size, psz);   /* one page each */
    void* prp_list_r = alloc_aligned(info.block_size, psz);
    if (!wbuf4 || !rbuf4 || !prp_list_w || !prp_list_r)
        step_fail(errno, "alloc_aligned 4-page buffers + PRP lists");

    uint64_t wbuf4_ioaddr[PRP_LIST_NR_DATA_PAGES];
    uint64_t rbuf4_ioaddr[PRP_LIST_NR_DATA_PAGES];
    uint64_t prp_list_w_ioaddr = 0;
    uint64_t prp_list_r_ioaddr = 0;
    memset(wbuf4_ioaddr, 0, sizeof(wbuf4_ioaddr));
    memset(rbuf4_ioaddr, 0, sizeof(rbuf4_ioaddr));
    {
        struct nvm_ioctl_map req;

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)wbuf4;
        req.n_pages     = PRP_LIST_NR_DATA_PAGES;
        req.ioaddrs     = wbuf4_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(wbuf4 x 4)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY wbuf4");

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)rbuf4;
        req.n_pages     = PRP_LIST_NR_DATA_PAGES;
        req.ioaddrs     = rbuf4_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(rbuf4 x 4)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY rbuf4");

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)prp_list_w;
        req.n_pages     = 1;
        req.ioaddrs     = &prp_list_w_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(prp_list_w)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY prp_list_w");

        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)prp_list_r;
        req.n_pages     = 1;
        req.ioaddrs     = &prp_list_r_ioaddr;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = 0;
        req.map_kind    = NVM_MAP_KIND_DATA;
        if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                     "NVM_MAP_HOST_MEMORY(prp_list_r)") < 0)
            step_fail(errno, "NVM_MAP_HOST_MEMORY prp_list_r");
    }

    /* Populate PRP lists.  Entries are 8-byte little-endian dma_addrs;
     * entry [0] = data page 1 (NOT page 0 -- page 0 went into PRP1).  */
    {
        uint64_t* lw = (uint64_t*)prp_list_w;
        uint64_t* lr = (uint64_t*)prp_list_r;
        for (unsigned i = 0; i < PRP_LIST_ENTRIES; i++) {
            lw[i] = wbuf4_ioaddr[i + 1];
            lr[i] = rbuf4_ioaddr[i + 1];
        }
    }
    step_ok("Phase 5c: 4-page data buffers + PRP_List pages registered "
            "(prp_list_w ioaddr=0x%llx)",
            (unsigned long long)prp_list_w_ioaddr);

    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = (uint16_t)(PRP_LIST_NR_DATA_PAGES - 1);
        const uint64_t LBA_PHASE_5C = TEST_LBA_BASE + 200;
        const unsigned NR_IO_5C = 4;
        const size_t io_bytes = PRP_LIST_NR_DATA_PAGES * info.block_size;

        struct test_queue* qw = &Q[0];
        struct test_queue* qr = &Q[1];
        char status_buf[64];

        for (unsigned i = 0; i < NR_IO_5C; i++) {
            uint64_t lba = LBA_PHASE_5C + 4u * i;       /* 4 LBA stride */
            uint8_t  pat = WRITE_PATTERN_BYTE(qw->qid, 200 + i);

            /* Stamp wbuf4 -- still mix sub-byte by 4 KiB page index so
             * any cross-page DMA misorder shows up as a single byte. */
            uint8_t* wbytes = (uint8_t*)wbuf4;
            for (size_t b = 0; b < io_bytes; b++)
                wbytes[b] = (uint8_t)(pat ^ (uint8_t)(b >> 12));

            uint16_t cid_w;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                         wbuf4_ioaddr[0], prp_list_w_ioaddr,
                         lba, nlb_zero_based, &cid_w);

            struct nvme_cqe cqe;
            int rc = tq_poll_one(qw, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5c Write %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qw->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5c Write %u: NVMe %s", i, status_buf);
            }

            /* Read back. */
            memset(rbuf4, 0, io_bytes);
            uint16_t cid_r;
            tq_submit_rw(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP, nsid,
                         rbuf4_ioaddr[0], prp_list_r_ioaddr,
                         lba, nlb_zero_based, &cid_r);
            rc = tq_poll_one(qr, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5c Read %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qr->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5c Read %u: NVMe %s", i, status_buf);
            }

            uint8_t* rbytes = (uint8_t*)rbuf4;
            for (size_t b = 0; b < io_bytes; b++) {
                uint8_t expect = (uint8_t)(pat ^ (uint8_t)(b >> 12));
                if (rbytes[b] != expect)
                    step_fail(0, "5c IO %u byte %zu: got 0x%02x, "
                                 "expected 0x%02x (page index %zu)",
                              i, b, rbytes[b], expect, b >> 12);
            }
        }
        step_ok("Phase 5c: PRP1 + PRP List x %u IOs, %u KiB each, "
                "LBA [%" PRIu64 "..%" PRIu64 "]",
                NR_IO_5C, (unsigned)(PRP_LIST_NR_DATA_PAGES * 4),
                LBA_PHASE_5C,
                LBA_PHASE_5C + 4u * (NR_IO_5C - 1) + 3);
    }

    /* ============================================================== */
    /* Phase 5d: SGL Data Block descriptor, conditional on the         */
    /* controller advertising SGL support in Identify Controller.     */
    /*                                                                 */
    /* SGL data block descriptor (NVMe 1.4 figure 105, Type=0):        */
    /*   bytes [0..7]   = address (data buffer dma_addr)               */
    /*   bytes [8..11]  = length (transfer size in bytes, little endian)*/
    /*   bytes [12..14] = reserved (zero)                              */
    /*   byte  [15]     = type (0x0) << 4 | subtype (0x0)              */
    /*                                                                 */
    /* Submitted by setting CDW0.PSDT = 01b (NVME_FLAG_PSDT_SGL) and   */
    /* packing the 16-byte descriptor into PRP1 (low 64 bits) +        */
    /* PRP2 (high 64 bits) of the SQE.                                 */
    /*                                                                 */
    /* Reuses the 4 KiB wbuf/rbuf from Phase 5 for simplicity -- this  */
    /* phase is just verifying that PSDT=01b is honoured for a 1-page  */
    /* transfer; multi-page SGL chains are out of scope for the smoke. */
    /* ============================================================== */
    if ((info.sgl_supported & 0x3) == 0) {
        step_ok("Phase 5d: SKIP -- controller advertises SGLS=0x%x, "
                "no SGL data block support (PRP-only controller)",
                info.sgl_supported);
    } else {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;             /* 1 LBA per IO */
        const uint64_t LBA_PHASE_5D = TEST_LBA_BASE + 300;
        const unsigned NR_IO_5D = 8;

        struct test_queue* qw = &Q[0];
        struct test_queue* qr = &Q[1];
        char status_buf[64];

        for (unsigned i = 0; i < NR_IO_5D; i++) {
            uint64_t lba = LBA_PHASE_5D + i;
            uint8_t pat = WRITE_PATTERN_BYTE(qw->qid, 300 + i);
            memset(wbuf, pat, info.block_size);

            /* Build the SGL data block descriptor in two 64-bit halves
             * that pack into the SQE's PRP1 (dptr0) and PRP2 (dptr1)
             * fields.  Spec: low 64 = address, high 64 has length in
             * the low 32 bits and the type/subtype byte at offset
             * 15 (i.e. high 8 bits of the dptr1 word).               */
            uint64_t sgl_addr = wbuf_ioaddr;
            uint64_t sgl_meta = ((uint64_t)info.block_size & 0xffffffffu)
                              | ((uint64_t)NVME_SGL_DESC_BYTE15 << 56);

            uint16_t cid_w;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_SGL, nsid,
                         sgl_addr, sgl_meta,
                         lba, nlb_zero_based, &cid_w);

            struct nvme_cqe cqe;
            int rc = tq_poll_one(qw, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5d Write %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qw->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5d Write %u: NVMe %s "
                             "(controller reported SGLS=0x%x but rejected "
                             "the descriptor; check PSDT bit packing)",
                          i, status_buf, info.sgl_supported);
            }

            /* Read back via SGL too. */
            memset(rbuf, 0, info.block_size);
            sgl_addr = rbuf_ioaddr;
            sgl_meta = ((uint64_t)info.block_size & 0xffffffffu)
                     | ((uint64_t)NVME_SGL_DESC_BYTE15 << 56);
            uint16_t cid_r;
            tq_submit_rw(qr, NVME_OPC_READ, NVME_FLAG_PSDT_SGL, nsid,
                         sgl_addr, sgl_meta,
                         lba, nlb_zero_based, &cid_r);
            rc = tq_poll_one(qr, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "5d Read %u (qid=%u, lba=%" PRIu64 ") timeout",
                          i, qr->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "5d Read %u: NVMe %s", i, status_buf);
            }

            uint8_t* rbytes = (uint8_t*)rbuf;
            for (size_t b = 0; b < info.block_size; b++) {
                if (rbytes[b] != pat)
                    step_fail(0, "5d IO %u byte %zu: got 0x%02x, "
                                 "expected 0x%02x",
                              i, b, rbytes[b], pat);
            }
        }
        step_ok("Phase 5d: SGL Data Block x %u IOs, 4 KiB each, "
                "LBA [%" PRIu64 "..%" PRIu64 "] (SGLS=0x%x)",
                NR_IO_5D, LBA_PHASE_5D, LBA_PHASE_5D + NR_IO_5D - 1,
                info.sgl_supported);
    }

    /* ============================================================== */
    /* Phase 6: SQ tail wrap test on qid=37.                          */
    /*                                                                 */
    /* q_depth=64 means SQ has 64 slots (sq_tail in [0..63]).  Phases */
    /* 5/5b/5c/5d advanced sq_tail by some amount on Q[0]; this phase */
    /* issues enough additional Writes that sq_tail definitely wraps  */
    /* past q_depth-1 back into [0..]; CQ phase on Q[0] also flips.   */
    /*                                                                 */
    /* LBA base bumped to TEST_LBA_BASE+1000 to stay clear of phases  */
    /* 5/5b/5c/5d (which use [0..15], 100, 200, 300 ranges).          */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;
        struct test_queue* qw = &Q[0];
        char status_buf[64];

        /* Issue (q_depth + 8) total IOs from this phase regardless of
         * what previous phases left in sq_tail -- guarantees at least
         * one full SQ wrap and one CQ phase flip.                    */
        unsigned cnt = info.q_depth + 8u;
        const uint64_t LBA_PHASE_6 = TEST_LBA_BASE + 1000;

        for (unsigned i = 0; i < cnt; i++) {
            uint64_t lba = LBA_PHASE_6 + i;
            uint8_t pat = WRITE_PATTERN_BYTE(qw->qid, 1000u + i);
            memset(wbuf, pat, info.block_size);

            uint16_t cid;
            tq_submit_rw(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP, nsid,
                         wbuf_ioaddr, /*dptr1=*/0,
                         lba, nlb_zero_based, &cid);

            struct nvme_cqe cqe;
            int rc = tq_poll_one(qw, &cqe, 5000);
            if (rc != 0)
                step_fail(-rc, "wrap Write %u (qid=%u, lba=%" PRIu64 ") "
                               "CQE poll timed out",
                          i, qw->qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "wrap Write %u: NVMe status %s",
                          i, status_buf);
            }
        }
        step_ok("SQ-tail-wrap stress: %u sequential 1-LBA Writes on qid=%u "
                "(sq_tail wrapped past q_depth=%u, cq_phase flipped)",
                cnt, qw->qid, info.q_depth);
    }

    /* ============================================================== */
    /* Phase 7: B6 fd-scoped DATA buffer survives group destroy.       */
    /*                                                                 */
    /* End of Phase 6 -- destroy the queue group while every data      */
    /* buffer (wbuf, rbuf, wbuf2, ..., prp_list_*) was registered      */
    /* with map_kind=NVM_MAP_KIND_DATA + group_id=0.  By design those  */
    /* are linked onto own->data_maps, NOT g->maps, so the cascade     */
    /* should drain ONLY the 4 ring maps (TEST_NR_QUEUES*2) and leave  */
    /* every DATA descriptor still pinned + DMA-mapped.                */
    /*                                                                 */
    /* Then re-create a group, re-allocate fresh SQ/CQ rings (the      */
    /* NVMe spec requires Create I/O SQ/CQ to point at fresh rings;    */
    /* nothing here re-uses the previous ring memory), re-issue        */
    /* NVM_ADD_USER_QUEUE -- and run a 4 KiB write+read+verify on the  */
    /* SAME wbuf / rbuf the previous group was using.  If anything in  */
    /* the B6 plumbing is wrong (DATA map got cascade-destroyed; the   */
    /* IOMMU mapping was torn down; the fd-scoped list lost the        */
    /* descriptor), the read either fails with -EFAULT, comes back     */
    /* with an NVMe status error, or returns the wrong bytes.          */
    /*                                                                 */
    /* Note: BAR0 is still mmap'd (we didn't munmap above), which is   */
    /* fine because the doorbell offsets the new ADD_USER_QUEUE        */
    /* returns are absolute BAR0 byte offsets -- they index the same   */
    /* mapping cleanly.                                                */
    /* ============================================================== */
    {
        /* a. Destroy the original group.  Should drain only rings.   */
        uint32_t gid_old = group_id;
        if (do_ioctl(fd_dev, NVM_DESTROY_QUEUE_GROUP, &gid_old,
                     "NVM_DESTROY_QUEUE_GROUP(B6 first destroy)") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP B6 first destroy");
        step_ok("Phase 7a: NVM_DESTROY_QUEUE_GROUP id=%u drains only %u "
                "ring map(s) (DATA buffers stay alive on own->data_maps)",
                group_id, TEST_NR_QUEUES * 2);

        /* b. Re-create a queue group on the same fd. */
        uint32_t group_id_b6 = 0;
        {
            struct nvm_ioctl_queue_group req;
            memset(&req, 0, sizeof(req));
            if (do_ioctl(fd_dev, NVM_CREATE_QUEUE_GROUP, &req,
                         "NVM_CREATE_QUEUE_GROUP(B6)") < 0)
                step_fail(errno, "NVM_CREATE_QUEUE_GROUP B6");
            group_id_b6 = req.group_id;
        }
        step_ok("Phase 7b: NVM_CREATE_QUEUE_GROUP -> group_id=%u (post-destroy)",
                group_id_b6);

        /* c. Re-allocate fresh rings + map them with kind=RING_*. */
        struct nvme_sqe* sq_buf_b6[TEST_NR_QUEUES];
        struct nvme_cqe* cq_buf_b6[TEST_NR_QUEUES];
        for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
            size_t sq_sz = (size_t)info.q_depth * sizeof(struct nvme_sqe);
            size_t cq_sz = (size_t)info.q_depth * sizeof(struct nvme_cqe);
            sq_buf_b6[i] = (struct nvme_sqe*)alloc_aligned(sq_sz, psz);
            cq_buf_b6[i] = (struct nvme_cqe*)alloc_aligned(cq_sz, psz);
            if (!sq_buf_b6[i] || !cq_buf_b6[i])
                step_fail(errno, "B6 alloc rings %u", i);
            memset(sq_buf_b6[i], 0, sq_sz);
            memset(cq_buf_b6[i], 0, cq_sz);
        }

        for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
            uint64_t throwaway[1];
            struct nvm_ioctl_map req;

            memset(&req, 0, sizeof(req));
            req.vaddr_start = (uint64_t)(uintptr_t)sq_buf_b6[i];
            req.n_pages     = 1;
            req.ioaddrs     = throwaway;
            req.ioq_idx     = -1;
            req.is_cq       = -1;
            req.group_id    = group_id_b6;
            req.map_kind    = NVM_MAP_KIND_RING_SQ;
            if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                         "NVM_MAP_HOST_MEMORY(B6 SQ)") < 0)
                step_fail(errno, "NVM_MAP_HOST_MEMORY B6 SQ %u", i);

            memset(&req, 0, sizeof(req));
            req.vaddr_start = (uint64_t)(uintptr_t)cq_buf_b6[i];
            req.n_pages     = 1;
            req.ioaddrs     = throwaway;
            req.ioq_idx     = -1;
            req.is_cq       = -1;
            req.group_id    = group_id_b6;
            req.map_kind    = NVM_MAP_KIND_RING_CQ;
            if (do_ioctl(fd_dev, NVM_MAP_HOST_MEMORY, &req,
                         "NVM_MAP_HOST_MEMORY(B6 CQ)") < 0)
                step_fail(errno, "NVM_MAP_HOST_MEMORY B6 CQ %u", i);
        }
        step_ok("Phase 7c: NVM_MAP_HOST_MEMORY x %u fresh ring(s) registered "
                "against group_id=%u; existing DATA maps untouched",
                TEST_NR_QUEUES * 2, group_id_b6);

        /* d. Issue NVM_ADD_USER_QUEUE for the fresh rings. */
        struct nvm_ioctl_add_user_queue add_req;
        memset(&add_req, 0, sizeof(add_req));
        add_req.group_id = group_id_b6;
        add_req.nr_pairs = TEST_NR_QUEUES;
        for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
            add_req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t)sq_buf_b6[i];
            add_req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t)cq_buf_b6[i];
        }
        if (do_ioctl(fd_dev, NVM_ADD_USER_QUEUE, &add_req,
                     "NVM_ADD_USER_QUEUE(B6)") < 0)
            step_fail(errno, "NVM_ADD_USER_QUEUE B6");
        step_ok("Phase 7d: NVM_ADD_USER_QUEUE created %u user queue(s) "
                "(qids %u..%u)", TEST_NR_QUEUES,
                add_req.out_pairs[0].qid,
                add_req.out_pairs[TEST_NR_QUEUES - 1].qid);

        /* e. Build per-queue runtime state and run a 4 KiB W+R+verify
         *    using the SAME wbuf/rbuf that survived the group destroy.   */
        struct test_queue qb6_w;
        struct test_queue qb6_r;
        memset(&qb6_w, 0, sizeof(qb6_w));
        memset(&qb6_r, 0, sizeof(qb6_r));
        qb6_w.sq      = sq_buf_b6[0];
        qb6_w.cq      = cq_buf_b6[0];
        qb6_w.q_depth = info.q_depth;
        qb6_w.qid     = (uint16_t)add_req.out_pairs[0].qid;
        qb6_w.cq_phase = 1;
        qb6_w.sq_db   = (volatile uint32_t*)
            ((uint8_t*)bar0 + add_req.out_pairs[0].sq_doorbell_offset);
        qb6_w.cq_db   = (volatile uint32_t*)
            ((uint8_t*)bar0 + add_req.out_pairs[0].cq_doorbell_offset);
        qb6_r.sq      = sq_buf_b6[1 % TEST_NR_QUEUES];
        qb6_r.cq      = cq_buf_b6[1 % TEST_NR_QUEUES];
        qb6_r.q_depth = info.q_depth;
        qb6_r.qid     = (uint16_t)add_req.out_pairs[1 % TEST_NR_QUEUES].qid;
        qb6_r.cq_phase = 1;
        qb6_r.sq_db   = (volatile uint32_t*)
            ((uint8_t*)bar0 + add_req.out_pairs[1 % TEST_NR_QUEUES].sq_doorbell_offset);
        qb6_r.cq_db   = (volatile uint32_t*)
            ((uint8_t*)bar0 + add_req.out_pairs[1 % TEST_NR_QUEUES].cq_doorbell_offset);

        /* Stamp wbuf with a B6-specific pattern so we know we're not
         * reading something the previous group left there.            */
        const uint8_t b6_pat = (uint8_t)(0x5A ^ qb6_w.qid);
        memset(wbuf, 0, info.block_size);
        for (size_t b = 0; b < info.block_size; b++)
            ((uint8_t*)wbuf)[b] = b6_pat ^ (uint8_t)(b >> 12);

        const uint64_t b6_lba = TEST_LBA_BASE + 50000;
        char status_buf[64];
        struct nvme_cqe cqe;
        uint16_t cid_w;

        tq_submit_rw(&qb6_w, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                     1, wbuf_ioaddr, 0, b6_lba, 0, &cid_w);
        int rc = tq_poll_one(&qb6_w, &cqe, 5000);
        if (rc) step_fail(-rc, "B6 Write poll rc=%d", rc);
        if ((cqe.status >> 1) != 0) {
            format_status(cqe.status, status_buf, sizeof(status_buf));
            step_fail(0, "B6 Write status %s", status_buf);
        }
        if (cqe.cid != cid_w)
            step_fail(0, "B6 Write CQE.cid=%u want %u", cqe.cid, cid_w);

        memset(rbuf, 0, info.block_size);
        uint16_t cid_r;
        tq_submit_rw(&qb6_r, NVME_OPC_READ, NVME_FLAG_PSDT_PRP,
                     1, rbuf_ioaddr, 0, b6_lba, 0, &cid_r);
        rc = tq_poll_one(&qb6_r, &cqe, 5000);
        if (rc) step_fail(-rc, "B6 Read poll rc=%d", rc);
        if ((cqe.status >> 1) != 0) {
            format_status(cqe.status, status_buf, sizeof(status_buf));
            step_fail(0, "B6 Read status %s", status_buf);
        }
        if (cqe.cid != cid_r)
            step_fail(0, "B6 Read CQE.cid=%u want %u", cqe.cid, cid_r);

        for (size_t b = 0; b < info.block_size; b++) {
            uint8_t want = b6_pat ^ (uint8_t)(b >> 12);
            if (((uint8_t*)rbuf)[b] != want)
                step_fail(0, "B6 readback mismatch at byte %zu: "
                             "got 0x%02x want 0x%02x (lba=%" PRIu64 ")",
                          b, ((uint8_t*)rbuf)[b], want, b6_lba);
        }
        step_ok("Phase 7e: 4 KiB W+R+verify on RECYCLED data buffers via "
                "post-destroy group (lba=%" PRIu64 ", qid_w=%u qid_r=%u)",
                b6_lba, qb6_w.qid, qb6_r.qid);

        /* f. Hand the fresh group back to Phase 8's destroy by
         *    overwriting group_id; Phase 8 will issue ONE final
         *    NVM_DESTROY_QUEUE_GROUP that picks up these new rings + 0
         *    data maps (DATA maps are already on own->data_maps and
         *    will be reaped at fd close instead).
         *
         *    The OLD rings (the sq_buf[i]/cq_buf[i] from Phase 2) are
         *    now orphaned at the user-space level: the kernel-side
         *    pinning was released by the Phase 7a DESTROY_QUEUE_GROUP,
         *    but the malloc'd memory itself is still ours to free.
         *    Drop it now before we lose the pointers.                 */
        for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
            free(sq_buf[i]);
            free(cq_buf[i]);
            sq_buf[i] = sq_buf_b6[i];
            cq_buf[i] = cq_buf_b6[i];
        }
        group_id = group_id_b6;
    }

    /* ============================================================== */
    /* Phase 8: Tear-down.                                            */
    /*                                                                 */
    /* munmap BAR0 first so the doorbell pointers in test_queue are    */
    /* invalidated before destroy_qgroup -- destroy_qgroup may issue  */
    /* admin commands that the kernel handles internally; we don't    */
    /* want to accidentally write a stale tail through the same       */
    /* pointer.  Then DESTROY_QUEUE_GROUP cascades through the        */
    /* Phase-7 fresh queue group's user queues + 4 ring maps; the     */
    /* DATA maps (wbuf/rbuf/wbuf2/rbuf2/wbuf4/rbuf4/prp_list_*) are    */
    /* released a moment later by close(fd_dev) via the per-fd        */
    /* data_maps cleanup in snvm_dev_release.                         */
    /* ============================================================== */
    if (munmap(bar0, info.bar0_size) < 0)
        step_fail(errno, "munmap BAR0");
    step_ok("munmap BAR0");

    {
        uint32_t gid = group_id;
        if (do_ioctl(fd_dev, NVM_DESTROY_QUEUE_GROUP, &gid,
                     "NVM_DESTROY_QUEUE_GROUP") < 0)
            step_fail(errno, "NVM_DESTROY_QUEUE_GROUP");
        /* Phase 7 only re-mapped 4 fresh ring buffers, all DATA maps
         * still live on own->data_maps and will be reaped at close.   */
        step_ok("NVM_DESTROY_QUEUE_GROUP id=%u cascades through %u user "
                "queue(s) + %u ring map(s) (8 DATA maps stay)",
                group_id, TEST_NR_QUEUES, TEST_NR_QUEUES * 2);
    }

    /* Free user-side ring + data buffers (the snvme-side maps are
     * already drained by NVM_DESTROY_QUEUE_GROUP).                    */
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        free(sq_buf[i]);
        free(cq_buf[i]);
    }
    free(wbuf);
    free(rbuf);
    free(wbuf2);
    free(rbuf2);
    free(wbuf4);
    free(rbuf4);
    free(prp_list_w);
    free(prp_list_r);

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

    fprintf(stderr, "\n=== snvme_smoke_io: all %d steps passed ===\n", g_step);
    return 0;
}
