/*
 * snvme_smoke_gpu.cu -- GPU end-to-end NVMe IO smoke test on B3 user
 * IO queues, with rings AND data buffers placed in GPU memory and
 * SQE submission / CQE polling driven from CUDA kernels.
 *
 * Mirrors snvme_smoke_io.c (same Phase numbering and Tier coverage)
 * but with the GPU-resident counterparts:
 *
 *   * SQ / CQ rings: cudaMalloc'd, registered with snvme via
 *     NVM_MAP_DEVICE_MEMORY (group-scoped, B2 path).  64 KiB
 *     GPU-page alignment is enforced by allocating one full GPU page
 *     per ring -- the q_depth=64 SQ (4 KiB) and CQ (1 KiB) only fill
 *     the first NVMe page of that 64 KiB allocation; the rest is
 *     unused but kept registered so the controller's PRP1 lookup
 *     stays trivial.
 *
 *   * Data buffers: same.  The 4 KiB / 8 KiB / 16 KiB tier transfers
 *     all share a single 64 KiB GPU allocation per direction, sliced
 *     by NVMe-page (4 KiB) offset.  Lets the same allocation cover
 *     PRP1, PRP1+PRP2, and PRP1+PRP_List without re-registration.
 *
 *   * Doorbells: BAR0 is mmap()d on the CPU side, then registered
 *     with cudaHostRegister(IoMemory) and translated to a GPU device
 *     pointer via cudaHostGetDevicePointer.  Submission CUDA kernels
 *     ring the SQ doorbell via a volatile uint32_t store directly
 *     from the GPU.
 *
 *   * SQE submission: built on the GPU using a one-thread CUDA kernel
 *     that fills the next slot in the GPU-resident SQ ring, then
 *     issues __threadfence_system() and rings the doorbell.
 *
 *   * CQE polling: GPU kernel spins on the phase bit of the next CQ
 *     slot, then writes the CQ head doorbell to release credit back
 *     to the controller.  Polling is bounded by an iteration counter
 *     (similar to the CPU smoke) so a misbehaving controller times
 *     out instead of hanging.
 *
 *   * DYNAMIC ALLOC/FREE LOOP: the entire data-plane (queue group +
 *     user IO queues + GPU rings + GPU data buffers + GPU PRP_Lists)
 *     is built up and torn down N times in a row (--rounds N,
 *     default 4).  Between rounds we cudaFree the GPU allocations
 *     and call NVM_DESTROY_QUEUE_GROUP, which forces the controller
 *     to execute Delete I/O SQ + Delete I/O CQ for every user queue
 *     and snvme to release every NVM_MAP_DEVICE_MEMORY descriptor.
 *     The next round re-creates everything from scratch.  This
 *     verifies:
 *       1. snvme's user QID pool is reclaimed correctly across
 *          DESTROY_QUEUE_GROUP cycles (no leak after N rounds).
 *       2. The controller accepts Create I/O SQ/CQ a second/third/...
 *          time on the same controller bind without misbehaving.
 *       3. GPU rings registered via NVM_MAP_DEVICE_MEMORY can be
 *          allocated and released repeatedly without breaking the
 *          NVIDIA p2p get_pages/put_pages refcount.
 *
 * Pre-conditions:
 *   - snvme-core.ko + snvme.ko loaded.
 *   - The proprietary NVIDIA driver loaded AND nvfs_nvidia_p2p_init()
 *     succeeded at module-load time (snvme refuses to load otherwise).
 *   - At least one GPU visible to CUDA.
 *   - Root (PCI bind requires CAP_SYS_ADMIN).
 *
 * DESTRUCTIVE: writes to LBAs starting at TEST_LBA_BASE (default
 * 2621440 = 10 GiB / 4 KiB).  Each round uses a non-overlapping LBA
 * window so the verifier can run independently per round.
 *
 * Build:        make snvme_smoke_gpu
 * Invoke:       sudo ./snvme_smoke_gpu [--gpu N] [--rounds N] <PCI_BDF>
 *
 * Exit codes:
 *   0  -- all steps passed.
 *   1  -- usage error.
 *   2  -- a smoke step failed; see stderr for which one.
 */

#include <cuda_runtime.h>

#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" {
#include "ioctl.h"
}

/* ------------------------------------------------------------------ */
/* NVMe spec constants.  Same definitions as snvme_smoke_io.c.        */
/* ------------------------------------------------------------------ */

#define NVME_OPC_WRITE              0x01u
#define NVME_OPC_READ               0x02u

#define NVME_SQE_SIZE               64u
#define NVME_CQE_SIZE               16u

#define TEST_LBA_BASE               2621440ULL   /* 10 GiB / 4 KiB */
#define TEST_NR_QUEUES              2u
#define TEST_NR_IO_PER_QUEUE        16u
#define TEST_DEFAULT_ROUNDS         4u
/* Per-round LBA window = 64 Ki LBAs = 256 MiB of 4 KiB sectors.
 * Rounds 0..N-1 occupy disjoint windows starting at TEST_LBA_BASE,
 * so a per-round verify never collides with another round.        */
#define TEST_LBA_PER_ROUND          0x10000ULL

/* GPU page size used by the kernel module
 * (snvme/map.c:GPU_PAGE_SHIFT=16).  Buffers passed to NVM_MAP_DEVICE_*
 * MUST be aligned to and sized in multiples of this.                  */
static constexpr size_t GPU_PAGE_SIZE = 1ULL << 16;     /* 64 KiB */

/* CDW0 PSDT bits (CDW0[15:14], appearing as bits [7:6] of the SQE
 * 'flags' byte).  PRP=00b, SGL data block=01b.                       */
#define NVME_FLAG_PSDT_PRP          (0u << 6)
#define NVME_FLAG_PSDT_SGL          (1u << 6)

/* SGL Data Block descriptor type|subtype byte.                        */
#define NVME_SGL_DESC_BYTE15        0x00u

/* Per-byte pattern is a function of (round, qid, ioidx) so a
 * cross-round byte mismatch is unambiguous in the failure log.    */
#define WRITE_PATTERN_BYTE(round, qid, ioidx) \
    ((uint8_t)(0xA5 ^ ((round) & 0xff) ^ ((qid) & 0xff) ^ ((ioidx) & 0xff)))

/* ------------------------------------------------------------------ */
/* Submission queue entry (Common Format, NVMe 1.4 figure 105).       */
/* Defined identically on host and device so __device__ kernels can   */
/* fill the same struct that the controller will then read via DMA.   */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【结构体】struct nvme_sqe —— NVMe 提交队列条目（Submission Queue Entry）
 * 【作用】描述一条要发给 NVMe 控制器的命令，固定 64 字节。GPU 核函数会
 *         往 GPU 显存里的 SQ 环填这个结构，控制器再通过 DMA 把它读走执行。
 * 【字段】
 *   opcode    —— 命令操作码（0x01=写、0x02=读）。
 *   flags     —— 命令标志位；高 2 位是 PSDT，决定用 PRP 还是 SGL 描述数据。
 *   cid       —— Command ID，命令唯一编号；完成时 CQE 会带回同一个 cid 供匹配。
 *   nsid      —— Namespace ID，目标命名空间（盘），本测试固定为 1。
 *   rsvd_2_3  —— 保留字段。
 *   metadata  —— 元数据指针（本测试不用）。
 *   prp1/prp2 —— 数据缓冲的物理/IO 地址（PRP = Physical Region Page）。
 *                Tier1 只用 prp1；Tier2 用 prp1+prp2；Tier3 prp2 指向 PRP_List。
 *   cdw10..15 —— Command Dword 10~15，命令相关参数。读写命令里 cdw10/11 放
 *                起始 LBA，cdw12 放“块数-1”(nlb_zero_based)。
 * 【在测试中的角色】这是 GPU 与 NVMe 控制器之间的命令“信件格式”，必须和
 *                 控制器约定的二进制布局严格一致，所以用 packed 且断言 64 字节。
 * 【新手提示】NVMe 工作方式：主机把命令写进 SQ 环 → 敲门铃(doorbell)通知控制器
 *           → 控制器执行 → 把结果写进 CQ 环。这里定义的就是“命令”那一半。
 * 【NPU 迁移提示】结构体本身是 NVMe 协议规定的，与 GPU/NPU 无关，迁移时不用改；
 *               但它会被 __device__/__global__ 核函数填写，那部分核函数迁移到
 *               昇腾时要改成 Ascend C / AIV kernel（见 k_submit_rw）。
 * ──────────────────────────────────────────────────────────── */
struct nvme_sqe {
    uint8_t  opcode;
    uint8_t  flags;
    uint16_t cid;
    uint32_t nsid;
    uint64_t rsvd_2_3;
    uint64_t metadata;
    uint64_t prp1;
    uint64_t prp2;
    uint32_t cdw10;
    uint32_t cdw11;
    uint32_t cdw12;
    uint32_t cdw13;
    uint32_t cdw14;
    uint32_t cdw15;
} __attribute__((packed));

static_assert(sizeof(nvme_sqe) == NVME_SQE_SIZE,
              "nvme_sqe must be exactly 64 bytes");

/* ────────────────────────────────────────────────────────────
 * 【结构体】struct nvme_cqe —— NVMe 完成队列条目（Completion Queue Entry）
 * 【作用】控制器执行完一条命令后，往 GPU 显存里的 CQ 环写入这个结构，固定 16 字节。
 *         GPU 轮询核函数读它来判断命令是否完成、成功还是失败。
 * 【字段】
 *   result  —— 命令特定返回值（读写命令一般为 0）。
 *   rsvd    —— 保留字段。
 *   sq_head —— 控制器已消费到的 SQ head 指针，用于回收 SQ 信用额度。
 *   sq_id   —— 这条完成项对应哪个提交队列。
 *   cid     —— 对应命令的 Command ID，与提交时的 cid 配对。
 *   status  —— 状态字段。最低位是 phase bit（相位位），用于判断该槽是否是本轮新写入；
 *              其余位是 SC(状态码)/SCT(状态码类型)，非 0 表示出错。
 * 【在测试中的角色】这是命令完成的“回执”格式；k_poll_one 通过比对 phase bit
 *                 判断有没有新回执，再用 status 判断成功失败。
 * 【新手提示】phase bit 机制：CQ 是环形缓冲，每绕一圈期望相位翻转一次。控制器写新
 *           条目时会把 phase 设成当前期望值，主机据此区分“新回执”和“上一圈的旧数据”。
 * 【NPU 迁移提示】协议结构体，迁移昇腾时无需修改；读取它的轮询核函数才需要改写。
 * ──────────────────────────────────────────────────────────── */
struct nvme_cqe {
    uint32_t result;
    uint32_t rsvd;
    uint16_t sq_head;
    uint16_t sq_id;
    uint16_t cid;
    uint16_t status;
} __attribute__((packed));

static_assert(sizeof(nvme_cqe) == NVME_CQE_SIZE,
              "nvme_cqe must be exactly 16 bytes");

/* ------------------------------------------------------------------ */
/* Logging helpers (host-side).                                       */
/* ------------------------------------------------------------------ */

static int g_step = 0;

/* ────────────────────────────────────────────────────────────
 * 【函数】step_ok
 * 【作用】打印一条“[ OK ] step=N ...”的成功日志（带可变参数，像 printf）。
 *         每调用一次全局步骤计数器 g_step 自增 1。
 * 【参数】fmt + ... —— printf 风格的格式串和参数。
 * 【返回】无。
 * 【在测试中的角色】每完成一个验证步骤就调一次，给人看测试进度。
 * 【新手提示】va_list/va_start/vfprintf 是 C 标准的可变参数转发写法。
 * ──────────────────────────────────────────────────────────── */
static void step_ok(const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[ OK ] step=%-3d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】step_fail
 * 【作用】打印一条“[FAIL] step=N ...”的失败日志，附带 errno 解释，然后
 *         直接 exit(2) 终止整个测试程序（noreturn，不会返回调用处）。
 * 【参数】
 *   err      —— errno 值；为 0 时打印 "n/a"，否则用 strerror 翻译成文字。
 *   fmt + ... —— printf 风格的失败原因描述。
 * 【返回】不返回（标记为 noreturn，进程退出码 2）。
 * 【在测试中的角色】任何一步出错就调它，立即中止并报告是哪一步挂了。
 * 【新手提示】退出码 2 在文件头注释里约定为“某个 smoke 步骤失败”。
 * ──────────────────────────────────────────────────────────── */
static void __attribute__((noreturn)) step_fail(int err, const char* fmt, ...) {
    va_list ap;
    g_step++;
    fprintf(stderr, "[FAIL] step=%-3d ", g_step);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, " errno=%d (%s)\n", err, err ? strerror(err) : "n/a");
    exit(2);
}

#define CUDA_OK(call_)                                                  \
    do {                                                                \
        cudaError_t _e = (call_);                                       \
        if (_e != cudaSuccess) {                                        \
            step_fail(0, "CUDA: %s -> %s", #call_, cudaGetErrorString(_e)); \
        }                                                               \
    } while (0)

/* ────────────────────────────────────────────────────────────
 * 【函数】parse_bdf
 * 【作用】把命令行里 "DDDD:BB:DD.F" 形式的 PCI 设备地址字符串解析成
 *         struct pci_device_addr（域:总线:槽.功能）。
 * 【参数】
 *   s   —— 输入字符串，例如 "0000:08:00.0"。
 *   out —— 输出结构体，填好 domain/bus/slot/func 四个字段。
 * 【返回】0 表示解析成功（恰好填满 4 个字段），-1 表示格式不对。
 * 【在测试中的角色】启动时把用户给的 BDF 转成内核 ioctl 需要的二进制地址。
 * 【新手提示】BDF = Bus:Device.Function，是 PCI 设备在系统里的唯一定位编号。
 * ──────────────────────────────────────────────────────────── */
static int parse_bdf(const char* s, struct pci_device_addr* out) {
    return sscanf(s, "%x:%x:%x.%x",
                  &out->domain, &out->bus, &out->slot, &out->func) == 4 ? 0 : -1;
}

/* ────────────────────────────────────────────────────────────
 * 【函数】do_ioctl
 * 【作用】对内核字符设备发一个 ioctl 调用，并在失败时打印带名字的错误信息，
 *         同时保留 errno 不被后续调用覆盖。
 * 【参数】
 *   fd   —— 已打开的设备文件描述符（/dev/snvm_control 或 /dev/ssnvmeN）。
 *   req  —— ioctl 命令号（如 NVM_CREATE_QUEUE_GROUP 等）。
 *   arg  —— 指向命令参数结构体的指针，内核会读/写它。
 *   what —— 这次调用的可读名字，仅用于出错日志。
 * 【返回】ioctl 的返回值；<0 表示失败（errno 已被设回原值）。
 * 【在测试中的角色】所有和 snvme 内核模块的控制面交互都走它，是测试与驱动沟通的总入口。
 * 【新手提示】ioctl 是 Linux 里“给设备下达特殊命令”的通用系统调用。
 * 【NPU 迁移提示】这里的 ioctl 命令号来自 snvme 驱动（ioctl.h），迁移到昇腾平台时
 *               需要换成华为对应的设备控制接口/驱动 ioctl 集合。
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
 * 【作用】把命令行用法说明打到 stderr（参数格式、举例、危险性提示）。
 * 【参数】prog —— 程序名（argv[0]），用于拼出示例命令。
 * 【返回】无。
 * 【在测试中的角色】参数解析出错或用户传 --help 时调用。
 * 【新手提示】DESTRUCTIVE 提示：本测试会真往磁盘写数据，跑之前要确认 LBA 区间没数据。
 * ──────────────────────────────────────────────────────────── */
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--gpu N] [--rounds N] <PCI_BDF>\n"
        "  e.g.: %s --gpu 0 --rounds 4 0000:08:00.0\n"
        "\n"
        "BINDS the target controller and writes/reads via GPU-resident\n"
        "rings + GPU-resident data buffers, repeated --rounds times\n"
        "(default %u) with full queue/buffer alloc-free between rounds.\n"
        "DESTRUCTIVE.\n",
        prog, prog, TEST_DEFAULT_ROUNDS);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】format_status
 * 【作用】把 CQE 的 16 位 status 字段拆解成人类可读字符串，分离出
 *         SC(状态码) 和 SCT(状态码类型)，写进调用者给的 buf。
 * 【参数】
 *   status —— CQE.status 原始 16 位值（最低位是 phase，本函数右移 1 位丢掉它）。
 *   buf    —— 输出缓冲区。
 *   cap    —— buf 容量（用 snprintf 防越界）。
 * 【返回】无（结果写进 buf）。
 * 【在测试中的角色】命令失败时把状态码格式化出来，方便定位 NVMe 错误原因。
 * 【新手提示】NVMe 状态字：bit0=phase；bits[8:1]=SC；bits[11:9]=SCT。SC/SCT 都为 0 表示成功。
 * ──────────────────────────────────────────────────────────── */
static void format_status(uint16_t status, char* buf, size_t cap) {
    uint16_t s   = status >> 1;
    uint8_t  sc  = s & 0xff;
    uint8_t  sct = (s >> 8) & 0x7;
    snprintf(buf, cap, "0x%04x (SC=0x%02x SCT=0x%x)", status, sc, sct);
}

/* ------------------------------------------------------------------ */
/* Per-queue runtime state.  Lives partly in host memory (the         */
/* `*_dev` device pointers) and partly on the GPU (the rings + the    */
/* doorbell GPU VA).  Submission/poll kernels receive a copy of the   */
/* whole struct by value.                                             */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【结构体】struct test_queue_dev —— 单个 IO 队列的“设备侧”句柄
 * 【作用】把一对 SQ/CQ 环加上它们的门铃地址、队列深度、队列号打包成一个
 *         可按值传给 GPU 核函数的结构。核函数拿到它就能独立完成提交和轮询。
 * 【字段】
 *   sq      —— SQ 环在 GPU 显存里的设备虚拟地址（核函数往这里写 SQE）。
 *   cq      —— CQ 环在 GPU 显存里的设备虚拟地址（核函数从这里读 CQE）。
 *   sq_db   —— SQ 门铃寄存器的 GPU 可见地址（映射到 BAR0 内），写它=通知控制器有新命令。
 *   cq_db   —— CQ 门铃寄存器的 GPU 可见地址，写它=告诉控制器“这些回执我收下了”。
 *   q_depth —— 队列深度（环里有多少个槽），用于 tail/head 的取模回绕。
 *   qid     —— 队列编号（控制器分配），用于日志和构造模式。
 * 【在测试中的角色】run_one_round 填好它，再交给 k_submit_rw / k_poll_one 在 GPU 上跑。
 * 【新手提示】门铃(doorbell)是 NVMe 控制器 BAR0 里的一个寄存器，主机写入“新的 tail/head”
 *           索引来通知控制器，是 GPU 直接驱动 NVMe 的关键。
 * 【NPU 迁移提示】sq/cq 指向 GPU 显存——昇腾上对应 device 内存（aclrtMalloc 得到的地址）；
 *               sq_db/cq_db 指向被映射进 GPU 地址空间的 BAR0 寄存器——昇腾上需要把 NVMe
 *               BAR0 注册成 AIV 可见地址后再取设备指针（替代 cudaHostRegister+GetDevicePointer）。
 * ──────────────────────────────────────────────────────────── */
struct test_queue_dev {
    nvme_sqe*           sq;             /* device VA */
    nvme_cqe*           cq;             /* device VA */
    volatile uint32_t*  sq_db;          /* GPU VA into BAR0 */
    volatile uint32_t*  cq_db;          /* GPU VA into BAR0 */
    uint16_t            q_depth;
    uint16_t            qid;
};

/* ------------------------------------------------------------------ */
/* GPU kernels for SQE submit / CQE poll / data fill / data verify.   */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】k_submit_rw（__global__ CUDA 核函数，单线程执行）
 * 【作用】在 GPU 上把一条读/写命令填进 SQ 环的下一个空槽，做一次 system-scope
 *         内存栅栏确保 SQE 对控制器 DMA 可见，然后写 SQ 门铃通知控制器、推进 tail。
 * 【参数】
 *   qd             —— 本队列的设备句柄（含 sq 环、sq_db 门铃、q_depth 等），按值传入。
 *   sq_tail_io     —— 指向统一内存里的 SQ tail 计数器，函数读它定位空槽并写回新值。
 *   cid            —— 本命令的 Command ID。
 *   opcode/flags   —— 命令操作码、标志位（PRP 或 SGL）。
 *   nsid           —— 命名空间 ID。
 *   dptr0/dptr1    —— 填入 prp1/prp2 的数据缓冲 IO 地址。
 *   slba           —— 起始 LBA（拆进 cdw10/cdw11）。
 *   nlb_zero_based —— 块数减 1（填进 cdw12）。
 * 【返回】无（结果体现在 SQ 环、门铃寄存器和 sq_tail_io 的更新上）。
 * 【在测试中的角色】submit_and_poll 的“提交”一半，演示由 GPU 核而非 CPU 来构造并下发 NVMe 命令。
 * 【新手提示】先清零 64 字节槽再填字段，避免残留旧数据；用 if(threadIdx==0&&blockIdx==0)
 *           保证只有一个线程干活，因为提交是单点串行操作。
 * 【NPU 迁移提示】
 *   - __global__ 核函数 → 改写为 Ascend C / AIV kernel。
 *   - __threadfence_system() → 换成 AIV 的 system-scope fence，保证 SQE 写入对 NVMe DMA 可见后才敲门铃。
 *   - *qd.sq_db 这种直接从核里写 BAR0 门铃 → 依赖 BAR0 被映射成设备可见地址，
 *     昇腾上需用华为 peer DMA / BAR 映射机制提供同等的“设备侧写 MMIO 寄存器”能力。
 * ──────────────────────────────────────────────────────────── */
__global__ void k_submit_rw(test_queue_dev qd,
                            uint16_t* sq_tail_io,
                            uint16_t cid,
                            uint8_t opcode,
                            uint8_t flags,
                            uint32_t nsid,
                            uint64_t dptr0,
                            uint64_t dptr1,
                            uint64_t slba,
                            uint16_t nlb_zero_based) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint16_t tail = *sq_tail_io;
    nvme_sqe* slot = &qd.sq[tail];

    /* Zero the 64-byte slot through the same path the controller will
     * see (memset_d-style writes via CUDA's volatile semantics).     */
    uint8_t* p = (uint8_t*)slot;
    #pragma unroll
    for (int i = 0; i < (int)sizeof(nvme_sqe); i++) p[i] = 0;

    slot->opcode = opcode;
    slot->flags  = flags;
    slot->cid    = cid;
    slot->nsid   = nsid;
    slot->prp1   = dptr0;
    slot->prp2   = dptr1;
    slot->cdw10  = (uint32_t)(slba & 0xffffffffu);
    slot->cdw11  = (uint32_t)(slba >> 32);
    slot->cdw12  = (uint32_t)(nlb_zero_based & 0xffffu);

    /* Make the SQE bytes visible to the device DMA engine BEFORE we
     * ring the doorbell.                                            */
    __threadfence_system();

    uint16_t new_tail = (uint16_t)((tail + 1) % qd.q_depth);
    *qd.sq_db = new_tail;
    *sq_tail_io = new_tail;
}

/* ────────────────────────────────────────────────────────────
 * 【函数】k_poll_one（__global__ CUDA 核函数，单线程执行）
 * 【作用】在 GPU 上自旋轮询 CQ 环当前 head 槽的 phase bit；一旦出现期望相位
 *         （表示有新回执），就把整条 16 字节 CQE 拷出来，推进 cq_head、必要时
 *         翻转 phase，敲 CQ 门铃归还信用。超过 max_iters 仍无回执则报超时。
 * 【参数】
 *   qd          —— 本队列设备句柄（含 cq 环和 cq_db 门铃）。
 *   cq_head_io  —— 统一内存里的 CQ head 指针，读+写回。
 *   cq_phase_io —— 统一内存里的当前期望相位，绕环一圈翻转一次。
 *   out_cqe     —— 输出：拷出的完整 CQE 给主机检查。
 *   timed_out   —— 输出：1=超时未等到回执，0=成功。
 *   max_iters   —— 自旋上限，防止控制器异常时核函数永久卡死。
 * 【返回】无（结果写进 out_cqe / timed_out / 推进 head 与 phase）。
 * 【在测试中的角色】submit_and_poll 的“轮询”一半，演示由 GPU 核直接收割 NVMe 完成项。
 * 【新手提示】CQ 是环形的，靠 phase bit 区分新旧回执；head 绕回 0 时期望相位异或 1。
 *           敲 CQ 门铃相当于告诉控制器“这些槽我读完了，可以复用”。
 * 【NPU 迁移提示】
 *   - __global__ 核 → Ascend C / AIV kernel。
 *   - 用 volatile 读 CQ 槽 + __threadfence_system() → 昇腾上需 AIV system-scope fence
 *     保证读到控制器经 DMA 写入显存的最新 CQE。
 *   - *qd.cq_db 直接写 BAR0 门铃 → 同样依赖 BAR0 设备可见映射（华为 peer DMA/BAR 接口）。
 * ──────────────────────────────────────────────────────────── */
/* Polls until either a CQE with the expected phase bit appears, or
 * `max_iters` iterations elapse without one (reported as
 * timed_out=1).  On success, copies the CQE out, advances cq_head,
 * flips cq_phase if the head wraps, and rings the CQ head doorbell.   */
__global__ void k_poll_one(test_queue_dev qd,
                           uint16_t* cq_head_io,
                           uint8_t* cq_phase_io,
                           nvme_cqe* out_cqe,
                           int* timed_out,
                           uint64_t max_iters) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint16_t head = *cq_head_io;
    uint8_t  expected = *cq_phase_io;

    uint64_t i = 0;
    for (;;) {
        volatile nvme_cqe* slot = &qd.cq[head];
        uint16_t status = slot->status;
        uint8_t phase = status & 0x1u;
        if (phase == expected) {
            /* Copy out the full 16-byte CQE before advancing. */
            nvme_cqe tmp;
            tmp.result  = slot->result;
            tmp.rsvd    = slot->rsvd;
            tmp.sq_head = slot->sq_head;
            tmp.sq_id   = slot->sq_id;
            tmp.cid     = slot->cid;
            tmp.status  = status;
            *out_cqe = tmp;

            uint16_t new_head = (uint16_t)((head + 1) % qd.q_depth);
            if (new_head == 0) expected ^= 1u;

            __threadfence_system();
            *qd.cq_db = new_head;
            *cq_head_io  = new_head;
            *cq_phase_io = expected;
            *timed_out = 0;
            return;
        }
        if (++i >= max_iters) {
            *timed_out = 1;
            return;
        }
    }
}

/* ------------------------------------------------------------------ */
/* GPU helpers: fill a buffer with a per-byte pattern; verify ditto.  */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】k_fill_pattern（__global__ CUDA 核函数，多线程并行）
 * 【作用】用多线程并行把 GPU 写缓冲填满一个可预测的数据模式：
 *         每字节 = pat ^ (字节偏移 >> 12)，即每 4 KiB 页换一个高位扰动。
 * 【参数】
 *   buf   —— 要填充的 GPU 缓冲区设备指针。
 *   bytes —— 填充字节数（决定要启动多少线程）。
 *   pat   —— 基准模式字节，由 (round,qid,ioidx) 算出，保证跨轮/跨队列唯一。
 * 【返回】无。
 * 【在测试中的角色】每次写命令之前先填好已知数据，写盘后再读回校验。
 * 【新手提示】idx = blockIdx.x*blockDim.x + threadIdx.x 是 CUDA 里计算全局线程号的标准式子；
 *           越界线程直接 return。
 * 【NPU 迁移提示】__global__ 核 → Ascend C / AIV kernel；全局线程索引换成 AIV 的
 *               block/thread 等价计算。这是普通显存填充，不涉及 BAR0/p2p，迁移最简单。
 * ──────────────────────────────────────────────────────────── */
__global__ void k_fill_pattern(uint8_t* buf, size_t bytes, uint8_t pat) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    buf[idx] = pat ^ (uint8_t)(idx >> 12);
}

/* ────────────────────────────────────────────────────────────
 * 【函数】k_verify_pattern（__global__ CUDA 核函数，多线程并行）
 * 【作用】在 GPU 上并行校验读回缓冲是否与当初 k_fill_pattern 写入的模式一致；
 *         发现任一字节不符就用原子操作记录第一个失配的字节下标。
 * 【参数】
 *   buf          —— 读回数据的 GPU 缓冲区设备指针（const）。
 *   bytes        —— 校验字节数。
 *   pat          —— 当初填充用的基准模式字节（须与写入时相同）。
 *   mismatch_idx —— 输出：统一内存里的失配下标；调用前置为 -1，仍为 -1 表示全部正确。
 * 【返回】无（结果体现在 mismatch_idx）。
 * 【在测试中的角色】读命令完成后比对数据，确认“写进去的”和“读出来的”一致，构成端到端正确性验证。
 * 【新手提示】atomicCAS(mismatch_idx, -1, idx)：只有第一个发现错误的线程能成功写入，
 *           保证多线程竞争下结果确定（“任一失配字节即可”）。
 * 【NPU 迁移提示】__global__ 核 → Ascend C / AIV kernel；atomicCAS 换成 AIV 对应的原子比较交换。
 *               同样是普通显存运算，不涉及 BAR0/p2p。
 * ──────────────────────────────────────────────────────────── */
__global__ void k_verify_pattern(const uint8_t* buf, size_t bytes,
                                 uint8_t pat, int* mismatch_idx) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) return;
    uint8_t expect = pat ^ (uint8_t)(idx >> 12);
    if (buf[idx] != expect) {
        /* Race-tolerant: any thread can win, the first idx written
         * wins for the host-side error report.  We just need ANY
         * mismatching byte.                                          */
        atomicCAS(mismatch_idx, -1, (int)idx);
    }
}

/* ------------------------------------------------------------------ */
/* Host helpers around the submit / poll kernels.                     */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【结构体】struct queue_state —— 单个队列的“主机侧”完整运行状态
 * 【作用】在 test_queue_dev（设备句柄）之外，再挂上一组放在统一内存里的计数器，
 *         让提交/轮询核函数能就地读写 tail/head/phase 而不必每次 IO 在主机和设备间来回拷贝。
 * 【字段】
 *   dev          —— 设备侧句柄（环、门铃、深度、qid），传给核函数用。
 *   sq_tail_um   —— 统一内存里的 SQ tail 计数器（1 个元素）。
 *   cq_head_um   —— 统一内存里的 CQ head 计数器。
 *   cq_phase_um  —— 统一内存里的当前 CQ 期望相位。
 *   out_cqe_um   —— 统一内存里存放轮询核拷出的 CQE。
 *   timed_out_um —— 统一内存里的超时标志。
 *   next_cid     —— 主机侧自增的下一个 Command ID。
 * 【在测试中的角色】run_one_round 给每个队列建一个，submit_and_poll 反复用它驱动一次次 IO。
 * 【新手提示】统一内存(Unified/Managed Memory)：一块 CPU 和 GPU 都能直接访问的内存，
 *           省去显式拷贝，特别适合这种 CPU/GPU 都要读写的小计数器。
 * 【NPU 迁移提示】*_um 字段依赖 CUDA Unified/Managed Memory（cudaMallocManaged）；
 *               昇腾上若无统一内存等价物，需改成 host 内存 + 显式 device 拷贝，或用华为
 *               的统一/共享内存接口。
 * ──────────────────────────────────────────────────────────── */
struct queue_state {
    test_queue_dev      dev;
    /* host-side counters; we keep these in unified-pinned memory so
     * the kernels can read+update them in place without a roundtrip */
    uint16_t*           sq_tail_um;     /* unified memory, 1 element */
    uint16_t*           cq_head_um;     /* ditto */
    uint8_t*            cq_phase_um;    /* ditto */
    nvme_cqe*           out_cqe_um;     /* unified memory, 1 element */
    int*                timed_out_um;   /* ditto */
    uint16_t            next_cid;
};

/* ────────────────────────────────────────────────────────────
 * 【函数】submit_and_poll
 * 【作用】完成一次完整的 NVMe IO：先启动 k_submit_rw 核在 GPU 上下发命令，
 *         再启动 k_poll_one 核在 GPU 上等回执，最后同步并把结果返回主机。
 * 【参数】
 *   qs             —— 目标队列状态（含设备句柄和统一内存计数器），引用传入。
 *   opcode/flags   —— 命令类型与 PRP/SGL 标志。
 *   nsid           —— 命名空间 ID。
 *   dptr0/dptr1    —— prp1/prp2 数据地址。
 *   slba           —— 起始 LBA。
 *   nlb_zero_based —— 块数减 1。
 *   cqe_out        —— 输出：本次命令的完成项。
 *   cid_out        —— 输出：本次分配的 Command ID（可为空）。
 * 【返回】0 成功；-EIO 核启动失败；-ETIMEDOUT 轮询超时。
 * 【在测试中的角色】各 Tier 测试反复调它，是“一次 IO”的主机侧封装；它把提交核和轮询核
 *                 串到默认流上顺序执行，再用 cudaDeviceSynchronize 等两核都跑完。
 * 【新手提示】两个核都丢到默认流（stream 0），默认流天然串行，所以提交一定先于轮询完成，
 *           无需在两者之间额外同步。
 * 【NPU 迁移提示】<<<1,1>>> 的核启动语法、cudaGetLastError、cudaDeviceSynchronize 都是
 *               CUDA 专有 → 昇腾上换成 AIV kernel 的下发与流/事件同步接口（aclrtSynchronizeStream 等）。
 * ──────────────────────────────────────────────────────────── */
static int submit_and_poll(queue_state& qs,
                           uint8_t opcode, uint8_t flags,
                           uint32_t nsid,
                           uint64_t dptr0, uint64_t dptr1,
                           uint64_t slba, uint16_t nlb_zero_based,
                           nvme_cqe* cqe_out, uint16_t* cid_out) {
    uint16_t cid = qs.next_cid++;
    if (cid_out) *cid_out = cid;

    k_submit_rw<<<1, 1>>>(qs.dev, qs.sq_tail_um, cid,
                          opcode, flags, nsid, dptr0, dptr1,
                          slba, nlb_zero_based);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "k_submit_rw launch: %s\n", cudaGetErrorString(e));
        return -EIO;
    }
    /* No explicit cudaDeviceSynchronize between submit and poll --
     * poll kernel will wait for SQE visibility implicitly through
     * the doorbell write.  But submit has to actually complete
     * before poll runs; the default stream serialises this. */

    /* 5-second budget at ~10 ns per iteration -> 5e8 iters; be
     * generous since GPU is slower than CPU on a tight spin loop. */
    constexpr uint64_t MAX_ITERS = 50000000ULL;
    k_poll_one<<<1, 1>>>(qs.dev, qs.cq_head_um, qs.cq_phase_um,
                         qs.out_cqe_um, qs.timed_out_um, MAX_ITERS);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "k_poll_one launch: %s\n", cudaGetErrorString(e));
        return -EIO;
    }
    CUDA_OK(cudaDeviceSynchronize());

    if (*qs.timed_out_um) return -ETIMEDOUT;
    *cqe_out = *qs.out_cqe_um;
    return 0;
}

/* ------------------------------------------------------------------ */
/* All per-round resources, allocated/freed by run_one_round().       */
/* B6: only the queue group + the SQ/CQ rings are per-round.  Data    */
/* buffers (wbuf / rbuf / prp_list_*) live in persistent_data_res     */
/* below and span every round on this fd.                             */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【结构体】struct round_resources —— 单轮(round)专属的资源集合
 * 【作用】把“每一轮都要重新创建、轮末又销毁”的东西打包：队列组 + 两对 SQ/CQ
 *         GPU 环及其 IO 地址 + 两个队列的运行状态。用于反复申请/释放以验证回收路径。
 * 【字段】
 *   group_id            —— 本轮 NVM_CREATE_QUEUE_GROUP 得到的队列组 ID。
 *   sq_dev[]/cq_dev[]   —— 每个队列的 SQ/CQ 环在 GPU 显存的地址（cudaMalloc 得到）。
 *   sq_ioaddr[]/cq_ioaddr[] —— 上述环经 NVM_MAP_DEVICE_MEMORY 注册后给控制器用的 IO 地址。
 *   QS[]                —— 每个队列的完整运行状态（queue_state）。
 * 【在测试中的角色】run_one_round 把它填满，teardown_one_round 再清空，是“动态分配/释放循环”的载体。
 * 【新手提示】数据缓冲不在这里——它们是跨轮持久的，放在 persistent_data_resources 里。
 * 【NPU 迁移提示】sq_dev/cq_dev 来自 cudaMalloc → 昇腾换 aclrtMalloc；ioaddr 来自
 *               NVM_MAP_DEVICE_MEMORY（依赖 nvidia_p2p_* 把显存暴露给 NVMe DMA）→ 昇腾需要
 *               华为 peer DMA 接口把 device 内存注册成 NVMe 可访问的总线地址。
 * ──────────────────────────────────────────────────────────── */
struct round_resources {
    uint32_t    group_id;
    void*       sq_dev[TEST_NR_QUEUES];
    void*       cq_dev[TEST_NR_QUEUES];
    uint64_t    sq_ioaddr[TEST_NR_QUEUES];
    uint64_t    cq_ioaddr[TEST_NR_QUEUES];
    queue_state QS[TEST_NR_QUEUES];
};

/* ------------------------------------------------------------------ */
/* B6: persistent fd-scoped data resources.  Allocated once in main(),*/
/* registered with map_kind=NVM_MAP_KIND_DATA + group_id=0 so the     */
/* kernel parks them on snvm_dev_owner.data_maps and they survive    */
/* every NVM_DESTROY_QUEUE_GROUP cascade.  Released on close(fd_dev). */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【结构体】struct persistent_data_resources —— 跨轮持久的数据缓冲资源
 * 【作用】打包“整个 fd 生命周期只申请一次、所有轮共用”的数据面缓冲：
 *         写缓冲、读缓冲、两个 PRP_List 缓冲，及它们各自的 IO 地址。
 * 【字段】
 *   wbuf_dev / rbuf_dev               —— 写/读数据缓冲的 GPU 显存地址。
 *   prp_list_w_dev / prp_list_r_dev   —— 写/读用的 PRP_List 缓冲 GPU 显存地址（Tier3 大 IO 用）。
 *   *_ioaddr                          —— 上述各缓冲注册后给控制器 DMA 用的 IO 地址。
 * 【在测试中的角色】在 main 的 Phase 2b 一次性分配并以 map_kind=DATA、group_id=0 注册，
 *                 之后每轮 IO 都直接复用，不随队列组销毁而释放，直到 close(fd) 才回收。
 * 【新手提示】PRP_List 是一张“地址表”：当一次 IO 跨多个页、prp1/prp2 装不下时，prp2 改指
 *           向这张表，表里列出后续各页的物理地址。
 * 【NPU 迁移提示】*_dev 来自 cudaMalloc → 昇腾 aclrtMalloc；*_ioaddr 来自
 *               NVM_MAP_DEVICE_MEMORY（NVIDIA p2p get_pages）→ 昇腾需用华为 peer DMA 接口
 *               注册 device 内存供 NVMe 直接读写。“注册一次、多组复用”的模式可保留。
 * ──────────────────────────────────────────────────────────── */
struct persistent_data_resources {
    void*       wbuf_dev;
    void*       rbuf_dev;
    void*       prp_list_w_dev;
    void*       prp_list_r_dev;
    uint64_t    wbuf_ioaddr;
    uint64_t    rbuf_ioaddr;
    uint64_t    prp_list_w_ioaddr;
    uint64_t    prp_list_r_ioaddr;
};

/* ────────────────────────────────────────────────────────────
 * 【函数】run_one_round
 * 【作用】跑完一整轮的数据面：建队列组、分配并注册 GPU 环、创建用户 IO 队列、
 *         构造每队列设备状态，然后做 Tier1~Tier4 + SQ 回绕压力测试的读写校验。
 * 【参数】
 *   fd_dev    —— 已绑定控制器的设备 fd。
 *   info      —— 控制器信息（块大小、队列深度、SGL 支持等）。
 *   bar0_gpu  —— BAR0 在 GPU 地址空间的基址，用于算出每队列门铃地址。
 *   round_idx —— 当前轮号，决定本轮独占的 LBA 窗口和数据模式条带。
 *   rr        —— 输出：本轮资源集合（调用方传入已清零的结构）。
 *   pdata     —— 跨轮持久的数据缓冲（读/写/PRP_List），本函数只用不分配。
 * 【返回】无（成功返回；任何一步失败直接 step_fail 退出进程）。
 * 【在测试中的角色】整个测试的核心一轮，按以下阶段推进：
 *   Phase R.1 创建队列组(NVM_CREATE_QUEUE_GROUP)；
 *   Phase R.2 cudaMalloc 各队列 SQ/CQ GPU 环（数据缓冲复用 pdata）；
 *   Phase R.3 NVM_MAP_DEVICE_MEMORY 注册环（RING_SQ/RING_CQ），拿到 IO 地址；
 *   Phase R.4 NVM_ADD_USER_QUEUE 让控制器创建用户队列，拿回 qid 和门铃偏移，
 *             组装 test_queue_dev 并分配统一内存计数器；
 *   Phase R.5 Tier1：4 KiB，仅 PRP1，写+读回校验 16 次×2 队列；
 *   Phase R.6 Tier2：8 KiB，PRP1+PRP2；
 *   Phase R.7 Tier3：16 KiB，PRP1+PRP_List（先把后续页地址拷进 PRP_List 缓冲）；
 *   Phase R.8 Tier4：SGL Data Block（控制器不支持 SGL 则跳过）；
 *   Phase R.9 SQ-tail-wrap：连发 q_depth+8 个写，强制 SQ 环回绕、CQ 相位翻转。
 * 【新手提示】每个 Tier 用更大的 IO 验证不同的数据描述方式（PRP1 / PRP1+PRP2 / PRP_List / SGL）；
 *           每轮用不重叠的 LBA 窗口，使各轮校验互不干扰。
 * 【NPU 迁移提示】本函数大量使用 CUDA 专有调用，迁移昇腾时需逐一替换：
 *   - cudaMalloc/cudaMemset/cudaMallocManaged → aclrtMalloc / aclrtMemset / 昇腾统一内存接口；
 *   - NVM_MAP_DEVICE_MEMORY（依赖 nvidia_p2p_get_pages 把显存暴露给 NVMe）→ 华为 peer DMA 注册接口；
 *   - 门铃地址 = bar0_gpu + offset，依赖 BAR0 被映射成 AIV 可见地址；
 *   - k_fill_pattern/k_verify_pattern/submit_and_poll 内部的核函数 → Ascend C / AIV kernel。
 *   ioctl 命令号（CREATE_QUEUE_GROUP / MAP_DEVICE_MEMORY / ADD_USER_QUEUE）是 snvme 专有，需对接华为驱动。
 * ──────────────────────────────────────────────────────────── */
/* Run all the per-round IO phases (formerly Phase 2..10).  Caller
 * supplies a fresh `rr` (zeroed) and the controller-wide state
 * (fd_dev, info, bar0_gpu).  Returns 0 on success, exits on error.
 *
 * `round_idx` selects a unique LBA window plus the per-byte
 * pattern stripe.  `kernel_ioq_cap` drives NVM_SET_KERNEL_IOQ_CAP
 * for the very first round only -- subsequent rounds reuse the cap
 * already negotiated with the controller (snvme keeps it across
 * group destroy/create cycles within a single bind).             */
static void run_one_round(int fd_dev,
                          const struct nvm_ioctl_dev& info,
                          void* bar0_gpu,
                          unsigned round_idx,
                          round_resources& rr,
                          const persistent_data_resources& pdata) {
    /* ============================================================== */
    /* Phase R.1: queue group.                                        */
    /* ============================================================== */
    {
        struct nvm_ioctl_queue_group req;
        memset(&req, 0, sizeof(req));
        if (do_ioctl(fd_dev, NVM_CREATE_QUEUE_GROUP, &req,
                     "NVM_CREATE_QUEUE_GROUP") < 0)
            step_fail(errno, "round=%u NVM_CREATE_QUEUE_GROUP", round_idx);
        rr.group_id = req.group_id;
        step_ok("round=%u NVM_CREATE_QUEUE_GROUP -> group_id=%u max_queues=%u",
                round_idx, rr.group_id, req.max_queues);
    }

    /* ============================================================== */
    /* Phase R.2: cudaMalloc rings only.  The data + PRP_List buffers */
    /* live in `pdata` and were registered ONCE in main() with        */
    /* map_kind=DATA + group_id=0; they outlive every per-round       */
    /* group and are reaped only at fd close.                         */
    /* ============================================================== */
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        CUDA_OK(cudaMalloc(&rr.sq_dev[i], GPU_PAGE_SIZE));
        CUDA_OK(cudaMalloc(&rr.cq_dev[i], GPU_PAGE_SIZE));
        CUDA_OK(cudaMemset(rr.sq_dev[i], 0, GPU_PAGE_SIZE));
        CUDA_OK(cudaMemset(rr.cq_dev[i], 0, GPU_PAGE_SIZE));
    }
    step_ok("round=%u cudaMalloc'd %u SQ + %u CQ GPU pages "
            "(%zu B each); reusing persistent wbuf/rbuf/prp_list",
            round_idx, TEST_NR_QUEUES, TEST_NR_QUEUES, GPU_PAGE_SIZE);

    /* ============================================================== */
    /* Phase R.3: NVM_MAP_DEVICE_MEMORY for the rings only, with      */
    /* explicit kind tags so NVM_ADD_USER_QUEUE will accept them and  */
    /* refuse any data-buffer vaddr in the same slot.                 */
    /* ============================================================== */
    auto map_ring = [&](void* gpu_va, unsigned long n_pages,
                        uint64_t* ioaddrs_out, uint8_t kind,
                        const char* what) {
        struct nvm_ioctl_map req;
        memset(&req, 0, sizeof(req));
        req.vaddr_start = (uint64_t)(uintptr_t)gpu_va;
        req.n_pages     = n_pages;
        req.ioaddrs     = ioaddrs_out;
        req.ioq_idx     = -1;
        req.is_cq       = -1;
        req.group_id    = rr.group_id;
        req.map_kind    = kind;
        if (do_ioctl(fd_dev, NVM_MAP_DEVICE_MEMORY, &req, what) < 0)
            step_fail(errno, "round=%u %s gpu_va=%p", round_idx, what, gpu_va);
    };

    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        map_ring(rr.sq_dev[i], 1, &rr.sq_ioaddr[i],
                 NVM_MAP_KIND_RING_SQ,
                 "NVM_MAP_DEVICE_MEMORY(SQ)");
        map_ring(rr.cq_dev[i], 1, &rr.cq_ioaddr[i],
                 NVM_MAP_KIND_RING_CQ,
                 "NVM_MAP_DEVICE_MEMORY(CQ)");
    }
    step_ok("round=%u NVM_MAP_DEVICE_MEMORY x %u ring(s) (RING_SQ/RING_CQ); "
            "data buffers (wbuf_ioaddr=0x%llx) come from persistent pool",
            round_idx, TEST_NR_QUEUES * 2,
            (unsigned long long)pdata.wbuf_ioaddr);

    /* ============================================================== */
    /* Phase R.4: NVM_ADD_USER_QUEUE.                                */
    /* ============================================================== */
    struct nvm_ioctl_add_user_queue add_req;
    memset(&add_req, 0, sizeof(add_req));
    add_req.group_id = rr.group_id;
    add_req.nr_pairs = TEST_NR_QUEUES;
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        add_req.pairs[i].sq_vaddr = (uint64_t)(uintptr_t)rr.sq_dev[i];
        add_req.pairs[i].cq_vaddr = (uint64_t)(uintptr_t)rr.cq_dev[i];
    }
    if (do_ioctl(fd_dev, NVM_ADD_USER_QUEUE, &add_req,
                 "NVM_ADD_USER_QUEUE") < 0)
        step_fail(errno, "round=%u NVM_ADD_USER_QUEUE", round_idx);
    step_ok("round=%u NVM_ADD_USER_QUEUE created %u user queue(s)",
            round_idx, TEST_NR_QUEUES);
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++)
        fprintf(stderr, "                pair[%u] qid=%u sq_db=0x%x cq_db=0x%x\n",
                i, add_req.out_pairs[i].qid,
                add_req.out_pairs[i].sq_doorbell_offset,
                add_req.out_pairs[i].cq_doorbell_offset);

    /* Build per-queue test_queue_dev structs (consumed by kernels).  */
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        rr.QS[i].dev.sq      = (nvme_sqe*)rr.sq_dev[i];
        rr.QS[i].dev.cq      = (nvme_cqe*)rr.cq_dev[i];
        rr.QS[i].dev.q_depth = info.q_depth;
        rr.QS[i].dev.qid     = (uint16_t)add_req.out_pairs[i].qid;
        rr.QS[i].dev.sq_db   = (volatile uint32_t*)
            ((char*)bar0_gpu + add_req.out_pairs[i].sq_doorbell_offset);
        rr.QS[i].dev.cq_db   = (volatile uint32_t*)
            ((char*)bar0_gpu + add_req.out_pairs[i].cq_doorbell_offset);

        /* Unified-memory counters / out-cqe so submit/poll kernels can
         * read+update them in place; avoids host<->device copies on
         * every IO.                                                  */
        CUDA_OK(cudaMallocManaged(&rr.QS[i].sq_tail_um,   sizeof(uint16_t)));
        CUDA_OK(cudaMallocManaged(&rr.QS[i].cq_head_um,   sizeof(uint16_t)));
        CUDA_OK(cudaMallocManaged(&rr.QS[i].cq_phase_um,  sizeof(uint8_t)));
        CUDA_OK(cudaMallocManaged(&rr.QS[i].out_cqe_um,   sizeof(nvme_cqe)));
        CUDA_OK(cudaMallocManaged(&rr.QS[i].timed_out_um, sizeof(int)));
        *rr.QS[i].sq_tail_um   = 0;
        *rr.QS[i].cq_head_um   = 0;
        *rr.QS[i].cq_phase_um  = 1;
        *rr.QS[i].timed_out_um = 0;
        rr.QS[i].next_cid = 0;
    }
    step_ok("round=%u per-queue device state ready (%u queues)",
            round_idx, TEST_NR_QUEUES);

    /* The data buffer's dma_addr lets us derive 4-KiB-grained NVMe
     * page addresses for tier 2/3.                                  */
    auto wpage = [&](unsigned npage) -> uint64_t {
        return pdata.wbuf_ioaddr + (uint64_t)npage * info.block_size;
    };
    auto rpage = [&](unsigned npage) -> uint64_t {
        return pdata.rbuf_ioaddr + (uint64_t)npage * info.block_size;
    };

    /* This round's LBA window starts here.  Each tier carves out a
     * disjoint sub-range.                                            */
    const uint64_t round_lba_base =
        TEST_LBA_BASE + (uint64_t)round_idx * TEST_LBA_PER_ROUND;

    /* ============================================================== */
    /* Phase R.5: Tier 1 -- 4 KiB IO, PRP1 only.                     */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;
        const size_t io_bytes = info.block_size;
        queue_state& qw = rr.QS[0];
        queue_state& qr = rr.QS[1];
        char status_buf[64];

        int* mismatch_um = nullptr;
        CUDA_OK(cudaMallocManaged(&mismatch_um, sizeof(int)));

        for (unsigned i = 0; i < TEST_NR_IO_PER_QUEUE; i++) {
            uint64_t lba = round_lba_base + i;
            uint8_t pat  = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            nvme_cqe cqe;
            uint16_t cid_w;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                                     nsid, wpage(0), 0,
                                     lba, nlb_zero_based, &cqe, &cid_w);
            if (rc) step_fail(-rc, "round=%u T1 Write %u (qid=%u, lba=%" PRIu64 ")",
                              round_idx, i, qw.dev.qid, lba);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T1 Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }
            if (cqe.cid != cid_w)
                step_fail(0, "round=%u T1 Write %u CQE.cid=%u expected %u",
                          round_idx, i, cqe.cid, cid_w);

            CUDA_OK(cudaMemset(pdata.rbuf_dev, 0, io_bytes));
            uint16_t cid_r;
            rc = submit_and_poll(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP,
                                 nsid, rpage(0), 0,
                                 lba, nlb_zero_based, &cqe, &cid_r);
            if (rc) step_fail(-rc, "round=%u T1 Read %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T1 Read %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            *mismatch_um = -1;
            k_verify_pattern<<<blocks, threads>>>((const uint8_t*)pdata.rbuf_dev,
                                                   io_bytes, pat, mismatch_um);
            CUDA_OK(cudaDeviceSynchronize());
            if (*mismatch_um != -1)
                step_fail(0, "round=%u T1 IO %u: byte %d mismatch (lba=%"
                          PRIu64 ")", round_idx, i, *mismatch_um, lba);
        }
        cudaFree(mismatch_um);
        step_ok("round=%u Tier 1 (PRP1, 4 KiB) write+verify x %u IOs, "
                "LBA [%" PRIu64 "..%" PRIu64 "]",
                round_idx, TEST_NR_IO_PER_QUEUE,
                round_lba_base,
                round_lba_base + TEST_NR_IO_PER_QUEUE - 1);
    }

    /* ============================================================== */
    /* Phase R.6: Tier 2 -- 8 KiB IO, PRP1 + PRP2.                   */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 1;          /* 2 LBAs per IO */
        const size_t   io_bytes = 2 * info.block_size;
        const uint64_t LBA_BASE = round_lba_base + 100;
        const unsigned NR = 8;
        queue_state& qw = rr.QS[0];
        queue_state& qr = rr.QS[1];
        char status_buf[64];

        int* mismatch_um = nullptr;
        CUDA_OK(cudaMallocManaged(&mismatch_um, sizeof(int)));

        for (unsigned i = 0; i < NR; i++) {
            uint64_t lba = LBA_BASE + 2u * i;
            uint8_t  pat = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, 100 + i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            nvme_cqe cqe;
            uint16_t cid_w;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                                     nsid, wpage(0), wpage(1),
                                     lba, nlb_zero_based, &cqe, &cid_w);
            if (rc) step_fail(-rc, "round=%u T2 Write %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T2 Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            CUDA_OK(cudaMemset(pdata.rbuf_dev, 0, io_bytes));
            uint16_t cid_r;
            rc = submit_and_poll(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP,
                                 nsid, rpage(0), rpage(1),
                                 lba, nlb_zero_based, &cqe, &cid_r);
            if (rc) step_fail(-rc, "round=%u T2 Read %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T2 Read %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            *mismatch_um = -1;
            k_verify_pattern<<<blocks, threads>>>((const uint8_t*)pdata.rbuf_dev,
                                                   io_bytes, pat, mismatch_um);
            CUDA_OK(cudaDeviceSynchronize());
            if (*mismatch_um != -1)
                step_fail(0, "round=%u T2 IO %u: byte %d mismatch",
                          round_idx, i, *mismatch_um);
        }
        cudaFree(mismatch_um);
        step_ok("round=%u Tier 2 (PRP1+PRP2, 8 KiB) x %u IOs, LBA [%"
                PRIu64 "..%" PRIu64 "]",
                round_idx, NR, LBA_BASE, LBA_BASE + 2u * (NR - 1) + 1);
    }

    /* ============================================================== */
    /* Phase R.7: Tier 3 -- 16 KiB IO, PRP1 + PRP_List.              */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 3;          /* 4 LBAs per IO */
        const size_t   io_bytes = 4 * info.block_size;
        const uint64_t LBA_BASE = round_lba_base + 200;
        const unsigned NR = 4;
        queue_state& qw = rr.QS[0];
        queue_state& qr = rr.QS[1];
        char status_buf[64];

        uint64_t prp_w_entries[3] = { wpage(1), wpage(2), wpage(3) };
        uint64_t prp_r_entries[3] = { rpage(1), rpage(2), rpage(3) };
        CUDA_OK(cudaMemcpy(pdata.prp_list_w_dev, prp_w_entries,
                           sizeof(prp_w_entries), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(pdata.prp_list_r_dev, prp_r_entries,
                           sizeof(prp_r_entries), cudaMemcpyHostToDevice));

        int* mismatch_um = nullptr;
        CUDA_OK(cudaMallocManaged(&mismatch_um, sizeof(int)));

        for (unsigned i = 0; i < NR; i++) {
            uint64_t lba = LBA_BASE + 4u * i;
            uint8_t  pat = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, 200 + i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            nvme_cqe cqe;
            uint16_t cid_w;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                                     nsid, wpage(0), pdata.prp_list_w_ioaddr,
                                     lba, nlb_zero_based, &cqe, &cid_w);
            if (rc) step_fail(-rc, "round=%u T3 Write %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T3 Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            CUDA_OK(cudaMemset(pdata.rbuf_dev, 0, io_bytes));
            uint16_t cid_r;
            rc = submit_and_poll(qr, NVME_OPC_READ, NVME_FLAG_PSDT_PRP,
                                 nsid, rpage(0), pdata.prp_list_r_ioaddr,
                                 lba, nlb_zero_based, &cqe, &cid_r);
            if (rc) step_fail(-rc, "round=%u T3 Read %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T3 Read %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            *mismatch_um = -1;
            k_verify_pattern<<<blocks, threads>>>((const uint8_t*)pdata.rbuf_dev,
                                                   io_bytes, pat, mismatch_um);
            CUDA_OK(cudaDeviceSynchronize());
            if (*mismatch_um != -1)
                step_fail(0, "round=%u T3 IO %u: byte %d mismatch",
                          round_idx, i, *mismatch_um);
        }
        cudaFree(mismatch_um);
        step_ok("round=%u Tier 3 (PRP1+PRP_List, 16 KiB) x %u IOs, LBA [%"
                PRIu64 "..%" PRIu64 "]",
                round_idx, NR, LBA_BASE, LBA_BASE + 4u * (NR - 1) + 3);
    }

    /* ============================================================== */
    /* Phase R.8: Tier 4 -- SGL Data Block (skipped on PRP-only).    */
    /* ============================================================== */
    if ((info.sgl_supported & 0x3) == 0) {
        step_ok("round=%u Tier 4: SKIP -- controller advertises SGLS=0x%x "
                "(PRP-only)", round_idx, info.sgl_supported);
    } else {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;
        const size_t io_bytes = info.block_size;
        const uint64_t LBA_BASE = round_lba_base + 300;
        const unsigned NR = 8;
        queue_state& qw = rr.QS[0];
        queue_state& qr = rr.QS[1];
        char status_buf[64];

        int* mismatch_um = nullptr;
        CUDA_OK(cudaMallocManaged(&mismatch_um, sizeof(int)));

        for (unsigned i = 0; i < NR; i++) {
            uint64_t lba = LBA_BASE + i;
            uint8_t pat = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, 300 + i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            uint64_t sgl_addr_w = wpage(0);
            uint64_t sgl_meta_w = ((uint64_t)io_bytes & 0xffffffffu)
                                | ((uint64_t)NVME_SGL_DESC_BYTE15 << 56);
            nvme_cqe cqe;
            uint16_t cid_w;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_SGL,
                                     nsid, sgl_addr_w, sgl_meta_w,
                                     lba, nlb_zero_based, &cqe, &cid_w);
            if (rc) step_fail(-rc, "round=%u T4 Write %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T4 Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            CUDA_OK(cudaMemset(pdata.rbuf_dev, 0, io_bytes));
            uint64_t sgl_addr_r = rpage(0);
            uint64_t sgl_meta_r = ((uint64_t)io_bytes & 0xffffffffu)
                                | ((uint64_t)NVME_SGL_DESC_BYTE15 << 56);
            uint16_t cid_r;
            rc = submit_and_poll(qr, NVME_OPC_READ, NVME_FLAG_PSDT_SGL,
                                 nsid, sgl_addr_r, sgl_meta_r,
                                 lba, nlb_zero_based, &cqe, &cid_r);
            if (rc) step_fail(-rc, "round=%u T4 Read %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u T4 Read %u: NVMe %s",
                          round_idx, i, status_buf);
            }

            *mismatch_um = -1;
            k_verify_pattern<<<blocks, threads>>>((const uint8_t*)pdata.rbuf_dev,
                                                   io_bytes, pat, mismatch_um);
            CUDA_OK(cudaDeviceSynchronize());
            if (*mismatch_um != -1)
                step_fail(0, "round=%u T4 IO %u: byte %d mismatch",
                          round_idx, i, *mismatch_um);
        }
        cudaFree(mismatch_um);
        step_ok("round=%u Tier 4 (SGL Data Block, 4 KiB) x %u IOs, LBA [%"
                PRIu64 "..%" PRIu64 "]",
                round_idx, NR, LBA_BASE, LBA_BASE + NR - 1);
    }

    /* ============================================================== */
    /* Phase R.9: SQ-tail-wrap stress on QS[0].                      */
    /* ============================================================== */
    {
        const uint32_t nsid = 1;
        const uint16_t nlb_zero_based = 0;
        queue_state& qw = rr.QS[0];
        char status_buf[64];

        unsigned cnt = info.q_depth + 8u;
        const uint64_t LBA_BASE = round_lba_base + 1000;
        const size_t io_bytes = info.block_size;

        for (unsigned i = 0; i < cnt; i++) {
            uint64_t lba = LBA_BASE + i;
            uint8_t pat = WRITE_PATTERN_BYTE(round_idx, qw.dev.qid, 1000u + i);

            int threads = 256, blocks = (int)((io_bytes + threads - 1) / threads);
            k_fill_pattern<<<blocks, threads>>>((uint8_t*)pdata.wbuf_dev,
                                                io_bytes, pat);
            CUDA_OK(cudaDeviceSynchronize());

            nvme_cqe cqe;
            uint16_t cid;
            int rc = submit_and_poll(qw, NVME_OPC_WRITE, NVME_FLAG_PSDT_PRP,
                                     nsid, wpage(0), 0,
                                     lba, nlb_zero_based, &cqe, &cid);
            if (rc) step_fail(-rc, "round=%u wrap Write %u", round_idx, i);
            if ((cqe.status >> 1) != 0) {
                format_status(cqe.status, status_buf, sizeof(status_buf));
                step_fail(0, "round=%u wrap Write %u: NVMe %s",
                          round_idx, i, status_buf);
            }
        }
        step_ok("round=%u SQ-tail-wrap: %u sequential Writes (sq wrapped "
                "past q_depth=%u, cq_phase flipped)",
                round_idx, cnt, info.q_depth);
    }
}

/* ────────────────────────────────────────────────────────────
 * 【函数】teardown_one_round
 * 【作用】把 run_one_round 在本轮建立的东西全部拆掉，为下一轮重新分配做准备。
 * 【参数】
 *   fd_dev    —— 设备 fd。
 *   round_idx —— 轮号（仅用于日志）。
 *   rr        —— 本轮资源集合；函数末尾会整体清零。
 * 【返回】无（失败时 step_fail 退出）。
 * 【在测试中的角色】与 run_one_round 配对，按严格顺序回收（顺序很关键）：
 *   Phase R.1（拆）先 cudaFree 每队列的统一内存计数器（sq_tail/cq_head/phase/out_cqe/timed_out）；
 *   再发 NVM_DESTROY_QUEUE_GROUP，由它级联删除控制器侧的用户 IO 队列(Delete I/O SQ/CQ)
 *   和 4 个环映射(NVM_MAP_DEVICE_MEMORY 描述符)；
 *   最后 cudaFree 各 SQ/CQ GPU 环页；末尾 memset 清零 rr。
 *   数据缓冲(pdata)不在此释放——它们跨轮存活，到 close(fd) 才回收。
 * 【新手提示】先销毁队列组再释放显存：让控制器先停止访问这些环，避免它还在 DMA 时显存被回收。
 * 【NPU 迁移提示】cudaFree → aclrtFree；NVM_DESTROY_QUEUE_GROUP 是 snvme 专有 ioctl，
 *               迁移时需对接华为驱动的队列组销毁接口，且其内部必须正确调用 peer DMA 的
 *               put_pages 释放对显存的引用（对应 NVIDIA 的 nvidia_p2p_put_pages 引用计数）。
 *               这条释放路径正是“动态分配/释放循环”要验证的引用计数回收正确性所在。
 * ──────────────────────────────────────────────────────────── */
/* Tear down everything that run_one_round() built up.  In B6 this
 * is just the queue group + its 4 ring maps + the 2 user IO queues
 * the controller created.  Data buffers (wbuf / rbuf / prp_list_*)
 * stay alive across rounds and are reaped at fd close by
 * snvm_dev_release walking the per-fd data_maps list.  Order
 * matters: free per-queue UM bookkeeping FIRST, then
 * NVM_DESTROY_QUEUE_GROUP (cascades through user queues + the 4
 * ring maps), then cudaFree the GPU ring pages.                 */
static void teardown_one_round(int fd_dev, unsigned round_idx,
                               round_resources& rr) {
    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        cudaFree(rr.QS[i].sq_tail_um);
        cudaFree(rr.QS[i].cq_head_um);
        cudaFree(rr.QS[i].cq_phase_um);
        cudaFree(rr.QS[i].out_cqe_um);
        cudaFree(rr.QS[i].timed_out_um);
    }

    {
        uint32_t gid = rr.group_id;
        if (do_ioctl(fd_dev, NVM_DESTROY_QUEUE_GROUP, &gid,
                     "NVM_DESTROY_QUEUE_GROUP") < 0)
            step_fail(errno, "round=%u NVM_DESTROY_QUEUE_GROUP", round_idx);
        step_ok("round=%u NVM_DESTROY_QUEUE_GROUP id=%u cascades through %u "
                "user queue(s) + %u ring map(s); data maps persist",
                round_idx, rr.group_id, TEST_NR_QUEUES,
                TEST_NR_QUEUES * 2);
    }

    for (unsigned i = 0; i < TEST_NR_QUEUES; i++) {
        cudaFree(rr.sq_dev[i]);
        cudaFree(rr.cq_dev[i]);
    }

    memset(&rr, 0, sizeof(rr));
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

/* ────────────────────────────────────────────────────────────
 * 【函数】main
 * 【作用】整个 GPU 版端到端 NVMe IO smoke 测试的入口：解析参数、初始化 CUDA、
 *         打开并绑定 snvme 控制器、映射 BAR0、分配持久数据缓冲，然后跑 N 轮
 *         队列组的“建立—IO—销毁”循环，最后清理收尾。
 * 【参数】
 *   argc/argv —— 命令行参数：[--gpu N] [--rounds N] <PCI_BDF>。
 * 【返回】0 全部通过；1 用法错误；2 某步骤失败（由 step_fail 触发）。
 * 【在测试中的角色】按以下完整流程把各部件串起来：
 *   解析参数 + parse_bdf；cudaSetDevice 选 GPU。
 *   Phase 0 控制面：open(/dev/snvm_control)、SNVM_CHRDEV_CREATE 创建字符设备、open(/dev/ssnvmeN)。
 *   Phase 1 绑定层：NVM_SET_KERNEL_IOQ_CAP 设内核 IOQ 上限、SNVM_DEVICE_BIND 绑定控制器、
 *           NVM_GET_DEV_INFO 取设备信息（含 4 KiB 块大小、队列深度等前置校验）。
 *   Phase 2 门铃：mmap BAR0 到 CPU，再 cudaHostRegister(IoMemory)+cudaHostGetDevicePointer
 *           得到 GPU 可见的 BAR0 地址，让核函数能直接写门铃。
 *   Phase 2b 持久数据缓冲：一次性 cudaMalloc 读/写/PRP_List 缓冲，以 map_kind=DATA、
 *           group_id=0 注册，使其跨所有轮存活。
 *   Phase 3+ rounds：循环 nr_rounds 次，每次 run_one_round + teardown_one_round。
 *   收尾：cudaHostUnregister + munmap BAR0；SNVM_DEVICE_UNBIND；close(fd_dev)（触发
 *        snvm_dev_release 释放 DATA 映射的 p2p 引用）；之后才 cudaFree 持久缓冲（顺序关键，
 *        否则 fput 时显存仍被 snvme 引用即泄漏）；SNVM_CHRDEV_REMOVE；close(fd_ctl)。
 * 【新手提示】前置条件（见文件头）：必须先加载 NVIDIA 驱动且 nvfs_nvidia_p2p_init() 成功，
 *           否则 snvme 拒绝加载——因为 GPUDirect 路径依赖 NVIDIA p2p 把显存暴露给 NVMe。
 *           本测试是破坏性的，会真往磁盘写数据。
 * 【NPU 迁移提示】main 串起了 NVIDIA GPUDirect 全套，迁移昇腾时的主要替换点：
 *   - cudaSetDevice/cudaGetDeviceProperties → aclrtSetDevice 等 ACL 设备初始化；
 *   - mmap BAR0 + cudaHostRegister(cudaHostRegisterIoMemory) + cudaHostGetDevicePointer
 *     → 把 NVMe BAR0 注册成 AIV 可见地址并取设备指针（这是让核函数写门铃的关键能力）；
 *   - cudaMalloc 持久缓冲 + NVM_MAP_DEVICE_MEMORY（NVIDIA p2p get/put_pages）
 *     → aclrtMalloc + 华为 peer DMA 注册/反注册接口；
 *   - 释放顺序“先解除设备对显存的 DMA 引用，再 free 显存”这一原则在昇腾上同样必须遵守，
 *     否则同样会泄漏（对应 NVIDIA nvidia_p2p_put_pages 引用计数）。
 *   - 所有 SNVM_ / NVM_ 系列 ioctl 是 snvme 专有，需对接华为侧驱动的等价控制接口。
 * ──────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    int cuda_device = 0;
    unsigned nr_rounds = TEST_DEFAULT_ROUNDS;
    const char* bdf_str = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            cuda_device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
            int n = atoi(argv[++i]);
            if (n <= 0) { usage(argv[0]); return 1; }
            nr_rounds = (unsigned)n;
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else if (argv[i][0] != '-' && !bdf_str) {
            bdf_str = argv[i];
        } else {
            usage(argv[0]); return 1;
        }
    }
    if (!bdf_str) { usage(argv[0]); return 1; }

    struct pci_device_addr orig_bdf;
    if (parse_bdf(bdf_str, &orig_bdf) != 0) {
        fprintf(stderr, "Bad BDF: '%s' (expected DDDD:BB:DD.F)\n", bdf_str);
        return 1;
    }

    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0) step_fail(errno, "sysconf(_SC_PAGESIZE)");

    CUDA_OK(cudaSetDevice(cuda_device));
    int n_dev = 0;
    CUDA_OK(cudaGetDeviceCount(&n_dev));
    if (cuda_device >= n_dev)
        step_fail(0, "--gpu %d invalid (CUDA sees %d devices)",
                  cuda_device, n_dev);
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, cuda_device));
    step_ok("CUDA setDevice(%d) name='%s' cap=%d.%d  rounds=%u",
            cuda_device, prop.name, prop.major, prop.minor, nr_rounds);

    /* ============================================================== */
    /* Phase 0: control plane + chrdev (lives across all rounds).    */
    /* ============================================================== */
    int fd_ctl = open("/dev/snvm_control", O_RDWR | O_NONBLOCK);
    if (fd_ctl < 0) step_fail(errno, "open(/dev/snvm_control)");
    step_ok("open(/dev/snvm_control) fd=%d", fd_ctl);

    struct pci_device_addr addr = orig_bdf;
    if (do_ioctl(fd_ctl, SNVM_CHRDEV_CREATE, &addr, "SNVM_CHRDEV_CREATE") < 0)
        step_fail(errno, "SNVM_CHRDEV_CREATE %s", bdf_str);
    int minor_n = addr.domain;
    step_ok("SNVM_CHRDEV_CREATE minor=%d", minor_n);

    char dev_path[64];
    snprintf(dev_path, sizeof(dev_path), "/dev/ssnvme%d", minor_n);
    int fd_dev = open(dev_path, O_RDWR);
    if (fd_dev < 0) step_fail(errno, "open(%s)", dev_path);
    step_ok("open(%s) fd=%d", dev_path, fd_dev);

    /* ============================================================== */
    /* Phase 1: kernel ioq cap + bind + dev info.  These are bind-   */
    /* level state, NOT per-round; they persist across rounds.       */
    /* ============================================================== */
    {
        uint32_t cap = 36;
        if (ioctl(fd_dev, NVM_SET_KERNEL_IOQ_CAP, &cap) != 0)
            step_fail(errno, "NVM_SET_KERNEL_IOQ_CAP cap=%u failed", cap);
        step_ok("NVM_SET_KERNEL_IOQ_CAP cap=%u", cap);
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
                info.disk_name[0] != '\0') { ok = 1; break; }
            usleep(100 * 1000);
        }
        if (!ok) step_fail(errno, "NVM_GET_DEV_INFO did not complete in 10s");
        step_ok("NVM_GET_DEV_INFO disk='%s' block_size=%zu q_depth=%u "
                "start_cq_idx=%u max_user_qid=%u sgls=0x%x",
                info.disk_name, info.block_size, info.q_depth,
                info.start_cq_idx, info.max_user_qid, info.sgl_supported);
    }
    if (info.block_size != 4096)
        step_fail(0, "smoke assumes 4 KiB-LBA controller; got %zu",
                  info.block_size);
    if ((size_t)info.q_depth * NVME_SQE_SIZE > GPU_PAGE_SIZE)
        step_fail(0, "GPU smoke: SQ ring (q_depth=%u * 64 = %zu B) "
                     "exceeds one GPU page (%zu B); lower io_queue_depth.",
                  info.q_depth,
                  (size_t)info.q_depth * NVME_SQE_SIZE,
                  GPU_PAGE_SIZE);

    /* ============================================================== */
    /* Phase 2: BAR0 mmap + cudaHostRegister (one-shot, lives across */
    /* rounds because doorbell offsets are stable per QID and QIDs   */
    /* are reused predictably by snvme's user_qid pool).              */
    /* ============================================================== */
    void* bar0_cpu = mmap(NULL, info.bar0_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd_dev, 0);
    if (bar0_cpu == MAP_FAILED)
        step_fail(errno, "mmap BAR0 (%u bytes)", info.bar0_size);
    CUDA_OK(cudaHostRegister(bar0_cpu, info.bar0_size,
                             cudaHostRegisterIoMemory));
    void* bar0_gpu = nullptr;
    CUDA_OK(cudaHostGetDevicePointer(&bar0_gpu, bar0_cpu, 0));
    step_ok("BAR0 mmap=%p gpu_va=%p (cudaHostRegister + GetDevicePointer)",
            bar0_cpu, bar0_gpu);

    /* ============================================================== */
    /* Phase 2b: persistent fd-scoped data buffers.  Allocated ONCE   */
    /* here and registered with map_kind=NVM_MAP_KIND_DATA +          */
    /* group_id=0; the kernel parks them on own->data_maps so they    */
    /* survive every per-round NVM_DESTROY_QUEUE_GROUP.  Reaped at    */
    /* fd close by snvm_dev_release.  This is the canonical B6       */
    /* usage pattern: one DMA pool spanning many short-lived queue   */
    /* groups, zero re-pinning per group.                             */
    /* ============================================================== */
    persistent_data_resources pdata;
    memset(&pdata, 0, sizeof(pdata));
    CUDA_OK(cudaMalloc(&pdata.wbuf_dev,       GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&pdata.rbuf_dev,       GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&pdata.prp_list_w_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMalloc(&pdata.prp_list_r_dev, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(pdata.wbuf_dev,       0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(pdata.rbuf_dev,       0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(pdata.prp_list_w_dev, 0, GPU_PAGE_SIZE));
    CUDA_OK(cudaMemset(pdata.prp_list_r_dev, 0, GPU_PAGE_SIZE));

    {
        auto map_data = [&](void* gpu_va, uint64_t* ioaddrs_out,
                            const char* what) {
            struct nvm_ioctl_map req;
            memset(&req, 0, sizeof(req));
            req.vaddr_start = (uint64_t)(uintptr_t)gpu_va;
            req.n_pages     = 1;
            req.ioaddrs     = ioaddrs_out;
            req.ioq_idx     = -1;
            req.is_cq       = -1;
            req.group_id    = 0;                       /* fd-scoped */
            req.map_kind    = NVM_MAP_KIND_DATA;
            if (do_ioctl(fd_dev, NVM_MAP_DEVICE_MEMORY, &req, what) < 0)
                step_fail(errno, "%s gpu_va=%p", what, gpu_va);
        };
        map_data(pdata.wbuf_dev,       &pdata.wbuf_ioaddr,
                 "NVM_MAP_DEVICE_MEMORY(wbuf, DATA, fd-scoped)");
        map_data(pdata.rbuf_dev,       &pdata.rbuf_ioaddr,
                 "NVM_MAP_DEVICE_MEMORY(rbuf, DATA, fd-scoped)");
        map_data(pdata.prp_list_w_dev, &pdata.prp_list_w_ioaddr,
                 "NVM_MAP_DEVICE_MEMORY(prpl_w, DATA, fd-scoped)");
        map_data(pdata.prp_list_r_dev, &pdata.prp_list_r_ioaddr,
                 "NVM_MAP_DEVICE_MEMORY(prpl_r, DATA, fd-scoped)");
    }
    step_ok("persistent DATA maps registered (wbuf_ioaddr=0x%llx, "
            "rbuf_ioaddr=0x%llx); these survive every round destroy",
            (unsigned long long)pdata.wbuf_ioaddr,
            (unsigned long long)pdata.rbuf_ioaddr);

    /* ============================================================== */
    /* Phase 3+: rounds.  Each round builds its OWN queue group +    */
    /* fresh GPU rings only, runs all 4 tiers + wrap against the      */
    /* persistent data buffers above, then tears down the group +     */
    /* the 4 ring maps.  Data maps are untouched across rounds.       */
    /* scratch every round to exercise the snvme alloc/free path.   */
    /* ============================================================== */
    for (unsigned round = 0; round < nr_rounds; round++) {
        round_resources rr;
        memset(&rr, 0, sizeof(rr));

        fprintf(stderr, "\n===== ROUND %u / %u BEGIN =====\n",
                round + 1, nr_rounds);
        run_one_round(fd_dev, info, bar0_gpu, round, rr, pdata);
        teardown_one_round(fd_dev, round, rr);
        fprintf(stderr, "===== ROUND %u / %u END   =====\n",
                round + 1, nr_rounds);
    }

    /* ============================================================== */
    /* Final tear-down.                                               */
    /* ============================================================== */
    CUDA_OK(cudaHostUnregister(bar0_cpu));
    if (munmap(bar0_cpu, info.bar0_size) < 0)
        step_fail(errno, "munmap BAR0");
    step_ok("munmap BAR0 + cudaHostUnregister");

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_DEVICE_UNBIND, &bdf, "SNVM_DEVICE_UNBIND") < 0)
            step_fail(errno, "SNVM_DEVICE_UNBIND");
        step_ok("SNVM_DEVICE_UNBIND %s", bdf_str);
    }
    if (close(fd_dev) < 0) step_fail(errno, "close(%s)", dev_path);
    step_ok("close(%s) -- snvm_dev_release cascades through %u DATA maps",
            dev_path, 4);

    /* cudaFree the persistent DATA buffers AFTER close(fd_dev) so the
     * snvm_dev_release path has already released the nvidia_p2p_get_pages
     * pin on them.  Doing it before close would leave the pages
     * referenced by snvme at fput() time, which is the leak the
     * snvm_dev_release hook is supposed to prevent in the first place
     * (see PORTING.md §7.3.1).                                       */
    CUDA_OK(cudaFree(pdata.wbuf_dev));
    CUDA_OK(cudaFree(pdata.rbuf_dev));
    CUDA_OK(cudaFree(pdata.prp_list_w_dev));
    CUDA_OK(cudaFree(pdata.prp_list_r_dev));

    {
        struct pci_device_addr bdf = orig_bdf;
        if (do_ioctl(fd_ctl, SNVM_CHRDEV_REMOVE, &bdf, "SNVM_CHRDEV_REMOVE") < 0)
            step_fail(errno, "SNVM_CHRDEV_REMOVE");
        step_ok("SNVM_CHRDEV_REMOVE %s", bdf_str);
    }
    close(fd_ctl);

    fprintf(stderr, "\n=== snvme_smoke_gpu: all %d steps passed across "
            "%u round(s) ===\n", g_step, nr_rounds);
    return 0;
}
