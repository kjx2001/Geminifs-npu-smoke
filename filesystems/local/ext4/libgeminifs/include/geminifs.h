#ifndef GEMINIFS_H
#define GEMINIFS_H

// #ifdef __cplusplus
// extern "C"{
// #endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "gemini_fiemap.h"

// GeminiFS specific file open flags
#define O_HOST      0x10000000  // Open for host-side operations
#define O_DEVICE    0x20000000  // Open for device-side operations

#define GPU_PAGE_SIZE 65536ul
#define ROUND_UP(x, align)(((uint64_t) (x) + ((uint64_t)align - 1)) & ~((uint64_t)align - 1))

typedef uint64_t vaddr_t;
typedef uint64_t rawfile_ofst_t;
typedef uint64_t nvme_ofst_t;

struct geminiFS_hdr {
	uint64_t magic_num;
        rawfile_ofst_t first_block_base; /* or length of metadata */
	uint64_t virtual_space_size; /* in bytes */
	int fd; /* dummy */
        uint8_t block_bit; /* block_size, in the form of bit num */
        uint8_t extent_count;
	struct gemini_fiemap_extent extents[];
};

#define GEMINI_HDR_MAX_SIZE (512)
#define GEMINI_HDR_MAX_EXTENTS ((GEMINI_HDR_MAX_SIZE - sizeof(struct geminiFS_hdr)) / sizeof(struct gemini_fiemap_extent))

extern union geminiFS_magic {
	uint64_t magic_num;
	char magic_cstr[8];
} the_geminiFS_magic;

struct geminifs_ctrl_params {
	std::string mount_path; // mount path
        std::string snvme_control_path;
	std::vector<std::string> pci_addr; // disk pci address in the form of 0000:00:00.0
	int cudaDevice; // gpu id cuda device id
	uint32_t ns_id;
	uint64_t queueDepth;
	uint64_t numQueues;
};

struct nvme_ctrl_param {
	std::string mount_path; // mount path
	std::string pci_addr; // disk pci address in the form of 0000:00:00.0
	int cudaDevice; // gpu id cuda device id
	uint32_t ns_id;
	uint64_t queueDepth;
	uint64_t numQueues;
	uint64_t maxIOsize;
};

typedef int fd_t;
typedef struct geminiFS_hdr* host_fd_t;

//using dev_fd_t = PageCache *;
typedef void *dev_fd_t;
struct pci_device_addr;
//-----------------host only------------------
extern void
host_open_all(
        const char *snvme_control_path,
        const char *pci_addr,
        const char *mount_path,
        uint32_t ns_id,
        uint64_t queueDepth,
        uint64_t numQueues);



extern host_fd_t
host_create_geminifs_file(const char *filename,
                          uint64_t block_size,
                          uint64_t virtual_space_size);
extern host_fd_t
host_create_geminifs_file(void *buf,
                          const char *filename,
                          uint64_t block_size,
                          uint64_t virtual_space_size);


extern host_fd_t
host_open_geminifs_file(const char *filename);

extern size_t
host_xfer_geminifs_file(host_fd_t fd_1,
                        vaddr_t va,
                        void *buf_1,
                        size_t nbyte,
                        int is_read);

extern void
host_refine_nvmeofst(host_fd_t fd);

extern void
host_close_geminifs_file(host_fd_t fd);

// extern void
// host_close_all();

//-----------------host for device------------------
extern dev_fd_t
host_open_geminifs_file_for_device(
        host_fd_t host_fd,
        uint64_t pagecache_capacity,
        int page_size,
        int pagecache_batching_size);

extern dev_fd_t
host_open_geminifs_file_for_device_1(
        host_fd_t host_fd,
        double cache_ratio,
        int page_size,
        int pagecache_batching_size);

extern dev_fd_t
host_open_geminifs_file_for_device_without_backing_file(
        int page_size,
        uint64_t pagecache_capacity,
        int pagecache_batching_size);

// extern __host__ void **
// geminifs_batch_create(int nr_device, int nr_files, size_t block_size, size_t file_size);
         
// #ifdef __cplusplus
// }
// #endif

/*----------------------------------------*/
// void *host_batch_create(Controller *ctrl, int block_size, int nr_files, size_t file_size)

#endif
