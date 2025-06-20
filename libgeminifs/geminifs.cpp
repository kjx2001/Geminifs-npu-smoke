#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <cuda_runtime.h>

#include "geminifs.h"
#include "nvm_error.h"


#define my_assert(code) do { \
    if (!(code)) { \
		printf("assert: %s:%d", __func__, __LINE__); \
        assert(0); \
    } \
} while(0)


union geminiFS_magic
the_geminiFS_magic = {
  	.magic_cstr = {'g', 'e', 'm', 'i', 'n', 'i', 'f', 's'}
};

static int one_nr__of__binary_int(unsigned long long i) {
	int count = 0;
	while (i != 0) {
		if ((i & 1) == 1)
		count++;
		i = i >> 1;
	}
	return count;
}

static rawfile_ofst_t host__convert_va__to(host_fd_t host_fd, vaddr_t va) {
	struct geminiFS_hdr *hdr = host_fd;
	return hdr->first_block_base + va;
}


#define ROUND_UP(x, align)(((uint64_t) (x) + ((uint64_t)align - 1)) & ~((uint64_t)align - 1))
host_fd_t host_create_geminifs_file_1(const char *filename,
                          uint64_t block_size,
			  uint64_t page_size,
                          uint64_t virtual_space_size) {
	return host_create_geminifs_file(filename, block_size, ROUND_UP(virtual_space_size, page_size));
}

#define FILE_BLOCK_SIZE 512 // disk block size
static inline struct fiemap *read_fiemap(int fd, u_int64_t fiemap_start, u_int64_t fiemap_length);

host_fd_t host_create_geminifs_file(const char *filename,
                          uint64_t block_size,
                          uint64_t virtual_space_size) {
	struct geminiFS_hdr *hdr;
	fd_t fd;

	my_assert(virtual_space_size % block_size == 0);

	auto nr_l1 = virtual_space_size / block_size;
	auto hdr_size = ROUND_UP(sizeof(struct geminiFS_hdr) + sizeof(nvme_ofst_t) * nr_l1, block_size);

	hdr = (struct geminiFS_hdr *)malloc(hdr_size);
	hdr->magic_num = the_geminiFS_magic.magic_num;
	hdr->virtual_space_size = ROUND_UP(virtual_space_size, block_size);
	hdr->block_bit = one_nr__of__binary_int(block_size - 1);
	hdr->nr_l1 = nr_l1;
	hdr->first_block_base = hdr_size;

	fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
	my_assert(0 <= fd);
	my_assert(0 ==
		fallocate(fd, 0, 0, hdr->first_block_base + hdr->virtual_space_size));
	my_assert((off_t)(-1) != lseek(fd, 0, SEEK_SET));
	my_assert(sizeof(*hdr) == write(fd, hdr, sizeof(*hdr)));
	
	hdr->fd = fd;

	host_refine_nvmeofst(hdr);

	return hdr;
}

host_fd_t host_create_geminifs_file(void *buf, 
									const char *filename,
									uint64_t block_size,
									uint64_t virtual_space_size) {
	struct geminiFS_hdr *hdr = (struct geminiFS_hdr *)buf;
	fd_t fd;

	my_assert(virtual_space_size % block_size == 0);

	hdr->magic_num = the_geminiFS_magic.magic_num;
	hdr->virtual_space_size = ROUND_UP(virtual_space_size, block_size);
	hdr->block_bit = one_nr__of__binary_int(block_size - 1);
	hdr->nr_l1 = hdr->virtual_space_size >> hdr->block_bit;
	hdr->first_block_base = ROUND_UP(sizeof(struct geminiFS_hdr) + sizeof(nvme_ofst_t) * hdr->nr_l1, block_size);
	
	fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
	my_assert(0 <= fd);
	my_assert(0 ==
		fallocate(fd, 0, 0, hdr->first_block_base + hdr->virtual_space_size));
	my_assert((off_t)(-1) != lseek(fd, 0, SEEK_SET));
	my_assert(sizeof(*hdr) == write(fd, hdr, sizeof(*hdr)));
	
	hdr->fd = fd;
	host_refine_nvmeofst(hdr);
	close(fd);
	
	return hdr;

}

host_fd_t host_open_geminifs_file(const char *filename) {
	struct geminiFS_hdr *hdr = (struct geminiFS_hdr *)malloc(sizeof(struct geminiFS_hdr));
	fd_t fd = open(filename, O_RDWR);
	my_assert(0 <= fd);

	my_assert((off_t)(-1) != lseek(fd, 0, SEEK_SET));
	my_assert(sizeof(*hdr) == read(fd, hdr, sizeof(*hdr)));

	hdr->fd = fd;

	my_assert(hdr->magic_num == the_geminiFS_magic.magic_num);

	return hdr;
}

size_t host_xfer_geminifs_file(host_fd_t host_fd,
                        vaddr_t va,
                        void *buf_1,
                        size_t nbyte,
                        int is_read) {
	struct geminiFS_hdr *hdr = host_fd;
	fd_t fd = hdr->fd;
	my_assert((off_t)(-1) != lseek(fd, host__convert_va__to(host_fd, va), SEEK_SET));

	size_t nbyte_already = 0;
	char *buf = (char *)buf_1;
	while (0 < nbyte) {
		ssize_t nbyte_this_time;
		if (is_read)
		nbyte_this_time = read(fd, buf, nbyte);
		else
		nbyte_this_time = write(fd, buf, nbyte);
		my_assert(nbyte_this_time != -1);

		nbyte -= nbyte_this_time;
		buf += nbyte_this_time;
		nbyte_already += nbyte_this_time;
	}
	if (!is_read)
		fsync(fd);
	return nbyte_already;
}

void host_close_geminifs_file(host_fd_t fd) {
	close(fd->fd);
	free(fd);
}

static inline struct fiemap *read_fiemap(int fd, u_int64_t fiemap_start, u_int64_t fiemap_length){
	struct fiemap *fiemap = NULL;
	struct fiemap *result_fiemap = NULL;
	struct fiemap *fm_tmp; /* need to store pointer on realloc */
	int extents_size;
	u_int32_t result_extents = 0;

	fiemap = (struct fiemap *)malloc(sizeof(struct fiemap));
	if (fiemap == NULL) {
		fprintf(stderr, "Out of memory allocating fiemap\n");
		return NULL;
	}

	result_fiemap = (struct fiemap *)malloc(sizeof(struct fiemap));
	if (result_fiemap == NULL) {
		fprintf(stderr, "Out of memory allocating fiemap\n");
		goto fail_cleanup;
	}
	
	memset(fiemap, 0, sizeof(struct fiemap));

	fiemap->fm_start = fiemap_start;
	fiemap->fm_length = fiemap_length;
    // fiemap->fm_flags = FIEMAP_FLAG_SYNC;

	/* Find out how many extents there are */
	if (ioctl(fd, FS_IOC_FIEMAP, fiemap) < 0) {
		fprintf(stderr, "fiemap ioctl() FS_IOC_FIEMAP failed");
		goto fail_cleanup;
	}

	/* Nothing to process */
	if (fiemap->fm_mapped_extents == 0)
		goto fail_cleanup;

	/* Result fiemap have to hold all the extents for the hole file */

	/* Read in the extents */
	extents_size = sizeof(struct fiemap_extent) *
							(fiemap->fm_mapped_extents);

	/* Resize fiemap to allow us to read in the extents */
	fm_tmp = (struct fiemap *)realloc(fiemap,
				sizeof(struct fiemap) + extents_size);
	if (!fm_tmp) {
		fprintf(stderr, "Out of memory reallocating fiemap\n");
		goto fail_cleanup;
	}
	fiemap = fm_tmp;

	memset(fiemap->fm_extents, 0, extents_size);
	fiemap->fm_extent_count = fiemap->fm_mapped_extents;
	fiemap->fm_mapped_extents = 0;

	if (ioctl(fd, FS_IOC_FIEMAP, fiemap) < 0) {
		fprintf(stderr, "fiemap ioctl() FS_IOC_FIEMAP failed\n");
		goto fail_cleanup;
	}

	extents_size = sizeof(struct fiemap_extent) *
							(result_extents +
					fiemap->fm_mapped_extents);

	/* Resize result_fiemap to allow us to read in the extents */
	fm_tmp = (struct fiemap *)realloc(result_fiemap,
				sizeof(struct fiemap) + extents_size);
	if (!fm_tmp) {
		fprintf(stderr, "Out of memory allocating fiemap\n");
		goto fail_cleanup;
	}
	result_fiemap = fm_tmp;

	memcpy(result_fiemap->fm_extents + result_extents,
			fiemap->fm_extents,
			sizeof(struct fiemap_extent) *
			fiemap->fm_mapped_extents);

	result_extents += fiemap->fm_mapped_extents;

	/* Highly unlikely that it is zero */
	if (fiemap->fm_mapped_extents) {
		const u_int32_t i = fiemap->fm_mapped_extents - 1;

		fiemap_start = fiemap->fm_extents[i].fe_logical +
					fiemap->fm_extents[i].fe_length;
	}

	result_fiemap->fm_mapped_extents = result_extents;
	free(fiemap);

	return result_fiemap;

fail_cleanup:
	if (result_fiemap)
		free(result_fiemap);

	if (fiemap)
		free(fiemap);

	return NULL;
}
// address embed 
void host_refine_nvmeofst(host_fd_t fd) {
	struct geminiFS_hdr *hdr = fd;
	struct geminiFS_hdr *file_mmap = (struct geminiFS_hdr *)
									mmap(NULL,
										hdr->first_block_base,
										PROT_WRITE | PROT_READ,
										MAP_SHARED,
										fd->fd,
										0);
	my_assert((void *) -1 != file_mmap);
	size_t idx = 0;
	struct fiemap *mapping = read_fiemap(hdr->fd, hdr->first_block_base, hdr->virtual_space_size);
	my_assert(NULL != mapping);
	for (size_t i = 0; i < mapping->fm_mapped_extents; ++i) {
		for (size_t j = 0; j < mapping->fm_extents[i].fe_length >> hdr->block_bit; ++j, ++idx) {
			file_mmap->l1[idx] = mapping->fm_extents[i].fe_physical + (j << hdr->block_bit);
			hdr->l1[idx] = file_mmap->l1[idx];
		}
	}

	munmap(file_mmap, hdr->first_block_base);
}