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
#include <linux/fs.h>
#include <cuda_runtime.h>

#include "geminifs.h"
#include "geminifs_helper.h"
#include "nvm_error.h"
#include "gemini_fiemap.h"

// Definition of the global magic number
union geminiFS_magic the_geminiFS_magic = {
    .magic_cstr = {'g', 'e', 'm', 'i', 'n', 'i', 'f', 's'}
};


#define my_assert(code) do { \
    if (!(code)) { \
		printf("assert: %s:%d", __func__, __LINE__); \
        assert(0); \
    } \
} while(0)



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

#define FILE_BLOCK_SIZE 512 // disk block size
static inline struct fiemap *read_fiemap(int fd, u_int64_t fiemap_start, u_int64_t fiemap_length);

host_fd_t host_create_geminifs_file(const char *filename,
                          uint64_t block_size,
                          uint64_t virtual_space_size) {
	struct geminiFS_hdr *hdr;
	fd_t fd;

	my_assert(virtual_space_size % block_size == 0);

	auto hdr_size = GEMINI_HDR_MAX_SIZE;

	hdr = (struct geminiFS_hdr *)malloc(hdr_size);
	hdr->magic_num = the_geminiFS_magic.magic_num;
	hdr->first_block_base = hdr_size;
	hdr->virtual_space_size = ROUND_UP(virtual_space_size, block_size);
	hdr->block_bit = one_nr__of__binary_int(block_size - 1);

	fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
	my_assert(0 <= fd);
	my_assert(0 ==
		fallocate(fd, 0, 0, hdr->first_block_base + hdr->virtual_space_size));
	
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
	hdr->first_block_base = GEMINI_HDR_MAX_SIZE;
	hdr->virtual_space_size = ROUND_UP(virtual_space_size, block_size);
	hdr->block_bit = one_nr__of__binary_int(block_size - 1);
	
	fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
	my_assert(0 <= fd);
	my_assert(0 ==
		fallocate(fd, 0, 0, hdr->first_block_base + hdr->virtual_space_size));
	
	hdr->fd = fd;
	host_refine_nvmeofst(hdr);
	close(fd);
	
	return hdr;

}

host_fd_t host_open_geminifs_file(const char *filename) {
	fd_t fd = open(filename, O_RDWR);
	my_assert(0 <= fd);

	// First, read just the header part to get the file size information
	struct geminiFS_hdr temp_hdr;
	my_assert((off_t)(-1) != lseek(fd, 0, SEEK_SET));
	my_assert(sizeof(temp_hdr) == read(fd, &temp_hdr, sizeof(temp_hdr)));

	// Validate magic number
	my_assert(temp_hdr.magic_num == the_geminiFS_magic.magic_num);

	// Calculate the actual size needed including the l1 array
	size_t hdr_size = temp_hdr.first_block_base;
	
	// Allocate the correct amount of memory
	struct geminiFS_hdr *hdr = (struct geminiFS_hdr *)malloc(hdr_size);
	my_assert(hdr != NULL);

	// Read the complete header including l1 array
	my_assert((off_t)(-1) != lseek(fd, 0, SEEK_SET));
	my_assert((ssize_t)temp_hdr.first_block_base == read(fd, hdr, temp_hdr.first_block_base));
	if(hdr->magic_num != the_geminiFS_magic.magic_num) {
		printf("Error: File %s is not a valid geminifs file (bad magic number)\n", filename);
		return nullptr;
	}
	my_assert(hdr->magic_num == the_geminiFS_magic.magic_num);
	geminifs_debug("File %s's extents count: %d, first blk base: %lu\n", filename, hdr->extent_count, hdr->first_block_base);
	for (uint32_t i = 0; i < hdr->extent_count; ++i) {
		geminifs_debug("File %s's extents: fe_physical %llx fe_len %llx\n", filename,
		 (unsigned long long)hdr->extents[i].fe_physical, (unsigned long long)hdr->extents[i].fe_length);
	}

	hdr->fd = fd;

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
	if (ioctl(fd, FS_IOC_FIEMAP, fiemap) != 0) {
		fprintf(stderr, "fiemap ioctl() FS_IOC_FIEMAP failed\n");
		goto fail_cleanup;
	}

	/* Nothing to process */
	if (fiemap->fm_mapped_extents == 0) {
		fprintf(stderr, "extent count: %d\n", fiemap->fm_extent_count);
		goto fail_cleanup;
	}

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
										/* FIXME: if first_block_base is not aligned to page size, mmap may fail */
										hdr->first_block_base,
										PROT_WRITE | PROT_READ,
										MAP_SHARED,
										fd->fd,
										0);
	my_assert(MAP_FAILED != file_mmap);

	struct fiemap *mapping = read_fiemap(hdr->fd, hdr->first_block_base, hdr->virtual_space_size);
	my_assert(NULL != mapping);

	
    gemini_fiemap* gemini_map = convert_fiemap_to_gemini_fiemap(mapping);
    my_assert(NULL != gemini_map);
    free(mapping); // Original fiemap is no longer needed
	
	if (gemini_map->fm_mapped_extents > GEMINI_HDR_MAX_EXTENTS) {
		fprintf(stderr, "FATAL: Allocated file has too many extents: %u, max allowed: %ld\n",
				gemini_map->fm_mapped_extents, GEMINI_HDR_MAX_EXTENTS);
		exit(EXIT_FAILURE);
	}

	file_mmap->magic_num = hdr->magic_num;
	file_mmap->first_block_base = hdr->first_block_base;
	file_mmap->virtual_space_size = hdr->virtual_space_size;
	file_mmap->fd = hdr->fd;
	file_mmap->block_bit = hdr->block_bit;
	file_mmap->extent_count = gemini_map->fm_mapped_extents;
	memcpy(file_mmap->extents, gemini_map->fm_extents,
		   gemini_map->fm_mapped_extents * sizeof(struct gemini_fiemap_extent));

    free(gemini_map); // Clean up the converted map
	munmap(file_mmap, hdr->first_block_base);
}