/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * FS_IOC_FIEMAP ioctl infrastructure.
 *
 * Some portions copyright (C) 2007 Cluster File Systems, Inc
 *
 * Authors: Mark Fasheh <mfasheh@suse.com>
 *          Kalpak Shah <kalpak.shah@sun.com>
 *          Andreas Dilger <adilger@sun.com>
 */

#ifndef _LINUX_FIEMAP_H
#define _LINUX_FIEMAP_H

#include <linux/types.h>
#include <stdlib.h> // Required for malloc
#include <stdint.h> // Required for uint32_t
 
struct gemini_fiemap_extent {
    __u64 fe_logical;  /* logical offset in bytes for the start of
                * the extent from the beginning of the file */
    __u64 fe_physical; /* physical offset in bytes for the start
                * of the extent from the beginning of the disk */
    __u64 fe_length;   /* length in bytes for this extent */
    __u32 fe_flags;    /* FIEMAP_EXTENT_* flags for this extent */
};

struct gemini_fiemap {
    __u64 fm_start;		/* logical offset (inclusive) at
                * which to start mapping (in) */
    __u64 fm_length;	/* logical length of mapping which
                * userspace wants (in) */
    __u32 fm_flags;		/* FIEMAP_FLAG_* flags for request (in/out) */
    __u32 fm_mapped_extents;/* number of extents that were mapped (out) */
    __u32 fm_extent_count;  /* size of fm_extents array (in) */
    __u32 fm_reserved;
    struct gemini_fiemap_extent fm_extents[0]; /* array of mapped extents (out) */
}; 

struct fiemap_extent {
	__u64 fe_logical;  /* logical offset in bytes for the start of
			    * the extent from the beginning of the file */
	__u64 fe_physical; /* physical offset in bytes for the start
			    * of the extent from the beginning of the disk */
	__u64 fe_length;   /* length in bytes for this extent */
	__u64 fe_reserved64[2];
	__u32 fe_flags;    /* FIEMAP_EXTENT_* flags for this extent */
	__u32 fe_reserved[3];
};

struct fiemap {
	__u64 fm_start;		/* logical offset (inclusive) at
				 * which to start mapping (in) */
	__u64 fm_length;	/* logical length of mapping which
				 * userspace wants (in) */
	__u32 fm_flags;		/* FIEMAP_FLAG_* flags for request (in/out) */
	__u32 fm_mapped_extents;/* number of extents that were mapped (out) */
	__u32 fm_extent_count;  /* size of fm_extents array (in) */
	__u32 fm_reserved;
	struct fiemap_extent fm_extents[0]; /* array of mapped extents (out) */
};

static inline gemini_fiemap_extent convert_fiemap_extent_to_gemini_fiemap_extent(fiemap_extent *fiemap_extent) {
    gemini_fiemap_extent gemini_extent;
    gemini_extent.fe_logical = fiemap_extent->fe_logical;
    gemini_extent.fe_physical = fiemap_extent->fe_physical;
    gemini_extent.fe_length = fiemap_extent->fe_length;
    gemini_extent.fe_flags = fiemap_extent->fe_flags;
    return gemini_extent;
}

static inline gemini_fiemap* convert_fiemap_to_gemini_fiemap(fiemap *fiemap) {
    if (fiemap == NULL) {
        return NULL;
    }

    size_t new_size = sizeof(gemini_fiemap) + fiemap->fm_mapped_extents * sizeof(gemini_fiemap_extent);
    gemini_fiemap* gemini_map = (gemini_fiemap*)malloc(new_size);
    if (gemini_map == NULL) {
        return NULL;
    }

    gemini_map->fm_start = fiemap->fm_start;
    gemini_map->fm_length = fiemap->fm_length;
    gemini_map->fm_flags = fiemap->fm_flags;
    gemini_map->fm_mapped_extents = fiemap->fm_mapped_extents;
    gemini_map->fm_extent_count = fiemap->fm_mapped_extents; // Gemini fiemap stores the actual count

    for (uint32_t i = 0; i < fiemap->fm_mapped_extents; ++i) {
        gemini_map->fm_extents[i] = convert_fiemap_extent_to_gemini_fiemap_extent(&fiemap->fm_extents[i]);
    }

    return gemini_map;
}
 
#define FIEMAP_MAX_OFFSET	(~0ULL)

#define FIEMAP_FLAG_SYNC	0x00000001 /* sync file data before map */
#define FIEMAP_FLAG_XATTR	0x00000002 /* map extended attribute tree */
#define FIEMAP_FLAG_CACHE	0x00000004 /* request caching of the extents */

#define FIEMAP_FLAGS_COMPAT	(FIEMAP_FLAG_SYNC | FIEMAP_FLAG_XATTR)

#define FIEMAP_EXTENT_LAST		0x00000001 /* Last extent in file. */
#define FIEMAP_EXTENT_UNKNOWN		0x00000002 /* Data location unknown. */
#define FIEMAP_EXTENT_DELALLOC		0x00000004 /* Location still pending.
                            * Sets EXTENT_UNKNOWN. */
#define FIEMAP_EXTENT_ENCODED		0x00000008 /* Data can not be read
                            * while fs is unmounted */
#define FIEMAP_EXTENT_DATA_ENCRYPTED	0x00000080 /* Data is encrypted by fs.
                            * Sets EXTENT_NO_BYPASS. */
#define FIEMAP_EXTENT_NOT_ALIGNED	0x00000100 /* Extent offsets may not be
                            * block aligned. */
#define FIEMAP_EXTENT_DATA_INLINE	0x00000200 /* Data mixed with metadata.
                            * Sets EXTENT_NOT_ALIGNED.*/
#define FIEMAP_EXTENT_DATA_TAIL		0x00000400 /* Multiple files in block.
                            * Sets EXTENT_NOT_ALIGNED.*/
#define FIEMAP_EXTENT_UNWRITTEN		0x00000800 /* Space allocated, but
                            * no data (i.e. zero). */
#define FIEMAP_EXTENT_MERGED		0x00001000 /* File does not natively
                            * support extents. Result
                            * merged for efficiency. */
#define FIEMAP_EXTENT_SHARED		0x00002000 /* Space shared with other
                            * files. */

#endif /* _LINUX_FIEMAP_H */
 