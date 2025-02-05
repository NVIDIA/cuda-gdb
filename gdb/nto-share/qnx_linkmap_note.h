/*
 * $QNXLicenseC:
 * Copyright 2013, QNX Software Systems. All Rights Reserved.
 *
 * You must obtain a written license from and pay applicable license fees to QNX
 * Software Systems before you may reproduce, modify or distribute this software,
 * or any work that includes all or part of this software.   Free development
 * licenses are available for evaluation and non-commercial purposes.  For more
 * information visit http://licensing.qnx.com or email licensing@qnx.com.
 *
 * This file may contain contributions from others.  Please review this entire
 * file for other proprietary rights or license notices, as well as the QNX
 * Development Suite License Guide at http://licensing.qnx.com/license-guide/
 * for other information.
 * $
 */


#ifndef __QNX_LINKMAP_NOTE_H_INCLUDED__
#define __QNX_LINKMAP_NOTE_H_INCLUDED__

/* This file describes QNT_LINK_MAP note format.
 *
 * It gets included in the tools that interpret
 * the note and therefore must maintain binary
 * compatibility across all supported hosts.
 *
 * The file contains structures with identical
 * layout as structures defined in sys/link.h,
 * but all pointers and types are rewritten as
 * appropriate explicitly sized integers. */

#include <stdint.h>

struct qnx_linkmap_note_header
{
	uint32_t version;
	uint32_t linkmapsz;
	uint32_t strtabsz;
	uint32_t buildidtabsz;
	uint32_t reserved[4];
};

/* Link_map structure. See sys/link.h for
 * details. */
struct qnx_link_map_32
{
    uint32_t l_addr;      /* base address           */
    uint32_t l_name;      /* full soname of lib     */
    uint32_t l_ld;        /* _DYNAMIC in lib        */
    uint32_t l_next;
    uint32_t l_prev;
    uint32_t l_refname;   /* matching soname of lib */
    uint32_t l_loaded;    /* time lib was loaded    */
    uint32_t l_path;      /* full pathname of lib   */
};

struct qnx_link_map_64
{
    uint64_t l_addr;      /* base address           */
    uint64_t l_name;      /* full soname of lib     */
    uint64_t l_ld;        /* _DYNAMIC in lib        */
    uint64_t l_next;
    uint64_t l_prev;
    uint64_t l_refname;   /* matching soname of lib */
    uint64_t l_loaded;    /* time lib was loaded    */
    uint64_t l_path;      /* full pathname of lib   */
};

struct qnx_r_debug_32
{
    int32_t  r_version;   /* R_DEBUG_VERSION         */
    uint32_t r_map;       /* Global link_map         */
    uint32_t r_brk;       /* void (*r_brk)(void)     */
    uint32_t r_state;     /* RT_*                    */
    uint32_t r_ldbase;    /* ldqnx.so.1 base address */
    uint32_t r_ldsomap;   /* ldqnx.so.1 link map     */
    uint32_t r_rdevent;   /* RD_*                    */
    uint32_t r_flags;     /* RD_FL_*                 */
};

struct qnx_r_debug_64
{
    int32_t  r_version;   /* R_DEBUG_VERSION         */
    int32_t  r_version_padding;
    uint64_t r_map;       /* Global link_map         */
    uint64_t r_brk;       /* void (*r_brk)(void)     */
    uint32_t r_state;     /* RT_*                    */
    uint32_t r_state_padding;
    uint64_t r_ldbase;    /* ldqnx.so.1 base address */
    uint64_t r_ldsomap;   /* ldqnx.so.1 link map     */
    uint32_t r_rdevent;   /* RD_*                    */
    uint32_t r_flags;     /* RD_FL_*                 */
};

struct qnx_linkmap_note_linkmap_32
{
    struct qnx_r_debug_32  r_debug;
    struct qnx_link_map_32 r_map[0];
};

struct qnx_linkmap_note_linkmap_64
{
    struct qnx_r_debug_64  r_debug;
    struct qnx_link_map_64 r_map[0];
};

#if __SIZEOF_POINTER__ == 4
    /* Native structure. */
#   define qnx_r_debug              qnx_r_debug_32
#   define qnx_linkmap_note_linkmap qnx_linkmap_note_linkmap_32
#   define qnx_link_map             qnx_link_map_32
#elif __SIZEOF_POINTER__ == 8
    /* Native structure. */
#   define qnx_r_debug              qnx_r_debug_64
#   define qnx_linkmap_note_linkmap qnx_linkmap_note_linkmap_64
#   define qnx_link_map             qnx_link_map_64
#else
#error Expected value for __SIZEOF_POINTER__
#endif

enum qnx_linkmap_note_buildid_type
{
	QNX_BUILDID_TYPE_GNU = 0,   /* .note.gnu.build-id */
};

struct qnx_linkmap_note_buildid
{
	uint16_t desctype;   /* One of qnx_linkmap_note_buildid_type */
	uint16_t descsz;
	uint8_t  desc[0];
};

#define QNX_LINKMAP_NOTE_BUILDID_MAXSZ 0xFFFFU

struct qnx_linkmap_note
{
	struct qnx_linkmap_note_header header;
	uint32_t data[0];
};

struct qnx_linkmap_note_buffer
{
	uint32_t size;                   /* Overall size of the buffer */
	struct qnx_linkmap_note *desc;   /* Allocated and built QNT_LINK_MAP
										note desc */
};


#ifdef __QNXNTO__
#include <sys/srcversion.h>
__SRCVERSION( "$URL$ $Rev$" )
#endif

#endif

