/*
 * $QNXtpLicenseC: $
*/

/*

   Copyright 2003 Free Software Foundation, Inc.

   Contributed by QNX Software Systems Ltd.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* NVIDIA CUDA Debugger CUDA-GDB
   Copyright (C) 2007-2024 NVIDIA Corporation
   Modified from the original GDB file referenced above by the CUDA-GDB
   team at NVIDIA <cudatools@nvidia.com>. */

/* __DEBUG_H_INCLUDED is Neutrino's native debug.h header.  We don't want
   these duplicate definitions if we're compiling natively and have already
   included it.  */
#ifndef __DEBUG_H_INCLUDED
#define __DEBUG_H_INCLUDED

#ifdef HAVE_TIME_H
#include <time.h>
#endif

#ifdef __MINGW32__
#include <time.h>
#endif // __MINGW32__

#include <stdint.h>

typedef int32_t nto_pid_t;
typedef int32_t nto_uid_t;
typedef int32_t nto_gid_t;

enum Elf_nto_note_types
{
  QNT_NULL = 0,
  QNT_DEBUG_FULLPATH,
  QNT_DEBUG_RELOC,
  QNT_STACK,
  QNT_GENERATOR,
  QNT_DEFAULT_LIB,
  QNT_CORE_SYSINFO,
  QNT_CORE_INFO,
  QNT_CORE_STATUS,
  QNT_CORE_GREG,
  QNT_CORE_FPREG,
  QNT_NUM
};

typedef struct
{
  uint32_t bits[2];
} nto_sigset_t;

union __sigval32 {
  int32_t   sival_int;
  uint32_t  sival_ptr;
};

union __sigval64 {
  int32_t   sival_int;
  uint64_t  sival_ptr;
};

union nto_sigval {
  union __sigval32 _32;
  union __sigval64 _64;
};

typedef struct _siginfo32 {
    int             si_signo;
    int             si_code;        /* if SI_NOINFO, only si_signo is valid */
    int             si_errno;
    union {
        int             __pad[7];
        struct {
            nto_pid_t       __pid;
            union {
                struct {
                    nto_uid_t       __uid;
                    union __sigval32 __value;
                }               __kill;     /* si_code <= 0 SI_FROMUSER */
                struct {
                    uint32_t   __utime;
                    int             __status;   /* CLD_EXITED status, else      signo */
                    uint32_t   __stime;
                }               __chld;     /* si_signo=SIGCHLD si_code=CLD_* */
            }               __pdata;
        }               __proc;
        struct {
            int             __fltno;
            uint32_t     __fltip;
            uint32_t     __addr;
            int             __bdslot;
        }               __fault;                /* si_signo=SIGSEGV,ILL,FPE,    TRAP,BUS */
    }               __data;
}               __siginfo32_t;

typedef struct _siginfo64 {
    int             si_signo;
    int             si_code;        /* if SI_NOINFO, only si_signo is valid */
    int             si_errno;
    union {
        int             __pad[7];
        struct {
            nto_pid_t       __pid;
            union {
                struct {
                    nto_uid_t       __uid;
                    union __sigval64 __value;
                }               __kill;     /* si_code <= 0 SI_FROMUSER */
                struct {
                    uint32_t   __utime;
                    int             __status;   /* CLD_EXITED status, else      signo */
                    uint32_t   __stime;
                }               __chld;     /* si_signo=SIGCHLD si_code=CLD_* */
            }               __pdata;
        }               __proc;
        struct {
            int             __fltno;
            uint64_t     __fltip;
            uint64_t     __addr;
            int             __bdslot;
        }               __fault;                /* si_signo=SIGSEGV,ILL,FPE,    TRAP,BUS */
    }               __data;
}               __siginfo64_t;

typedef union nto_siginfo
{
  __siginfo32_t _32;
  __siginfo64_t _64;
} nto_siginfo_t;

#define nto_si_pid	__data.__proc.__pid
#define nto_si_value	__data.__proc.__pdata.__kill.__value
#define nto_si_uid	__data.__proc.__pdata.__kill.__uid
#define nto_si_status	__data.__proc.__pdata.__chld.__status
#define nto_si_utime	__data.__proc.__pdata.__chld.__utime
#define nto_si_stime	__data.__proc.__pdata.__chld.__stime
#define nto_si_fltno	__data.__fault.__fltno
#define nto_si_trapno	nto_si_fltno
#define nto_si_addr	__data.__fault.__addr
#define nto_si_fltip	__data.__fault.__fltip
#define nto_si_bdslot	__data.__fault.__bdslot

#ifdef __QNX__
__BEGIN_DECLS
#include <_pack64.h>
#endif
#define _DEBUG_FLAG_STOPPED			0x00000001	/* Thread is not running.  */
#define DEBUG_FLAG_ISTOP			0x00000002	/* Stopped at point of interest.  */
#define _DEBUG_FLAG_IPINVAL			0x00000010	/* IP is not valid.  */
#define _DEBUG_FLAG_ISSYS			0x00000020	/* System process.  */
#define _DEBUG_FLAG_SSTEP			0x00000040	/* Stopped because of single step.  */
#define _DEBUG_FLAG_CURTID			0x00000080	/* Thread is current thread.  */
#define DEBUG_FLAG_TRACE_EXEC		0x00000100	/* Stopped because of breakpoint.  */
#define _DEBUG_FLAG_TRACE_RD		0x00000200	/* Stopped because of read access.  */
#define _DEBUG_FLAG_TRACE_WR		0x00000400	/* Stopped because of write access.  */
#define _DEBUG_FLAG_TRACE_MODIFY	0x00000800	/* Stopped because of modified memory.  */
#define _DEBUG_FLAG_RLC				0x00010000	/* Run-on-Last-Close flag is set.  */
#define _DEBUG_FLAG_KLC				0x00020000	/* Kill-on-Last-Close flag is set.  */
#define _DEBUG_FLAG_FORK			0x00040000	/* Child inherits flags (Stop on fork/spawn).  */
#define _DEBUG_FLAG_MASK			0x000f0000	/* Flags that can be changed.  */
  enum
{
  _DEBUG_WHY_REQUESTED,
  _DEBUG_WHY_SIGNALLED,
  _DEBUG_WHY_FAULTED,
  _DEBUG_WHY_JOBCONTROL,
  _DEBUG_WHY_TERMINATED,
  _DEBUG_WHY_CHILD,
  _DEBUG_WHY_EXEC,
  _DEBUG_WHY_THREAD
};

/* WHAT constants are specific to WHY reason they are associated with.
   Actual numeric value may be reused between different WHY reasons.  */
#define _DEBUG_WHAT_DESTROYED		0x00000000U	/* WHY_THREAD */
#define _DEBUG_WHAT_CREATED			0x00000001U	/* WHY_THREAD */
#define _DEBUG_WHAT_FORK			0x00000000U	/* WHY_CHILD */
#define _DEBUG_WHAT_VFORK			0x00000001U	/* WHY_CHILD */

#define _DEBUG_RUN_CLRSIG			0x00000001	/* Clear pending signal */
#define _DEBUG_RUN_CLRFLT			0x00000002	/* Clear pending fault */
#define DEBUG_RUN_TRACE			0x00000004	/* Trace mask flags interesting signals */
#define DEBUG_RUN_HOLD				0x00000008	/* Hold mask flags interesting signals */
#define DEBUG_RUN_FAULT			0x00000010	/* Fault mask flags interesting faults */
#define _DEBUG_RUN_VADDR			0x00000020	/* Change ip before running */
#define _DEBUG_RUN_STEP				0x00000040	/* Single step only one thread */
#define _DEBUG_RUN_STEP_ALL			0x00000080	/* Single step one thread, other threads run */
#define _DEBUG_RUN_CURTID			0x00000100	/* Change current thread (target thread) */
#define DEBUG_RUN_ARM				0x00000200	/* Deliver event at point of interest */

typedef struct _debug_process_info
{
  nto_pid_t pid;
  nto_pid_t parent;
  unsigned flags;
  unsigned umask;
  nto_pid_t child;
  nto_pid_t sibling;
  nto_pid_t pgrp;
  nto_pid_t sid;
  uint64_t base_address;
  uint64_t initial_stack;
  nto_uid_t uid;
  nto_gid_t gid;
  nto_uid_t euid;
  nto_gid_t egid;
  nto_uid_t suid;
  nto_gid_t sgid;
  nto_sigset_t sig_ignore;
  nto_sigset_t sig_queue;
  nto_sigset_t sig_pending;
  unsigned num_chancons;
  unsigned num_fdcons;
  unsigned num_threads;
  unsigned num_timers;
  uint64_t reserved[5];	  /* Process times.  */
  unsigned char priority; /* Process base priority.  */
  unsigned char reserved2[7];
  unsigned char extsched[8];
  uint64_t pls;	  /* Address of process local storage.  */
  uint64_t sigstub; /* Address of process signal trampoline.  */
  uint64_t canstub; /* Address of process thread cancellation trampoline. */
  uint64_t reserved3[10];
} nto_procfs_info;

typedef struct _debug_thread_info32
{
  nto_pid_t pid;
  unsigned tid;
  unsigned flags;
  unsigned short why;
  unsigned short what;
  uint64_t ip;
  uint64_t sp;
  uint64_t stkbase;
  uint64_t tls;
  unsigned stksize;
  unsigned tid_flags;
  unsigned char priority;
  unsigned char real_priority;
  unsigned char policy;
  unsigned char state;
  short syscall;
  unsigned short last_cpu;
  unsigned timeout;
  int last_chid;
  nto_sigset_t sig_blocked;
  nto_sigset_t sig_pending;
  __siginfo32_t info;
  unsigned reserved1;
  union
  {
    struct
    {
      unsigned tid;
    } join;
    struct
    {
      int id;
      int sync;
    } sync;
    struct
    {
      unsigned nid;
      nto_pid_t pid;
      int coid;
      int chid;
      int scoid;
    } connect;
    struct
    {
      int chid;
    } channel;
    struct
    {
      nto_pid_t pid;
      unsigned flags;
      int64_t vaddr;
    } waitpage;
    struct
    {
      unsigned size;
    } stack;
    struct
    {
      unsigned  tid;
    }          thread_event;
    struct
    {
      nto_pid_t  child;
    }          fork_event;
    uint64_t filler[4];
  } blocked;
  uint64_t                    start_time;     /* thread start time in nsec */
  uint64_t                    sutime;         /* thread system + user running time in nsec */
  uint8_t                     extsched[8];
  uint64_t                    nsec_since_block;   /*how long thread has been  blocked. 0 for STATE_READY or STATE_RUNNING. 
					in nsec, but ms resolution. */

  uint64_t reserved2[8];
} _debug_thread_info32;

typedef struct _debug_thread_info64
{
  nto_pid_t pid;
  unsigned tid;
  unsigned flags;
  unsigned short why;
  unsigned short what;
  uint64_t ip;
  uint64_t sp;
  uint64_t stkbase;
  uint64_t tls;
  unsigned stksize;
  unsigned tid_flags;
  unsigned char priority;
  unsigned char real_priority;
  unsigned char policy;
  unsigned char state;
  short syscall;
  unsigned short last_cpu;
  unsigned timeout;
  int last_chid;
  nto_sigset_t sig_blocked;
  nto_sigset_t sig_pending;
  __siginfo32_t __info32;
  union
  {
    struct
    {
      unsigned tid;
    } join;
    struct
    {
      int64_t id;
      uint64_t sync;
    } sync;
    struct
    {
      unsigned nid;
      nto_pid_t pid;
      int coid;
      int chid;
      int scoid;
    } connect;
    struct
    {
      int chid;
    } channel;
    struct
    {
      nto_pid_t pid;
      unsigned flags;
      uint64_t vaddr;
    } waitpage;
    struct
    {
      uint64_t size;
    } stack;
    struct {
	unsigned tid;
    }	  thread_event;
    struct {
	nto_pid_t child;
    }	  fork_event;
    uint64_t filler[4];
  } blocked;

  uint64_t                    start_time;     /* thread start time in nsec */
  uint64_t                    sutime;         /* thread system + user running time in nsec */
  uint8_t                     extsched[8];
  uint64_t                    nsec_since_block;   /*how long thread has been  blocked. 0 for STATE_READY or STATE_RUNNING. 
					in nsec, but ms resolution. */
  union {
      __siginfo32_t           info32;
      __siginfo64_t           info64;
  };


  uint64_t reserved2[4];
} _debug_thread_info64;

typedef union nto_procfs_status
{
    _debug_thread_info32 _32;
    _debug_thread_info64 _64;
} nto_procfs_status;

/* From debug.h to interpret ldd events.
   _r_debug global variable of this type exists in libc. We read it to 
   figure out what the event is about.  */
#define	R_DEBUG_VERSION	2

/* The following is to satisfy things we do not currently use.  */

typedef enum {
  RT_CONSISTENT,	/* link_maps are consistent */
  RT_ADD,		/* Adding to link_map */
  RT_DELETE		/* Removeing a link_map */
} r_state_e;

typedef enum {
  RD_FL_NONE =	0,
  RD_FL_DBG =	(1<<1)	/* process may be being debugged */
} rd_flags_e;

typedef enum {
  RD_NONE = 0,
  RD_PREINIT,	/* Before .init() */
  RD_POSTINIT,	/* After .init() */
  RD_DLACTIVITY	/* dlopen() or dlclose() occured */
} rd_event_e;

#ifdef __QNX__
#include <_packpop.h>

__END_DECLS
#endif
#endif /* __DEBUG_H_INCLUDED */
