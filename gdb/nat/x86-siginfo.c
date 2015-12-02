/* Low-level siginfo manipulation (i386 and x86-64).

   Copyright (C) 2002-2015 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <signal.h>
#include "common-defs.h"
#include "x86-siginfo.h"

/* When GDB is built as a 64-bit application on linux, the
   PTRACE_GETSIGINFO data is always presented in 64-bit layout.  Since
   debugging a 32-bit inferior with a 64-bit GDB should look the same
   as debugging it with a 32-bit GDB, we do the 32-bit <-> 64-bit
   conversion in-place ourselves.  With the presence of possible different
   fields for host and target we have to guarantee that we use the
   Also, the first step is to make a copy from the original bits to the
   internal structure which is the super set. */

/* These types below (native_*) define a siginfo type that is layout
   the most complete siginfo available for the architecture.  */



/* These types below (compat_*) define a siginfo type that is layout
   compatible with the siginfo type exported by the 32-bit userspace
   support.  */

typedef int compat_int_t;
typedef unsigned int compat_uptr_t;

typedef int compat_time_t;
typedef int compat_timer_t;
typedef int compat_clock_t;

struct compat_timeval
{
  compat_time_t tv_sec;
  int tv_usec;
};

typedef union compat_sigval
{
  compat_int_t sival_int;
  compat_uptr_t sival_ptr;
} compat_sigval_t;

typedef struct compat_siginfo
{
  int si_signo;
  int si_errno;
  int si_code;

  union
  {
    int _pad[((128 / sizeof (int)) - 3)];

    /* kill() */
    struct
    {
      unsigned int _pid;
      unsigned int _uid;
    } _kill;

    /* POSIX.1b timers */
    struct
    {
      compat_timer_t _tid;
      int _overrun;
      compat_sigval_t _sigval;
    } _timer;

    /* POSIX.1b signals */
    struct
    {
      unsigned int _pid;
      unsigned int _uid;
      compat_sigval_t _sigval;
    } _rt;

    /* SIGCHLD */
    struct
    {
      unsigned int _pid;
      unsigned int _uid;
      int _status;
      compat_clock_t _utime;
      compat_clock_t _stime;
    } _sigchld;

    /* SIGILL, SIGFPE, SIGSEGV, SIGBUS */
    struct
    {
      unsigned int _addr;
      short int _addr_lsb;
      struct
      {
	unsigned int _lower;
	unsigned int _upper;
      } si_addr_bnd;
    } _sigfault;

    /* SIGPOLL */
    struct
    {
      int _band;
      int _fd;
    } _sigpoll;
  } _sifields;
} compat_siginfo_t;

typedef long __attribute__ ((__aligned__ (4))) compat_x32_clock_t;

typedef struct compat_x32_siginfo
{
  int si_signo;
  int si_errno;
  int si_code;

  union
  {
    int _pad[((128 / sizeof (int)) - 3)];

    /* kill() */
    struct
    {
      unsigned int _pid;
      unsigned int _uid;
    } _kill;

    /* POSIX.1b timers */
    struct
    {
      compat_timer_t _tid;
      int _overrun;
      compat_sigval_t _sigval;
    } _timer;

    /* POSIX.1b signals */
    struct
    {
      unsigned int _pid;
      unsigned int _uid;
      compat_sigval_t _sigval;
    } _rt;

    /* SIGCHLD */
    struct
    {
      unsigned int _pid;
      unsigned int _uid;
      int _status;
      compat_x32_clock_t _utime;
      compat_x32_clock_t _stime;
    } _sigchld;

    /* MPX fields are not needed here; X32 not supported.  */
    /* SIGILL, SIGFPE, SIGSEGV, SIGBUS.  */
    struct
    {
      unsigned int _addr;
      unsigned int _addr_lsb;
    } _sigfault;

    /* SIGPOLL */
    struct
    {
      int _band;
      int _fd;
    } _sigpoll;
  } _sifields;
} compat_x32_siginfo_t __attribute__ ((__aligned__ (8)));

#define cpt_si_pid _sifields._kill._pid
#define cpt_si_uid _sifields._kill._uid
#define cpt_si_timerid _sifields._timer._tid
#define cpt_si_overrun _sifields._timer._overrun
#define cpt_si_status _sifields._sigchld._status
#define cpt_si_utime _sifields._sigchld._utime
#define cpt_si_stime _sifields._sigchld._stime
#define cpt_si_ptr _sifields._rt._sigval.sival_ptr
#define cpt_si_addr _sifields._sigfault._addr
#define cpt_si_addr_lsb _sifields._sigfault._addr_lsb
#define cpt_si_band _sifields._sigpoll._band
#define cpt_si_fd _sifields._sigpoll._fd

/* glibc at least up to 2.3.2 doesn't have si_timerid, si_overrun.
   In their place is si_timer1,si_timer2.  */
#ifndef si_timerid
#define si_timerid si_timer1
#endif
#ifndef si_overrun
#define si_overrun si_timer2
#endif

static void
compat_siginfo_from_siginfo (compat_siginfo_t *to, siginfo_t *from)
{

  memset (to, 0, sizeof (*to));

  to->si_signo = from->si_signo;
  to->si_errno = from->si_errno;
  to->si_code = from->si_code;

  if (to->si_code == SI_TIMER)
    {
      to->cpt_si_timerid = from->si_timerid;
      to->cpt_si_overrun = from->si_overrun;
      to->cpt_si_ptr = (intptr_t) from->si_ptr;
    }
  else if (to->si_code == SI_USER)
    {
      to->cpt_si_pid = from->si_pid;
      to->cpt_si_uid = from->si_uid;
    }
  else if (to->si_code < 0)
    {
      to->cpt_si_pid = from->si_pid;
      to->cpt_si_uid = from->si_uid;
      to->cpt_si_ptr = (intptr_t) from->si_ptr;
    }
  else
    {
      switch (to->si_signo)
	{
	case SIGCHLD:
	  to->cpt_si_pid = from->si_pid;
	  to->cpt_si_uid = from->si_uid;
	  to->cpt_si_status = from->si_status;
	  to->cpt_si_utime = from->si_utime;
	  to->cpt_si_stime = from->si_stime;
	  break;
	case SIGILL:
	case SIGFPE:
	case SIGSEGV:
	case SIGBUS:
	  to->cpt_si_addr = (intptr_t) from->si_addr;
	  to->cpt_si_addr_lsb = (short) from->si_addr_lsb;
	  break;
	case SIGPOLL:
	  to->cpt_si_band = from->si_band;
	  to->cpt_si_fd = from->si_fd;
	  break;
	default:
	  to->cpt_si_pid = from->si_pid;
	  to->cpt_si_uid = from->si_uid;
	  to->cpt_si_ptr = (intptr_t) from->si_ptr;
	  break;
	}
    }
}

static void
siginfo_from_compat_siginfo (siginfo_t *to, compat_siginfo_t *from)
{
  memset (&to, 0, sizeof (to));

  to->si_signo = from->si_signo;
  to->si_errno = from->si_errno;
  to->si_code = from->si_code;

  if (to->si_code == SI_TIMER)
    {
      to->si_timerid = from->cpt_si_timerid;
      to->si_overrun = from->cpt_si_overrun;
      to->si_ptr = (void *) (intptr_t) from->cpt_si_ptr;
    }
  else if (to->si_code == SI_USER)
    {
      to->si_pid = from->cpt_si_pid;
      to->si_uid = from->cpt_si_uid;
    }
  if (to->si_code < 0)
    {
      to->si_pid = from->cpt_si_pid;
      to->si_uid = from->cpt_si_uid;
      to->si_ptr = (void *) (intptr_t) from->cpt_si_ptr;
    }
  else
    {
      switch (to->si_signo)
	{
	case SIGCHLD:
	  to->si_pid = from->cpt_si_pid;
	  to->si_uid = from->cpt_si_uid;
	  to->si_status = from->cpt_si_status;
	  to->si_utime = from->cpt_si_utime;
	  to->si_stime = from->cpt_si_stime;
	  break;
	case SIGILL:
	case SIGFPE:
	case SIGSEGV:
	case SIGBUS:
	  to->si_addr = (void *) (intptr_t) from->cpt_si_addr;
	  to->si_addr_lsb = (short) from->cpt_si_addr_lsb;
	  break;
	case SIGPOLL:
	  to->si_band = from->cpt_si_band;
	  to->si_fd = from->cpt_si_fd;
	  break;
	default:
	  to->si_pid = from->cpt_si_pid;
	  to->si_uid = from->cpt_si_uid;
	  to->si_ptr = (void* ) (intptr_t) from->cpt_si_ptr;
	  break;
	}
    }
}


static void
compat_x32_siginfo_from_siginfo (compat_x32_siginfo_t *to,
				 siginfo_t *from)
{
  memset (to, 0, sizeof (*to));

  to->si_signo = from->si_signo;
  to->si_errno = from->si_errno;
  to->si_code = from->si_code;

  if (to->si_code == SI_TIMER)
    {
      to->cpt_si_timerid = from->si_timerid;
      to->cpt_si_overrun = from->si_overrun;
      to->cpt_si_ptr = (intptr_t) from->si_ptr;
    }
  else if (to->si_code == SI_USER)
    {
      to->cpt_si_pid = from->si_pid;
      to->cpt_si_uid = from->si_uid;
    }
  else if (to->si_code < 0)
    {
      to->cpt_si_pid = from->si_pid;
      to->cpt_si_uid = from->si_uid;
      to->cpt_si_ptr = (intptr_t) from->si_ptr;
    }
  else
    {
      switch (to->si_signo)
	{
	case SIGCHLD:
	  to->cpt_si_pid = from->si_pid;
	  to->cpt_si_uid = from->si_uid;
	  to->cpt_si_status = from->si_status;
	  memcpy (&to->cpt_si_utime, &from->si_utime,
		  sizeof (to->cpt_si_utime));
	  memcpy (&to->cpt_si_stime, &from->si_stime,
		  sizeof (to->cpt_si_stime));
	  break;
	case SIGILL:
	case SIGFPE:
	case SIGSEGV:
	case SIGBUS:
	  to->cpt_si_addr = (intptr_t) from->si_addr;
	  to->cpt_si_addr_lsb = (short) from->si_addr_lsb;
	  break;
	case SIGPOLL:
	  to->cpt_si_band = from->si_band;
	  to->cpt_si_fd = from->si_fd;
	  break;
	default:
	  to->cpt_si_pid = from->si_pid;
	  to->cpt_si_uid = from->si_uid;
	  to->cpt_si_ptr = (intptr_t) from->si_ptr;
	  break;
	}
    }
}


static void
siginfo_from_compat_x32_siginfo (siginfo_t *to,
				 compat_x32_siginfo_t *from)
{
  memset (to, 0, sizeof (*to));

  to->si_signo = from->si_signo;
  to->si_errno = from->si_errno;
  to->si_code = from->si_code;

  if (to->si_code == SI_TIMER)
    {
      to->si_timerid = from->cpt_si_timerid;
      to->si_overrun = from->cpt_si_overrun;
      to->si_ptr = (void *) (intptr_t) from->cpt_si_ptr;
    }
  else if (to->si_code == SI_USER)
    {
      to->si_pid = from->cpt_si_pid;
      to->si_uid = from->cpt_si_uid;
    }
  if (to->si_code < 0)
    {
      to->si_pid = from->cpt_si_pid;
      to->si_uid = from->cpt_si_uid;
      to->si_ptr = (void *) (intptr_t) from->cpt_si_ptr;
    }
  else
    {
      switch (to->si_signo)
	{
	case SIGCHLD:
	  to->si_pid = from->cpt_si_pid;
	  to->si_uid = from->cpt_si_uid;
	  to->si_status = from->cpt_si_status;
	  memcpy (&to->si_utime, &from->cpt_si_utime,
		  sizeof (to->si_utime));
	  memcpy (&to->si_stime, &from->cpt_si_stime,
		  sizeof (to->si_stime));
	  break;
	case SIGILL:
	case SIGFPE:
	case SIGSEGV:
	case SIGBUS:
	  to->si_addr = (void *) (intptr_t) from->cpt_si_addr;
	  to->si_addr_lsb = (short) from->cpt_si_addr_lsb;
	  break;
	case SIGPOLL:
	  to->si_band = from->cpt_si_band;
	  to->si_fd = from->cpt_si_fd;
	  break;
	default:
	  to->si_pid = from->cpt_si_pid;
	  to->si_uid = from->cpt_si_uid;
	  to->si_ptr = (void* ) (intptr_t) from->cpt_si_ptr;
	  break;
	}
    }
}

/* Convert a native/host siginfo object, into/from the siginfo in the
   layout of the inferiors' architecture.  Returns true if any
   conversion was done; false otherwise.  If DIRECTION is 1, then copy
   from INF to NATIVE.  If DIRECTION is 0, copy from NATIVE to
   INF.  */

int
amd64_linux_siginfo_fixup_low (siginfo_t *native, gdb_byte *inf,
			       int direction,
			       enum amd64_siginfo_fixup_mode mode)
{

  if (mode == COMPAT32_FIXUP)
    {
      gdb_assert (sizeof (siginfo_t) == sizeof (compat_siginfo_t));

      if (direction == 0)
	compat_siginfo_from_siginfo ((struct compat_siginfo *) inf, native);
      else
	siginfo_from_compat_siginfo (native, (struct compat_siginfo *) inf);

      return 1;
    }
  /* No fixup for native x32 GDB.  */
  else if (mode == X32_FIXUP)
    {
      gdb_assert (sizeof (siginfo_t) == sizeof (compat_x32_siginfo_t));

      if (direction == 0)
	compat_x32_siginfo_from_siginfo ((struct compat_x32_siginfo *) inf,
					 native);
      else
	siginfo_from_compat_x32_siginfo (native,
					 (struct compat_x32_siginfo *) inf);

      return 1;
    }
  else
    return 0;
}
