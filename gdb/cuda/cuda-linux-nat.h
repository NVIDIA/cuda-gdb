/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2024 NVIDIA Corporation
 * Written by CUDA-GDB team at NVIDIA <cudatools@nvidia.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _CUDA_LINUX_NAT_H
#define _CUDA_LINUX_NAT_H 1

#include "defs.h"

#include <sys/signal.h>

#include <objfiles.h>

#include "bfd.h"
#include "cudadebugger.h"
#include "elf-bfd.h"

#include "arch-utils.h"
#include "cuda-commands.h"
#include "cuda-convvars.h"
#include "cuda-events.h"
#include "cuda-exceptions.h"
#include "cuda-notifications.h"
#include "cuda-options.h"
#include "cuda-packet-manager.h"
#include "cuda-state.h"
#include "cuda-tdep.h"
#include "cuda-utils.h"
#include "gdbthread.h"
#include "inf-child.h"
#include "linux-nat.h"

/* Various state used by cuda-linux-nat-template.h */
extern struct cuda_cudart_symbols_st cuda_cudart_symbols;
extern bool cuda_debugging_enabled;
extern int cuda_host_want_singlestep;

extern void cuda_signal_set_silent (int sig, struct cuda_signal_info_st *save);
extern void cuda_signal_restore_settings (int sig,
					  struct cuda_signal_info_st *save);
extern void cuda_sigtrap_restore_settings (void);
extern void cuda_sigtrap_set_silent (void);
extern bool cuda_check_pending_sigint (pid_t pid);
extern struct objfile *cuda_create_builtins_objfile (void);

class inf_child_target;

template <class BaseTarget> struct cuda_nat_linux : public BaseTarget
{
public:
  target_info m_info = { 0 };
  bool m_resumed_from_fatal_exception = false;
  bool m_send_event_ack = false;

  cuda_nat_linux ();

  /* Return a reference to this target's unique target_info
     object.  */
  virtual const target_info &
  info () const override
  {
    return m_info;
  }

  /* Disable scheduler locking for cuda targets */
  thread_control_capabilities
  get_thread_control_capabilities () override
  {
    return tc_none;
  }

  virtual void kill () override;

  virtual enum target_xfer_status
  xfer_partial (enum target_object object, const char *annex,
		gdb_byte *readbuf, const gdb_byte *writebuf, ULONGEST offset,
		ULONGEST len, ULONGEST *xfered_len) override;

  virtual void mourn_inferior () override;

  virtual void resume (ptid_t arg0,
		       int TARGET_DEBUG_PRINTER (target_debug_print_step) arg1,
		       enum gdb_signal arg2) override;

  void resume (ptid_t ptid, int sstep, int host_sstep,
	       enum gdb_signal ts);

  virtual ptid_t wait (ptid_t arg0, struct target_waitstatus *arg1,
		       target_wait_flags arg2) override;

  virtual void fetch_registers (struct regcache *, int) override;

  virtual void store_registers (struct regcache *, int) override;

  virtual int insert_breakpoint (struct gdbarch *,
				 struct bp_target_info *) override;

  virtual int remove_breakpoint (struct gdbarch *, struct bp_target_info *,
				 enum remove_bp_reason) override;

  virtual struct gdbarch *thread_architecture (ptid_t) override;

  virtual void detach (inferior *, int) override;

private:
  enum target_xfer_status
  cuda_xfer_siginfo (enum target_object object, const char *annex,
		     gdb_byte *readbuf, const gdb_byte *writebuf,
		     ULONGEST offset, LONGEST len, ULONGEST *xfered_len);
};

#endif
