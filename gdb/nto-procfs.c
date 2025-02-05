/* Machine independent support for QNX Neutrino /proc (process file system)
   for GDB.  Written by Colin Burgess at QNX Software Systems Limited.

   Copyright (C) 2003-2022 Free Software Foundation, Inc.

   Contributed by QNX Software Systems Ltd.

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

#include "defs.h"

#include <fcntl.h>
#include <spawn.h>
#include <sys/debug.h>
#include <sys/procfs.h>
#include <sys/neutrino.h>
#include <sys/syspage.h>
#include <dirent.h>
#include <sys/netmgr.h>
#include <sys/auxv.h>
#include <signal.h>

#include "gdbcore.h"
#include "inferior.h"
#include "target.h"
#include "target-descriptions.h"
#include "objfiles.h"
#include "gdbthread.h"
#include "nto-tdep.h"
#include "command.h"
#include "regcache.h"
#include "solib.h"
#include "inf-child.h"
#include "gdbsupport/filestuff.h"
#include "observable.h"

#include <sys/procmgr.h>

#define NULL_PID    0
#define _DEBUG_FLAG_TRACE  (_DEBUG_FLAG_TRACE_EXEC|_DEBUG_FLAG_TRACE_RD|\
    _DEBUG_FLAG_TRACE_WR|_DEBUG_FLAG_TRACE_MODIFY)

typedef debug_thread64_t nto_procfs_status;

int ctl_fd = -1;
// class members will need access to these for pulses
int com_coid {-1};
int com_chid {-1};
struct sigevent com_event;

bool initialized_wait {false};

static procfs_run run;

static ptid_t do_attach (ptid_t ptid);

static void pulse_setup ();

static const target_info nto_procfs_target_info = {
  "native",
  N_("QNX Neutrino local process"),
  N_("QNX Neutrino local process (started by the \"run\" command).")
};

/*
 * todo it may make sense to put the session data in here too..
 */
class nto_procfs final : public inf_child_target
{
public:
  /* we have no private data yet but that may follow..
  ~nto_procfs () override;
  */

  const target_info &info () const override
  { return nto_procfs_target_info; }

  static void nto_open (const char *, int);
  // void close () override;

  bool can_attach () override {
    return true;
  }

  void attach ( const char*, int ) override;
  void post_attach( int ) override;
  void detach (inferior *, int) override;
//  void disconnect (const char *, int) override;
  void resume (ptid_t, int TARGET_DEBUG_PRINTER (target_debug_print_step), enum gdb_signal) override;
  ptid_t wait (ptid_t, struct target_waitstatus *, target_wait_flags) override;
  void fetch_registers (struct regcache *, int) override;
  void store_registers (struct regcache *, int) override;
//  void prepare_to_store (struct regcache *) override;
  void files_info () override;
  int insert_breakpoint (struct gdbarch *, struct bp_target_info *) override;
  int remove_breakpoint (struct gdbarch *, struct bp_target_info *, enum remove_bp_reason) override;
  int can_use_hw_breakpoint (enum bptype, int, int) override;
  int insert_hw_breakpoint (struct gdbarch *, struct bp_target_info *) override;
  int remove_hw_breakpoint (struct gdbarch *, struct bp_target_info *) override;
  int remove_watchpoint (CORE_ADDR, int, enum target_hw_bp_type, struct expression *) override;
  int insert_watchpoint (CORE_ADDR, int, enum target_hw_bp_type, struct expression *) override;
  bool stopped_by_watchpoint () override {
    return nto_stopped_by_watchpoint();
  }
#if 0
  void terminal_init () override;
  void terminal_inferior () override;
  void terminal_ours_for_output () override;
  void terminal_info (const char *, int) override;
#endif
  void kill () override;
  void load (const char *args, int from_tty ) override {
    return generic_load (args, from_tty);
  }
  bool can_create_inferior () override {
    return true;
  }
/* todo */
  //int insert_fork_catchpoint (int) override;
  //int remove_fork_catchpoint (int) override;
  //int insert_vfork_catchpoint (int) override;
  //int remove_vfork_catchpoint (int) override;
  //int follow_fork (int, int) override;
  //int insert_exec_catchpoint (int) override;
  //int remove_exec_catchpoint (int) override;
  void create_inferior (const char *, const std::string &, char **, int) override;
  void mourn_inferior () override;
  bool can_run () override {
    return false;
  }
  bool thread_alive (ptid_t ptid) override;
  void update_thread_list () override;
  std::string pid_to_str (ptid_t ptid) override {
    return nto_pid_to_str (ptid);
  }
  char *pid_to_exec_file (int pid) override;
  const char *extra_thread_info (thread_info *ti ) override {
    return nto_extra_thread_info( ti );
  }

  bool can_async_p () override {
    /* Not yet. */
    return false;
  };
  bool supports_non_stop () override {
    /* Not yet. */
    return false;
  }
  const struct target_desc *read_description () override;
  void stop (ptid_t) override;
  void interrupt () override;
  void pass_ctrlc () override;
  bool supports_multi_process () override {
    return true;
  }
//  int verify_memory (const gdb_byte *data, CORE_ADDR memaddr, ULONGEST size) override;
  enum target_xfer_status xfer_partial (enum target_object object,
                const char *annex,
                gdb_byte *readbuf,
                const gdb_byte *writebuf,
                ULONGEST offset, ULONGEST len,
                ULONGEST *xfered_len) override;

  void pass_signals (gdb::array_view<const unsigned char>) override;

public:
  bool have_continuable_watchpoint (void) {
    return true;
  }
};

static nto_procfs nto_procfs_target;

/*
 * wrapper for default close()
 */
static void
close1 ( int fd ) {
  close( fd );
}

/*  Set up pulse notification.  */
static void pulse_setup () {
  /*
   * If channel exists, we must have an event registered.
   */

  if (com_chid != -1) {
    goto debug_and_return;
  }

  if ((com_chid = ChannelCreate(_NTO_CHF_PRIVATE)) == -1) {
     error(_("Failed to create channel"));
  }

  /*
   * Connect to the channel
   */
  if ((com_coid = ConnectAttach(0, 0, com_chid, _NTO_SIDE_CHANNEL, 0)) == -1) {
     ChannelDestroy(com_chid);
     com_chid = -1;
     error(_("Failed to connect to channel"));
  }
  /*
   * Set up a pulse event to be received on process's channel from kernel.
   */
  SIGEV_PULSE_INIT(&com_event, com_coid, -1, _PULSE_CODE_MINAVAIL, NULL);
  if((MsgRegisterEvent(&com_event, SYSMGR_COID)) != EOK) {
     ChannelDestroy(com_chid);
     com_chid = -1;
     error(_("Failed to register pulse event!"));
  }
debug_and_return:
  nto_trace(0)("Successfully established connection with kernel using coid: %d on chid: %d!\n", com_coid, com_chid);

  return;

}

/* This is called when we call 'target native' or 'target procfs
   <arg>' from the (gdb) prompt. */
void
nto_procfs::nto_open (const char *arg, int from_tty)
{
  char buffer[50];
  int fd, total_size;
  procfs_sysinfo *sysinfo;
//  struct cleanup *cleanups;
  char nto_procfs_path[PATH_MAX]="/proc";

  /* Offer to kill previous inferiors before opening this target.  */
  target_preopen (from_tty);

  init_thread_list ();

  fd = open (nto_procfs_path, O_RDONLY);
  if (fd == -1)
    {
      printf_filtered ("Error opening %s : %d (%s)\n", nto_procfs_path, errno,
           safe_strerror (errno));
      error (_("Invalid procfs arg"));
    }
//  cleanups = make_cleanup_close (fd);

  sysinfo = (procfs_sysinfo *) buffer;
  if (devctl (fd, DCMD_PROC_SYSINFO, sysinfo, sizeof buffer, 0) != EOK)
    {
      printf_filtered ("Error getting size: %d (%s)\n", errno,
           safe_strerror (errno));
      error (_("Devctl failed."));
    }
  else
    {
      total_size = sysinfo->total_size;
      sysinfo = (procfs_sysinfo *) alloca (total_size);
      if (sysinfo == NULL)
  {
    printf_filtered ("Memory error: %d (%s)\n", errno,
         safe_strerror (errno));
    error (_("alloca failed."));
  }
      else
  {
    if (devctl (fd, DCMD_PROC_SYSINFO, sysinfo, total_size, 0) != EOK)
      {
        printf_filtered ("Error getting sysinfo: %d (%s)\n", errno,
             safe_strerror (errno));
        error (_("Devctl failed."));
      }
    else
      {
        if (sysinfo->type !=
      nto_map_arch_to_cputype (gdbarch_bfd_arch_info
             (target_gdbarch ())->arch_name))
    error (_("Invalid target CPU."));
      }
  }
    }
//  do_cleanups (cleanups);
  close1(fd);

  inf_child_open_target (arg, from_tty);
  printf_filtered ("Debugging using %s\n", nto_procfs_path);
}

static void
procfs_set_thread (ptid_t ptid)
{
  int tid;

  tid = ptid.lwp();
  devctl (ctl_fd, DCMD_PROC_CURTHREAD, &tid, sizeof (tid), 0);
}

/* wrapper to access native kill() function */
static int
kill_1( pid_t pid, int sig )
{
  return kill( pid, sig );
}

/*  Return nonzero if the ptid is still alive.  */
bool
nto_procfs::thread_alive (ptid_t ptid)
{
  int tid;
  pid_t pid;
  procfs_status status;
  int err;

  tid = ptid.lwp ();
  pid = ptid.pid ();

  if ( kill_1 (pid, 0) == -1)
    return false;

  status.tid = tid;
  if ((err = devctl (ctl_fd, DCMD_PROC_TIDSTATUS,
         &status, sizeof (status), 0)) != EOK)
    return false;

  /* Thread is alive or dead but not yet joined,
     or dead and there is an alive (or dead unjoined) thread with
     higher tid.
     If the tid is not the same as requested, requested tid is dead.  */
  return (status.tid == tid) && (status.state != STATE_DEAD);
}

/* the NTO specific way of gathering thread information */
static void
update_thread_private_data (struct thread_info *new_thread,
          pthread_t tid, int state, int flags)
{
  struct nto_thread_info *pti;
  procfs_info pidinfo;
  struct _thread_name *tn;
  procfs_threadctl tctl;

  gdb_assert (new_thread != NULL);

  if (devctl (ctl_fd, DCMD_PROC_INFO, &pidinfo,
        sizeof(pidinfo), 0) != EOK)
    return;

  memset (&tctl, 0, sizeof (tctl));
  tctl.cmd = _NTO_TCTL_NAME;
  tn = (struct _thread_name *) (&tctl.data);

  /* Fetch name for the given thread.  */
  tctl.tid = tid;
  tn->name_buf_len = sizeof (tctl.data) - sizeof (*tn);
  tn->new_name_len = -1; /* Getting, not setting.  */
  if (devctl (ctl_fd, DCMD_PROC_THREADCTL, &tctl, sizeof (tctl), NULL) != EOK)
    tn->name_buf[0] = '\0';

  tn->name_buf[_NTO_THREAD_NAME_MAX] = '\0';

  if (!new_thread->priv)
    new_thread->priv.reset(new nto_thread_info());

  pti = (struct nto_thread_info *) new_thread->priv.get();
  new_thread->name=xstrdup(tn->name_buf);
  pti->state = state;
  pti->flags = flags;
}

/**
 * Checks all active threads of the current process and creates/updates
 * the thread information.
 */
void
nto_procfs::update_thread_list ()
{
  procfs_status status;
  int pid;
  ptid_t ptid;
  struct thread_info *current_thread;

  if (ctl_fd == -1)
    return;

  prune_threads ();
  pid = inferior_ptid.pid ();
  /* NTO thread IDs start with 1. */
  status.tid = 1;

  /* Check for thread info. If the thread ID status.tid does not exist but another
   * one with a higher tid does, the devctl updates the status.tid accordingly */
  while (devctl (ctl_fd, DCMD_PROC_TIDSTATUS, &status, sizeof(status), 0) == EOK)
    {
      ptid = ptid_t (pid, status.tid, 0);
      /* check if the thread is already known */
      current_thread = nto_find_thread (ptid);
      if (!current_thread)
	current_thread = add_thread (this, ptid);
      update_thread_private_data (current_thread, status.tid, status.state, 0);
      /* fetch the next thread */
      status.tid++;
    }
  return;
}


static void
procfs_pidlist (const char *args, int from_tty)
{
  DIR *dp = NULL;
  struct dirent *dirp = NULL;
  char buf[PATH_MAX];
  procfs_info *pidinfo = NULL;
  procfs_debuginfo *info = NULL;
  procfs_status *status = NULL;
  int num_threads = 0;
  int pid;
  char name[PATH_MAX];
  /* todo: adapt to new/current cleanup mechanism */
//  struct cleanup *cleanups;
  char procfs_dir[PATH_MAX]="/proc";

  dp = opendir (procfs_dir);
  if (dp == NULL)
    {
      fprintf_unfiltered (gdb_stderr, "failed to opendir \"%s\" - %d (%s)",
        procfs_dir, errno, safe_strerror (errno));
      return;
    }

//  cleanups = make_cleanup (do_closedir_cleanup, dp);

  /* Start scan at first pid.  */
  rewinddir (dp);

  do
    {
      int fd;
//      struct cleanup *inner_cleanup;

      /* Get the right pid and procfs path for the pid.  */
      do
        {
          dirp = readdir (dp);
          if (dirp == NULL)
            {
//              do_cleanups (cleanups);
              return;
            }
          snprintf (buf, sizeof (buf), "/proc/%s/as", dirp->d_name);
          pid = atoi (dirp->d_name);
        }
      while (pid == 0);

      /* Open the procfs path.  */
      fd = open (buf, O_RDONLY);
      if (fd == -1)
        {
          fprintf_unfiltered (gdb_stderr, "failed to open %s - %d (%s)\n",
              buf, errno, safe_strerror (errno));
          continue;
        }
//      inner_cleanup = make_cleanup_close (fd);

      pidinfo = (procfs_info *) buf;
      if (devctl (fd, DCMD_PROC_INFO, pidinfo, sizeof (buf), 0) != EOK)
        {
          fprintf_unfiltered (gdb_stderr,
              "devctl DCMD_PROC_INFO failed - %d (%s)\n",
              errno, safe_strerror (errno));
          break;
        }
      num_threads = pidinfo->num_threads;

      info = (procfs_debuginfo *) buf;
      if (devctl (fd, DCMD_PROC_MAPDEBUG_BASE, info, sizeof (buf), 0) != EOK)
         strcpy (name, "unavailable");
      else
         strcpy (name, info->path);

      /* Collect state info on all the threads.  */
      status = (procfs_status *) buf;
      for (status->tid = 1; status->tid <= num_threads; status->tid++)
        {
          const int err
              = devctl (fd, DCMD_PROC_TIDSTATUS, status, sizeof (buf), 0);
          printf_filtered ("%s - %d", name, pid);
          if (err == EOK && status->tid != 0)
             printf_filtered ("/%d\n", status->tid);
          else
            {
              printf_filtered ("\n");
              break;
            }
        }

//      do_cleanups (inner_cleanup);
    }
  while (dirp != NULL);

//  do_cleanups (cleanups);
  closedir (dp);
  return;
}

static void
procfs_meminfo (const char *args, int from_tty)
{
  procfs_mapinfo *mapinfos = NULL;
  static int num_mapinfos = 0;
  procfs_mapinfo *mapinfo_p, *mapinfo_p2;
  int flags = ~0, err, num, i, j;

  struct
  {
    procfs_debuginfo info;
    char buff[_POSIX_PATH_MAX];
  } map;

  // todo: unsigned is not enough for 64bit!
  struct info
  {
    unsigned addr;
    unsigned size;
    unsigned flags;
    unsigned debug_vaddr;
    unsigned long long offset;
  };

  struct printinfo
  {
    unsigned long long ino;
    unsigned dev;
    struct info text;
    struct info data;
    char name[256];
  } printme;

  /* can we even get something? */
  if (ctl_fd == -1)
    {
      warning("Not attached to a process yet.");
      return;
    }

  /* Get the number of map entrys.  */
  err = devctl (ctl_fd, DCMD_PROC_MAPINFO, NULL, 0, &num);
  if (err != EOK)
    {
      printf ("failed devctl num mapinfos - %d (%s)\n", err,
        safe_strerror (err));
      return;
    }

  mapinfos = XNEWVEC (procfs_mapinfo, num);

  num_mapinfos = num;
  mapinfo_p = mapinfos;

  /* Fill the map entrys.  */
  err = devctl (ctl_fd, DCMD_PROC_MAPINFO, mapinfo_p, num
    * sizeof (procfs_mapinfo), &num);
  if (err != EOK)
    {
      printf ("failed devctl mapinfos - %d (%s)\n", err, safe_strerror (err));
      xfree (mapinfos);
      return;
    }

  num = (num < num_mapinfos)?num:num_mapinfos;

  /* Run through the list of mapinfos, and store the data and text info
     so we can print it at the bottom of the loop.  */
  for (mapinfo_p = mapinfos, i = 0; i < num; i++, mapinfo_p++)
    {
      if (!(mapinfo_p->flags & flags))
          mapinfo_p->ino = 0;

      if (mapinfo_p->ino == 0)  /* Already visited.  */
          continue;

      map.info.vaddr = mapinfo_p->vaddr;

      err = devctl (ctl_fd, DCMD_PROC_MAPDEBUG, &map, sizeof (map), 0);
      if (err != EOK)
          continue;

      memset (&printme, 0, sizeof printme);
      printme.dev = mapinfo_p->dev;
      printme.ino = mapinfo_p->ino;
      printme.text.addr = mapinfo_p->vaddr;
      printme.text.size = mapinfo_p->size;
      printme.text.flags = mapinfo_p->flags;
      printme.text.offset = mapinfo_p->offset;
      printme.text.debug_vaddr = map.info.vaddr;
      strcpy (printme.name, map.info.path);

      /* Check for matching data.  */
      for (mapinfo_p2 = mapinfos, j = 0; j < num; j++, mapinfo_p2++)
        {
          if (mapinfo_p2->vaddr != mapinfo_p->vaddr
              && mapinfo_p2->ino == mapinfo_p->ino
              && mapinfo_p2->dev == mapinfo_p->dev)
            {
              map.info.vaddr = mapinfo_p2->vaddr;
              err =
                  devctl (ctl_fd, DCMD_PROC_MAPDEBUG, &map, sizeof (map), 0);
              if (err != EOK)
                  continue;

              if (strcmp (map.info.path, printme.name))
                  continue;

              /* Lower debug_vaddr is always text, if necessary, swap.  */
              if ((int) map.info.vaddr < (int) printme.text.debug_vaddr)
                {
                  memcpy (&(printme.data), &(printme.text),
                  sizeof (printme.data));
                  printme.text.addr = mapinfo_p2->vaddr;
                  printme.text.size = mapinfo_p2->size;
                  printme.text.flags = mapinfo_p2->flags;
                  printme.text.offset = mapinfo_p2->offset;
                  printme.text.debug_vaddr = map.info.vaddr;
                }
              else
                {
                  printme.data.addr = mapinfo_p2->vaddr;
                  printme.data.size = mapinfo_p2->size;
                  printme.data.flags = mapinfo_p2->flags;
                  printme.data.offset = mapinfo_p2->offset;
                  printme.data.debug_vaddr = map.info.vaddr;
                }
              mapinfo_p2->ino = 0;
            }
        }
      mapinfo_p->ino = 0;

      printf_filtered ("%s\n", printme.name);
      printf_filtered ("\ttext=%08x bytes @ 0x%08x\n", printme.text.size,
           printme.text.addr);
      printf_filtered ("\t\tflags=%08x\n", printme.text.flags);
      printf_filtered ("\t\tdebug=%08x\n", printme.text.debug_vaddr);
      printf_filtered ("\t\toffset=%s\n", phex (printme.text.offset, 8));
      if (printme.data.size)
        {
          printf_filtered ("\tdata=%08x bytes @ 0x%08x\n", printme.data.size,
             printme.data.addr);
          printf_filtered ("\t\tflags=%08x\n", printme.data.flags);
          printf_filtered ("\t\tdebug=%08x\n", printme.data.debug_vaddr);
          printf_filtered ("\t\toffset=%s\n", phex (printme.data.offset, 8));
        }
      printf_filtered ("\tdev=0x%x\n", printme.dev);
      printf_filtered ("\tino=0x%x\n", (unsigned int) printme.ino);
    }
  xfree (mapinfos);
  return;
}

/* Print status information about what we're accessing.  */
void
nto_procfs::files_info ( )
{
  struct inferior *inf = current_inferior ();

  printf_unfiltered ("\tUsing the running image of %s %s.\n",
         inf->attach_flag ? "attached" : "child",
         target_pid_to_str (inferior_ptid).c_str () );
}

/* Map a pid to a filename
 * needed for exec_file_locate_attach  */

char *
nto_procfs::pid_to_exec_file (int pid)
{
  int proc_fd;
  static char proc_path[PATH_MAX];
  ssize_t rd;

  /* Read exe file name.  */
  snprintf (proc_path, sizeof (proc_path), "/proc/%d/exefile", pid);
  proc_fd = open (proc_path, O_RDONLY);
  if (proc_fd == -1)
      return NULL;

  rd = read (proc_fd, proc_path, sizeof (proc_path) - 1);
  close1 (proc_fd);
  if (rd <= 0)
    {
      warning("Pid %i has no executable name!", pid);
      proc_path[0] = '\0';
      return NULL;
    }
  proc_path[rd] = '\0';

  if (strrchr(proc_path, '/'))
    return strrchr(proc_path, '/')+1;

  return proc_path;
}

/* Attach to process PID, then initialize for debugging it.  */
void
nto_procfs::attach (const char *args, int from_tty)
{
  int pid;
  struct inferior *inf;

  pid = parse_pid_to_attach (args);

  if (pid == getpid ())
    error (_("Attaching GDB to itself is not a good idea..."));

  ptid_t ptid = ptid_t (pid, 0, 0);

  const char *exec_file = get_exec_file (0);

  if (from_tty)
    {
      if (exec_file)
          printf_unfiltered ("Attaching to program `%s', %s\n", exec_file,
              nto_pid_to_str (ptid).c_str ());
      else
          printf_unfiltered ("Attaching to %s\n",
              nto_pid_to_str (ptid).c_str ());

      gdb_flush (gdb_stdout);
    }

  /* do_attach may change the tid */
  ptid = do_attach (ptid);
  inferior_ptid = ptid;

  inf = nto_add_inferior (this, pid);
  inf->attach_flag = true;
  inf->removable = false;

  /* add and switch to the current thread */
  thread_info *tp = add_thread (this, ptid);
  switch_to_thread_no_regs (tp);

  // No file has been loaded, so use the pid to identify the name
  if (exec_file == NULL)
    exec_file_locate_attach(pid, false, from_tty);

  /* update all thread information */
  update_thread_list ();
}

void
nto_procfs::post_attach (int pid)
{
  nto_trace(0) ("%s (%i)\n", __func__, pid );

  if (current_program_space->exec_bfd ())
    solib_create_inferior_hook (0);
}

static ptid_t
do_attach (ptid_t ptid)
{
  procfs_status status;
  char path[PATH_MAX];
  nto_trace(0)("%s\n", __func__);

  snprintf (path, PATH_MAX - 1, "/proc/%d/as", ptid.pid () );
  ctl_fd = open (path, O_RDWR);
  if (ctl_fd == -1)
      error (_("Couldn't open proc file %s, error %d (%s)"), path, errno,
          safe_strerror (errno));
  if (devctl (ctl_fd, DCMD_PROC_STOP, &status, sizeof (status), 0) != EOK)
      error (_("Couldn't stop process"));


  pulse_setup ();
  if (devctl (ctl_fd, DCMD_PROC_EVENT, &com_event, sizeof (com_event), 0) != 0)
      error (_("Failed to assign event"));

  if (devctl (ctl_fd, DCMD_PROC_STATUS, &status, sizeof (status), 0) == EOK
      && ( status.flags & _DEBUG_FLAG_STOPPED) )
      SignalKill ( ND_LOCAL_NODE, ptid.pid (), 0, SIGCONT, 0, 0);

  return ptid_t (ptid.pid (), status.tid, 0);
}

static const char* nto_get_status(procfs_status status)
{
  if (status.flags & _DEBUG_FLAG_SSTEP)
    {
      return("single step");
    }
  /* Was it a break-/watchpoint?  */
  else if (status.flags & _DEBUG_FLAG_TRACE_EXEC)
    {
      return("hit breakpoint");
    }
  else if (status.flags & _DEBUG_FLAG_TRACE)
    {
      return("hit watchpoint");
    }
  else if (status.flags & _DEBUG_FLAG_ISTOP)
    {
      switch (status.why)
        {
        case _DEBUG_WHY_SIGNALLED:
          return ("stop on signal");
        break;
        case _DEBUG_WHY_FAULTED:
          return ("stop on fault");
        break;
        case _DEBUG_WHY_TERMINATED:
          return ("terminated");
        break;

        case _DEBUG_WHY_REQUESTED:
          return ("stop on interrupt");
        break;
        default:
          return ("stopped for the weather");
        }
    }
  else if (status.flags & _DEBUG_FLAG_STOPPED)
    {
      return ("process stopped");
    }
  else if (status.flags & _DEBUG_FLAG_IPINVAL)
    {
      return ("IP is not valid");
    }
  else
    {
      return ("unknown signal!");
    }
}

ptid_t
nto_procfs::wait (ptid_t ptid, struct target_waitstatus *ourstatus, target_wait_flags options)
{
  struct _pulse rpulse;
  procfs_status status;
  static enum gdb_signal exit_signo = GDB_SIGNAL_0;  /* To track signals that cause termination.  */
  int err = 0;

  nto_trace (0) ("procfs_wait pid %d, inferior pid %d tid %ld\n",
     ptid.pid(), inferior_ptid.pid(), ptid.lwp());

  ourstatus->kind = TARGET_WAITKIND_SPURIOUS;

  if ( inferior_ptid == null_ptid )
    {
      gdb_assert(current_inferior() != NULL);
      if (current_inferior()->pid == 0)
	{
	  warning ("No current inferior process!");
	  ourstatus->kind = TARGET_WAITKIND_STOPPED;
	  ourstatus->value.sig = GDB_SIGNAL_0;
	  exit_signo = GDB_SIGNAL_0;
	  return null_ptid;
	}

      nto_trace(0) ("  setting current inferior\n");
      inferior_ptid = ptid_t( current_inferior()->pid );
      nto_trace(0) ("%s: set current inferior to %d\n",__func__, inferior_ptid.pid());
    }

  /*
   * if process is already stopped first time we wait, it should be started first.
   * this is so we can arm the run with proper flags to get events from kernel.
   * don't wait on kernel if that's the case and just move on.
   */
  err = devctl (ctl_fd, DCMD_PROC_STATUS, &status, sizeof (status), 0);
  if (err != EOK) {
	      error(_("%s: error checking status %s\n"), __func__, strerror(err));
  }

  // check only if neither stopped nor first time running wait
  if (!((status.flags & _DEBUG_FLAG_STOPPED) && initialized_wait == false)) {
	  // Block until kernel notifies us
	  err = MsgReceivePulse(com_chid, &rpulse, sizeof(rpulse), NULL);
	  nto_trace(0)("Process received pulse! %d\n", rpulse.value.sival_int);
	  if (err == -1)
	  {
	      error(_("%s: error receiving pulse"), __func__);
	  }
  }

  // after the first wait call, we should have run properly armed.
  initialized_wait = true;

  // check status to see why kernel notified us
  err = devctl (ctl_fd, DCMD_PROC_STATUS, &status, sizeof (status), 0);
  if (err != EOK)
  {
      /*
       * Hack to circumvent the fact that we have no way of knowing what
       * happened to the thread if it no longer exists.
       * Eventually we want kernel to tell us in pulse event why it stopped us
       * and then we can avoid this whole thing.
       */
      if (err == ESRCH)
        {
          // from a development pov this should be a warning, from a user pov
          // it's irrelevant, so just report when asked for.
          nto_trace(0)("Process died without termination message!\n");
          status.flags = _DEBUG_FLAG_ISTOP;
          status.why = _DEBUG_WHY_TERMINATED;
          status.pid = inferior_ptid.pid();
          status.tid = inferior_ptid.tid();
        }
      else
      {
	      error(_("%s: error checking status %s\n"), __func__, strerror(err));
      }
  }
  nto_inferior_data (NULL)->stopped_flags = status.flags;
  nto_inferior_data (NULL)->stopped_pc = status.ip;

  nto_trace(0) ("%s:  status.flags: 0x%x status.ip: 0x%p status.what %d status.why %d\n",
      __func__, status.flags, (void *)status.ip, status.what, status.why);
  nto_trace(0) ("%s:  status: %s from %i/%i\n",__func__, nto_get_status(status), status.pid, status.tid);

  if (status.flags & _DEBUG_FLAG_SSTEP)
    {
      ourstatus->kind = TARGET_WAITKIND_STOPPED;
      ourstatus->value.sig = GDB_SIGNAL_TRAP;
    }
  /* Was it a breakpoint?  */
  else if (status.flags & _DEBUG_FLAG_TRACE)
    {
      ourstatus->kind = TARGET_WAITKIND_STOPPED;
      ourstatus->value.sig = GDB_SIGNAL_TRAP;
    }
  else if (status.flags & _DEBUG_FLAG_ISTOP)
    {
      switch (status.why)
        {
        case _DEBUG_WHY_SIGNALLED:
          ourstatus->kind = TARGET_WAITKIND_STOPPED;
          ourstatus->value.sig =
              gdb_signal_from_host (status.info.si_signo);
          exit_signo = GDB_SIGNAL_0;
        break;
        case _DEBUG_WHY_FAULTED:
          ourstatus->kind = TARGET_WAITKIND_STOPPED;
          if (status.info.si_signo == SIGTRAP)
            {
              ourstatus->value.sig = GDB_SIGNAL_0;
              exit_signo = GDB_SIGNAL_0;
            }
          else
            {
              ourstatus->value.sig =
                  gdb_signal_from_host (status.info.si_signo);
              exit_signo = ourstatus->value.sig;
            }
        break;

        case _DEBUG_WHY_TERMINATED:
          {
            int waitval = 0;

            nto_trace(0)("Waiting for the process to die..\n");
            waitpid (inferior_ptid.pid(), &waitval, WNOHANG);
            if (exit_signo)
              {
                /* Abnormal death.  */
                ourstatus->kind = TARGET_WAITKIND_SIGNALLED;
                ourstatus->value.sig = exit_signo;
                nto_trace(0) ("Abnormal end!\n");
              }
            else
              {
                /* Normal death.  */
                ourstatus->kind = TARGET_WAITKIND_EXITED;
                ourstatus->value.integer = WEXITSTATUS (waitval);
                nto_trace(0) ("Normal exit: %x\n", WEXITSTATUS (waitval));
              }
            exit_signo = GDB_SIGNAL_0;
          }
          break;

        case _DEBUG_WHY_REQUESTED:
          nto_trace(0) ("  got ISTOP/REQUESTED reply\n");
          /* We are assuming a requested stop is due to a SIGINT.  */
          ourstatus->kind = TARGET_WAITKIND_STOPPED;
          ourstatus->value.sig = GDB_SIGNAL_INT;
          exit_signo = GDB_SIGNAL_0;
        break;
        }
    }

  return ptid_t (status.pid, status.tid, 0);
}

/* Read the current values of the inferior's registers, both the
   general register set and floating point registers (if supported)
   and update gdb's idea of their current values.  */
void
nto_procfs::fetch_registers (struct regcache *regcache, int regno)
{
  union
  {
    procfs_greg greg;
    procfs_fpreg fpreg;
    procfs_altreg altreg;
  }
  reg;
  int regsize;
  ptid_t ptid=regcache->ptid ();

  nto_trace(0)("nto_procfs::fetch_registers (#%i) for %s\n", regno, nto_pid_to_str(ptid).c_str());
  procfs_set_thread (ptid);
  int rv = devctl (ctl_fd, DCMD_PROC_GETGREG, &reg, sizeof (reg), &regsize);
  if ( rv == EOK)
    {
      nto_supply_gregset (regcache, (const gdb_byte *) &reg.greg, regsize);
    }
  else
    {
      warning("Could not fetch general registers: %s!", strerror(rv));
    }
  if (devctl (ctl_fd, DCMD_PROC_GETFPREG, &reg, sizeof (reg), &regsize) == EOK)
    {
      nto_supply_fpregset (regcache, (const gdb_byte *) &reg.fpreg, regsize);
    }
  if (devctl (ctl_fd, DCMD_PROC_GETALTREG, &reg, sizeof (reg), &regsize) == EOK)
    {
      nto_supply_altregset (regcache, (const gdb_byte*) &reg.altreg, regsize);
    }
}

/* Helper for procfs_xfer_partial that handles memory transfers.
   Arguments are like target_xfer_partial.  */
static enum target_xfer_status
procfs_xfer_memory (gdb_byte *readbuf, const gdb_byte *writebuf,
        ULONGEST memaddr, ULONGEST len, ULONGEST *xfered_len)
{
  int nbytes;

  if (lseek (ctl_fd, (off_t) memaddr, SEEK_SET) != (off_t) memaddr)
      return TARGET_XFER_E_IO;

  if (writebuf != NULL)
      nbytes = write (ctl_fd, writebuf, len);
  else
      nbytes = read (ctl_fd, readbuf, len);
  if (nbytes <= 0)
      return TARGET_XFER_E_IO;
  *xfered_len = nbytes;
  return TARGET_XFER_OK;
}

/* Target to_xfer_partial implementation.  */
enum target_xfer_status
nto_procfs::xfer_partial ( enum target_object object,
         const char *annex, gdb_byte *readbuf,
         const gdb_byte *writebuf, ULONGEST offset, ULONGEST len,
         ULONGEST *xfered_len)
{
  switch (object)
    {
    case TARGET_OBJECT_MEMORY:
      return procfs_xfer_memory (readbuf, writebuf, offset, len, xfered_len);
    case TARGET_OBJECT_SIGNAL_INFO:
      if (readbuf != NULL)
        {
          siginfo_t siginfo;
          nto_procfs_status status;
          const size_t sizeof_status = sizeof (nto_procfs_status);
          const size_t sizeof_siginfo = sizeof (siginfo);

          int err;
          LONGEST mylen = len;

          if ((err = devctl (ctl_fd, DCMD_PROC_STATUS, &status, sizeof_status,
              0)) != EOK)
              return TARGET_XFER_E_IO;
          if ((offset + mylen) > sizeof (siginfo))
            {
              if (offset < sizeof_siginfo)
                  mylen = sizeof (siginfo) - offset;
              else
                  return TARGET_XFER_EOF;
            }
          nto_get_siginfo_from_procfs_status (&status, &siginfo);
          memcpy (readbuf, (gdb_byte *)&siginfo + offset, mylen);
          *xfered_len = len;
          return len? TARGET_XFER_OK : TARGET_XFER_EOF;
        }
      /* fallthrough */
      /* no break */
    case TARGET_OBJECT_AUXV:
      if (readbuf != NULL)
        {
          int err;
          CORE_ADDR initial_stack;
          debug_process_t procinfo;
          /* For 32-bit architecture, size of auxv_t is 8 bytes.  */
          const unsigned int sizeof_auxv_t = sizeof (auxv_t);
          const unsigned int sizeof_tempbuf = 20 * sizeof_auxv_t;
          int tempread;
          gdb_byte *const tempbuf = (gdb_byte *) alloca (sizeof_tempbuf);

          if (tempbuf == NULL)
              return TARGET_XFER_E_IO;

          err = devctl (ctl_fd, DCMD_PROC_INFO, &procinfo,
              sizeof procinfo, 0);
          if (err != EOK)
              return TARGET_XFER_E_IO;

          initial_stack = procinfo.initial_stack;

          /* procfs is always 'self-hosted', no byte-order manipulation.  */
          tempread = nto_read_auxv_from_initial_stack (initial_stack, tempbuf,
                   sizeof_tempbuf,
                   sizeof (auxv_t));
          tempread = ((tempread<len)?tempread:len) - offset;
          memcpy (readbuf, tempbuf + offset, tempread);
          *xfered_len = tempread;

          return tempread ? TARGET_XFER_OK : TARGET_XFER_EOF;
        }
      /* fallthrough */
      /* no break */
    default:
      return this->beneath()->xfer_partial (object, annex, readbuf,
                  writebuf, offset, len, xfered_len);
    }
}

/* Take a program previously attached to and detaches it.
   The program resumes execution and will no longer stop
   on signals, etc.  We'd better not have left any breakpoints
   in the program or it'll die when it hits one.  */
void
nto_procfs::detach (inferior *inf, int from_tty)
{
  nto_trace(0) ("detach %d\n", inf->pid);
  target_announce_detach (from_tty);

  close1 (ctl_fd);
  ctl_fd = -1;

  detach_inferior(inf);
  mourn_inferior ();
  maybe_unpush_target();
}

static int
procfs_breakpoint (CORE_ADDR addr, int type, int size)
{
  procfs_break brk;

  nto_trace(0) ("%sset breakpoint at %p\n", (size == -1)?"un":"  ", (void*)addr);

  brk.type = type;
  brk.addr = addr;
  brk.size = size;
  errno = devctl (ctl_fd, DCMD_PROC_BREAK, &brk, sizeof (brk), 0);
  if (errno != EOK)
      return 1;
  return 0;
}

int
nto_procfs::insert_breakpoint (struct gdbarch *gdbarch,
        struct bp_target_info *bp_tgt)
{
  bp_tgt->placed_address = bp_tgt->reqstd_address;
  return procfs_breakpoint (bp_tgt->placed_address, _DEBUG_BREAK_EXEC,
          nto_breakpoint_size (bp_tgt->placed_address));
}

int
nto_procfs::remove_breakpoint (struct gdbarch *gdbarch,
        struct bp_target_info *bp_tgt,
        enum remove_bp_reason reason)
{
  return procfs_breakpoint (bp_tgt->placed_address, _DEBUG_BREAK_EXEC, -1);
}

int
nto_procfs::insert_hw_breakpoint (struct gdbarch *gdbarch,
           struct bp_target_info *bp_tgt)
{
  bp_tgt->placed_address = bp_tgt->reqstd_address;
  return procfs_breakpoint (bp_tgt->placed_address,
          _DEBUG_BREAK_EXEC | _DEBUG_BREAK_HW, 0);
}

int
nto_procfs::remove_hw_breakpoint (struct gdbarch *gdbarch,
           struct bp_target_info *bp_tgt)
{
  return procfs_breakpoint (bp_tgt->placed_address,
          _DEBUG_BREAK_EXEC | _DEBUG_BREAK_HW, -1);
}

void
nto_procfs::resume ( ptid_t ptid, int step, enum gdb_signal signo)
{
  int signal_to_pass;
  procfs_status status;
  sigset_t *run_fault = (sigset_t *) (void *) &run.fault;
  nto_trace(0)("%s\n", __func__);
  if ( inferior_ptid == null_ptid )
      return;

  procfs_set_thread ((ptid == minus_one_ptid) ? inferior_ptid :
         ptid);

  run.flags = _DEBUG_RUN_FAULT | _DEBUG_RUN_TRACE;
  if (step)
      run.flags |= _DEBUG_RUN_STEP;

  sigemptyset (run_fault);
  sigaddset (run_fault, FLTBPT);
  sigaddset (run_fault, FLTTRACE);
  sigaddset (run_fault, FLTILL);
  sigaddset (run_fault, FLTPRIV);
  sigaddset (run_fault, FLTBOUNDS);
  sigaddset (run_fault, FLTIOVF);
  sigaddset (run_fault, FLTIZDIV);
  sigaddset (run_fault, FLTFPE);
  /* Peter V will be changing this at some point.  */
  /* todo has this been done yet? */
  sigaddset (run_fault, FLTPAGE);

  run.flags |= _DEBUG_RUN_ARM;

  signal_to_pass = gdb_signal_to_host (signo);

  if (signal_to_pass)
    {
      devctl (ctl_fd, DCMD_PROC_STATUS, &status, sizeof (status), 0);
      signal_to_pass = gdb_signal_to_host (signo);
      if (status.why & (_DEBUG_WHY_SIGNALLED | _DEBUG_WHY_FAULTED))
        {
          if (signal_to_pass != status.info.si_signo)
            {
              SignalKill ( ND_LOCAL_NODE, inferior_ptid.pid(), 0,
                  signal_to_pass, 0, 0);
              run.flags |= _DEBUG_RUN_CLRFLT | _DEBUG_RUN_CLRSIG;
            }
        }
    }
  else
      run.flags |= _DEBUG_RUN_CLRSIG | _DEBUG_RUN_CLRFLT;

  errno = devctl (ctl_fd, DCMD_PROC_RUN, &run, sizeof (run), 0);
  if (errno == EBUSY)
    {
      // there are still messages waiting, so keep the process stopped
      // and poll for the next status update.
      errno = EOK;
      nto_trace(0)("Can't run, there are still unhandled events!\n");
    }
  if (errno != EOK)
    {
      nto_trace(0)("Can't run, %i!\n", errno);
      perror (_("run error!\n"));
    }
}

void
nto_procfs::mourn_inferior ( )
{
  struct inferior *inf = current_inferior ();
  struct nto_inferior_data *inf_data = nto_inferior_data (inf);

  nto_trace(0)("%s\n", __func__);

  gdb_assert(inf_data != NULL);

  if (ctl_fd != -1)
    {
      /* detach from debug interface */
      close1 (ctl_fd);
      ctl_fd = -1;
    }

  generic_mourn_inferior ();
  inf_data->has_execution = 0;
  inf_data->has_stack = 0;
  inf_data->has_registers = 0;
  inf_data->has_memory = 0;

}

/* This function breaks up an argument string into an argument
   vector suitable for passing to execvp().
   E.g., on "run a b c d" this routine would get as input
   the string "a b c d", and as output it would fill in argv with
   the four arguments "a", "b", "c", "d".  The only additional
   functionality is simple quoting.  The gdb command:
    run a "b c d" e
   will fill in argv with the three args "a", "b c d", "e".  */
static void
breakup_args (char *scratch, const char **argv)
{
  char *pp, *cp = scratch;
  char quoting = 0;

  for (;;)
    {
      /* Scan past leading separators.  */
      quoting = 0;
      while (*cp == ' ' || *cp == '\t' || *cp == '\n')
          cp++;

      /* Break if at end of string.  */
      if (*cp == '\0')
          break;

      /* Take an arg.  */
      if (*cp == '"')
        {
           cp++;
           quoting = strchr (cp, '"') ? 1 : 0;
        }

      *argv++ = cp;

      /* Scan for next arg separator.  */
      pp = cp;
      if (quoting)
          cp = strchr (pp, '"');
      if ((cp == NULL) || (!quoting))
          cp = strchr (pp, ' ');
      if (cp == NULL)
          cp = strchr (pp, '\t');
      if (cp == NULL)
          cp = strchr (pp, '\n');

      /* No separators => end of string => break.  */
      if (cp == NULL)
        {
          pp = cp;
          break;
        }

      /* Replace the separator with a terminator.  */
      *cp++ = '\0';
    }

  /* Execv requires a null-terminated arg vector.  */
  *argv = NULL;
}

static unsigned nto_get_cpuflags (void)
{
  return SYSPAGE_ENTRY (cpuinfo)->flags;
}

/*
 * todo: by default GDB fork()'s the new process internally, while we are
 *       spawn()ing it on our own.
 *       Probably it makes sense to take the default approach instead of
 *       re-inventing the wheel. Otherwise this means that using GDB through
 *       QNet will no longer work as expected.
 *
 * todo: QNet is dead!
 */
void
nto_procfs::create_inferior (const char *exec_file, const std::string &allargs,
    char **env, int from_tty)
{
  struct inheritance inherit;
  int pid;
  const char **argv;
  char *args;
  const char *in = "", *out = "", *err = "";
  int fd, fds[3];
//  const char *inferior_io_terminal = get_inferior_io_terminal ();

  nto_trace(0) ("create_inferior (%s, ...)\n", exec_file);

  /* todo: the whole argument vector handling is QUESTIONABLE! */
  args = xstrdup (allargs.c_str());

  argv = (const char **) xmalloc (((strlen (args) + 1U) / 2U + 2U) *
      sizeof (*argv));
  argv[0] = get_exec_file (1);
  if (!argv[0])
    {
      if (exec_file)
          argv[0] = xstrdup(exec_file);
      else
          return;
    }

  breakup_args (args, (exec_file != NULL) ? &argv[1] : &argv[0]);

  argv = nto_parse_redirection (argv, &in, &out, &err);

  fds[0] = STDIN_FILENO;
  fds[1] = STDOUT_FILENO;
  fds[2] = STDERR_FILENO;

  /* If the user specified I/O via gdb's --tty= arg, use it, but only
     if the i/o is not also being specified via redirection.  */
/*  if (inferior_io_terminal)
    {
      if (!in[0])
  in = inferior_io_terminal;
      if (!out[0])
  out = inferior_io_terminal;
      if (!err[0])
  err = inferior_io_terminal;
    }
*/
  if (in[0])
    {
      fd = open (in, O_RDONLY);
      if (fd == -1)
          perror (in);
      else
          fds[0] = fd;
    }
  if (out[0])
    {
      fd = open (out, O_WRONLY);
      if (fd == -1)
          perror (out);
      else
          fds[1] = fd;
    }
  if (err[0])
    {
      fd = open (err, O_WRONLY);
      if (fd == -1)
          perror (err);
      else
          fds[2] = fd;
    }

  memset (&inherit, 0, sizeof (inherit));

  inherit.flags |= SPAWN_SETGROUP | SPAWN_HOLD;
  inherit.pgroup = SPAWN_NEWPGROUP;

  pid = spawnp (argv[0], 3, fds, &inherit, (char * const *)argv, env );
  xfree (args);

  if (pid == -1)
      error (_("Error spawning %s: %d (%s)"), argv[0], errno,
          safe_strerror (errno));

  if (fds[0] != STDIN_FILENO)
      close1 (fds[0]);
  if (fds[1] != STDOUT_FILENO)
      close1 (fds[1]);
  if (fds[2] != STDERR_FILENO)
      close1 (fds[2]);

  inferior_ptid = do_attach (ptid_t (pid, 1, 0));
  nto_trace(0) ("attached to %s\n", nto_pid_to_str(inferior_ptid).c_str());

  inferior *inf = nto_add_inferior(this, pid);
  inf->attach_flag = false;
  inf->removable = true;

  /* add and switch to the current thread */
  thread_info *tp = add_thread (this, inferior_ptid);
  switch_to_thread_no_regs (tp);

  update_thread_list ( );

  terminal_init ();
}

/**
 * added trace to check if each of these is really necessary
 */
void
nto_procfs::stop (ptid_t ptid)
{
  nto_trace(0)("stop()\n");
  devctl (ctl_fd, DCMD_PROC_STOP, NULL, 0, 0);
}

void
nto_procfs::interrupt ()
{
  nto_trace(0)("interrupt()\n");
  devctl (ctl_fd, DCMD_PROC_STOP, NULL, 0, 0);
}

void
nto_procfs::pass_ctrlc ()
{
  nto_trace(0)("pass_ctrlc()\n");
  devctl (ctl_fd, DCMD_PROC_STOP, NULL, 0, 0);
}

/**
 * terminates the current inferior and cleans up the data structures
 */
void
nto_procfs::kill ( )
{
  nto_trace (0) ("%s ()\n", __func__);

  kill_1( current_inferior()->pid, SIGKILL);
  mourn_inferior ( );
}

/* Fill buf with regset and return devctl cmd to do the setting.  Return
   -1 if we fail to get the regset.  Store size of regset in bufsize.  */
static int
get_regset (int regset, char *buf, int *bufsize)
{
  int dev_get, dev_set;
  switch (regset)
    {
    case NTO_REG_GENERAL:
      dev_get = DCMD_PROC_GETGREG;
      dev_set = DCMD_PROC_SETGREG;
      break;

    case NTO_REG_FLOAT:
      dev_get = DCMD_PROC_GETFPREG;
      dev_set = DCMD_PROC_SETFPREG;
      break;

    case NTO_REG_ALT:
      dev_get = DCMD_PROC_GETALTREG;
      dev_set = DCMD_PROC_SETALTREG;
      break;

    case NTO_REG_SYSTEM:
    default:
      return -1;
    }
  if (devctl (ctl_fd, dev_get, buf, *bufsize, bufsize) != EOK)
      return -1;

  return dev_set;
}

void
nto_procfs::store_registers (struct regcache *regcache, int regno)
{
  union
  {
    procfs_greg greg;
    procfs_fpreg fpreg;
    procfs_altreg altreg;
  } reg;
  int regsize, err, dev_set, regset;

  if ( inferior_ptid == null_ptid )
      return;
  procfs_set_thread (inferior_ptid);

  for (regset = NTO_REG_GENERAL; regset < NTO_REG_END; regset++)
    {
      regsize = sizeof (reg);
      dev_set = get_regset (regset, (char *) &reg, &regsize);
      if (dev_set == -1)
          continue;

      if (nto_regset_fill (regcache, regset, (gdb_byte *) &reg, regsize) == -1)
          continue;

      err = devctl (ctl_fd, dev_set, &reg, sizeof (reg), 0);
      if (err != EOK)
          fprintf_unfiltered (gdb_stderr,
              "Warning unable to write regset %d: %s\n",
              regno, safe_strerror (err));
    }
}

/* Set list of signals to be handled in the target.  */

void
nto_procfs::pass_signals (gdb::array_view<const unsigned char> pass_signals)
{
  int signo;

  sigfillset (&run.trace);

  for (signo = 1; signo < NSIG; signo++)
    {
      int target_signo = gdb_signal_from_host (signo);
      if (target_signo < pass_signals.size () && pass_signals[target_signo])
        sigdelset (&run.trace, signo);
    }
}

const struct target_desc *
nto_procfs::read_description ( )
{
  if (ntoops_read_description)
      return ntoops_read_description (nto_get_cpuflags ());
  else
    {
      warning("Target description unavailable!");
      return NULL;
    }
}

static int
procfs_hw_watchpoint (CORE_ADDR addr, int len, enum target_hw_bp_type type)
{
  procfs_break brk;
  nto_trace(0) ("%sset watchpoint at %p\n", (len == -1)?"un":"  ", (void*)addr);

  switch (type)
    {
    case hw_read:
#if defined(__x86_64__)
      // No BREAK_RD support, fall back to BREAK_RW
      warning("No support for read only watchpoints on ntox86_64!");
      brk.type = _DEBUG_BREAK_RW;
#else
      brk.type = _DEBUG_BREAK_RD;
#endif
      break;
    case hw_access:
      brk.type = _DEBUG_BREAK_RW;
      break;
    default:      /* Modify.  */
      brk.type = _DEBUG_BREAK_MODIFY;
    }
  brk.type |= _DEBUG_BREAK_HW;  /* Always ask for HW.  */
  brk.addr = addr;
  brk.size = len;

  errno = devctl (ctl_fd, DCMD_PROC_BREAK, &brk, sizeof (brk), 0);
  if (errno != EOK)
    {
      nto_trace(0)("addr: %p, errno: %i, ctl_fd: %i\n", (void*)addr, errno, ctl_fd);
      if (len == -1)
        perror (_("Failed to remove hardware watchpoint"));
      else
        perror (_("Failed to set hardware watchpoint"));
      return -1;
    }
  return 0;
}

int
nto_procfs::can_use_hw_breakpoint (enum bptype type,
            int cnt, int othertype)
{
  nto_trace(0)("can_use_hw_breakpoint(%i, %i, %i)\n", (int)type, cnt, othertype);
  return 1;
}

int
nto_procfs::remove_watchpoint (CORE_ADDR addr, int len,
           enum target_hw_bp_type type,
           struct expression *cond)
{
  return procfs_hw_watchpoint (addr, -1, type);
}

int
nto_procfs::insert_watchpoint (CORE_ADDR addr, int len,
           enum target_hw_bp_type type,
           struct expression *cond)
{
  return procfs_hw_watchpoint (addr, len, type);
}

/* Create the "native" and "procfs" targets.  */

#define OSTYPE_NTO 1

void _initialize_nto_procfs (void);
void
_initialize_nto_procfs ()
{
  int rv;

  nto_trace (0) ("%s ()\n", __func__);

  if (getuid() != 0)
    warning("GDB is not running as root!\nStarting/attaching to processes will not work!");
  else
    {
      /* we need all the abilities we can get */
      rv = procmgr_ability(0, PROCMGR_ADN_ROOT|PROCMGR_AOP_ALLOW|PROCMGR_AOP_INHERIT_YES|PROCMGR_AID_EOL);
      if (rv != EOK)
          warning("Could not set procmgr abilities for root! (%s)", strerror(rv));
    }
  /* Initially, make sure all signals are reported.  */
  sigfillset (&run.trace);

  add_info ("pidlist", procfs_pidlist, _("pidlist"));
  add_info ("meminfo", procfs_meminfo, _("memory information"));

  add_inf_child_target (&nto_procfs_target);
}
