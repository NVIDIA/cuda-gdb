/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2025 NVIDIA Corporation
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

/* Utility functions for cuda-gdb */
#ifdef GDBSERVER
#include "gdbsupport/gdb_locale.h"
#include "server.h"
#else
#include "defs.h"

#include "cuda-options.h"
#include "exceptions.h"
#include "gdb/signals.h"
#include "gdbthread.h"
#include "inferior.h"
#include "objfiles.h"
#include "stack.h"
#include "utils.h"
#endif

#include <ctype.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef __QNXHOST__
#include <ftw.h>
#else
#include <dirent.h>
#endif
#include <string.h>
#include <unistd.h>

#include "cuda-defs.h"
#ifndef GDBSERVER
#include "cuda-api.h"
#include "cuda-options.h"
#endif
#include "cuda-utils.h"
#include <fcntl.h>

static char cuda_gdb_tmp_basedir[CUDA_GDB_TMP_BUF_SIZE] = { 0 };
static char *cuda_gdb_tmp_dir = NULL;

int
cuda_gdb_dir_create (const char *dir_name, uint32_t permissions,
		     bool override_umask, bool *dir_exists)
{
  int ret;
  mode_t old_umask = 0;

  /* Save the old umask and reset it */
  if (override_umask)
    old_umask = umask (0);

  ret = mkdir (dir_name, permissions);
  if ((ret < 0) && (errno == EEXIST))
    {
      /* Preexisting directory (may have lost a race to create it),
	 report dir_exists and success */
      *dir_exists = true;
      ret = 0;
    }
  else
    *dir_exists = false;

  /* Restore the old umask */
  if (override_umask)
    umask (old_umask);

  return ret;
}

static void
cuda_gdb_tmpdir_create_basedir (void)
{
  int ret = 0;
  bool dir_exists = false;
  bool override_umask = true;

  if (getenv ("TMPDIR"))
    snprintf (cuda_gdb_tmp_basedir, sizeof (cuda_gdb_tmp_basedir),
	      "%s/cuda-dbg", getenv ("TMPDIR"));
  else
    snprintf (cuda_gdb_tmp_basedir, sizeof (cuda_gdb_tmp_basedir),
	      "/tmp/cuda-dbg");

  ret = cuda_gdb_dir_create (cuda_gdb_tmp_basedir, S_IRWXU | S_IRWXG | S_IRWXO,
			     override_umask, &dir_exists);
  if (ret)
    error (_ ("Error creating temporary directory %s\n"),
	   cuda_gdb_tmp_basedir);
}

static char *
cuda_gdb_get_tmp_basedir (void)
{
  if (cuda_gdb_tmp_basedir[0] == 0)
    cuda_gdb_tmpdir_create_basedir ();

  return cuda_gdb_tmp_basedir;
}

#ifdef __QNXHOST__
static int
fn_cleanup_dir (const char *path, const struct stat *stat, int flag,
		struct FTW *ftw)
{
  switch (flag)
    {
    case FTW_DP:
      rmdir (path);
      break;
    case FTW_F:
    case FTW_SL:
    case FTW_SLN:
      unlink (path);
      break;
    case FTW_DNR:
    case FTW_NS:
      /* Silently ignore read errors */
      break;
    }
  return 0;
}

void
cuda_gdb_tmpdir_cleanup_dir (const char *dirpath)
{
  const int maxDepth = 10;
  nftw (dirpath, fn_cleanup_dir, maxDepth, FTW_DEPTH);
}

#else  /* __QNXHOST__*/

void
cuda_gdb_dir_cleanup_files (const char *dirpath)
{
  char path[CUDA_GDB_TMP_BUF_SIZE];
  DIR *dir = opendir (dirpath);
  struct dirent *dir_ent = NULL;

  if (!dir)
    return;

  while ((dir_ent = readdir (dir)))
    {
      if (!strcmp (dir_ent->d_name, ".") || !strcmp (dir_ent->d_name, ".."))
	continue;
      snprintf (path, sizeof (path), "%s/%s", dirpath, dir_ent->d_name);
      if (dir_ent->d_type == DT_DIR)
	{
	  cuda_gdb_dir_cleanup_files (path);
	  rmdir (path);
	}
      else
	unlink (path);
    };

  closedir (dir);
}

static void
cuda_gdb_tmpdir_cleanup_dir (const char *dirpath)
{
  cuda_gdb_dir_cleanup_files (dirpath);
  rmdir (dirpath);
}
#endif /* __QNXHOST__*/

void
cuda_gdb_tmpdir_cleanup_self (void *unused)
{
  if (!cuda_gdb_tmp_dir)
    return;

  cuda_gdb_tmpdir_cleanup_dir (cuda_gdb_tmp_dir);
  xfree (cuda_gdb_tmp_dir);
  cuda_gdb_tmp_dir = NULL;
}

static void
cuda_gdb_tmpdir_setup (void)
{
  /* This needs to fit cuda_gdb_get_tmp_basedir + getpid output */
  char dirpath[2 * CUDA_GDB_TMP_BUF_SIZE];
  int ret;
  bool dir_exists = false;
  bool override_umask = false;

  snprintf (dirpath, sizeof (dirpath), "%s/%u", cuda_gdb_get_tmp_basedir (),
	    getpid ());

#ifdef __QNXHOST__
  cuda_gdb_tmpdir_cleanup_dir (
      dirpath); /* Try to remove the dir if it exists */
#endif		/* __QNXHOST__ */
  ret = cuda_gdb_dir_create (dirpath, S_IRWXU | S_IRWXG | S_IXOTH,
			     override_umask, &dir_exists);
  if (ret)
    error (_ ("Error creating temporary directory %s\n"), dirpath);

#ifndef __QNXHOST__
  if (dir_exists)
    cuda_gdb_dir_cleanup_files (dirpath);
#endif /* __QNXHOST__ */

  size_t buflen = strlen (dirpath) + 1;
  cuda_gdb_tmp_dir = (char *)xmalloc (buflen);
  strncpy (cuda_gdb_tmp_dir, dirpath, buflen);

  /* No final cleanup chain at server side,
   * cleanup function is called explicitly when server quits */
#ifndef GDBSERVER
  make_final_cleanup (cuda_gdb_tmpdir_cleanup_self, NULL);
#endif
}

const char *
cuda_gdb_tmpdir_getdir (void)
{
  return cuda_gdb_tmp_dir;
}

// Reserve 0 as a starting point for timestamp comparisons
static cuda_clock_t cuda_clock_ = 1;

cuda_clock_t
cuda_clock (void)
{
  return cuda_clock_;
}

void
cuda_clock_increment (void)
{
  ++cuda_clock_;
  if (cuda_clock_ == 0)
    warning (_ ("The internal clock counter used for cuda debugging wrapped "
		"around.\n"));
}

#ifndef GDBSERVER
static unsigned char *
cuda_nat_save_gdb_signal_handlers (void)
{
  unsigned char *sigs;
  int i, j;
  static int (*sighand_savers[]) (int)
      = { signal_stop_state, signal_print_state, signal_pass_state };

  sigs = (unsigned char *)xmalloc (GDB_SIGNAL_LAST
				   * ARRAY_SIZE (sighand_savers));

  for (i = 0; i < ARRAY_SIZE (sighand_savers); i++)
    for (j = 0; j < GDB_SIGNAL_LAST; j++)
      sigs[i * GDB_SIGNAL_LAST + j] = sighand_savers[i](j);

  return sigs;
}

static void
cuda_nat_restore_gdb_signal_handlers (unsigned char *sigs)
{
  int i, j;
  static int (*sighand_updaters[]) (int, int)
      = { signal_stop_update, signal_print_update, signal_pass_update };

  for (i = 0; i < ARRAY_SIZE (sighand_updaters); i++)
    for (j = 0; j < GDB_SIGNAL_LAST; j++)
      sighand_updaters[i](j, sigs[i * GDB_SIGNAL_LAST + j]);
}

void
cuda_nat_bypass_signals_cleanup (unsigned char *sigs)
{
  cuda_nat_restore_gdb_signal_handlers (sigs);
  xfree (sigs);
}

unsigned char *
cuda_gdb_bypass_signals (void)
{
  unsigned char *sigs;
  constexpr unsigned cuda_stop_signal = GDB_SIGNAL_URG;
  int i;

  sigs = cuda_nat_save_gdb_signal_handlers ();
  for (i = 0; i < GDB_SIGNAL_LAST; i++)
    {
      if (i == cuda_stop_signal || i == GDB_SIGNAL_TRAP || i == GDB_SIGNAL_KILL
	  || i == GDB_SIGNAL_STOP || i == GDB_SIGNAL_CHLD
	  || i >= GDB_SIGNAL_CUDA_UNKNOWN_EXCEPTION)
	continue;
      signal_stop_update (i, 0);
      signal_pass_update (i, 1);
      signal_print_update (i, 1);
    }

  return sigs;
}

/* CUDA PTX registers cache */
struct cuda_ptx_cache_element
{
  struct frame_id frame_id;
  cuda_coords coords;
  int dwarf_regnum;
  char data[16];
  int len;
};
typedef struct cuda_ptx_cache_element cuda_ptx_cache_element_t;

static std::vector<cuda_ptx_cache_element_t> cuda_ptx_register_cache;

/* Searches for cache entry containing given dwarf register for a given frame
 */
static std::vector<cuda_ptx_cache_element_t>::iterator
cuda_ptx_cache_find_element (struct frame_id frame_id, int dwarf_regnum)
{

  if (!cuda_current_focus::isDevice ())
    return cuda_ptx_register_cache.end ();

  const auto &coords = cuda_current_focus::get ();

  for (auto it = cuda_ptx_register_cache.begin ();
       it != cuda_ptx_register_cache.end (); ++it)
    if ((coords == it->coords) && (frame_id == it->frame_id)
	&& (dwarf_regnum == it->dwarf_regnum))
      return it;

  return cuda_ptx_register_cache.end ();
}

/**
 *  Adds PTX register to a cache.
 *  PTX register cache is considered valid only for given lane within given
 * frame
 */
void
cuda_ptx_cache_store_register (frame_info_ptr frame, int dwarf_regnum,
			       struct value *value)
{
  struct cuda_ptx_cache_element new_elem;

  /* If focus is not on device - return */
  if (!cuda_current_focus::isDevice ())
    return;

  /* Store the current focus */
  new_elem.coords = cuda_current_focus::get ();

  /* Store information about cached register in temporary cache element*/
  new_elem.frame_id = get_frame_id (frame);
  new_elem.dwarf_regnum = dwarf_regnum;
  new_elem.len = value->type ()->length ();
  /* If element can not be cached - return */
  if (new_elem.len > sizeof (new_elem.data))
    return;
  memcpy (new_elem.data, value->contents_raw ().data (), new_elem.len);

  auto elem = cuda_ptx_cache_find_element (new_elem.frame_id, dwarf_regnum);
  if (elem != cuda_ptx_register_cache.end ())
    {
      elem->len = new_elem.len;
      memcpy (elem->data, new_elem.data, elem->len);
      return;
    }

  /* Add new element to the cache */
  cuda_ptx_register_cache.emplace_back (std::move (new_elem));
}

/**
 * Retrieves previously cached value from PTX register cache.
 * \return either a previously cached value
 *         or struct value with optimized_out flag set
 */

struct value *
cuda_ptx_cache_get_register (frame_info_ptr frame, int dwarf_regnum,
			     struct type *type)
{
  struct value *retval;

  retval = value::allocate (type);

  auto elem = cuda_ptx_cache_find_element (get_frame_id (frame), dwarf_regnum);
  if ((elem == cuda_ptx_register_cache.end ())
      || (elem->len != type->length ())
      || !cuda_options_variable_value_cache_enabled ())
    {
      retval->set_lval (not_lval);
      retval->mark_bytes_optimized_out (0, type->length ());
      return retval;
    }

  retval->mark_bytes_optimized_out (0, 0);
  retval->set_cached (true);
  memcpy (retval->contents_raw ().data (), elem->data, elem->len);
  return retval;
}

struct cuda_ptx_cache_update_local_vars_data
{
  frame_info_ptr m_frame;

  void
  operator() (const char *print_name, struct symbol *sym)
  {
    cuda_ptx_cache_local_vars_iterator (print_name, sym, m_frame);
  }
};

static void
cuda_ptx_cache_update_local_vars (void)
{
  struct cuda_ptx_cache_update_local_vars_data cb_data;
  const struct block *block = NULL;

  if (!cuda_current_focus::isDevice ()
      || current_inferior ()
	     ->process_target ()
	     ->find_thread (inferior_ptid)
	     ->executing ())
    return;
  cb_data.m_frame = get_current_frame ();
  if (!cb_data.m_frame)
    return;

  block = get_frame_block (cb_data.m_frame, 0);
  if (!block)
    return;

  iterate_over_block_local_vars (block, cb_data);
}

/**
 * Refresh cuda ptx register cache
 * If cache is not empty but the focus was changed - clean up the cache
 * Then try to cache all local variables mapped to PTX/GPU registers.
 */
void
cuda_ptx_cache_refresh (void)
{
  if (!cuda_ptx_register_cache.empty () && cuda_current_focus::isDevice ())
    {
      auto elem = cuda_ptx_register_cache.begin ();
      const auto &coords = cuda_current_focus::get ();
      /* If focus is still on the same lane - keep the cache intact */
      if (coords != elem->coords
	  || !cuda_options_variable_value_cache_enabled ())
	{
	  cuda_ptx_register_cache.clear ();
	}
    }

  if (!cuda_options_variable_value_cache_enabled ())
    return;
  cuda_ptx_cache_update_local_vars ();
}

bool
cuda_managed_msymbol_p (struct bound_minimal_symbol &bmsym)
{
  struct obj_section *section = bmsym.obj_section ();
  struct objfile *obj = section ? section->objfile : NULL;
  struct gdbarch *arch = obj ? obj->arch () : NULL;

  if (arch == NULL)
    return false;

  return gdbarch_bfd_arch_info (arch)->arch == bfd_arch_m68k
	     ?
	     /* If this a device symbol MSYMBOL_TARGET_FLAG indicates if it is
		managed */
	     (bmsym.minsym->target_flag_1 () != 0)
	     :
	     /* Managed host symbols must be located in __nv_managed_data__
		section */
	     (section->the_bfd_section != NULL
	      && strcmp (section->the_bfd_section->name, "__nv_managed_data__")
		     == 0);
}

/* CUDA managed memory region list */
typedef struct
{
  CORE_ADDR begin;
  CORE_ADDR end;
} memory_region_t;

static std::vector<memory_region_t> cuda_managed_memory_regions;
bool cuda_managed_memory_regions_populated = false;

/* Must be called right after inferior was suspended */
void
cuda_managed_memory_clean_regions (void)
{
  cuda_managed_memory_regions_populated = false;
  cuda_managed_memory_regions.clear ();
}

void
cuda_managed_memory_add_region (CORE_ADDR begin, CORE_ADDR end)
{
  memory_region_t new_reg = { begin, end };

  cuda_managed_memory_regions.emplace_back (std::move (new_reg));
}

static void
cuda_managed_memory_populate_regions (void)
{
  CUDBGMemoryInfo regions[16];
  uint32_t regions_returned;
  uint64_t end;
  uint32_t cnt;
  uint64_t start_addr = 0;

  /* Check if information about managed memory regions has been queried already
   */
  if (cuda_managed_memory_regions_populated)
    return;

  do
    {
      cuda_debugapi::get_managed_memory_region_info (
	  start_addr, regions, ARRAY_SIZE (regions), &regions_returned);
      if (regions_returned == 0)
	return;
      /* Add fetched queries to the list and updated start address*/
      for (cnt = 0; cnt < regions_returned; cnt++)
	{
	  end = regions[cnt].startAddress + regions[cnt].size;
	  if (start_addr < end)
	    start_addr = end;
	  cuda_managed_memory_add_region (regions[cnt].startAddress, end);
	}
    }
  while (regions_returned == ARRAY_SIZE (regions));
  cuda_managed_memory_regions_populated = true;
}

bool
cuda_managed_address_p (CORE_ADDR addr)
{
  cuda_managed_memory_populate_regions ();

  for (auto &elem : cuda_managed_memory_regions)
    if (elem.begin <= addr && elem.end > addr)
      return true;
  return false;
}

bool
cuda_is_value_managed_pointer (struct value *value)
{
  bool result = false;
  struct type *type = value ? value->type () : NULL;

  /* Sanity checks */
  if (type == NULL || type->code () != TYPE_CODE_PTR)
    return result;

  try
    {
      result = cuda_managed_address_p (
	  unpack_pointer (type, value->contents_for_printing ().data ()));
    }
  catch (const gdb_exception_error &e)
    {
      if (e.reason != 0)
	return false;
    }

  return result;
}
#endif /* GDBSERVER */

static int
cuda_gdb_uid_from_pid (int pid)
{
  int uid = -1;
  FILE *procfile;
  char buffer[MAXPATHLEN], fname[MAXPATHLEN];

  /* Determine the uid by reading /proc/$pid/status */
  sprintf (fname, "/proc/%d/status", pid);
  procfile = fopen (fname, "r");
  if (procfile == NULL)
    return uid;

  while (fgets (buffer, MAXPATHLEN, procfile) != NULL)
    {
      if (strncmp (buffer, "Uid:\t", 5) != 0)
	continue;
      if (sscanf (buffer + 5, "%d", &uid) != 1)
	uid = -1;
      break;
    }
  fclose (procfile);

  return uid;
}

bool
cuda_gdb_chown_to_pid_uid (int pid, const char *path)
{
  int uid;

  if (pid <= 0)
    return true;

  uid = cuda_gdb_uid_from_pid (pid);
  if (uid == -1)
    return true;

  return chown (path, uid, -1) == 0;
}

static bool cuda_app_uses_uvm = false;

bool
cuda_is_uvm_used (void)
{
  return cuda_app_uses_uvm;
}

void
cuda_set_uvm_used (bool value)
{
  cuda_app_uses_uvm = value;
}

static bool cuda_app_uses_device_launch = false;

bool
cuda_is_device_launch_used (void)
{
  return cuda_app_uses_device_launch;
}

void
cuda_set_device_launch_used (bool value)
{
  cuda_app_uses_device_launch = value;
}

void _initialize_cuda_utils ();
void
_initialize_cuda_utils ()
{
  /* Create the base temporary directory */
  cuda_gdb_tmpdir_create_basedir ();

  /* Populate the temporary directory with a unique subdirectory for this
   * instance. */
  cuda_gdb_tmpdir_setup ();
}

#ifndef GDBSERVER
/* Drain the CUDA log queue from the backend and print all logs
 * according to the current log level setting. This should be called
 * whenever we need to consume logs from the backend.
 */
void
cuda_consume_and_print_driver_logs (void)
{
  CUDBGCudaLogMessage logMessages[cuda_debugapi::CudaLogMessagesFetchLimit];
  uint32_t num_consumed = 0;
  // Keep fetching logs until no more are available
  do
  {
    cuda_debugapi::consume_cuda_logs(
        logMessages, cuda_debugapi::CudaLogMessagesFetchLimit, &num_consumed);

    for (uint32_t i = 0; i < num_consumed; i++)
    {
      bool drop_log_record = false;
      const auto output_log_level = cuda_options_driver_log_level();
      const char *log_level_str;
      switch (logMessages[i].logLevel)
      {
      case CUDBG_CUDA_LOG_LEVEL_ERROR:
        log_level_str = "error";
        break;
      case CUDBG_CUDA_LOG_LEVEL_WARNING:
        log_level_str = "warning";
        if (output_log_level == cuda_log_level_t::CUDA_LOG_LEVEL_ERROR)
          drop_log_record = true;
        break;
      default:
        log_level_str = "unknown";
        if (output_log_level != cuda_log_level_t::CUDA_LOG_LEVEL_EXTRA)
          drop_log_record = true;
        break;
      }
      if (drop_log_record)
        continue;
      // Print the log message prefix
      printf_unfiltered(_("warning: Cuda Driver %s detected"), log_level_str);
      // print the log timestamp and thread id when driver_log_level is extra
      if (output_log_level == cuda_log_level_t::CUDA_LOG_LEVEL_EXTRA)
        printf_unfiltered(_(" (thread=0x%x timestamp_ns=%llu)"),
                          logMessages[i].osThreadId,
                          (unsigned long long)logMessages[i].unixTimestampNs);
      // print the log message
      printf_unfiltered(_(": %s\n"), logMessages[i].message);
    }
  } while (num_consumed > 0);
}
#endif /* GDBSERVER */
