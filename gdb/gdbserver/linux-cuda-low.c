#include <sys/types.h>
#include <stdio.h>

#include "server.h"
#include "target.h"
#include "cuda-tdep-server.h"
#include "../cuda-utils.h"
#include "../cuda-notifications.h"

extern struct cuda_sym cuda_symbol_list[];

/* Copy of the original host set of target operations. When a CUDA target op
   does not apply because we are dealing with the host code/cpu, use those
   routines instead. */
static struct target_ops host_target_ops;
bool cuda_debugging_enabled = false;
bool all_cuda_symbols_looked_up = false;
ptid_t last_ptid;
struct target_waitstatus last_ws;


/* Debugger API statistics collection is disabled on cuda-gdbserver side */
bool
cuda_options_statistics_collection_enabled (void)
{
  return false;
}

void
initialize_cuda_remote (void)
{
  /* Check the required CUDA debugger files are present */
  if (cuda_get_debugger_api ())
    {
      warning ("CUDA support disabled: could not obtain the CUDA debugger API\n");
      cuda_debugging_enabled = false;
      return;
    }

  /* Initialize the CUDA modules */
  cuda_utils_initialize ();
  cuda_notification_initialize ();

  cuda_debugging_enabled = true;
}

void
cuda_linux_mourn (struct process_info *process)
{
  cuda_cleanup ();
  host_target_ops.mourn (process);
}

ptid_t
cuda_linux_wait (ptid_t ptid, struct target_waitstatus *ws, int target_options)
{
  last_ptid = host_target_ops.wait (ptid, ws, target_options);
  last_ws = *ws;

  return last_ptid;
}

void
cuda_linux_look_up_symbols (void)
{
  int i;

  host_target_ops.look_up_symbols ();

  if (all_cuda_symbols_looked_up)
    return;

  all_cuda_symbols_looked_up = true;
  for (i = 0; i < cuda_get_symbol_cache_size (); i++)
    if (look_up_one_symbol (cuda_symbol_list[i].name, &(cuda_symbol_list[i].addr), 1) == 0)
      all_cuda_symbols_looked_up = false;
}

void
initialize_cuda_target_ops (struct target_ops *t)
{
  if (!cuda_debugging_enabled)
    return;

  /* Save the original set of target operations */
  host_target_ops = *t;

  /* Override what we need to */
  t->mourn                = cuda_linux_mourn;
  t->wait                 = cuda_linux_wait;
  t->look_up_symbols      = cuda_linux_look_up_symbols;
}
