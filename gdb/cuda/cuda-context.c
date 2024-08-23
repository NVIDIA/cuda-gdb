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
#include "defs.h"

#include "cuda-context.h"
#include "cuda-options.h"
#include "cuda-tdep.h"
#include "cuda-state.h"
#include "cuda-modules.h"


/******************************************************************************
 *
 *                                  Context
 *
 *****************************************************************************/

context_t
context_new (uint64_t context_id, uint32_t dev_id)
{
  context_t context;

  cuda_trace ("create context dev_id %u context_id 0x%llx",
              dev_id, (unsigned long long)context_id);

  context = (context_t) xmalloc (sizeof *context);
  context->context_id = context_id;
  context->dev_id     = dev_id;
  context->modules    = modules_new ();

  return context;
}

void
context_delete (context_t ctx)
{
  gdb_assert (ctx);
  gdb_assert (get_current_context () != ctx);

  cuda_trace ("delete context dev_id %u context_id 0x%llx",
              ctx->dev_id, (unsigned long long)ctx->context_id);

  modules_delete (context_get_modules (ctx));
  xfree (ctx);
}

uint64_t
context_get_id (context_t ctx)
{
  gdb_assert (ctx);

  return ctx->context_id;
}

uint32_t
context_get_device_id (context_t ctx)
{
  gdb_assert (ctx);

  return ctx->dev_id;
}

modules_t
context_get_modules (context_t ctx)
{
  gdb_assert (ctx);

  return ctx->modules;
}

void
context_print (context_t ctx)
{
  gdb_assert (ctx);

  cuda_trace ("context %p context_id 0x%llx dev_id %u modules %p",
              ctx, (unsigned long long)ctx->context_id,
              ctx->dev_id, ctx->modules);

  modules_print (ctx->modules);
}

static context_t
context_remove_context_from_list (context_t context, list_elt_t *list_head)
{
  list_elt_t elt, prev_elt;
  context_t removed_context;

  /* Find the list element associated with the given context. Use the head of
     the list if not specified. */
  if (context)
    for (elt = *list_head; elt && elt->context != context; elt = elt->next);
  else
    elt = *list_head;

  /* Nothing to do if not found or list is empty. */
  if (!elt)
    return NULL;

  removed_context = elt->context;

  /* Remove the element from the list. */
  if (elt == *list_head)
    *list_head = elt->next;
  else
  {
    for (prev_elt = *list_head; prev_elt && prev_elt->next != elt; prev_elt = prev_elt->next);
    prev_elt->next = elt->next;
  }

  /* Free the element */
  xfree (elt);

  /* Return the context associated with the removed element */
  return removed_context;
}

/******************************************************************************
 *
 *                                  Contexts
 *
 *****************************************************************************/

contexts_t
contexts_new (void)
{
  contexts_t contexts;

  contexts = (contexts_t) xcalloc (1, sizeof *contexts);

  return contexts;
}

void
contexts_delete (contexts_t ctx)
{
  context_t  context;
  uint32_t   ctxtid;
  uint32_t   tid;

  gdb_assert (ctx);

  for (ctxtid = 0; ctxtid < ctx->num_ctxtids; ++ctxtid)
    while (ctx->stacks[ctxtid])
      {
        tid = ctx->ctxtid_to_tid[ctxtid];
        contexts_unstack_context (ctx, tid);
      }

  while (ctx->list)
    {
      context = context_remove_context_from_list (NULL, &ctx->list);
      ctx->list_size--;
      context_delete (context);
    }

  xfree (ctx->stacks);
  xfree (ctx->ctxtid_to_tid);
}

void
contexts_print (contexts_t ctx)
{
  list_elt_t     elt;
  context_t      context;

  gdb_assert (ctx);

  cuda_trace (" Contexts: ");

  for (elt = ctx->list; elt ; elt = ctx->list->next)
    {
      context = elt->context;
      context_print (context);
    }
}

void
contexts_add_context (contexts_t ctx, context_t context)
{
  list_elt_t elt;

  gdb_assert (ctx);

  /* Create a new list element to contain the context. */
  elt = (list_elt_t) xmalloc (sizeof *elt);
  elt->context = context;
  elt->next    = ctx->list;

  /* Add the element at the head of the list */
  ctx->list = elt;
  ctx->list_size++;
}

static list_elt_t * 
contexts_find_stack_by_tid (contexts_t ctx, uint32_t tid)
{
  uint32_t ctxtid;
  bool     found = false;

  for (ctxtid = 0; ctxtid < ctx->num_ctxtids; ++ctxtid)
    if (tid == ctx->ctxtid_to_tid[ctxtid])
      {
        found = true;
        break;
      }
  
  if (found)
    return &ctx->stacks[ctxtid];
  else
    return NULL;
}

context_t
contexts_remove_context (contexts_t ctx, context_t context)
{
  uint32_t ctxtid;
  context_t removed_context = context;

  gdb_assert (ctx);

  /* Remove context from all device stacks */
  for (ctxtid = 0; ctxtid < ctx->num_ctxtids; ctxtid++)
    {
      /* The context may have been pushed multiple times */
      while (removed_context)
      {
        removed_context = context_remove_context_from_list (context, &ctx->stacks[ctxtid]);
      }
    }

  /* Remove context from the list of all device contexts */
  context_remove_context_from_list (context, &ctx->list);
  ctx->list_size--;

  return context;
}

static void 
contexts_add_stack_for_tid (contexts_t ctx, uint32_t tid)
{
  uint32_t ctxtid;

  ctxtid = ctx->num_ctxtids++;

  ctx->stacks = (list_elt_t *) xrealloc (ctx->stacks,
					 ctx->num_ctxtids * sizeof (*ctx->stacks));
  ctx->ctxtid_to_tid = (uint32_t *) xrealloc (ctx->ctxtid_to_tid,
					      ctx->num_ctxtids * sizeof (*ctx->ctxtid_to_tid));

  ctx->stacks[ctxtid] = NULL; 
  ctx->ctxtid_to_tid[ctxtid] = tid;
}

void
contexts_stack_context (contexts_t ctx, context_t context, uint32_t tid)
{
  uint32_t     ctxtid;
  list_elt_t   elt;
  list_elt_t  *stack;

  gdb_assert (ctx);

  /* Make sure ctx context exists on the device */
  gdb_assert (contexts_find_context_by_id (ctx, context_get_id (context)));

  /* Get the stack to add ctx context to */
  stack = contexts_find_stack_by_tid (ctx, tid);

  if (NULL == stack)
    {
      /* Add a new stack */
      contexts_add_stack_for_tid (ctx, tid); 
      ctxtid = ctx->num_ctxtids - 1;
      stack = &ctx->stacks[ctxtid];
    }

  /* Insert the element at the top of the stack list */
  elt = (list_elt_t) xmalloc (sizeof *elt);
  elt->context = context;
  elt->next = *stack;
  *stack = elt;
}

/* Pop the topmost context from the stack for this tid */
context_t
contexts_unstack_context (contexts_t ctx, uint32_t tid)
{
  list_elt_t  elt;
  context_t   context;
  list_elt_t *stack;

  gdb_assert (ctx);

  /* Get the stack for this tid */
  stack = contexts_find_stack_by_tid (ctx, tid);

  if (NULL == stack)
      return NULL;

  /* Remove the context from the top of the stack. */
  elt = *stack;

  gdb_assert (elt);

  *stack = elt->next;

  context = elt->context;

  xfree (elt);

  return context;
}

context_t
contexts_find_context_by_id  (contexts_t ctx, uint64_t context_id)
{
  list_elt_t elt;

  gdb_assert (ctx);

  /* Look for the context in the context list */
  for (elt = ctx->list; elt; elt = elt->next)
    if (context_get_id (elt->context) == context_id)
      return elt->context;

  /* Return NULL if not found */
  return NULL;
}

context_t
contexts_get_active_context (contexts_t ctx, uint32_t tid)
{
  list_elt_t *stack;

  gdb_assert (ctx);

  /* Get the stack for this tid */
  stack = contexts_find_stack_by_tid (ctx, tid);

  if (NULL == stack || NULL == *stack)
    return NULL;

  return (*stack)->context;
}

/* expensive: traverse all the objfiles to see if addr corresponds to
   any of them. Return the context if found, 0 otherwise. */
context_t
contexts_find_context_by_address (contexts_t ctx, CORE_ADDR addr)
{
  list_elt_t elt;

  gdb_assert (ctx);

  for (elt = ctx->list; elt; elt = elt->next)
    if (modules_find_module_by_address (elt->context->modules, addr))
      return elt->context;

  return NULL;
}

bool
contexts_is_any_context_present (contexts_t ctx)
{
  gdb_assert (ctx);

  return (NULL != ctx->list);
}

bool
contexts_is_active_context (contexts_t ctx, context_t context)
{
  uint32_t ctxtid;
  gdb_assert (ctx);

  for (ctxtid = 0; ctxtid < ctx->num_ctxtids; ++ctxtid)
    if (ctx->stacks[ctxtid] && context == ctx->stacks[ctxtid]->context)
      return true;

  return false;
}

uint32_t
contexts_get_list_size (contexts_t ctx)
{
  return ctx->list_size;
}

/******************************************************************************
 *
 *                             Current Context
 *
 *****************************************************************************/

context_t current_context;

context_t
get_current_context (void)
{
  return current_context;
}

void
set_current_context (context_t context)
{
  current_context = context;
}
