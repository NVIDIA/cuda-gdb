/*
 * NVIDIA CUDA Debugger CUDA-GDB
 * Copyright (C) 2007-2023 NVIDIA Corporation
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

#ifndef _CUDA_CONTEXT_H
#define _CUDA_CONTEXT_H 1

#include "defs.h"
#include "cuda-defs.h"


/* Context */
context_t      context_new (uint64_t context_id, uint32_t dev_id);
void           context_delete (context_t);

uint64_t       context_get_id            (context_t);
uint32_t       context_get_device_id     (context_t);
modules_t      context_get_modules       (context_t);
void           context_print             (context_t);

module_t       context_find_module_by_id (context_t, uint64_t module_id);

/* Contexts */
contexts_t     contexts_new              (void);
void           contexts_delete           (contexts_t);
void           contexts_print            (contexts_t);

void           contexts_add_context             (contexts_t, context_t context);
context_t      contexts_remove_context          (contexts_t, context_t context);
void           contexts_stack_context           (contexts_t, context_t context, uint32_t tid);
context_t      contexts_unstack_context         (contexts_t, uint32_t tid);
context_t      contexts_get_active_context      (contexts_t, uint32_t tid);
bool           contexts_is_any_context_present  (contexts_t);
bool           contexts_is_active_context       (contexts_t, context_t context);

context_t      contexts_find_context_by_id      (contexts_t, uint64_t context_id);
context_t      contexts_find_context_by_address (contexts_t, CORE_ADDR addr);

uint32_t       contexts_get_list_size           (contexts_t);

/* Current Context */
context_t get_current_context     (void);
void      set_current_context     (context_t context);

/*--------------------------------------------------------------------------*/

/* DO NOT USE DIRECTLY the structs below. Those structs are provided to allow
   the use of structs instead of pointers for the iterator. Thus, no
   deallocation is required when done iterating. (Would be unnecessary in C++
   when using destructors). */

struct context_st {
  uint64_t    context_id;            /* the CUcontext handle */
  uint32_t    dev_id;                /* index of the parent device state */
  modules_t   modules;               /* list of modules in that context */
};

struct list_elt_st {
  context_t           context;      /* the context */
  struct list_elt_st *next;         /* pointer to the next element */
};

typedef struct list_elt_st     *list_elt_t;

struct contexts_st {
  uint32_t   *ctxtid_to_tid;
  uint32_t    num_ctxtids;
  uint32_t    list_size;            /* size of the context list */
  list_elt_t  list;                 /* list of all contexts on the device */
  list_elt_t *stacks;               /* context stacks for each host thread */
};

#endif

