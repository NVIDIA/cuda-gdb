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

%{

#include "defs.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "cuda/cuda-parser.h"

/*Input string */
struct cuda_parser_input_st {
  char     *buffer;  // copy of the string to parse
  uint32_t  size;    // size of the buffer in bytes
  char     *ptr;     // ptr to the next chunk to parse within the string
} cuda_parser_input;

/*Intermediate Buffers */
static request_value_t value_buffer;

/*Detect when the same coordinate type is specified more than one */
filter_type_t handled_filters;

/*Default value when a coordinate is omitted */
cuda_coords_special_value_t default_value;

/*Command type (to decide which parser to use) */
command_t start_token; /* read once, then set to CMD_NONE. See Bison manual 11.5. */

/*Results */
static cuda_parser_result_t  result;

static void handle_command (command_t cmd);
static void handle_query (filter_type_t type);
static void handle_condition (filter_type_t type, compare_t cmp);
static void handle_switch (filter_type_t type);
static void handle_filter (filter_type_t type);
static void handle_scalar (uint32_t index, uint32_t value);

int  cuda_parser_lex (void);
int  cuda_parser_get_next_input (char *buf, int max_size);
void cuda_parser_error (const char *msg);
void cuda_parser_reset_lexer (void);

%}

%union
  {
    uint32_t  value;
    compare_t cmp;
  }

%token START_QUERY_OR_SWITCH START_CONDITIONS START_FILTER START_FILTER_KERNEL
%token DEVICE SM WARP LANE
%token KERNEL GRID BLOCK THREAD
%token BLOCKIDX_X BLOCKIDX_Y BLOCKIDX_Z
%token THREADIDX_X THREADIDX_Y THREADIDX_Z
%token CURRENT WILDCARD
%token <value> VALUE
%token <cmp> CMP
%token COMMA
%token OPENPAR CLOSEPAR
%token LOGICAL_AND LOGICAL_OR
%token PARSING_ERROR

%%

command : START_QUERY_OR_SWITCH queries              { handle_command (CMD_QUERY); }
        | START_CONDITIONS      and_conditions       { handle_command (CMD_COND_AND); }
        | START_CONDITIONS      or_conditions        { handle_command (CMD_COND_OR); }
        | START_QUERY_OR_SWITCH switches             { handle_command (CMD_SWITCH); }
        | START_FILTER          filters              { handle_command (CMD_FILTER); }
        | START_FILTER_KERNEL   kernel               { handle_command (CMD_FILTER_KERNEL); }
        ;

filters : filter
        | filters filter
        ;

filter : device
       | sm
       | warp
       | lane 
       | kernel
       | grid
       | block
       | thread
       ;

device     : DEVICE scalar           { handle_filter (FILTER_TYPE_DEVICE); }
           ;
sm         : SM scalar               { handle_filter (FILTER_TYPE_SM); }
           ;
warp       : WARP scalar             { handle_filter (FILTER_TYPE_WARP); }
           ;
lane       : LANE scalar             { handle_filter (FILTER_TYPE_LANE); }
           ;
kernel     : KERNEL scalar           { handle_filter (FILTER_TYPE_KERNEL); }
           ;
grid       : GRID scalar             { handle_filter (FILTER_TYPE_GRID); }
           ;
block      : BLOCK cudim3            { handle_filter (FILTER_TYPE_BLOCK); }
           ;
thread     : THREAD cudim3           { handle_filter (FILTER_TYPE_THREAD); }
           ;

queries : query
        | queries query
        ;

query : DEVICE            { handle_query (FILTER_TYPE_DEVICE); }
      | SM                { handle_query (FILTER_TYPE_SM); }
      | WARP              { handle_query (FILTER_TYPE_WARP); }
      | LANE              { handle_query (FILTER_TYPE_LANE); }
      | KERNEL            { handle_query (FILTER_TYPE_KERNEL); }
      | GRID              { handle_query (FILTER_TYPE_GRID); }
      | BLOCK             { handle_query (FILTER_TYPE_BLOCK); }
      | THREAD            { handle_query (FILTER_TYPE_THREAD); }
      ;

and_conditions : condition
               | and_conditions LOGICAL_AND condition
               ;

or_conditions : condition LOGICAL_OR condition
              | or_conditions LOGICAL_OR condition
              :

condition : BLOCKIDX_X CMP scalar     { handle_condition (FILTER_TYPE_BLOCKIDX_X, $2); }
          | BLOCKIDX_Y CMP scalar     { handle_condition (FILTER_TYPE_BLOCKIDX_Y, $2); }
          | BLOCKIDX_Z CMP scalar     { handle_condition (FILTER_TYPE_BLOCKIDX_Z, $2); }
          | THREADIDX_X CMP scalar    { handle_condition (FILTER_TYPE_THREADIDX_X, $2); }
          | THREADIDX_Y CMP scalar    { handle_condition (FILTER_TYPE_THREADIDX_Y, $2); }
          | THREADIDX_Z CMP scalar    { handle_condition (FILTER_TYPE_THREADIDX_Z, $2); }
          ;

switches : switch
         | switches switch
         ;

switch : DEVICE scalar           { handle_switch (FILTER_TYPE_DEVICE); }
       | SM scalar               { handle_switch (FILTER_TYPE_SM); }
       | WARP scalar             { handle_switch (FILTER_TYPE_WARP); }
       | LANE scalar             { handle_switch (FILTER_TYPE_LANE); }
       | KERNEL scalar           { handle_switch (FILTER_TYPE_KERNEL); }
       | GRID scalar             { handle_switch (FILTER_TYPE_GRID); }
       | BLOCK cudim3            { handle_switch (FILTER_TYPE_BLOCK); }
       | THREAD cudim3           { handle_switch (FILTER_TYPE_THREAD); }
       ;

cudim3 : OPENPAR triplet CLOSEPAR
       | triplet
       ;

scalar : OPENPAR x CLOSEPAR
       | x
       ;

triplet : x COMMA y COMMA z   { ; }
        | x COMMA y           { handle_scalar (2, default_value); }
        | x                   { handle_scalar (1, default_value); handle_scalar (2, default_value); }
        ;

x : VALUE    { handle_scalar (0, $1); }
  | CURRENT  { handle_scalar (0, CUDA_CURRENT); }
  | WILDCARD { handle_scalar (0, CUDA_WILDCARD); }
  ;

y : VALUE    { handle_scalar (1, $1); }
  | CURRENT  { handle_scalar (1, CUDA_CURRENT); }
  | WILDCARD { handle_scalar (1, CUDA_WILDCARD); }
  ;

z : VALUE    { handle_scalar (2, $1); }
  | CURRENT  { handle_scalar (2, CUDA_CURRENT); }
  | WILDCARD { handle_scalar (2, CUDA_WILDCARD); }
  ;

%%

#include "cuda-parser.lex.c"

static void
new_request (filter_type_t type, request_value_t value, compare_t cmp)
{
  request_t *request;

  ++result.num_requests;

  if (result.num_requests > result.max_requests)
    {
      if (result.max_requests == 0)
        result.max_requests = 4;
      else
        result.max_requests *= 2;

      result.requests = (request_t *) xrealloc (result.requests, result.max_requests * sizeof (*request));
    }

  request = &result.requests[result.num_requests-1];
  request->type  = type;
  request->value = value;
  request->cmp   = cmp;
}

static void
reset_value_buffer (void)
{
  memset (&value_buffer, ~0, sizeof (value_buffer));
}

static void
handle_command (command_t cmd)
{
  if (result.command != CMD_ERROR)
    result.command = cmd;
}

static void
handle_query (filter_type_t type)
{
  if (handled_filters & type)
    cuda_parser_error ("Duplicate filter type");
  handled_filters = (filter_type_t) (handled_filters | type);
  value_buffer.cudim3 = (CuDim3) {CUDA_CURRENT, CUDA_CURRENT, CUDA_CURRENT};
  new_request (type, value_buffer, CMP_NONE);
  reset_value_buffer ();
}

static void
handle_condition (filter_type_t type, compare_t cmp)
{
  new_request (type, value_buffer, cmp);
  reset_value_buffer ();
}

static void
handle_switch (filter_type_t type)
{
  if (handled_filters & type)
    cuda_parser_error ("Duplicate filter type");
  handled_filters = (filter_type_t) (handled_filters | type);
  new_request (type, value_buffer, CMP_NONE);
  reset_value_buffer ();
}

static void
handle_filter (filter_type_t type)
{
  if (handled_filters & type)
    cuda_parser_error ("Duplicate filter type");
  handled_filters = (filter_type_t) (handled_filters | type);
  new_request (type, value_buffer, CMP_NONE);
  reset_value_buffer ();
}

static void
handle_scalar (uint32_t index, uint32_t value)
{
  switch (index)
    {
    case 0: value_buffer.cudim3.x = value; break;
    case 1: value_buffer.cudim3.y = value; break;
    case 2: value_buffer.cudim3.z = value; break;
    default: cuda_parser_error ("Unexpected index in handle_scalar.");
    }
}

static void
print_coord  (uint32_t value)
{
  if (value == ~0)
    printf ("x");
  else
    printf ("%u", value);
}

static void
print_request (request_t *request)
{
  if (request->type & FILTER_TYPE_DEVICE) printf ("device");
  if (request->type & FILTER_TYPE_SM)     printf ("sm");
  if (request->type & FILTER_TYPE_WARP)   printf ("warp");
  if (request->type & FILTER_TYPE_LANE)   printf ("lane");
  if (request->type & FILTER_TYPE_KERNEL) printf ("kernel");
  if (request->type & FILTER_TYPE_GRID)   printf ("grid");
  if (request->type & FILTER_TYPE_BLOCK)  printf ("block");
  if (request->type & FILTER_TYPE_THREAD) printf ("thread");
  if (request->type & FILTER_TYPE_BLOCKIDX_X)  printf ("blockIdx.x");
  if (request->type & FILTER_TYPE_BLOCKIDX_Y)  printf ("blockIdx.y");
  if (request->type & FILTER_TYPE_BLOCKIDX_Z)  printf ("blockIdx.z");
  if (request->type & FILTER_TYPE_THREADIDX_X) printf ("threadIdx.z");
  if (request->type & FILTER_TYPE_THREADIDX_Y) printf ("threadIdx.y");
  if (request->type & FILTER_TYPE_THREADIDX_Z) printf ("threadIdx.z");

  if (request->cmp == CMP_EQ) printf (" == ");
  if (request->cmp == CMP_NE) printf (" != ");
  if (request->cmp == CMP_LT) printf (" < ");
  if (request->cmp == CMP_GT) printf (" > ");
  if (request->cmp == CMP_LE) printf (" <= ");
  if (request->cmp == CMP_GE) printf (" >= ");
  if (request->cmp == CMP_NONE) printf (" ");

  if (request->type & FILTER_TYPE_THREAD ||
      request->type & FILTER_TYPE_BLOCK)
    {
      printf ("(");
      print_coord (request->value.cudim3.x);
      printf (",");
      print_coord (request->value.cudim3.y);
      printf (",");
      print_coord (request->value.cudim3.z);
      printf (")");
    }
  else
    print_coord (request->value.scalar);

  printf ("\n");
}


void
cuda_parser_error (const char *msg)
{
  //fprintf(stderr, "CUDA parser error: %s\n", msg);
  result.command = CMD_ERROR;
}

static void
cuda_parser_initialize (const char *input, command_t command,
                        cuda_coords_special_value_t dflt_value)
{
  uint32_t size = strlen (input) + 1;

  /* Initialized the result */
  result.command = CMD_NONE;
  result.num_requests = 0;
  result.max_requests = 0;
  result.requests = NULL;

  /* Initialize the input buffer */
  if (cuda_parser_input.buffer)
    xfree (cuda_parser_input.buffer);
  cuda_parser_input.buffer = (char *) xmalloc (size);
  cuda_parser_input.size   = size - 1; // do not pass the '\0' character
  cuda_parser_input.ptr    = (char *) memcpy (cuda_parser_input.buffer, input, size);

  /* Initialize the temporary buffer */
  reset_value_buffer ();

  /* Initialize the sanity check variables */
  handled_filters = FILTER_TYPE_NONE;

  /* Initialize the requested command */
  if (command & CMD_SWITCH ||
      command & CMD_QUERY ||
      command & CMD_FILTER ||
      command & CMD_FILTER_KERNEL)
{
    start_token = command;
}
  else
    cuda_parser_error ("Incorrect requested command\n");

  /* Initialize the default value */
  default_value = dflt_value;
}

int
cuda_parser_get_next_input (char *buf, int max_size)
{
  int remaining_size = cuda_parser_input.buffer + cuda_parser_input.size - cuda_parser_input.ptr;
  int copy_size = remaining_size > max_size ? max_size : remaining_size;

  if (copy_size == 0)
    return 0;

  memcpy (buf, cuda_parser_input.ptr, copy_size);
  cuda_parser_input.ptr += copy_size;
  return copy_size;
}

void
cuda_parser (const char *input, command_t command, cuda_parser_result_t **res,
             cuda_coords_special_value_t dflt_value)
{
  cuda_parser_reset_lexer ();
  cuda_parser_initialize (input, command, dflt_value);
  cuda_parser_parse ();

  *res = &result;
}

void
cuda_parser_print (cuda_parser_result_t *result)
{
  request_t *request;
  uint32_t i;

  printf ("Command: ");
  switch (result->command)
    {
    case CMD_NONE: printf ("none"); break;
    case CMD_ERROR: printf ("error"); break;
    case CMD_QUERY: printf ("query"); break;
    case CMD_COND_AND: printf ("and conditions"); break;
    case CMD_COND_OR: printf ("or conditions"); break;
    case CMD_SWITCH: printf ("switch"); break;
    default: break;
    }
  printf ("\n");

  if (result->command == CMD_ERROR)
    return;

  for (i = 0; i < result->num_requests; ++i)
    {
      request = &result->requests[i];
      if (i > 0)
        if (result->command == CMD_COND_AND)
          printf (" && ");
        else if (result->command == CMD_COND_OR)
          printf (" || ");
        else
          printf ("    ");
      else
        printf ("    ");
      print_request (request);
    }
}
