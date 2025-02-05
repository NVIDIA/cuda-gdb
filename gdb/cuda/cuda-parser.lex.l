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

WHITE	[ \t\n]
COMMA   [,]
DIG	[0-9]
NUM10	({DIG}({DIG})*)
HEXDIG	[0-9a-fA-F]
NUM16	(0x{HEXDIG}({HEXDIG})*)
CONV    (\$[0-9a-zA-Z_]+)

%{

#include "defs.h"

#include <stdlib.h>
#include "value.h"
#include "cuda/cuda-parser.h"
#include "cuda-parser.tab.h"

extern int cuda_parser_lex (void);
int  cuda_parser_get_next_input (char *buf, int max_size);
void cuda_parser_reset_lexer (void);

/*Redefine YY_INPUT so that we can take a string as input */
#undef YY_INPUT
#define YY_INPUT(BUF, RESULT, MAX_SIZE)                        \
  (RESULT) = cuda_parser_get_next_input ((BUF), (MAX_SIZE));   \

#define YY_NO_UNPUT 1

%}

/*100% input available when parsing -> faster */
%option  never-interactive

/*generates error if unidentified token -> safer */
%option  nodefault

/*only one input, no need to define yywrap -> cleaner */
%option  noyywrap

/*don't generate input or yyunput, which are never used */
%option noinput
%option nounput

/* uncomment to debug the parser */
/* %option debug */

%%

%{
  /* Handle multiple start-symbols. See Bison manual 11.5. */
  if (start_token & CMD_SWITCH ||
      start_token & CMD_QUERY)
    {
      start_token = CMD_NONE;
      return START_QUERY_OR_SWITCH;
    }
  else if (start_token & CMD_COND_OR ||
           start_token & CMD_COND_AND)
    {
      start_token = CMD_NONE;
      return START_CONDITIONS;
    }
  else if (start_token & CMD_FILTER)
    {
      start_token = CMD_NONE;
      return START_FILTER;
    }
  else if (start_token & CMD_FILTER_KERNEL)
    {
      start_token = CMD_NONE;
      return START_FILTER_KERNEL;
    }
  else if (start_token != CMD_NONE)
    return PARSING_ERROR;

%}

device          { return DEVICE; }
sm              { return SM; }
warp            { return WARP; }
lane            { return LANE; }

kernel          { return KERNEL; }
grid            { return GRID; }
block           { return BLOCK; }
thread          { return THREAD; }

threadIdx\.x    { return THREADIDX_X; }
threadIdx\.y    { return THREADIDX_Y; }
threadIdx\.z    { return THREADIDX_Z; }
blockIdx\.x     { return BLOCKIDX_X; }
blockIdx\.y     { return BLOCKIDX_Y; }
blockIdx\.z     { return BLOCKIDX_Z; }

{NUM10}         { cuda_parser_lval.value = strtol (yytext, NULL, 10); return VALUE; }
{NUM16} 	{ cuda_parser_lval.value = strtol (yytext, NULL, 16); return VALUE; }
{CONV}          { cuda_parser_lval.value = value_as_long (parse_to_comma_and_eval ((const char **)&yytext)); return VALUE; }

current         { return CURRENT; }
any             { return WILDCARD; }

"&&"		{ return LOGICAL_AND; }
"||"		{ return LOGICAL_OR; }

"=="		{ cuda_parser_lval.cmp = CMP_EQ; return CMP; }
"!="		{ cuda_parser_lval.cmp = CMP_NE; return CMP; }
"<"		{ cuda_parser_lval.cmp = CMP_LT; return CMP; }
">"		{ cuda_parser_lval.cmp = CMP_GT; return CMP; }
"<="		{ cuda_parser_lval.cmp = CMP_LE; return CMP; }
">="		{ cuda_parser_lval.cmp = CMP_GE; return CMP; }

"("		{ return OPENPAR; }
")"		{ return CLOSEPAR; }
{COMMA}         { return COMMA; }

{WHITE}         { /* ignore */ }

.               { return PARSING_ERROR; }

%%

void cuda_parser_reset_lexer (void)
{
  yyrestart (yyin);
  BEGIN(INITIAL);
}
