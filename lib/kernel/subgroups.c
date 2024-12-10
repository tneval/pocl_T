/* OpenCL built-in library: subroups functionality

   Copyright (c) 2022-2024 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

/* The default implementation of subgroups for CPU drivers. It uses work-group
   sized local buffers for exchanging the data. The subgroup size is by default
   the local X dimension side, unless restricted with the
   intel_reqd_sub_group_size metadata.
 */

#include <math.h>

#include "work_group_alloca.h"
#include <stdio.h>
#include <string.h>

size_t _CL_OVERLOADABLE get_local_id (unsigned int dimindx);
size_t _CL_OVERLOADABLE get_local_linear_id (void);
size_t _CL_OVERLOADABLE get_local_size (unsigned int dimindx);


struct String{
    char str[100];
    int val;
    float fval;
};


void _CL_OVERLOADABLE 
printFromKernel(struct String s)
{
  fprintf(stderr, "%s\n",s.str);
}

void _CL_OVERLOADABLE 
printIntFromKernel(struct String s){

  fprintf(stderr,"%s\tVAL: %d\n",s.str, s.val);
}

void _CL_OVERLOADABLE
printFloatFromKernel(struct String s){
  fprintf(stderr,"%s: %d\n",s.str,s.val);
}



/* Magic variable that is expanded in Workgroup.cc */
extern uint _pocl_sub_group_size;

uint _CL_OVERLOADABLE
get_sub_group_size (void)
{
  return _pocl_sub_group_size;
}

uint _CL_OVERLOADABLE
get_max_sub_group_size (void)
{
  return get_sub_group_size ();
}

uint _CL_OVERLOADABLE
get_num_sub_groups (void)
{
  return (uint)get_local_size (0) * get_local_size (1) * get_local_size (2)
         / get_max_sub_group_size ();
}

uint _CL_OVERLOADABLE
get_enqueued_num_sub_groups (void)
{
  return 1;
}

uint _CL_OVERLOADABLE
get_sub_group_id (void)
{
  return (uint)get_local_linear_id () / get_max_sub_group_size ();
}

uint _CL_OVERLOADABLE
get_sub_group_local_id (void)
{
  return (uint)get_local_linear_id () % get_max_sub_group_size ();
}

static size_t _CL_OVERLOADABLE
get_first_llid (void)
{
  return get_sub_group_id () * get_max_sub_group_size ();
}

void _CL_OVERLOADABLE sub_group_barrier (cl_mem_fence_flags flags);

#define SUB_GROUP_SHUFFLE_PT(PREFIX, TYPE)                                    \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle (TYPE val, uint index)      \
  {                                                                           \
    volatile TYPE *temp_storage                                               \
        = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE), 0);         \
    temp_storage[get_local_linear_id ()] = val;                               \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return temp_storage[get_first_llid () + index % get_sub_group_size ()];   \
  }

/* Define both the non-prefixed (khr) and Intel-prefixed shuffles. */
#define SUB_GROUP_SHUFFLE_T(TYPE)                                             \
  SUB_GROUP_SHUFFLE_PT (, TYPE)                                               \
  SUB_GROUP_SHUFFLE_PT (intel_, TYPE)

SUB_GROUP_SHUFFLE_T (char)
SUB_GROUP_SHUFFLE_T (uchar)
SUB_GROUP_SHUFFLE_T (short)
SUB_GROUP_SHUFFLE_T (ushort)
SUB_GROUP_SHUFFLE_T (int)
SUB_GROUP_SHUFFLE_T (uint)
SUB_GROUP_SHUFFLE_T (long)
SUB_GROUP_SHUFFLE_T (ulong)
#ifdef cl_khr_fp16
SUB_GROUP_SHUFFLE_T (half)
/* OpenCL C mangles half 'h' whereas C (clang) mangles it 'fp16'.
   We need to provide a wrapper for the OpenCL C compatible mangling. */
half
_Z23intel_sub_group_shuffleDhj (half val, uint mask)
{
  return intel_sub_group_shuffle (val, mask);
}
#endif
SUB_GROUP_SHUFFLE_T (float)
SUB_GROUP_SHUFFLE_T (double)

#define SUB_GROUP_SHUFFLE_XOR_PT(PREFIX, TYPE)                                \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE PREFIX##sub_group_shuffle_xor (TYPE val, uint mask)   \
  {                                                                           \
    volatile TYPE *temp_storage                                               \
        = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE), 0);         \
    temp_storage[get_local_linear_id ()] = val;                               \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return temp_storage[get_first_llid ()                                     \
                        + (get_sub_group_local_id () ^ mask)                  \
                              % get_sub_group_size ()];                       \
  }

/* Define both the non-prefixed (khr) and Intel-prefixed shuffles. */
#define SUB_GROUP_SHUFFLE_XOR_T(TYPE)                                         \
  SUB_GROUP_SHUFFLE_XOR_PT (, TYPE)                                           \
  SUB_GROUP_SHUFFLE_XOR_PT (intel_, TYPE)

SUB_GROUP_SHUFFLE_XOR_T (char)
SUB_GROUP_SHUFFLE_XOR_T (uchar)
SUB_GROUP_SHUFFLE_XOR_T (short)
SUB_GROUP_SHUFFLE_XOR_T (ushort)
SUB_GROUP_SHUFFLE_XOR_T (int)
SUB_GROUP_SHUFFLE_XOR_T (uint)
SUB_GROUP_SHUFFLE_XOR_T (long)
SUB_GROUP_SHUFFLE_XOR_T (ulong)
SUB_GROUP_SHUFFLE_XOR_T (float)
SUB_GROUP_SHUFFLE_XOR_T (double)

#define SUB_GROUP_BROADCAST_T(TYPE)                                           \
  __attribute__ ((always_inline)) TYPE _CL_OVERLOADABLE sub_group_broadcast ( \
      TYPE val, uint id)                                                      \
  {                                                                           \
    return sub_group_shuffle (val, id);                                       \
  }

SUB_GROUP_BROADCAST_T (int)
SUB_GROUP_BROADCAST_T (uint)
SUB_GROUP_BROADCAST_T (long)
SUB_GROUP_BROADCAST_T (ulong)
SUB_GROUP_BROADCAST_T (float)
SUB_GROUP_BROADCAST_T (double)
#ifdef cl_khr_fp16
SUB_GROUP_BROADCAST_T (half)
/* OpenCL C mangles half 'h' whereas C (clang) mangles it 'fp16'.
   We need to provide a wrapper for the OpenCL C compatible mangling. */
half
_Z19sub_group_broadcastDhj (half val, uint mask)
{
  return sub_group_broadcast (val, mask);
}

#endif

#define SUB_GROUP_REDUCE_OT(OPNAME, OPERATION, TYPE)                          \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE sub_group_reduce_##OPNAME (TYPE val)                  \
  {                                                                           \
  /* struct String string = {"<PoCL/Kernel-lib> sub_group_reduce()"}; */ \
    /* printFromKernel(string); */ \
    /* struct String ss1 = {"pocl",val,val}; */\
            /* printFloatFromKernel(ss1); */ \
  /* struct String lid = {"localLinearId",get_local_linear_id(),get_local_linear_id()}; */ \
 /*  printFloatFromKernel(lid);  */    \
  /* struct String firstId = {"firstllid", get_first_llid(),get_first_llid()}; */ \
  /* printFloatFromKernel(firstId); */   \
    volatile TYPE *temp_storage                                               \
        = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE), 0);         \
    temp_storage[get_local_linear_id ()] = val;                               \
    /* struct String first_val = {"wi value:",temp_storage[get_local_linear_id()],temp_storage[get_local_linear_id()]}; */   \
    /* printFloatFromKernel(first_val); */ \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    /* struct String first_val2 = {"wi value2:",temp_storage[get_local_linear_id()],temp_storage[get_local_linear_id()]}; */   \
    /* printFloatFromKernel(first_val2); */ \
    /* int c = temp_storage[get_local_linear_id() + 1]; */ \
    /* struct String cs = {"c_var", c, c}; */ \
    /* printFloatFromKernel(cs); */ \
    if (get_sub_group_local_id () == 0)                                       \
      {     \
        /* struct String printti = {"subgrouplocalid == 0"}; */ \
         /* printFromKernel(printti); */                                                                  \
        for (uint i = 1; i < get_sub_group_size (); ++i)                      \
          {                                                                   \
            TYPE a = temp_storage[get_first_llid ()],                         \
                 b = temp_storage[get_first_llid () + i];                     \
            /* temp_storage[get_first_llid ()] = OPERATION; */                      \
            temp_storage[get_first_llid ()] = a+b; \
            /* struct String as = {"a_var", a, a}; */ \
            /* struct String bs = {"b_var", b, b}; */ \
            /* printFloatFromKernel(as); */  \
            /* printFloatFromKernel(bs); */ \
            /* struct String ss2 = {"pocl",0,temp_storage[get_first_llid()]}; */    \
            /* printFloatFromKernel(ss2); */                                        \
          }                                                                   \
      }                                                                       \
      /* struct String first_val3 = {"wi value3:",temp_storage[get_local_linear_id()],temp_storage[get_local_linear_id()]};  */  \
    /* printFloatFromKernel(first_val3);  */                                                                  \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return temp_storage[get_first_llid ()];                                   \
  }

#define SUB_GROUP_REDUCE_T(OPNAME, OPERATION)                                 \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, int)                                \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, uint)                               \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, long)                               \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, ulong)                              \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, float)                              \
  SUB_GROUP_REDUCE_OT (OPNAME, OPERATION, double)

SUB_GROUP_REDUCE_T (add, a + b)
SUB_GROUP_REDUCE_T (min, a > b ? b : a)
SUB_GROUP_REDUCE_T (max, a > b ? a : b)

#ifdef cl_khr_fp16
SUB_GROUP_REDUCE_OT (add, a + b, half)
SUB_GROUP_REDUCE_OT (max, a > b ? a : b, half)

half
_Z20sub_group_reduce_maxDh (half val)
{
  return sub_group_reduce_max (val);
}

half
_Z20sub_group_reduce_addDh (half val)
{
  return sub_group_reduce_add (val);
}
#endif

#define SUB_GROUP_SCAN_INCLUSIVE_OT(OPNAME, OPERATION, TYPE)                  \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE sub_group_scan_inclusive_##OPNAME (TYPE val)          \
  {                                                                           \
    volatile TYPE *data                                                       \
        = __pocl_work_group_alloca (sizeof (TYPE), sizeof (TYPE), 0);         \
    data[get_local_linear_id ()] = val;                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    if (get_sub_group_local_id () == 0)                                       \
      {                                                                       \
        for (uint i = 1; i < get_sub_group_size (); ++i)                      \
          {                                                                   \
            TYPE a = data[get_first_llid () + i - 1],                         \
                 b = data[get_first_llid () + i];                             \
            data[get_first_llid () + i] = OPERATION;                          \
          }                                                                   \
      }                                                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return data[get_local_linear_id ()];                                      \
  }

#define SUB_GROUP_SCAN_INCLUSIVE_T(OPNAME, OPERATION)                         \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, int)                        \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, uint)                       \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, long)                       \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, ulong)                      \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, float)                      \
  SUB_GROUP_SCAN_INCLUSIVE_OT (OPNAME, OPERATION, double)

SUB_GROUP_SCAN_INCLUSIVE_T (add, a + b)
SUB_GROUP_SCAN_INCLUSIVE_T (min, a > b ? b : a)
SUB_GROUP_SCAN_INCLUSIVE_T (max, a > b ? a : b)

#define SUB_GROUP_SCAN_EXCLUSIVE_OT(OPNAME, OPERATION, TYPE, ID)              \
  __attribute__ ((always_inline))                                             \
  TYPE _CL_OVERLOADABLE sub_group_scan_exclusive_##OPNAME (TYPE val)          \
  {                                                                           \
    volatile TYPE *data = __pocl_work_group_alloca (                          \
        sizeof (TYPE), sizeof (TYPE), sizeof (TYPE));                         \
    data[get_local_linear_id () + 1] = val;                                   \
    data[get_first_llid ()] = ID;                                             \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    if (get_sub_group_local_id () == 0)                                       \
      {                                                                       \
        for (uint i = 1; i < get_sub_group_size (); ++i)                      \
          {                                                                   \
            TYPE a = data[get_first_llid () + i - 1],                         \
                 b = data[get_first_llid () + i];                             \
            data[get_first_llid () + i] = OPERATION;                          \
          }                                                                   \
      }                                                                       \
    sub_group_barrier (CLK_LOCAL_MEM_FENCE);                                  \
    return data[get_local_linear_id ()];                                      \
  }

SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, int, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, uint, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, long, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, ulong, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, float, 0.0f)
SUB_GROUP_SCAN_EXCLUSIVE_OT (add, a + b, double, 0.0)

SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, int, INT_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, uint, UINT_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, long, LONG_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, ulong, ULONG_MAX)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, float, +INFINITY)
SUB_GROUP_SCAN_EXCLUSIVE_OT (min, a > b ? b : a, double, +INFINITY)

SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, int, INT_MIN)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, uint, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, long, LONG_MIN)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, ulong, 0)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, float, -INFINITY)
SUB_GROUP_SCAN_EXCLUSIVE_OT (max, a > b ? a : b, double, -INFINITY)

__attribute__ ((always_inline)) uint4 _CL_OVERLOADABLE
sub_group_ballot (int predicate)
{
  /* The results for the ballot for all of the WG's SGs. */
  uint4 *flags = __pocl_work_group_alloca (sizeof (uint4), sizeof (uint4), 0);
  /* Temporary storage for the predicate flags of all WIs in the WG. */
  char *res = __pocl_work_group_alloca (sizeof (char), 4, 0);
  res[get_local_linear_id ()] = !!predicate;
  sub_group_barrier (CLK_LOCAL_MEM_FENCE);
  if (get_sub_group_local_id () == 0)
    {
      flags[get_sub_group_id ()] = 0;
      uint *f = (uint*) (flags + get_sub_group_id ());
      for (uint i = 0; i < get_sub_group_size () && i < 128; ++i)
        {
          f[i / 32] |= res[get_first_llid () + i] << (i % 32);
        }
    }
  sub_group_barrier (CLK_LOCAL_MEM_FENCE);
  return flags[get_sub_group_id ()];
}
