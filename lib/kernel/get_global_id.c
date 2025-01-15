/* OpenCL built-in library: get_global_id()

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

extern const size_t _local_size_x;
extern const size_t _local_size_y;
extern const size_t _local_size_z;

extern const size_t _group_id_x;
extern const size_t _group_id_y;
extern const size_t _group_id_z;

extern const size_t _local_id_x;
extern const size_t _local_id_y;
extern const size_t _local_id_z;

extern const size_t _global_id_x;
extern const size_t _global_id_y;
extern const size_t _global_id_z;

extern const size_t _global_offset_x;
extern const size_t _global_offset_y;
extern const size_t _global_offset_z;

extern const size_t _global_id_x;
extern const size_t _global_id_y;
extern const size_t _global_id_z;

/* We generate the global id in WorkitemLoops using the loop iterator to make
   the life easier for loopvec. This will be inlined and removed. */
#if _MSC_VER
size_t
__identifier ("?get_global_id@@$$J0YAKI@Z") (unsigned dim)
#else
size_t
_Z13get_global_idj (unsigned dim)
#endif
{
  switch (dim)
    {
    case 0:
      return _global_id_x;
    case 1:
      return _global_id_y;
    case 2:
      return _global_id_z;
    default:
      return 0;
    }
}
