#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014 pocl developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

#cl.hpp: $(top_srcdir)/include/CL/cl.hpp.in $(top_srcdir)/tools/patches/khronos_cl.hpp.patch

if(INSTALL_OPENCL_HEADERS)
  install(FILES cl.h
                cl_d3d10.h
                cl_d3d11.h
                cl_dx9_media_sharing.h
                cl_dx9_media_sharing_intel.h
                cl_ext.h
                cl_egl.h
                cl_exp_tensor.h
                cl_exp_defined_builtin_kernels.h
                cl_ext_intel.h
                cl_ext_pocl.h
                cl_gl.h
                cl_gl_ext.h
                cl_half.h
                cl_icd.h
                cl_platform.h
                cl_va_api_media_sharing_intel.h
                cl_version.h
                opencl.h
                cl2.hpp opencl.hpp
          DESTINATION "${POCL_INSTALL_OPENCL_HEADER_DIR_REL}" COMPONENT "dev")
endif()
