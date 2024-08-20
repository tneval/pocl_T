/* pocl_llvm_api.cc: internally used header for pocl's LLVM API sources.

   Copyright (c) 2013 Kalle Raiskila
                 2013-2017 Pekka Jääskeläinen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl_llvm.h"

#ifndef POCL_LLVM_API_H
#define POCL_LLVM_API_H

#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <map>
#include <string>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* The LLVM API interface functions are not thread safe at the moment;
 * Pocl needs to ensure only one thread is using this layer at the time.
 *
 * Pocl used a llvm::sys::Mutex class variable before, unfortunately,
 * using llvm::sys::Mutex is not safe. Reason:
 *
 * if pocl is dlopened from a C++ program, pocl dlopens libLLVM. At exit time,
 * LLVM is unloaded before pocl, and its destructors are called.
 *
 * This causes the Mutex to be destroyed,
 * and if the program's dtors call clReleaseProgram()
 * -> pocl_free_llvm_irs() -> llvm::PoclMutexGuard guard_variable(Mutex)
 * ... the program will freeze/segfault.
 *
 * This happens with many ViennaCL examples.
 *
 * This class is a replacement that uses a simple pthread lock
 */

class PoclCompilerMutexGuard {
  PoclCompilerMutexGuard(const PoclCompilerMutexGuard &) = delete;
  void operator=(const PoclCompilerMutexGuard &) = delete;
  pocl_lock_t *lock;

public:
  PoclCompilerMutexGuard (pocl_lock_t *ptr);
  ~PoclCompilerMutexGuard();
};

llvm::Module *parseModuleIR (const char *path, llvm::LLVMContext *c);
void parseModuleGVarSize (cl_program program, unsigned device_i,
                          llvm::Module *ProgramBC);
void writeModuleIRtoString(const llvm::Module *mod, std::string& dest);
llvm::Module *parseModuleIRMem (const char *input_stream, size_t size,
                                llvm::LLVMContext *c);
std::string getDiagString (cl_context ctx);
std::string getDiagString (void *PoclCtx);

void setModuleIntMetadata (llvm::Module *mod, const char *key, unsigned long data);
void setModuleStringMetadata (llvm::Module *mod, const char *key,
                              const char *data);
void setModuleBoolMetadata (llvm::Module *mod, const char *key, bool data);

POCL_EXPORT bool getModuleIntMetadata (const llvm::Module &mod,
                                       const char *key, unsigned long &data);
POCL_EXPORT bool getModuleStringMetadata (const llvm::Module &mod,
                                          const char *key, std::string &data);
POCL_EXPORT bool getModuleBoolMetadata (const llvm::Module &mod,
                                        const char *key, bool &data);

/**
 * Creates a PassBuilder and populates a PassManager with LLMV optimization
 * passes. Optionally runs it on a Module.
 *
 * Passes - optional. If non-NULL, it must be a pointer to PassManager which
 * the PassBuilder will use, otherwise a hidden PM is used. Currently only
 * required in one place (runKernelCompilerPasses for legacy PM).
 * Module - optional. If non-NULL, it must be a pointer to Module.
 * The constructed PassManager instance will run the passes on the Module.
 * OptL - optimize for speed (0 to 3 are valid)
 * SizeL - optimize for size
 * Vectorize - whether to invoke the vectorizer (only used for legacy PM)
 */
POCL_EXPORT void populateModulePM (void *Passes,
                                   void *Module,
                                   unsigned OptL,
                                   unsigned SizeL,
                                   bool Vectorize = true,
                                   llvm::TargetMachine *TM = nullptr);

extern std::string CurrentWgMethod;

extern const char *PoclGVarPrefix;
extern const char *PoclGVarBufferName;
extern const char *PoclGVarMDName;

typedef std::map<cl_device_id, llvm::Module *> kernelLibraryMapTy;
struct PoclLLVMContextData
{
  pocl_lock_t Lock;
  llvm::LLVMContext *Context;
  unsigned number_of_IRs;
  std::string *poclDiagString;
  llvm::raw_string_ostream *poclDiagStream;
  llvm::DiagnosticPrinterRawOStream *poclDiagPrinter;
  kernelLibraryMapTy *kernelLibraryMap;
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
