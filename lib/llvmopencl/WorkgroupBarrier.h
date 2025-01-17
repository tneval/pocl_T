// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2011-2019 Pekka Jääskeläinen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef POCL_WGBARRIER_H
#define POCL_WGBARRIER_H

#include "config.h"
#include "Barrier.h"

namespace pocl {
  // Class for work-group barrier instructions, inherits from barrier class. 
  // This is semantically same as previous implementation of Barrier.
  class WorkgroupBarrier : public Barrier {
  public:

    /// Ensures there is a workgroup barrier call in the basic block before 
    /// the given instruction.
    ///
    /// Otherwise, creates a new one there.
    ///
    /// \returns The workgroup barrier.
    static WorkgroupBarrier *create(llvm::Instruction *InsertBefore) {
      llvm::Module *M = InsertBefore->getParent()->getParent()->getParent();

      if (InsertBefore != &InsertBefore->getParent()->front() &&
          llvm::isa<Barrier>(InsertBefore->getPrevNode()))
        return llvm::cast<WorkgroupBarrier>(InsertBefore->getPrevNode());

      llvm::FunctionCallee FC =
        M->getOrInsertFunction(WGBARRIER_FUNCTION_NAME,
                                llvm::Type::getVoidTy(M->getContext()));
      llvm::Function *F = llvm::cast<llvm::Function>(FC.getCallee());
      F->addFnAttr(llvm::Attribute::Convergent);
      return llvm::cast<pocl::WorkgroupBarrier>
        (llvm::CallInst::Create(F, "", InsertBefore));
    }

    static bool classof(const Barrier *) { return true; }
    static bool classof(const llvm::CallInst *C) {
      return C->getCalledFunction() != NULL &&
        C->getCalledFunction()->getName() == WGBARRIER_FUNCTION_NAME;
    }
    static bool classof(const Instruction *I) {
      return (llvm::isa<llvm::CallInst>(I) &&
              classof(llvm::cast<llvm::CallInst>(I)));
    }
    static bool classof(const User *U) {
      return (llvm::isa<Instruction>(U) &&
              classof(llvm::cast<llvm::Instruction>(U)));
    }
    static bool classof(const Value *V) {
      return (llvm::isa<User>(V) &&
              classof(llvm::cast<llvm::User>(V)));
    }
  };
  
}

#endif
