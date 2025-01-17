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

#ifndef POCL_SGBARRIER_H
#define POCL_SGBARRIER_H

#include "config.h"
#include "Barrier.h"

namespace pocl{
  // Class for subgroup barrier instruction, inherits from barrier class.
  // Used by fiber work-group method to handle control flow of diverging 
  // sub-groups.
  // Other work-group methods identify this as "barrier" and is given a
  // work-group barrier treatment.
  class SubgroupBarrier : public Barrier {
  public:
    
    static bool classof(const SubgroupBarrier *S) { return true; }

    static bool classof(const llvm::CallInst *C) {
      return C->getCalledFunction() != nullptr &&
        C->getCalledFunction()->getName() == SGBARRIER_FUNCTION_NAME;
    }
    static bool classof(const llvm::Instruction *I) {
      return llvm::isa<SubgroupBarrier>(I);
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