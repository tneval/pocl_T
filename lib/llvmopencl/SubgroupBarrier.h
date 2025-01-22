#ifndef POCL_SGBARRIER_H
#define POCL_SGBARRIER_H

#include "Barrier.h"

#define SGBARRIER_FUNCTION_NAME "pocl.subgroup_barrier"

namespace pocl {

  class SubgroupBarrier : public Barrier {
  public:
    
    static bool classof(const SubgroupBarrier *) { return true; }

    static bool classof(const Barrier *B) {
        if (auto *C = llvm::dyn_cast<llvm::CallInst>(B)) {
            return classof(C);
        }
        return false;
    }

    static bool classof(const llvm::CallInst *C) {
        return C->getCalledFunction() != NULL &&
               C->getCalledFunction()->getName() == SGBARRIER_FUNCTION_NAME;
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


    static bool hasSGBarrier(const llvm::BasicBlock *BB) {
      for (llvm::BasicBlock::const_iterator I = BB->begin(), E = BB->end();
           I != E; ++I)
        if (llvm::isa<SubgroupBarrier>(I))
          return true;
      return false;
    }


    /* static bool classof(const SubgroupBarrier *) { return true; }

    static bool classof(const llvm::CallInst *C) {
      return C->getCalledFunction() != NULL &&
        C->getCalledFunction()->getName() == SGBARRIER_FUNCTION_NAME;
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
    } */
  };

}

#endif
