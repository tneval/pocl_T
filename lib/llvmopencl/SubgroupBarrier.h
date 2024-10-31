#ifndef POCL_SGBARRIER_H
#define POCL_SGBARRIER_H

#include "config.h"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/Support/Casting.h>

#define SGBARRIER_FUNCTION_NAME "pocl.subgroup_barrier"

namespace pocl {

  class SubgroupBarrier : public llvm::CallInst {
  public:
    
    /// Ensures there is a barrier call in the basic block before the given
    /// instruction.
    ///
    /// Otherwise, creates a new one there.
    ///
    /// \returns The barrier.
    static SubgroupBarrier *create(llvm::Instruction *InsertBefore) {
      llvm::Module *M = InsertBefore->getParent()->getParent()->getParent();

      if (InsertBefore != &InsertBefore->getParent()->front() &&
          llvm::isa<SubgroupBarrier>(InsertBefore->getPrevNode()))
        return llvm::cast<SubgroupBarrier>(InsertBefore->getPrevNode());

      llvm::FunctionCallee FC =
        M->getOrInsertFunction(SGBARRIER_FUNCTION_NAME,
                                llvm::Type::getVoidTy(M->getContext()));
      llvm::Function *F = llvm::cast<llvm::Function>(FC.getCallee());
      F->addFnAttr(llvm::Attribute::Convergent);
      return llvm::cast<pocl::SubgroupBarrier>
        (llvm::CallInst::Create(F, "", InsertBefore));
    }

    static bool classof(const SubgroupBarrier *) { return true; }
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

    static bool hasOnlyBarrier(const llvm::BasicBlock *BB) {
      return endsWithSGBarrier(BB) && BB->size() == 2;
    }

    static bool hasSGBarrier(const llvm::BasicBlock *BB) {
      for (llvm::BasicBlock::const_iterator I = BB->begin(), E = BB->end();
           I != E; ++I)
        if (llvm::isa<SubgroupBarrier>(I))
          return true;
      return false;
    }

    // Returns true in case the given basic block starts with a barrier,
    // that is, contains a branch instruction after possible PHI nodes.
    static bool startsWithSGBarrier(const llvm::BasicBlock *BB) {
      const llvm::Instruction *Inst = BB->getFirstNonPHI();
      if (Inst == NULL)
        return false;
      return llvm::isa<SubgroupBarrier>(Inst);
    }

    // Returns true in case the given basic block ends with a barrier,
    // that is, contains only a branch instruction after a barrier call.
    static bool endsWithSGBarrier(const llvm::BasicBlock *BB) {
      const llvm::Instruction *Inst = BB->getTerminator();
      if (Inst == NULL)
        return false;
      return BB->size() > 1 && Inst->getPrevNode() != NULL &&
          llvm::isa<SubgroupBarrier>(Inst->getPrevNode());
    }
  };

}

#endif
