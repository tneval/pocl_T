#ifndef POCL_FIBER_H
#define POCL_FIBER_H

#include "config.h"

#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>

namespace pocl {


class Fiber : public llvm::PassInfoMixin<Fiber>{

public:
    static void registerWithPB(llvm::PassBuilder &B);
    llvm::PreservedAnalyses run(llvm::Function &F,llvm::FunctionAnalysisManager &AM);
    static bool isRequired() { return true; }

};

}

#endif