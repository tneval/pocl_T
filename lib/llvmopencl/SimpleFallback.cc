#include "LLVMUtils.h"
#include "SimpleFallback.h"
#include "WorkitemHandlerChooser.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "llvm/IR/IRBuilder.h"
#include "DebugHelpers.h"
#include "KernelCompilerUtils.h"


#include <iostream>

#define PASS_NAME "simplefallback"
#define PASS_CLASS pocl::SimpleFallback
#define PASS_DESC "Simple and robust work group function generator"


namespace pocl{




class SimpleFallbackImpl : public pocl::WorkitemHandler{

public:
    SimpleFallbackImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                    llvm::PostDominatorTree &PDT,
                    VariableUniformityAnalysisResult &VUA)
      : WorkitemHandler(), DT(DT), LI(LI), PDT(PDT), VUA(VUA) {}

    virtual bool runOnFunction(llvm::Function &F);


protected:
    llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr) override;
    //llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr,size_t Dim) override;


// TODO: Check what is actually needed, these are from wiloops
private:
    using BasicBlockVector = std::vector<llvm::BasicBlock *>;
    using InstructionIndex = std::set<llvm::Instruction *>;
    using InstructionVec = std::vector<llvm::Instruction *>;
    using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;

    llvm::DominatorTree &DT;
    llvm::LoopInfo &LI;
    llvm::PostDominatorTree &PDT;
    llvm::Module *M;
    llvm::Function *F;

    VariableUniformityAnalysisResult &VUA;




    ParallelRegion::ParallelRegionVector OriginalParallelRegions;
    
    StrInstructionMap ContextArrays;

    std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;

    size_t TempInstructionIndex;



    

    ParallelRegion *regionOfBlock(llvm::BasicBlock *BB);

    llvm::Value *getLinearWiIndex(llvm::IRBuilder<> &Builder, llvm::Module *M, ParallelRegion *Region);

    llvm::Instruction *addContextSave(llvm::Instruction *Def, llvm::AllocaInst *AllocaI);


    //bool processFunction(llvm::Function &F);

};


llvm::Instruction * SimpleFallbackImpl::addContextSave(llvm::Instruction *Def, llvm::AllocaInst *AllocaI) {

  if (llvm::isa<llvm::AllocaInst>(Def)) {
    // If the variable to be context saved is itself an alloca, we have created
    // one big alloca that stores the data of all the work-items and return
    // pointers to that array. Thus, we need no initialization code other than
    // the context data alloca itself.
    return NULL;
  }

  //Save the produced variable to the array.
  llvm::BasicBlock::iterator definition = (llvm::dyn_cast<llvm::Instruction>(Def))->getIterator();
  ++definition;
  while (llvm::isa<llvm::PHINode>(definition)) ++definition;

  // TO CLEAN: Refactor by calling CreateContextArrayGEP.
  llvm::IRBuilder<> builder(&*definition);
  std::vector<llvm::Value *> gepArgs;

  
  ParallelRegion *region = regionOfBlock(Def->getParent());
  assert ("Adding context save outside any region produces illegal code." && 
          region != NULL);

  if (WGDynamicLocalSize) {
    llvm::Module *M = AllocaI->getParent()->getParent()->getParent();
    gepArgs.push_back(getLinearWiIndex(builder, M, region));
  } else {
    gepArgs.push_back(llvm::ConstantInt::get(ST, 0));
    gepArgs.push_back(region->getOrCreateIDLoad(LID_G_NAME(2)));
    gepArgs.push_back(region->getOrCreateIDLoad(LID_G_NAME(1)));
    gepArgs.push_back(region->getOrCreateIDLoad(LID_G_NAME(0)));
  }

  return builder.CreateStore(
      Def,
#if LLVM_MAJOR < 15
      builder.CreateGEP(AllocaI->getType()->getPointerElementType(), AllocaI,
                        gepArgs));
#else
      builder.CreateGEP(AllocaI->getAllocatedType(), AllocaI, gepArgs));
#endif

}




llvm::Value *SimpleFallbackImpl::getLinearWIIndexInRegion(llvm::Instruction *Instr) {
  ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
  assert(ParRegion != nullptr);
  llvm::IRBuilder<> Builder(Instr);
  return getLinearWiIndex(Builder, M, ParRegion);
}





// TO CLEAN: Refactor into getLinearWIIndexInRegion.
llvm::Value *SimpleFallbackImpl::getLinearWiIndex(llvm::IRBuilder<> &Builder,llvm::Module *M,ParallelRegion *Region) {

  llvm::GlobalVariable *LocalSizeXPtr = llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal("_local_size_x", ST));
  llvm::GlobalVariable *LocalSizeYPtr = llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal("_local_size_y", ST));

  assert(LocalSizeXPtr != NULL && LocalSizeYPtr != NULL);

  llvm::LoadInst *LoadX = Builder.CreateLoad(ST, LocalSizeXPtr, "ls_x");
  llvm::LoadInst *LoadY = Builder.CreateLoad(ST, LocalSizeYPtr, "ls_y");

  
  llvm::Value* LocalSizeXTimesY = Builder.CreateBinOp(llvm::Instruction::Mul, LoadX, LoadY, "ls_xy");

  llvm::Value *ZPart =Builder.CreateBinOp(llvm::Instruction::Mul, LocalSizeXTimesY,Region->getOrCreateIDLoad(LID_G_NAME(2)), "tmp");

  llvm::Value *YPart = Builder.CreateBinOp(llvm::Instruction::Mul, LoadX,Region->getOrCreateIDLoad(LID_G_NAME(1)), "ls_x_y");

  llvm::Value* ZYSum = Builder.CreateBinOp(llvm::Instruction::Add, ZPart, YPart,"zy_sum");

  return Builder.CreateBinOp(llvm::Instruction::Add, ZYSum,Region->getOrCreateIDLoad(LID_G_NAME(0)),"linear_xyz_idx");
}


 

 
ParallelRegion *SimpleFallbackImpl::regionOfBlock(llvm::BasicBlock *BB) {
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *PRegion = (*PRI);
    if (PRegion->hasBlock(BB))
      return PRegion;
  }
  return nullptr;
}




 


bool SimpleFallbackImpl::runOnFunction(llvm::Function &Func) {

    M = Func.getParent();
    F = &Func;

    M->dump();

    Initialize(llvm::cast<Kernel>(&Func));

    // This will add on module level:
    //@_global_id_x = external global i64
    //@_global_id_y = external global i64
    //@_global_id_z = external global i64

    GlobalIdIterators = {
    llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(0), ST)),
    llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(1), ST)),
    llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(2), ST))};

    M->dump();

    TempInstructionIndex = 0;

    //bool Changed = processFunction(Func);    

    //return llvm::PreservedAnalyses::none();
    return true;

}

llvm::PreservedAnalyses SimpleFallback::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
    
    // We only want to process kernel functions
    if (!isKernelToProcess(F)){
        return llvm::PreservedAnalyses::all();
    }

    
    WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;

    if (WIH != WorkitemHandlerType::FALLBACK)
    {
        return llvm::PreservedAnalyses::all();
    }
    
    if(WIH == WorkitemHandlerType::FALLBACK){
        std::cout << "WIH  is of type FALLBACK" << std::endl;
    }

    llvm::errs() << F.getName() << "\n";

    for (auto &BasicBlock : F) {
        for (auto &Instr : BasicBlock) {
            
            // Check for function calls
            if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(&Instr)) {
                
                llvm::Function *calledFunc = callInst->getCalledFunction();

                if(calledFunc) {
                    llvm::errs() << "Found a call to: " << calledFunc->getName().str() << "\n";
                } 

                //llvm::errs() << "Found a call to: " << callInst->getName().str() << "\n";
                
                /* if (callInst->getCalledFunction()->getName() == "__switch_to_work_item") {
                    llvm::IRBuilder<> builder(callInst);

                    // Context save (example)
                    llvm::Value *contextSave = builder.CreateAlloca(llvm::Type::getInt32Ty(Func.getContext()), nullptr, "context_save");
                    builder.CreateStore(callInst->getOperand(0), contextSave);

                    // Context restore (example)
                    llvm::Value *contextRestore = builder.CreateLoad(contextSave, "context_restore");

                    // Remove the original function call
                    callInst->eraseFromParent();
                } */
            }
        }
    }
        

    



    auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
    auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
    auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
    auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);

    // Not sure what these do
    llvm::PreservedAnalyses PAChanged = llvm::PreservedAnalyses::none();
    PAChanged.preserve<VariableUniformityAnalysis>();
    PAChanged.preserve<WorkitemHandlerChooser>();

    

    SimpleFallbackImpl WIL(DT, LI, PDT, VUA);

    bool ret_val = WIL.runOnFunction(F);
    

   

    //return ret_val ? PAChanged : llvm::PreservedAnalyses::all();
    
    return llvm::PreservedAnalyses::all();
    

}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);
}