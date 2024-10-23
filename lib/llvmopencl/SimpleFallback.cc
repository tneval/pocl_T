#include "LLVMUtils.h"
#include "SimpleFallback.h"

#define PASS_NAME "simplefallback"
#define PASS_CLASS pocl::SimpleFallback
#define PASS_DESC "Simple and robust work group function generator"


namespace pocl{
/* 
class SimpleFallBackImpl : public pocl::WorkitemHandler{

public:


private:


};
 */



llvm::PreservedAnalyses SimpleFallback::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  
    llvm::errs() << F.getName() << "\n";
    return llvm::PreservedAnalyses::all();

}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);
}