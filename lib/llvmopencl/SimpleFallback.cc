#include "LLVMUtils.h"
#include "SimpleFallback.h"
#include "WorkitemHandlerChooser.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "llvm/IR/IRBuilder.h"
#include "DebugHelpers.h"
#include "KernelCompilerUtils.h"

#include <llvm/IR/Verifier.h>

#include "Barrier.h"
#include "SubgroupBarrier.h"

#include "pocl_llvm_api.h"

#include <iostream>

#define PASS_NAME "simplefallback"
#define PASS_CLASS pocl::SimpleFallback
#define PASS_DESC "Simple and robust work group function generator"

//#define DBG

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
  //llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr, size_t Dim) override;


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

    void initializeLocalIds(llvm::BasicBlock *Entry, llvm::IRBuilder<> *bldr);

    void initializeGlobalIterators();

    VariableUniformityAnalysisResult &VUA;
    
    StrInstructionMap ContextArrays;

    std::array<llvm::GlobalVariable *, 3> LocalIdIterators;
    std::array<llvm::GlobalVariable *, 3> LocalSizeIterators;
    std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;
    std::array<llvm::GlobalVariable *, 3> GroupIdIterators;
    std::array<llvm::Value *, 3> LocalSizeValues;
    llvm::ConstantInt* sgSize;

    size_t TempInstructionIndex;

    // An alloca in the kernel which stores the first iteration to execute
    // in the inner (dimension 0) loop. This is set to 1 in an peeled iteration
    // to skip the 0, 0, 0 iteration in the loops.
    llvm::Value *LocalIdXFirstVar;

    std::map<llvm::Instruction *, unsigned> TempInstructionIds;

    bool shouldNotBeContextSaved(llvm::Instruction *Instr);

    llvm::Instruction *addContextRestore(llvm::Value *Val, llvm::AllocaInst *AllocaI,llvm::Type *LoadInstType, bool PaddingWasAdded,llvm::Instruction *Before = nullptr, bool isAlloca = false);

    llvm::AllocaInst *getContextArray(llvm::Instruction *Inst,bool &PoclWrapperStructAdded);

    std::vector<llvm::Instruction*> contextVars;
    std::vector<llvm::AllocaInst*> contextAllocas;

    void identifyContextVars();

    void allocateContextVars();
    void addSave();
    llvm::GetElementPtrInst* getGEP(llvm::AllocaInst *CtxArrayAlloca,llvm::Instruction *Before,bool AlignPadding);

    llvm::AllocaInst *allocateJumpStorage(llvm::IRBuilder<> &builder);

    llvm::Value *getNumberOfWIs(llvm::IRBuilder<> &Builder);

};


// Get the linear wi ID
llvm::Value *SimpleFallbackImpl::getLinearWIIndexInRegion(llvm::Instruction* Instr) {

    assert(LocalSizeIterators[0] != NULL && LocalSizeIterators[1] != NULL);

    llvm::IRBuilder<> Builder(Instr);
    
  /* Form linear index from xyz coordinates:
       local_size_x * local_size_y * local_id_z  (z dimension)
     + local_size_x * local_id_y                 (y dimension)
     + local_id_x                                (x dimension)
  */

    llvm::LoadInst *LoadXSize = Builder.CreateLoad(ST, LocalSizeIterators[0], "ls_x");
    llvm::LoadInst *LoadYSize = Builder.CreateLoad(ST, LocalSizeIterators[1], "ls_y");

    llvm::LoadInst *LoadXId = Builder.CreateLoad(ST, LocalIdIterators[0], "id_x");
    llvm::LoadInst *LoadYId = Builder.CreateLoad(ST, LocalIdIterators[1], "id_y");
    llvm::LoadInst *LoadZId = Builder.CreateLoad(ST, LocalIdIterators[2], "id_z");

    llvm::Value* LocalSizeXTimesY =
    Builder.CreateBinOp(llvm::Instruction::Mul, LoadXSize, LoadYSize, "ls_xy");

    llvm::Value *ZPart =
      Builder.CreateBinOp(llvm::Instruction::Mul, LocalSizeXTimesY, LoadZId, "tmp");

    llvm::Value *YPart =
      Builder.CreateBinOp(llvm::Instruction::Mul, LoadXSize, LoadYId, "ls_x_y");

    llvm::Value* ZYSum =
        Builder.CreateBinOp(llvm::Instruction::Add, ZPart, YPart,
                        "zy_sum");

    return Builder.CreateBinOp(llvm::Instruction::Add, ZYSum,
                             LoadXId,"linear_xyz_idx");
}


///////////////////////////////////////////////////////////////////
// THE NEW CONTEXT SAVE

void SimpleFallbackImpl::identifyContextVars()
{
    int added = 0;

    for (auto &BB : *F) {
        for (auto &Instr : BB) {
            
            if (shouldNotBeContextSaved(&Instr)){
                continue;
            }

            for (llvm::Instruction::use_iterator UI = Instr.use_begin(),UE = Instr.use_end();UI != UE; ++UI) {
            
                llvm::Instruction *User = llvm::dyn_cast<llvm::Instruction>(UI->getUser());

                if (User == NULL)
                    continue;

                // User is in same block = NO CONTEXT SAVE needed
                llvm::BasicBlock* currentBlock = Instr.getParent();

                llvm::BasicBlock* userBlock = User->getParent();

                if(currentBlock == userBlock){
                    continue;
                }

                contextVars.push_back(&Instr);
                added++;
                break;
            }
            
        }
    }
} // identifyContextVars()

void SimpleFallbackImpl::allocateContextVars()
{
    for(auto &instr : contextVars){
        // Allocate the context data array for the variable.
        bool PaddingAdded = false;
        llvm::AllocaInst *Alloca = getContextArray(instr, PaddingAdded);

        contextAllocas.push_back(Alloca);
    }
}

void SimpleFallbackImpl::addSave()
{

    llvm::Type *ST = llvm::Type::getInt64Ty(M->getContext());
   
    for(int i = 0; i< contextVars.size(); i++){


        llvm::BasicBlock::iterator definition = (llvm::dyn_cast<llvm::Instruction>(contextVars[i]))->getIterator();
        ++definition;
        while (llvm::isa<llvm::PHINode>(definition)) ++definition;

        // TO CLEAN: Refactor by calling CreateContextArrayGEP.
        llvm::IRBuilder<> ctxSaveBuilder(&*definition);

        llvm::Value *local_x = ctxSaveBuilder.CreateLoad(ST,LocalIdIterators[0],"local_x");
        llvm::Value *local_y = ctxSaveBuilder.CreateLoad(ST,LocalIdIterators[1],"local_y");
        llvm::Value *local_z = ctxSaveBuilder.CreateLoad(ST,LocalIdIterators[2],"local_z");

        // These are the indices for context arrays
        std::vector<llvm::Value *> gepArgs;

        gepArgs.push_back(llvm::ConstantInt::get(ST, 0));

        gepArgs.push_back(local_z);
        gepArgs.push_back(local_y);
        gepArgs.push_back(local_x);

        llvm::Instruction* TheStore = ctxSaveBuilder.CreateStore(contextVars[i],ctxSaveBuilder.CreateGEP(contextAllocas[i]->getAllocatedType(), contextAllocas[i], gepArgs));

        InstructionVec Uses;

        for (llvm::Instruction::use_iterator UI = contextVars[i]->use_begin(), UE = contextVars[i]->use_end();UI != UE; ++UI) {

            llvm::Instruction *User = llvm::cast<llvm::Instruction>(UI->getUser());

            if (User == NULL || User == TheStore) continue;

            Uses.push_back(User);
        }

        for (InstructionVec::iterator I = Uses.begin(); I != Uses.end(); ++I) {
    
            llvm::Instruction *UserI = *I;
                    
            llvm::Instruction *ContextRestoreLocation = UserI;

            llvm::Value* LoadedValue = addContextRestore(UserI, contextAllocas[i], contextVars[i]->getType(), false, ContextRestoreLocation, llvm::isa<llvm::AllocaInst>(contextVars[i]));
            
            UserI->replaceUsesOfWith(contextVars[i], LoadedValue);
        
        }
    }
}


llvm::GetElementPtrInst* SimpleFallbackImpl::getGEP(llvm::AllocaInst *CtxArrayAlloca,llvm::Instruction *Before,bool AlignPadding)
{

    llvm::Type *ST = llvm::Type::getInt64Ty(M->getContext());

    llvm::IRBuilder<> ctxLoadBuilder(Before);

    llvm::Value *local_x = ctxLoadBuilder.CreateLoad(ST,LocalIdIterators[0],"local_x");
    llvm::Value *local_y = ctxLoadBuilder.CreateLoad(ST,LocalIdIterators[1],"local_y");
    llvm::Value *local_z = ctxLoadBuilder.CreateLoad(ST,LocalIdIterators[2],"local_z");


    std::vector<llvm::Value *> GEPArgs;
    
    GEPArgs.push_back(llvm::ConstantInt::get(ST, 0));
    GEPArgs.push_back(local_z);
    GEPArgs.push_back(local_y);
    GEPArgs.push_back(local_x);

    if (AlignPadding)
        GEPArgs.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(CtxArrayAlloca->getContext()), 0));
    

    //std::cout << "inserting GEP (getGEP)\n";
    llvm::GetElementPtrInst *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(ctxLoadBuilder.CreateGEP(
      CtxArrayAlloca->getAllocatedType(), CtxArrayAlloca, GEPArgs));
    return GEP;

}

////////////////////////////////////////////////////////////////////

llvm::AllocaInst *SimpleFallbackImpl::getContextArray(llvm::Instruction *Inst,bool &PaddingAdded) {
    
    PaddingAdded = false;

    std::ostringstream Var;
    Var << ".";

    if (std::string(Inst->getName().str()) != "") {
        Var << Inst->getName().str();
    } else if (TempInstructionIds.find(Inst) != TempInstructionIds.end()) {
        Var << TempInstructionIds[Inst];
    } else {
        // Unnamed temp instructions need a name generated for the context array.
        // Create one using a running integer.
        TempInstructionIds[Inst] = TempInstructionIndex++;
        Var << TempInstructionIds[Inst];
    }

    Var << ".pocl_context";
    std::string CArrayName = Var.str();

    if (ContextArrays.find(CArrayName) != ContextArrays.end())
        return ContextArrays[CArrayName];

    llvm::BasicBlock &Entry = K->getEntryBlock();
    return ContextArrays[CArrayName] = createAlignedAndPaddedContextAlloca(
                Inst, &*(Entry.getFirstInsertionPt()), CArrayName, PaddingAdded);
}


llvm::Instruction *SimpleFallbackImpl::addContextRestore(
    llvm::Value *Val, llvm::AllocaInst *AllocaI, llvm::Type *LoadInstType,
    bool PaddingWasAdded, llvm::Instruction *Before, bool isAlloca) {

    assert(Before != nullptr);

    llvm::Instruction* GEP = getGEP(AllocaI, Before, PaddingWasAdded);

    if (isAlloca) {
        /* In case the context saved instruction was an alloca, we created a
        context array with pointed-to elements, and now want to return a
        pointer to the elements to emulate the original alloca. */
        return GEP;
    }

    llvm::IRBuilder<> Builder(Before);
    return Builder.CreateLoad(LoadInstType, GEP);
}


// DECIDE WHETHER VARIABLE SHOULD BE CONTEXT SAVED
bool SimpleFallbackImpl::shouldNotBeContextSaved(llvm::Instruction *Instr) {

    if (llvm::isa<llvm::BranchInst>(Instr)){

        return true;
    } 

    return false;

    // The local memory allocation call is uniform, the same pointer to the
    // work-group shared memory area is returned to all work-items. It must
    // not be replicated.
    if (llvm::isa<llvm::CallInst>(Instr)) {
        llvm::Function *F = llvm::cast<llvm::CallInst>(Instr)->getCalledFunction();
        if (F && (F == LocalMemAllocaFuncDecl || F == WorkGroupAllocaFuncDecl))
        return true;
    }

    //return false;

    llvm::LoadInst *Load = llvm::dyn_cast<llvm::LoadInst>(Instr);
    if (Load != NULL && (Load->getPointerOperand() == LocalIdGlobals[0] ||
                        Load->getPointerOperand() == LocalIdGlobals[1] ||
                        Load->getPointerOperand() == LocalIdGlobals[2] ||
                        Load->getPointerOperand() == GlobalIdGlobals[0] ||
                        Load->getPointerOperand() == GlobalIdGlobals[1] ||
                        Load->getPointerOperand() == GlobalIdGlobals[2])){

        return true;                       

    }
    
    
    if (!VUA.shouldBePrivatized(Instr->getParent()->getParent(), Instr)) {


        return true;
    }

    return false;
}


// Initialize local ids as zero
void SimpleFallbackImpl::initializeLocalIds(llvm::BasicBlock *Entry, llvm::IRBuilder<> *builder) {

    llvm::GlobalVariable *GVX = LocalIdIterators[0];
    if (GVX != NULL)
        builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GVX);

    llvm::GlobalVariable *GVY = LocalIdIterators[1];
    if (GVY != NULL)
        builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GVY);

    llvm::GlobalVariable *GVZ = LocalIdIterators[2];
    if (GVZ != NULL)
        builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GVZ);
}

// Cast values stored in WorkitemHandler to global variable pointers.
void SimpleFallbackImpl::initializeGlobalIterators(){

    for(int i = 0; i < 3; i++){
        // _local_id_xyz
        LocalIdIterators[i] = 
        llvm::cast<llvm::GlobalVariable>(LocalIdGlobals[i]);

        // _local_size_xyz
        LocalSizeIterators[i] = 
        llvm::cast<llvm::GlobalVariable>(LocalSizeGlobals[i]);

        // _global_id_xyz
        GlobalIdIterators[i] =
        llvm::cast<llvm::GlobalVariable>(GlobalIdGlobals[i]);

        // _group_id_xyz
        GroupIdIterators[i] =
        llvm::cast<llvm::GlobalVariable>(GroupIdGlobals[i]);
    }
}

// Calculate the number of work items in wg.
llvm::Value *SimpleFallbackImpl::getNumberOfWIs(llvm::IRBuilder<> &Builder){

    llvm::Value *nWI;

    if(WGDynamicLocalSize){

        llvm::Instruction *loadX = Builder.CreateLoad(ST, LocalSizeGlobals[0]);
        llvm::Instruction *loadY = Builder.CreateLoad(ST, LocalSizeGlobals[1]);
        llvm::Instruction *loadZ = Builder.CreateLoad(ST, LocalSizeGlobals[2]);

        LocalSizeValues[0] = loadX;
        LocalSizeValues[1] = loadY;
        LocalSizeValues[2] = loadZ;

        llvm::Value *xy = Builder.CreateBinOp(llvm::Instruction::Mul, loadX, loadY);
        llvm::Value *xyz = Builder.CreateBinOp(llvm::Instruction::Mul, xy, loadZ);
        xyz->setName("nWI");

        nWI = xyz;

    }else{

        LocalSizeValues[0] = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeX, false);
        LocalSizeValues[1] = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeY, false);
        LocalSizeValues[2] = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeZ, false);
        nWI = llvm::ConstantInt::get(ST,WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ);
        
    }    
    
    return nWI;
}


llvm::AllocaInst *SimpleFallbackImpl::allocateJumpStorage(llvm::IRBuilder<> &entryBlockBuilder){

    // Create stack storage for index variable that directs wi to correct
    // next block after jumping from dispatcher.
    llvm::Value *dummyValue = entryBlockBuilder.CreateLoad(ST, LocalIdIterators[0], "dummy");
    llvm::LoadInst *dummyInst = llvm::dyn_cast<llvm::LoadInst>(dummyValue);
    bool PaddingAdded = false;

    // For storing the block ids, create alloca with dynamic size, using context save machinery.
    // For this, we need to use proper instruction as a parameter. First instruction of entry block is guaranteed to be like that.
    llvm::AllocaInst *nextExitBlockArray = createAlignedAndPaddedContextAlloca(dummyInst,dummyInst,"jump_indices", PaddingAdded);
    
    dummyInst->eraseFromParent();


    llvm::Value *nWI = getNumberOfWIs(entryBlockBuilder);

   

    //llvm::Value *size = entryBlockBuilder.getInt64(2);
    llvm::Value *zero = entryBlockBuilder.getInt8(0);
    //llvm::Value *align = entryBlockBuilder.getInt32(4);
    llvm::MaybeAlign maybeAlign(nextExitBlockArray->getAlign().value());
    entryBlockBuilder.CreateMemSet(nextExitBlockArray, zero, nWI, maybeAlign);

   
    return nextExitBlockArray;

}



bool SimpleFallbackImpl::runOnFunction(llvm::Function &Func) {

    M = Func.getParent();
    F = &Func;

    Initialize(llvm::cast<Kernel>(&Func));

    // Initialize pointers to global variables:
    initializeGlobalIterators();

    // Expand workitem function calls.
    handleWorkitemFunctions();

    TempInstructionIndex = 0;

    /////////////////////////
    // Context save/restore
    //identifyContextVars();
    //allocateContextVars();
    //addSave();
    /////////////////////////

    /////////////////////////////////////////////////////////
    // Begin processing actual function

    llvm::BasicBlock *EntryBlock = &Func.getEntryBlock();

    llvm::IRBuilder<> entryBlockBuilder(&*(Func.getEntryBlock().getFirstInsertionPt()));

    // Initialize local ids to 0
    initializeLocalIds(EntryBlock, &entryBlockBuilder);

    // Array for exit block indices
    llvm::Type *Int64Ty = llvm::Type::getInt64Ty(M->getContext());

    // Alloca for storing the index of "next" block after dispatcher for each WI
    // Also initializes to zero
    llvm::AllocaInst *nextJumpIndices = allocateJumpStorage(entryBlockBuilder);
    
    
     
    ////////////////////////////////////////////////////////////////////////////////////////////
    // Init call: Instead of relying global variables, use the metadata

     // Create function call to __pocl_sched_init
    llvm::Function *schedFuncI = M->getFunction("__pocl_sched_init");
    llvm::outs() << "Function signature: " << *(schedFuncI->getType()) << "\n";
    
    if (llvm::MDNode *SGSizeMD = F->getMetadata("intel_reqd_sub_group_size")) {
        // Use the constant from the metadata.
        llvm::ConstantAsMetadata *ConstMD = llvm::cast<llvm::ConstantAsMetadata>(SGSizeMD->getOperand(0));
        sgSize = llvm::cast<llvm::ConstantInt>(ConstMD->getValue());    
    }else{
        sgSize = llvm::cast<llvm::ConstantInt>(LocalSizeValues[0]);
    }
    
    // This will pass the sg size and local size to init function.
    entryBlockBuilder.CreateCall(schedFuncI, {LocalSizeValues[0], LocalSizeValues[1], LocalSizeValues[2], sgSize});
    
    ////////////////////////////////////////////////////////////////////////////////////////////
    Func.dump();

    return true;
    // Store exit blocks after barriers
    std::vector<llvm::BasicBlock*> barrierExits;

    llvm::BasicBlock *currBlock = EntryBlock;


    // Store barrier blocks
    std::vector<llvm::BasicBlock*> barrierBlocks;

   
    llvm::Value *zeroIndex = llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()), 0);

    for(auto &Block : Func){

        // Save blocks that have barriers or sg barriers
        if(Barrier::hasBarrier(&Block) || SubgroupBarrier::hasSGBarrier(&Block)){
            barrierBlocks.push_back(&Block);
        }
        
    }

    //std::cerr << "Num of barriers : " << barrierBlocks.size() << std::endl;
    // Store pointer to old exit here
    llvm::BasicBlock* oldExitBlock = nullptr;


    // Create new block for dispatcher; dispathcer block manipulation is done later. Need this for reference below
    llvm::BasicBlock *dispatcherBlock = llvm::BasicBlock::Create(F->getContext(), "dispatcher", F);
/* 
    // Modify the barrier blocks
    for(auto &BBlock : barrierBlocks){


        // This is the entry barrier block
        // Is this bad way to check entry barrier?
        if(BBlock == &Func.getEntryBlock()){

            //std::cout << "ENTRY" << std::endl;

            // In some cases there are none
            if(BBlock->getTerminator()->getNumSuccessors()>0){
                barrierExits.push_back(BBlock->getTerminator()->getSuccessor(0));
                //std::cout << BBlock->getTerminator()->getSuccessor(0)->getName().str()<< std::endl;
            }

            //std::cout << "BARRIER-entry: " << BBlock->getName().str() << std::endl;

            // Add branch to dispatcher
            llvm::IRBuilder<> entryBuilder(BBlock->getTerminator());
            entryBuilder.CreateBr(dispatcherBlock);

            // This removes the old branch
            BBlock->getTerminator()->eraseFromParent();

        // This is the "return" block
        }else if(BBlock->getTerminator()->getNumSuccessors() == 0){
            //std::cout << "BARRIER: " << BBlock->getName().str() << std::endl;
            
            // Create new kernel exit where we come out as "one"
            llvm::BasicBlock *newExitBlock = llvm::BasicBlock::Create(F->getContext(), "exit_block", F);
            
            // This will be the last jump where we exit from the kernel
            barrierExits.push_back(newExitBlock);


            // Handle for old return block
            llvm::IRBuilder<> oldExitBlockBuilder(BBlock->getTerminator());

            llvm::Value *local_z = oldExitBlockBuilder.CreateLoad(ST,LocalIdIterators[2],"local_z");
            llvm::Value *local_y = oldExitBlockBuilder.CreateLoad(ST,LocalIdIterators[1],"local_y");
            llvm::Value *local_x = oldExitBlockBuilder.CreateLoad(ST,LocalIdIterators[0],"local_x");
            
            //llvm::Value *next_block_ptr = oldExitBlockBuilder.CreateGEP(exitBlockIdxs, nextExitBlockArray, {zeroIndex, local_x}, "exit_block_ptr");

            llvm::Value *next_block_ptr = oldExitBlockBuilder.CreateGEP(ContextArrayType,nextExitBlockArray, {zeroIndex, local_z, local_y, local_x}, "exit_block_ptr");

            llvm::Value *next_block_idx = llvm::ConstantInt::get(Int64Ty, barrierExits.size()-1);
            oldExitBlockBuilder.CreateStore(next_block_idx, next_block_ptr);

            llvm::Function *barrierReached = M->getFunction("__pocl_barrier_reached");           
            oldExitBlockBuilder.CreateCall(barrierReached,{local_x, local_y, local_z});
            
            // Add branch to dispatcher
            oldExitBlockBuilder.CreateBr(dispatcherBlock);

            // This removes the "old" ret void
            BBlock->getTerminator()->eraseFromParent();

            // Add ret void instr to new return block
            llvm::IRBuilder<> newExitBlockBuilder(newExitBlock);


            llvm::Function *schedClean = M->getFunction("__pocl_sched_clean");

            newExitBlockBuilder.CreateCall(schedClean);

            newExitBlockBuilder.CreateRetVoid();

           
        // These are "Explicit" barriers
        }else{
            
            
#ifdef DBG
            if(Barrier::hasBarrier(BBlock)){
                std::cout << "BARRIER: " << BBlock->getName().str() << std::endl;
            }else if(SubgroupBarrier::hasSGBarrier(BBlock)){
                std::cout << "SG BARRIER:" << BBlock->getName().str() << std::endl;
            }
#endif

             // This is the next exit block
            barrierExits.push_back(BBlock->getTerminator()->getSuccessor(0));
            //std::cout << BBlock->getTerminator()->getSuccessor(0)->getName().str()<< std::endl;

            // These contain either barriers or sg barriers, but not the "entry" or "exit" barrier
            llvm::IRBuilder<> barrierBlockBuilder(BBlock->getTerminator());
            

            llvm::Value *local_z = barrierBlockBuilder.CreateLoad(ST,LocalIdIterators[2],"local_z");
            llvm::Value *local_y = barrierBlockBuilder.CreateLoad(ST,LocalIdIterators[1],"local_y");
            llvm::Value *local_x = barrierBlockBuilder.CreateLoad(ST,LocalIdIterators[0],"local_x");
            

            llvm::Value *next_block_ptr = barrierBlockBuilder.CreateGEP(ContextArrayType,nextExitBlockArray, {zeroIndex, local_z, local_y, local_x}, "exit_block_ptr");
            
            //llvm::Value *next_block_ptr = barrierBlockBuilder.CreateGEP(exitBlockIdxs, nextExitBlockArray, {zeroIndex, local_x}, "exit_block_ptr");

            llvm::Value *next_block_idx = llvm::ConstantInt::get(Int64Ty, barrierExits.size()-1);
            barrierBlockBuilder.CreateStore(next_block_idx, next_block_ptr);

            // Register barrier entry
            if(Barrier::hasBarrier(BBlock)){
                llvm::Function *barrierReached = M->getFunction("__pocl_barrier_reached");           
                barrierBlockBuilder.CreateCall(barrierReached,{local_x, local_y, local_z});
            }else if(SubgroupBarrier::hasSGBarrier(BBlock)){
                llvm::Function *sgbarrierReached = M->getFunction("__pocl_sg_barrier_reached");
                barrierBlockBuilder.CreateCall(sgbarrierReached,{local_x, local_y, local_z});
            }

            // Add branch to dispatcher
            barrierBlockBuilder.CreateBr(dispatcherBlock);

            // This removes the old branch
            BBlock->getTerminator()->eraseFromParent();

        }
    } */
    /* 
    ////////////////////////////////////////////////////////////
    // Actual dispatcher implementation

    // Build the dispatcher block
    llvm::IRBuilder<> bBuilder(dispatcherBlock);


    // Create function call to __pocl_sched_work_item to retrieve next WI id
    llvm::Function *schedFunc = M->getFunction("__pocl_sched_work_item");

    // Retrieve the return value, i.e. WI id
    llvm::Value *linearWI = bBuilder.CreateCall(schedFunc);
    linearWI->setName("next_linear_wi");


    llvm::Value *xSize = llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()),WGLocalSizeX);
    llvm::Value *ySize = llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()),WGLocalSizeY);
    llvm::Value *zSize = llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()),WGLocalSizeZ);

    // X 
    llvm::Value *loc_x = bBuilder.CreateBinOp(llvm::Instruction::BinaryOps::SRem, linearWI ,xSize, "loc_id_x");

    // Y
    unsigned int mult_xy_sizes = WGLocalSizeX*WGLocalSizeY;
    llvm::Value *xy_mult = llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()),mult_xy_sizes);
    llvm::Value *loc_y_tmp = bBuilder.CreateBinOp(llvm::Instruction::BinaryOps::SRem, linearWI,xy_mult, "loc_id_y_tmp");
    llvm::Value *loc_y = bBuilder.CreateBinOp(llvm::Instruction::UDiv, loc_y_tmp, xSize, "loc_id_y");

    // Z
    llvm::Value *loc_z = bBuilder.CreateBinOp(llvm::Instruction::UDiv, linearWI, xy_mult, "loc_id_z");


    // Store new ids
    bBuilder.CreateStore(loc_x, LocalIdIterators[0]);
    bBuilder.CreateStore(loc_y, LocalIdIterators[1]);
    bBuilder.CreateStore(loc_z, LocalIdIterators[2]);

    llvm::Value* x_gid = bBuilder.CreateLoad(ST, GroupIdGlobals[0], "group_id_x");
    llvm::Value* y_gid = bBuilder.CreateLoad(ST, GroupIdGlobals[1], "group_id_y");
    llvm::Value* z_gid = bBuilder.CreateLoad(ST, GroupIdGlobals[2], "group_id_z");


    llvm::Value* multX = bBuilder.CreateMul(llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeX, false), x_gid, "mulx");
    llvm::Value* gid_x = bBuilder.CreateAdd(multX, loc_x, "gid_x");


    llvm::Value* multY = bBuilder.CreateMul(llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeY, false), y_gid, "muly");
    llvm::Value* gid_y = bBuilder.CreateAdd(multY, loc_y, "gid_y");

    llvm::Value* multZ = bBuilder.CreateMul(llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeZ, false), z_gid, "mulz");
    llvm::Value* gid_z = bBuilder.CreateAdd(multZ, loc_z, "gid_z");


    // These will store global ids
    bBuilder.CreateStore(gid_x, GlobalIdIterators[0]);

    bBuilder.CreateStore(gid_y, GlobalIdIterators[1]);

    bBuilder.CreateStore(gid_z, GlobalIdIterators[2]);

    //global_id(dim)=global_offset(dim)+local_work_size(dim)×group_id(dim)+local_id(dim)

    // Pointer to exit index array
    //llvm::Value *next_block_ptr = bBuilder.CreateGEP(exitBlockIdxs, nextExitBlockArray, {zeroIndex, nextWI}, "exit_block_ptr");
    llvm::Value *next_block_ptr = bBuilder.CreateGEP(ContextArrayType,nextExitBlockArray, {zeroIndex, loc_z, loc_y, loc_x}, "exit_block_ptr");

    // Retrieve exit index based for current local_id_x
    llvm::Value *loadedValue = bBuilder.CreateLoad(bBuilder.getInt64Ty(), next_block_ptr, "next_exit_block");
    
    llvm::Function *nextI = M->getFunction("__pocl_next_jump");
    bBuilder.CreateCall(nextI, {loadedValue});
    
    
    // Create switch statement for exit blocks
    if(barrierExits.size() > 0){

        llvm::ConstantInt *zero = llvm::ConstantInt::get(bBuilder.getInt64Ty(),0);
        llvm::SwitchInst *switchInst = bBuilder.CreateSwitch(loadedValue, barrierExits[0]);
    

        for(int i = 1; i < barrierExits.size(); i++){
            
            llvm::ConstantInt *caseValue = llvm::ConstantInt::get(bBuilder.getInt64Ty(), i);
            
            switchInst->addCase(caseValue, barrierExits[i]);
        }

    }
 */
    // End of dispatcher manipulation
    ////////////////////////////////////////////////////////////

   /*  std::string Log;
    llvm::raw_string_ostream OS(Log);
    bool BrokenDebugInfo = false;
 
    llvm::verifyModule(*M, &OS, &BrokenDebugInfo);
    if (!Log.empty()) {
        std::cerr << "Module verification errors:\n" << Log << std::endl;
    }
  
    llvm::verifyFunction(Func);

    handleLocalMemAllocas();

    // added 5.12; trying to fix domination issue 
    fixUndominatedVariableUses(DT, Func); */

    return true;

}

llvm::PreservedAnalyses SimpleFallback::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
    
    // We only want to process kernel functions
    if (!isKernelToProcess(F)){
        return llvm::PreservedAnalyses::all();
    }

    F.dump();

    
    WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;

    if (WIH != WorkitemHandlerType::FALLBACK)
    {
        return llvm::PreservedAnalyses::all();
    }
    

#ifdef DBG

    if(WIH == WorkitemHandlerType::FALLBACK){
        std::cout << "WIH  is of type FALLBACK" << std::endl;
    }
    llvm::errs() << F.getName() << "\n";
#endif

    //F.dump();

    //dumpCFG(F, F.getName().str() + "_before_fallback.dot", nullptr,nullptr);

    auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
    auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
    auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
    auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);

    // Not sure what these do
    llvm::PreservedAnalyses PAChanged = llvm::PreservedAnalyses::none();
    PAChanged.preserve<VariableUniformityAnalysis>();
    PAChanged.preserve<WorkitemHandlerChooser>();

    SimpleFallbackImpl WIL(DT, LI, PDT, VUA);


    //dumpCFG(F, F.getName().str() + "_before_fallback.dot", nullptr,nullptr);

#ifdef DBG 
    F.dump();
#endif

    bool ret_val = WIL.runOnFunction(F);

#ifdef DBG 
    F.dump();
#endif
    //F.dump();
    //dumpCFG(F, F.getName().str() + "AFTER_FALLBACK.dot", nullptr,nullptr);

    //return ret_val ? PAChanged : llvm::PreservedAnalyses::all();
    
    return llvm::PreservedAnalyses::all();
    
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl