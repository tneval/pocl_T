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


namespace pocl{


static constexpr const char LocalIdGlobalNameX[] = "_local_id_x";
static constexpr const char LocalIdGlobalNameY[] = "_local_id_y";
static constexpr const char LocalIdGlobalNameZ[] = "_local_id_z";
static constexpr const char NextWI[] = "_next_wi_x";



// Note: this is renamed version of subcfgformation.
// This should initialize local id to 0
void insertLocalIdInit_(llvm::BasicBlock *Entry) {

    llvm::IRBuilder<> Builder(Entry, Entry->getFirstInsertionPt());

    llvm::Module *M = Entry->getParent()->getParent();

    unsigned long address_bits;
    getModuleIntMetadata(*M, "device_address_bits", address_bits);

    llvm::Type *SizeT = llvm::IntegerType::get(M->getContext(), address_bits);

    llvm::GlobalVariable *GVX = M->getGlobalVariable(LocalIdGlobalNameX);
    if (GVX != NULL)
        Builder.CreateStore(llvm::ConstantInt::getNullValue(SizeT), GVX);

    llvm::GlobalVariable *GVY = M->getGlobalVariable(LocalIdGlobalNameY);
    if (GVY != NULL)
        Builder.CreateStore(llvm::ConstantInt::getNullValue(SizeT), GVY);

    llvm::GlobalVariable *GVZ = M->getGlobalVariable(LocalIdGlobalNameZ);
    if (GVZ != NULL)
        Builder.CreateStore(llvm::ConstantInt::getNullValue(SizeT), GVZ);

 
}



class SimpleFallbackImpl : public pocl::WorkitemHandler{

public:
    SimpleFallbackImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                    llvm::PostDominatorTree &PDT,
                    VariableUniformityAnalysisResult &VUA)
      : WorkitemHandler(), DT(DT), LI(LI), PDT(PDT), VUA(VUA) {}

    virtual bool runOnFunction(llvm::Function &F);


protected:
    //llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr) override;
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

    llvm::BasicBlock* dispatcher;


    ParallelRegion::ParallelRegionVector OriginalParallelRegions;
    
    StrInstructionMap ContextArrays;

    std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;

    size_t TempInstructionIndex;

    // An alloca in the kernel which stores the first iteration to execute
    // in the inner (dimension 0) loop. This is set to 1 in an peeled iteration
    // to skip the 0, 0, 0 iteration in the loops.
    llvm::Value *LocalIdXFirstVar;

    std::map<llvm::Instruction *, unsigned> TempInstructionIds;


    llvm::Instruction *addContextSave(llvm::Instruction *Def, llvm::AllocaInst *AllocaI);


    void releaseParallelRegions();

    void fixMultiRegionVariables(ParallelRegion *region);

    bool shouldNotBeContextSaved(llvm::Instruction *Instr);

    void addContextSaveRestore(llvm::Instruction *instruction);

    llvm::Instruction *addContextRestore(llvm::Value *Val, llvm::AllocaInst *AllocaI,llvm::Type *LoadInstType, bool PaddingWasAdded,llvm::Instruction *Before = nullptr, bool isAlloca = false);


    llvm::AllocaInst *getContextArray(llvm::Instruction *Inst,bool &PoclWrapperStructAdded);

    //bool processFunction(llvm::Function &F);
    void ctxSaveRestore();

    void addCtxSaveRstr(llvm::Instruction *Def);




    std::vector<llvm::Instruction*> contextVars;
    std::vector<llvm::AllocaInst*> contextAllocas;

    void identifyContextVars();

    void allocateContextVars();
    void addSave();
    void addLoad();

    llvm::GetElementPtrInst* getGEP(llvm::AllocaInst *CtxArrayAlloca,llvm::Instruction *Before,bool AlignPadding);


};


///////////////////////////////////////////////////////////////////
// THE NEW CONTEXT SAVE

void SimpleFallbackImpl::identifyContextVars()
{

    //std::cout << "identifyContextVars called\n" << std::endl;

    int added = 0;

    for (auto &BB : *F) {
        for (auto &Instr : BB) {
            
            /* if(added > 4){
                continue;
            } */

            if (shouldNotBeContextSaved(&Instr)){
                continue;
            }

            //std::cout << "Current instr: \n";
            //Instr.print(llvm::outs());
            //std::cout << "\n";
            for (llvm::Instruction::use_iterator UI = Instr.use_begin(),UE = Instr.use_end();UI != UE; ++UI) {
            
                llvm::Instruction *User = llvm::dyn_cast<llvm::Instruction>(UI->getUser());

                if (User == NULL)
                    continue;

                //std::cout << "  User: \n";
                //User->print(llvm::outs());
                //std::cout <<"\n";


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

    /* std::cout << "\nFixing: "<<std::endl;
    for(auto &inst : contextVars){
        inst->print(llvm::outs());
        std::cout << "\n";
    } */


} // identifyContextVars()

void SimpleFallbackImpl::allocateContextVars()
{

    //std::cout << "allocateContextVars called\n";

    for(auto &instr : contextVars){
        // Allocate the context data array for the variable.
        bool PaddingAdded = false;
        llvm::AllocaInst *Alloca = getContextArray(instr, PaddingAdded);

        contextAllocas.push_back(Alloca);
    }
}

void SimpleFallbackImpl::addSave()
{

    /* for(int i = 0; i< contextVars.size(); i++){
        std::cout << "contextAlloca: " << contextAllocas[i]->getName().str() << "\n";
        contextAllocas[i]->print(llvm::outs());
        std::cout << "\n";
    } */
    
    


    // cant insert here due to dominance
    //llvm::IRBuilder<> ctxSaveBuilder(dispatcher, dispatcher->begin());

    llvm::Type *uType = llvm::Type::getInt64Ty(M->getContext());
    llvm::GlobalVariable *localIdXPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_x"));
    llvm::GlobalVariable *localIdYPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_y"));
    llvm::GlobalVariable *localIdZPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_z"));


    for(int i = 0; i< contextVars.size(); i++){


        llvm::BasicBlock::iterator definition = (llvm::dyn_cast<llvm::Instruction>(contextVars[i]))->getIterator();
        ++definition;
        while (llvm::isa<llvm::PHINode>(definition)) ++definition;

        // TO CLEAN: Refactor by calling CreateContextArrayGEP.
        llvm::IRBuilder<> ctxSaveBuilder(&*definition);

        llvm::Value *local_x = ctxSaveBuilder.CreateLoad(uType,localIdXPtr,"local_x");
        llvm::Value *local_y = ctxSaveBuilder.CreateLoad(uType,localIdYPtr,"local_y");
        llvm::Value *local_z = ctxSaveBuilder.CreateLoad(uType,localIdZPtr,"local_z");

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
            
            //std::cout << "Loaded value\n";
            //LoadedValue->print(llvm::outs());
            //std::cout << "\n";
            
            UserI->replaceUsesOfWith(contextVars[i], LoadedValue);
        
        }


    }
    
}




llvm::GetElementPtrInst* SimpleFallbackImpl::getGEP(llvm::AllocaInst *CtxArrayAlloca,llvm::Instruction *Before,bool AlignPadding)
{

    

    llvm::Type *uType = llvm::Type::getInt64Ty(M->getContext());
    llvm::GlobalVariable *localIdXPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_x"));
    llvm::GlobalVariable *localIdYPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_y"));
    llvm::GlobalVariable *localIdZPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_z"));

    //llvm::IRBuilder<> ctxLoadBuilder(dispatcher);

    llvm::IRBuilder<> ctxLoadBuilder(Before);

    llvm::Value *local_x = ctxLoadBuilder.CreateLoad(uType,localIdXPtr,"local_x");
    llvm::Value *local_y = ctxLoadBuilder.CreateLoad(uType,localIdYPtr,"local_y");
    llvm::Value *local_z = ctxLoadBuilder.CreateLoad(uType,localIdZPtr,"local_z");


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





/// Returns the context array (alloca) for the given \param Inst, creates it if
/// not found.
///
/// \param PaddingAdded will be set to true in case a wrapper struct was
/// added for padding in order to enforce proper alignment to the elements of
/// the array. Such padding might be needed to ensure aligned accessed from
/// single work-items accessing aggregates in the context data.
llvm::AllocaInst *SimpleFallbackImpl::getContextArray(llvm::Instruction *Inst,bool &PaddingAdded) {
    
    PaddingAdded = false;


    //std::cout << "getContextArray called\n";

    std::ostringstream Var;
    Var << ".";

    if (std::string(Inst->getName().str()) != "") {
        Var << Inst->getName().str();
        //std::cout << "Instr: " << Inst->getName().str() << std::endl;
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


    //std::cout << "addContextRestore called\n";



    assert(Before != nullptr);

    //llvm::Instruction *GEP = createContextArrayGEP(AllocaI, Before, PaddingWasAdded);


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


    //Instr->print(llvm::outs());

    if (llvm::isa<llvm::BranchInst>(Instr)){

        //llvm::errs()<<"\nReason: branch instruction";
        return true;
    } 

    // The local memory allocation call is uniform, the same pointer to the
    // work-group shared memory area is returned to all work-items. It must
    // not be replicated.
    if (llvm::isa<llvm::CallInst>(Instr)) {
        llvm::Function *F = llvm::cast<llvm::CallInst>(Instr)->getCalledFunction();
        if (F && (F == LocalMemAllocaFuncDecl || F == WorkGroupAllocaFuncDecl))

        //llvm::errs()<<"\nReason: local memory allocation call is uniform";

        return true;
    }

    llvm::LoadInst *Load = llvm::dyn_cast<llvm::LoadInst>(Instr);
    if (Load != NULL && (Load->getPointerOperand() == LocalIdGlobals[0] ||
                        Load->getPointerOperand() == LocalIdGlobals[1] ||
                        Load->getPointerOperand() == LocalIdGlobals[2] ||
                        Load->getPointerOperand() == GlobalIdGlobals[0] ||
                        Load->getPointerOperand() == GlobalIdGlobals[1] ||
                        Load->getPointerOperand() == GlobalIdGlobals[2])){

        //llvm::errs()<<"\nReason: loading generated ids";

        return true;                       

    }

    
    if (!VUA.shouldBePrivatized(Instr->getParent()->getParent(), Instr)) {


        return true;
    }

    return false;
}





void SimpleFallbackImpl::releaseParallelRegions() {
  for (auto PRI = OriginalParallelRegions.begin(),
            PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *P = *PRI;
    delete P;
  }
}
 


bool SimpleFallbackImpl::runOnFunction(llvm::Function &Func) {

    M = Func.getParent();
    F = &Func;


    

    Initialize(llvm::cast<Kernel>(&Func));


        //Func.dump();

    // This will add on module level:
    //@_global_id_x = external global i64
    //@_global_id_y = external global i64
    //@_global_id_z = external global i64

    GlobalIdIterators = {
    llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(0), ST)),
    llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(1), ST)),
    llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(2), ST))};

    //llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(std::string("_pocl_sub_group_size"),ST));
    

    TempInstructionIndex = 0;

    

    handleWorkitemFunctions();

    //Func.dump();

    // Pointers to local ids
    llvm::GlobalVariable *localIdXPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_x"));
    llvm::GlobalVariable *localIdYPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_y"));
    llvm::GlobalVariable *localIdZPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_z"));

    llvm::Type *uType = llvm::Type::getInt64Ty(M->getContext());
    


    // Create new block
    llvm::BasicBlock *dispatcherBlock = llvm::BasicBlock::Create(F->getContext(), "dispatcher", F);
    

    dispatcher = dispatcherBlock;

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
    bBuilder.CreateStore(loc_x, localIdXPtr);
    bBuilder.CreateStore(loc_y, localIdYPtr);
    bBuilder.CreateStore(loc_z, localIdZPtr);


    llvm::Instruction* tsti = getGlobalIdOrigin(0);




    std::cout << "1\n";

    ///// Dispatcher part ends here

    ////////////  
    identifyContextVars();
    allocateContextVars();
    addSave();
    

    std::cout << "2\n";

    llvm::Module *M = Func.getParent();


    llvm::BasicBlock *Entry = &Func.getEntryBlock();

    // Initialize local id to 0
    insertLocalIdInit_(Entry);


    llvm::IRBuilder<> entryBlockBuilder(Entry, Entry->begin());


    // Array for exit block indices
    llvm::Type *Int64Ty = llvm::Type::getInt64Ty(M->getContext());

    unsigned int n_wi = WGLocalSizeX*WGLocalSizeY*WGLocalSizeZ;

    //llvm::ArrayType *exitBlockIdxs = llvm::ArrayType::get(Int64Ty, n_wi);
    
    //llvm::AllocaInst *nextExitBlockArray = entryBlockBuilder.CreateAlloca(exitBlockIdxs, nullptr, "next_exit_block_array");

    llvm::Type *ContextArrayType = llvm::ArrayType::get(
        llvm::ArrayType::get(llvm::ArrayType::get(Int64Ty, WGLocalSizeX), WGLocalSizeY),WGLocalSizeZ);

    llvm::AllocaInst *nextExitBlockArray = entryBlockBuilder.CreateAlloca(ContextArrayType, nullptr, "exit_blocks");


    // Initialize first jump indices
    for (int i = 0; i < WGLocalSizeZ; i++) {
        for(int j = 0; j< WGLocalSizeY; j++){
            for(int k = 0; k< WGLocalSizeX; k++){
                llvm::Value *index_Z = entryBlockBuilder.getInt64(i);
                llvm::Value *index_Y = entryBlockBuilder.getInt64(j);
                llvm::Value *index_X = entryBlockBuilder.getInt64(k);
                llvm::Value *exitBidxPtr = entryBlockBuilder.CreateGEP(ContextArrayType, nextExitBlockArray, {entryBlockBuilder.getInt64(0), index_Z,index_Y,index_X});
                entryBlockBuilder.CreateStore(entryBlockBuilder.getInt64(0), exitBidxPtr);
            }
        }
        
    }

    
    std::cout << "3\n";


    //M->dump();
    

    ////////////////////////////////////////////////////////////////////////////////////////////
    // Init call: Instead of relying global variables, use the metadata

     // Create function call to __pocl_sched_init
    llvm::Function *schedFuncI = M->getFunction("__pocl_sched_init");

    llvm::ConstantInt* x_size = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeX, false);
    llvm::ConstantInt* y_size = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeY, false);
    llvm::ConstantInt* z_size = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeZ, false);

    llvm::ConstantInt* sgSize;

    if (llvm::MDNode *SGSizeMD = F->getMetadata("intel_reqd_sub_group_size")) {
        // Use the constant from the metadata.
        llvm::ConstantAsMetadata *ConstMD = llvm::cast<llvm::ConstantAsMetadata>(SGSizeMD->getOperand(0));
        sgSize = llvm::cast<llvm::ConstantInt>(ConstMD->getValue());    
    }else{
        sgSize = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeX, false);
    }

    // This will pass the sg size and local size to init function.
    entryBlockBuilder.CreateCall(schedFuncI, {x_size, y_size, z_size, sgSize});

    ////////////////////////////////////////////////////////////////////////////////////////////

    // Store exit blocks after barriers
    std::vector<llvm::BasicBlock*> barrierExits;


    llvm::BasicBlock *currBlock = Entry;


    // Store barrier blocks
    std::vector<llvm::BasicBlock*> barrierBlocks;

    std::cout << "4\n";
   
    llvm::Value *zeroIndex = llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()), 0);

    for(auto &Block : Func){

        // Save blocks that have barriers or sg barriers
        if(Barrier::hasBarrier(&Block) || SubgroupBarrier::hasSGBarrier(&Block)){
            barrierBlocks.push_back(&Block);
        }
    }


    // Store pointer to old exit here
    llvm::BasicBlock* oldExitBlock = nullptr;

    std::cout << "5\n";
    // Modify the barrier blocks
    for(auto &BBlock : barrierBlocks){

        std::cout << "6\n";

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


        // This is the "return" block
        }else if(BBlock->getTerminator()->getNumSuccessors() == 0){
            //std::cout << "BARRIER: " << BBlock->getName().str() << std::endl;
            
            
            
            // Create new kernel exit where we come out as "one"
            llvm::BasicBlock *newExitBlock = llvm::BasicBlock::Create(F->getContext(), "exit_block", F);
            
            // This will be the last jump where we exit from the kernel
            barrierExits.push_back(newExitBlock);


            // Handle for old return block
            llvm::IRBuilder<> oldExitBlockBuilder(BBlock->getTerminator());

            llvm::Value *local_z = oldExitBlockBuilder.CreateLoad(uType,localIdZPtr,"local_z");
            llvm::Value *local_y = oldExitBlockBuilder.CreateLoad(uType,localIdYPtr,"local_y");
            llvm::Value *local_x = oldExitBlockBuilder.CreateLoad(uType,localIdXPtr,"local_x");
            
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
            

            if(Barrier::hasBarrier(BBlock)){
                std::cout << "BARRIER: " << BBlock->getName().str() << std::endl;
            }else if(SubgroupBarrier::hasSGBarrier(BBlock)){
                std::cout << "SG BARRIER:" << BBlock->getName().str() << std::endl;
            }



             // This is the next exit block
            barrierExits.push_back(BBlock->getTerminator()->getSuccessor(0));
            //std::cout << BBlock->getTerminator()->getSuccessor(0)->getName().str()<< std::endl;

            // These contain either barriers or sg barriers, but not the "entry" or "exit" barrier
            llvm::IRBuilder<> barrierBlockBuilder(BBlock->getTerminator());
            

            llvm::Value *local_z = barrierBlockBuilder.CreateLoad(uType,localIdZPtr,"local_z");
            llvm::Value *local_y = barrierBlockBuilder.CreateLoad(uType,localIdYPtr,"local_y");
            llvm::Value *local_x = barrierBlockBuilder.CreateLoad(uType,localIdXPtr,"local_x");
            

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


        // Note that we can have varying number of barriers in the kernel.
        // even just one, if there are no explicit barriers.
        // Above code does not handle that so check that here, refactor later maybe
        if(barrierExits.size() < 2){

        }



    }

    std::cout << "7\n";

    llvm::Value *local_z = bBuilder.CreateLoad(uType,localIdZPtr,"local_z");
    llvm::Value *local_y = bBuilder.CreateLoad(uType,localIdYPtr,"local_y");
    llvm::Value *local_x = bBuilder.CreateLoad(uType,localIdXPtr,"local_x");

    // Pointer to exit index array
    //llvm::Value *next_block_ptr = bBuilder.CreateGEP(exitBlockIdxs, nextExitBlockArray, {zeroIndex, nextWI}, "exit_block_ptr");
    llvm::Value *next_block_ptr = bBuilder.CreateGEP(ContextArrayType,nextExitBlockArray, {zeroIndex, local_z, local_y, local_x}, "exit_block_ptr");

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

    //Func.dump();

    //M->dump();

    std::string Log;
    llvm::raw_string_ostream OS(Log);
    bool BrokenDebugInfo = false;
 
    llvm::verifyModule(*M, &OS, &BrokenDebugInfo);
    if (!Log.empty()) {
    std::cerr << "Module verification errors:\n" << Log << std::endl;
}
  


    llvm::verifyFunction(Func);


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


    dumpCFG(F, F.getName().str() + "_before_fallback.dot", nullptr,nullptr);

    F.dump();

    bool ret_val = WIL.runOnFunction(F);
    
    F.dump();
   
    dumpCFG(F, F.getName().str() + "AFTER_FALLBACK.dot", nullptr,nullptr);

    //return ret_val ? PAChanged : llvm::PreservedAnalyses::all();
    
    return llvm::PreservedAnalyses::all();
    

}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);
}