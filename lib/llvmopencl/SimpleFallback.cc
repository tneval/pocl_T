#include "LLVMUtils.h"
#include "SimpleFallback.h"
#include "WorkitemHandlerChooser.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "llvm/IR/IRBuilder.h"
#include "DebugHelpers.h"
#include "KernelCompilerUtils.h"



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



    // This is "iterator". Don't know if this is the proper place to create it.
    llvm::GlobalVariable *NWI = M->getGlobalVariable(NextWI);
    if(!NWI){
        // If it doesn't exist, define it
        NWI = new llvm::GlobalVariable(*M,SizeT,false,llvm::GlobalValue::ExternalLinkage,llvm::ConstantInt::getNullValue(SizeT),NEXT_WI);
    }

    Builder.CreateStore(llvm::ConstantInt::getNullValue(SizeT), NWI);
}



class SimpleFallbackImpl : public pocl::WorkitemHandler{

public:
    SimpleFallbackImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                    llvm::PostDominatorTree &PDT,
                    VariableUniformityAnalysisResult &VUA)
      : WorkitemHandler(), DT(DT), LI(LI), PDT(PDT), VUA(VUA) {}

    virtual bool runOnFunction(llvm::Function &F);


protected:
    llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr) override;
    llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr,size_t Dim) override;


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

    // An alloca in the kernel which stores the first iteration to execute
    // in the inner (dimension 0) loop. This is set to 1 in an peeled iteration
    // to skip the 0, 0, 0 iteration in the loops.
    llvm::Value *LocalIdXFirstVar;

    std::map<llvm::Instruction *, unsigned> TempInstructionIds;

    

    ParallelRegion *regionOfBlock(llvm::BasicBlock *BB);

    llvm::Value *getLinearWiIndex(llvm::IRBuilder<> &Builder, llvm::Module *M, ParallelRegion *Region);

    llvm::Instruction *addContextSave(llvm::Instruction *Def, llvm::AllocaInst *AllocaI);


    void releaseParallelRegions();

    void fixMultiRegionVariables(ParallelRegion *region);

    bool shouldNotBeContextSaved(llvm::Instruction *Instr);

    void addContextSaveRestore(llvm::Instruction *instruction);

    llvm::Instruction *addContextRestore(llvm::Value *Val, llvm::AllocaInst *AllocaI,llvm::Type *LoadInstType, bool PaddingWasAdded,llvm::Instruction *Before = nullptr, bool isAlloca = false);


    llvm::AllocaInst *getContextArray(llvm::Instruction *Inst,bool &PoclWrapperStructAdded);

    //bool processFunction(llvm::Function &F);

};



// Overrides virtual method in WorkitemHandler class
llvm::Instruction *SimpleFallbackImpl::getLocalIdInRegion(llvm::Instruction *Instr, size_t Dim) {
    
    ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
    
    if (ParRegion != nullptr) {
        return ParRegion->getOrCreateIDLoad(LID_G_NAME(Dim));
    }
    llvm::IRBuilder<> Builder(Instr);

    return Builder.CreateLoad(ST, LocalIdGlobals[Dim]);
}




/// Returns the context array (alloca) for the given \param Inst, creates it if
/// not found.
///
/// \param PaddingAdded will be set to true in case a wrapper struct was
/// added for padding in order to enforce proper alignment to the elements of
/// the array. Such padding might be needed to ensure aligned accessed from
/// single work-items accessing aggregates in the context data.
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

    llvm::Instruction *GEP = createContextArrayGEP(AllocaI, Before, PaddingWasAdded);
    if (isAlloca) {
        /* In case the context saved instruction was an alloca, we created a
        context array with pointed-to elements, and now want to return a
        pointer to the elements to emulate the original alloca. */
        return GEP;
    }

    llvm::IRBuilder<> Builder(Before);
    return Builder.CreateLoad(LoadInstType, GEP);
}





/// Adds context save/restore code for the value produced by the
/// given instruction.
///
/// \todo add only one restore per variable per region.
/// \todo add only one load of the id variables per region.
/// Could be done by having a context restore BB in the beginning of the
/// region and a context save BB at the end.
/// \todo ignore work group variables completely (the iteration variables)
/// The LLVM should optimize these away but it would improve
/// the readability of the output during debugging.
/// \todo rematerialize some values such as extended values of global
/// variables (especially global id which is computed from local id) or kernel
/// argument values instead of allocating stack space for them.
void SimpleFallbackImpl::addContextSaveRestore(llvm::Instruction *Def) {

    std::cerr << "void SimpleFallbackImpl::addContextSaveRestore called\nInstruction: ";

    Def->print(llvm::errs());
    

    // Allocate the context data array for the variable.
    bool PaddingAdded = false;
    llvm::AllocaInst *Alloca = getContextArray(Def, PaddingAdded);
    llvm::Instruction *TheStore = addContextSave(Def, Alloca);



    std::cerr << "\nalloca: ";
    Alloca->print(llvm::errs());
    std::cerr << "\nTheStore: ";
    TheStore->print(llvm::errs());
    std::cerr<< "\n";

    InstructionVec Uses;
    // Restore the produced variable before each use to ensure the correct
    // context copy is used.


    // TODO: This is now obsolete:
    // We could add the restore only to other regions outside the variable
    // defining region and use the original variable in the defining region due
    // to the SSA virtual registers being unique. However, alloca variables can
    // be redefined also in the same region, thus we need to ensure the correct
    // alloca context position is written, not the original unreplicated one.
    // These variables can be generated by volatile variables, private arrays,
    // and due to the PHIs to allocas pass.


    // Find out the uses to fix first as fixing them invalidates the iterator.
    for (llvm::Instruction::use_iterator UI = Def->use_begin(), UE = Def->use_end();UI != UE; ++UI) {

        llvm::Instruction *User = llvm::cast<llvm::Instruction>(UI->getUser());

        if (User == NULL || User == TheStore) continue;

        Uses.push_back(User);
    }

    for (InstructionVec::iterator I = Uses.begin(); I != Uses.end(); ++I) {

        llvm::Instruction *UserI = *I;
        llvm::Instruction *ContextRestoreLocation = UserI;
        // If the user is in a block that doesn't belong to a region, the variable
        // itself must be a "work group variable", that is, not dependent on the
        // work item. Most likely an iteration variable of a for loop with a
        // barrier.
        if (regionOfBlock(UserI->getParent()) == NULL) continue;

        llvm::PHINode* Phi = llvm::dyn_cast<llvm::PHINode>(UserI);
        if (Phi != NULL) {
            // In case of PHI nodes, we cannot just insert the context restore code
            // before it in the same basic block because it is assumed there are no
            // non-phi Instructions before PHIs which the context restore code
            // constitutes to. Add the context restore to the incomingBB instead.

            // There can be values in the PHINode that are incoming from another
            // region even though the decision BB is within the region. For those
            // values we need to add the context restore code in the incoming BB
            // (which is known to be inside the region due to the assumption of not
            // having to touch PHI nodes in PRentry BBs).

            // PHINodes at region entries are broken down earlier.
            assert ("Cannot add context restore for a PHI node at the region entry!" && regionOfBlock(Phi->getParent())->entryBB() != Phi->getParent());

            //std::cerr << "### adding context restore code before PHI" << std::endl;
            //UserI->dump();
            //std::cerr << "### in BB:" << std::endl;
            //UserI->getParent()->dump();

            llvm::BasicBlock *IncomingBB = NULL;
            for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues();
                ++Incoming) {
                llvm::Value *Val = Phi->getIncomingValue(Incoming);
                llvm::BasicBlock *BB = Phi->getIncomingBlock(Incoming);
                if (Val == Def)
                IncomingBB = BB;
            }
            assert(IncomingBB != NULL);
            ContextRestoreLocation = IncomingBB->getTerminator();
        }
        
        llvm::Value *LoadedValue = addContextRestore(UserI, Alloca, Def->getType(), PaddingAdded, ContextRestoreLocation, llvm::isa<llvm::AllocaInst>(Def));
        
        UserI->replaceUsesOfWith(Def, LoadedValue);
    
        //std::cerr << "### done, the user was converted to:" << std::endl;
        //UserI->dump();
   
    }
}








// Add context save/restore code to variables that are defined in
// the given region and are used outside the region.
//
// Each such variable gets a slot in the stack frame. The variable
// is restored from the stack whenever it's used.
void SimpleFallbackImpl::fixMultiRegionVariables(ParallelRegion *Region) {

  InstructionIndex InstructionsInRegion;
  InstructionVec InstructionsToFix;

  // Construct an index of the region's instructions so it's fast to figure
  // out if the variable uses are all in the region.
  for (BasicBlockVector::iterator I = Region->begin(); I != Region->end();++I) {
    for (llvm::BasicBlock::iterator Instr = (*I)->begin(); Instr != (*I)->end();
         ++Instr) {
      

      // Collect instrucions in given region, note that this is a set.
      InstructionsInRegion.insert(&*Instr);
    }
  }

    //std::cerr << "instructions in region\n";
    for(auto &inst : InstructionsInRegion){
        //inst->print(llvm::errs());
        //llvm::errs()<<"\n";
    }
    //std::cerr << "\ndone\n";


  // Find all the instructions that define new values and check if they need
  // to be context saved.
    for (BasicBlockVector::iterator R = Region->begin(); R != Region->end();++R) {

        for (llvm::BasicBlock::iterator I = (*R)->begin(); I != (*R)->end(); ++I) {

            llvm::Instruction *Instr = &*I;

            if (shouldNotBeContextSaved(&*Instr)){
                //llvm::errs()<<"\nNot saved\n";
                continue;
            }else{
                //llvm::errs()<<"\nSAVED\n";
            }

            for (llvm::Instruction::use_iterator UI = Instr->use_begin(),UE = Instr->use_end();UI != UE; ++UI) {
            
                llvm::Instruction *User = llvm::dyn_cast<llvm::Instruction>(UI->getUser());

                if (User == NULL)
                    continue;

                // Allocas (originating from OpenCL C private arrays) should be
                // privatized always. Otherwise we end up reading the same array,
                // but replicating only the GEP pointing to it.
                if (llvm::isa<llvm::AllocaInst>(Instr) ||
                    // If the instruction is used also inside another region (not
                    // in a regionless BB like the B-loop construct BBs), we need
                    // to context save it to pass the private data over.
                    (InstructionsInRegion.find(User) ==
                    InstructionsInRegion.end() &&
                    regionOfBlock(User->getParent()) != NULL)) {
                        InstructionsToFix.push_back(Instr);
                        break;
                }

            } // for (llvm::Instruction::use_iterator UI = Instr->use_begin(),UE = Instr->use_end();UI != UE; ++UI)

        } // for (llvm::BasicBlock::iterator I = (*R)->begin(); I != (*R)->end(); ++I)

    } // for (BasicBlockVector::iterator R = Region->begin(); R != Region->end();++R)



    for (InstructionVec::iterator I = InstructionsToFix.begin();I != InstructionsToFix.end(); ++I) {

        std::cerr << "### adding context/save restore for" << std::endl;
        (*I)->dump();

        addContextSaveRestore(*I);
  }
}


// DECIDE WHETHER VARIABLE SHOULD BE CONTEXT SAVED
bool SimpleFallbackImpl::shouldNotBeContextSaved(llvm::Instruction *Instr) {


    //Instr->print(llvm::errs());

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

    // Generated id loads should not be replicated as it leads to problems in
    // conditional branch case where the header node of the region is shared
    // across the peeled branches and thus the header node's ID loads might get
    // context saved which leads to egg-chicken problems.
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

        

    // In case of uniform variables (same value for all work-items), there is no
    // point to create a context array slot for them, but just use the original
    // value everywhere.

    // Allocas are problematic since they include the de-phi induction variables
    // of the b-loops. In those case each work item has a separate loop iteration
    // variable in LLVM IR but which is really a parallel region loop invariant.
    // But because we cannot separate such loop invariant variables at this point
    // sensibly, let's just replicate the iteration variable to each work item
    // and hope the latter optimizations reduce them back to a single induction
    // variable outside the parallel loop.
    if (!VUA.shouldBePrivatized(Instr->getParent()->getParent(), Instr)) {

        //std::cerr << "### based on VUA, not context saving:";
        //Instr->dump();
        //llvm::errs()<<"\nReason: based on VUA?";
        return true;
    }

    return false;
}



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


    //M->print(llvm::outs(), nullptr);


    //M->dump();

    Initialize(llvm::cast<Kernel>(&Func));

    // This will add on module level:
    //@_global_id_x = external global i64
    //@_global_id_y = external global i64
    //@_global_id_z = external global i64

    GlobalIdIterators = {
    llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(0), ST)),
    llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(1), ST)),
    llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(2), ST))};

    TempInstructionIndex = 0;

    

    // Deletes parallelregions
    releaseParallelRegions();

    K->getParallelRegions(LI, &OriginalParallelRegions);

    handleWorkitemFunctions();

    llvm::IRBuilder<> builder2(&*(Func.getEntryBlock().getFirstInsertionPt()));
    LocalIdXFirstVar = builder2.CreateAlloca(ST, 0, ".pocl.local_id_x_init");

    /* for (ParallelRegion::ParallelRegionVector::iterator PRI = OriginalParallelRegions.begin(),PRE = OriginalParallelRegions.end();PRI != PRE; ++PRI) {
        ParallelRegion *Region = (*PRI);

        std::cerr << "### Adding context save/restore for PR: ";
        Region->dumpNames();

        //entryCounts[Region->entryBB()]++;

        fixMultiRegionVariables(Region);
    }

     */
    llvm::Module *M = Func.getParent();


    int added = 0;

    llvm::BasicBlock *Entry = &Func.getEntryBlock();
    insertLocalIdInit_(Entry);


    llvm::Instruction *term = Entry->getTerminator();

    llvm::IRBuilder<> builderInit(term);

    // Move insertion point to before after call, rather than after
    //builderI.SetInsertPoint(&*++builderI.GetInsertPoint());


     // Create function call to __pocl_sched_init
    llvm::Function *schedFuncI = M->getFunction("__pocl_sched_init");

    llvm::GlobalVariable *sgSizePtr = llvm::cast<llvm::GlobalVariable>(Func.getParent()->getGlobalVariable("_pocl_sub_group_size"));
    llvm::Type *uType = llvm::Type::getInt32Ty(M->getContext());
    llvm::Value *sg_size = builderInit.CreateLoad(uType,sgSizePtr,"sg_size");
    
    llvm::GlobalVariable *xSizePtr = llvm::cast<llvm::GlobalVariable>(Func.getParent()->getGlobalVariable("_local_size_x"));

    llvm::Value *x_size = builderInit.CreateLoad(uType,xSizePtr,"x_size");

    builderInit.CreateCall(schedFuncI, {sg_size,x_size});

    //F->dump();
    M->dump();

    // Store pointers here
    // Blocks where we jump back to
    std::vector<llvm::BasicBlock*> loopBlocks;
    // Blocks where we should proceed next
    std::vector<llvm::BasicBlock*> procBlocks;


    for (auto &BasicBlock : Func) {
        for (auto &Instr : BasicBlock) {

            
            // Check for function calls
            if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(&Instr)) {
                
                llvm::BasicBlock *currentBB = callInst->getParent();

                llvm::Function *calledFunc = callInst->getCalledFunction();

                if(calledFunc->getName().str() == "pocl.barrier") {
                    


                    // Do not loop the entry and exit
                    /* if(currentBB->getName().str() == "exit.barrier"){
                        // We cant skip this
                        //continue;
                    } */

                    if(currentBB->getName().str() == "entry.barrier"){
                        
                        // Store next looping block, dont loop back to barrier.entry, but entry instead
                        loopBlocks.push_back(currentBB->getNextNode());

                        continue;

                    }else{
                        // This will be the next block after barrier
                        if(currentBB->getName().str() != "exit.barrier"){
                            procBlocks.push_back(currentBB->getNextNode());
                        }
                    }

                    std::cout << "popping next loop block: size: " << loopBlocks.size() << std::endl;
                    // Get next block target and remove it from the list
                    llvm::BasicBlock* loopBlock = loopBlocks.back();
                    loopBlocks.pop_back();


                    llvm::IRBuilder<> builder(callInst);

                    // Move insertion point to before after call, rather than after
                    builder.SetInsertPoint(&*++builder.GetInsertPoint());


                    llvm::Function *barrierReached = M->getFunction("__pocl_barrier_reached");

                    /* llvm::Function *sg_local_id_f = M->getFunction("_Z22get_sub_group_local_idv");
                    llvm::Value *sg_local_id = builder.CreateCall(sg_local_id_f);
                    sg_local_id->setName("sg_local_id_for_scheduler");

                    llvm::Function *sg_id_f = M->getFunction("_Z16get_sub_group_idv");
                    llvm::Value *sg_id = builder.CreateCall(sg_id_f);
                    sg_id->setName("sg_id_for_scheduler"); */

                    llvm::GlobalVariable *localx = llvm::cast<llvm::GlobalVariable>(Func.getParent()->getGlobalVariable("_local_id_x"));
                    llvm::Value *local_x = builderInit.CreateLoad(uType,xSizePtr,"local_x");
                    builder.CreateCall(barrierReached,{local_x});


                    // Create function call to __pocl_sched_work_item to retrieve next WI id
                    llvm::Function *schedFunc = M->getFunction("__pocl_sched_work_item");

                
                    // Retrieve the return value, i.e. WI id
                    llvm::Value *returnValue = builder.CreateCall(schedFunc);

                    llvm::LLVMContext &context = Func.getContext();

                   
                    llvm::GlobalVariable *localIdXPtr = llvm::cast<llvm::GlobalVariable>(Func.getParent()->getGlobalVariable("_local_id_x"));

                    // Use this temp iterator instead of _local_id_x
                    llvm::GlobalVariable *nextWiptr = llvm::cast<llvm::GlobalVariable>(Func.getParent()->getGlobalVariable("_next_wi_x"));
                    
                    // Here 16 is hardcoded for now, change.
                    llvm::Value *compVal = llvm::ConstantInt::get(llvm::Type::getInt64Ty(Func.getContext()), 16);

                    // Result of comparison; check if WI id from scheduler is less than max wg id
                    llvm::Value *comparisonIter = builder.CreateICmpSLT(returnValue, compVal, "check_iterator");


                    //llvm::Value *localIdXValue = builder.CreateLoad(localIdXPtr->getValueType(), localIdXPtr, "loaded_local_id_x");

                    llvm::Value *selectedValue = builder.CreateSelect(comparisonIter, returnValue ,builder.getInt64(0));


                    // Store new WI id            
                    builder.CreateStore(returnValue, nextWiptr);

                    
                    



                    // Store new WI id            
                    builder.CreateStore(selectedValue, localIdXPtr);


                    // NOTE: THIS IS NEEDED IN THE VECTOR ADDITION AT LEAST
                    llvm::GlobalVariable *globalIdXPtr = llvm::cast<llvm::GlobalVariable>(Func.getParent()->getGlobalVariable("_global_id_x"));
                    builder.CreateStore(selectedValue, globalIdXPtr);

                    // One of the conditional jump targets. This is where to jump after all WIs have cleared the barrier
                    //llvm::BasicBlock *nextBlock = BasicBlock.getNextNode();

                    
                    //llvm::Value *localIdXValue = builder.CreateLoad(localIdXPtr->getValueType(), localIdXPtr, "loaded_local_id_x");
                    
                    // Here 16 is hardcoded for now, change.
                    //llvm::Value *comparisonValue = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 15);


                    // Result of comparison; check if WI id from scheduler is less than max wg id
                    //llvm::Value *comparisonResult = builder.CreateICmpSLT(localIdXValue, comparisonValue, "is_less_than");


                    // This is another jump target when we loop back
                    // NOTE: FOR NOW THIS IS HARDCODED to be next from the entry. CHANGE                  
                    //llvm::BasicBlock &entryBlock = F.getEntryBlock();
                    //llvm::BasicBlock *tmp = entryBlock.getNextNode();


                    bool isExitBlock = false;

                    // For now, check if last instruction is ret void. Don't know if this is enough to deduce terminal block.
                    // There might be several like this, but does it matter?
                    llvm::Instruction* lastOfBlock = currentBB->getTerminator();

                    if(auto* retInst = llvm::dyn_cast<llvm::ReturnInst>(lastOfBlock)){
                        if(retInst->getReturnValue() == nullptr){
                            isExitBlock = true;
                        }
                    }


                    if(!isExitBlock){
                    //if(currentBB->getName().str() != "exit.barrier"){
                        std::cout << "Popping next proc block: size: " << procBlocks.size() << std::endl;
                        llvm::BasicBlock* nextBlock = procBlocks.back();
                        procBlocks.pop_back();


                        // Next loop block will be the one that we jump into now
                        loopBlocks.push_back(nextBlock);

                        // create branch instr
                        builder.CreateCondBr(comparisonIter, loopBlock, nextBlock);

                        // Remove last branch as we just created new conditional one above
                        llvm::Instruction *lastInst = currentBB->getTerminator();
                        if (lastInst) {
                            lastInst->eraseFromParent();  // This removes and deallocates the instruction
                        }
                    // Handle exit barrier
                    }else{




                        //Create new exit block that only contains ret void
                        llvm::BasicBlock* return_block = llvm::BasicBlock::Create(context, "return_block", &Func);
                        llvm::IRBuilder<> bldr(return_block);
                        bldr.CreateRetVoid();

                        // Get handle to "old" ret void instr
                        llvm::IRBuilder<> InsertBuilder(currentBB->getTerminator());


                       
                        InsertBuilder.CreateCondBr(comparisonIter, loopBlock,return_block);

                        llvm::Instruction* last_ret = currentBB->getTerminator();

                        last_ret->eraseFromParent();



                    }


                    

                    

                    // Just to add first loop
                    //added ++;


                    llvm::errs() << "Found a call to: " << calledFunc->getName().str() << "\n";
                }else if(calledFunc->getName().str() == "pocl.subgroup_barrier"){

                    



                }

                //llvm::errs() << "Found a call to: " << callInst->getName().str() << "\n";
                
               
            }

            
        }
    }










    
    //std::cerr << "\n 1###########################################\n" << std::endl;
    //F->dump();

    


    

    //std::cerr << "\n2 ###########################################\n" << std::endl;

    //handleWorkitemFunctions();
    
    //F->dump();


    /* llvm::IRBuilder<> builder(&*(F->getEntryBlock().getFirstInsertionPt()));
    LocalIdXFirstVar = builder.CreateAlloca(ST, 0, ".pocl.local_id_x_init"); */

    

    //std::cerr << "\n3 ###########################################\n" << std::endl;
    //F->dump();



    /* Count how many parallel regions share each entry node to
     detect diverging regions that need to be peeled. */
    std::map<llvm::BasicBlock*, int> entryCounts;

    /* for (ParallelRegion::ParallelRegionVector::iterator PRI = OriginalParallelRegions.begin(),PRE = OriginalParallelRegions.end();PRI != PRE; ++PRI) {
        ParallelRegion *Region = (*PRI);

        std::cerr << "### Adding context save/restore for PR: ";
        Region->dumpNames();

       

        entryCounts[Region->entryBB()]++;

        //fixMultiRegionVariables(Region);
    } */



    
    //bool Changed = processFunction(Func);    

    //return llvm::PreservedAnalyses::none();
    return true;

}

llvm::PreservedAnalyses SimpleFallback::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
    
    // We only want to process kernel functions
    if (!isKernelToProcess(F)){
        return llvm::PreservedAnalyses::all();
    }


    dumpCFG(F, F.getName().str() + "_before_fallback.dot", nullptr,nullptr);

    
    WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;

    if (WIH != WorkitemHandlerType::FALLBACK)
    {
        return llvm::PreservedAnalyses::all();
    }
    
    if(WIH == WorkitemHandlerType::FALLBACK){
        std::cout << "WIH  is of type FALLBACK" << std::endl;
    }

    llvm::errs() << F.getName() << "\n";

    F.dump();


    /* llvm::Module *M = F.getParent();


    int added = 0;

    auto *Entry = &F.getEntryBlock();
    insertLocalIdInit_(Entry);

    // Store pointers here
    // Blocks where we jump back to
    std::vector<llvm::BasicBlock*> loopBlocks;
    // Blocks where we should proceed next
    std::vector<llvm::BasicBlock*> procBlocks; */


    


    

    /* for (auto &BasicBlock : F) {
        for (auto &Instr : BasicBlock) {

            
            // Check for function calls
            if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(&Instr)) {
                
                llvm::BasicBlock *currentBB = callInst->getParent();

                llvm::Function *calledFunc = callInst->getCalledFunction();

                if(calledFunc->getName().str() == "pocl.barrier") {
            
                    // Do not loop the entry and exit
                    if(currentBB->getName().str() == "exit.barrier"){
                        continue;
                    }

                    if(currentBB->getName().str() == "entry.barrier"){
                        
                        // Store next looping block, dont loop back to barrier.entry, but entry instead
                        loopBlocks.push_back(currentBB->getNextNode());

                        continue;

                    }else{
                        // This will be the next block after barrier
                        if(currentBB->getName().str() != "exit.barrier"){
                            procBlocks.push_back(currentBB->getNextNode());
                        }
                    }

                    std::cout << "popping next loop block: size: " << loopBlocks.size() << std::endl;
                    // Get next block target and remove it from the list
                    llvm::BasicBlock* loopBlock = loopBlocks.back();
                    loopBlocks.pop_back();


                    llvm::IRBuilder<> builder(callInst);

                    // Move insertion point to before after call, rather than after
                    builder.SetInsertPoint(&*++builder.GetInsertPoint());

                    // Create function call to __pocl_sched_work_item to retrieve next WI id
                    llvm::FunctionType *funcType = llvm::FunctionType::get(builder.getInt64Ty(),{},false);
                    llvm::Function *schedFunc = M->getFunction("__pocl_sched_work_item");

                    if (!schedFunc) {
                        schedFunc = llvm::Function::Create(funcType,llvm::Function::ExternalLinkage,"__pocl_sched_work_item",M);
                    }

                    // Retrieve the return value, i.e. WI id
                    llvm::Value *returnValue = builder.CreateCall(schedFunc);

                    llvm::LLVMContext &context = F.getContext();

                   
                    llvm::GlobalVariable *localIdXPtr = llvm::cast<llvm::GlobalVariable>(F.getParent()->getGlobalVariable("_local_id_x"));

                    // Use this temp iterator instead of _local_id_x
                    llvm::GlobalVariable *nextWiptr = llvm::cast<llvm::GlobalVariable>(F.getParent()->getGlobalVariable("_next_wi_x"));
                    
                    // Here 16 is hardcoded for now, change.
                    llvm::Value *compVal = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 16);

                    // Result of comparison; check if WI id from scheduler is less than max wg id
                    llvm::Value *comparisonIter = builder.CreateICmpSLT(returnValue, compVal, "check_iterator");


                    //llvm::Value *localIdXValue = builder.CreateLoad(localIdXPtr->getValueType(), localIdXPtr, "loaded_local_id_x");

                    llvm::Value *selectedValue = builder.CreateSelect(comparisonIter, returnValue ,builder.getInt64(0));


                    // Store new WI id            
                    builder.CreateStore(returnValue, nextWiptr);

                    
                    



                    // Store new WI id            
                    builder.CreateStore(selectedValue, localIdXPtr);

                    // One of the conditional jump targets. This is where to jump after all WIs have cleared the barrier
                    //llvm::BasicBlock *nextBlock = BasicBlock.getNextNode();

                    
                    //llvm::Value *localIdXValue = builder.CreateLoad(localIdXPtr->getValueType(), localIdXPtr, "loaded_local_id_x");
                    
                    // Here 16 is hardcoded for now, change.
                    //llvm::Value *comparisonValue = llvm::ConstantInt::get(llvm::Type::getInt64Ty(F.getContext()), 15);


                    // Result of comparison; check if WI id from scheduler is less than max wg id
                    //llvm::Value *comparisonResult = builder.CreateICmpSLT(localIdXValue, comparisonValue, "is_less_than");


                    // This is another jump target when we loop back
                    // NOTE: FOR NOW THIS IS HARDCODED to be next from the entry. CHANGE                  
                    //llvm::BasicBlock &entryBlock = F.getEntryBlock();
                    //llvm::BasicBlock *tmp = entryBlock.getNextNode();


                    std::cout << "Popping next proc block: size: " << procBlocks.size() << std::endl;
                    llvm::BasicBlock* nextBlock = procBlocks.back();
                    procBlocks.pop_back();


                    // Next loop block will be the one that we jump into now
                    loopBlocks.push_back(nextBlock);

                    // create branch instr
                    builder.CreateCondBr(comparisonIter, loopBlock, nextBlock);

                    // Remove last branch as we just created new conditional one above
                    llvm::Instruction *lastInst = currentBB->getTerminator();
                    if (lastInst) {
                        lastInst->eraseFromParent();  // This removes and deallocates the instruction
                    }

                    

                    // Just to add first loop
                    //added ++;


                    llvm::errs() << "Found a call to: " << calledFunc->getName().str() << "\n";
                } 

                //llvm::errs() << "Found a call to: " << callInst->getName().str() << "\n";
                
               
            }

            
        }
    } */
        

    



    auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
    auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
    auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
    auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);

    // Not sure what these do
    llvm::PreservedAnalyses PAChanged = llvm::PreservedAnalyses::none();
    PAChanged.preserve<VariableUniformityAnalysis>();
    PAChanged.preserve<WorkitemHandlerChooser>();

    

    SimpleFallbackImpl WIL(DT, LI, PDT, VUA);


    //dumpCFG(F, F.getName().str() + "_after_fallback.dot", nullptr,nullptr);

    bool ret_val = WIL.runOnFunction(F);
    
    F.dump();
   
    dumpCFG(F, F.getName().str() + "_after_fallback.dot", nullptr,nullptr);

    //return ret_val ? PAChanged : llvm::PreservedAnalyses::all();
    
    return llvm::PreservedAnalyses::all();
    

}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);
}