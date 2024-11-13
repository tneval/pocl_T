#include "LLVMUtils.h"
#include "SimpleFallback.h"
#include "WorkitemHandlerChooser.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "llvm/IR/IRBuilder.h"
#include "DebugHelpers.h"
#include "KernelCompilerUtils.h"



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

    

    //ParallelRegion *regionOfBlock(llvm::BasicBlock *BB);

    //llvm::Value *getLinearWiIndex(llvm::IRBuilder<> &Builder, llvm::Module *M, ParallelRegion *Region);

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

    std::cout << "identifyContextVars called\n" << std::endl;

    int added = 0;

    for (auto &BB : *F) {
        for (auto &Instr : BB) {


            if (shouldNotBeContextSaved(&Instr)){
                continue;
            }

            std::cout << "Current instr: \n";
            Instr.print(llvm::outs());
            std::cout << "\n";
            for (llvm::Instruction::use_iterator UI = Instr.use_begin(),UE = Instr.use_end();UI != UE; ++UI) {
            
                llvm::Instruction *User = llvm::dyn_cast<llvm::Instruction>(UI->getUser());

                if (User == NULL)
                    continue;

                std::cout << "  User: \n";
                User->print(llvm::outs());
                std::cout <<"\n";


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

    std::cout << "\nFixing: "<<std::endl;
    for(auto &inst : contextVars){
        inst->print(llvm::outs());
        std::cout << "\n";
    }


} // identifyContextVars()

void SimpleFallbackImpl::allocateContextVars()
{

    std::cout << "allocateContextVars called\n";

    for(auto &instr : contextVars){
        // Allocate the context data array for the variable.
        bool PaddingAdded = false;
        llvm::AllocaInst *Alloca = getContextArray(instr, PaddingAdded);

        contextAllocas.push_back(Alloca);
    }
}

void SimpleFallbackImpl::addSave()
{

    for(int i = 0; i< contextVars.size(); i++){
        std::cout << "contextAlloca: " << contextAllocas[i]->getName().str() << "\n";
        contextAllocas[i]->print(llvm::outs());
        std::cout << "\n";
    }
    
    


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
        llvm::Value *local_y = ctxSaveBuilder.CreateLoad(uType,localIdXPtr,"local_y");
        llvm::Value *local_z = ctxSaveBuilder.CreateLoad(uType,localIdXPtr,"local_z");

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

            llvm::Value* LoadedValue = addContextRestore(UserI, contextAllocas[i], contextAllocas[i]->getType(), false, ContextRestoreLocation, llvm::isa<llvm::AllocaInst>(contextVars[i]));
            
            
            
            UserI->replaceUsesOfWith(contextVars[i], LoadedValue);
        
        }


    }
    
}

void SimpleFallbackImpl::addLoad()
{

    bool PaddingAdded=false;
    


    llvm::Instruction* ContextRestoreLocation = dispatcher->getTerminator();

    /* for(int i = 0; i< contextAllocas.size(); i++){

        std::cout << "hep: " << i <<std::endl;

        bool isAlloca = llvm::isa<llvm::AllocaInst>(contextVars[i]);

        llvm::Instruction *GEP = createContextArrayGEP(contextAllocas[i], ContextRestoreLocation, PaddingAdded);

        std::cout << "got gep" << std::endl;

        llvm::IRBuilder<> Builder(ContextRestoreLocation);
        
        Builder.CreateLoad(contextVars[i]->getType(), GEP);

        std::cout << "hop: " <<i <<std::endl;
    } */

    getGEP(contextAllocas[0],ContextRestoreLocation,PaddingAdded);


    std::cout << "addLoad called\n";


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
    llvm::Value *local_y = ctxLoadBuilder.CreateLoad(uType,localIdXPtr,"local_y");
    llvm::Value *local_z = ctxLoadBuilder.CreateLoad(uType,localIdXPtr,"local_z");


    std::vector<llvm::Value *> GEPArgs;
    
    GEPArgs.push_back(llvm::ConstantInt::get(ST, 0));
    GEPArgs.push_back(local_z);
    GEPArgs.push_back(local_y);
    GEPArgs.push_back(local_x);
    
    

    if (AlignPadding)
        GEPArgs.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(CtxArrayAlloca->getContext()), 0));
    

    std::cout << "inserting GEP (getGEP)\n";
    llvm::GetElementPtrInst *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(ctxLoadBuilder.CreateGEP(
      CtxArrayAlloca->getAllocatedType(), CtxArrayAlloca, GEPArgs));

    //CtxArrayAlloca->getAllocatedType()->print(llvm::outs());
  

    /* for(int i = 0; i<contextVars.size(); i++){
        std::cout << "getGEP3\n";
        llvm::GetElementPtrInst *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(ctxLoadBuilder.CreateGEP(
      contextAllocas[i]->getAllocatedType(), contextAllocas[i], GEPArgs));
        std::cout << "getGEP4\n";
    } */

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


    std::cout << "getContextArray called\n";

    std::ostringstream Var;
    Var << ".";

    if (std::string(Inst->getName().str()) != "") {
        Var << Inst->getName().str();
        std::cout << "Instr: " << Inst->getName().str() << std::endl;
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


    std::cout << "addContextRestore called\n";



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



void SimpleFallbackImpl::addCtxSaveRstr(llvm::Instruction *Def){


    std::cout << "void SimpleFallbackImpl::addCtxSaveRstr called\n";

    Def->print(llvm::outs());
    std::cout<<"\n";
    

    // Allocate the context data array for the variable.
    bool PaddingAdded = false;
    llvm::AllocaInst *Alloca = getContextArray(Def, PaddingAdded);
    
    
    // Restoring always happens at the dispatcher
    llvm::Instruction* ContextRestoreLocation = &*dispatcher->begin();

    //llvm::Value *LoadedValue = addContextRestore(UserI, Alloca, Def->getType(), PaddingAdded, ContextRestoreLocation, llvm::isa<llvm::AllocaInst>(Def));
    //llvm::Value *LoadedValue = addContextRestore(nullptr,Alloca,Def->getType(),PaddingAdded, ContextRestoreLocation, llvm::isa<llvm::AllocaInst>(Def));
    
    llvm::Instruction *TheStore = addContextSave(Def, Alloca);


    

    InstructionVec Uses;


    for (llvm::Instruction::use_iterator UI = Def->use_begin(), UE = Def->use_end();UI != UE; ++UI) {

        llvm::Instruction *User = llvm::cast<llvm::Instruction>(UI->getUser());

        if (User == NULL || User == TheStore) continue;

        Uses.push_back(User);
    }

    bool set = false;


    llvm::Value *LoadedValue;

    for (InstructionVec::iterator I = Uses.begin(); I != Uses.end(); ++I) {
 
        llvm::Instruction *UserI = *I;
         
        
        if(!set){
            LoadedValue = addContextRestore(UserI, Alloca, Def->getType(), PaddingAdded, ContextRestoreLocation, llvm::isa<llvm::AllocaInst>(Def));
            set = true;
        
        }
        
        UserI->replaceUsesOfWith(Def, LoadedValue);
    
    }

    


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

    std::cout << "void SimpleFallbackImpl::addContextSaveRestore called\n";

    Def->print(llvm::outs());
    std::cout<<"\n";
    

    // Allocate the context data array for the variable.
    bool PaddingAdded = false;
    llvm::AllocaInst *Alloca = getContextArray(Def, PaddingAdded);
    llvm::Instruction *TheStore = addContextSave(Def, Alloca);



    std::cout << "\nalloca: ";
    Alloca->print(llvm::outs());
    std::cout << "\nTheStore: ";
    TheStore->print(llvm::outs());
    std::cout<< "\n";

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
        //if (regionOfBlock(UserI->getParent()) == NULL) continue;

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
            //assert ("Cannot add context restore for a PHI node at the region entry!" && regionOfBlock(Phi->getParent())->entryBB() != Phi->getParent());

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


void SimpleFallbackImpl::ctxSaveRestore()
{
    InstructionIndex InstructionsInFunction;
    InstructionVec InstructionsToFix;

    for (auto &BB : *F) {
        for (auto &Instr : BB) {
            InstructionsInFunction.insert(&Instr);
        }
    }


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

                // User is in same block = NO CONTEXT SAVE
                llvm::BasicBlock* currentBlock = Instr.getParent();

                llvm::BasicBlock* userBlock = User->getParent();

                if(currentBlock == userBlock){
                    continue;
                }


                // THIS IS FORCE TO NOT CONTEXT SAVE PRINTF CALLS
                /* if(added > 1){
                    continue;
                } */


                InstructionsToFix.push_back(&Instr);
                added++;
                break;
            }
            


        }
    }

    std::cout << "Fixing: "<<std::endl;
    for(auto &inst : InstructionsToFix){
        inst->print(llvm::outs());
        std::cout << "\n";
    }

    for (InstructionVec::iterator I = InstructionsToFix.begin();I != InstructionsToFix.end(); ++I) {

       /*  std::cerr << "### adding context/save restore for" << std::endl;
        (*I)->dump(); */

        addCtxSaveRstr(*I);
  }




}





// Add context save/restore code to variables that are defined in
// the given region and are used outside the region.
//
// Each such variable gets a slot in the stack frame. The variable
// is restored from the stack whenever it's used.
void SimpleFallbackImpl::fixMultiRegionVariables(ParallelRegion *Region) {


    std::cout << "fixMultiRegionVariables called" << std::endl;


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
               /*  if (llvm::isa<llvm::AllocaInst>(Instr) ||
                    // If the instruction is used also inside another region (not
                    // in a regionless BB like the B-loop construct BBs), we need
                    // to context save it to pass the private data over.
                    (InstructionsInRegion.find(User) ==
                    InstructionsInRegion.end() &&
                    regionOfBlock(User->getParent()) != NULL)) {
                        std::cout << "hep" << std::endl;
                        InstructionsToFix.push_back(Instr);
                        break;
                } */

            } // for (llvm::Instruction::use_iterator UI = Instr->use_begin(),UE = Instr->use_end();UI != UE; ++UI)

        } // for (llvm::BasicBlock::iterator I = (*R)->begin(); I != (*R)->end(); ++I)

    } // for (BasicBlockVector::iterator R = Region->begin(); R != Region->end();++R)
    for(auto &inst : InstructionsToFix){
        inst->print(llvm::outs());
        std::cout << "\n";
    }


    for (InstructionVec::iterator I = InstructionsToFix.begin();I != InstructionsToFix.end(); ++I) {

        //std::cerr << "### adding context/save restore for" << std::endl;
        //(*I)->dump();

        addContextSaveRestore(*I);
  }
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

  
  //ParallelRegion *region = regionOfBlock(Def->getParent());
  //assert ("Adding context save outside any region produces illegal code." && region != NULL);

  if (WGDynamicLocalSize) {
    //llvm::Module *M = AllocaI->getParent()->getParent()->getParent();
    //gepArgs.push_back(getLinearWiIndex(builder, M, region));
    std::cout << "dynamic" << std::endl;
  } else {
    std::cout << "not dynamic" << std::endl;
    gepArgs.push_back(llvm::ConstantInt::get(ST, 0));
    //gepArgs.push_back(region->getOrCreateIDLoad(LID_G_NAME(2)));
    //gepArgs.push_back(region->getOrCreateIDLoad(LID_G_NAME(1)));
    //gepArgs.push_back(region->getOrCreateIDLoad(LID_G_NAME(0)));
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

    TempInstructionIndex = 0;

    

    // Deletes parallelregions
    releaseParallelRegions();

    K->getParallelRegions(LI, &OriginalParallelRegions);
    
    handleWorkitemFunctions();

    //Func.dump();

    llvm::IRBuilder<> builder2(&*(Func.getEntryBlock().getFirstInsertionPt()));

    LocalIdXFirstVar = builder2.CreateAlloca(ST, 0, ".pocl.local_id_x_init");
    

   

    /* for (ParallelRegion::ParallelRegionVector::iterator PRI = OriginalParallelRegions.begin(),PRE = OriginalParallelRegions.end();PRI != PRE; ++PRI) {
        ParallelRegion *Region = (*PRI);

        std::cerr << "### Adding context save/restore for PR: ";
        Region->dumpNames();

        //entryCounts[Region->entryBB()]++;

        fixMultiRegionVariables(Region);

    } */


    llvm::GlobalVariable *localIdXPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_x"));

    // Create new block
    llvm::BasicBlock *dispatcherBlock = llvm::BasicBlock::Create(F->getContext(), "dispatcher", F);
    

    dispatcher = dispatcherBlock;

    // Build the dispatcher block
    llvm::IRBuilder<> bBuilder(dispatcherBlock);


    // Create function call to __pocl_sched_work_item to retrieve next WI id
    llvm::Function *schedFunc = M->getFunction("__pocl_sched_work_item");

    // Retrieve the return value, i.e. WI id
    llvm::Value *nextWI = bBuilder.CreateCall(schedFunc);
    nextWI->setName("next_wi");


    // Store new id as global
    bBuilder.CreateStore(nextWI, localIdXPtr);

  
    //ctxSaveRestore();
    //identifyContextVars();
    //allocateContextVars();
    //addSave();
    //addLoad();


    llvm::Module *M = Func.getParent();


    llvm::BasicBlock *Entry = &Func.getEntryBlock();

    insertLocalIdInit_(Entry);


    llvm::IRBuilder<> entryBlockBuilder(Entry, Entry->begin());
    llvm::Type *int32Type = llvm::Type::getInt32Ty(M->getContext());


    // Array for exit block indices
    llvm::Type *Int64Ty = llvm::Type::getInt64Ty(M->getContext());
    llvm::ArrayType *exitBlockIdxs = llvm::ArrayType::get(Int64Ty, 16);
    llvm::AllocaInst *nextExitBlockArray = entryBlockBuilder.CreateAlloca(exitBlockIdxs, nullptr, "next_exit_block_array");


    for (int i = 0; i < 16; i++) {
        llvm::Value *index = entryBlockBuilder.getInt32(i);
        llvm::Value *exitBidxPtr = entryBlockBuilder.CreateGEP(exitBlockIdxs, nextExitBlockArray, {entryBlockBuilder.getInt64(0), index});
        entryBlockBuilder.CreateStore(entryBlockBuilder.getInt64(0), exitBidxPtr);
    }


    


     // Create function call to __pocl_sched_init
    llvm::Function *schedFuncI = M->getFunction("__pocl_sched_init");

    llvm::GlobalVariable *sgSizePtr = llvm::cast<llvm::GlobalVariable>(Func.getParent()->getGlobalVariable("_pocl_sub_group_size"));
    llvm::Type *uType = llvm::Type::getInt64Ty(M->getContext());
    
    llvm::Value *sg_size = entryBlockBuilder.CreateLoad(uType,sgSizePtr,"sg_size");
    
    llvm::GlobalVariable *xSizePtr = llvm::cast<llvm::GlobalVariable>(Func.getParent()->getGlobalVariable("_local_size_x"));

    llvm::Value *x_size = entryBlockBuilder.CreateLoad(uType,xSizePtr,"x_size");

    entryBlockBuilder.CreateCall(schedFuncI, {sg_size,x_size});

    


    std::vector<llvm::BasicBlock*> barrierExits;


    /* // Create new block
    llvm::BasicBlock *dispatcherBlock = llvm::BasicBlock::Create(F->getContext(), "dispatcher", F);
    

    dispatcher = dispatcherBlock; */

    llvm::BasicBlock *currBlock = Entry;


    std::vector<llvm::BasicBlock*> barrierBlocks;

    /* llvm::GlobalVariable *localIdXPtr = llvm::cast<llvm::GlobalVariable>(M->getGlobalVariable("_local_id_x")); */
   
    llvm::Value *zeroIndex = llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 0);

    for(auto &Block : Func){

        if(Barrier::hasBarrier(&Block) || SubgroupBarrier::hasSGBarrier(&Block)){
            barrierBlocks.push_back(&Block);
            
        }
    }


    // Store pointer to old exit here
    llvm::BasicBlock* oldExitBlock = nullptr;


    // Modify the barrier blocks
    for(auto &BBlock : barrierBlocks){

        // This is the entry barrier block
        // Is this bad way to check entry barrier?
        if(BBlock == &Func.getEntryBlock()){

            std::cout << "ENTRY" << std::endl;

            // We dont want to loop over the entry.barrier
            // Do not jump to dispatcher from here
    
            // This is the next exit block


            // In some cases there are none
            if(BBlock->getTerminator()->getNumSuccessors()>0){
                barrierExits.push_back(BBlock->getTerminator()->getSuccessor(0));
                //std::cout << BBlock->getTerminator()->getSuccessor(0)->getName().str()<< std::endl;
            }


            // REMOVE branch to entry and branch to dispatcher instead
            // THIS DOES NOT WORK
           /*  llvm::BasicBlock &entry_barrier = F->getEntryBlock();

            llvm::Instruction* entry_term = entry_barrier.getTerminator();

            llvm::IRBuilder<> ebuilder(&entry_barrier);
            ebuilder.CreateBr(dispatcher);

            entry_term->eraseFromParent(); */



            std::cout << "BARRIER-entry: " << BBlock->getName().str() << std::endl;


        // This is the "return" block
        }else if(BBlock->getTerminator()->getNumSuccessors() == 0){
            std::cout << "BARRIER: " << BBlock->getName().str() << std::endl;
            
            
            
            // Create new kernel exit where we come out as "one"
            llvm::BasicBlock *newExitBlock = llvm::BasicBlock::Create(F->getContext(), "exit_block", F);
            
            // This will be the last jump where we exit from the kernel
            barrierExits.push_back(newExitBlock);


            // Handle for old return block
            llvm::IRBuilder<> oldExitBlockBuilder(BBlock->getTerminator());

            
            llvm::Value *local_x = oldExitBlockBuilder.CreateLoad(uType,localIdXPtr,"local_x");
            
            llvm::Value *next_block_ptr = oldExitBlockBuilder.CreateGEP(exitBlockIdxs, nextExitBlockArray, {zeroIndex, local_x}, "exit_block_ptr");

            llvm::Value *next_block_idx = llvm::ConstantInt::get(Int64Ty, barrierExits.size()-1);
            oldExitBlockBuilder.CreateStore(next_block_idx, next_block_ptr);

            llvm::Function *barrierReached = M->getFunction("__pocl_barrier_reached");           
            oldExitBlockBuilder.CreateCall(barrierReached,{local_x});
            
            // Add branch to dispatcher
            oldExitBlockBuilder.CreateBr(dispatcherBlock);

            // This removes the "old" ret void
            BBlock->getTerminator()->eraseFromParent();

            // Add ret void instr to new return block
            llvm::IRBuilder<> newExitBlockBuilder(newExitBlock);


            /* llvm::Function *schedClean = M->getFunction("__pocl_sched_clean");

            newExitBlockBuilder.CreateCall(schedClean); */

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
            

            llvm::Value *local_x = barrierBlockBuilder.CreateLoad(uType,localIdXPtr,"local_x");
            
            llvm::Value *next_block_ptr = barrierBlockBuilder.CreateGEP(exitBlockIdxs, nextExitBlockArray, {zeroIndex, local_x}, "exit_block_ptr");

            llvm::Value *next_block_idx = llvm::ConstantInt::get(Int64Ty, barrierExits.size()-1);
            barrierBlockBuilder.CreateStore(next_block_idx, next_block_ptr);

            // Register barrier entry
            if(Barrier::hasBarrier(BBlock)){
                llvm::Function *barrierReached = M->getFunction("__pocl_barrier_reached");           
                barrierBlockBuilder.CreateCall(barrierReached,{local_x});
            }else if(SubgroupBarrier::hasSGBarrier(BBlock)){
                llvm::Function *sgbarrierReached = M->getFunction("__pocl_sg_barrier_reached");
                barrierBlockBuilder.CreateCall(sgbarrierReached,{local_x});
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

    /* // Build the dispatcher block
    llvm::IRBuilder<> bBuilder(dispatcherBlock);


    // Create function call to __pocl_sched_work_item to retrieve next WI id
    llvm::Function *schedFunc = M->getFunction("__pocl_sched_work_item");

    // Retrieve the return value, i.e. WI id
    llvm::Value *nextWI = bBuilder.CreateCall(schedFunc);
    nextWI->setName("next_wi");


    // Store new id as global
    bBuilder.CreateStore(nextWI, localIdXPtr); */

    // CONTEXT LOADS HERE

    //ctxSaveRestore();


    // Pointer to exit index array
    llvm::Value *next_block_ptr = bBuilder.CreateGEP(exitBlockIdxs, nextExitBlockArray, {zeroIndex, nextWI}, "exit_block_ptr");

    // Retrieve exit index based for current local_id_x
    llvm::Value *loadedValue = bBuilder.CreateLoad(bBuilder.getInt64Ty(), next_block_ptr, "next_exit_block");
    
    /* llvm::Function *nextI = M->getFunction("__pocl_next_jump");
    bBuilder.CreateCall(nextI, {loadedValue}); */
    
    
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

    M->dump();

  

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

    dumpCFG(F, F.getName().str() + "_before_fallback.dot", nullptr,nullptr);


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
    
    //F.dump();
   
    dumpCFG(F, F.getName().str() + "AFTER_FALLBACK.dot", nullptr,nullptr);

    //return ret_val ? PAChanged : llvm::PreservedAnalyses::all();
    
    return llvm::PreservedAnalyses::all();
    

}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);
}