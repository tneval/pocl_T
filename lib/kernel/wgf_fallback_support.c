#include <stdio.h>

// This can be referenced from kernel because its visibility is global
typedef struct {
    // Set in kernel code
    unsigned long local_size_x;
    unsigned long local_size_y;
    unsigned long local_size_z;
    unsigned long subgroup_size;
    // Set in scheduler init
    unsigned long n_subgroups;
    unsigned long waiting_count;
    unsigned long sg_barriers_active;
    unsigned long *sg_wi_counter;
    unsigned long *sg_barrier_counter;
} wgState;
 

//#define DBG


void __pocl_context_test(wgState *wgState,unsigned long *sg_wi_counter, unsigned long *sg_barrier_counter)
{  
    printf("hello\n");
    for(int i = 0; i< 4; i++){
        printf("barrier counter (%d): %ld\n",i, sg_wi_counter[i]);
        sg_wi_counter[i]++;
    }
    wgState->sg_wi_counter = sg_wi_counter;
    wgState->sg_barrier_counter = sg_barrier_counter;

}

// Init needs group ids as well?
void __pocl_sched_init(wgState *wgState, unsigned long *sg_wi_counter, unsigned long *sg_barrier_counter)
{
   

    wgState->n_subgroups = (wgState->local_size_x*wgState->local_size_y*wgState->local_size_z)/wgState->subgroup_size;
    wgState->waiting_count = 0;
    wgState->sg_barriers_active = 0;

    wgState->sg_wi_counter = sg_wi_counter;
    wgState->sg_barrier_counter = sg_barrier_counter;

#ifdef DBG
 


    printf("struct (init): %d, %d, %d, %d\n", wgState->local_size_x, wgState->local_size_y, wgState->local_size_z, wgState->subgroup_size);

    printf("sg_wi_counter: ");
    for(int i = 0; i< wgState->n_subgroups; i++){
        printf("%d ",wgState->sg_wi_counter[i]);
    }
    printf("\nsg_barrier_counter: ");
    for(int i = 0; i< wgState->n_subgroups; i++){
        printf("%d ",wgState->sg_barrier_counter[i]);
    }
    printf("\n");

    printf("sgsize: %d, n_subgroups: %d, waiting_count: %d, sg_barriers_active: %d\n", wgState->subgroup_size,wgState->n_subgroups, wgState->waiting_count, wgState->sg_barriers_active);
#endif

}


void __pocl_next_jump(long idx){
#ifdef DBG
    printf("jumping next to index: %d\n",idx);
#endif
}



static void resolve_barriers(wgState *wgState)
{   

#ifdef DBG
    printf("SCHEDULER>> BEFORE resolving barriers\n");
    printf("SCHEDULER>> waiting_count: %d\tsg_barriers_active: %d\t",wgState->waiting_count,wgState->sg_barriers_active);

    for(int i = 0; i<wgState->n_subgroups; i++){
        printf("sg_wi_counter[%d]: %d\t",i,wgState->sg_wi_counter[i]);
    }
    for(int i = 0; i<wgState->n_subgroups; i++){
        printf("sg_barrier_status[%d]: %d\t",i,wgState->sg_barrier_counter[i]);
    }
    printf("\n");

#endif

    // Case when no sg barriers are encountered
    if(!wgState->sg_barriers_active){
        
        //Resolve wg barrier when all wis have reached it
        // zero the counters
        if(wgState->waiting_count == wgState->subgroup_size*wgState->n_subgroups){
            
            for(int i = 0; i<wgState->n_subgroups; i++){
                wgState->sg_wi_counter[i] = 0;
            }

            wgState->waiting_count = 0;
        }
        // Else, nothing to do
    
    // Here, there are sg barriers in play.
    // Check if some of the subgroups have all wis reached the sg-barrier
    }else{

        for(int i = 0; i< wgState->n_subgroups; i++){

            // sg barrier is encountered for this subgroup and all of its members have reached the barrier
            if(wgState->sg_barrier_counter[i] == 1 && wgState->sg_wi_counter[i] == wgState->subgroup_size){

                wgState->sg_wi_counter[i] = 0;

                wgState->sg_barrier_counter[i] = 0;

                wgState->waiting_count = wgState->waiting_count - wgState->subgroup_size;

                wgState->sg_barriers_active--;
            }
        }
    }

#ifdef DBG
    printf("SCHEDULER>> AFTER resolving barriers\n");
    printf("SCHEDULER>> waiting_count: %d\tsg_barriers_active: %d\t",wgState->waiting_count,wgState->sg_barriers_active);

    for(int i = 0; i<wgState->n_subgroups; i++){
        printf("sg_wi_counter[%d]: %d\t",i,wgState->sg_wi_counter[i]);
    }
    for(int i = 0; i<wgState->n_subgroups; i++){
        printf("sg_barrier_status[%d]: %d\t",i,wgState->sg_barrier_counter[i]);
    }
    printf("\n");
#endif


}



static void print_barrier_status(wgState *wgState){

    for(int i = 0; i < wgState->n_subgroups; i++){
        
        printf(" %d ", wgState->sg_wi_counter[i]);
    }
    printf("\n");

}



void __pocl_barrier_reached(long local_id_x, long local_id_y, long local_id_z, wgState *wgState)
{

    // Linearize wg id
    unsigned int linearId = ((local_id_z*wgState->local_size_y*wgState->local_size_x)+(local_id_y*wgState->local_size_x)+local_id_x);

    unsigned int sg_id = linearId / wgState->subgroup_size;
    unsigned int sg_local_id = linearId % wgState->subgroup_size;

    wgState->waiting_count++;

    wgState->sg_wi_counter[sg_id]++;

#ifdef DBG
    printf("SCHEDULER>> BARRIER REACHED\n");
    printf("SCHEDULER>> linearID: %d\tlocal_id_x: %d\tlocal_id_y: %d\tlocal_id_z: %d\tsg_id: %d\t sg_local_id: %d\n",linearId,local_id_x,local_id_y,local_id_z,sg_id, sg_local_id);
    print_barrier_status(wgState);
#endif
    

    //resolve_barriers();

}


void __pocl_sg_barrier_reached(long local_id_x, long local_id_y, long local_id_z, wgState *wgState)
{

    // Linearize wg id
    unsigned int linearId = ((local_id_z*wgState->local_size_y*wgState->local_size_x)+(local_id_y*wgState->local_size_x)+local_id_x);

    int sg_id = linearId / wgState->subgroup_size;
    int sg_local_id = linearId % wgState->subgroup_size;


    // Only increase the sg barrier counter when the first wi of the subgroup comes in
    if(wgState->sg_wi_counter[sg_id] == 0){
        wgState->sg_barriers_active++;
        wgState->sg_barrier_counter[sg_id]++;
    }
    
    wgState->sg_wi_counter[sg_id]++;
    wgState->waiting_count++;
   
#ifdef DBG
    printf("SCHEDULER>> SG BARRIER REACHED\n");
    printf("SCHEDULER>> sg_id: %d\t sg_local_id: %d\n",sg_id, sg_local_id);
    print_barrier_status(wgState);
    
#endif


    // Resolve barrier immediately if possible, to avoid extra function calls
    // if(sg_wi_counter[sg_id] == sub_group_size) .. 
}


long __pocl_sched_work_item(wgState *wgState)
{

    resolve_barriers(wgState);

    long next_wi = 0;

    // Go through all subgroups
    for(int i = 0; i< wgState->n_subgroups; i++){
        
        // If wi-counter is not "full", we will schedule from this subgroup.
        if(wgState->sg_wi_counter[i] < wgState->subgroup_size){
            // Adjust the id so that return value will be linearized id of the workgroup
            next_wi = i*wgState->subgroup_size + wgState->sg_wi_counter[i];
            break;
        }
    }

#ifdef DBG
            printf("SCHEDULER>> NEXT WI :%ld\n",next_wi);
#endif

    return next_wi;
}


void __pocl_sched_clean()
{
   
#ifdef DBG
    printf("SCHEDULER>> clean called\n");
#endif

    //free(sg_wi_counter);
    //free(sg_barrier_status);

}

