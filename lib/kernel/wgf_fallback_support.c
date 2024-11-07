#include <stdio.h>
#include <stdlib.h>


static unsigned int sub_group_size;
/* 
static int** wi_barrier_status;

static int* sub_group_barrier_status; */

static int n_subgroups;


static int waiting_count;

static int sg_barriers_active;


// alternative approach, dont store arrays with individual wis, but count wis per sg

static int* sg_wi_counter;
static int* sg_barrier_status;


#define DBG


void __pocl_next_jump(int idx){
    fprintf(stdout, "jumping next to index: %d\n",idx);
}



static void resolve_barriers()
{   

#ifdef DBG
    fprintf(stdout, "SCHEDULER>> resolving barriers\n");
    fprintf(stdout, "SCHEDULER>> waiting_count: %d\tsg_barriers_active: %d\tsg_wi_counter[0]: %d\tsg_wi_counter[1]: %d\tsg_barrier_status[0]: %d\tsg_barrier_status[1]:%d\n",waiting_count,sg_barriers_active,sg_wi_counter[0],sg_wi_counter[1],sg_barrier_status[0],sg_barrier_status[1]);
#endif

    // Case when no sg barriers are encountered
    if(!sg_barriers_active){
        
        //Resolve wg barrier when all wis have reached it
        // zero the counters
        if(waiting_count == sub_group_size*n_subgroups){
            
            for(int i = 0; i<n_subgroups; i++){
                sg_wi_counter[i] = 0;
            }

            waiting_count = 0;
        }
        // Else, nothing to do
    
    // Here, there are sg barriers in play.
    // Check if some of the subgroups have all wis reached the sg-barrier
    }else{

        for(int i = 0; i< n_subgroups; i++){

            // sg barrier is encountered for this subgroup and all of its members have reached the barrier
            if(sg_barrier_status[i] == 1 && sg_wi_counter[i] == sub_group_size){

                sg_wi_counter[i] = 0;

                sg_barrier_status[i] = 0;

                waiting_count = waiting_count - sub_group_size;

                sg_barriers_active--;
            }
        }
    }
}



static void print_barrier_status(){

    for(int i = 0; i < n_subgroups; i++){
        
        fprintf(stdout, " %d ", sg_wi_counter[i]);
    }
    fprintf(stdout,"\n");

}


void __pocl_sched_init(unsigned int sg_size, unsigned int x_size)
{

#ifdef DBG
    fprintf(stdout, "SCHEDULER>> init called %u\t%d\n",sg_size,x_size);
#endif

    sub_group_size = sg_size;

    n_subgroups = x_size/sg_size;

    waiting_count = 0;

    sg_barriers_active = 0;

    sg_wi_counter = malloc(n_subgroups * sizeof(int));
    sg_barrier_status = malloc(n_subgroups * sizeof(int));

    for(int i = 0; i< n_subgroups; i++){
        sg_wi_counter[i] = 0;
        sg_barrier_status[i] = 0;
    }
}


void __pocl_barrier_reached(int local_id_x)
{
    

    int sg_id = local_id_x / sub_group_size;
    int sg_local_id = local_id_x % sub_group_size;



    waiting_count++;

    sg_wi_counter[sg_id]++;

#ifdef DBG
    fprintf(stdout, "SCHEDULER>> BARRIER REACHED\n");
    fprintf(stdout, "SCHEDULER>> sg_id: %d\t sg_local_id: %d\n",sg_id, sg_local_id);
    print_barrier_status();
#endif
    

    //resolve_barriers();

}


void __pocl_sg_barrier_reached(int local_id_x)
{

    

    int sg_id = local_id_x / sub_group_size;
    int sg_local_id = local_id_x % sub_group_size;


    // Only increase the sg barrier counter when the first wi of the subgroup comes in
    if(sg_wi_counter[sg_id] == 0){
        sg_barriers_active++;
        sg_barrier_status[sg_id]++;
    }
    
    sg_wi_counter[sg_id]++;
    waiting_count++;
   
#ifdef DBG
    fprintf(stdout, "SCHEDULER>> SG BARRIER REACHED\n");
    fprintf(stdout, "SCHEDULER>> sg_id: %d\t sg_local_id: %d\n",sg_id, sg_local_id);
    print_barrier_status();
#endif


    

   


    // Resolve barrier immediately if possible, to avoid extra function calls
    // if(sg_wi_counter[sg_id] == sub_group_size) .. 
}


long __pocl_sched_work_item()
{

    resolve_barriers();

    long next_wi = 0;

    for(int i = 0; i< n_subgroups; i++){

        if(sg_wi_counter[i] < sub_group_size){
            next_wi = i*sub_group_size + sg_wi_counter[i];

#ifdef DBG
            fprintf(stdout, "SCHEDULER>> NEXT WI :%ld\n",next_wi);
#endif

            break;
        }
    }
    return next_wi;
}


void __pocl_sched_clean()
{
    /* free(sub_group_barrier_status);

    for(int sg_i = 0; sg_i < n_subgroups; sg_i++){
        free(wi_barrier_status[sg_i]);
    }

    free(wi_barrier_status); */


    // ALT

    free(sg_wi_counter);
    free(sg_barrier_status);

}