#include <stdio.h>
#include <stdlib.h>


// WG dimensions
static unsigned int local_size_x;
static unsigned int local_size_y;
static unsigned int local_size_z;


// Subgroup size
static unsigned int sub_group_size;

// Number of subgroups
static unsigned int n_subgroups;

static int waiting_count;

static int sg_barriers_active;


static unsigned int* sg_wi_counter;
static unsigned int* sg_barrier_status;




//#define DBG



void __pocl_sched_init(long x_size, long y_size, long z_size, long sg_size)
{

#ifdef DBG
    fprintf(stdout, "SCHEDULER>> init called %ld\t%ld\t%ld\t%ld\n",sg_size,x_size,y_size,z_size);
#endif

    sub_group_size = sg_size;

    // Set wg dimensions for scheduler
    local_size_x = x_size;
    local_size_y = y_size;
    local_size_z = z_size;


    n_subgroups = (x_size*y_size*z_size)/sg_size;

    waiting_count = 0;

    sg_barriers_active = 0;

    sg_wi_counter = malloc(n_subgroups * sizeof(int));
    sg_barrier_status = malloc(n_subgroups * sizeof(int));

    for(int i = 0; i< n_subgroups; i++){
        sg_wi_counter[i] = 0;
        sg_barrier_status[i] = 0;
    }
}





void __pocl_next_jump(long idx){
    fprintf(stdout, "jumping next to index: %d\n",idx);
}



static void resolve_barriers()
{   

#ifdef DBG
    fprintf(stdout, "SCHEDULER>> BEFORE resolving barriers\n");
    fprintf(stdout, "SCHEDULER>> waiting_count: %d\tsg_barriers_active: %d\t",waiting_count,sg_barriers_active);

    for(int i = 0; i<n_subgroups; i++){
        fprintf(stdout,"sg_wi_counter[%d]: %d\t",i,sg_wi_counter[i]);
    }
    for(int i = 0; i<n_subgroups; i++){
        fprintf(stdout,"sg_barrier_status[%d]: %d\t",i,sg_barrier_status[i]);
    }
    fprintf(stdout, "\n");

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

#ifdef DBG
    fprintf(stdout, "SCHEDULER>> AFTER resolving barriers\n");
    fprintf(stdout, "SCHEDULER>> waiting_count: %d\tsg_barriers_active: %d\t",waiting_count,sg_barriers_active);

    for(int i = 0; i<n_subgroups; i++){
        fprintf(stdout,"sg_wi_counter[%d]: %d\t",i,sg_wi_counter[i]);
    }
    for(int i = 0; i<n_subgroups; i++){
        fprintf(stdout,"sg_barrier_status[%d]: %d\t",i,sg_barrier_status[i]);
    }
    fprintf(stdout, "\n");
#endif


}



static void print_barrier_status(){

    for(int i = 0; i < n_subgroups; i++){
        
        fprintf(stdout, " %d ", sg_wi_counter[i]);
    }
    fprintf(stdout,"\n");

}



void __pocl_barrier_reached(long local_id_x, long local_id_y, long local_id_z)
{
    

    // Linearize wg id
    unsigned int linearId = ((local_id_z*local_size_y*local_size_x)+(local_id_y*local_size_x)+local_id_x);

    int sg_id = linearId / sub_group_size;
    int sg_local_id = linearId % sub_group_size;

    waiting_count++;

    sg_wi_counter[sg_id]++;

#ifdef DBG
    fprintf(stdout, "SCHEDULER>> BARRIER REACHED\n");
    fprintf(stdout, "SCHEDULER>> linearID: %d\tlocal_id_x: %d\tlocal_id_y: %d\tlocal_id_z: %d\tsg_id: %d\t sg_local_id: %d\n",linearId,local_id_x,local_id_y,local_id_z,sg_id, sg_local_id);
    print_barrier_status();
#endif
    

    //resolve_barriers();

}


void __pocl_sg_barrier_reached(long local_id_x, long local_id_y, long local_id_z)
{

    // Linearize wg id
    unsigned int linearId = ((local_id_z*local_size_y*local_size_x)+(local_id_y*local_size_x)+local_id_x);

    int sg_id = linearId / sub_group_size;
    int sg_local_id = linearId % sub_group_size;


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

    fprintf(stdout, "SCHEDULER>> clean called\n");

    free(sg_wi_counter);
    free(sg_barrier_status);

}