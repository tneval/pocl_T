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




static void resolve_barriers()
{

   /*  // Just check if waiting count is equal to total n of wis
    if(!sg_barriers_active){
        fprintf(stdout, "wainting_count: %d\n",waiting_count);
        // Can pass the barrier
        if(waiting_count == sub_group_size*n_subgroups){
            

            waiting_count = 0;
        }
    }     */

    
    if(!sg_barriers_active){
        
        //Resolve wg barrier
        if(waiting_count == sub_group_size*n_subgroups){
            
            for(int i = 0; i<n_subgroups; i++){
                sg_wi_counter[i] = 0;
            }

            waiting_count = 0;
        }
        // Else, nothing to do
    
    // Check if one of the subgroups have all wis reached the sg-barrier
    }else{

        for(int i = 0; i< n_subgroups; i++){

            // sg barrier is active for this subgroup and all of its members have reached the barrier
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
    fprintf(stdout, "init called %u\t%d\n",sg_size,x_size);

    sub_group_size = sg_size;

    n_subgroups = x_size/sg_size;

    waiting_count = 0;

    sg_barriers_active = 0;

    /* sub_group_barrier_status = malloc(n_subgroups * sizeof(int));

    wi_barrier_status = malloc(x_size * sizeof(int*));

    for(int sg_i = 0; sg_i < n_subgroups; sg_i++){
        wi_barrier_status[sg_i] = malloc(sub_group_size * sizeof(int));
    }


    // Init values

    for(int i = 0; i< n_subgroups; i++){
        sub_group_barrier_status[i] = 0;

        for(int j = 0; j < sub_group_size; j++){
            wi_barrier_status[i][j] = 0;
        }
    } */

    // ALT approach

    sg_wi_counter = malloc(n_subgroups * sizeof(int));
    sg_barrier_status = malloc(n_subgroups * sizeof(int));

    for(int i = 0; i< n_subgroups; i++){
        sg_wi_counter[i] = 0;
        sg_barrier_status[i] = 0;
    }
}


void __pocl_barrier_reached(int local_id_x)
{

    fprintf(stdout, "BARRIER\n");

    int sg_id = local_id_x / sub_group_size;
    int sg_local_id = local_id_x % sub_group_size;

    fprintf(stdout, "SCHEDULER>> sg_id: %d\t sg_local_id: %d\n",sg_id, sg_local_id);


    //wi_barrier_status[sg_id][sg_local_id]++;

    waiting_count++;


    // ALT; increase counter for corresponding subgroup
    sg_wi_counter[sg_id]++;


    print_barrier_status();

    resolve_barriers();

}


void __pocl_sg_barrier_reached(int local_id_x)
{

    fprintf(stdout, "SG BARRIER\n");

    int sg_id = local_id_x / sub_group_size;
    int sg_local_id = local_id_x % sub_group_size;

    //wi_barrier_status[sg_id][sg_local_id]++;

    //sub_group_barrier_status[sg_id]++;

    sg_barriers_active++;
    

    // ALT approach:
    sg_wi_counter[sg_id]++;
    waiting_count++;
    sg_barrier_status[sg_id]++;


    print_barrier_status();

    // Check if barriers can be passed
    //resolve_barriers();

}


long __pocl_sched_work_item()
{

    long next_wi = 0;

    for(int i = 0; i< n_subgroups; i++){

        if(sg_wi_counter[i] < sub_group_size){
            next_wi = i*sub_group_size + sg_wi_counter[i];
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