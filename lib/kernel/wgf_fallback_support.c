#include <stdio.h>
#include <stdlib.h>

#define START_IDX 0
#define WG_SIZE 16

static unsigned int sub_group_size;

int** barrier_status;

int* sub_group_status;

int n_subgroups;



void __pocl_sched_init(unsigned int sg_size, unsigned int x_size)
{
    fprintf(stdout, "init called %u\t%d\n",sg_size,x_size);

    sub_group_size = sg_size;

    n_subgroups = x_size/sg_size;

    sub_group_status = malloc(n_subgroups * sizeof(int));

    barrier_status = malloc(x_size * sizeof(int*));

    for(int sg_i = 0; sg_i < n_subgroups; sg_i++){
        barrier_status[sg_i] = malloc(sub_group_size * sizeof(int));
    }


    // Init values

    for(int i = 0; i< n_subgroups; i++){
        sub_group_status[i] = 0;

        for(int j = 0; j < sub_group_size; j++){
            barrier_status[i][j] = 0;
        }
    }


}


void __pocl_barrier_reached(int local_id_x)
{

    fprintf(stdout, "BARRIER\n");

    int sg_id = local_id_x / sub_group_size;
    int sg_local_id = local_id_x % sub_group_size;

    fprintf(stdout, "sg_id: %d\t sg_local_id: %d\n",sg_id, sg_local_id);

}



long __pocl_sched_work_item()
{

    static long next_id = START_IDX;

    next_id++;

    // Temp solution for now
    if(next_id == WG_SIZE+1){
        next_id = 1;
    }

    // Implement bookkeeping here
    fprintf(stdout, ">> SCHEDULER: next_id is %d\n",next_id);



    return next_id;

}


void __pocl_sched_clean()
{
    free(sub_group_status);

    for(int sg_i = 0; sg_i < n_subgroups; sg_i++){
        free(barrier_status[sg_i]);
    }

    free(barrier_status);

}