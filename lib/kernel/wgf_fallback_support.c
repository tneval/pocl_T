#include <stdio.h>


#define START_IDX 0
#define WG_SIZE 16

long __pocl_sched_work_item()
{

    //static int WIs[WG_SIZE] = {0};

    long start_idx = START_IDX;

    static long next_id = START_IDX;

    next_id++;

    if(next_id == WG_SIZE){
        next_id = 0;
    }

    // Implement bookkeeping here
    fprintf(stdout, ">> SCHEDULER: next_id is %d\n",next_id);



    return next_id;

}
