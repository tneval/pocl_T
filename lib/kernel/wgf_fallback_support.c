#include <stdio.h>


int _pocl_sched_work_item()
{
    // Implement bookkeeping here
    fprintf(stderr, "called from kernel\n");

    

    /* unsigned int n_sg = get_num_sub_groups();
    unsigned int sg_size = get_sub_group_size();

    fprintf(stderr, "n_sg: %u\tsg_size: %u\n",n_sg,sg_size);
 */
    int id = 1;

    return id;

}