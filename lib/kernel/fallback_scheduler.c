#include <stdio.h>

void _pocl_sched_work_item()
{
    fprintf(stderr, "called from kernel\n");
}