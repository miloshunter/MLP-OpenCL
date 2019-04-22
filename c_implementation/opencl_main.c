#include "opencl_utils.h"

void main()
{
    // Container variables
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;

    init_opencl(context, command_queue);

    read_and_build_kernel_program(context, program, "./kernel_mlp.cl");

    
}

