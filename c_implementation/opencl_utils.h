#ifndef OPENCL_UTILS
#define OPENCL_UTILS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_SOURCE_SIZE (0x100000)


void init_opencl(cl_context context, cl_command_queue command_queue);

size_t read_kernel_source(char* source_filename, char** source_string);

void read_and_build_kernel_program(cl_context context, cl_program program,
                                     char* source_filename);


#endif // !OPENCL_UTILS





