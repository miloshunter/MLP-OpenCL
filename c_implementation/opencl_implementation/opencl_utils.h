#ifndef OPENCL_UTILS
#define OPENCL_UTILS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

cl_int ret;
cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_context context;
cl_command_queue command_queue;
cl_int ret;
cl_program program;
cl_device_id device_id;
cl_kernel kernel;
cl_mem w1_mem_obj;
cl_mem w2_mem_obj;
cl_mem w3_mem_obj;
cl_mem wout_mem_obj;
cl_mem w_mem_obj_array[4];

size_t global_work_size[3];
size_t local_item_size[3];


void init_opencl();

size_t read_kernel_source(char* source_filename, char** source_string);

void read_and_build_kernel_program(char* source_filename);

void prepare_and_run_kernel(char* kernel_name, size_t args_num,
                                 void* kernel_args[10], size_t in_len,
                                 size_t out_len);

void copy_weights_and_biases(const int *LAYER_SIZE, float *w1, float *w2, float *w3, float *wout);
#endif // !OPENCL_UTILS





