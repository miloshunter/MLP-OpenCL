#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "opencl_utils.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int main()
{
    // Create the two input vectors
    //  --- -   -   -   -   -   -   Testing
    #define N_IN 5
    #define N_OUT 5
    printf("Testiranje racunanja matrica!\n");
    float in_mat[N_IN] = {1, 2, 3, 4, 5};
    float out_mat[N_OUT] = {0, 0, 0, 0, 0};
    float biases[N_OUT] = {0, 0, 0, 0, 0};
    

    float weights[N_OUT][N_IN] = {
        {1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2},
        {3, 3, 3, 3, 3},
        {4, 4, 4, 4, 4},
        {5, 5, 5, 5, 5}
    };
    // float in_mat[N_IN] = {1, 2, 3};
    // float out_mat[N_OUT] = {0, 0};
    // float biases[N_OUT] = {0, 0};
    

    // float weights[N_OUT][N_IN] = {
    //     {1, 1, 1},
    //     {2, 2, 2}
    // };
    int i;
    const int LIST_SIZE = 64;

    init_opencl();
    read_and_build_kernel_program("new_kernel.cl");
    struct timeval  tv1, tv2;

    cl_mem in_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    (N_IN)*sizeof(float), NULL, &ret);
    cl_mem w_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    (N_IN*N_OUT)*sizeof(float), NULL, &ret);
    cl_mem calc_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    (N_IN*N_OUT)*sizeof(float), NULL, &ret);
    cl_mem out_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    (N_OUT)*sizeof(float), NULL, &ret);
    
    
    ret = clEnqueueWriteBuffer(command_queue, w_mem_obj, CL_TRUE, 0,
                    (N_IN*N_OUT)*sizeof(float), weights, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, in_mem_obj, CL_TRUE, 0,
                    (N_IN)*sizeof(float), in_mat, 0, NULL, NULL);
    printf("Iskopirao...\n");
    kernel = clCreateKernel(program, "weights_mul", &ret);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&w_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&calc_mem_obj);
    
    global_work_size[0] = N_IN;
    global_work_size[1] = N_OUT;
    printf("Global work size: %d + %d\n", 
                global_work_size[0], global_work_size[1]);
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            global_work_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);

    printf("\nSabiranje izracunatog!\n");
    read_and_build_kernel_program("add_kernel.cl");

    kernel = clCreateKernel(program, "add_calculated", &ret);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&calc_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&out_mem_obj);
    cl_mem num_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    sizeof(int), NULL, &ret);
    int num = N_IN;
    ret = clEnqueueWriteBuffer(command_queue, num_mem_obj, CL_TRUE, 0,
                    sizeof(int), &num, 0, NULL, NULL);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&num_mem_obj);
    global_work_size[0] = N_OUT;

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            global_work_size, NULL, 0, NULL, NULL);

    ret = clEnqueueReadBuffer(command_queue, out_mem_obj, CL_TRUE, 0, 
            N_OUT * sizeof(float), out_mat, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Could not Read buffer from GPU. Error code: %d\n", ret);
        exit(0);
    }
  
    // Display the result to the screen
    printf("\nRezultat: \n");
    for(i = 0; i < N_OUT; i++){
        printf("%f\t", out_mat[i]);
    }
    printf("\n");

    exit(0);
}