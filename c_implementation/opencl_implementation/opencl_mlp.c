#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opencl_utils.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 64;
    char *msg = "Pozdrav, Milos\n\0";

    int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
    double out[3] = {0, 0, 0};
    double w[3*5] = {
            1, 0, 0.3333, 0, 0,
            0, 0, 0, 0, 2,
            1, 0, 0, 0, 0
    };
    double b[3] = {1, 2, 3};
    double x[5] = {1, 2, 3, 4, 5};
    for(i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }
    
    init_opencl();
    read_and_build_kernel_program("opencl_mlp_kernel.cl");

    cl_mem w_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            (3*5) * sizeof(double), NULL, &ret);
    cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            5 * sizeof(double), NULL, &ret);
    cl_mem l1_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            3 * sizeof(double), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, w_mem_obj, CL_TRUE, 0,
                                (3*5) * sizeof(double), w, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, x_mem_obj, CL_TRUE, 0,
                                (5) * sizeof(double), x, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, l1_mem_obj, CL_TRUE, 0,
                                (3) * sizeof(double), out, 0, NULL, NULL);

    // Execute the OpenCL kernel on the list
    
    void* args[10];
    args[0] = x_mem_obj;
    args[1] = w_mem_obj;
    args[2] = l1_mem_obj;
    prepare_and_run_kernel("double_array", 1, args, 5, 3);

    // Read the memory buffer C on the device to the local variable C
    double *C = (double*)malloc(sizeof(double)*3);
    ret = clEnqueueReadBuffer(command_queue, l1_mem_obj, CL_TRUE, 0, 
            3 * sizeof(double), out, 0, NULL, NULL);
    
    printf("Procitao: %d?\n", ret);
    //Display the result to the screen
    printf("Rezultat: ");
    for(i = 0; i < 3; i++)
        printf("%d\t", out[i]);
    printf("\n");

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return 0;
}
