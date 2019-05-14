#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "opencl_utils.h"
#include "../read_image.h"
#include "../load_parameters.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// #define DEBUG

#define MAX_SOURCE_SIZE (0x100000)

struct timeval  tv1, tv2, tv3, tv4;
struct timeval  tvlaystart, tvlayend;

static void softmax(float *input, size_t input_len) {
    float * Z = input;
    int K = input_len;

    float sum = 0;
    for(int j = 0; j < K; j++)
    {
        sum += exp(Z[j]);
    }

    for(int i = 0; i < K; i++)
    {
        input[i] = exp(input[i])/sum;
    } 

}

static void relu(float *input, int input_len){
    for(int i = 0; i < input_len; i++)
    {
        input[i] = fmax((float)0, input[i]);
    }
}

//  Activation function: 1 - Relu; 2 - softmax
void calculate_layer(
        int layer_num, int*layer_size, float* input_matrix,
        float* biases, float* out_matrix, int activation_function)
{
    gettimeofday(&tvlaystart, NULL);
    int i, j;
    #ifdef DEBUG
    printf("Calculating layer: %d\n", layer_num);
    #endif // DEBUG
    
    gettimeofday(&tv1, NULL);
    int N_IN = layer_size[layer_num-1];
    int N_OUT = layer_size[layer_num];

    gettimeofday(&tv3, NULL);
    cl_mem in_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    (N_IN)*sizeof(float), NULL, &ret);
    cl_mem calc_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    (N_IN*N_OUT)*sizeof(float), NULL, &ret);
    cl_mem out_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    (N_OUT)*sizeof(float), NULL, &ret);
    clFinish(command_queue);
    gettimeofday(&tv4, NULL);
    #ifdef DEBUG
    printf ("\tCreating buffers time = %f microseconds \n",
         (float) (tv4.tv_usec - tv3.tv_usec) +
         (float) 1000000*(tv4.tv_sec - tv3.tv_sec));
    #endif // DEBUG
    gettimeofday(&tv3, NULL);
    ret = clEnqueueWriteBuffer(command_queue, in_mem_obj, CL_TRUE, 0,
                    (N_IN)*sizeof(float), input_matrix, 0, NULL, NULL);
    clFinish(command_queue);
    gettimeofday(&tv4, NULL);
    #ifdef DEBUG
    printf ("\tWritting input to GPU time = %f microseconds \n",
         (float) (tv4.tv_usec - tv3.tv_usec) +
         (float) 1000000*(tv4.tv_sec - tv3.tv_sec));
    #endif // DEBUG
    kernel = clCreateKernel(program, "weights_mul", &ret);
    clFinish(command_queue);

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&w_mem_obj_array[layer_num-1]);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&calc_mem_obj);
    clFinish(command_queue);
    
    global_work_size[0] = N_IN;
    global_work_size[1] = N_OUT;
    #ifdef DEBUG
    
    printf("Global work size: %d + %d\n", 
                global_work_size[0], global_work_size[1]);
    #endif // DEBUG
    gettimeofday(&tv1, NULL);
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            global_work_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);
    gettimeofday(&tv2, NULL);
    #ifdef DEBUG
    printf ("Calculate mul time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
    #endif // DEBUG

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
    clFinish(command_queue);
    gettimeofday(&tv1, NULL);    
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            global_work_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);
    gettimeofday(&tv2, NULL);
    #ifdef DEBUG
    printf ("Calculate add time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
    #endif // DEBUG
    
    gettimeofday(&tv1, NULL);    
    ret = clEnqueueReadBuffer(command_queue, out_mem_obj, CL_TRUE, 0, 
            N_OUT * sizeof(float), out_matrix, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Could not Read buffer from GPU. Error code: %d\n", ret);
        exit(0);
    } 
    clFinish(command_queue);
    gettimeofday(&tv2, NULL);
    #ifdef DEBUG
    printf ("Read calculated time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
    #endif // DEBUG
   
    for(size_t i = 0; i < layer_size[layer_num]; i++)
    {
        out_matrix[i] += biases[i];
    }
     
    if (activation_function == 1) {
        relu(out_matrix, N_OUT);
    } else if (activation_function == 2) {
        softmax(out_matrix, N_OUT);
    }   

    if(layer_num > 0){
        clRetainMemObject(w_mem_obj_array[layer_num-1]);
    }
    clFinish(command_queue);
    gettimeofday(&tvlayend, NULL);
    #ifdef DEBUG
    printf ("Total layer time = %f microseconds\n",
         (float) (tvlayend.tv_usec - tvlaystart.tv_usec) +
         (float) 1000000*(tvlayend.tv_sec - tvlaystart.tv_sec));
    printf("***************************************** ");
    #endif // DEBUG

}


int main(int argc, char **argv) {
    struct timeval  tv1, tv2;
    char *network_name = argv[1];
    int *layer_sizes;
    float ***weights;
    float **biases;

    int layer_num = load_parameters(network_name, &layer_sizes, 
                            &weights, &biases);

    float *flatten_image;
    char tmp[50];
    strcpy(tmp, argv[2]);
    read_png_file(tmp, &flatten_image);

    init_opencl();
    read_and_build_kernel_program(
        "c_implementation/opencl_implementation/kernels.cl"
    );
    //Prepare pointers
    w_mem_obj_array = (cl_mem*) malloc(layer_num*sizeof(cl_mem));
    
    // Convert to 1D weights for OpenCL
    float ** tmp_weights = (float **) malloc(layer_num*sizeof(float*));
    for (int n = 0; n < layer_num; n++)
    {
        tmp_weights[n] = (float *) malloc(
            layer_sizes[n]*layer_sizes[n+1]*sizeof(float)
        );
        for (int i = 0; i < layer_sizes[n+1]; i++)
        {
            for (int j = 0; j < layer_sizes[n]; j++)
            {
                tmp_weights[n][i*layer_sizes[n]+j] = weights[n][i][j];
            }   
        }
    }
    
    copy_weights_to_device(layer_num, layer_sizes, 
        tmp_weights);

    gettimeofday(&tv1, NULL);

    float **L = (float**) malloc(layer_num*sizeof(float*));
    for (size_t i = 0; i < layer_num; i++)
    {
        L[i] = (float*) calloc(layer_sizes[i+1], sizeof(float));
        // printf("Allocating %d floats for layer %d\n", layer_size[i+1], i);
    }
    
    calculate_layer(1, layer_sizes, flatten_image, biases[0], L[0], 1); 
    for (size_t i = 1; i < layer_num; i++)
    {
        int act_fn = 1;
        if(i==layer_num-1) act_fn = 2;
        calculate_layer(i+1, layer_sizes, L[i-1], 
                        biases[i], L[i], act_fn);
    }

    gettimeofday(&tv2, NULL);
    printf ("\n\t\t*\tTotal time OpenCL = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));


    // Display the result to the screen
    for(int i = 0; i < layer_sizes[layer_num]; i++){
        printf("  %d\t", i);
    }
    printf("\n");
    for(int i = 0; i < layer_sizes[layer_num]; i++){
        printf("%.2f\t", L[layer_num-1][i]);
    }
    printf("\n");

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    
    
   return 0;
}
