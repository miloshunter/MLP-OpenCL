#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "opencl_utils.h"
#include "../weights/network3/w1.h"
#include "../weights/network3/b1.h"
#include "../weights/network3/w2.h"
#include "../weights/network3/b2.h"
#include "../weights/network3/w3.h"
#include "../weights/network3/b3.h"
#include "../weights/network3/wout.h"
#include "../weights/network3/bout.h"
#include "../weights/slike.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

#define PIC_SIZE 28
#define INPUT_SIZE 28*28
#define CLASS_NUM 10

struct timeval  tv1, tv2;


const int LAYER_SIZE[5] = {784, 4096, 2048, 2048, 10};
#define n_input  784
#define n_layer1  4096
#define n_layer2  2048
#define n_layer3  2048
#define n_output  10

//  Placeholders for calculations
float input[INPUT_SIZE];
float L1[n_layer1];
float L2[n_layer2];
float L3[n_layer3];
float Loutput[n_output];

void load_input(float* image)
{
	for (int i = 0; i < n_input; i++)
	{
		input[i] = image[i];
	}
}

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

void compare_1d_array(float *x, float *y, int length)
{
	float max_error = 0;
    int max_index = -1;
	for (int i = 0; i < length; i++){
		if (fabs((x[i]-y[i])) > max_error)
		{
			max_error = fabs((x[i] - y[i]));
            max_index = i;
		}
        if (0){
            printf("%d : \t%f\t%f razlika = %f\n",
                      i,  x[i], y[i], fabs((x[i] - y[i])));
        }
	}
	printf("Max error = %f on index %d\n", max_error, max_index);
}


//  Activation function: 1 - Relu; 2 - softmax
void calculate_layer(int layer_number, float* input_matrix, float *weights,
                        float* biases, float* out_matrix, int activation_function)
{
    int i, j;
    printf("Calculating layer: %d\n", layer_number);
    int X = LAYER_SIZE[layer_number-1];
    int Y = LAYER_SIZE[layer_number];
    
    // Layer 1

    // float *flatten_w = (float *)malloc((LAYER_SIZE[layer_number-1]*LAYER_SIZE[layer_number]) * sizeof(float));
    // for(size_t i = 0; i < Y; i++)
    // {
    //     for(size_t j = 0; j < X; j++)
    //     {
    //         flatten_w[i*X+j] = (*((weights+i*X) + j));
    //     } 
    // }

    gettimeofday(&tv1, NULL);
    cl_mem w_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            (LAYER_SIZE[layer_number-1]*LAYER_SIZE[layer_number]) * sizeof(float), NULL, &ret);
    cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            LAYER_SIZE[layer_number-1] * sizeof(float), NULL, &ret);
    cl_mem l1_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            LAYER_SIZE[layer_number] * sizeof(float), NULL, &ret);
    gettimeofday(&tv2, NULL);
    printf ("Create buffer time = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
    
    gettimeofday(&tv1, NULL);
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, w_mem_obj, CL_TRUE, 0,
                                (LAYER_SIZE[layer_number-1]*LAYER_SIZE[layer_number]) * sizeof(float), weights, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, x_mem_obj, CL_TRUE, 0,
                                (LAYER_SIZE[layer_number-1]) * sizeof(float), input_matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, l1_mem_obj, CL_TRUE, 0,
                                (LAYER_SIZE[layer_number]) * sizeof(float), out_matrix, 0, NULL, NULL);
    gettimeofday(&tv2, NULL);
    printf ("Copy data time = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
    // Execute the OpenCL kernel on the list
    
    void* args[10];
    size_t arg_num = 3;
    args[0] = x_mem_obj;
    args[1] = w_mem_obj;
    args[2] = l1_mem_obj;
    
    gettimeofday(&tv1, NULL);
    prepare_and_run_kernel("dense_mul", arg_num, args,
                            LAYER_SIZE[layer_number-1], LAYER_SIZE[layer_number]);
    gettimeofday(&tv2, NULL);
    printf ("Layer calc time = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
    
    // Read the memory buffer C on the device to the local variable C
    gettimeofday(&tv1, NULL);
    ret = clEnqueueReadBuffer(command_queue, l1_mem_obj, CL_TRUE, 0, 
            LAYER_SIZE[layer_number] * sizeof(float), out_matrix, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Could not Read buffer from GPU. Error code: %d\n", ret);
        exit(0);
    }
    gettimeofday(&tv2, NULL);
    printf ("Read time = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));

    gettimeofday(&tv1, NULL);    
    for(size_t i = 0; i < LAYER_SIZE[layer_number]; i++)
    {
        out_matrix[i] += biases[i];
    }
    gettimeofday(&tv2, NULL);
    printf ("Add bias time = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
   
    
    gettimeofday(&tv1, NULL);
    if (activation_function == 1) {
        relu(out_matrix, Y);
    } else {
        softmax(out_matrix, Y);
    }   
    gettimeofday(&tv2, NULL);
    printf ("Activation function time = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));

}


int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 64;

    load_input(sedmica);

    init_opencl();
    read_and_build_kernel_program("opencl_mlp_kernel.cl");

    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);

    calculate_layer(1, input, (float *)w1, b1, L1, 1);
    calculate_layer(2, L1, (float *)w2, b2, L2, 1);
    calculate_layer(3, L2, (float *)w3, b3, L3, 1);
    calculate_layer(4, L3, (float *)wout, bout, Loutput, 0);
  
    gettimeofday(&tv2, NULL);
    printf ("Total time = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));


    //Display the result to the screen
    // printf("Rezultat: ");
    // for(i = 0; i < LAYER_SIZE[4]; i++){
    //     printf("%f\t", Loutput[i]);
    // }
    // printf("\n");

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
}
