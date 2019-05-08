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

struct timeval  tv1, tv2, tv3, tv4;
struct timeval  tvlaystart, tvlayend;


const int LAYER_SIZE[5] = {784, 4096, 2048, 2048, 10};
#define n_input  784
#define n_layer1  4096
#define n_layer2  2048 
#define n_layer3  1024
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
        if (1){
            printf("%d : \t%f\t%f razlika = %f\n",
                      i,  x[i], y[i], fabs((x[i] - y[i])));
        }
	}
	printf("Max error = %f on index %d \n", max_error, max_index);
}


//  Activation function: 1 - Relu; 2 - softmax
void calculate_layer(int layer_number, float* input_matrix, float *weights,
                        float* biases, float* out_matrix, int activation_function)
{
    gettimeofday(&tvlaystart, NULL);
    int i, j;
    printf("Calculating layer: %d\n", layer_number);
    
    gettimeofday(&tv1, NULL);
    int N_IN = LAYER_SIZE[layer_number-1];
    int N_OUT = LAYER_SIZE[layer_number];

    gettimeofday(&tv3, NULL);
    cl_mem in_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    (N_IN)*sizeof(float), NULL, &ret);
    cl_mem calc_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                    (N_IN*N_OUT)*sizeof(float), NULL, &ret);
    cl_mem out_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    (N_OUT)*sizeof(float), NULL, &ret);
    clFinish(command_queue);
    gettimeofday(&tv4, NULL);
    printf ("\tCreating buffers time = %f microseconds \n",
         (float) (tv4.tv_usec - tv3.tv_usec) +
         (float) 1000000*(tv4.tv_sec - tv3.tv_sec));
    gettimeofday(&tv3, NULL);
    ret = clEnqueueWriteBuffer(command_queue, in_mem_obj, CL_TRUE, 0,
                    (N_IN)*sizeof(float), input_matrix, 0, NULL, NULL);
    clFinish(command_queue);
    gettimeofday(&tv4, NULL);
    printf ("\tWritting input to GPU time = %f microseconds \n",
         (float) (tv4.tv_usec - tv3.tv_usec) +
         (float) 1000000*(tv4.tv_sec - tv3.tv_sec));
    gettimeofday(&tv3, NULL);
    kernel = clCreateKernel(program, "weights_mul", &ret);
    clFinish(command_queue);
    gettimeofday(&tv4, NULL);
    printf ("\tCreating kernel time = %f microseconds \n",
         (float) (tv4.tv_usec - tv3.tv_usec) +
         (float) 1000000*(tv4.tv_sec - tv3.tv_sec));

    gettimeofday(&tv3, NULL);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&in_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&w_mem_obj_array[layer_number-1]);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&calc_mem_obj);
    clFinish(command_queue);
    gettimeofday(&tv4, NULL);
    printf ("\tSetting kernel args time = %f microseconds \n",
         (float) (tv4.tv_usec - tv3.tv_usec) +
         (float) 1000000*(tv4.tv_sec - tv3.tv_sec));
    
    global_work_size[0] = N_IN;
    global_work_size[1] = N_OUT;
    printf("Global work size: %d + %d\n", 
                global_work_size[0], global_work_size[1]);
    gettimeofday(&tv2, NULL);
    printf ("Prepare mul time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
    gettimeofday(&tv1, NULL);
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            global_work_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);
    gettimeofday(&tv2, NULL);
    printf ("Calculate mul time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));

    gettimeofday(&tv1, NULL);

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
    gettimeofday(&tv2, NULL);
    printf ("Prepare add time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
    gettimeofday(&tv1, NULL);    
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            global_work_size, NULL, 0, NULL, NULL);
    clFinish(command_queue);
    gettimeofday(&tv2, NULL);
    printf ("Calculate add time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
    
    gettimeofday(&tv1, NULL);    
    ret = clEnqueueReadBuffer(command_queue, out_mem_obj, CL_TRUE, 0, 
            N_OUT * sizeof(float), out_matrix, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Could not Read buffer from GPU. Error code: %d\n", ret);
        exit(0);
    } 
    clFinish(command_queue);
    gettimeofday(&tv2, NULL);
    printf ("Read calculated time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
   
    gettimeofday(&tv1, NULL);    
    for(size_t i = 0; i < LAYER_SIZE[layer_number]; i++)
    {
        out_matrix[i] += biases[i];
    }
    gettimeofday(&tv2, NULL);
    printf ("Add bias time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));
   
    
    gettimeofday(&tv1, NULL);
    if (activation_function == 1) {
        relu(out_matrix, N_OUT);
    } else if (activation_function == 2) {
        softmax(out_matrix, N_OUT);
    }   
    gettimeofday(&tv2, NULL);
    printf ("Activation function time = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));

    if(layer_number > 0){
        clRetainMemObject(w_mem_obj_array[layer_number-1]);
    }
    clFinish(command_queue);
    gettimeofday(&tvlayend, NULL);
    printf ("Total layer time = %f microseconds\n",
         (float) (tvlayend.tv_usec - tvlaystart.tv_usec) +
         (float) 1000000*(tvlayend.tv_sec - tvlaystart.tv_sec));
    printf("***************************************** ");

}


int main(void) {
    load_input(petica);

    init_opencl();
    read_and_build_kernel_program("new_kernel.cl");

    struct timeval  tv1, tv2;

    copy_weights_and_biases(LAYER_SIZE, (float *)w1, (float *)w2, 
                                (float *)w3, (float *)wout);

    gettimeofday(&tv1, NULL);

    calculate_layer(1, input, (float *)w1, b1, L1, 1);
    calculate_layer(2, L1, (float *)w2, b2, L2, 1);
    calculate_layer(3, L2, (float *)w3, b3, L3, 1);
    calculate_layer(4, L3, (float *)wout, bout, Loutput, 2);
  
    gettimeofday(&tv2, NULL);
    printf ("\nTotal time = %f microseconds\n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));


    // Display the result to the screen
    printf("Rezultat: \n");
    for(int i = 0; i < LAYER_SIZE[4]; i++){
        printf("  %d\t", i);
    }
    printf("\n");

    for(int i = 0; i < LAYER_SIZE[4]; i++){
        printf("%.2f\t", Loutput[i]);
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
