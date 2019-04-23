#include <stdio.h>
#include <stdlib.h>
#include "weights/network3/w1.h"
#include "weights/network3/w2.h"
#include "weights/network3/w3.h"
#include "weights/network3/wout.h"
#include "weights/network3/b1.h"
#include "weights/network3/b2.h"
#include "weights/network3/b3.h"
#include "weights/network3/bout.h"
#include "weights/slike.h"
#include "weights/network3/layer1.h"
#include "weights/network3/layer2.h"
#include "weights/network3/layer3.h"
#include "weights/network3/output.h"
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <gperftools/profiler.h>

#define PIC_SIZE 28
#define INPUT_SIZE 28*28
#define CLASS_NUM 10

const int LAYER_SIZE[5] = {784, 4096, 2048, 2048, 10};
#define n_input  784
#define n_layer1  4096
#define n_layer2  2048
#define n_layer3  2048
#define n_output  10

static void softmax(double *input, size_t input_len) {
    double * Z = input;
    int K = input_len;

    double sum = 0;
    for(int j = 0; j < K; j++)
    {
        sum += exp(Z[j]);
    }

    for(int i = 0; i < K; i++)
    {
        input[i] = exp(input[i])/sum;
    } 

}

static void relu(double *input, int input_len){
    for(int i = 0; i < input_len; i++)
    {
        input[i] = fmax((double)0, input[i]);
    }
}


//  Placeholders for calculations
double input[INPUT_SIZE];
double L1[n_layer1];
double L2[n_layer2];
double L3[n_layer3];
double Loutput[n_output];

void flatten_image(double ** image)
{
    for(size_t i = 0; i < PIC_SIZE; i++)
    {
        for(size_t j = 0; j < PIC_SIZE; j++)
        {
            input[i*PIC_SIZE+j] = image[i][j];
        }
    }
}

//  Activation function: 1 - Relu; 2 - softmax
void calculate_layer(int layer_num, double* input_matrix, double *weights,
                        double* biases, double* out_matrix, int activation_function)
{
    int i, j;
    //printf("Racunanje sloja: %d\n", layer_num);
    int X = LAYER_SIZE[layer_num-1];
    int Y = LAYER_SIZE[layer_num];
    int x = 20, y=10;
    #pragma omp parallel private(i,j) shared(Y, X, out_matrix, input_matrix, weights, biases)
    {
        #pragma omp for schedule(static)
        for(i = 0; i < Y; i++)
        {
            for(j = 0; j < X; j++)
            {
                out_matrix[i] += input_matrix[j]* (*((weights+i*X) + j));
            }
            out_matrix[i] += biases[i];
        }
    }

    if (activation_function == 1) {
        relu(out_matrix, Y);
    } else {
        softmax(out_matrix, Y);
    }   
    
}

void load_input(double* image)
{
	for (int i = 0; i < n_input; i++)
	{
		input[i] = image[i];
	}
}

void compare_1d_array(double *x, double *y, int length)
{
	double max_error = 0;
    int max_index = -1;
	for (int i = 0; i < length; i++){
		if (fabs((x[i]-y[i])) > max_error)
		{
			max_error = fabs((x[i] - y[i]));
            max_index = i;
		}
	}
	printf("Max error = %f on index %d\n", max_error, max_index);
}

void forward_propagation(){
    calculate_layer(1, input, (double *)w1, b1, L1, 1);
    calculate_layer(2, L1, (double *)w2, b2, L2, 1);
    calculate_layer(3, L2, (double *)w3, b3, L3, 1);
    calculate_layer(4, L3, (double *)wout, bout, Loutput, 0);
}

void main()
{
    
	load_input(sestica);

    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);
    // for(int i = 0; i < 1000; i++)
    // {
        
    forward_propagation();
        
        
    // }
    gettimeofday(&tv2, NULL);

    printf ("Total time = %f microseconds\n",
         (double) (tv2.tv_usec - tv1.tv_usec) +
         (double) 1000000*(tv2.tv_sec - tv1.tv_sec));

    printf("Izracunat izlaz: ");
    for(size_t i = 0; i < CLASS_NUM; i++)
    {
        printf("\t%.2f", Loutput[i]);
    }
    printf("\n");
    
	
}