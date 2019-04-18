#include <stdio.h>
#include <stdlib.h>
#include "weights/w1.h"
#include "weights/w2.h"
#include "weights/w3.h"
#include "weights/wout.h"
#include "weights/b1.h"
#include "weights/b2.h"
#include "weights/b3.h"
#include "weights/bout.h"
#include "weights/sedmica.h"
#include "weights/layer1.h"
#include <math.h>
#include <assert.h>

#define PIC_SIZE 28
#define INPUT_SIZE 28*28
#define CLASS_NUM 10

const int LAYER_SIZE[5] = {784, 512, 256, 128, 10};
#define n_input  784
#define n_layer1  512
#define n_layer2  256
#define n_layer3  128
#define n_output  10
//#define LAYER_SIZE(index) {784, 512, 256, 128, 10}

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
        input[i] = max(0, input[i]);
    }
}


//  Placeholders for calculations
double input[INPUT_SIZE];
double L1[n_layer1];
double L2[n_layer2];
double L3[n_layer3];
double output[n_output];

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
void calculate_layer(int layer_num, double* input, double (*weights)[],
                        double* biases, double* output, int activation_function)
{
    printf("Racunanje sloja: %d\n", layer_num);
    int X = LAYER_SIZE[layer_num-1];
    int Y = LAYER_SIZE[layer_num];

    for(int i = 0; i < Y; i++)
    {
        for(int j = 0; j < X; j++)
        {
            output[i] += input[j]*(*weights)[i,j];
        }
    }

    for(int i = 0; i < Y; i++)
    {
        output[i] += biases[i];
    }

    if (activation_function == 1) {
        relu(output, Y);
    } else {
        //softmax(output, Y);
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
	for (int i = 0; i < length; i++){
		if (fabs((x[i]-y[i]) > max_error))
		{
			max_error = fabs((x[i] - y[i]));
		}
	}
	printf("Max error = %f\n", max_error);
}

void main()
{
	load_input(sedmica);

    calculate_layer(1, input, w1, b1, L1, 1);

	compare_1d_array(L1, layer1, LAYER_SIZE[0]);
    //calculate_layer(2, L1, w2, b2, L2, 2);

    /*printf("Izracunat izlaz: ");
    for(size_t i = 0; i < CLASS_NUM; i++)
    {
        printf("\t%.2f", L2[i]);
    }
    printf("\n"); */   
    
	
}