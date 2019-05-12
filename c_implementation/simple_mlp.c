#include <stdio.h>
#include <stdlib.h>
#include "load_parameters.h"
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <gperftools/profiler.h>

#define PIC_SIZE 28
#define INPUT_SIZE 28*28
#define CLASS_NUM 10

#define n_input  784
#define n_layer1  4096
#define n_layer2  2048 
#define n_layer3  2048
#define n_output  10
const int LAYER_SIZE[5] = {n_input, n_layer1, n_layer2, n_layer3, n_output};

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


//  Placeholders for calculations
float input[INPUT_SIZE];
float L1[n_layer1];
float L2[n_layer2];
float L3[n_layer3];
float Loutput[n_output];

void flatten_image(float ** image)
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
void calculate_layer(int layer_num, float* input_matrix, float *weights,
                        float* biases, float* out_matrix, int activation_function)
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
    } else if(activation_function == 2) {
        softmax(out_matrix, Y);
    }   
    
}

void load_input(float* image)
{
	for (int i = 0; i < n_input; i++)
	{
		input[i] = image[i];
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
	}
	printf("Max error = %f on index %d\n", max_error, max_index);
}

// void forward_propagation(){
//     calculate_layer(1, input, (float *)w1, b1, L1, 1);    
//     calculate_layer(2, L1, (float *)w2, b2, L2, 1);
//     calculate_layer(3, L2, (float *)w3, b3, L3, 1);
//     calculate_layer(4, L3, (float *)wout, bout, Loutput, 2);
// }

void main(int argc, char **argv)
{
    char *network_name = argv[1];
    int *layer_sizes;

    load_parameters(network_name, layer_sizes);
 
	// load_input();

    struct timeval  tv1, tv2;
    gettimeofday(&tv1, NULL);
   
    // forward_propagation();

    gettimeofday(&tv2, NULL);

    printf ("Total time = %f microseconds \n",
         (float) (tv2.tv_usec - tv1.tv_usec) +
         (float) 1000000*(tv2.tv_sec - tv1.tv_sec));

    printf("Izracunat izlaz: ");
    for(size_t i = 0; i < CLASS_NUM; i++)
    {
        printf("\t%.2f", Loutput[i]);
    }
    printf("\n");
    
	
}
