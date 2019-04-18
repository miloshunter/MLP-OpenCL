#include <stdio.h>
#include "weights/L1_b.c"
#include "weights/L1_w.c"
#include "weights/L2_b.c"
#include "weights/L2_w.c"
#include "weights/dvojka.c"
#include "weights/image.c"
#include <libpng12/png.h>
#include "utils.h"
#include <math.h>
#include <assert.h>

#define PIC_SIZE 28
#define INPUT_SIZE 28*28
#define L1_NUM 512
#define L2_NUM 10
#define CLASS_NUM 10

int LAYER_SIZE[3] = {784, 512, 10};

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

static void relu(double *input, size_t input_len){
    for(int i = 0; i < input_len; i++)
    {
        input[i] = max(0, input[i]);
    }
}


//  Placeholders for calculations
double input[INPUT_SIZE];
double L1[L1_NUM];
double L2[L2_NUM];

void flatten_image()
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
        for(size_t j = 0; j < X; j++)
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


int main()
{
    flatten_image();


    calculate_layer(1, input, L1_w, L1_b, L1, 1);
    calculate_layer(2, L1, L2_w, L2_b, L2, 2);

    printf("Izracunat izlaz: ");
    for(size_t i = 0; i < CLASS_NUM; i++)
    {
        printf("\t%.2f", L2[i]);
    }
    printf("\n");    
    
}