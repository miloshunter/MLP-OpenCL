#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>


void read_config(char* filename, int **array_sizes, int *layer_number)
{
    FILE *fp;
    int c;
    char tmp[50];
    sprintf(tmp, "../%s", filename);

    fp = fopen(tmp, "r");
    if (fp == 0){
        printf("Cannot open file: %s\n", tmp);
        exit(-1);
    }
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    for(int i=0; i<3; i++) getline(&line, &len, fp);    //Skip header
    read = getline(&line, &len, fp);
    *layer_number = strtol(line, NULL, 10); //Read number of layers
    (*array_sizes) = (int *) malloc((*layer_number)*sizeof(int));
    read = getline(&line, &len, fp);
    for(int i=0; i<(*layer_number)+1; i++){
        read = getline(&line, &len, fp);
        int num = strtol(line, NULL, 10);
        (*array_sizes)[i] = num;
    }
}

void print_network_config(int layer_num, int *layer_sizes)
{
    printf("\nNumber of layers is: %d\n", layer_num);
    printf("Layer sizes are: \n");
    printf("\tInput %d: %d\n", 0, layer_sizes[0]);
    for(int i=1; i<layer_num+1; i++){
        printf("\tLayer %d: %d\n", i, layer_sizes[i]);
    }
    printf("\n");
}

void read_parameters(char* filename, float ***weights,
                        float **biases, int *sizes, int layer_num)
{
    FILE *fp;
    char tmp[50];
    sprintf(tmp, "../parameters/%s_weights.bin", 
                strsep(&filename, "."));

    printf("Reading weights from %s\n", tmp);
    fp = fopen(tmp, "rb");
    if (fp == 0){
        printf("Cannot open file: %s\n", tmp);
        exit(-1);
    }
    print_network_config(layer_num, sizes);
    for(int n=0; n<layer_num; n++){
        for(int i=0; i<sizes[n+1]; i++){
            for(int j=0; j<sizes[n]; j++){
                float buffer;
                fread(&buffer, sizeof(float), 1, fp);
                //printf("%d %d %d : %f\n", n, i, j, buffer);
                weights[n][i][j] = buffer;
            }
        }
    }
    for(int n=0; n<layer_num; n++){
        for(int i=0; i<sizes[n+1]; i++){
            float buffer;
            fread(&buffer, sizeof(float), 1, fp);
            //printf("%d %d %d : %f\n", n, i, j, buffer);
            biases[n][i] = buffer;
        }
    }
}

void skip_line(FILE *file){
    while(getc(file) != '\n');
}

int main(int argc, char **argv)
{
    char *network_name = argv[1];

    printf("Reading parameters for: %s\n", network_name);
    int *layer_sizes;
    int layer_num;

    read_config(network_name, &layer_sizes, &layer_num);

    float ***weights;
    weights = (float***) malloc((layer_num)*sizeof(float**));
    printf("\nAllocated %d ** pointers for Weights\n", layer_num);
    for(int n=0; n<layer_num; n++){
        printf("\tAllocated %d * pointers each %d floats for layer %d\n", layer_sizes[n+1], layer_sizes[n], n);
        weights[n] = (float**) malloc((layer_sizes[n+1])*sizeof(float*));
        for(int i=0; i<layer_sizes[n+1]; i++){
            weights[n][i] = (float*) malloc((layer_sizes[n])*sizeof(float));
        }
    }
    float **biases;
    biases = (float**) malloc((layer_num)*sizeof(float*));
    printf("\nAllocated %d * pointers for Biases\n", layer_num);
    for(int n=0; n<layer_num; n++){
        printf("\tAllocated %d floats for layer %d\n", layer_sizes[n+1], n);
        biases[n] = (float*) malloc((layer_sizes[n+1])*sizeof(float));
    }

    read_parameters(network_name, weights, biases, layer_sizes, layer_num);

    // Print weights and biases for testing
    // int n = layer_num-2;
    // for(int i=0; i<layer_sizes[n+1]; i++){
    //     printf("\t %f\n", biases[n][i]);
    // }
}




