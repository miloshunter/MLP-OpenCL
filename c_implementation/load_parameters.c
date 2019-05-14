#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

// #define DEBUG


void read_config(char* filepath, int **array_sizes, int *layer_number)
{
    FILE *fp;
    int c;
    char tmp[50];
    strcpy(tmp, filepath);

    fp = fopen(tmp, "r");
    if (fp == 0){
        printf("Cannot open file: %s\n", filepath);
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
    printf("****Number of layers is: %d\t[  ", layer_num);
    for(int i=0; i<layer_num+1; i++){
        printf("%d:%d\t", i, layer_sizes[i]);
    }
    printf("]\n");
}

void read_parameters(char* filename, float ***weights,
                        float **biases, int *sizes, int layer_num)
{
    FILE *fp;
    char tmp[50];
    char fname[50];
    strcpy(fname, filename);
    sprintf(tmp, "parameters/%s_weights.bin", 
                strsep(&filename, "."));

    // printf("Reading weights from %s\n", tmp);
    fp = fopen(tmp, "rb");
    if (fp == 0){
        printf("Cannot open file: %s\n", tmp);
        printf("\n\tPlease check if the network is trained?\n");
        printf("\tIn order to train network use: make train CONF=%s\n\n", fname);
        exit(-1);
    }
#ifdef DEBUG
    print_network_config(layer_num, sizes);
#endif // DEBUG
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

int load_parameters(char* network_name, int ** ls_adr, 
                        float **** weights_adr, float *** biases_adr)
{

    printf("Reading parameters for: %s\n", network_name);
    int layer_num;

    int *layer_sizes;

    read_config(network_name, &layer_sizes, &layer_num);
    (*ls_adr) = layer_sizes;
    
    (*weights_adr) = (float***) malloc((layer_num)*sizeof(float**));
    float ***weights = (*weights_adr);

    // printf("\nAllocated %d ** pointers for Weights\n", layer_num);
    for(int n=0; n<layer_num; n++){
        // printf("\tAllocated %d * pointers each %d floats for layer %d\n", layer_sizes[n+1], layer_sizes[n], n);
        weights[n] = (float**) malloc((layer_sizes[n+1])*sizeof(float*));
        for(int i=0; i<layer_sizes[n+1]; i++){
            weights[n][i] = (float*) malloc((layer_sizes[n])*sizeof(float));
        }
    }
    (*biases_adr) = (float**) malloc((layer_num)*sizeof(float*));
    float **biases = (*biases_adr);
    // printf("\nAllocated %d * pointers for Biases\n", layer_num);
    for(int n=0; n<layer_num; n++){
        // printf("\tAllocated %d floats for layer %d\n", layer_sizes[n+1], n);
        biases[n] = (float*) malloc((layer_sizes[n+1])*sizeof(float));
    }

    read_parameters(network_name, weights, biases, layer_sizes, layer_num);

    // Print weights and biases for testing

#ifdef DEBUG
    int n = layer_num-1;
    for(int i=0; i<layer_sizes[n+1]; i++){
        printf("\t %f\n", biases[n][i]);
    }
#endif // DEBUG
    
    print_network_config(layer_num, layer_sizes);
    return layer_num;
}




