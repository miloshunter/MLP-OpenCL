#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>

#define LAYER_NUM 4

int array_sizes[LAYER_NUM];

float readfloat(FILE *f) {
  float v;
  fread((void*)(&v), sizeof(v), 1, f);
  return v;
}

void read_weights(char* filename, float ***weights, int *sizes)
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

    
    for(int n=0; n<LAYER_NUM; n++){
        for(int i=0; i<sizes[n]; i++){
            for(int j=0; j<sizes[n+1]; j++){
                float buffer;
                fread(&buffer, sizeof(float), 1, fp);
                weights[n][i][j] = buffer;
            }
        }
    }
    
}

void skip_line(FILE *file){
    while(getc(file) != '\n');
}

void read_config(char* filename, int *array_sizes)
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
    for(int i=0; i<LAYER_NUM+1; i++) getline(&line, &len, fp);
    for(int i=0; i<LAYER_NUM+1; i++){
        read = getline(&line, &len, fp);
        int num = strtol(line, NULL, 10);
        printf("%d\n", num);
        array_sizes[i] = num;
    }
}

int main(int argc, char **argv)
{
    char *network_name = argv[1];

    printf("Reading parameters for: %s\n", network_name);


    int layer_sizes[LAYER_NUM];
    read_config(network_name, layer_sizes);

    // Placeholders
    float **weights;

    for(int i; i<LAYER_NUM+5; i++){
        weights[i] = (float*) malloc(layer_sizes[i]*sizeof(float));
    }

    read_weights(network_name, weights, layer_sizes);


}
