#include "opencl_utils.h"

cl_int ret;

void init_opencl(cl_context context, cl_command_queue command_queue)
{
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    printf("Found %d platforms\n", ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, 
            &device_id, &ret_num_devices);
    
    // Create an OpenCL context
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("Could not create context. Error code: %d\n", ret);
        exit(0);
    }
    
    // Create a command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("Could not create command queue.Error code: %d\n", ret);
        exit(0);
    }


}

// Returns Source Size
size_t read_kernel_source(char* source_filename, char** source_string){
    FILE *fp;
    size_t source_size;
    
    /* Load the source code containing the kernel*/
    fp = fopen(source_filename, "r");
    printf("Reading kernel: %s\n", source_filename);
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    char* result = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(result, 1, MAX_SOURCE_SIZE, fp);
    //printf("Read: %s\n\n", source_string);
    fclose(fp);

    *source_string = result;

    return source_size;
}

// Reads Kernel code, build it and builds Kernel Program
void read_and_build_kernel_program(cl_context context, cl_program program,
                                     char* source_filename){
    
    char *source_string = NULL;
    size_t source_size = read_kernel_source("./kernel_mlp.cl", &source_string);
    printf("Code size: %d\n %s\n", source_size, source_string);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_string,
                                            (const size_t *)&source_size, &ret);


}
