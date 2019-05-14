#include "opencl_utils.h"


#define MAX_SOURCE_SIZE (0x10000)


void init_opencl()
{
    // Get platform and device information
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    printf("Found %d platforms\n", ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, 
            &device_id, &ret_num_devices);
    
    // Create an OpenCL context
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        printf("Could not create context.  Error code: %d\n", ret);
        exit(0);
    }
    
    // Create a command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret != CL_SUCCESS) {
        printf("Could not create command queue. Error code: %d\n", ret);
        exit(0);
    }


}

// Returns Source Size
size_t read_kernel_source(char* source_filename, char** source_string){
    FILE *fp;
    size_t source_size;
    
    /* Load the source code containing the kernel*/
    fp = fopen(source_filename, "r");
    // printf("Reading kernel: %s\n", source_filename);
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
void read_and_build_kernel_program(char* source_filename){
    
    char *source_string = NULL;
    size_t source_size = read_kernel_source(source_filename, &source_string);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_string,
                                         (const size_t *)&source_size, &ret);

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Could not build program. Error code: %d\n", ret);
        exit(0);
    }
    clFinish(command_queue);
}

void prepare_and_run_kernel(char* kernel_name, size_t args_num,
                                 void* kernel_args[10], size_t in_len,
                                 size_t out_len)
{
    // Create the OpenCL kernel
    kernel = clCreateKernel(program, kernel_name, &ret);

    global_work_size[0] = out_len;
    global_work_size[1] = 1;
    global_work_size[2] = 1;
    
    // Set the arguments of the kernel
    for(size_t i = 0; i < args_num; i++)
    {
        ret = clSetKernelArg(kernel, i, sizeof(cl_mem), (void *)&kernel_args[i]);
        if (ret != CL_SUCCESS) {
            printf("Could not set Kernel ARG. Error code: %d\n", ret);
            exit(0);
        }
    }
    ret = clSetKernelArg(kernel, args_num, sizeof(int), &out_len);
    if (ret != CL_SUCCESS) {
        printf("Could not set Kernel ARG. Error code: %d\n", ret);
        exit(0);
    }
    
    printf("Global work size: %d + %d\n", global_work_size[0], global_work_size[1]);
    ret = clEnqueueNDRangeKernel(command_queue, kernel, args_num-1, NULL, 
            global_work_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("Could not Enqueue NDRange kernel. Error code: %d\n", ret);
        exit(0);
    }

}

void copy_weights_to_device(int layer_num, 
        const int *layer_sizes, float** weights)
{
    for (int n = 1; n < layer_num+1; n++)
    {
        
        w_mem_obj_array[n-1] = clCreateBuffer(
            context, CL_MEM_READ_ONLY, 
            (layer_sizes[n-1]*layer_sizes[n]) * sizeof(float),
            NULL, &ret
        );

        ret = clEnqueueWriteBuffer(
                command_queue, w_mem_obj_array[n-1], CL_TRUE, 0,
                (layer_sizes[n-1]*layer_sizes[n]) * sizeof(float), 
                weights[n-1], 0, NULL, NULL
        );
    }
    
}

